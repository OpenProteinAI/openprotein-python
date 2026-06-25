import sys
import threading
import time
import warnings
from typing import Mapping, Sequence
from urllib.parse import urljoin

import requests
import requests.auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

from openprotein._version import __version__
from openprotein.auth_store import TokenRecord, TokenStore
from openprotein.errors import AuthError, HTTPError

# Refresh the access token this many seconds before it actually expires,
# to absorb clock skew and request latency. 30s is a conservative buffer for
# typical clock drift plus a round-trip against a ~15-minute access token.
SKEW_SECONDS = 30


class BearerAuth(requests.auth.AuthBase):
    """
    See https://stackoverflow.com/a/58055668
    """

    def __init__(self, username, token):
        self._username = username
        self._token = token

    @property
    def token(self):
        warnings.warn(
            "DeprecationWarning: Accessing session.auth.token is deprecated and will throw an error in the future. Use session.auth._token if you need it."
        )
        return self._token

    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self._token
        return r


class APISession(requests.Session):
    """
    A class to handle API sessions. This class provides a connection session to the OpenProtein API.
    """

    def __init__(
        self,
        username: str,
        password: str,
        backend: str,
        timeout: int = 180,
    ):
        if not username or not password:
            raise AuthError(
                "Expected username and password. Or use environment variables `OPENPROTEIN_USERNAME` and `OPENPROTEIN_PASSWORD`. Or provide these variables (`username` and `password`) in ~/.openprotein/config.toml."
            )
        super().__init__()
        self.backend = backend
        self.verify = True
        self.timeout = timeout

        # Custom retry strategies
        # auto retry for pesky connection reset errors and others
        # 503 will catch if BE is refreshing
        retry = Retry(
            total=4,
            backoff_factor=3,  # 0,1,4,13s
            status_forcelist=[500, 502, 503, 504, 101, 104],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("https://", adapter)
        self.headers["User-Agent"] = f"openprotein-python/{__version__}"

        # Auth / refresh state. Initialised before any network call so that
        # request() can read _refreshing / _legacy_mode safely.
        self._username = username
        self._password = password
        self._token_store = TokenStore()
        self._identity = self._token_store.identity(backend, username)
        # _refresh_lock serializes concurrent refreshes across threads.
        # _refreshing (see the _refreshing property) is a thread-local recursion
        # guard: the refresh POST flows back through request(), so the refreshing
        # thread must skip the refresh logic, but OTHER threads sharing this
        # session (e.g. via stream_parallel) must not be suppressed by it.
        self._refresh_lock = threading.Lock()
        self._tl = threading.local()
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0
        self._legacy_mode: bool = False

        self._init_auth()

    @property
    def _refreshing(self) -> bool:
        return getattr(self._tl, "refreshing", False)

    @_refreshing.setter
    def _refreshing(self, value: bool) -> None:
        self._tl.refreshing = value

    def _init_auth(self):
        """Adopt a valid stored record if present, otherwise log in fresh."""
        record = self._token_store.load(self._identity)
        if record is not None and record.refresh_token:
            self._adopt(record)
        else:
            self.login(self._username, self._password)

    def post(self, url, data=None, json=None, **kwargs):
        r"""Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """
        timeout = self.timeout
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")

        return self.request(
            "POST", url, data=data, json=json, timeout=timeout, **kwargs
        )

    def login(self, username: str, password: str):
        """
        Authenticate connection to OpenProtein with your credentials.

        Parameters
        -----------
        username: str
            username
        password: str
            password
        """
        # unset the auth first
        self.auth = None
        self._get_auth_token(username, password)

    def _auth_post(self, url: str, **kwargs):
        """POST for the auth endpoints, with the refresh logic suppressed.

        Sets ``_refreshing`` so the proactive/reactive refresh in request()
        does not recurse while we are logging in or refreshing.
        """
        self._refreshing = True
        try:
            return self.post(url, **kwargs)
        finally:
            self._refreshing = False

    def _get_auth_token(self, username: str, password: str):
        url = urljoin(self.backend, "v1/auth/login")
        try:
            response = self._auth_post(
                url,
                data={"username": username, "password": password},
                headers={"X-Token-Delivery": "body"},
                timeout=3,
            )
        except HTTPError as e:
            if e.status_code == 404:
                # Backend not yet upgraded -> use the legacy endpoint.
                return self._legacy_login(username, password)
            raise AuthError(
                "Authentication failed. Please check your credentials and connection."
            ) from e
        self._apply_token_response(response.json(), username)

    def _legacy_login(self, username: str, password: str):
        url = urljoin(self.backend, "v1/login/access-token")
        try:
            response = self._auth_post(
                url,
                data={"username": username, "password": password},
                timeout=3,
            )
        except HTTPError as e:
            raise AuthError(
                "Authentication failed. Please check your credentials and connection."
            ) from e
        token = response.json().get("access_token")
        if token is None:
            raise AuthError("Unable to authenticate with given credentials.")
        self._legacy_mode = True
        self._refresh_token = None
        self._expires_at = 0.0
        self._set_token(username, token)

    def _apply_token_response(self, result: dict, username: str):
        """Parse a login/refresh response, update memory + store, set the bearer."""
        access_token = result.get("access_token")
        if access_token is None:
            raise AuthError("Unable to authenticate with given credentials.")
        refresh_token = result.get("refresh_token")
        if refresh_token is None:
            # Cookie-mode / pre-rotation backend -> no refresh available.
            self._legacy_mode = True
            self._refresh_token = None
            self._expires_at = 0.0
        else:
            self._legacy_mode = False
            self._refresh_token = refresh_token
            # A missing/zero expires_in is treated as "expired now", which makes
            # the proactive check refresh on the next request rather than trust a
            # token of unknown lifetime.
            self._expires_at = time.time() + (result.get("expires_in") or 0)
            self._token_store.save(
                self._identity,
                TokenRecord(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_at=self._expires_at,
                ),
            )
        self._set_token(username, access_token)

    def _set_token(self, username: str, access_token: str):
        if isinstance(self.auth, BearerAuth):
            self.auth._token = access_token
        else:
            self.auth = BearerAuth(username=username, token=access_token)

    def _adopt(self, record: TokenRecord):
        """Adopt an existing token record (from disk or a fresher in-store copy).

        Callers must only pass a record whose ``refresh_token`` is set — a
        record without one is not adoptable (it would leave the session in
        non-legacy mode with nothing to refresh).
        """
        self._legacy_mode = False
        self._refresh_token = record.refresh_token
        self._expires_at = record.expires_at
        self._set_token(self._username, record.access_token)

    def _refresh(self):
        """Refresh the access token, serialized across threads and processes.

        Re-reads the store under the lock first: if another thread/process
        already rotated the token, adopt that instead of refreshing again
        (replaying a superseded refresh token would revoke the session).
        """
        with self._refresh_lock, self._token_store.lock():
            record = self._token_store.load(self._identity)
            if (
                record is not None
                and record.refresh_token
                and record.expires_at > self._expires_at
            ):
                self._adopt(record)
                return

            if self._refresh_token is None:
                self._relogin()
                return

            url = urljoin(self.backend, "v1/auth/refresh")
            try:
                response = self._auth_post(
                    url,
                    json={"refresh_token": self._refresh_token},
                    headers={"X-Token-Delivery": "body"},
                    timeout=10,
                )
            except HTTPError as e:
                # Recover by re-logging-in when the refresh token is rejected
                # (401/403) or when the endpoint is absent (404 from a backend
                # that issued a refresh token but doesn't expose /auth/refresh,
                # e.g. mid-rollback) — re-login itself drops to legacy mode if
                # the backend has no refresh support. Other statuses are real
                # errors and propagate.
                if e.status_code in (401, 403, 404):
                    self._relogin()
                    return
                raise
            self._apply_token_response(response.json(), self._username)

    def _relogin(self):
        """Re-establish a session from the in-memory credentials."""
        try:
            self.login(self._username, self._password)
        except AuthError:
            raise
        except Exception as e:
            raise AuthError("Session expired and re-login failed.") from e

    def _should_refresh_proactively(self) -> bool:
        return (
            not self._refreshing
            and not self._legacy_mode
            and self._refresh_token is not None
            and time.time() >= self._expires_at - SKEW_SECONDS
        )

    def request(self, method: str, url: str, *args, **kwargs):
        if self._should_refresh_proactively():
            self._refresh()

        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)

        # Reactive refresh: a 401 means the access token expired between the
        # proactive check and the call (or skew). Refresh once and retry once.
        if (
            response.status_code == 401
            and not self._refreshing
            and not self._legacy_mode
        ):
            self._refresh()
            response = super().request(method, full_url, *args, **kwargs)

        if (js := kwargs.get("json")) and js is not None:
            if _total_size(js) > 1e6:
                warnings.warn(
                    "The requested payload is >1MB. There might be some delays or issues in processing. If the request fails, please try again with smaller sizes."
                )

        # intercept CloudFront errors
        if "cloudfront" in response.headers.get("Server", "").lower():
            if response.status_code in (502, 503):
                raise CloudFrontError(
                    f"We're experiencing a temporary backend issue via CloudFront. Please try again later. Error {response.status_code}."
                )
            elif response.status_code == 504:
                raise TimeoutError(
                    "Your request took too long to process likely due to it's size. Please try breaking it up into smaller requests if possible."
                )
        elif not response.ok:
            # raise custom exception that prints better error message than requests.HTTPError
            raise HTTPError(response)
        return response


def _total_size(o: Sequence | Mapping | object, seen=None):
    """Recursively finds size of objects including contents."""
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(o)
    if isinstance(o, dict):
        size += sum((_total_size(k, seen) + _total_size(v, seen)) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        size += sum(_total_size(i, seen) for i in o)
    return size


class TimeoutError(requests.exceptions.HTTPError):
    """
    An Exception raised due to timeout, possibly from overly large
    requests.
    """


class CloudFrontError(requests.exceptions.HTTPError):
    """
    An Exception raised due to CloudFront.

    This is usually due to the strict timeout from CloudFront.
    AWS CloudFront limits responses to return within 2 minutes.
    This can be a bit prohibitive for our system that tends to
    deal with large data. It is usually safe to just ignore/retry upon
    hitting this error. Our system will scale up and still handle
    the job.
    """
