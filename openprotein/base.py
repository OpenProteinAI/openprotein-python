import contextvars
import re
import sys
import threading
import time
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from urllib.parse import urljoin

import requests
import requests.auth
from requests.adapters import HTTPAdapter
from requests.models import complexjson
from requests.packages.urllib3.util.retry import Retry  # type: ignore

from openprotein._version import __version__
from openprotein.auth_store import TokenRecord, TokenStore
from openprotein.errors import AuthError, HTTPError

__all__ = [
    "APISession",
    "BearerAuth",
    "CloudFrontError",
    "TimeoutError",
]


class _NoReplayRetry(Retry):
    """Resend a POST only when it failed to connect, so it was never sent.

    urllib3 retries an unclassified error such as a TLS failure for every
    method, after the server may already have received the request.
    """

    def increment(
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        if (
            (method or "").upper() == "POST"
            and error is not None
            and not self._is_connection_error(error)
        ):
            raise error.with_traceback(_stacktrace)
        return super().increment(method, url, response, error, _pool, _stacktrace)


class _ModelSubmissionRetry(_NoReplayRetry):
    """Retry model POSTs on 503 or a failure to establish a connection."""

    def is_retry(self, method, status_code, has_retry_after=False):
        if method.upper() == "POST":
            return status_code == 503
        return super().is_retry(method, status_code, has_retry_after)


# Each POST to these creates a new sampling job, so a resend samples twice.
_SAMPLING_PATHS = (
    "v1/data/prompts/sample_msa",
    "v1/prompt/sample_msa",
    "v1/align/prompt",
)


# Refresh the access token this many seconds before it expires, to absorb
# clock skew and request latency.
_SKEW_SECONDS = 30
_IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def _freeze_idempotent_json(kwargs: dict) -> None:
    """Make a keyed JSON request's body stable across session-level retries.

    urllib3 retries one prepared request, but the reactive authentication retry
    below prepares a request again. Supplying the already encoded bytes for both
    paths keeps the body byte-for-byte identical while leaving ordinary SDK
    requests on the existing ``json=`` path.
    """
    headers = kwargs.get("headers")
    if not isinstance(headers, Mapping) or "json" not in kwargs:
        return
    if not any(
        isinstance(name, str) and name.lower() == _IDEMPOTENCY_KEY_HEADER.lower()
        for name in headers
    ):
        return
    if "data" in kwargs and kwargs["data"] is not None:
        return
    if kwargs["json"] is None:
        return

    kwargs["data"] = complexjson.dumps(kwargs["json"], allow_nan=False).encode("utf-8")
    request_headers = dict(headers)
    if not any(
        isinstance(name, str) and name.lower() == "content-type"
        for name in request_headers
    ):
        request_headers["Content-Type"] = "application/json"
    kwargs["headers"] = request_headers


# `299 <agent> "<text>"`: a persistent warning, which the API sends on deprecated calls
_SERVER_WARNING = re.compile(r'299\s+\S+\s+"(?P<text>[^"]*)"')


_server_warnings_silenced: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "server_warnings_silenced", default=False
)


@contextmanager
def _silence_server_warnings() -> Generator[None]:
    """Drop the server's `Warning: 299` header on requests made in this block.

    For SDK functions that emit their own versioned deprecation warning.
    """
    token = _server_warnings_silenced.set(True)
    try:
        yield
    finally:
        _server_warnings_silenced.reset(token)


@contextmanager
def _surface_server_warnings() -> Generator[None]:
    """Keep the server's `Warning: 299` header on requests made in this block,
    inside a call that silences it for its own request.
    """
    token = _server_warnings_silenced.set(False)
    try:
        yield
    finally:
        _server_warnings_silenced.reset(token)


def _warn_from_server(response: requests.Response) -> None:
    """Surface a response's `Warning: 299` header as a `FutureWarning`."""
    if _server_warnings_silenced.get():
        return
    match = _SERVER_WARNING.match(response.headers.get("Warning", ""))
    if match is not None:
        warnings.warn(match["text"], FutureWarning, stacklevel=2)


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
        self._init_transport(backend, timeout)

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

    def _init_transport(self, backend: str, timeout: float) -> None:
        """Configure the HTTP transport: retry policy, adapters, and headers."""
        super().__init__()
        self.backend = backend
        self.verify = True
        self.timeout = timeout

        retry = Retry(
            total=4,
            backoff_factor=3,  # 0,1,4,13s
            status_forcelist=[500, 502, 503, 504, 101, 104],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        for scheme in ("https://", "http://"):
            self.mount(scheme, adapter)
        model_retry = _ModelSubmissionRetry(
            total=retry.total,
            backoff_factor=retry.backoff_factor,
            status_forcelist=retry.status_forcelist,
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self.mount(urljoin(backend, "v1/models/"), HTTPAdapter(max_retries=model_retry))
        sampling_retry = _NoReplayRetry(
            total=retry.total,
            backoff_factor=retry.backoff_factor,
            status_forcelist=retry.status_forcelist,
            raise_on_status=False,
        )
        for path in _SAMPLING_PATHS:
            self.mount(urljoin(backend, path), HTTPAdapter(max_retries=sampling_retry))
        self.headers["User-Agent"] = f"openprotein-python/{__version__}"

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
                # Older servers have no token endpoint; use the legacy login.
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
            # No refresh token issued, so refresh is unavailable.
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
        with (
            self._refresh_lock,
            self._token_store.lock(),
            _surface_server_warnings(),
        ):
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
                # A rejected (401/403) or missing (404) refresh endpoint falls back
                # to a fresh login, which drops to legacy mode when the server has
                # no refresh support. Other statuses propagate.
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
            and time.time() >= self._expires_at - _SKEW_SECONDS
        )

    def request(self, method: str, url: str, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        if self._should_refresh_proactively():
            self._refresh()

        # Freeze the JSON body once so a reactive 401 retry resends the same
        # payload under the same idempotency key.
        if method.upper() == "POST":
            _freeze_idempotent_json(kwargs)

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

        _warn_from_server(response)

        if (js := kwargs.get("json")) and _total_size(js) > 1e6:
            warnings.warn(
                "The requested payload is >1MB. There might be some delays or issues in processing. If the request fails, please try again with smaller sizes."
            )

        if "cloudfront" in response.headers.get("Server", "").lower():
            if response.status_code in (502, 503):
                raise CloudFrontError(
                    f"We're experiencing a temporary backend issue via CloudFront. Please try again later. Error {response.status_code}."
                )
            elif response.status_code == 504:
                raise TimeoutError(
                    "Your request took too long to process likely due to it's size. Please try breaking it up into smaller requests if possible."
                )
        if not response.ok:
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
