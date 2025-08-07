import os
import sys
import warnings
from collections.abc import Container, Mapping
from typing import Union
from urllib.parse import urljoin

import requests
import requests.auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

import openprotein.config as config
from openprotein.errors import APIError, AuthError, HTTPError

USERNAME = os.getenv("OPENPROTEIN_USERNAME")
PASSWORD = os.getenv("OPENPROTEIN_PASSWORD")
BACKEND = os.getenv("OPENPROTEIN_API_BACKEND", "https://api.openprotein.ai/api/")


class BearerAuth(requests.auth.AuthBase):
    """
    See https://stackoverflow.com/a/58055668
    """

    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.token
        return r


class APISession(requests.Session):
    """
    A class to handle API sessions. This class provides a connection session to the OpenProtein API.

    Parameters
    ----------
    username : str
        The username of the user.
    password : str
        The password of the user.

    Examples
    --------
    >>> session = APISession("username", "password")
    """

    def __init__(
        self,
        username: str | None = USERNAME,
        password: str | None = PASSWORD,
        backend: str = BACKEND,
        timeout: int = 180,
    ):
        if not username or not password:
            raise AuthError(
                "Expected username and password. Or use environment variables `OPENPROTEIN_USERNAME` and `OPENPROTEIN_PASSWORD`"
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
        self.login(username, password)

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
        self.auth = self._get_auth_token(username, password)

    def _get_auth_token(self, username: str, password: str):
        endpoint = "v1/login/access-token"
        url = urljoin(self.backend, endpoint)
        try:
            response = self.post(
                url, data={"username": username, "password": password}, timeout=3
            )
        except HTTPError as e:
            # if an error occured during auth, we raise an AuthError with reference to the HTTPError
            raise AuthError(
                f"Authentication failed. Please check your credentials and connection."
            ) from e

        result = response.json()
        token = result.get("access_token")
        if token is None:
            raise AuthError("Unable to authenticate with given credentials.")
        return BearerAuth(token)

    def request(self, method: str, url: str, *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)

        if (js := kwargs.get("json")) and js is not None:
            if total_size(js) > 1e6:
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


def total_size(o, seen=None):
    """Recursively finds size of objects including contents."""
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(o)
    if isinstance(o, dict):
        size += sum((total_size(k, seen) + total_size(v, seen)) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        size += sum(total_size(i, seen) for i in o)
    return size


class RestEndpoint:
    pass


class TimeoutError(requests.exceptions.HTTPError):
    pass


class CloudFrontError(requests.exceptions.HTTPError):
    pass
