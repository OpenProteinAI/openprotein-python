from typing import Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import openprotein.config as config
from openprotein.errors import APIError, AuthError, HTTPError


class BearerAuth(requests.auth.AuthBase):
    """
    See https://stackoverflow.com/a/58055668
    """

    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
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
        username: str,
        password: str,
        backend: str = "https://api.openprotein.ai/api/",
        timeout: int = 180,
    ):
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

    def request(
        self, method: Union[str, bytes], url: Union[str, bytes], *args, **kwargs
    ):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        if not response.ok:
            # raise custom exception that prints better error message than requests.HTTPError
            raise HTTPError(response)
        return response


class RestEndpoint:
    pass
