import openprotein.config as config

import requests
from urllib.parse import urljoin
from typing import Union

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from openprotein.errors import APIError, InvalidParameterError, MissingParameterError, AuthError

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
    

    def __init__(self, username:str,
                 password:str,
                 backend:str = "https://api.openprotein.ai/api/",
                 timeout:int = 180):
        super().__init__()
        self.backend = backend
        self.verify = True
        self.timeout = timeout

        # Custom retry strategies
        #auto retry for pesky connection reset errors and others
        # 503 will catch if BE is refreshing
        retry = Retry(total=4,
                      backoff_factor=3, #0,1,4,13s
                      status_forcelist=[500, 502, 503, 504, 101, 104]) 
        adapter = HTTPAdapter(max_retries=retry)
        self.mount('https://', adapter)
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
        if 'timeout' in kwargs:
            timeout = kwargs.pop('timeout')
  
        return self.request("POST",
                            url,
                            data=data,
                            json=json,
                            timeout=timeout,
                            **kwargs)
    
    def login(self, username:str, password:str):
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

    def _get_auth_token(self, username:str, password:str):
        endpoint = "v1/login/user-access-token"
        url = urljoin(self.backend, endpoint)
        response = self.post(
            url, params={"username": username, "password": password}, timeout=3
        )
        if response.status_code == 200:
            result = response.json()
            token = result["access_token"]
            return BearerAuth(token)
        else:
            raise AuthError(
                f"Unable to authenticate with given credentials: {response.status_code} : {response.text}"
            )

    def request(
        self, method: Union[str, bytes], url: Union[str, bytes], *args, **kwargs
    ):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        # allow 400 to pass to get caught by autherror
        if response.status_code not in [200, 201, 202, 400]:
            raise APIError(
                f"Request failed: \n\t status: {response.status_code} \n\t message: {response.text} "
            )
        return response


class RestEndpoint:
    pass
