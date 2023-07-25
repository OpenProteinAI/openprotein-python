import openprotein.config as config

import requests
from urllib.parse import urljoin
from typing import Union
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
    """Connection session."""

    def __init__(self, username:str,
                 password:str,
                 backend:str = "https://dev.api.openprotein.ai/api/" ):
        super().__init__()
        self.backend = backend
        self.login(username, password)
        self.verify = True

    def login(self, username:str, password:str):
        self.auth = self.get_auth_token(username, password)

    def get_auth_token(self, username:str, password:str):
        endpoint = "v1/login/user-access-token"
        url = urljoin(self.backend, endpoint)
        response = requests.post(
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
        if response.status_code not in [200, 201, 202]:
            raise APIError(
                f"Request failed with status {response.status_code} and message {response.text} "
            )
        return response


class RestEndpoint:
    pass
