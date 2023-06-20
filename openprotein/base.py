
import openprotein.config as config

import requests
from urllib.parse import urljoin
from typing import Union


class BearerAuth(requests.auth.AuthBase):
    """
    See https://stackoverflow.com/a/58055668
    """
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers['authorization'] = 'Bearer ' + self.token
        return r


class APISession(requests.Session):
    def __init__(self, username, password, backend=config.Backend.PROD):
        super().__init__()
        self.backend = backend
        self.login(username, password)
        self.verify = True

    def login(self, username, password):
        self.auth = self.get_auth_token(username, password)

    def get_auth_token(self, username, password):
        endpoint = 'v1/login/access-token'
        url = urljoin(self.backend, endpoint)
        response = requests.post(url, data={'username': username, 'password': password})
        result = response.json()
        token = result['access_token']
        return BearerAuth(token)

    def request(self, method: Union[str, bytes], url: Union[str, bytes], *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        return response


class RestEndpoint:
    pass