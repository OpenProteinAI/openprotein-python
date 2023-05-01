
import openprotein.config as config

import requests
from urllib.parse import urljoin
from typing import Union


class APISession(requests.Session):
    def __init__(self, username, password, backend=config.Backend.PROD):
        super().__init__()

        self.backend = backend
        self.auth = (username, password)
        self.verify = True

    def request(self, method: Union[str, bytes], url: Union[str, bytes], *args, **kwargs):
        full_url = urljoin(self.backend, url)
        #print(full_url)
        #print(args)
        #print(kwargs)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        #print(response, response.json())
        return response


class RestEndpoint:
    pass