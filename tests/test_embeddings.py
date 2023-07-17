import pytest
from unittest.mock import MagicMock
from openprotein.base import APISession
from datetime import datetime
from openprotein.api.poet import *
import io
from urllib.parse import urljoin

from typing import List, Optional, Union
from io import BytesIO
from unittest.mock import ANY
import json
from openprotein.base import BearerAuth

class ResponseMock:
    def __init__(self):
        super().__init__()
        self._json = {}
        self.headers = {}
        self.iter_content = MagicMock()
        self._content = None
        self.status_code = 200
        self.raw = io.BytesIO()  # Create an empty raw bytes stream
        self.text = "blank"

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def json(self):
        return self._json


class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = "test_username"
        password = "test_password"
        super().__init__(username, password)

    def get_auth_token(self, username, password):
        return BearerAuth('AUTHORIZED')

    def post(self, endpoint, data=None, json=None, **kwargs):
        return ResponseMock()

    def get(self, endpoint, **kwargs):
        return ResponseMock()

    def request(self, method, url, *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        return response

@pytest.fixture
def api_session_mock():
    sess = APISessionMock()
    yield sess