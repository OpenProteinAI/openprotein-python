import pytest
from unittest.mock import MagicMock, patch
from openprotein_python.base import AuthError, APISession
import io
import requests
import json
from urllib.parse import urljoin

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

@pytest.fixture
def response_mock_unauthenticated():
    mock = ResponseMock()
    mock.status_code = 401
    mock.text = "Authentication failed"
    return mock

@pytest.fixture
def response_mock_authenticated():
    mock = ResponseMock()
    mock.status_code = 200
    token = 'testtoken'
    mock.json = MagicMock(return_value={'access_token': token})
    return mock


def test_APISession_authenticate_failed(response_mock_unauthenticated):
    username = 'testuser'
    password = 'testpassword'

    with patch.object(APISession, 'post', return_value=response_mock_unauthenticated):
        with pytest.raises(AuthError) as exc:
            APISession(username, password)

        assert "Unable to authenticate with given credentials" in str(exc.value) 
        APISession.post.assert_called_once_with(
            'https://api.openprotein.ai/api/v1/login/user-access-token',
            params={'username': username, 'password': password}, timeout=3
        )

def test_APISession_authenticate_successful(response_mock_authenticated):
    username = 'testuser'
    password = 'testpassword'
    token = 'testtoken'

    with patch.object(APISession, 'post', return_value=response_mock_authenticated):
        session = APISession(username, password)

        assert session.auth.token == token
        session.post.assert_called_once_with(
            'https://api.openprotein.ai/api/v1/login/user-access-token',
            params={'username': username, 'password': password}, timeout=3
        )
