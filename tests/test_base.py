import io
from unittest.mock import MagicMock, patch

import pytest
from openprotein.base import APISession, AuthError, BearerAuth

from tests.conf import BACKEND


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
    token = "testtoken"
    mock.json = MagicMock(return_value={"access_token": token})  # ty: ignore[invalid-assignment]
    return mock


def test_APISession_authenticate_failed(response_mock_unauthenticated):
    username = "testuser"
    password = "testpassword"

    # post() is patched, so request()'s status handling never runs: the failure
    # this exercises is an empty response body (no access_token), not the 401.
    response_mock_unauthenticated.json = MagicMock(return_value={})
    with patch("openprotein.base.TokenStore") as store_cls:
        store_cls.return_value.load.return_value = None
        store_cls.return_value.identity.return_value = "id"
        with patch.object(
            APISession, "post", return_value=response_mock_unauthenticated
        ):
            with pytest.raises(AuthError) as exc:
                APISession(username, password, backend=BACKEND)

            assert "Unable to authenticate with given credentials" in str(exc.value)
            APISession.post.assert_called_once_with(  # ty: ignore[unresolved-attribute]
                f"{BACKEND}v1/auth/login",
                data={"username": username, "password": password},
                headers={"X-Token-Delivery": "body"},
                timeout=3,
            )


def test_APISession_authenticate_successful(response_mock_authenticated):
    username = "testuser"
    password = "testpassword"
    token = "testtoken"

    with patch("openprotein.base.TokenStore") as store_cls:
        store_cls.return_value.load.return_value = None
        store_cls.return_value.identity.return_value = "id"
        with patch.object(APISession, "post", return_value=response_mock_authenticated):
            session = APISession(username, password, backend=BACKEND)

            assert isinstance(session.auth, BearerAuth)
            assert session.auth._token == token
            # No refresh_token in the response -> legacy mode, access token used as-is.
            assert session._legacy_mode is True
            APISession.post.assert_called_once_with(  # ty: ignore[unresolved-attribute]
                f"{BACKEND}v1/auth/login",
                data={"username": username, "password": password},
                headers={"X-Token-Delivery": "body"},
                timeout=3,
            )
