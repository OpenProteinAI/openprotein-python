import io
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest
import requests

from openprotein.base import APISession, AuthError, BearerAuth, CloudFrontError
from openprotein.base import TimeoutError as GatewayTimeoutError
from openprotein.errors import HTTPError
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


@pytest.fixture
def session(tmp_path, monkeypatch):
    from openprotein import auth_store

    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )
    login = MagicMock(status_code=200, ok=True, headers={})
    login.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 900,
    }
    with patch.object(APISession, "post", return_value=login):
        session = APISession("user@x.com", "pw", backend=BACKEND)
    session._expires_at = time.time() + 10_000
    return session


def _answer(status: int, headers: dict[str, str]) -> MagicMock:
    response = MagicMock(status_code=status, ok=status < 400, headers=headers, text="")
    response.url = "u"
    return response


DEPRECATED = {
    "Deprecation": "true",
    "Warning": '299 - "POST /align/prompt is deprecated; use POST /prompt/sample_msa"',
}


def test_a_deprecated_call_warns_with_the_servers_text(session):
    deprecated = _answer(200, DEPRECATED)
    with (
        patch.object(requests.Session, "request", return_value=deprecated),
        pytest.warns(FutureWarning) as caught,
    ):
        session.request("POST", "v1/align/prompt")
    assert [str(w.message) for w in caught] == [
        "POST /align/prompt is deprecated; use POST /prompt/sample_msa"
    ]


def test_a_failed_deprecated_call_warns_before_raising(session):
    missing = _answer(404, DEPRECATED)
    with (
        patch.object(requests.Session, "request", return_value=missing),
        pytest.warns(FutureWarning, match="deprecated"),
        pytest.raises(HTTPError),
    ):
        session.request("POST", "v1/align/prompt")


@pytest.mark.parametrize(
    "headers",
    [{}, {"Warning": '199 - "transient"'}, {"Warning": "299 - unquoted"}],
    ids=["none", "other-code", "malformed"],
)
def test_other_responses_warn_nothing(session, headers):
    with (
        patch.object(requests.Session, "request", return_value=_answer(200, headers)),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        session.request("GET", "v1/jobs")


@pytest.mark.parametrize("status", [400, 403, 404])
def test_an_unsuccessful_status_raises_http_error_whatever_the_server(session, status):
    failed = _answer(status, {"Server": "CloudFront"})
    with (
        patch.object(requests.Session, "request", return_value=failed),
        pytest.raises(HTTPError) as raised,
    ):
        session.request("GET", "v1/jobs")
    assert raised.value.status_code == status


@pytest.mark.parametrize(
    "status,error",
    [(502, CloudFrontError), (503, CloudFrontError), (504, GatewayTimeoutError)],
)
def test_a_gateway_error_raises_its_own_error(session, status, error):
    failed = _answer(status, {"Server": "CloudFront"})
    with (
        patch.object(requests.Session, "request", return_value=failed),
        pytest.raises(error),
    ):
        session.request("GET", "v1/jobs")


def test_a_plain_http_backend_retries_a_stalled_read(tmp_path, monkeypatch):
    from openprotein import auth_store

    seen = []

    class Stalls(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if len(seen) == 1:
                time.sleep(1)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stalls)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )
    login = MagicMock(status_code=200, ok=True, headers={})
    login.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 900,
    }
    backend = f"http://127.0.0.1:{server.server_port}/"
    try:
        with patch.object(APISession, "post", return_value=login):
            session = APISession("user@x.com", "pw", backend=backend)
        session._expires_at = time.time() + 10_000
        assert session.get("v1/jobs/j", timeout=0.3).status_code == 200
    finally:
        server.shutdown()
    assert seen == ["/v1/jobs/j", "/v1/jobs/j"]


def _retry_responses(status: int, headers: dict[str, str]):
    import urllib3

    def make_request(*args, **kwargs):
        return urllib3.HTTPResponse(
            body=io.BytesIO(b"{}"),
            status=status,
            headers=headers,
            preload_content=False,
        )

    return make_request


@pytest.mark.parametrize(
    "status,headers,error",
    [
        (503, {"Server": "CloudFront"}, CloudFrontError),
        (502, {"Server": "CloudFront"}, CloudFrontError),
        (504, {"Server": "CloudFront"}, GatewayTimeoutError),
        (500, {}, HTTPError),
        (503, {}, HTTPError),
    ],
)
def test_a_spent_retry_budget_raises_the_sdks_own_error(
    session, status, headers, error
):
    """A retryable status that outlasts the retries surfaces as the SDK's error."""
    make_request = MagicMock(side_effect=_retry_responses(status, headers))
    with (
        patch("urllib3.connectionpool.HTTPConnectionPool._make_request", make_request),
        patch("urllib3.util.retry.Retry.sleep"),
        pytest.raises(error) as raised,
    ):
        session.request("GET", "v1/jobs/j")
    assert not isinstance(raised.value, requests.exceptions.RetryError)
    assert make_request.call_count == 5


@pytest.mark.parametrize(
    "method,path,attempts",
    [
        ("POST", "v1/models/esm2/embed", 5),
        ("GET", "v1/models/esm2", 5),
        ("POST", "v1/align/msa", 1),
    ],
)
def test_a_spent_model_route_budget_raises_the_sdks_error(
    session, method, path, attempts
):
    """Repeated 503s on a model route surface as the SDK's error, not a retry error."""
    make_request = MagicMock(side_effect=_retry_responses(503, {}))
    with (
        patch("urllib3.connectionpool.HTTPConnectionPool._make_request", make_request),
        patch("urllib3.util.retry.Retry.sleep"),
        pytest.raises(HTTPError) as raised,
    ):
        session.request(method, path, json={} if method == "POST" else None)
    assert not isinstance(raised.value, requests.exceptions.RetryError)
    assert make_request.call_count == attempts


def test_a_request_without_a_timeout_uses_the_session_timeout(session):
    """Every method sends the session timeout unless the caller passes one."""
    session.timeout = 42
    with patch.object(
        requests.Session, "request", return_value=_answer(200, {})
    ) as send:
        session.get("v1/jobs/j")
        session.put("v1/jobs/j", json={})
        session.delete("v1/jobs/j")
    assert [call.kwargs["timeout"] for call in send.call_args_list] == [42, 42, 42]


@pytest.mark.parametrize("timeout", [7, None])
def test_an_explicit_timeout_wins_over_the_session_timeout(session, timeout):
    """A timeout passed to a request, including None, is sent as given."""
    session.timeout = 42
    with patch.object(
        requests.Session, "request", return_value=_answer(200, {})
    ) as send:
        session.get("v1/jobs/j", timeout=timeout)
        session.post("v1/jobs/j", timeout=timeout)
    assert [call.kwargs["timeout"] for call in send.call_args_list] == [timeout] * 2
