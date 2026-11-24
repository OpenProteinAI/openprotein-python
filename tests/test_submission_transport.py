"""Transport retry policy on the submission prefixes, against a local server."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
import urllib3
from requests.packages.urllib3.exceptions import NewConnectionError, SSLError  # type: ignore
from requests.packages.urllib3.util.retry import Retry  # type: ignore

from openprotein import auth_store
from openprotein.base import APISession


class _Server:
    def __init__(self, script):
        self.requests: list[tuple[str, str, bytes, str | None]] = []
        outer = self
        answers = list(script)

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                outer.requests.append(
                    (
                        self.command,
                        self.path,
                        body,
                        self.headers.get("Idempotency-Key"),
                    )
                )
                answer = answers.pop(0) if len(answers) > 1 else answers[0]
                if answer == "drop":
                    self.close_connection = True
                    self.connection.close()
                    return
                status, headers = answer
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _handle

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(Retry, "_sleep_backoff", lambda self: None)


@pytest.fixture
def serve():
    servers: list[_Server] = []

    def start(script):
        server = _Server(script)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )

    def build(backend):
        login = MagicMock(status_code=200, ok=True, headers={})
        login.json.return_value = {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 900,
        }
        with patch.object(APISession, "post", return_value=login):
            session = APISession("user@x.com", "pw", backend=backend)
        session._expires_at = time.time() + 10_000
        return session

    return build


def _send(session, method, url, **kwargs):
    return requests.Session.request(session, method, url, timeout=5, **kwargs)


@pytest.mark.parametrize("status", [413, 429, 500, 502, 504])
def test_a_submission_post_is_not_resent_after_an_error_response(
    serve, make_session, status
):
    server = serve([(status, {"Retry-After": "0"})])
    session = make_session(server.url)

    response = _send(session, "POST", server.url + "v1/models/m/in/out", json={"a": 1})

    assert response.status_code == status
    assert len(server.requests) == 1


@pytest.mark.parametrize("headers", [{}, {"Retry-After": "0"}])
def test_a_submission_post_is_resent_unchanged_after_a_503(
    serve, make_session, headers
):
    server = serve([(503, headers), (202, {})])
    session = make_session(server.url)

    response = _send(
        session,
        "POST",
        server.url + "v1/models/m/in/out",
        json={"a": 1},
        headers={"Idempotency-Key": "key-1"},
    )

    assert response.status_code == 202
    assert len(server.requests) == 2
    assert server.requests[0] == server.requests[1]
    assert server.requests[0][3] == "key-1"


def test_a_submission_post_is_not_resent_after_the_connection_drops(
    serve, make_session
):
    server = serve(["drop"])
    session = make_session(server.url)

    with pytest.raises(requests.exceptions.ConnectionError):
        _send(session, "POST", server.url + "v1/models/m/in/out", json={"a": 1})

    assert len(server.requests) == 1


def test_a_model_post_does_not_retry_an_unclassified_transport_error(make_session):
    session = make_session("https://test.invalid/")
    retry = session.get_adapter("https://test.invalid/v1/models/m/in/out").max_retries
    error = SSLError("TLS error")

    with pytest.raises(SSLError) as caught:
        retry.increment(method="POST", error=error)

    assert caught.value is error
    assert retry.increment(method="GET", error=error).total == retry.total - 1


def test_a_model_post_can_retry_a_failed_connection(make_session):
    session = make_session("https://test.invalid/")
    retry = session.get_adapter("https://test.invalid/v1/models/m/in/out").max_retries

    assert (
        retry.increment(method="POST", error=NewConnectionError(None, "refused")).total
        == retry.total - 1
    )


SAMPLING_PATHS = [
    "v1/data/prompts/sample_msa",
    "v1/prompt/sample_msa",
    "v1/align/prompt",
]


@pytest.mark.parametrize("path", SAMPLING_PATHS)
def test_a_sampling_post_is_not_resent_after_an_unclassified_error(make_session, path):
    session = make_session("https://test.invalid/")
    retry = session.get_adapter("https://test.invalid/" + path).max_retries
    error = SSLError("TLS error")

    with pytest.raises(SSLError) as caught:
        retry.increment(method="POST", error=error)

    assert caught.value is error
    assert retry.increment(method="GET", error=error).total == retry.total - 1
    assert (
        retry.increment(method="POST", error=NewConnectionError(None, "refused")).total
        == retry.total - 1
    )


@pytest.mark.parametrize("path", SAMPLING_PATHS)
@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_sampling_post_is_not_resent_after_a_server_error(
    serve, make_session, path, status
):
    server = serve([(status, {"Retry-After": "0"}), (202, {})])
    session = make_session(server.url)

    response = _send(session, "POST", server.url + path, json={"msa_id": "m"})

    assert response.status_code == status
    assert len(server.requests) == 1


def test_a_sampling_post_that_fails_after_it_was_sent_is_sent_once(
    serve, make_session, monkeypatch
):
    server = serve([(202, {})])
    session = make_session(server.url)
    send = urllib3.connectionpool.HTTPConnectionPool._make_request

    def fail_after_send(pool, *args, **kwargs):
        response = send(pool, *args, **kwargs)
        response.close()
        raise SSLError("TLS failure reading the response")

    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "_make_request", fail_after_send
    )

    with pytest.raises(requests.exceptions.SSLError):
        _send(session, "POST", server.url + "v1/data/prompts/sample_msa", json={"a": 1})

    assert len(server.requests) == 1


def test_a_submission_get_is_still_retried_after_a_502(serve, make_session):
    server = serve([(502, {}), (200, {})])
    session = make_session(server.url)

    response = _send(session, "GET", server.url + "v1/models/m")

    assert response.status_code == 200
    assert len(server.requests) == 2
