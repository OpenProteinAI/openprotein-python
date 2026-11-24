"""``wait(timeout=)`` raises ``TimeoutException`` when polls stall past the timeout."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest
import requests
from urllib3.util.retry import Retry

from openprotein import auth_store
from openprotein.base import APISession
from openprotein.errors import TimeoutException
from openprotein.jobs import Future, Job


class _Stalled(Future):
    def _get(self, **kwargs):
        return None


def _job() -> Job:
    return Job.model_validate(
        {
            "job_id": "j",
            "job_type": "/models",
            "status": "RUNNING",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )


@pytest.fixture
def stalled_backend(tmp_path, monkeypatch):
    class Stalls(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(1.5)

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
    with patch.object(APISession, "post", return_value=login):
        session = APISession(
            "user@x.com", "pw", backend=f"http://127.0.0.1:{server.server_port}/"
        )
    session._expires_at = time.time() + 10_000
    try:
        yield session
    finally:
        server.shutdown()


def test_a_stalled_poll_past_the_timeout_raises_timeout_exception(stalled_backend):
    future = _Stalled(stalled_backend, _job())
    with patch.object(Retry, "sleep"), pytest.raises(TimeoutException):
        future.wait_until_done(interval=0, timeout=0.3)


def test_a_failed_poll_within_the_timeout_raises_the_request_error(stalled_backend):
    stalled_backend.timeout = 0.2
    future = _Stalled(stalled_backend, _job())
    with (
        patch.object(Retry, "sleep"),
        pytest.raises(requests.exceptions.RequestException),
    ):
        future.wait_until_done(interval=0, timeout=600)
