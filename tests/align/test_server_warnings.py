"""Server deprecation warnings on align calls that carry an SDK deprecation."""

import io
import threading
import time
import warnings
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from openprotein import auth_store
from openprotein.align import api
from openprotein.base import APISession
from tests.conf import BACKEND

SERVER_WARNING = "POST /align/prompt is deprecated; use POST /prompt/sample_msa"
JOB = {
    "job_id": "job123",
    "status": "PENDING",
    "job_type": "/align/align",
    "created_date": "2023-01-01T00:00:00",
}


@pytest.fixture
def session(tmp_path, monkeypatch):
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


def _answer(body: dict) -> MagicMock:
    response = MagicMock(
        status_code=200,
        ok=True,
        headers={"Warning": f'299 - "{SERVER_WARNING}"'},
    )
    response.json.return_value = body
    return response


def _warnings(call) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


def _send(session, body):
    return patch.object(requests.Session, "request", return_value=_answer(body))


def test_prompt_post_warns_only_with_its_own_deprecation(session):
    """A deprecated `prompt_post` surfaces its SDK warning, not the server's."""
    with _send(session, JOB):
        messages = _warnings(lambda: api.prompt_post(session, "msa1", num_sequences=5))
    assert len(messages) == 1
    assert "openprotein.align.api.prompt_post" in messages[0]


def test_msa_post_warns_only_with_its_own_deprecation(session):
    """A deprecated `msa_post` surfaces its SDK warning, not the server's."""
    with _send(session, JOB):
        messages = _warnings(lambda: api.msa_post(session, seed="MKT"))
    assert len(messages) == 1
    assert "openprotein.align.api.msa_post" in messages[0]


def test_the_msa_file_upload_surfaces_no_server_warning(session):
    """The upload helper behind `AlignAPI.upload_msa` surfaces no server warning."""
    with _send(session, JOB):
        messages = _warnings(lambda: api._msa_post(session, io.BytesIO(b">a\nMKT")))
    assert messages == []


def test_other_align_calls_surface_the_server_warning(session):
    """A call without an SDK deprecation still surfaces the server's warning."""
    with _send(session, {"sequence": "MKT"}):
        messages = _warnings(lambda: api.get_seed(session, "msa1"))
    assert messages == [SERVER_WARNING]


def test_server_warnings_return_after_a_deprecated_call_fails(session):
    """Calls after a failed deprecated call surface server warnings again."""
    with _send(session, {"sequence": "MKT"}):
        with pytest.raises(ValidationError):
            api.prompt_post(session, "msa1", num_sequences=5)
        messages = _warnings(lambda: api.get_seed(session, "msa1"))
    assert messages == [SERVER_WARNING]


def _by_url(responses: dict[str, MagicMock]):
    def request(self, method, url, *args, **kwargs):
        return next(r for suffix, r in responses.items() if url.endswith(suffix))

    return patch.object(requests.Session, "request", request)


def test_a_token_refresh_inside_a_silenced_call_surfaces_its_server_warning(session):
    """Only the silenced request drops the server warning, not a refresh it triggers."""
    session._expires_at = time.time() - 1
    refreshed = _answer(
        {"access_token": "new", "refresh_token": "next", "expires_in": 900}
    )
    with (
        patch.object(session._token_store, "load", return_value=None),
        _by_url({"v1/auth/refresh": refreshed, "v1/align/msa": _answer(JOB)}),
    ):

        def silenced_call():
            with api._silence_server_warnings():
                session.post("v1/align/msa")

        messages = _warnings(silenced_call)
    assert messages == [SERVER_WARNING]


def test_a_call_on_another_thread_during_a_silenced_call_surfaces_its_warning(
    session,
):
    """Silencing applies to the calling thread's requests only."""
    other: list[str] = []

    def on_another_thread():
        thread = threading.Thread(
            target=lambda: other.extend(
                _warnings(lambda: api.get_seed(session, "msa1"))
            )
        )
        thread.start()
        thread.join()

    with _send(session, {"sequence": "MKT"}), api._silence_server_warnings():
        on_another_thread()
        inside = _warnings(lambda: api.get_seed(session, "msa1"))
    assert inside == []
    assert other == [SERVER_WARNING]
