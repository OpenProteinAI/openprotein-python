"""Shared submission retry: a 425 answer is resubmitted, body and key unchanged."""

from unittest.mock import MagicMock

import pytest

from openprotein.errors import HTTPError
from openprotein.jobs._submission import post_with_retry


def _response(status_code: int, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    return response


def test_post_with_retry_returns_the_first_success():
    session = MagicMock()
    session.post.return_value = _response(200)

    response = post_with_retry(
        session, "v1/models/m/in/out", json={"a": 1}, headers={"Idempotency-Key": "k"}
    )

    assert response.status_code == 200
    session.post.assert_called_once_with(
        "v1/models/m/in/out", json={"a": 1}, headers={"Idempotency-Key": "k"}
    )


def test_post_with_retry_resubmits_the_identical_body_and_key_on_425(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "openprotein.jobs._submission.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    session = MagicMock()
    session.post.side_effect = [
        HTTPError(_response(425, {"Retry-After": "2"})),
        _response(200),
    ]

    response = post_with_retry(
        session, "v1/models/m/in/out", json={"a": 1}, headers={"Idempotency-Key": "k"}
    )

    assert response.status_code == 200
    assert sleeps == [2.0]
    calls = session.post.call_args_list
    assert calls[0] == calls[1]


def test_post_with_retry_gives_up_after_the_bound(monkeypatch):
    monkeypatch.setattr("openprotein.jobs._submission.time.sleep", lambda _: None)
    session = MagicMock()
    session.post.side_effect = HTTPError(_response(425, {"Retry-After": "0"}))

    with pytest.raises(HTTPError) as exc:
        post_with_retry(
            session,
            "v1/models/m/in/out",
            json={},
            headers={"Idempotency-Key": "k"},
            max_retries=2,
        )

    assert exc.value.status_code == 425
    assert session.post.call_count == 3


def test_post_with_retry_reraises_other_statuses():
    session = MagicMock()
    session.post.side_effect = HTTPError(_response(422))

    with pytest.raises(HTTPError) as exc:
        post_with_retry(
            session, "v1/models/m/in/out", json={}, headers={"Idempotency-Key": "k"}
        )

    assert exc.value.status_code == 422
    assert session.post.call_count == 1
