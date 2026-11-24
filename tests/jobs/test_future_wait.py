"""`Future.wait` on jobs that end unsuccessfully."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openprotein.errors import JobFailedException
from openprotein.jobs import Job
from openprotein.jobs.futures import Future

JOB: dict[str, Any] = {
    "job_id": "job-1",
    "job_type": "/models",
    "status": "RUNNING",
    "created_date": "2026-09-05T08:00:00Z",
    "provider_id": "esm2_t33_650M_UR50D",
    "path": "/batch/embeddings",
    "args": {"sequences": ["MKT"]},
}


def _future(session, final_status: str) -> Future:
    session.get.return_value.json.return_value = JOB | {
        "status": final_status,
        "failure_message": "boom" if final_status == "FAILURE" else None,
    }
    return Future(session, Job.model_validate(JOB))


def test_wait_raises_for_a_canceled_job(mock_session: MagicMock) -> None:
    """`wait()` raises `JobFailedException` saying the job was canceled."""
    future = _future(mock_session, "CANCELED")
    with (
        patch("time.sleep"),
        patch.object(Future, "get") as get,
        pytest.raises(JobFailedException, match="canceled") as raised,
    ):
        future.wait(interval=0)
    assert raised.value.job_id == "job-1"
    get.assert_not_called()


def test_wait_raises_for_a_failed_job(mock_session: MagicMock) -> None:
    """`wait()` on a failed job raises with the job's failure message."""
    future = _future(mock_session, "FAILURE")
    with patch("time.sleep"), pytest.raises(JobFailedException, match="boom"):
        future.wait(interval=0)


def test_wait_until_done_returns_for_a_canceled_job(mock_session: MagicMock) -> None:
    """`wait_until_done()` reports the terminal status without raising."""
    future = _future(mock_session, "CANCELED")
    with patch("time.sleep"):
        assert future.wait_until_done(interval=0) is True
    assert future.cancelled()
