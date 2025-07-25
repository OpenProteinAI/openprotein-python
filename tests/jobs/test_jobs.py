from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.errors import TimeoutException
from openprotein.jobs.futures import Future
from openprotein.jobs.jobs import JobsAPI
from openprotein.jobs.schemas import Job, JobStatus, JobType


class ConcreteFuture(Future):
    """A concrete implementation of the abstract Future for testing purposes."""

    def get(self, verbose: bool = False, **kwargs) -> str:
        return "Result"


@pytest.fixture
def mock_job_object() -> Job:
    """Provides a reusable mock Job object."""
    return Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
    )


def test_jobs_api_get_job(mock_session: MagicMock, mock_job_object: Job) -> None:
    """Tests that JobsAPI.get_job() calls the underlying api function."""
    # Configure the mock session to return the job dictionary
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = mock_job_object.model_dump(
        mode="json"
    )

    api = JobsAPI(mock_session)
    job = api.get_job("job_123")

    mock_session.get.assert_called_once_with("v1/jobs/job_123")
    assert job.job_id == mock_job_object.job_id


@pytest.fixture
def pending_job_obj() -> Job:
    """Provides a reusable mock Job object in PENDING state."""
    return Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.PENDING,
        created_date=datetime.now(),
    )


def test_future_wait_until_done_success(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """Tests the polling logic of a future for a successful job."""
    mock_session.get.return_value.json.side_effect = [
        {
            "job_id": "job_123",
            "status": "RUNNING",
            "job_type": "stub",
            "created_date": datetime.now().isoformat(),
        },
        {
            "job_id": "job_123",
            "status": "SUCCESS",
            "job_type": "stub",
            "created_date": datetime.now().isoformat(),
        },
    ]

    future = ConcreteFuture(mock_session, pending_job_obj)
    result = future.wait_until_done(interval=0)

    assert result is True
    assert future.status == JobStatus.SUCCESS
    assert mock_session.get.call_count == 2
    mock_session.get.assert_called_with("v1/jobs/job_123")


def test_future_wait_until_done_failure(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """Tests the polling logic of a future for a failed job."""
    mock_session.get.return_value.json.side_effect = [
        {
            "job_id": "job_123",
            "status": "FAILURE",
            "job_type": "stub",
            "created_date": datetime.now().isoformat(),
        },
    ]

    future = ConcreteFuture(mock_session, pending_job_obj)
    result = future.wait_until_done()

    assert result is True
    assert future.status == JobStatus.FAILURE
    assert mock_session.get.call_count == 1


def test_future_wait_until_done_timeout(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """Tests that the future correctly times out if the job never finishes."""
    mock_session.get.return_value.json.return_value = {
        "job_id": "job_123",
        "status": "RUNNING",
        "job_type": "stub",
        "created_date": datetime.now().isoformat(),
    }

    future = ConcreteFuture(mock_session, pending_job_obj)

    with pytest.raises(TimeoutException):
        future.wait_until_done(interval=0.01, timeout=0.05)
