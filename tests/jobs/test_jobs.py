from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.errors import JobFailedException, TimeoutException
from openprotein.jobs.futures import Future
from openprotein.jobs.jobs import JobsAPI
from openprotein.jobs.schemas import Job, JobStatus, JobType


class ConcreteFuture(Future):
    """A concrete implementation of the abstract Future for testing purposes."""

    def _get(self, verbose: bool = False, **kwargs) -> str:
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


def test_jobs_api_list_with_page_params(
    mock_session: MagicMock, mock_job_object: Job
) -> None:
    """JobsAPI.list forwards page_size/page_offset to the wire."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        mock_job_object.model_dump(mode="json")
    ]

    api = JobsAPI(mock_session)
    jobs = api.list(page_size=20, page_offset=40)

    mock_session.get.assert_called_once_with(
        "v1/jobs", params={"page_size": 20, "page_offset": 40}
    )
    assert len(jobs) == 1
    assert jobs[0].job_id == mock_job_object.job_id


def test_jobs_api_list_default_sends_page_size(
    mock_session: MagicMock, mock_job_object: Job
) -> None:
    """Default JobsAPI.list call sends `page_size=100`."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        mock_job_object.model_dump(mode="json")
    ]

    api = JobsAPI(mock_session)
    api.list()

    mock_session.get.assert_called_once_with("v1/jobs", params={"page_size": 100})


def test_jobs_api_list_legacy_limit_aliases_page_size(
    mock_session: MagicMock, mock_job_object: Job
) -> None:
    """Deprecated `limit` arg is forwarded as `page_size` on the wire."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        mock_job_object.model_dump(mode="json")
    ]

    api = JobsAPI(mock_session)
    api.list(limit=25)

    mock_session.get.assert_called_once_with("v1/jobs", params={"page_size": 25})


def test_jobs_api_list_rejects_both_limit_and_page_size(
    mock_session: MagicMock,
) -> None:
    """Passing both `limit` and `page_size` raises ValueError."""
    api = JobsAPI(mock_session)
    with pytest.raises(ValueError, match="page_size or limit"):
        api.list(limit=10, page_size=20)
    mock_session.get.assert_not_called()


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


def test_future_get_raises_on_failed_job(
    mock_session: MagicMock,
) -> None:
    """`.get()` on a failed Future raises JobFailedException with the failure_message."""
    failed_job = Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.FAILURE,
        created_date=datetime.now(),
        failure_message="something went wrong",
    )

    future = ConcreteFuture(mock_session, failed_job)

    with pytest.raises(JobFailedException) as exc_info:
        future.get()

    assert exc_info.value.job_id == "job_123"
    assert exc_info.value.failure_message == "something went wrong"
    assert "something went wrong" in str(exc_info.value)


def test_future_get_raises_on_failed_job_without_message(
    mock_session: MagicMock,
) -> None:
    """`.get()` on a failed Future without a failure_message still raises."""
    failed_job = Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.FAILURE,
        created_date=datetime.now(),
    )

    future = ConcreteFuture(mock_session, failed_job)

    with pytest.raises(JobFailedException) as exc_info:
        future.get()

    assert exc_info.value.job_id == "job_123"
    assert exc_info.value.failure_message is None


def test_future_wait_raises_on_failed_job(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """`.wait()` on a job that polls into FAILURE raises JobFailedException."""
    mock_session.get.return_value.json.side_effect = [
        {
            "job_id": "job_123",
            "status": "FAILURE",
            "job_type": "stub",
            "created_date": datetime.now().isoformat(),
            "failure_message": "boom",
        },
    ]

    future = ConcreteFuture(mock_session, pending_job_obj)

    with patch("time.sleep"):
        with pytest.raises(JobFailedException) as exc_info:
            future.wait(interval=0)

    assert exc_info.value.failure_message == "boom"


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
        future.wait_until_done(interval=0.01, timeout=1)
