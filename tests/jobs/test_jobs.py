import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from openprotein.errors import JobFailedException, TimeoutException
from openprotein.jobs.futures import Future
from openprotein.jobs.jobs import JobsAPI
from openprotein.jobs.schemas import Job, JobStatus, JobType


class ConcreteFuture(Future):
    def _get(self, verbose: bool = False, **kwargs) -> str:
        return "Result"


@pytest.fixture
def mock_job_object() -> Job:
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
    return Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.PENDING,
        created_date=datetime.now(),
    )


def test_future_wait_until_done_success(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
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
    mock_session.get.return_value.json.return_value = {
        "job_id": "job_123",
        "status": "RUNNING",
        "job_type": "stub",
        "created_date": datetime.now().isoformat(),
    }

    future = ConcreteFuture(mock_session, pending_job_obj)

    with pytest.raises(TimeoutException):
        future.wait_until_done(interval=0.01, timeout=1)


def test_jobs_api_list_with_name(mock_session: MagicMock, mock_job_object: Job) -> None:
    """JobsAPI.list forwards `name` to the wire alongside the page size."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        mock_job_object.model_dump(mode="json")
    ]

    api = JobsAPI(mock_session)
    api.list(name="my-run")

    mock_session.get.assert_called_once_with(
        "v1/jobs", params={"name": "my-run", "page_size": 100}
    )


def test_jobs_api_update(mock_session: MagicMock, mock_job_object: Job) -> None:
    """JobsAPI.update issues the PUT and returns the updated job."""
    mock_session.put.return_value.json.return_value = mock_job_object.model_dump(
        mode="json"
    ) | {"name": "my-run"}

    api = JobsAPI(mock_session)
    job = api.update("job_123", name="my-run")

    mock_session.put.assert_called_once_with("v1/jobs/job_123", json={"name": "my-run"})
    assert job.name == "my-run"


def test_future_rename(mock_session: MagicMock, mock_job_object: Job) -> None:
    """rename swaps in the job the server returns."""
    mock_session.put.return_value.json.return_value = mock_job_object.model_dump(
        mode="json"
    ) | {"name": "my-run", "description": "first pass"}

    future = ConcreteFuture(session=mock_session, job=mock_job_object)
    future.rename(name="my-run", description="first pass")

    mock_session.put.assert_called_once_with(
        "v1/jobs/job_123",
        json={"name": "my-run", "description": "first pass"},
    )
    assert future.name == "my-run"
    assert future.description == "first pass"


def test_future_rename_name_only_omits_description(
    mock_session: MagicMock, mock_job_object: Job
) -> None:
    """Renaming with only a name sends no description key."""
    mock_session.put.return_value.json.return_value = mock_job_object.model_dump(
        mode="json"
    ) | {"name": "my-run"}

    future = ConcreteFuture(session=mock_session, job=mock_job_object)
    future.rename(name="my-run")

    mock_session.put.assert_called_once_with("v1/jobs/job_123", json={"name": "my-run"})


def test_future_rename_preserves_extra_fields(mock_session: MagicMock) -> None:
    """rename carries over extra fields the same way refresh does."""
    job_with_extra = Job(
        job_id="job_123",
        job_type=JobType.stub,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
        N=5,
    )
    mock_session.put.return_value.json.return_value = {
        "job_id": "job_123",
        "job_type": "stub",
        "status": "SUCCESS",
        "created_date": datetime.now().isoformat(),
        "name": "my-run",
    }

    future = ConcreteFuture(session=mock_session, job=job_with_extra)
    future.rename(name="my-run")

    assert future.job.N == 5


def test_jobs_api_list_takes_more_recent_than_fourth(mock_session: MagicMock) -> None:
    """Positional order is status, job_type, assay_id, more_recent_than."""
    mock_session.get.return_value.json.return_value = []

    JobsAPI(mock_session).list(None, None, None, "2026-01-01T00:00:00")

    mock_session.get.assert_called_once_with(
        "v1/jobs",
        params={"more_recent_than": "2026-01-01T00:00:00", "page_size": 100},
    )


def test_jobs_api_list_takes_name_by_keyword_only(mock_session: MagicMock) -> None:
    with pytest.raises(TypeError):
        JobsAPI(mock_session).list(None, None, None, None, None, None, None, "my-run")


class _Clock:
    """A fake clock whose sleeps advance it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _running(mock_session: MagicMock) -> None:
    mock_session.get.return_value.json.return_value = {
        "job_id": "job_123",
        "status": "RUNNING",
        "job_type": "stub",
        "created_date": datetime.now().isoformat(),
    }


def test_wait_caps_each_poll_and_sleep_by_the_remaining_timeout(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """With a wait timeout, each poll and sleep is bounded by the remaining budget."""
    _running(mock_session)
    clock = _Clock()
    future = ConcreteFuture(mock_session, pending_job_obj)

    with patch("openprotein.jobs.futures.time", clock), pytest.raises(TimeoutException):
        future.wait_until_done(interval=3, timeout=5)

    assert clock.sleeps == [3, 2]
    assert [c.kwargs["timeout"] for c in mock_session.get.call_args_list] == [5, 2]


def test_wait_turns_a_poll_that_times_out_into_the_wait_timeout(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """A poll that fails once the budget is spent raises `TimeoutException`."""

    def slow(*args, **kwargs):
        time.sleep(0.1)
        raise requests.exceptions.ReadTimeout("slow")

    mock_session.get.side_effect = slow
    future = ConcreteFuture(mock_session, pending_job_obj)

    with pytest.raises(TimeoutException):
        future.wait_until_done(interval=0, timeout=0.05)


def test_a_refresh_without_a_timeout_polls_with_the_session_timeout(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """`refresh()` passes no timeout of its own, even after a budgeted wait."""
    _running(mock_session)
    future = ConcreteFuture(mock_session, pending_job_obj)
    with pytest.raises(TimeoutException):
        future.wait_until_done(interval=0, timeout=0)

    mock_session.get.reset_mock()
    future.refresh()

    mock_session.get.assert_called_once_with("v1/jobs/job_123")


def test_wait_raises_a_poll_error_while_the_budget_remains(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """A poll that fails with budget left raises its own request error."""
    mock_session.get.side_effect = requests.exceptions.ReadTimeout("slow")
    future = ConcreteFuture(mock_session, pending_job_obj)

    with pytest.raises(requests.exceptions.ReadTimeout):
        future.wait_until_done(interval=0, timeout=5)


def test_wait_without_a_timeout_polls_with_the_session_timeout(
    mock_session: MagicMock, pending_job_obj: Job
) -> None:
    """Without a wait timeout, polls pass no timeout of their own."""
    mock_session.get.return_value.json.return_value = {
        "job_id": "job_123",
        "status": "SUCCESS",
        "job_type": "stub",
        "created_date": datetime.now().isoformat(),
    }
    future = ConcreteFuture(mock_session, pending_job_obj)
    future.wait_until_done(interval=0)
    mock_session.get.assert_called_once_with("v1/jobs/job_123")
