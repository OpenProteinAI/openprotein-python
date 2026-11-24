from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from openprotein.jobs import _api as jobs_wire
from openprotein.jobs import api as jobs_api
from openprotein.jobs.schemas import Job, JobStatus, JobType


@pytest.fixture
def job_response_json() -> dict[str, Any]:
    return {
        "job_id": "job_123",
        "job_type": "stub",
        "status": "SUCCESS",
        "created_date": datetime.now().isoformat(),
    }


def test_job_get(mock_session: MagicMock, job_response_json: dict[str, Any]) -> None:
    mock_session.get.return_value.json.return_value = job_response_json

    job = jobs_api.job_get(mock_session, "job_123")

    mock_session.get.assert_called_once_with("v1/jobs/job_123")
    assert isinstance(job, Job)
    assert job.job_id == "job_123"
    assert job.status == JobStatus.SUCCESS


def test_jobs_list_no_params(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs = jobs_api.jobs_list(mock_session)

    mock_session.get.assert_called_once_with("v1/jobs", params={})
    assert isinstance(jobs, list)
    assert len(jobs) == 1
    assert jobs[0].job_id == "job_123"


def test_jobs_list_with_all_params(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    mock_session.get.return_value.json.return_value = [job_response_json]

    now = datetime.now()

    jobs_api.jobs_list(
        session=mock_session,
        status=JobStatus.SUCCESS,
        job_type=JobType.stub,
        assay_id="assay_456",
        name="my-run",
        more_recent_than=now.isoformat(),
        limit=50,
    )

    expected_params = {
        "status": "SUCCESS",
        "job_type": "stub",
        "assay_id": "assay_456",
        "name": "my-run",
        "more_recent_than": now.isoformat(),
        "limit": 50,
    }

    mock_session.get.assert_called_once_with("v1/jobs", params=expected_params)


def test_jobs_list_with_page_params(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs_api.jobs_list(session=mock_session, page_size=25, page_offset=50)

    mock_session.get.assert_called_once_with(
        "v1/jobs", params={"page_size": 25, "page_offset": 50}
    )


def test_jobs_list_page_offset_only(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """page_offset can be sent on its own."""
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs_api.jobs_list(session=mock_session, page_offset=10)

    mock_session.get.assert_called_once_with("v1/jobs", params={"page_offset": 10})


def test_jobs_list_legacy_limit_unchanged(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """Existing callers using `limit` still send only `limit` (backwards compat)."""
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs_api.jobs_list(session=mock_session, limit=10)

    mock_session.get.assert_called_once_with("v1/jobs", params={"limit": 10})


def test_jobs_list_with_name(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """`name` is forwarded as a query param."""
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs_api.jobs_list(session=mock_session, name="my-run")

    mock_session.get.assert_called_once_with("v1/jobs", params={"name": "my-run"})


def test_job_get_parses_name_and_description(mock_session: MagicMock) -> None:
    """Name and description are declared fields, not extras."""
    mock_session.get.return_value.json.return_value = {
        "job_id": "job_123",
        "job_type": "stub",
        "status": "SUCCESS",
        "created_date": datetime.now().isoformat(),
        "name": "my-run",
        "description": "first pass",
    }

    job = jobs_api.job_get(mock_session, "job_123")

    assert "name" in Job.model_fields
    assert "description" in Job.model_fields
    assert job.name == "my-run"
    assert job.description == "first pass"


def test_job_update_sends_only_supplied_fields(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """A name-only rename does not clear the description."""
    mock_session.put.return_value.json.return_value = job_response_json | {
        "name": "my-run"
    }

    job = jobs_wire.job_update(mock_session, "job_123", name="my-run")

    mock_session.put.assert_called_once_with("v1/jobs/job_123", json={"name": "my-run"})
    assert job.name == "my-run"


def test_job_update_sends_both_fields(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """Name and description are sent together when both are given."""
    mock_session.put.return_value.json.return_value = job_response_json | {
        "name": "my-run",
        "description": "first pass",
    }

    jobs_wire.job_update(
        mock_session, "job_123", name="my-run", description="first pass"
    )

    mock_session.put.assert_called_once_with(
        "v1/jobs/job_123", json={"name": "my-run", "description": "first pass"}
    )


def test_job_update_with_neither_field_sends_empty_body(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """Calling job_update with no name or description sends an empty body."""
    mock_session.put.return_value.json.return_value = job_response_json

    jobs_wire.job_update(mock_session, "job_123")

    mock_session.put.assert_called_once_with("v1/jobs/job_123", json={})


def test_jobs_list_takes_more_recent_than_fourth(mock_session: MagicMock) -> None:
    mock_session.get.return_value.json.return_value = []

    jobs_api.jobs_list(mock_session, None, None, None, "2026-01-01T00:00:00")

    mock_session.get.assert_called_once_with(
        "v1/jobs", params={"more_recent_than": "2026-01-01T00:00:00"}
    )
