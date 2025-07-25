from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from openprotein.jobs import api as jobs_api
from openprotein.jobs.schemas import Job, JobStatus, JobType


@pytest.fixture
def job_response_json() -> dict[str, Any]:
    """Provides a sample JSON response for a single job."""
    return {
        "job_id": "job_123",
        "job_type": "stub",
        "status": "SUCCESS",
        "created_date": datetime.now().isoformat(),
    }


def test_job_get(mock_session: MagicMock, job_response_json: dict[str, Any]) -> None:
    """Tests that job_get calls the correct endpoint and parses the response."""
    mock_session.get.return_value.json.return_value = job_response_json

    job = jobs_api.job_get(mock_session, "job_123")

    mock_session.get.assert_called_once_with("v1/jobs/job_123")
    assert isinstance(job, Job)
    assert job.job_id == "job_123"
    assert job.status == JobStatus.SUCCESS


def test_jobs_list_no_params(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """Tests jobs_list with no optional parameters."""
    mock_session.get.return_value.json.return_value = [job_response_json]

    jobs = jobs_api.jobs_list(mock_session)

    mock_session.get.assert_called_once_with("v1/jobs", params={})
    assert isinstance(jobs, list)
    assert len(jobs) == 1
    assert jobs[0].job_id == "job_123"


def test_jobs_list_with_all_params(
    mock_session: MagicMock, job_response_json: dict[str, Any]
) -> None:
    """Tests jobs_list with all optional parameters provided."""
    mock_session.get.return_value.json.return_value = [job_response_json]

    now = datetime.now()

    jobs_api.jobs_list(
        session=mock_session,
        status=JobStatus.SUCCESS,
        job_type=JobType.stub,
        assay_id="assay_456",
        more_recent_than=now.isoformat(),
        limit=50,
    )

    expected_params = {
        "status": "SUCCESS",
        "job_type": "stub",
        "assay_id": "assay_456",
        "more_recent_than": now.isoformat(),
        "limit": 50,
    }

    mock_session.get.assert_called_once_with("v1/jobs", params=expected_params)
