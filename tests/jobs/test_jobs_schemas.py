from datetime import datetime
from unittest.mock import MagicMock
from typing import Any

import pytest
from requests import Response

from openprotein.jobs.schemas import Job, JobStatus, JobType


def test_job_status_done() -> None:
    """Tests the done() method of the JobStatus enum."""
    assert JobStatus.SUCCESS.done() is True
    assert JobStatus.FAILURE.done() is True
    assert JobStatus.CANCELED.done() is True
    assert JobStatus.PENDING.done() is False
    assert JobStatus.RUNNING.done() is False
    assert JobStatus.RETRYING.done() is False


def test_job_status_cancelled() -> None:
    """Tests the cancelled() method of the JobStatus enum."""
    assert JobStatus.CANCELED.cancelled() is True
    assert JobStatus.SUCCESS.cancelled() is False
    assert JobStatus.FAILURE.cancelled() is False


@pytest.fixture
def base_job_dict() -> dict[str, Any]:
    """Provides a basic dictionary to instantiate a Job model."""
    return {
        "job_id": "job_123",
        "job_type": JobType.stub,
        "status": JobStatus.PENDING,
        "created_date": datetime.now(),
        "start_date": datetime.now(),
        "end_date": datetime.now(),
        "prerequisite_job_id": "job_000",
        "progress_message": "Starting up",
        "progress_counter": 0,
        "sequence_length": 150,
    }


def test_job_model_validation_success(base_job_dict: dict[str, Any]) -> None:
    """Tests successful validation of the Job model from a dictionary."""
    job = Job.model_validate(base_job_dict)
    assert job.job_id == base_job_dict["job_id"]
    assert job.status == JobStatus.PENDING


def test_job_model_missing_optional_fields(base_job_dict: dict[str, Any]) -> None:
    """Tests that the Job model validates correctly when optional fields are missing."""
    del base_job_dict["start_date"]
    del base_job_dict["end_date"]
    del base_job_dict["prerequisite_job_id"]

    job = Job.model_validate(base_job_dict)
    assert job.job_id == "job_123"
    assert job.start_date is None
    assert job.end_date is None
    assert job.prerequisite_job_id is None


def test_job_create_from_dict(base_job_dict: dict[str, Any]) -> None:
    """Tests the Job.create() factory method from a dictionary."""
    job = Job.create(base_job_dict)
    assert isinstance(job, Job)
    assert job.job_id == "job_123"


def test_job_create_from_mock_response(base_job_dict: dict[str, Any]) -> None:
    """Tests the Job.create() factory method from a mocked Response object."""
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = base_job_dict

    job = Job.create(mock_response)
    assert isinstance(job, Job)
    assert job.job_id == "job_123"
    mock_response.json.assert_called_once()
