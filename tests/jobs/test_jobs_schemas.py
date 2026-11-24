from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from requests import Response

from openprotein.jobs.schemas import Job, JobStatus, JobType


def test_job_status_done() -> None:
    assert JobStatus.SUCCESS.done() is True
    assert JobStatus.FAILURE.done() is True
    assert JobStatus.CANCELED.done() is True
    assert JobStatus.PENDING.done() is False
    assert JobStatus.RUNNING.done() is False
    assert JobStatus.RETRYING.done() is False


def test_job_status_cancelled() -> None:
    assert JobStatus.CANCELED.cancelled() is True
    assert JobStatus.SUCCESS.cancelled() is False
    assert JobStatus.FAILURE.cancelled() is False


@pytest.fixture
def base_job_dict() -> dict[str, Any]:
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
    job = Job.model_validate(base_job_dict)
    assert job.job_id == base_job_dict["job_id"]
    assert job.status == JobStatus.PENDING


def test_job_model_missing_optional_fields(base_job_dict: dict[str, Any]) -> None:
    del base_job_dict["start_date"]
    del base_job_dict["end_date"]
    del base_job_dict["prerequisite_job_id"]

    job = Job.model_validate(base_job_dict)
    assert job.job_id == "job_123"
    assert job.start_date is None
    assert job.end_date is None
    assert job.prerequisite_job_id is None


def test_job_create_from_dict(base_job_dict: dict[str, Any]) -> None:
    job = Job.create(base_job_dict)
    assert isinstance(job, Job)
    assert job.job_id == "job_123"


def test_job_metadata_round_trips(base_job_dict: dict[str, Any]) -> None:
    job = Job.model_validate(
        base_job_dict | {"metadata": {"svd": {"transform_job_id": "t"}}}
    )
    assert job.metadata == {"svd": {"transform_job_id": "t"}}


def test_job_metadata_defaults_to_none(base_job_dict: dict[str, Any]) -> None:
    assert Job.model_validate(base_job_dict).metadata is None


@pytest.mark.parametrize(
    "field",
    ["units_digest", "unit_count", "provider_id", "path", "version", "outputs"],
)
def test_job_does_not_declare_model_fields(field: str) -> None:
    assert field not in Job.model_fields


def test_job_create_from_mock_response(base_job_dict: dict[str, Any]) -> None:
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = base_job_dict

    job = Job.create(mock_response)
    assert isinstance(job, Job)
    assert job.job_id == "job_123"
    mock_response.json.assert_called_once()


@pytest.mark.parametrize(
    ("job_type", "schema"),
    [
        ("/models", "Job"),
        ("/models/esm2_t33_650M_UR50D/batch/embeddings", "Job"),
        ("/search/mmseqs2", "Job"),
        ("/align/prompt", "PromptJob"),
        ("/prompt/sample_msa", "PromptJob"),
    ],
)
def test_a_job_loads_as_the_schema_its_type_names(
    base_job_dict: dict[str, Any], job_type: str, schema: str
) -> None:
    import openprotein.models  # noqa: F401
    import openprotein.prompt  # noqa: F401

    job = Job.create(base_job_dict | {"job_type": job_type})

    assert type(job).__name__ == schema
