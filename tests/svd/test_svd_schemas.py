"""Test the schemas for the svd domain."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from openprotein.jobs import JobStatus, JobType
from openprotein.svd.schemas import SVDEmbeddingsJob, SVDFitJob, SVDMetadata


def test_svd_metadata_is_done():
    """Test the is_done() method on SVDMetadata."""
    metadata_done = SVDMetadata(
        id="svd-1",
        status=JobStatus.SUCCESS,
        model_id="model-1",
        n_components=10,
    )
    metadata_not_done = SVDMetadata(
        id="svd-1",
        status=JobStatus.PENDING,
        model_id="model-1",
        n_components=10,
    )
    assert metadata_done.is_done() is True
    assert metadata_not_done.is_done() is False


@pytest.mark.parametrize(
    "job_class, expected_job_type",
    [
        (SVDFitJob, JobType.svd_fit),
        (SVDEmbeddingsJob, JobType.svd_embed),
    ],
)
def test_job_schemas_job_type(job_class, expected_job_type):
    """Test that each job schema has the correct job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": expected_job_type.value,  # Must provide the correct type
    }
    job = job_class(**job_data)
    assert job.job_type == expected_job_type


def test_svd_job_schema_invalid_job_type():
    """Test that validation fails for an incorrect job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": "some_other_type",  # Invalid job type
    }
    with pytest.raises(ValidationError):
        SVDFitJob(**job_data)
