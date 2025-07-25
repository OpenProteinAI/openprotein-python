"""Test the schemas for the umap domain."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType
from openprotein.umap.schemas import UMAPEmbeddingsJob, UMAPFitJob, UMAPMetadata


def test_umap_metadata_is_done():
    """Test the is_done() method on UMAPMetadata."""
    metadata_done = UMAPMetadata(
        id="umap-1",
        status=JobStatus.SUCCESS,
        model_id="model-1",
        feature_type=FeatureType.PLM,
    )
    metadata_not_done = UMAPMetadata(
        id="umap-1",
        status=JobStatus.PENDING,
        model_id="model-1",
        feature_type=FeatureType.PLM,
    )
    assert metadata_done.is_done() is True
    assert metadata_not_done.is_done() is False


@pytest.mark.parametrize(
    "job_class, expected_job_type",
    [
        (UMAPFitJob, JobType.umap_fit),
        (UMAPEmbeddingsJob, JobType.umap_embed),
    ],
)
def test_job_schemas_job_type(job_class, expected_job_type):
    """Test that each job schema has the correct job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": expected_job_type.value,
    }
    job = job_class(**job_data)
    assert job.job_type == expected_job_type


def test_umap_job_schema_invalid_job_type():
    """Test that validation fails for an incorrect job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": "some_other_type",
    }
    with pytest.raises(ValidationError):
        UMAPFitJob(**job_data)
