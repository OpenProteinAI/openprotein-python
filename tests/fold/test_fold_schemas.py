"""Test the schemas for the fold domain."""

import json
from datetime import datetime

from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs.schemas import BatchJob, Job, JobStatus, JobType


def test_fold_metadata_creation():
    """Test successful creation of FoldMetadata."""
    metadata = FoldMetadata(
        job_id="test_job_id", model_id="test_model_id", args={"param": "value"}
    )
    assert metadata.job_id == "test_job_id"
    assert metadata.model_id == "test_model_id"
    assert metadata.args == {"param": "value"}


def test_fold_metadata_deserialization():
    """Test successful deserialization of FoldMetadata from JSON."""
    json_data = {
        "job_id": "test_job_id",
        "model_id": "test_model_id",
        "args": {"param": "value"},
    }
    metadata = FoldMetadata.model_validate(json_data)
    assert metadata.job_id == "test_job_id"
    assert metadata.model_id == "test_model_id"
    assert metadata.args == {"param": "value"}


def test_fold_job_creation():
    """Test successful creation of FoldJob."""
    now = datetime.now()
    job = FoldJob(
        job_id="test_job_id",
        job_type=JobType.embeddings_fold,
        status=JobStatus.PENDING,
        created_date=now,
    )
    assert job.job_id == "test_job_id"
    assert job.job_type == JobType.embeddings_fold
    assert job.status == "PENDING"
    assert job.created_date == now
    assert isinstance(job, Job)
    assert isinstance(job, BatchJob)
