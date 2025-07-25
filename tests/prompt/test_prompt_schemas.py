from datetime import datetime

import pytest
from pydantic import ValidationError

from openprotein.jobs import JobStatus, JobType
from openprotein.prompt.schemas import PromptJob, PromptMetadata, QueryMetadata


def test_prompt_job_schema():
    """Test successful creation of a PromptJob."""
    job_data = {
        "job_id": "prompt-123",
        "job_type": JobType.align_prompt,
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
    }
    job = PromptJob(**job_data)
    assert job.job_id == "prompt-123"
    assert job.prompt_id == "prompt-123"


def test_prompt_metadata_schema():
    """Test successful creation of PromptMetadata."""
    metadata_data = {
        "id": "prompt-meta-123",
        "name": "Test Prompt",
        "description": "A test prompt.",
        "created_date": datetime.now(),
        "num_replicates": 5,
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
    }
    metadata = PromptMetadata(**metadata_data)
    assert metadata.id == "prompt-meta-123"
    assert metadata.name == "Test Prompt"


def test_prompt_metadata_schema_optional_fields():
    """Test PromptMetadata with optional fields omitted."""
    metadata_data = {
        "id": "prompt-meta-456",
        "name": "Another Test Prompt",
        "created_date": datetime.now(),
        "num_replicates": 2,
        "status": JobStatus.PENDING,
    }
    metadata = PromptMetadata(**metadata_data)
    assert metadata.description is None
    assert metadata.job_id is None


def test_prompt_metadata_validation_error():
    """Test validation error for missing required fields in PromptMetadata."""
    with pytest.raises(ValidationError):
        PromptMetadata(
            id="prompt-meta-789",
            name="Incomplete Prompt",
            num_replicates=1,
            status=JobStatus.PENDING,
            # Missing created_date
        )  # type: ignore - meant to be missing


def test_query_metadata_schema():
    """Test successful creation of QueryMetadata."""
    query_data = {
        "id": "query-123",
        "created_date": datetime.now(),
    }
    query = QueryMetadata(**query_data)
    assert query.id == "query-123"


def test_query_metadata_validation_error():
    """Test validation error for missing required fields in QueryMetadata."""
    with pytest.raises(ValidationError):
        QueryMetadata(id="query-456")  # type: ignore - meant to be missing
