"""Test the schemas for the embeddings domain."""

import numpy as np
import pytest
from pydantic import ValidationError

from openprotein.embeddings.schemas import (
    AttnJob,
    EmbeddedSequence,
    EmbeddingsJob,
    GenerateJob,
    LogitsJob,
    ScoreIndelJob,
    ScoreJob,
    ScoreSingleSiteJob,
)
from openprotein.jobs import JobStatus, JobType


def test_embedded_sequence_behavior():
    """Test the tuple-like behavior of EmbeddedSequence."""
    sequence = b"ACGT"
    embedding = np.array([1.0, 2.0, 3.0])
    embedded_sequence = EmbeddedSequence(sequence=sequence, embedding=embedding)

    # Test __iter__
    s, e = embedded_sequence
    assert s == sequence
    assert np.array_equal(e, embedding)

    # Test __len__
    assert len(embedded_sequence) == 2

    # Test __getitem__
    assert embedded_sequence[0] == sequence
    assert np.array_equal(embedded_sequence[1], embedding)


def test_embedded_sequence_with_invalid_index():
    """Test that IndexError is raised for out-of-bounds access."""
    embedded_sequence = EmbeddedSequence(
        sequence=b"ACGT", embedding=np.array([1.0, 2.0])
    )
    with pytest.raises(IndexError):
        _ = embedded_sequence[2]


@pytest.mark.parametrize(
    "job_class, expected_job_type",
    [
        (EmbeddingsJob, JobType.embeddings_embed),
        (AttnJob, JobType.embeddings_attn),
        (LogitsJob, JobType.embeddings_logits),
        (ScoreJob, JobType.poet_score),
        (ScoreIndelJob, JobType.poet_score_indel),
        (ScoreSingleSiteJob, JobType.poet_single_site),
        (GenerateJob, JobType.poet_generate),
    ],
)
def test_job_schemas_job_type(job_class, expected_job_type):
    """Test that each job schema has the correct default job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": "2023-01-01T00:00:00",
    }
    job = job_class(**job_data)
    assert job.job_type == expected_job_type


def test_embeddings_job_with_reduced_type():
    """Test that EmbeddingsJob can handle the reduced embedding job type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_embed_reduced,
        "created_date": "2023-01-01T00:00:00",
    }
    job = EmbeddingsJob(**job_data)
    assert job.job_type == JobType.embeddings_embed_reduced


def test_job_schemas_invalid_job_type():
    """Test that validation fails for an incorrect job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": "some_other_type",  # Invalid job type
        "created_date": "2023-01-01T00:00:00",
    }
    with pytest.raises(ValidationError):
        EmbeddingsJob(**job_data)
