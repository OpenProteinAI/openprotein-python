"""Test the model logic for the svd domain."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.jobs import JobStatus, JobType, JobsAPI
from openprotein.svd.models import SVDModel, SVDEmbeddingResultFuture
from openprotein.svd.schemas import SVDEmbeddingsJob, SVDFitJob, SVDMetadata


@pytest.fixture
def mock_svd_metadata():
    """Fixture for SVDMetadata."""
    return SVDMetadata(
        id="svd-123",
        status=JobStatus.SUCCESS,
        model_id="model-1",
        n_components=10,
    )


@pytest.fixture
def mock_svd_fit_job():
    """Fixture for a valid SVDFitJob instance."""
    return SVDFitJob(
        job_id="svd-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.svd_fit,
        created_date=datetime(2023, 1, 1),
    )


def test_svd_model_init_with_metadata(
    mock_session: MagicMock, mock_svd_metadata, mock_svd_fit_job
):
    """Test SVDModel initialization with metadata."""
    mock_session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    mock_session.jobs.get_job.return_value = mock_svd_fit_job  # type: ignore
    model = SVDModel(session=mock_session, metadata=mock_svd_metadata)
    mock_session.jobs.get_job.assert_called_once_with(job_id=mock_svd_metadata.id)  # type: ignore
    assert model.id == "svd-123"


def test_svd_model_init_with_job(
    mock_session: MagicMock, mock_svd_fit_job, mock_svd_metadata
):
    """Test SVDModel initialization with a job object."""
    with patch("openprotein.svd.api.svd_get", return_value=mock_svd_metadata) as mock_api_get:
        model = SVDModel(session=mock_session, job=mock_svd_fit_job)
        mock_api_get.assert_called_once_with(
            session=mock_session, svd_id=mock_svd_fit_job.job_id
        )
        assert model.id == "svd-123"


def test_svd_model_embed(mock_session: MagicMock, mock_svd_metadata, mock_svd_fit_job):
    """Test the embed method of SVDModel."""
    mock_session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    mock_session.jobs.get_job.return_value = mock_svd_fit_job  # type: ignore
    model = SVDModel(session=mock_session, metadata=mock_svd_metadata)

    sequences = [b"ACGT"]
    mock_embed_job_dict = {
        "job_id": "embed-job-123",
        "job_type": JobType.svd_embed.value,
        "status": JobStatus.SUCCESS,
        "created_date": "2023-01-01T00:00:00",
    }
    with patch(
        "openprotein.svd.api.svd_embed_post", return_value=mock_embed_job_dict
    ) as mock_embed_post:
        future = model.embed(sequences)
        mock_embed_post.assert_called_once_with(
            session=mock_session, svd_id=model.id, sequences=sequences
        )
        assert isinstance(future, SVDEmbeddingResultFuture)
        assert future.id == "embed-job-123"


def test_svd_embedding_result_future_get_item(mock_session: MagicMock):
    """Test the get_item override in SVDEmbeddingResultFuture."""
    mock_job = MagicMock(spec=SVDEmbeddingsJob)
    mock_job.job_id = "job-456"
    future = SVDEmbeddingResultFuture(session=mock_session, job=mock_job)
    sequence = b"TEST"

    with patch("openprotein.svd.api.embed_get_sequence_result") as mock_get_result, patch(
        "openprotein.svd.api.embed_decode"
    ) as mock_decode:
        future.get_item(sequence)
        mock_get_result.assert_called_once_with(mock_session, "job-456", sequence)
        mock_decode.assert_called_once()
