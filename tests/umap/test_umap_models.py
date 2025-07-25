"""Test the model logic for the umap domain."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType, JobsAPI
from openprotein.umap.models import UMAPModel
from openprotein.umap.schemas import UMAPFitJob, UMAPMetadata


@pytest.fixture
def mock_umap_metadata():
    """Fixture for UMAPMetadata."""
    return UMAPMetadata(
        id="umap-123",
        status=JobStatus.SUCCESS,
        model_id="model-1",
        feature_type=FeatureType.PLM,
    )


@pytest.fixture
def mock_umap_fit_job():
    """Fixture for UMAPFitJob."""
    return UMAPFitJob(
        job_id="umap-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.umap_fit,
        created_date=datetime(2023, 1, 1),
    )


def test_umap_model_init_with_job(
    mock_session: MagicMock, mock_umap_fit_job, mock_umap_metadata
):
    """Test UMAPModel initialization with a job object."""
    with patch("openprotein.umap.api.umap_get", return_value=mock_umap_metadata) as mock_api_get:
        model = UMAPModel(session=mock_session, job=mock_umap_fit_job)
        mock_api_get.assert_called_once_with(mock_session, mock_umap_fit_job.job_id)
        assert model.id == "umap-123"


def test_umap_model_get_embeddings(
    mock_session: MagicMock, mock_umap_metadata, mock_umap_fit_job
):
    """Test the embeddings property."""
    mock_session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    mock_session.jobs.get_job.return_value = mock_umap_fit_job  # type: ignore
    model = UMAPModel(session=mock_session, metadata=mock_umap_metadata)

    # Mock the api calls made by the 'embeddings' property
    with patch(
        "openprotein.umap.api.embed_get_batch_result", return_value=b""
    ) as mock_get_batch, patch(
        "openprotein.umap.api.embed_batch_decode", return_value=[(1, 2)]
    ) as mock_decode, patch(
        "openprotein.umap.models.UMAPModel.get_inputs", return_value=["ACGT"]
    ):
        embeddings = model.embeddings
        mock_get_batch.assert_called_once_with(session=mock_session, job_id=model.id)
        mock_decode.assert_called_once()
        assert embeddings == [("ACGT", (1, 2))]
