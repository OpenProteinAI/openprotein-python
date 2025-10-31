"""L2 integration tests for the umap domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.common import FeatureType
from openprotein.embeddings.models import EmbeddingModel
from openprotein.jobs import JobsAPI, JobStatus, JobType
from openprotein.umap.models import UMAPModel
from openprotein.umap.schemas import UMAPFitJob
from openprotein.umap.umap import UMAPAPI


@pytest.fixture
def umap_api(mock_session: MagicMock):
    """Fixture to create a UMAPAPI instance."""
    return UMAPAPI(mock_session)


@pytest.fixture
def mock_umap_fit_job():
    """Fixture for a valid UMAPFitJob instance."""
    return UMAPFitJob(
        job_id="umap-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.umap_fit,
        created_date=datetime(2023, 1, 1),
    )


def test_list_umap(umap_api: UMAPAPI, mock_umap_fit_job):
    """Test the list_umap method."""
    umap_api.session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    umap_api.session.jobs.get_job.return_value = mock_umap_fit_job  # type: ignore
    with patch("openprotein.umap.api.umap_list_get", return_value=[MagicMock()]):
        with patch("openprotein.umap.models.UMAPModel"):
            umap_api.list_umap()


def test_get_umap(umap_api: UMAPAPI, mock_umap_fit_job, mock_session: MagicMock):
    """Test the get_umap method."""
    umap_id = "umap-123"
    umap_api.session = mock_session
    umap_api.session.jobs = MagicMock(spec=JobsAPI)
    umap_api.session.jobs.get_job.return_value = mock_umap_fit_job

    with patch("openprotein.umap.api.umap_get"):
        with patch("openprotein.umap.models.UMAPModel"):
            umap_api.get_umap(umap_id)


def test_fit_umap(umap_api: UMAPAPI, mock_session: MagicMock):
    """Test the fit_umap method."""
    model = MagicMock(spec=EmbeddingModel)
    model.id = "model-1"

    mock_fit_job = MagicMock(spec=UMAPFitJob)
    mock_fit_job.job_id = "job-123"

    mock_session.get.return_value.json.return_value = {
        "id": "umap-123",
        "status": "PENDING",
        "model_id": "model-123",
        "feature_type": "PLM",
    }

    with patch(
        "openprotein.umap.api.umap_fit_post", return_value=mock_fit_job
    ) as mock_fit_post:
        umap_api.fit_umap(model=model, reduction="MEAN")
        mock_fit_post.assert_called_once()
