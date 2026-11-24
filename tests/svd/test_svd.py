"""L2 integration tests for the svd domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.base import APISession
from openprotein.embeddings.embeddings import EmbeddingsAPI
from openprotein.embeddings.models import EmbeddingModel
from openprotein.jobs import JobsAPI, JobStatus, JobType
from openprotein.svd.models import SVDModel
from openprotein.svd.schemas import SVDFitJob, SVDMetadata
from openprotein.svd.svd import SVDAPI


@pytest.fixture
def svd_api(mock_session: MagicMock):
    return SVDAPI(mock_session)


@pytest.fixture
def mock_svd_fit_job():
    return SVDFitJob(
        job_id="svd-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.svd_fit,
        created_date=datetime(2023, 1, 1),
    )


def test_list_svd(svd_api: SVDAPI, mock_svd_fit_job):
    svd_api.session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    svd_api.session.jobs.get_job.return_value = mock_svd_fit_job  # type: ignore

    # Mock the api call to return a list with one mock item
    with patch(
        "openprotein.svd.api.svd_list_get", return_value=[MagicMock()]
    ) as mock_list_get:
        # We also need to mock the SVDModel constructor that gets called
        with patch("openprotein.svd.models.SVDModel"):
            svd_api.list_svd()
            mock_list_get.assert_called_once()


def test_get_svd(svd_api: SVDAPI, mock_svd_fit_job):
    svd_id = "svd-123"
    svd_api.session.jobs = MagicMock(spec=JobsAPI)  # type: ignore
    svd_api.session.jobs.get_job.return_value = mock_svd_fit_job  # type: ignore

    with patch("openprotein.svd.api.svd_get") as mock_get:
        # Mock the SVDModel constructor so we don't test its internals here
        with patch("openprotein.svd.models.SVDModel"):
            svd_api.get_svd(svd_id)
            mock_get.assert_called_once_with(svd_api.session, svd_id)


def test_fit_svd_accepts_a_models_namespace_handle():
    """Both fit paths take a session.models handle without a string lookup."""
    from openprotein.embeddings.poet2 import PoET2Model
    from openprotein.prompt import PromptAPI
    from openprotein.svd import api as svd_api_module

    session = MagicMock()
    session.svd = SVDAPI(session)
    session.embedding = EmbeddingsAPI(session)
    session.prompt = MagicMock(spec=PromptAPI)
    handle = PoET2Model(session, "poet-2", metadata=MagicMock())

    job = MagicMock(spec=SVDFitJob)
    job.job_id = "svd-123"
    metadata = SVDMetadata(
        id="svd-123",
        status=JobStatus.PENDING,
        model_id="poet-2",
        n_components=1024,
    )
    with (
        patch.object(svd_api_module, "svd_fit_post", return_value=job) as fit_post,
        patch.object(svd_api_module, "svd_get", return_value=metadata),
    ):
        via_api = session.svd.fit_svd(model=handle, sequences=[b"ACGT"])
        via_handle = handle.fit_svd(sequences=[b"ACGT"])

    assert isinstance(via_api, SVDModel)
    assert isinstance(via_handle, SVDModel)
    assert fit_post.call_count == 2
    assert fit_post.call_args_list[0].kwargs["model_id"] == "poet-2"


@patch("openprotein.svd.api.svd_get")
def test_fit_svd(mock_get, svd_api: SVDAPI, mock_session: MagicMock):
    model_id = "model-1"
    sequences = [b"ACGT"]

    mock_fit_job = MagicMock(spec=SVDFitJob)
    mock_fit_job.job_id = "job-123"

    mock_get.return_value.json.return_value = SVDMetadata(
        id="svd-123",
        status=JobStatus.PENDING,
        model_id="model-123",
        n_components=1024,
    )

    mock_embedding_model = MagicMock(spec=EmbeddingModel)
    mock_embedding_model.id = model_id

    mock_embeddings_api = MagicMock(spec=EmbeddingsAPI)
    mock_embeddings_api.get_model.return_value = mock_embedding_model

    svd_api.session = MagicMock(spec=APISession)
    svd_api.session.embedding = mock_embeddings_api

    with patch(
        "openprotein.svd.api.svd_fit_post", return_value=mock_fit_job
    ) as mock_fit_post:
        result = svd_api.fit_svd(model=model_id, sequences=sequences)
        mock_embeddings_api.get_model.assert_called_once_with(model_id)
        mock_fit_post.assert_called_once()
        assert isinstance(result, SVDModel)
