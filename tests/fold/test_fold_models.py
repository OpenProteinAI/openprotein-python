"""Test the models for the fold domain."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from openprotein.base import APISession
from openprotein.common.model_metadata import ModelDescription, ModelMetadata
from openprotein.fold.alphafold2 import AlphaFold2Model
from openprotein.fold.esmfold import ESMFoldModel
from openprotein.fold.future import FoldResultFuture
from openprotein.fold.models import FoldModel
from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs.schemas import JobStatus, JobType


@pytest.fixture
def mock_session():
    """Fixture for a mocked APISession."""
    return MagicMock(spec=APISession)


def test_fold_model_create(mock_session):
    """Test FoldModel.create."""
    # Test creating a known model
    model = FoldModel.create(mock_session, "esmfold")
    assert isinstance(model, ESMFoldModel)

    # Test creating another known model
    model = FoldModel.create(mock_session, "alphafold2")
    assert isinstance(model, AlphaFold2Model)

    # Test creating a generic model
    model = FoldModel.create(mock_session, "some-other-model", default=FoldModel)
    assert isinstance(model, FoldModel)
    assert not isinstance(model, (ESMFoldModel, AlphaFold2Model))

    # Test ValueError for unsupported model
    with pytest.raises(ValueError):
        FoldModel.create(mock_session, "unsupported-model")


@patch("openprotein.fold.api.fold_model_get")
def test_fold_model_get_metadata(mock_fold_model_get, mock_session):
    """Test FoldModel.get_metadata and caching."""
    mock_metadata = ModelMetadata(
        model_id="esmfold",
        description=ModelDescription(summary="A test model"),
        dimension=128,
        output_types=["pdb"],
        input_tokens=["protein"],
        token_descriptions=[],
    )
    mock_fold_model_get.return_value = mock_metadata

    model = ESMFoldModel(session=mock_session, model_id="esmfold")

    # Access metadata for the first time
    metadata = model.metadata
    assert metadata == mock_metadata
    mock_fold_model_get.assert_called_once_with(mock_session, "esmfold")

    # Access metadata again to test caching
    metadata2 = model.metadata
    assert metadata2 == mock_metadata
    # Assert that the mock was not called again
    mock_fold_model_get.assert_called_once()


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_fold_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test FoldModel.fold."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="esmfold",
    )

    model = ESMFoldModel(session=mock_session, model_id="esmfold")
    future = model.fold(sequences=["SEQ1"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="esmfold",
        sequences=["SEQ1"],
        num_recycles=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"
