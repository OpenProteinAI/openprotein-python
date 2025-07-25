"""Test the model logic for the predictor domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobsAPI, JobStatus, JobType
from openprotein.predictor.models import PredictorModel, PredictorModelGroup
from openprotein.predictor.schemas import (
    Dataset,
    Features,
    ModelSpec,
    PredictorMetadata,
    PredictorTrainJob,
    PredictorType,
)


@pytest.fixture
def mock_predictor_metadata():
    """Fixture for a valid PredictorMetadata instance."""
    return PredictorMetadata(
        id="p1",
        name="p1",
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
        model_spec=ModelSpec(
            type=PredictorType.GP,
            features=Features(type=FeatureType.PLM, model_id="m1", reduction="mean"),
        ),
        training_dataset=Dataset(assay_id="a1", properties=["p1"]),
    )


@pytest.fixture
def mock_predictor_train_job():
    """Fixture for a valid PredictorTrainJob instance."""
    return PredictorTrainJob(
        job_id="p1",
        status=JobStatus.SUCCESS,
        job_type=JobType.predictor_train,
        created_date=datetime.now(),
    )


def test_predictor_model_init(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    """Test PredictorModel initialization with metadata."""
    # The constructor expects session.jobs to be a JobsAPI instance
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)
    assert model.id == "p1"


def test_predictor_model_predict(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    """Test the predict method of PredictorModel."""
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)

    with patch("openprotein.predictor.api.predictor_predict_post") as mock_predict_post:
        with patch("openprotein.predictor.prediction.PredictionResultFuture.create"):
            model.predict(sequences=[b"ACGT"])
            mock_predict_post.assert_called_once()


def test_predictor_model_group(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    """Test the PredictorModelGroup logic."""
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model1 = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)
    model2 = PredictorModel(
        session=mock_session,
        metadata=mock_predictor_metadata.model_copy(update={"id": "p2"}),
    )

    group = model1 | model2
    assert isinstance(group, PredictorModelGroup)
    assert len(group.__models__) == 2

    with patch(
        "openprotein.predictor.api.predictor_predict_multi_post"
    ) as mock_predict_multi:
        with patch("openprotein.predictor.prediction.PredictionResultFuture.create"):
            group.predict(sequences=[b"ACGT"])
            mock_predict_multi.assert_called_once()
            call_args, call_kwargs = mock_predict_multi.call_args
            assert call_kwargs["predictor_ids"] == ["p1", "p2"]
