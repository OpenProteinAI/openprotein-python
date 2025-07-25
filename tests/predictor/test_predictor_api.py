"""Test the api for the predictor domain."""
from unittest.mock import MagicMock

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobType
from openprotein.predictor import api
from openprotein.predictor.schemas import PredictorMetadata


def test_predictor_list(mock_session: MagicMock):
    """Test predictor_list."""
    mock_session.get.return_value.json.return_value = []
    api.predictor_list(mock_session, limit=50, offset=5)
    mock_session.get.assert_called_once_with(
        "v1/predictor",
        params={"limit": 50, "offset": 5, "stats": False, "curve": False},
    )


def test_predictor_get(mock_session: MagicMock):
    """Test predictor_get."""
    mock_session.get.return_value.json.return_value = {
        "id": "p1",
        "name": "p1",
        "status": "SUCCESS",
        "created_date": "2023-01-01T00:00:00",
        "model_spec": {"type": "GP"},
        "training_dataset": {"assay_id": "a1", "properties": ["p1"]},
    }
    api.predictor_get(mock_session, "p1", include_stats=True)
    mock_session.get.assert_called_once_with(
        "v1/predictor/p1", params={"stats": True, "curve": False}
    )


def test_predictor_fit_gp_post(mock_session: MagicMock):
    """Test predictor_fit_gp_post."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job1",
        "status": "SUCCESS",
        "created_date": "2023-01-01T00:00:00",
        "job_type": JobType.predictor_train.value,
    }
    api.predictor_fit_gp_post(
        mock_session,
        assay_id="a1",
        properties=["p1"],
        feature_type="PLM",
        model_id="m1",
        reduction="mean",
    )
    mock_session.post.assert_called_once()
    call_args, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["features"]["type"] == "PLM"
    assert call_kwargs["json"]["features"]["reduction"] == "mean"
