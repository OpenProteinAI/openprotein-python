"""Test the Future classes for the predictor domain."""
from unittest.mock import MagicMock, patch

import numpy as np

from openprotein.predictor.prediction import PredictionResultFuture
from openprotein.predictor.validate import CVResultFuture


def test_prediction_result_future_get(mock_session: MagicMock):
    """Test the get() method of PredictionResultFuture."""
    mock_job = MagicMock()
    mock_job.job_id = "job-123"
    future = PredictionResultFuture(session=mock_session, job=mock_job)

    with patch(
        "openprotein.predictor.api.predictor_predict_get_batched_result"
    ) as mock_get_result, patch(
        "openprotein.predictor.api.decode_predict", return_value=(1, 2)
    ) as mock_decode:
        mu, var = future.get()
        mock_get_result.assert_called_once_with(mock_session, "job-123")
        mock_decode.assert_called_once()
        assert mu == 1
        assert var == 2


def test_cv_result_future_get(mock_session: MagicMock):
    """Test the get() method of CVResultFuture."""
    mock_job = MagicMock()
    mock_job.job_id = "job-456"
    future = CVResultFuture(session=mock_session, job=mock_job)

    with patch(
        "openprotein.predictor.api.predictor_crossvalidate_get"
    ) as mock_get_cv, patch(
        "openprotein.predictor.api.decode_crossvalidate", return_value=(1, 2, 3)
    ) as mock_decode_cv:
        y, mu, var = future.get()
        mock_get_cv.assert_called_once_with(mock_session, "job-456")
        mock_decode_cv.assert_called_once()
        assert y == 1
        assert mu == 2
        assert var == 3
