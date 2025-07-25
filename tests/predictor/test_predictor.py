"""L2 integration tests for the predictor domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType
from openprotein.predictor.predictor import PredictorAPI
from openprotein.svd.models import SVDModel
from openprotein.svd.schemas import SVDMetadata
from openprotein.svd.svd import SVDAPI


@pytest.fixture
def predictor_api(mock_session: MagicMock):
    """Fixture to create a PredictorAPI instance with a mocked session."""
    return PredictorAPI(mock_session)


def test_fit_gp_with_svd_model(predictor_api: PredictorAPI, mock_session: MagicMock):
    """
    Test the integration of fit_gp when using an SVD model.
    """
    # 1. Setup
    # The final call in the chain is predictor_fit_gp_post
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "job_type": JobType.predictor_train.value,
        "created_date": "2023-01-01T00:00:00",
    }
    mock_svd_model = MagicMock(spec=SVDModel)
    predictor_api.session.svd = MagicMock(spec=SVDAPI)  # type: ignore - needed to mock
    predictor_api.session.svd.get_svd.return_value = mock_svd_model  # type: ignore - mock

    # 2. Execution
    predictor_api.fit_gp(
        assay="assay-1",
        properties=["p1"],
        model="svd-1",
        feature_type=FeatureType.SVD,
    )
    # 3. Verification
    predictor_api.session.svd.get_svd.assert_called_once_with("svd-1")  # type: ignore - mock
    mock_svd_model.fit_gp.assert_called_once()
