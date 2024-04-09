from unittest.mock import patch, MagicMock, ANY
from openprotein.api.predict import (
    PredictService,
    PredictFuture,
    get_prediction_results,
    get_single_site_prediction_results,
    SequenceDataset,
    SequenceData,
    Prediction,
    PredictJob,
    PredictSingleSiteJob,
)
from openprotein.base import APISession
from openprotein.api.train import TrainFuture
from openprotein.api.data import AssayMetadata

from urllib.parse import urljoin
import pytest
from datetime import datetime
from tests.conf import BACKEND
from openprotein.jobs import Job, JobType
from openprotein.base import BearerAuth
from requests import Response
import io

# pending refactor
pytest.skip(allow_module_level=True)


class ResponseMock:
    def __init__(self):
        super().__init__()
        self._json = {}
        self.headers = {}
        self.iter_content = MagicMock()
        self._content = None
        self.status_code = 200
        self.raw = io.BytesIO()  # Create an empty raw bytes stream
        self.text = "blank"

    @property
    def __class__(self):
        return Response

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def json(self):
        return self._json

    def get(self, key, default=None):
        return self._json.get(key, default)


class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = "test_username"
        password = "test_password"
        super().__init__(username, password, backend=BACKEND)

    def _get_auth_token(self, username, password):
        return BearerAuth("AUTHORIZED")

    def post(self, endpoint, data=None, json=None, **kwargs):
        return ResponseMock()

    def get(self, endpoint, **kwargs):
        return ResponseMock()

    def request(self, method, url, *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        return response


metadata_mock = AssayMetadata(
    assay_name="Test Assay",
    assay_description="Description",
    assay_id="1234",
    original_filename="file.csv",
    created_date=datetime.now(),
    num_rows=1000,
    num_entries=2000,
    measurement_names=["m1", "m2"],
    sequence_length=3,
)


@pytest.fixture
def api_session_mock():
    sess = APISessionMock()
    yield sess


def test_create_predict_job():
    # mock objects
    session = MagicMock(APISession)
    train_future = MagicMock(TrainFuture)
    train_future.id = "1234"
    train_future.status = "SUCCESS"
    train_future.assaymetadata = metadata_mock

    job = MagicMock(Job)
    job.job_id = "5678"

    sequences = ["AAA", "CCC"]

    # configure
    session.post.return_value.json.return_value = {
        "job_id": job.job_id,
        "status": "PENDING",
        "job_type": "/workflow/predict",
    }
    predict_api = PredictService(session)
    predict_api.create_predict_job(sequences, train_future)

    payload = {"sequences": sequences, "train_job_id": train_future.id}

    # Check that post was called with the correct arguments.
    session.post.assert_called_with("v1/workflow/predict", json=payload)


def test_create_predict_single_site():
    # mock objects
    session = MagicMock(APISession)
    train_future = MagicMock(TrainFuture)
    train_future.id = "1234"
    train_future.assaymetadata = metadata_mock

    job = MagicMock(Job)
    job.job_id = "5678"

    sequence = "AAA"

    # configure
    session.post.return_value.json.return_value = {
        "job_id": job.job_id,
        "status": "PENDING",
        "job_type": "/workflow/predict",
    }
    predict_api = PredictService(session)
    predict_api.create_predict_single_site(sequence, train_future)

    payload = {"sequence": sequence, "train_job_id": train_future.id}

    # Check that post was called with the correct arguments.
    session.post.assert_called_with("v1/workflow/predict/single_site", json=payload)


def test_get_prediction_results():
    # Given
    job_id = "1234"
    session = MagicMock()
    response_data = {
        "job_id": job_id,
        "status": "SUCCESS",
        "job_type": "/workflow/predict",
        "result": [
            {
                "sequence": "AAA",
                "predictions": [
                    {
                        "model_id": "M1",
                        "model_name": "Model1",
                        "properties": {"prop1": {"val1": 0.5}},
                    }
                ],
            }
        ],
    }
    session.get.return_value.json.return_value = response_data

    # When
    job = get_prediction_results(session, job_id)

    # Then
    assert isinstance(job, PredictJob)
    assert job.job_id == job_id
    assert job.result[0].sequence == "AAA"
    assert len(job.result[0].predictions) == 1
    assert job.result[0].predictions[0].model_id == "M1"


def test_get_single_site_prediction_results():
    # Given
    job_id = "1234"
    session = MagicMock()
    response_data = {
        "job_id": job_id,
        "status": "SUCCESS",
        "job_type": "/workflow/predict/single_site",
        "result": [
            {
                "position": 1,
                "amino_acid": "A",
                "predictions": [
                    {
                        "model_id": "M1",
                        "model_name": "Model1",
                        "properties": {"prop1": {"val1": 0.5}},
                    }
                ],
            }
        ],
    }
    session.get.return_value.json.return_value = response_data

    # When
    job = get_single_site_prediction_results(session, job_id)

    # Then
    assert isinstance(job, PredictSingleSiteJob)
    assert job.job_id == job_id
    assert job.result[0].position == 1
    assert job.result[0].amino_acid == "A"
    assert len(job.result[0].predictions) == 1
    assert job.result[0].predictions[0].model_id == "M1"
