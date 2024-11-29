from datetime import datetime
from unittest.mock import MagicMock

import pytest
from openprotein.api.data import AssayDataset, AssayMetadata
from openprotein.api.train import (
    TrainFuture,
    TrainingAPI,
    TrainJob,
    _create_train_job_br,
    _create_train_job_gp,
    create_train_job,
    get_training_results,
)
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.jobs import Job, JobType
from tests.conf import BACKEND

# pending refactor
pytest.skip(allow_module_level=True)


@pytest.fixture
def mock_setup():
    session_mock = MagicMock(spec=APISession)
    metadata_mock = AssayMetadata(
        assay_name="Test Assay",
        assay_description="Description",
        assay_id="1234",
        original_filename="file.csv",
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=["m1", "m2"],
    )
    dataset_mock = AssayDataset(session_mock, metadata_mock)
    job_mock = Job(job_id="5678", status="PENDING", job_type="/workflow/train")
    return session_mock, dataset_mock, job_mock


def test_create_train_job(mock_setup):
    session_mock, dataset_mock, job_mock = mock_setup
    session_mock.post.return_value.json.return_value = job_mock.dict()
    job = create_train_job(session_mock, dataset_mock, "m1", "model1", True)
    assert job.job_id == "5678"
    session_mock.post.assert_called_with(
        "v1/workflow/train",
        params={"force_preprocess": "true"},
        json={"assay_id": "1234", "measurement_name": ["m1"], "model_name": "model1"},
    )


def test_create_train_job_br(mock_setup):
    session_mock, dataset_mock, job_mock = mock_setup
    session_mock.post.return_value.json.return_value = job_mock.dict()
    job = _create_train_job_br(session_mock, dataset_mock, "m1", "model1", True)
    assert job.job_id == "5678"
    session_mock.post.assert_called_with(
        "v1/workflow/train/br",
        params={"force_preprocess": "true"},
        json={"assay_id": "1234", "measurement_name": ["m1"], "model_name": "model1"},
    )


def test_create_train_job_gp(mock_setup):
    session_mock, dataset_mock, job_mock = mock_setup
    session_mock.post.return_value.json.return_value = job_mock.dict()
    job = _create_train_job_gp(session_mock, dataset_mock, "m1", "model1", True)
    assert job.job_id == "5678"
    session_mock.post.assert_called_with(
        "v1/workflow/train/gp",
        params={"force_preprocess": "true"},
        json={"assay_id": "1234", "measurement_name": ["m1"], "model_name": "model1"},
    )


def test_create_train_job_invalid_measurement_name(mock_setup):
    session_mock, dataset_mock, _ = mock_setup
    with pytest.raises(InvalidParameterError):
        create_train_job(session_mock, dataset_mock, "m3")


def test_create_train_job_not_enough_data_points(mock_setup):
    session_mock, dataset_mock, _ = mock_setup
    dataset_mock.metadata.num_rows = 2
    with pytest.raises(InvalidParameterError):
        create_train_job(session_mock, dataset_mock, "m1")


def test_get_training_results(mock_setup):
    session_mock, _, _ = mock_setup
    response_data = {
        "traingraph": [
            {"step": 1, "loss": 0.123, "tag": "taggy", "tags": {"tag1": "value1"}},
            {"step": 2, "loss": 0.112, "tag": "taggy2", "tags": {"tag2": "value2"}},
        ],
        "created_date": "2023-01-01T01:01:01",
        "job_id": "5678",
    }
    session_mock.get.return_value.json.return_value = response_data
    train_graph = get_training_results(session_mock, "5678")
    session_mock.get.assert_called_once_with("v1/workflow/train/5678")
    assert isinstance(train_graph, TrainJob)
    assert len(train_graph.traingraph) == 2
    assert train_graph.traingraph[0].step == 1
    assert train_graph.traingraph[0].loss == 0.123
    assert train_graph.traingraph[0].tag == "taggy"
    assert train_graph.traingraph[0].tags == {"tag1": "value1"}
    assert train_graph.created_date.isoformat() == "2023-01-01T01:01:01"
    assert train_graph.job_id == "5678"


def test_get_training_results2():
    session = MagicMock(spec=APISession)
    job_id = "1234"

    expected_train_graph = TrainJob(
        traingraph=[], created_date=datetime.now(), job_id="1234"
    )

    session.get.return_value.json.return_value = expected_train_graph.dict()

    training_api = TrainingAPI(session)

    result = training_api.get_training_results(job_id)

    assert isinstance(result, TrainFuture)
    session.get.assert_called_once_with(f"v1/workflow/train/{job_id}")


@pytest.mark.parametrize(
    "job_creator, endpoint",
    [
        (TrainingAPI.create_training_job, "v1/workflow/train"),
        (TrainingAPI._create_training_job_br, "v1/workflow/train/br"),
        (TrainingAPI._create_training_job_gp, "v1/workflow/train/gp"),
    ],
)
def test_create_training_job(job_creator, endpoint):
    measurement_name = "test_measurement"
    model_name = "test_model"
    force_preprocess = False

    mocked_response = MagicMock()
    mocked_response.json.return_value = {
        "status": "PENDING",
        "job_id": "mock_id",
        "job_type": JobType.workflow_train,
    }

    session = MagicMock(spec=APISession)
    session.post.return_value = mocked_response

    metadata_mock = AssayMetadata(
        assay_name="Test Assay",
        assay_description="Description",
        assay_id="1234",
        original_filename="file.csv",
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=[measurement_name, "test_measurement2"],
    )
    dataset_mock = AssayDataset(session, metadata_mock)

    # Ini
    training_api = TrainingAPI(session)

    result = job_creator(
        training_api, dataset_mock, measurement_name, model_name, force_preprocess
    )

    assert isinstance(result, TrainFuture)
    assert result.status == "PENDING"
    assert result.id == "mock_id"

    session.post.assert_called_once_with(
        endpoint,
        params={"force_preprocess": "false"},
        json={
            "assay_id": "1234",
            "measurement_name": ["test_measurement"],
            "model_name": "test_model",
        },
    )
