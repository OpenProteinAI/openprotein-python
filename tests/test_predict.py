from unittest.mock import patch, MagicMock, ANY
from openprotein.api.predict import PredictFuture, PredictAPI, get_prediction_results, get_single_site_prediction_results
from openprotein.base import APISession
from openprotein.api.train import TrainFuture
from openprotein.models import SequenceDataset, SequenceData, AssayMetadata, Prediction, PredictJob, PredictSingleSiteJob
from openprotein.api.jobs import Job
import pytest 
from datetime import datetime 



metadata_mock = AssayMetadata(
    assay_name='Test Assay',
    assay_description='Description',
    assay_id='1234',
    original_filename='file.csv',
    created_date=datetime.now(),
    num_rows=1000,
    num_entries=2000,
    measurement_names=['m1', 'm2'],
    sequence_length=3
)

def test_create_predict_job():
    # mock objects
    session = MagicMock(APISession)
    train_future = MagicMock(TrainFuture)
    train_future.id = '1234'
    train_future.status = "SUCCESS"
    train_future.assaymetadata = metadata_mock

    job = MagicMock(Job)
    job.job_id = '5678'
    
    sequences =['AAA', 'CCC']

    # configure 
    session.post.return_value.json.return_value = {
            'job_id': job.job_id,
            'status': 'PENDING',  # Or any valid status
            'job_type': '/workflow/predict',  # Or any valid job type
        }
    predict_api = PredictAPI(session)
    predict_api.create_predict_job(sequences, train_future)

    payload = {'sequences': sequences, 'train_job_id': train_future.id}

    # Check that post was called with the correct arguments.
    session.post.assert_called_with('v1/workflow/predict', json=payload)

def test_create_predict_single_site():
    # mock objects
    session = MagicMock(APISession)
    train_future = MagicMock(TrainFuture)
    train_future.id = '1234'
    train_future.assaymetadata = metadata_mock

    job = MagicMock(Job)
    job.job_id = '5678'
    
    sequence = 'AAA'

    # configure 
    session.post.return_value.json.return_value = {
            'job_id': job.job_id,
            'status': 'PENDING',  # Or any valid status
            'job_type': '/workflow/predict',  # Or any valid job type
        }
    predict_api = PredictAPI(session)
    predict_api.create_predict_single_site(sequence, train_future)

    payload = {'sequence': sequence, 'train_job_id': train_future.id}

    # Check that post was called with the correct arguments.
    session.post.assert_called_with('v1/workflow/predict/single_site', json=payload)

# testing the functions get_prediction_results and get_single_site_prediction_results
def test_get_prediction_results():
    # Given
    job_id = '1234'
    session = MagicMock()
    response_data = {
        "job_id": job_id,
        "status": "SUCCESS",
        "job_type": "/workflow/predict",
        "result": [
            {"sequence": "AAA", "predictions": [{"model_id": "M1", "model_name": "Model1", "properties": {"prop1": {"val1": 0.5}}}]}
        ]
    }
    session.get.return_value.json.return_value = response_data

    # When
    job = get_prediction_results(session, job_id)

    # Then
    assert isinstance(job, PredictJob)
    assert job.job_id == job_id
    assert job.result[0].sequence == "AAA"
    assert len(job.result[0].predictions) == 1
    assert job.result[0].predictions[0].model_id == 'M1'


def test_get_single_site_prediction_results():
    # Given
    job_id = '1234'
    session = MagicMock()
    response_data = {
        "job_id": job_id,
        "status": "SUCCESS",
        "job_type": "/workflow/predict/single_site",
        "result": [
            {"position": 1, "amino_acid": "A", "predictions": [{"model_id": "M1", "model_name": "Model1", "properties": {"prop1": {"val1": 0.5}}}]}
        ]
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
    assert job.result[0].predictions[0].model_id == 'M1'
