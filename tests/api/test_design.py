import pytest
from unittest.mock import MagicMock
from typing import List, Optional, Union
import io 
from unittest.mock import ANY
import json
from urllib.parse import urljoin
from datetime import datetime 


from openprotein.base import APISession
from openprotein.api.jobs import Job
from openprotein.models import DesignJobCreate, DesignResults, JobType, ModelCriterion, Criterion
from openprotein.api.design import load_job, create_design_job, get_design_results, DesignAPI, DesignFuture
from openprotein.base import BearerAuth

class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = "test_username"
        password = "test_password"
        super().__init__(username, password)

    def _get_auth_token(self, username, password):
        return BearerAuth('AUTHORIZED')

    def post(self, endpoint, data=None, json=None, **kwargs):
        return ResponseMock()

    def get(self, endpoint, **kwargs):
        return ResponseMock()

    def request(self, method, url, *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        return response


class ResponseMock:
    def __init__(self):
        super().__init__()
        self._json = {}
        self.headers = {}
        self.iter_content = MagicMock()
        self._content = None
        self.status_code =200
        self.raw = io.BytesIO()  # Create an empty raw bytes stream
        self.text = "blank"

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def json(self):
        return self._json

@pytest.fixture
def api_session_mock():
    return APISessionMock()

def test_load_job(api_session_mock):
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '12345', 'status':'SUCCESS','job_type': JobType.design}
    api_session_mock.get = MagicMock(return_value=response_mock)
    
    result = load_job(api_session_mock, job_id='12345')

    api_session_mock.get.assert_called_once_with('v1/jobs/12345')
    assert result.job_id == '12345'
    assert result.job_type == JobType.design


def test_create_design_job(api_session_mock):
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '12345', 'status':'SUCCESS','job_type': JobType.design}
    api_session_mock.post = MagicMock(return_value=response_mock)

    # assuming the correct data structure for `criteria` and the value for `assay_id`.
    # Modify as necessary
    criteria = [[ModelCriterion(criterion_type='model',
                                criterion = Criterion(target=1, weight=1, direction=">"),
                                measurement_name="activity",
                                model_id="model1232",
                                assay_id="123")]]
    assay_id = 'assay123'
    
    design_job = DesignJobCreate(assay_id=assay_id, criteria=criteria)
    result = create_design_job(api_session_mock, design_job)

    api_session_mock.post.assert_called_once_with('v1/workflow/design/genetic-algorithm', json=design_job.dict(exclude_none=True))
    assert result.job_id == '12345'
    assert result.job_type == JobType.design

def test_get_design_results(api_session_mock):
    response_mock = ResponseMock()
    response_mock._json = {
        'status': 'SUCCESS',
        'job_id': '12345',
        'created_date': '2023-07-25',
        'job_type': '/design',
        'start_date': '2023-07-25',
        'end_date': '2023-07-25',
        'assay_id': 'assay123',
        'num_rows': 1,
        'result': [
            {
                'step': 1,
                'sample_index': 0,
                'sequence': 'APLPA',
                'scores': [1, 2, 3],  # this should be a list of integers
                'subscores_metadata': [
                    [
                        {'score': 1, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 2, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 3, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}}
                    ]
                ],
                'umap1': 0.5,
                'umap2': 0.6
            }
        ]
    }
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = get_design_results(api_session_mock, job_id='12345')

    api_session_mock.get.assert_called_once_with('v1/workflow/design/12345', params={})
    assert result.job_id == '12345'
    assert result.status == 'SUCCESS'
    assert len(result.result) == 1
    assert result.result[0].sequence == 'APLPA'

def test_DesignAPI_load_job(api_session_mock):
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '12345', 'status':'SUCCESS','job_type': JobType.design}
    api_session_mock.get = MagicMock(return_value=response_mock)

    design_api = DesignAPI(api_session_mock)
    result = design_api.load_job(job_id='12345')

    api_session_mock.get.assert_called_once_with('v1/jobs/12345')
    assert isinstance(result, DesignFuture)
    assert result.job.job_id == '12345'
    assert result.job.job_type == JobType.design

def test_design_api_create_design_job(api_session_mock):
    # This is the input for create_design_job
    assay_id = 'assay123'
    criteria = [[ModelCriterion(criterion_type='model',
                                criterion = Criterion(target=1, weight=1, direction=">"),
                                measurement_name="activity",
                                model_id="model1232",
                                assay_id=assay_id)]]
    job_create_sample = DesignJobCreate(assay_id=assay_id, criteria=criteria)

    job_response = {
        'status': "SUCCESS",
        'job_id': "123",
        'job_type': JobType.design
    }
    
    response_mock = ResponseMock()
    response_mock._json = job_response
    api_session_mock.post = MagicMock(return_value=response_mock)

    design_api = DesignAPI(api_session_mock)
    design_future = design_api.create_design_job(job_create_sample)

    api_session_mock.post.assert_called_once_with(
        'v1/workflow/design/genetic-algorithm',
        json=job_create_sample.dict(exclude_none=True) 
    )
    
    # Check the returned object
    assert isinstance(design_future, DesignFuture)
    assert design_future.id == '123'

def test_design_future_get(api_session_mock):
    # This is what get_results will return
    results = {
        'status': 'SUCCESS',
        'job_id': '12345',
        'created_date': '2023-07-25',
        'job_type': '/design',
        'start_date': '2023-07-25',
        'end_date': '2023-07-25',
        'assay_id': 'assay123',
        'num_rows': 1,
        'result': [
            {
                'step': 1,
                'sample_index': 0,
                'sequence': 'APLPA',
                'scores': [1, 2, 3],  # this should be a list of integers
                'subscores_metadata': [
                    [
                        {'score': 1, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 2, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 3, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}}
                    ]
                ],
                'umap1': 0.5,
                'umap2': 0.6
            }
        ]
    }

    design_results = DesignResults(**results)

    DesignFuture.get_results = MagicMock(return_value=design_results)

    job_sample = {
        'status': 'SUCCESS',
        'job_id': '12345',
        'created_date': datetime.now(),
        'job_type': JobType.design,
        'start_date': datetime.now(),
        'end_date': datetime.now(),
        'assay_id': 'assay123',
    }
    job_instance = Job(**job_sample)

    design_future = DesignFuture(api_session_mock, job_instance)
    results = design_future.get(verbose=False)

    # Verify that get_results was called with the correct arguments
    DesignFuture.get_results.assert_called_once_with(page_offset=0, page_size=1000)

    # Verify that the returned results are correct
    assert results == design_results.result

def test_design_api_get_design_results(api_session_mock):
    # Define the sample job
    job_sample = {
        'status': 'SUCCESS',
        'job_id': '12345',
        'created_date': '2023-07-25',
        'job_type': '/design',
        'start_date': '2023-07-25',
        'end_date': '2023-07-25',
        'assay_id': 'assay123',
        'num_rows': 1,
        'result': [
            {
                'step': 1,
                'sample_index': 0,
                'sequence': 'APLPA',
                'scores': [1, 2, 3],
                'subscores_metadata': [
                    [
                        {'score': 1, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 2, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}},
                        {'score': 3, 'metadata': {'y_mu': 0.1, 'y_var': 0.2}}
                    ]
                ],
                'umap1': 0.5,
                'umap2': 0.6
            }
        ]
    }

    response_mock = ResponseMock()
    response_mock._json = job_sample
    api_session_mock.get = MagicMock(return_value=response_mock)  # Mock the get method

    # Instantiate DesignAPI and call get_design_results
    api = DesignAPI(api_session_mock)
    results = api.get_design_results(job_id='12345', page_size=100, page_offset=0)

    # Verify that get_design_results was called with the correct arguments
    api_session_mock.get.assert_called_once_with('v1/workflow/design/12345', params={'page_size': 100, 'page_offset': 0})

    # Verify that the correct results were returned
    assert results == DesignResults(**job_sample)
