import pytest
from unittest.mock import MagicMock
from openprotein.base import APISession
from datetime import datetime
import io
from urllib.parse import urljoin

from typing import List, Optional, Union
from io import BytesIO
from unittest.mock import ANY
import json
from openprotein.base import BearerAuth
from openprotein.api.embedding import *
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
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def json(self):
        return self._json


class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = "test_username"
        password = "test_password"
        super().__init__(username, password)

    def get_auth_token(self, username, password):
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

@pytest.fixture
def api_session_mock():
    sess = APISessionMock()
    yield sess

PATH_PREFIX = 'v1/embeddings'

def test_embedding_models_get(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = ["model1", "model2"]
    api_session_mock.get = MagicMock(return_value=mock_response)
    assert embedding_models_get(api_session_mock) == ["model1", "model2"]
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + '/models')

def test_embedding_model_get(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = {
        "model_id": "1234",
        "description": {"citation_title":"citation_title", "summary":"summary", "doi":"doi"},
        "created_date": "2022-01-01T01:01:01",
        "model_name": "Model 1",
        "dimension": 128,
        "output_types": ["type1", "type2"],
        "input_tokens": ["token1", "token2"],
        "output_tokens": ["token3", "token4"],
        "token_descriptions":[]
    }
    api_session_mock.get = MagicMock(return_value=mock_response)
    metadata = embedding_model_get(api_session_mock, "1234")
    assert isinstance(metadata, ModelMetadata)
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + '/models/1234')


def test_embedding_get(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = ["AAAAA", "BBBBB"]
    api_session_mock.get = MagicMock(return_value=mock_response)
    sequences = embedding_get(api_session_mock, "1234")
    assert sequences == ["AAAAA".encode(), "BBBBB".encode()]
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + '/1234')

def test_embedding_get_sequences(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = ["XXX", "ZZZ"]
    api_session_mock.get = MagicMock(return_value=mock_response)
    sequences = embedding_get_sequences(api_session_mock, "1234")
    assert sequences == ["XXX".encode(), "ZZZ".encode()]
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + '/1234/sequences')

def test_embedding_get_sequence_result(api_session_mock):
    mock_response = ResponseMock()
    mock_response._content = b'someresult'
    api_session_mock.get = MagicMock(return_value=mock_response)
    job_id = "1234"
    sequence = b'AAA'

    result = embedding_get_sequence_result(api_session_mock, job_id, sequence)

    assert result == b'someresult'
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + f'/{job_id}/{sequence.decode()}')

def test_embedding_model_post(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = {'job_id': '1234', 'status':'SUCCESS','job_type':"/embeddings"}

    api_session_mock.post = MagicMock(return_value=mock_response)
    model_id = "model1"
    sequences = [b'sequence1', b'sequence2']
    reduction = "MEAN"

    job = embedding_model_post(api_session_mock, model_id, sequences, reduction)

    assert isinstance(job, Job)
    assert job.job_id == "1234"
    assert job.status == "SUCCESS"
    api_session_mock.post.assert_called_once_with(
        PATH_PREFIX + f'/models/{model_id}/embed',
        json={
            'sequences': [s.decode() for s in sequences],
            'reduction': reduction
        }
    )

def test_embedding_api_list_models(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = ['model1', 'model2']
    api_session_mock.get = MagicMock(return_value=mock_response)
    
    api = EmbeddingAPI(api_session_mock)
    models = api.list_models()

    assert len(models) == 2
    assert isinstance(models[0], ProtembedModel)
    assert isinstance(models[1], ProtembedModel)
    api_session_mock.get.assert_called_once_with(PATH_PREFIX + '/models')


def test_embedding_api_get_model(api_session_mock):
    model_id = 'model1'
    
    api = EmbeddingAPI(api_session_mock)
    model = api.get_model(model_id)

    assert isinstance(model, ProtembedModel)


def test_embedding_api_embed(api_session_mock):
    mock_response = ResponseMock()
    mock_response._json = {'job_id': '1234', 'status':'SUCCESS','job_type':"/embeddings"}
    api_session_mock.post = MagicMock(return_value=mock_response)
    model_id = "model1"
    sequences = [b'AAA', b'CCC']
    reduction = "MEAN"
    model = ProtembedModel(api_session_mock, model_id)

    api = EmbeddingAPI(api_session_mock)
    job = api.embed(model, sequences, reduction)

    assert isinstance(job, EmbeddingResultFuture)
    api_session_mock.post.assert_called_once_with(
        PATH_PREFIX + f'/models/{model_id}/embed',
        json={
            'sequences': [s.decode() for s in sequences],
            'reduction': reduction
        }
    )

def test_embedding_api_get_results(api_session_mock):
    # Given
    job = Job(**{"job_id":'1234',
              "status":"SUCCESS",
              'job_type':"/embeddings"})

    # When
    api = EmbeddingAPI(api_session_mock)
    result_future = api.get_results(job)

    # Then
    assert isinstance(result_future, EmbeddingResultFuture)

