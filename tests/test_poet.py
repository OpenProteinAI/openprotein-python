import pytest
import io
import csv
import requests
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from typing import List, Dict, Any
from requests import Response
from openprotein.api.poet import (
    APISession,
    poet_single_site_post,
    poet_single_site_get,
    poet_generate_post,
    poet_generate_get,
    poet_score_get,
    get_align_job_inputs,
    prompt_post,
    poet_score_post,
    msa_post,
    upload_prompt_post,
    MSASamplingMethod,
    PoetInputType,
    InvalidParameterError,
    MissingParameterError,
    APIError,
    Job
)
from unittest.mock import ANY
import json

with open('secrets.config') as f:
    secrets = json.load(f)

class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = secrets['username']  # Provide username
        password = secrets['password']  # Provide password
        super().__init__(username, password)

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
        self.raw = io.BytesIO()  # Create an empty raw bytes stream

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


def test_poet_single_site_post(api_session_mock):
    variant = 'AUGUCA'
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '12345', 'status':'SUCCESS','job_type':"/poet/single_site"}
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = poet_single_site_post(api_session_mock, prompt_id='prompt123', variant=variant)

    api_session_mock.post.assert_called_once_with(
        'v1/poet/single_site',
        params={'variant': variant, 'prompt_id': 'prompt123'}
    )
    assert result.job_id == '12345'


def test_poet_single_site_get(api_session_mock):
    job_id = '12345'
    results =   {"score": [ -1.0],"name": "name1", "sequence": b"X0X"}
    response_mock = ResponseMock()
    response_mock._json = {'result': [results], 'status':'SUCCESS','job_type':"/poet/single_site", "job_id":job_id}
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = poet_single_site_get(api_session_mock, job_id)

    api_session_mock.get.assert_called_once_with(
        'v1/poet/single_site',
        params={'job_id': job_id, 'page_size': 100, 'page_offset': 0}    )
    assert result.result == [results] 


def test_poet_generate_post(api_session_mock):
    prompt_id = 'prompt123'
    seed =111
    jobid = "gen123"
    response_mock = ResponseMock()
    response_mock._json = {'job_id': jobid, 'status':'PENDING','job_type':"/poet/generate", 'seed':seed}
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = poet_generate_post(api_session_mock, prompt_id=prompt_id, random_seed=seed)

    api_session_mock.post.assert_called_once_with(
        'v1/poet/generate',
        params={
            'prompt_id': prompt_id,
            'generate_n': 100,
            'temperature': 1.0,
            'maxlen': 1000,
            'seed': seed
        }
    )
    assert result.job_id == jobid

def test_poet_generate_get(api_session_mock):
    job_id = '12345'

    # Prepare a file stream (CSV) with headers
    csv_data = 'sequence,score,name\nsequence-1,1.0,name1\nsequence-2,2.0,name2\n'
    csv_stream = io.BytesIO(csv_data.encode())

    response_mock = ResponseMock()
    response_mock.headers = {'status': 'SUCCESS', 'job_type': '/poet/generate', 'job_id': job_id}
    response_mock.iter_content = MagicMock(return_value=iter([csv_data.encode()]))
    response_mock.content = csv_stream

    api_session_mock.get = MagicMock(return_value=response_mock)
    result = poet_generate_get(api_session_mock, job_id)

    api_session_mock.get.assert_called_once_with(
        'v1/poet/generate',
        params={'job_id': job_id},
        stream=True
    )

    # Assert the returned stream and headers
    assert result.content == csv_stream
    assert result.headers == {'status': 'SUCCESS', 'job_type': '/poet/generate', 'job_id': job_id}



def test_poet_score_get(api_session_mock):
    job_id = '12345'
    results =   { "sequence": b"AAHAA", "score": [ -1.0],"name": "name1",}
    response_mock = ResponseMock()
    response_mock._json = {'result': [results], 'status':'SUCCESS','job_type':"/poet/score", "job_id":job_id}

    api_session_mock.get = MagicMock(return_value=response_mock)

    result = poet_score_get(api_session_mock, job_id, page_size=100)

    api_session_mock.get.assert_called_once_with(
        'v1/poet/score',
        params={'job_id': job_id, 'page_size': 100, 'page_offset': 0}
    )
    assert len(result.result) == 1
    assert result.result[0].sequence == results['sequence']
    assert result.result[0].score == results['score']
    assert result.result[0].name == results['name']


def test_poet_score_post(api_session_mock):
    prompt_id = '12345'
    queries = [b'AAA', b'AAH', b'AAL']
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '67890', 'status': 'SUCCESS', 'job_type': '/poet/score'}

    api_session_mock.post = MagicMock(return_value=response_mock)

    result = poet_score_post(api_session_mock, prompt_id, queries)

    api_session_mock.post.assert_called_once_with(
        'v1/poet/score',
        files={'variant_file': ANY},
        params={'prompt_id': prompt_id}
    )
    assert result.job_id == '67890'


def test_msa_post(api_session_mock):
    msa_fasta = f">test\nAAALHAAA".encode()
    response_mock = ResponseMock()
    response_mock._json = {'job_id': '12345','msa_id': '12345', 'status': 'SUCCESS', 'job_type': '/align/align'}
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = msa_post(api_session_mock, msa_file=msa_fasta)

    api_session_mock.post.assert_called_once_with(
        'v1/poet/align/msa',
        files={'msa_file': msa_fasta},
        params={'is_seed': False}
    )
    assert result.msa_id == '12345'

def test_upload_prompt_post(api_session_mock):
    prompt_fasta = f">test\nAAALHAAA".encode()
    response_mock = ResponseMock()
    response_mock._json = {'job_id': 'j123','prompt_id': 'j123', 'status': 'SUCCESS', 'job_type': '/align/align'}
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = upload_prompt_post(api_session_mock, prompt_file=prompt_fasta)

    api_session_mock.post.assert_called_once_with(
        'v1/align/upload_prompt',
        body={'prompt_file': prompt_fasta}
    )
    assert result.job_id == 'j123'

def test_prompt_post(api_session_mock):
    msa_id = '12345'
    job_id = '67890'
    response_mock = ResponseMock()
    response_mock._json = {'job_id':job_id ,'msa_id': msa_id,'prompt_id': job_id, 'status': 'SUCCESS', 'job_type': '/align/prompt'}
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = prompt_post(
        api_session_mock,
        msa_id=msa_id,
        num_sequences=10,
        num_residues=None,
        method=MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
        homology_level=0.8,
        max_similarity=1.0,
        min_similarity=0.0,
        always_include_seed_sequence=False,
        num_ensemble_prompts=1,
        random_seed=12345
    )

    api_session_mock.post.assert_called_once_with(
        'v1/poet/align/prompt',
        params={
            'msa_id': msa_id,
            'msa_method': MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
            'homology_level': 0.8,
            'max_similarity': 1.0,
            'min_similarity': 0.0,
            'force_include_first': False,
            'replicates': 1,
            'seed': 12345,
            'max_msa_sequences': 10
        }
    )
    assert result.job_id == '67890'

def test_get_align_job_inputs(api_session_mock):
    job_id = '12345'
    input_type = PoetInputType.INPUT

    response_mock = MagicMock()
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = get_align_job_inputs(api_session_mock, job_id, input_type)

    api_session_mock.get.assert_called_once_with(
        'v1/poet/align/inputs',
        params={'job_id': job_id, 'msa_type': input_type},
        stream=True
    )
    assert result == response_mock

def test_get_align_job_inputs_prompt_index(api_session_mock):
    job_id = '12345'
    input_type = PoetInputType.PROMPT
    prompt_index =1

    response_mock = MagicMock()
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = get_align_job_inputs(api_session_mock, job_id, input_type, prompt_index=prompt_index)

    api_session_mock.get.assert_called_once_with(
        'v1/poet/align/inputs',
        params={'job_id': job_id, 'msa_type': input_type, "replicate":prompt_index},
        stream=True
    )
    assert result == response_mock


def test_poet_single_site_post_invalid_parameter(api_session_mock):
    variant = 'HLALA'
    error_message = ""
    response_mock = MagicMock()
    api_session_mock.post = MagicMock(return_value=response_mock)

    with pytest.raises(InvalidParameterError) as exc:
        poet_single_site_post(api_session_mock, variant=variant)

    assert "parent_id or prompt_id must be set" in str(exc.value)  # Assertion for the backend exception message

    with pytest.raises(InvalidParameterError) as exc:
        poet_single_site_post(api_session_mock, variant=variant, parent_id='123', prompt_id='123')

    assert "parent_id or prompt_id must be set" in str(exc.value)  # Assertion for the backend exception message

def test_poet_generate_post_invalid_parameter(api_session_mock):
    prompt_id = '1234'
    num_samples = 100
    response_mock = MagicMock()
    api_session_mock.post = MagicMock(return_value=response_mock)

    with pytest.raises(InvalidParameterError) as exc:
        poet_generate_post(api_session_mock, prompt_id=prompt_id, num_samples=num_samples, temperature=100)
    assert  "'temperature' must be between" in str(exc.value)  # Assertion for the backend exception message

    with pytest.raises(InvalidParameterError) as exc:
        poet_generate_post(api_session_mock, prompt_id=prompt_id, num_samples=num_samples, topk=100, temperature=1)
    assert  "'topk' must be between" in str(exc.value)  # Assertion for the backend exception message

    with pytest.raises(InvalidParameterError) as exc:
        poet_generate_post(api_session_mock, prompt_id=prompt_id, num_samples=num_samples, topp=-1)
    assert  "'topp' must be between" in str(exc.value)  # Assertion for the backend exception message

    with pytest.raises(InvalidParameterError) as exc:
        poet_generate_post(api_session_mock, prompt_id=prompt_id, num_samples=num_samples, random_seed=-100)
    assert  "'random_seed' must be between" in str(exc.value)  # Assertion for the backend exception message



def test_poet_score_post_invalid_parameter(api_session_mock):
    queries = ["LHAALA", "AAAHAAA"]
    queries = [i.encode() for i in queries]
    response_mock = MagicMock()
    api_session_mock.post = MagicMock(return_value=response_mock)

    with pytest.raises(MissingParameterError) as exc:
        poet_score_post(api_session_mock, prompt_id='123', queries=[])

    assert "include queries" in str(exc.value)  # Assertion for the backend exception message

    with pytest.raises(MissingParameterError) as exc:
        poet_score_post(api_session_mock, prompt_id=None, queries=queries)

    assert "include prompt" in str(exc.value)  # Assertion for the backend exception message

