"""Test the api for the embeddings domain."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from openprotein.embeddings import api
from openprotein.jobs import JobStatus, JobType


def test_list_models(mock_session: MagicMock):
    """Test the list_models API function."""
    mock_session.get.return_value.json.return_value = ["model1", "model2"]
    result = api.list_models(mock_session)
    mock_session.get.assert_called_once_with("v1/embeddings/models")
    assert result == ["model1", "model2"]


def test_get_model(mock_session: MagicMock):
    """Test the get_model API function."""
    mock_session.get.return_value.json.return_value = {
        "model_id": "model1",
        "description": {"summary": "A test model"},
        "dimension": 128,
        "output_types": ["embedding"],
        "input_tokens": ["protein"],
        "token_descriptions": [],
    }
    api.get_model(mock_session, "model1")
    mock_session.get.assert_called_once_with("v1/embeddings/models/model1")


def test_get_request_sequences(mock_session: MagicMock):
    """Test get_request_sequences."""
    job_id = "job_123"
    job_type = JobType.embeddings_embed
    expected_sequences = [b"seq1", b"seq2"]
    mock_session.get.return_value.json.return_value = expected_sequences

    sequences = api.get_request_sequences(mock_session, job_id, job_type)

    path = "v1" + job_type.value
    endpoint = path + f"/{job_id}/sequences"
    mock_session.get.assert_called_once_with(endpoint)
    assert sequences == expected_sequences


def test_request_get_sequence_result(mock_session: MagicMock):
    """Test request_get_sequence_result."""
    job_id = "job_123"
    sequence = b"seq1"
    job_type = JobType.embeddings_embed
    expected_content = b"result_content"
    mock_session.get.return_value.content = expected_content

    content = api.request_get_sequence_result(mock_session, job_id, sequence, job_type)

    path = "v1" + job_type.value
    endpoint = path + f"/{job_id}/{sequence.decode()}"
    mock_session.get.assert_called_once_with(endpoint)
    assert content == expected_content


def test_result_decode():
    """Test result_decode."""
    arr = np.array([1.0, 2.0, 3.0])
    encoded = arr.tobytes()
    with pytest.raises(ValueError):
        api.result_decode(encoded)


@pytest.mark.parametrize(
    "api_func, schema_class, job_type",
    [
        (api.request_post, api.EmbeddingsJob, JobType.embeddings_embed),
        (api.request_logits_post, api.LogitsJob, JobType.embeddings_logits),
        (api.request_attn_post, api.AttnJob, JobType.embeddings_attn),
        (api.request_score_post, api.ScoreJob, JobType.poet_score),
    ],
)
def test_embedding_request_posts(
    mock_session: MagicMock, api_func, schema_class, job_type
):
    """Test POST requests for various embedding-related jobs."""
    model_id = "test_model"
    sequences = ["ACGT"]
    job_id = "job-123"

    mock_response_json = {
        "job_id": job_id,
        "status": JobStatus.SUCCESS,
        "job_type": job_type,
        "created_date": "2023-01-01T00:00:00",
    }
    mock_session.post.return_value.json.return_value = mock_response_json

    kwargs = {"prompt_id": "prompt1"} if "score" in api_func.__name__ else {}
    job = api_func(mock_session, model_id, sequences, **kwargs)

    assert isinstance(job, schema_class)
    assert job.job_id == job_id
    mock_session.post.assert_called_once()
    # Further checks on the endpoint and body could be added here
