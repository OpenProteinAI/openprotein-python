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
    mock_session.get.assert_called_once_with("v1/embeddings/models", params={})
    assert result == ["model1", "model2"]


def test_list_models_verbose(mock_session: MagicMock):
    """Test the list_models API function with verbose=True returns ModelMetadata."""
    mock_session.get.return_value.json.return_value = [
        {
            "model_id": "model1",
            "description": {"summary": "Model 1"},
            "dimension": 128,
            "output_types": ["embedding"],
            "input_tokens": ["protein"],
            "token_descriptions": [],
        },
        {
            "model_id": "model2",
            "description": {"summary": "Model 2"},
            "dimension": 256,
            "output_types": ["embedding"],
            "input_tokens": ["protein"],
            "token_descriptions": [],
        },
    ]
    result = api.list_models(mock_session, verbose=True)
    mock_session.get.assert_called_once_with(
        "v1/embeddings/models", params={"verbose": "true"}
    )
    assert len(result) == 2
    assert all(isinstance(m, api.ModelMetadata) for m in result)
    assert result[0].id == "model1"
    assert result[1].id == "model2"


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


def test_request_generate_post_with_query_id_list(mock_session: MagicMock):
    """Test generation POST body when query_id is provided as a list."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_generate,
        "created_date": "2023-01-01T00:00:00",
    }

    api.request_generate_post(
        session=mock_session,
        model_id="proteinmpnn",
        num_samples=4,
        query_id=["q-1", "q-2"],
        use_query_structure_in_decoder=False,
        random_seed=123,
    )

    mock_session.post.assert_called_once()
    _, kwargs = mock_session.post.call_args
    assert kwargs["json"]["query_id"] == ["q-1", "q-2"]
    assert kwargs["json"]["use_query_structure_in_decoder"] is False


def test_request_generate_post_rejects_query_id_list_for_poet(mock_session: MagicMock):
    """poet v1 does not support query_id in generation (including list form)."""
    with pytest.raises(AssertionError, match="Model with id poet does not support query"):
        api.request_generate_post(
            session=mock_session,
            model_id="poet",
            query_id=["q-1"],
        )


# force_recompute tests for all 7 api functions


def test_request_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_embed,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_post(mock_session, "test_model", ["ACGT"], force_recompute=True)
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_embed,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_post(mock_session, "test_model", ["ACGT"])
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_logits_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true for logits."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_logits,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_logits_post(mock_session, "test_model", ["ACGT"], force_recompute=True)
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_logits_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent for logits."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_logits,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_logits_post(mock_session, "test_model", ["ACGT"])
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_attn_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true for attn."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_attn,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_attn_post(mock_session, "test_model", ["ACGT"], force_recompute=True)
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_attn_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent for attn."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_attn,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_attn_post(mock_session, "test_model", ["ACGT"])
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_score_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true for score."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_score,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_post(
        mock_session, "test_model", ["ACGT"], prompt_id="prompt1", force_recompute=True
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_score_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent for score."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_score,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_post(mock_session, "test_model", ["ACGT"], prompt_id="prompt1")
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_score_indel_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true for score/indel."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_score_indel,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_indel_post(
        mock_session,
        "test_model",
        "ACGT",
        insert="A",
        prompt_id="prompt1",
        force_recompute=True,
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_score_indel_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent for score/indel."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_score_indel,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_indel_post(
        mock_session, "test_model", "ACGT", insert="A", prompt_id="prompt1"
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_score_single_site_post_force_recompute_sends_param(
    mock_session: MagicMock,
):
    """force_recompute=True sends ?force=true for score_single_site."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_single_site,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_single_site_post(
        mock_session, "test_model", "ACGT", prompt_id="prompt1", force_recompute=True
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_score_single_site_post_no_force_recompute_by_default(
    mock_session: MagicMock,
):
    """Without force_recompute, no ?force param is sent for score_single_site."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.poet_single_site,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_score_single_site_post(
        mock_session, "test_model", "ACGT", prompt_id="prompt1"
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None


def test_request_generate_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true for generate."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_generate,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_generate_post(
        mock_session,
        "test_model",
        num_samples=4,
        prompt_id="prompt1",
        force_recompute=True,
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_request_generate_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent for generate."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.embeddings_generate,
        "created_date": "2023-01-01T00:00:00",
    }
    api.request_generate_post(
        mock_session, "test_model", num_samples=4, prompt_id="prompt1"
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None
