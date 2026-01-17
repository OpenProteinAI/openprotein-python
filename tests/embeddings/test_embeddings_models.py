"""Test the model logic for the embeddings domain."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.embeddings.esm import ESMModel
from openprotein.embeddings.models import EmbeddingModel
from openprotein.embeddings.openprotein import OpenProteinModel
from openprotein.embeddings.poet import PoETModel
from openprotein.embeddings.poet2 import PoET2Model
from openprotein.jobs import JobType


@pytest.mark.parametrize(
    "model_id, expected_class",
    [
        ("esm1b_t33_650M_UR50S", ESMModel),
        ("esm2_t33_650M_UR50D", ESMModel),
        ("prot-seq", OpenProteinModel),
        ("poet", PoETModel),
        ("poet-2", PoET2Model),
        ("unsupported-model", EmbeddingModel),  # Falls back to default
    ],
)
def test_embedding_model_create(model_id, expected_class):
    """Test the EmbeddingModel.create factory method."""
    mock_session = MagicMock()
    # The factory calls get_metadata, which we can ignore for this test
    with patch.object(
        expected_class, "get_metadata", return_value=MagicMock()
    ) as mock_get_metadata:
        model = EmbeddingModel.create(
            session=mock_session, model_id=model_id, default=EmbeddingModel
        )
        assert isinstance(model, expected_class)
        if expected_class != EmbeddingModel:
            # The base class won't have this called if it's the one created
            mock_get_metadata.assert_called_once()


def test_embedding_model_create_unsupported_error():
    """Test that creating an unsupported model without a default raises an error."""
    mock_session = MagicMock()
    with pytest.raises(
        ValueError, match="Unsupported model_id type: unsupported-model"
    ):
        EmbeddingModel.create(session=mock_session, model_id="unsupported-model")


def test_embedding_model_embed(mock_session: MagicMock):
    """Test the base EmbeddingModel.embed method."""
    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = EmbeddingModel(session=mock_session, model_id="test-model")

    sequences = [b"ACGT"]
    mock_job_dict = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "job_type": JobType.embeddings_embed.value,
        "created_date": "2023-01-01T00:00:00",
    }
    with patch(
        "openprotein.embeddings.api.request_post", return_value=mock_job_dict
    ) as mock_request_post:
        model.embed(sequences=sequences)
        mock_request_post.assert_called_once()
        call_args, call_kwargs = mock_request_post.call_args
        assert call_kwargs["session"] == mock_session
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["sequences"] == sequences


def test_poet_model_embed(mock_session: MagicMock):
    """Test that PoETModel.embed correctly passes the prompt_id."""
    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = PoETModel(session=mock_session, model_id="poet")

    sequences = [b"ACGT"]
    prompt_id = "prompt-123"

    mock_job_dict = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "job_type": JobType.embeddings_embed.value,
        "created_date": "2023-01-01T00:00:00",
    }

    with patch(
        "openprotein.embeddings.api.request_post", return_value=mock_job_dict
    ) as mock_request_post:
        model.embed(sequences=sequences, prompt=prompt_id)
        mock_request_post.assert_called_once()
        call_args, call_kwargs = mock_request_post.call_args
        assert call_kwargs["prompt_id"] == prompt_id


def test_poet2_model_embed():
    """Test that PoET2Model.embed correctly passes prompt and query IDs."""
    from openprotein.prompt import PromptAPI

    mock_session = MagicMock()
    # Configure the 'prompt' attribute to be a mock of PromptAPI
    mock_session.prompt = MagicMock(spec=PromptAPI)
    mock_session.prompt._resolve_query.return_value = "query-456"

    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = PoET2Model(session=mock_session, model_id="poet-2")

    sequences = [b"ACGT"]
    prompt_id = "prompt-123"
    query_sequence = b"TGCA"

    mock_job_dict = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "job_type": JobType.embeddings_embed.value,
        "created_date": "2023-01-01T00:00:00",
    }

    with patch(
        "openprotein.embeddings.api.request_post", return_value=mock_job_dict
    ) as mock_request_post:
        model.embed(sequences=sequences, prompt=prompt_id, query=query_sequence)
        mock_request_post.assert_called_once()
        call_args, call_kwargs = mock_request_post.call_args
        assert call_kwargs["prompt_id"] == prompt_id
        assert call_kwargs["query_id"] == "query-456"
