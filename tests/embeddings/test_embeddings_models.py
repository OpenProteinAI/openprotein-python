"""Test the model logic for the embeddings domain."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.embeddings.ablang import AbLang2Model
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
        ("esm1v_t33_650M_UR90S_1", ESMModel),
        ("esm2_t33_650M_UR50D", ESMModel),
        ("esmc-300m", ESMModel),
        ("esmc-600m", ESMModel),
        ("esmc-6b", ESMModel),
        ("ablang2", AbLang2Model),
        ("prot-seq", OpenProteinModel),
        ("poet", PoETModel),
        ("poet-2", PoET2Model),
        ("unsupported-model", EmbeddingModel),  # Falls back to default
    ],
)
def test_embedding_model_create(model_id, expected_class):
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


def test_embedding_model_create_with_metadata():
    mock_session = MagicMock()
    mock_metadata = MagicMock()
    mock_metadata.id = "prot-seq"

    with patch.object(
        OpenProteinModel, "get_metadata", return_value=MagicMock()
    ) as mock_get_metadata:
        model = EmbeddingModel.create(
            session=mock_session,
            model_id="prot-seq",
            default=EmbeddingModel,
            metadata=mock_metadata,
        )
        assert isinstance(model, OpenProteinModel)
        mock_get_metadata.assert_not_called()
        assert model._metadata is mock_metadata


def test_embedding_model_create_unsupported_error():
    mock_session = MagicMock()
    with pytest.raises(
        ValueError, match="Unsupported model_id type: unsupported-model"
    ):
        EmbeddingModel.create(session=mock_session, model_id="unsupported-model")


def test_embedding_model_embed(mock_session: MagicMock):
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


def test_poet2_model_generate_with_query_list():
    from openprotein.prompt import PromptAPI

    mock_session = MagicMock()
    mock_session.prompt = MagicMock(spec=PromptAPI)
    mock_session.prompt._resolve_query.return_value = ["query-1", "query-2"]

    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = PoET2Model(session=mock_session, model_id="poet-2")

    with (
        patch(
            "openprotein.embeddings.poet.api.request_generate_post",
            return_value=MagicMock(),
        ) as mock_request_post,
        patch(
            "openprotein.embeddings.poet.EmbeddingsGenerateFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.generate(prompt=None, query=["query-1", "query-2"], num_samples=2)

    mock_session.prompt._resolve_query.assert_called_once_with(
        query=["query-1", "query-2"]
    )
    _, kwargs = mock_request_post.call_args
    assert kwargs["query_id"] == ["query-1", "query-2"]


@pytest.mark.parametrize(
    "model_class, model_id",
    [
        (ESMModel, "esm1b_t33_650M_UR50S"),
        (ESMModel, "esm2_t33_650M_UR50D"),
        (ESMModel, "esmc-300m"),
    ],
)
def test_single_site_forwards_base_sequence(model_class, model_id):
    """single_site() posts the base sequence and returns the single-site future."""
    mock_session = MagicMock()

    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = model_class(session=mock_session, model_id=model_id)

    with (
        patch(
            "openprotein.embeddings.esm.api.request_score_single_site_post",
            return_value=MagicMock(),
        ) as mock_request_post,
        patch(
            "openprotein.embeddings.esm.EmbeddingsScoreSingleSiteFuture.create",
            return_value=MagicMock(),
        ) as mock_future_create,
    ):
        model.single_site(sequence="MKTLL")

    _, kwargs = mock_request_post.call_args
    assert kwargs["model_id"] == model_id
    assert kwargs["base_sequence"] == "MKTLL"
    mock_future_create.assert_called_once()


def test_ablang2_single_site_forwards_base_sequence():
    """single_site() posts the base sequence and returns the single-site future."""
    mock_session = MagicMock()

    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = AbLang2Model(session=mock_session, model_id="ablang2")

    with (
        patch(
            "openprotein.embeddings.ablang.api.request_score_single_site_post",
            return_value=MagicMock(),
        ) as mock_request_post,
        patch(
            "openprotein.embeddings.ablang.EmbeddingsScoreSingleSiteFuture.create",
            return_value=MagicMock(),
        ) as mock_future_create,
    ):
        model.single_site(sequence="EVQL:DIQM")

    _, kwargs = mock_request_post.call_args
    assert kwargs["model_id"] == "ablang2"
    assert kwargs["base_sequence"] == "EVQL:DIQM"
    mock_future_create.assert_called_once()


@pytest.mark.parametrize(
    ("model_class", "model_id"), [(PoETModel, "poet"), (PoET2Model, "poet-2")]
)
def test_a_poet_model_sends_the_id_of_a_stored_prompt_that_is_not_ready(
    model_class, model_id
):
    from openprotein.data.prompts import Prompt, PromptSummary
    from openprotein.jobs import JobStatus
    from openprotein.prompt import PromptAPI

    session = MagicMock()
    session.prompt = MagicMock(spec=PromptAPI)
    session.prompt._resolve_query.return_value = None
    prompt = Prompt(
        session,
        PromptSummary(id="prompt-9", status=JobStatus.PENDING, job_id="prompt-9"),
    )
    job = {
        "job_id": "job-123",
        "status": "PENDING",
        "job_type": JobType.embeddings_embed.value,
        "created_date": "2023-01-01T00:00:00",
    }
    with patch("openprotein.embeddings.api.get_model", return_value=MagicMock()):
        model = model_class(session=session, model_id=model_id)

    with patch("openprotein.embeddings.api.request_post", return_value=job) as post:
        model.embed(sequences=[b"ACGT"], prompt=prompt)

    assert post.call_args.kwargs["prompt_id"] == "prompt-9"
