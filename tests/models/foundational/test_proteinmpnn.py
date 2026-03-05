"""Tests for the ProteinMPNN foundation model."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.jobs import JobStatus
from openprotein.models.foundation.proteinmpnn import ProteinMPNNModel
from openprotein.models.structure_generation import (
    StructureGenerationFuture,
    StructureGenerationJob,
)
from openprotein.prompt import PromptAPI


@pytest.fixture
def mock_model_session():
    """Create a mocked API session configured for ProteinMPNN tests."""
    session = MagicMock()
    session.prompt = MagicMock(spec=PromptAPI)
    session.prompt._resolve_query.return_value = "query-123"
    session.get.return_value.json.return_value = {
        "model_id": "proteinmpnn",
        "description": {"summary": "ProteinMPNN"},
        "dimension": 0,
        "output_types": ["sequence"],
        "input_tokens": ["protein"],
        "token_descriptions": [],
        "max_sequence_length": 1024,
    }
    return session


def test_generate_with_design_id_without_query(mock_model_session: MagicMock):
    """ProteinMPNN generate forwards design_id when query is omitted."""
    model = ProteinMPNNModel(session=mock_model_session)

    with patch(
        "openprotein.models.foundation.proteinmpnn.embeddings_api.request_generate_post",
        return_value=MagicMock(),
    ) as mock_request_post, patch(
        "openprotein.models.foundation.proteinmpnn.EmbeddingsGenerateFuture.create",
        return_value=MagicMock(),
    ):
        model.generate(design="design-abc", num_samples=5)

    mock_model_session.prompt._resolve_query.assert_not_called()
    _, kwargs = mock_request_post.call_args
    assert kwargs["design_id"] == "design-abc"
    assert kwargs["query_id"] is None
    assert kwargs["num_samples"] == 5


def test_generate_with_query_list_without_query(mock_model_session: MagicMock):
    """ProteinMPNN generate forwards list-valued query via query_id."""
    model = ProteinMPNNModel(session=mock_model_session)
    mock_model_session.prompt._resolve_query.return_value = ["q-1", "q-2"]

    with patch(
        "openprotein.models.foundation.proteinmpnn.embeddings_api.request_generate_post",
        return_value=MagicMock(),
    ) as mock_request_post, patch(
        "openprotein.models.foundation.proteinmpnn.EmbeddingsGenerateFuture.create",
        return_value=MagicMock(),
    ):
        model.generate(query=["q-1", "q-2"], num_samples=3)

    mock_model_session.prompt._resolve_query.assert_called_once_with(
        query=["q-1", "q-2"]
    )
    _, kwargs = mock_request_post.call_args
    assert kwargs["query_id"] == ["q-1", "q-2"]
    assert kwargs["num_samples"] == 3


def test_generate_with_structure_generation_future(mock_model_session: MagicMock):
    """ProteinMPNN generate extracts design_id from StructureGenerationFuture."""
    model = ProteinMPNNModel(session=mock_model_session)
    design_future = StructureGenerationFuture(
        session=mock_model_session,
        job=StructureGenerationJob(
            job_id="design-job-789",
            job_type="/models/design",
            status=JobStatus.SUCCESS,
            created_date="2026-01-01T00:00:00",
        ),
        N=1,
        result_format="pdb",
    )

    with patch(
        "openprotein.models.foundation.proteinmpnn.embeddings_api.request_generate_post",
        return_value=MagicMock(),
    ) as mock_request_post, patch(
        "openprotein.models.foundation.proteinmpnn.EmbeddingsGenerateFuture.create",
        return_value=MagicMock(),
    ):
        model.generate(query=b"ACDE", design=design_future)

    mock_model_session.prompt._resolve_query.assert_called_once_with(query=b"ACDE")
    _, kwargs = mock_request_post.call_args
    assert kwargs["design_id"] == "design-job-789"
    assert kwargs["query_id"] == "query-123"


def test_generate_requires_query_or_design(mock_model_session: MagicMock):
    """ProteinMPNN generate validates at least one input source."""
    model = ProteinMPNNModel(session=mock_model_session)
    with pytest.raises(
        ValueError,
        match="Expected either `query` or `design` to be provided",
    ):
        model.generate()
