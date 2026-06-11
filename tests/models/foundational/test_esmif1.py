"""Tests for the ESM-IF1 foundation model."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.embeddings.future import EmbeddingsScoreSingleSiteFuture
from openprotein.embeddings.schemas import ScoreSingleSiteJob
from openprotein.jobs import JobStatus
from openprotein.models.foundation.esmif1 import ESMIF1Model
from openprotein.models.structure_generation import (
    StructureGenerationFuture,
    StructureGenerationJob,
)
from openprotein.prompt import PromptAPI

# 1UBQ (ubiquitin) native sequence, 76 residues, single chain.
UBQ_NATIVE_SEQ = (
    b"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
assert len(UBQ_NATIVE_SEQ) == 76

# Real UUIDs (rather than `"query-123"`-style stubs) so reviewers see
# what an actual server response looks like. Specific values are arbitrary;
# only their shape matters to the SDK forwarding logic these tests cover.
QUERY_ID = "9305b5de-9c9b-4313-95b9-2efb635cb908"
DESIGN_ID = "a1b2c3d4-5678-90ab-cdef-1234567890ab"
DESIGN_JOB_ID = "fb7e0c1a-2c33-4d6e-b8a9-7e5d4f2a1b6c"

# Bytes form of `query=` accepts raw structure-file content
CIF_BYTES_STUB = (
    b"data_1UBQ\n"
    b"_entry.id 1UBQ\n"
    b"loop_\n"
    b"_atom_site.group_PDB\n"
    b"_atom_site.id\n"
    b"...\n"
)


@pytest.fixture
def mock_model_session():
    """Mocked API session configured for ESM-IF1 tests."""
    session = MagicMock()
    session.prompt = MagicMock(spec=PromptAPI)
    session.prompt._resolve_query.return_value = QUERY_ID
    session.get.return_value.json.return_value = {
        "model_id": "esm-if1",
        "description": {
            "summary": (
                "ESM Inverse Folding: structure-conditioned encoder + "
                "autoregressive decoder. Score sequences against coordinates "
                "or generate new sequences from a structure."
            ),
        },
        "dimension": -1,
        "output_types": ["score", "generate"],
        "input_tokens": list("ACDEFGHIKLMNPQRSTVWY"),
        "token_descriptions": [],
        "max_sequence_length": 500,
    }
    return session


def test_score_forwards_query_and_sequences(mock_model_session: MagicMock):
    """score() resolves query and forwards sequences + query_id to the API."""
    model = ESMIF1Model(session=mock_model_session)
    sequences = [UBQ_NATIVE_SEQ]

    with (
        patch(
            "openprotein.models.foundation.esmif1.embeddings_api.request_score_post",
            return_value=MagicMock(),
        ) as mock_post,
        patch(
            "openprotein.models.foundation.esmif1.EmbeddingsScoreFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.score(sequences=sequences, query=CIF_BYTES_STUB)

    mock_model_session.prompt._resolve_query.assert_called_once_with(
        query=CIF_BYTES_STUB
    )
    _, kwargs = mock_post.call_args
    assert kwargs["model_id"] == "esm-if1"
    assert kwargs["sequences"] == sequences
    assert kwargs["query_id"] == QUERY_ID


def test_single_site_forwards_query_and_returns_single_site_future(
    mock_model_session: MagicMock,
):
    """single_site() forwards base_sequence + query_id and returns the single-site future.

    The job-type factory in Future.create dispatches a poet_single_site job to
    EmbeddingsScoreSingleSiteFuture regardless of which Future subclass `.create`
    is called on, so this also pins the declared return type to the concrete type.
    """
    model = ESMIF1Model(session=mock_model_session)
    job = ScoreSingleSiteJob(
        job_id="11111111-2222-3333-4444-555555555555",
        status=JobStatus.PENDING,
        created_date="2026-01-01T00:00:00",
    )

    with patch(
        "openprotein.models.foundation.esmif1.embeddings_api.request_score_single_site_post",
        return_value=job,
    ) as mock_post:
        future = model.single_site(sequence=UBQ_NATIVE_SEQ, query=CIF_BYTES_STUB)

    mock_model_session.prompt._resolve_query.assert_called_once_with(
        query=CIF_BYTES_STUB
    )
    _, kwargs = mock_post.call_args
    assert kwargs["model_id"] == "esm-if1"
    assert kwargs["base_sequence"] == UBQ_NATIVE_SEQ
    assert kwargs["query_id"] == QUERY_ID
    assert isinstance(future, EmbeddingsScoreSingleSiteFuture)


def test_generate_forwards_query_and_sampling_params(mock_model_session: MagicMock):
    """generate() forwards query_id, num_samples, temperature, and seed."""
    model = ESMIF1Model(session=mock_model_session)

    with (
        patch(
            "openprotein.models.foundation.esmif1.embeddings_api.request_generate_post",
            return_value=MagicMock(),
        ) as mock_post,
        patch(
            "openprotein.models.foundation.esmif1.EmbeddingsGenerateFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.generate(
            query=CIF_BYTES_STUB,
            num_samples=5,
            temperature=0.5,
            seed=42,
        )

    mock_model_session.prompt._resolve_query.assert_called_once_with(
        query=CIF_BYTES_STUB
    )
    _, kwargs = mock_post.call_args
    assert kwargs["model_id"] == "esm-if1"
    assert kwargs["query_id"] == QUERY_ID
    assert kwargs["num_samples"] == 5
    assert kwargs["temperature"] == 0.5
    assert kwargs["random_seed"] == 42


def test_generate_with_design_id_without_query(mock_model_session: MagicMock):
    """generate() forwards design_id when query is omitted."""
    model = ESMIF1Model(session=mock_model_session)

    with (
        patch(
            "openprotein.models.foundation.esmif1.embeddings_api.request_generate_post",
            return_value=MagicMock(),
        ) as mock_post,
        patch(
            "openprotein.models.foundation.esmif1.EmbeddingsGenerateFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.generate(design=DESIGN_ID, num_samples=5)

    mock_model_session.prompt._resolve_query.assert_not_called()
    _, kwargs = mock_post.call_args
    assert kwargs["model_id"] == "esm-if1"
    assert kwargs["design_id"] == DESIGN_ID
    assert kwargs["query_id"] is None
    assert kwargs["num_samples"] == 5


def test_generate_with_structure_generation_future(mock_model_session: MagicMock):
    """generate() extracts design_id from a StructureGenerationFuture."""
    model = ESMIF1Model(session=mock_model_session)
    design_future = StructureGenerationFuture(
        session=mock_model_session,
        job=StructureGenerationJob(
            job_id=DESIGN_JOB_ID,
            job_type="/models/design",
            status=JobStatus.SUCCESS,
            created_date="2026-01-01T00:00:00",
        ),
        N=1,
        result_format="pdb",
    )

    with (
        patch(
            "openprotein.models.foundation.esmif1.embeddings_api.request_generate_post",
            return_value=MagicMock(),
        ) as mock_post,
        patch(
            "openprotein.models.foundation.esmif1.EmbeddingsGenerateFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.generate(design=design_future, num_samples=2)

    mock_model_session.prompt._resolve_query.assert_not_called()
    _, kwargs = mock_post.call_args
    assert kwargs["design_id"] == DESIGN_JOB_ID
    assert kwargs["query_id"] is None


def test_generate_with_both_query_and_design(mock_model_session: MagicMock):
    """generate() forwards both ids when both are provided."""
    model = ESMIF1Model(session=mock_model_session)

    with (
        patch(
            "openprotein.models.foundation.esmif1.embeddings_api.request_generate_post",
            return_value=MagicMock(),
        ) as mock_post,
        patch(
            "openprotein.models.foundation.esmif1.EmbeddingsGenerateFuture.create",
            return_value=MagicMock(),
        ),
    ):
        model.generate(query=CIF_BYTES_STUB, design=DESIGN_ID)

    mock_model_session.prompt._resolve_query.assert_called_once_with(
        query=CIF_BYTES_STUB
    )
    _, kwargs = mock_post.call_args
    assert kwargs["query_id"] == QUERY_ID
    assert kwargs["design_id"] == DESIGN_ID


def test_generate_requires_query_or_design(mock_model_session: MagicMock):
    """generate() validates that at least one of query or design is provided."""
    model = ESMIF1Model(session=mock_model_session)
    with pytest.raises(
        ValueError,
        match="Expected either `query` or `design` to be provided",
    ):
        model.generate()
