"""E2E tests for the embeddings domain."""

from pathlib import Path

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.errors import HTTPError
from openprotein.molecules import Protein
from tests.e2e.config import scaled_timeout
from tests.utils.sequences import random_sequence_fake

# Model configurations: (model_id, expected_dimension)
EMBEDDING_MODELS = [
    ("esm2_t33_650M_UR50D", 1280),
    ("prot-seq", 1024),
    ("poet", 1024),
    ("poet-2", 1024),
]

REDUCTION_TYPES = [
    ReductionType.MEAN,
    ReductionType.SUM,
]

SEQ_LEN = 1000
NUM_SEQS_SMALL = 10
NUM_SEQS_MEDIUM = 100
TIMEOUT = scaled_timeout(1.0)
GENERATE_TIMEOUT = scaled_timeout(2.0)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,expected_dim", EMBEDDING_MODELS)
def test_embedding_single_model(session: OpenProtein, model_id: str, expected_dim: int):
    """
    Test embedding workflow for a single model.
    Validates model metadata and embedding output shape.
    """
    # Get the model
    model = session.embedding.get_model(model_id)
    assert model is not None, f"Failed to get model {model_id}"
    assert model.metadata.dimension == expected_dim, (
        f"Expected dimension {expected_dim} for {model_id}, "
        f"got {model.metadata.dimension}"
    )

    # Embed a small batch of sequences
    sequences = [random_sequence_fake(SEQ_LEN).encode() for _ in range(NUM_SEQS_SMALL)]
    future = model.embed(sequences=sequences)

    # Validate results
    results = future.wait(timeout=TIMEOUT)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert (
        len(results) == NUM_SEQS_SMALL
    ), f"Expected {NUM_SEQS_SMALL} results, got {len(results)}"

    # Validate first result
    sequence, embedding = results[0]
    assert sequence == sequences[0], "Sequence mismatch in results"
    assert isinstance(embedding, np.ndarray), "Embedding is not a numpy array"
    assert embedding.shape == (
        expected_dim,
    ), f"Expected shape ({expected_dim},), got {embedding.shape}"


@pytest.mark.e2e
@pytest.mark.parametrize("reduction", REDUCTION_TYPES)
def test_embedding_reduction_types(session: OpenProtein, reduction: ReductionType):
    """
    Test different reduction types for embeddings.
    Uses ESM2 model as baseline.
    """
    model = session.embedding.esm2
    sequences = [random_sequence_fake(SEQ_LEN).encode() for _ in range(NUM_SEQS_SMALL)]

    future = model.embed(sequences=sequences, reduction=reduction)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == NUM_SEQS_SMALL
    sequence, embedding = results[0]
    assert sequence == sequences[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (model.metadata.dimension,)


@pytest.mark.e2e
def test_embedding_parallel_models(session: OpenProtein, test_sequences_short):
    """
    Test submitting embedding jobs to multiple models in parallel.
    Validates that the backend can handle concurrent jobs.
    """
    # Submit jobs to all models in parallel
    futures = []
    for model_id, expected_dim in EMBEDDING_MODELS:
        model = session.embedding.get_model(model_id)
        future = model.embed(sequences=test_sequences_short)
        futures.append((model_id, expected_dim, future))

    # Wait for all jobs and validate
    for model_id, expected_dim, future in futures:
        results = future.wait(timeout=TIMEOUT)
        assert len(results) == len(test_sequences_short), (
            f"Model {model_id}: expected {len(test_sequences_short)} results, "
            f"got {len(results)}"
        )
        _, embedding = results[0]
        assert embedding.shape == (expected_dim,), (
            f"Model {model_id}: expected shape ({expected_dim},), "
            f"got {embedding.shape}"
        )


@pytest.mark.e2e
@pytest.mark.parametrize("num_seqs", [1, 10, 100])
def test_embedding_batch_sizes(session: OpenProtein, num_seqs: int):
    """
    Test embedding with different batch sizes.
    Validates scalability and batch processing.
    """
    model = session.embedding.esm2
    sequences = [random_sequence_fake(SEQ_LEN).encode() for _ in range(num_seqs)]

    future = model.embed(sequences=sequences)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == num_seqs
    for i, (seq, emb) in enumerate(results):
        assert seq == sequences[i]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (model.metadata.dimension,)


@pytest.mark.e2e
def test_embedding_varied_sequence_lengths(session: OpenProtein, test_sequences_varied):
    """
    Test embedding sequences of varying lengths.
    Validates handling of short, medium, long, and very long sequences.
    """
    model = session.embedding.esm2

    future = model.embed(sequences=test_sequences_varied)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == len(test_sequences_varied)
    for i, (seq, emb) in enumerate(results):
        assert seq == test_sequences_varied[i], f"Sequence {i} mismatch"
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (model.metadata.dimension,)


@pytest.mark.e2e
def test_embedding_empty_sequence_handling(session: OpenProtein):
    """
    Test error handling for edge cases like empty sequences.
    """
    model = session.embedding.esm2

    with pytest.raises(HTTPError, match="Status code"):
        future = model.embed(sequences=[b""])
        future.wait(timeout=TIMEOUT)


@pytest.mark.e2e
def test_embedding_invalid_amino_acids(session: OpenProtein):
    """
    Test error handling for sequences with invalid amino acids.
    """
    model = session.embedding.esm2

    # Sequence with invalid characters
    invalid_seq = b"ACDEFGHIKLMNPQRSTVWYXBZJ123"

    with pytest.raises(HTTPError, match="Status code"):
        future = model.embed(sequences=[invalid_seq])
        future.wait(timeout=TIMEOUT)


def _supports_model(session: OpenProtein, model_id: str) -> bool:
    return model_id in {model.id for model in session.embedding.list_models()}


@pytest.mark.e2e
def test_e2e_embedding_logits(session: OpenProtein):
    """Validate logits retrieval for a model that supports logits output."""
    if not _supports_model(session, "esm2_t33_650M_UR50D"):
        pytest.skip("esm2_t33_650M_UR50D model is not available in this backend")

    model = session.embedding.esm2
    sequence = b"ACDEFGHIKLMNPQRSTVWY"
    results = model.logits(sequences=[sequence]).wait(timeout=TIMEOUT)

    assert len(results) == 1
    returned_seq, logits = results[0]
    assert returned_seq == sequence
    assert isinstance(logits, np.ndarray)
    assert logits.ndim >= 2
    assert logits.size > 0
    assert np.isfinite(logits).all()


@pytest.mark.e2e
def test_e2e_embedding_attn(session: OpenProtein):
    """Validate attention output for a model that supports attention."""
    if not _supports_model(session, "esm2_t33_650M_UR50D"):
        pytest.skip("esm2_t33_650M_UR50D model is not available in this backend")

    model = session.embedding.esm2
    sequence = b"ACDEFGHIKLMNPQRSTVWY"
    results = model.attn(sequences=[sequence]).wait(timeout=TIMEOUT)

    assert len(results) == 1
    returned_seq, attn = results[0]
    assert returned_seq == sequence
    assert isinstance(attn, np.ndarray)
    assert attn.ndim >= 3
    assert attn.size > 0
    assert np.isfinite(attn).all()


@pytest.mark.e2e
def test_e2e_poet2_score_single_site_and_indel(session: OpenProtein):
    """Validate PoET-2 score, single-site, and indel workflows when supported."""
    if not _supports_model(session, "poet-2"):
        pytest.skip("poet-2 model is not available in this backend")

    model = session.embedding.poet2
    base_sequence = b"ACDEFGHIKLMNPQRSTVWY"
    batch_sequences = [base_sequence, b"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"]

    score_results = model.score(sequences=batch_sequences).wait(timeout=TIMEOUT)
    assert len(score_results) == len(batch_sequences)
    for row in score_results:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()

    single_site_results = model.single_site(sequence=base_sequence).wait(timeout=TIMEOUT)
    assert len(single_site_results) > 0
    for row in single_site_results[:10]:
        assert isinstance(row.mut_code, str)
        assert len(row.mut_code) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()

    indel_results = model.indel(sequence=base_sequence, insert="A").wait(timeout=TIMEOUT)
    assert len(indel_results) > 0
    for row in indel_results[:10]:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()


@pytest.mark.e2e
def test_e2e_ablang2_embeddings(session: OpenProtein):
    """Validate end-to-end embeddings with AbLang2 when available."""
    available_models = {model.id for model in session.embedding.list_models()}
    if "ablang2" not in available_models:
        pytest.skip("ablang2 model is not available in this backend")

    model = session.embedding.get_model("ablang2")
    sequences = [
        b"QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPG:KGLEWVSAISWNSGSIGYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGYYYGMDVWGQGTTVTVSS",
        b"EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGQ:GLEWMGIINPSNGGTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARDRGYSSSWYFDVWGQGTLVTVSS",
    ]

    future = model.embed(sequences=sequences)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == len(sequences)
    for i, (seq, embedding) in enumerate(results):
        assert seq == sequences[i]
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (model.metadata.dimension,)


def _assert_generated_sequences(results: list, expected_count: int) -> None:
    assert isinstance(results, list)
    assert len(results) == expected_count
    for entry in results:
        assert isinstance(entry.sequence, str)
        assert len(entry.sequence) > 0
        assert isinstance(entry.score, np.ndarray)


@pytest.mark.e2e
def test_e2e_structure_generate_then_sequence_generate_with_design(
    session: OpenProtein,
):
    """
    Run a structure-generation -> sequence-generation workflow.

    This validates that a structure generation future can be consumed directly
    by sequence generation through `design`.
    """
    n_designs = 1
    n_sequences = 2
    design_future = session.models.rfdiffusion.generate(contigs=60, N=n_designs)
    assert design_future.wait_until_done(timeout=GENERATE_TIMEOUT)

    designs = design_future.get()
    assert len(designs) == n_designs

    seq_future = session.models.proteinmpnn.generate(
        design=design_future,
        num_samples=n_sequences,
        temperature=0.1,
    )
    assert seq_future.wait_until_done(timeout=GENERATE_TIMEOUT)
    results = seq_future.get()
    _assert_generated_sequences(results=results, expected_count=n_designs * n_sequences)


@pytest.mark.e2e
def test_e2e_proteinmpnn_generate_with_query_fanout(session: OpenProtein):
    """Validate list-valued `query` fan-out behavior for ProteinMPNN generation."""
    num_queries = 2
    n_sequences = 2
    n_unmasked_positions = 18

    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    # Use a full structure example recommended for ProteinMPNN inverse folding.
    query_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")

    # Remove N-terminal linker residues with undefined structure.
    sdn_linker = b"MHHHHHHSDN"
    query_protein = query_protein[len(sdn_linker) :]

    structured_positions = np.where(~query_protein.get_structure_mask())[0] + 1
    assert len(structured_positions) >= n_unmasked_positions
    rng = np.random.default_rng(0)

    query_ids = []
    for _ in range(num_queries):
        unmasked_positions = np.sort(
            rng.choice(
                structured_positions,
                size=n_unmasked_positions,
                replace=False,
            )
        )
        masked_query = query_protein.copy().mask_sequence_except_at(unmasked_positions)
        query = session.prompt.create_query(query=masked_query, force_structure=True)
        query_ids.append(query.id)

    assert len(query_ids) == num_queries

    future = session.models.proteinmpnn.generate(
        query=query_ids,
        num_samples=n_sequences,
        temperature=0.1,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    results = future.get()
    _assert_generated_sequences(
        results=results,
        expected_count=num_queries * n_sequences,
    )


@pytest.mark.e2e
def test_e2e_poet2_generate_with_query_fanout(session: OpenProtein):
    """Validate list-valued `query` fan-out behavior for PoET2 generation."""
    n_sequences = 2
    query_ids = [
        session.prompt.create_query("ACDEFGHIXXXXPQRSTVWY").id,
        session.prompt.create_query("MKTAYIAKQRQISXXXXXFSRQLEERLGLIEVQ").id,
    ]

    future = session.embedding.poet2.generate(
        prompt=None,
        query=query_ids,
        num_samples=n_sequences,
        temperature=1.0,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    results = future.get()
    _assert_generated_sequences(
        results=results,
        expected_count=len(query_ids) * n_sequences,
    )


@pytest.mark.e2e
def test_e2e_poet2_generate_with_prompt(session: OpenProtein):
    """Validate PoET2 generate with a prompt that has already reached SUCCESS."""
    n_sequences = 2
    context = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"]
    prompt = session.prompt.create_prompt(context)
    assert prompt.wait_until_done(timeout=TIMEOUT)

    future = session.embedding.poet2.generate(
        prompt=prompt,
        num_samples=n_sequences,
        temperature=1.0,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    results = future.get()
    _assert_generated_sequences(results=results, expected_count=n_sequences)


@pytest.mark.e2e
def test_e2e_proteinmpnn_score_not_implemented(session: OpenProtein):
    with pytest.raises(NotImplementedError, match="Score not yet implemented"):
        session.models.proteinmpnn.score(
            sequences=[b"ACDEFGHIKLMNPQRSTVWY"],
            query=b"ACDEFGHIKLMNPQRSTVWY",
        )


@pytest.mark.e2e
def test_e2e_proteinmpnn_indel_not_implemented(session: OpenProtein):
    with pytest.raises(NotImplementedError, match="Score indel not yet implemented"):
        session.models.proteinmpnn.indel(
            sequence=b"ACDEFGHIKLMNPQRSTVWY",
            query=b"ACDEFGHIKLMNPQRSTVWY",
            insert="A",
        )


@pytest.mark.e2e
def test_e2e_proteinmpnn_single_site_not_implemented(session: OpenProtein):
    with pytest.raises(NotImplementedError, match="Score indel not yet implemented"):
        session.models.proteinmpnn.single_site(
            sequence=b"ACDEFGHIKLMNPQRSTVWY",
            query=b"ACDEFGHIKLMNPQRSTVWY",
        )
