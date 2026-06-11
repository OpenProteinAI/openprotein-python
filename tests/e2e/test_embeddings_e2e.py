"""E2E tests for the embeddings domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.errors import HTTPError
from tests.e2e.config import scaled_timeout
from tests.utils.sequences import (
    ANTIBODY_HEAVY_SEQUENCE,
    ANTIBODY_LIGHT_SEQUENCE,
    mutate_sequence,
    random_sequence_fake,
    random_sequence_real,
)

def _build_single_chain_sequence(seq_len: int) -> bytes:
    """Default sequence builder: a single-chain random AA sequence."""
    return random_sequence_fake(seq_len).encode()


def _build_paired_chain_sequence(seq_len: int) -> bytes:
    """
    Antibody paired-chain sequence builder: 'heavy:light'.

    Antibody models such as ablang2 reject random amino-acid strings, so we
    start from real antibody variable-region sequences and apply random
    substitutions to each chain. The substitutions keep the chains valid
    antibodies while making every generated pair unique, which busts
    server-side caches across test runs.

    ``seq_len`` is ignored: antibody variable domains have a fixed natural
    length.
    """
    heavy = mutate_sequence(ANTIBODY_HEAVY_SEQUENCE, mutation_rate=0.02)
    light = mutate_sequence(ANTIBODY_LIGHT_SEQUENCE, mutation_rate=0.02)
    return f"{heavy}:{light}".encode()


# Model configurations: (model_id, expected_dimension, sequence_builder).
# Once the upcoming dev rollout lets every model accept ':' separators, every
# model can use _build_paired_chain_sequence and the per-model entry can
# collapse back to (model_id, expected_dimension).
EMBEDDING_MODELS = [
    ("esm2_t33_650M_UR50D", 1280, _build_single_chain_sequence),
    ("esmc-300m", None, _build_single_chain_sequence),
    ("prot-seq", 1024, _build_single_chain_sequence),
    ("poet", 1024, _build_single_chain_sequence),
    ("poet-2", 1024, _build_single_chain_sequence),
    ("ablang2", None, _build_paired_chain_sequence),
]

REDUCTION_TYPES = [
    ReductionType.MEAN,
    ReductionType.SUM,
]

# Encoder-style PLMs exercised by the generic per-model checks below
# (reduction, batch sizes, varied lengths, error handling, logits, attn).
PER_MODEL_TEST_IDS = ["esm2_t33_650M_UR50D", "esmc-300m"]

SEQ_LEN = 1000
NUM_SEQS_SMALL = 10
NUM_SEQS_MEDIUM = 100
TIMEOUT = scaled_timeout(1.0)
GENERATE_TIMEOUT = scaled_timeout(2.0)


def _supports_model(session: OpenProtein, model_id: str) -> bool:
    return model_id in {model.id for model in session.embedding.list_models()}


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,expected_dim,build_sequence", EMBEDDING_MODELS)
def test_embedding_single_model(
    session: OpenProtein,
    model_id: str,
    expected_dim: int | None,
    build_sequence,
):
    """
    Test embedding workflow for a single model.
    Validates model metadata and embedding output shape.
    """
    if model_id not in {m.id for m in session.embedding.list_models()}:
        pytest.skip(f"{model_id} is not available in this backend")

    model = session.embedding.get_model(model_id)
    assert model is not None, f"Failed to get model {model_id}"
    if expected_dim is not None:
        assert model.metadata.dimension == expected_dim, (
            f"Expected dimension {expected_dim} for {model_id}, "
            f"got {model.metadata.dimension}"
        )
    effective_dim = model.metadata.dimension

    sequences = [build_sequence(SEQ_LEN) for _ in range(NUM_SEQS_SMALL)]
    future = model.embed(sequences=sequences)

    results = future.wait(timeout=TIMEOUT)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert (
        len(results) == NUM_SEQS_SMALL
    ), f"Expected {NUM_SEQS_SMALL} results, got {len(results)}"

    sequence, embedding = results[0]
    assert sequence == sequences[0], "Sequence mismatch in results"
    assert isinstance(embedding, np.ndarray), "Embedding is not a numpy array"
    assert embedding.shape == (
        effective_dim,
    ), f"Expected shape ({effective_dim},), got {embedding.shape}"


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
@pytest.mark.parametrize("reduction", REDUCTION_TYPES)
def test_embedding_reduction_types(
    session: OpenProtein, reduction: ReductionType, model_id: str
):
    """Test different reduction types for embeddings."""
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")
    model = session.embedding.get_model(model_id)
    sequences = [random_sequence_fake(SEQ_LEN).encode() for _ in range(NUM_SEQS_SMALL)]

    future = model.embed(sequences=sequences, reduction=reduction)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == NUM_SEQS_SMALL
    sequence, embedding = results[0]
    assert sequence == sequences[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (model.metadata.dimension,)


@pytest.mark.e2e
def test_embedding_parallel_models(session: OpenProtein):
    """
    Test submitting embedding jobs to multiple models in parallel.
    Validates that the backend can handle concurrent jobs.

    Uses a per-model sequence builder so each model gets sequences in its
    accepted format (e.g. ablang2 needs paired heavy:light).
    """
    available_models = {m.id for m in session.embedding.list_models()}

    futures = []
    for model_id, _expected_dim, build_sequence in EMBEDDING_MODELS:
        if model_id not in available_models:
            continue
        model = session.embedding.get_model(model_id)
        sequences = [build_sequence(64) for _ in range(10)]
        future = model.embed(sequences=sequences)
        futures.append((model_id, model.metadata.dimension, sequences, future))

    if not futures:
        pytest.skip("No embedding models from EMBEDDING_MODELS available in this backend")

    for model_id, dim, sequences, future in futures:
        results = future.wait(timeout=TIMEOUT)
        assert len(results) == len(sequences), (
            f"Model {model_id}: expected {len(sequences)} results, "
            f"got {len(results)}"
        )
        _, embedding = results[0]
        assert embedding.shape == (dim,), (
            f"Model {model_id}: expected shape ({dim},), "
            f"got {embedding.shape}"
        )


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
@pytest.mark.parametrize("num_seqs", [1, 10, 100])
def test_embedding_batch_sizes(session: OpenProtein, num_seqs: int, model_id: str):
    """
    Test embedding with different batch sizes.
    Validates scalability and batch processing.
    """
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")
    model = session.embedding.get_model(model_id)
    sequences = [random_sequence_fake(SEQ_LEN).encode() for _ in range(num_seqs)]

    future = model.embed(sequences=sequences)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == num_seqs
    for i, (seq, emb) in enumerate(results):
        assert seq == sequences[i]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (model.metadata.dimension,)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
def test_embedding_varied_sequence_lengths(
    session: OpenProtein, test_sequences_varied, model_id: str
):
    """
    Test embedding sequences of varying lengths.
    Validates handling of short, medium, long, and very long sequences.
    """
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")
    model = session.embedding.get_model(model_id)

    future = model.embed(sequences=test_sequences_varied)
    results = future.wait(timeout=TIMEOUT)

    assert len(results) == len(test_sequences_varied)
    for i, (seq, emb) in enumerate(results):
        assert seq == test_sequences_varied[i], f"Sequence {i} mismatch"
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (model.metadata.dimension,)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
def test_embedding_empty_sequence_handling(session: OpenProtein, model_id: str):
    """Test error handling for edge cases like empty sequences."""
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")
    model = session.embedding.get_model(model_id)

    with pytest.raises(HTTPError, match="Status code"):
        future = model.embed(sequences=[b""])
        future.wait(timeout=TIMEOUT)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
def test_embedding_invalid_amino_acids(session: OpenProtein, model_id: str):
    """Test error handling for sequences with invalid amino acids."""
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")
    model = session.embedding.get_model(model_id)

    # Sequence with invalid characters
    invalid_seq = b"ACDEFGHIKLMNPQRSTVWYXBZJ123"

    with pytest.raises(HTTPError, match="Status code"):
        future = model.embed(sequences=[invalid_seq])
        future.wait(timeout=TIMEOUT)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
def test_e2e_embedding_logits(session: OpenProtein, model_id: str):
    """Validate logits retrieval for a model that supports logits output."""
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")

    model = session.embedding.get_model(model_id)
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
@pytest.mark.parametrize("model_id", PER_MODEL_TEST_IDS)
def test_e2e_embedding_attn(session: OpenProtein, model_id: str):
    """Validate attention output for a model that supports attention."""
    if not _supports_model(session, model_id):
        pytest.skip(f"{model_id} model is not available in this backend")

    model = session.embedding.get_model(model_id)
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
    expected_count = len(query_ids) * n_sequences
    assert isinstance(results, list)
    assert len(results) == expected_count
    for entry in results:
        assert isinstance(entry.sequence, str)
        assert len(entry.sequence) > 0
        assert isinstance(entry.score, np.ndarray)


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
    assert isinstance(results, list)
    assert len(results) == n_sequences
    for entry in results:
        assert isinstance(entry.sequence, str)
        assert len(entry.sequence) > 0
        assert isinstance(entry.score, np.ndarray)


@pytest.mark.e2e
def test_e2e_msa_to_prompt_to_poet_score(session: OpenProtein):
    """Chain MSA -> prompt sample -> PoET score."""
    if "poet" not in {m.id for m in session.embedding.list_models()}:
        pytest.skip("poet model is not available in this backend")

    seed_sequence = random_sequence_real(200).encode()
    msa_future = session.align.create_msa(seed=seed_sequence)
    prompt_future = msa_future.sample_prompt(num_ensemble_prompts=1)
    assert prompt_future.wait_until_done(timeout=GENERATE_TIMEOUT)

    score_sequences = [seed_sequence]
    score_results = session.embedding.poet.score(
        sequences=score_sequences, prompt=prompt_future
    ).wait(timeout=GENERATE_TIMEOUT)

    assert len(score_results) == len(score_sequences)
    for row in score_results:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()


@pytest.mark.e2e
def test_e2e_msa_to_prompt_to_poet2_score(session: OpenProtein):
    """Chain MSA -> prompt sample -> PoET-2 score."""
    if "poet-2" not in {m.id for m in session.embedding.list_models()}:
        pytest.skip("poet-2 model is not available in this backend")

    seed_sequence = random_sequence_real(200).encode()
    msa_future = session.align.create_msa(seed=seed_sequence)
    prompt_future = msa_future.sample_prompt(num_ensemble_prompts=1)
    assert prompt_future.wait_until_done(timeout=GENERATE_TIMEOUT)

    score_sequences = [seed_sequence]
    score_results = session.embedding.poet2.score(
        sequences=score_sequences, prompt=prompt_future
    ).wait(timeout=GENERATE_TIMEOUT)

    assert len(score_results) == len(score_sequences)
    for row in score_results:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()


# Models whose inference is conditioned on a prompt (and, for PoET-2, a query).
PROMPT_MODELS = ["poet", "poet-2"]


def _novel_multichain_context() -> list[str]:
    """A two-chain context joined by ':' — novel each run to bust server-side caches."""
    chain_a = random_sequence_real(80)
    chain_b = random_sequence_real(80)
    return [f"{chain_a}:{chain_b}"]


def _require_prompt_model(session: OpenProtein, model_id: str):
    """Skip if the model is unavailable on this backend; otherwise return it."""
    if model_id not in {m.id for m in session.embedding.list_models()}:
        pytest.skip(f"{model_id} is not available in this backend")
    return session.embedding.get_model(model_id)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PROMPT_MODELS)
def test_e2e_multichain_prompt_embed(session: OpenProtein, model_id: str):
    """Embed a sequence conditioned on a multichain (':'-delimited) prompt."""
    model = _require_prompt_model(session, model_id)
    prompt = session.prompt.create_prompt(_novel_multichain_context())
    assert prompt.wait_until_done(timeout=TIMEOUT)

    sequences = [random_sequence_real(80).encode()]
    results = model.embed(sequences=sequences, prompt=prompt).wait(
        timeout=GENERATE_TIMEOUT
    )

    assert len(results) == len(sequences)
    sequence, embedding = results[0]
    assert sequence == sequences[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (model.metadata.dimension,)
    assert np.isfinite(embedding).all()


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PROMPT_MODELS)
def test_e2e_multichain_prompt_score(session: OpenProtein, model_id: str):
    """Score a sequence against a multichain (':'-delimited) prompt."""
    model = _require_prompt_model(session, model_id)
    prompt = session.prompt.create_prompt(_novel_multichain_context())
    assert prompt.wait_until_done(timeout=TIMEOUT)

    sequences = [random_sequence_real(80).encode()]
    results = model.score(sequences=sequences, prompt=prompt).wait(
        timeout=GENERATE_TIMEOUT
    )

    assert len(results) == len(sequences)
    for row in results:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", PROMPT_MODELS)
def test_e2e_multichain_prompt_generate(session: OpenProtein, model_id: str):
    """Generate sequences conditioned on a multichain (':'-delimited) prompt."""
    model = _require_prompt_model(session, model_id)
    n_samples = 2
    prompt = session.prompt.create_prompt(_novel_multichain_context())
    assert prompt.wait_until_done(timeout=TIMEOUT)

    future = model.generate(prompt=prompt, num_samples=n_samples, temperature=1.0)
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == n_samples
    for entry in results:
        assert isinstance(entry.sequence, str)
        assert len(entry.sequence) > 0
        assert isinstance(entry.score, np.ndarray)


@pytest.mark.e2e
def test_e2e_multichain_query_poet2_embed(session: OpenProtein):
    """PoET-2 embed with a multichain (':'-delimited) query."""
    _require_prompt_model(session, "poet-2")
    query = f"{random_sequence_real(80)}:{random_sequence_real(80)}"

    sequences = [random_sequence_real(80).encode()]
    results = session.embedding.poet2.embed(sequences=sequences, query=query).wait(
        timeout=GENERATE_TIMEOUT
    )

    assert len(results) == len(sequences)
    sequence, embedding = results[0]
    assert sequence == sequences[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (session.embedding.poet2.metadata.dimension,)
    assert np.isfinite(embedding).all()


@pytest.mark.e2e
def test_e2e_multichain_query_poet2_score(session: OpenProtein):
    """PoET-2 score with a multichain (':'-delimited) query."""
    _require_prompt_model(session, "poet-2")
    query = f"{random_sequence_real(80)}:{random_sequence_real(80)}"

    sequences = [random_sequence_real(80).encode()]
    results = session.embedding.poet2.score(sequences=sequences, query=query).wait(
        timeout=GENERATE_TIMEOUT
    )

    assert len(results) == len(sequences)
    for row in results:
        assert isinstance(row.sequence, str)
        assert len(row.sequence) > 0
        assert isinstance(row.score, np.ndarray)
        assert row.score.size > 0
        assert np.isfinite(row.score).all()
