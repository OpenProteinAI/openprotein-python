"""E2E tests for the embeddings domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
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
TIMEOUT = 10 * 60


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
    _, embedding = results[0]
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
    for seq, emb in results:
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

    # Test with empty sequence - should raise an error or handle gracefully
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
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

    # Should raise an error or handle gracefully
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        future = model.embed(sequences=[invalid_seq])
        future.wait(timeout=TIMEOUT)
