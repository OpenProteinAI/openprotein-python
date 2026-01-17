"""E2E tests for the svd domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.svd.models import SVDModel

# Model configurations for SVD fitting
SVD_MODELS = [
    ("esm2_t33_650M_UR50D", 1280),
    ("prot-seq", 1024),
    ("poet", 1024),
    ("poet-2", 1024),
]

TIMEOUT = 10 * 60  # 10 minutes for SVD fitting


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,expected_base_dim", SVD_MODELS)
@pytest.mark.parametrize("n_components", [8, 32, 128])
def test_svd_single_model(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    model_id: str,
    expected_base_dim: int,
    n_components: int,
):
    """
    Test fitting SVD models with different embedding models and n_components.
    Validates SVD workflow and reduced embedding output.
    """
    # Get the embedding model
    embedding_model = session.embedding.get_model(model_id)

    # Fit the SVD model
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=n_components
    )

    # Wait for the model to be ready
    svd_model = svd_future.wait()
    assert isinstance(
        svd_model, SVDModel
    ), f"SVD model fitting failed for {model_id} with {n_components} components"
    assert svd_model.n_components == n_components

    # Use the SVD model to embed a sequence
    embedding_future = svd_model.embed(sequences=[test_sequences_same_length[0]])
    assert embedding_future.wait_until_done(), f"SVD embed failed for {model_id}"
    results = embedding_future.get()

    # Validate the output
    assert isinstance(results, list)
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_same_length[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (
        n_components,
    ), f"Expected SVD embedding shape of ({n_components},), but got {embedding.shape} for {model_id}"


@pytest.mark.e2e
def test_svd_parallel_models(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test fitting multiple SVD models in parallel.
    Validates that the backend can handle concurrent SVD fitting jobs.
    """
    n_components = 32

    # Submit SVD fitting jobs for multiple models in parallel
    futures: list[tuple[str, SVDModel]] = []
    for model_id, expected_dim in SVD_MODELS:
        embedding_model = session.embedding.get_model(model_id)
        svd_future = embedding_model.fit_svd(
            sequences=test_sequences_same_length, n_components=n_components
        )
        futures.append((model_id, svd_future))

    # Wait for all SVD fitting jobs to complete
    svd_models: list[tuple[str, SVDModel]] = []
    for model_id, future in futures:
        future.wait_until_done(timeout=TIMEOUT)
        svd_model = future
        assert isinstance(svd_model, SVDModel), f"SVD fitting failed for {model_id}"
        assert svd_model.n_components == n_components
        svd_models.append((model_id, svd_model))

    # Validate all SVD models can embed sequences
    test_sequence = test_sequences_same_length[0]
    for model_id, svd_model in svd_models:
        embedding_future = svd_model.embed(sequences=[test_sequence])
        results = embedding_future.wait()
        assert len(results) == 1
        seq, embedding = results[0]
        assert seq == test_sequences_same_length[0]
        assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize("reduction", [ReductionType.MEAN, ReductionType.SUM])
def test_svd_reduction_types(
    session: OpenProtein, test_sequences_varied: list[bytes], reduction: ReductionType
):
    """
    Test SVD with different reduction types for variable-length sequences.
    """
    embedding_model = session.embedding.esm2
    n_components = 32

    # Fit SVD with specified reduction
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_varied, n_components=n_components, reduction=reduction
    )

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    assert svd_model.reduction == reduction

    # Embed sequences with the SVD model
    embedding_future = svd_model.embed(sequences=[test_sequences_varied[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_varied[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize("num_sequences", [2, 10, 50])
def test_svd_batch_embedding_same_length(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    num_sequences: int,
):
    """
    Test SVD embedding with different batch sizes (same-length sequences).
    """
    embedding_model = session.embedding.esm2
    n_components = 32

    # Fit SVD model
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=n_components
    )
    svd_future.wait(timeout=TIMEOUT)
    svd_model = svd_future

    # Generate test sequences
    test_sequences = test_sequences_same_length[:num_sequences]

    # Embed batch
    embedding_future = svd_model.embed(sequences=test_sequences)
    results = embedding_future.wait()

    # Validate batch output
    assert len(results) == num_sequences
    for seq, embedding in results:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "assay_fixture",
    ["assay_small", "assay_medium"],
)
def test_svd_from_assay(session: OpenProtein, assay_fixture: str, request):
    """
    Test fitting SVD from assay datasets of different sizes.
    """
    # Get the fixture dynamically
    try:
        assay = request.getfixturevalue(assay_fixture)
    except pytest.FixtureLookupError:
        pytest.skip(f"Fixture {assay_fixture} not available")

    embedding_model = session.embedding.esm2
    n_components = 32

    # Fit SVD from assay
    svd_future = embedding_model.fit_svd(assay=assay, n_components=n_components)

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    assert svd_model.n_components == n_components

    # Get sequences from the SVD model
    sequences = svd_model.get_inputs()
    assert len(sequences) > 0

    # Embed one sequence
    embedding_future = svd_model.embed(sequences=[sequences[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == sequences[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_svd_retrieval_by_id(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test retrieving an SVD model by ID.
    """
    embedding_model = session.embedding.esm2
    n_components = 32

    # Fit SVD model
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=n_components
    )
    original_svd = svd_future.wait(timeout=TIMEOUT)

    # Retrieve by ID
    retrieved_svd = session.svd.get_svd(original_svd.id)

    assert retrieved_svd.id == original_svd.id
    assert retrieved_svd.n_components == n_components

    # Validate embedding works with retrieved model
    embedding_future = retrieved_svd.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_same_length[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_svd_edge_case_n_components_large(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """
    Test SVD with n_components exceeding the number of sequences.
    The backend should reduce n_components to min(M, N) where M is the number
    of sequences and N is the embedding dimension.
    """
    embedding_model = session.embedding.esm2
    # Use n_components larger than number of sequences
    n_components = 256
    num_sequences = 5

    # Fit SVD with few sequences
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length[:num_sequences], n_components=n_components
    )

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    # The backend should reduce n_components to min(num_sequences, embedding_dim)
    # Since we have 5 sequences and ESM2 has 1280 dimensions, it should be reduced to 5
    assert svd_model.n_components == num_sequences, (
        f"Expected n_components to be reduced to {num_sequences}, "
        f"but got {svd_model.n_components}"
    )
    
    # Verify embedding works with the reduced components
    embedding_future = svd_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_same_length[0]
    assert embedding.shape == (num_sequences,)


@pytest.mark.e2e
def test_svd_edge_case_small_n_components(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """
    Test SVD with very small n_components (e.g., 2).
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # Fit SVD with minimal components
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=n_components
    )

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    assert svd_model.n_components == n_components

    # Embed and validate
    embedding_future = svd_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_same_length[0]
    assert embedding.shape == (n_components,)



@pytest.mark.e2e
def test_svd_edge_case_n_components_exceeds_embedding_dim(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """
    Test SVD with n_components exceeding the embedding dimension.
    The backend should reduce n_components to min(M, N) where M is the number
    of sequences and N is the embedding dimension.
    """
    embedding_model = session.embedding.esm2
    # ESM2 has 1280 dimensions, use more than that
    n_components = 2000
    num_sequences = 50  # More sequences than embedding dim

    # Fit SVD with n_components > embedding_dim
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length[:num_sequences], n_components=n_components
    )

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    # The backend should reduce n_components to min(num_sequences, embedding_dim)
    # Since we have 50 sequences and ESM2 has 1280 dimensions, it should be reduced to 50
    expected_components = min(num_sequences, 1280)
    assert svd_model.n_components == expected_components, (
        f"Expected n_components to be reduced to {expected_components}, "
        f"but got {svd_model.n_components}"
    )
    
    # Verify embedding works with the reduced components
    embedding_future = svd_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_same_length[0]
    assert embedding.shape == (expected_components,)

@pytest.mark.e2e
def test_svd_error_different_length_sequences_no_reduction(
    session: OpenProtein, test_sequences_varied: list[bytes]
):
    """
    Test that SVD raises an error when fitting on different-length sequences
    without reduction. SVD requires same-length embeddings, which means
    same-length sequences when no reduction is applied.
    """
    embedding_model = session.embedding.esm2
    n_components = 32

    # Try to fit SVD with different-length sequences and no reduction
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        svd_future = embedding_model.fit_svd(
            sequences=test_sequences_varied, n_components=n_components
        )
        svd_future.wait(timeout=TIMEOUT)


@pytest.mark.e2e
def test_svd_different_length_sequences_with_reduction(
    session: OpenProtein, test_sequences_varied: list[bytes]
):
    """
    Test that SVD works with different-length sequences when reduction is used.
    Reduction (MEAN/SUM) produces fixed-size embeddings regardless of sequence length.
    """
    embedding_model = session.embedding.esm2
    n_components = 32

    # Fit SVD with different-length sequences using MEAN reduction
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_varied,
        n_components=n_components,
        reduction=ReductionType.MEAN,
    )

    svd_model = svd_future.wait(timeout=TIMEOUT)
    assert isinstance(svd_model, SVDModel)
    assert svd_model.n_components == n_components
    assert svd_model.reduction == ReductionType.MEAN

    # Embed a sequence and validate
    embedding_future = svd_model.embed(sequences=[test_sequences_varied[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert seq == test_sequences_varied[0]
    assert embedding.shape == (n_components,)
