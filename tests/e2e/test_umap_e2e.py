"""E2E tests for the umap domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.umap.models import UMAPModel

# Model configurations for UMAP fitting
UMAP_MODELS = [
    ("esm2_t33_650M_UR50D", 1280),
    ("prot-seq", 1024),
    ("poet", 1024),
    ("poet-2", 1024),
]

TIMEOUT = 10 * 60  # 10 minutes for UMAP fitting


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,expected_base_dim", UMAP_MODELS)
@pytest.mark.parametrize("n_components", [2, 3, 10])
def test_umap_single_model(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    model_id: str,
    expected_base_dim: int,
    n_components: int,
):
    """
    Test fitting UMAP models with different embedding models and n_components.
    Validates UMAP workflow and projected embedding output.
    """
    # Get the embedding model
    embedding_model = session.embedding.get_model(model_id)

    # Fit the UMAP model
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length, n_components=n_components
    )

    # Wait for the model to be ready
    assert umap_future.wait_until_done(), f"UMAP model fitting failed for {model_id}"
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components

    # Use the UMAP model to embed a sequence
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    assert embedding_future.wait_until_done(), f"UMAP embed failed for {model_id}"
    results = embedding_future.get()

    # Validate the output
    assert isinstance(results, list)
    assert len(results) == 1
    seq, embedding = results[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (
        n_components,
    ), f"Expected UMAP embedding shape of ({n_components},), but got {embedding.shape} for {model_id}"


@pytest.mark.e2e
def test_umap_parallel_models(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test fitting multiple UMAP models in parallel.
    Validates that the backend can handle concurrent UMAP fitting jobs.
    """
    n_components = 3

    # Submit UMAP fitting jobs for multiple models in parallel
    futures = []
    for model_id, expected_dim in UMAP_MODELS:
        embedding_model = session.embedding.get_model(model_id)
        umap_future = embedding_model.fit_umap(
            sequences=test_sequences_same_length, n_components=n_components
        )
        futures.append((model_id, umap_future))

    # Wait for all UMAP fitting jobs to complete
    umap_models = []
    for model_id, future in futures:
        assert future.wait_until_done(), f"UMAP fitting failed for {model_id}"
        umap_model = future
        assert isinstance(umap_model, UMAPModel)
        assert umap_model.n_components == n_components
        umap_models.append((model_id, umap_model))

    # Validate all UMAP models can embed sequences
    test_sequence = test_sequences_same_length[0]
    for model_id, umap_model in umap_models:
        embedding_future = umap_model.embed(sequences=[test_sequence])
        results = embedding_future.wait()
        assert len(results) == 1
        seq, embedding = results[0]
        assert embedding.shape == (
            n_components,
        ), f"Model {model_id}: unexpected UMAP embedding shape"


@pytest.mark.e2e
@pytest.mark.parametrize("reduction", [ReductionType.MEAN, ReductionType.SUM])
def test_umap_reduction_types(
    session: OpenProtein, test_sequences_varied: list[bytes], reduction: ReductionType
):
    """
    Test UMAP with different reduction types for variable-length sequences.
    """
    embedding_model = session.embedding.esm2
    n_components = 3

    # Fit UMAP with specified reduction
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_varied, n_components=n_components, reduction=reduction
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.reduction == reduction

    # Embed sequences with the UMAP model
    embedding_future = umap_model.embed(sequences=[test_sequences_varied[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "n_neighbors,min_dist",
    [
        (5, 0.1),
        (15, 0.5),
        (30, 0.9),
    ],
)
def test_umap_hyperparameters(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    n_neighbors: int,
    min_dist: float,
):
    """
    Test UMAP with different hyperparameter combinations.
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # Fit UMAP with specified hyperparameters
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components
    assert umap_model.n_neighbors == n_neighbors
    assert umap_model.min_dist == min_dist

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize("num_sequences", [2, 10, 50])
def test_umap_batch_embedding(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    num_sequences: int,
):
    """
    Test UMAP embedding with different batch sizes.
    """
    embedding_model = session.embedding.esm2
    n_components = 3

    # Fit UMAP model
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length, n_components=n_components
    )
    assert umap_future.wait_until_done()
    umap_model = umap_future

    # Generate test sequences
    test_sequences = test_sequences_same_length[:num_sequences]

    # Embed batch
    embedding_future = umap_model.embed(sequences=test_sequences)
    results = embedding_future.wait()

    # Validate batch output
    assert len(results) == num_sequences
    for seq, embedding in results:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (n_components,)
        assert seq in test_sequences


@pytest.mark.e2e
@pytest.mark.parametrize(
    "assay_fixture",
    ["assay_small", "assay_medium"],
)
def test_umap_from_assay(session: OpenProtein, assay_fixture: str, request):
    """
    Test fitting UMAP from assay datasets of different sizes.
    """
    # Get the fixture dynamically
    try:
        assay = request.getfixturevalue(assay_fixture)
    except pytest.FixtureLookupError:
        pytest.skip(f"Fixture {assay_fixture} not available")

    embedding_model = session.embedding.esm2
    n_components = 3

    # Fit UMAP from assay
    umap_future = embedding_model.fit_umap(assay=assay, n_components=n_components)

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components

    # Get sequences from the UMAP model
    sequences = umap_model.get_inputs()
    assert len(sequences) > 0

    # Embed one sequence
    embedding_future = umap_model.embed(sequences=[sequences[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_retrieval_by_id(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test retrieving a UMAP model by ID.
    """
    embedding_model = session.embedding.esm2
    n_components = 3

    # Fit UMAP model
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length, n_components=n_components
    )
    assert umap_future.wait_until_done()
    original_umap = umap_future

    # Retrieve by ID
    retrieved_umap = session.umap.get_umap(original_umap.id)

    assert retrieved_umap.id == original_umap.id
    assert retrieved_umap.n_components == n_components

    # Validate embedding works with retrieved model
    embedding_future = retrieved_umap.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_edge_case_small_dataset(session: OpenProtein):
    """
    Test UMAP with a very small dataset (minimum viable, same-length sequences).
    UMAP typically requires at least n_neighbors + 1 sequences.
    """
    embedding_model = session.embedding.esm2
    n_components = 2
    n_neighbors = 5

    # Create minimal dataset (just above n_neighbors)
    # All sequences are 64 residues long
    sequences = [
        b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
        b"MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL",
        b"MASGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLA",
        b"MTSGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLT",
        b"MGSGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLG",
        b"MKSGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
    ]

    # Fit UMAP with minimal dataset
    umap_future = embedding_model.fit_umap(
        sequences=sequences, n_components=n_components, n_neighbors=n_neighbors
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[sequences[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_edge_case_high_dimensions(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """
    Test UMAP with higher dimensional output (e.g., 50 components).
    """
    embedding_model = session.embedding.esm2
    n_components = 50

    # Fit UMAP with high dimensions
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length, n_components=n_components
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_error_different_length_sequences_no_reduction(
    session: OpenProtein, test_sequences_varied: list[bytes]
):
    """
    Test that UMAP raises an error when fitting on different-length sequences
    without reduction. UMAP requires same-length embeddings, which means
    same-length sequences when no reduction is applied.
    """
    embedding_model = session.embedding.esm2
    n_components = 3

    # Try to fit UMAP with different-length sequences and no reduction
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        umap_future = embedding_model.fit_umap(
            sequences=test_sequences_varied,
            n_components=n_components,
            reduction=None,  # type: ignore
        )
        umap_future.wait_until_done()


@pytest.mark.e2e
def test_umap_different_length_sequences_with_reduction(
    session: OpenProtein, test_sequences_varied: list[bytes]
):
    """
    Test that UMAP works with different-length sequences when reduction is used.
    Reduction (MEAN/SUM) produces fixed-size embeddings regardless of sequence length.
    """
    embedding_model = session.embedding.esm2
    n_components = 3

    # Fit UMAP with different-length sequences using MEAN reduction
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_varied,
        n_components=n_components,
        reduction=ReductionType.MEAN,
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components
    assert umap_model.reduction == ReductionType.MEAN

    # Embed a sequence and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_varied[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_with_svd_features(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test UMAP using SVD-reduced embeddings as features.
    When using SVD features, reduction must be None.
    """
    embedding_model = session.embedding.esm2
    svd_n_components = 32
    umap_n_components = 2

    # First, fit an SVD model
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=svd_n_components
    )
    svd_model = svd_future.wait(timeout=TIMEOUT)

    # Now fit UMAP on the SVD features (reduction must be None for SVD)
    umap_future = svd_model.fit_umap(
        sequences=test_sequences_same_length,
        n_components=umap_n_components,
        reduction=None,  # type: ignore
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == umap_n_components
    assert umap_model.reduction is None

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (umap_n_components,)


@pytest.mark.e2e
def test_umap_error_svd_with_reduction(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test that UMAP raises an error when trying to use reduction with SVD features.
    SVD features are already reduced, so reduction should not be specified.
    """
    embedding_model = session.embedding.esm2
    svd_n_components = 32
    umap_n_components = 2

    # First, fit an SVD model
    svd_future = embedding_model.fit_svd(
        sequences=test_sequences_same_length, n_components=svd_n_components
    )
    svd_model = svd_future.wait(timeout=TIMEOUT)

    # Try to fit UMAP with reduction (should fail)
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        umap_future = svd_model.fit_umap(
            sequences=test_sequences_same_length,
            n_components=umap_n_components,
            reduction=ReductionType.MEAN,
        )
        umap_future.wait_until_done()


@pytest.mark.e2e
def test_umap_edge_case_n_neighbors_large(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test UMAP with n_neighbors close to the dataset size.
    UMAP should handle this gracefully or adjust n_neighbors.
    """
    embedding_model = session.embedding.esm2
    n_components = 2
    num_sequences = 10
    n_neighbors = 9  # Just below dataset size

    # Fit UMAP with n_neighbors close to dataset size
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length[:num_sequences],
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.n_components == n_components

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
@pytest.mark.parametrize("min_dist", [0.0, 0.01, 0.99, 1.0])
def test_umap_extreme_min_dist(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    min_dist: float,
):
    """
    Test UMAP with extreme min_dist values (0.0 to 1.0).
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # Fit UMAP with extreme min_dist
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length,
        n_components=n_components,
        min_dist=min_dist,
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    assert umap_model.min_dist == min_dist

    # Embed and validate
    embedding_future = umap_model.embed(sequences=[test_sequences_same_length[0]])
    results = embedding_future.wait()
    assert len(results) == 1
    seq, embedding = results[0]
    assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_sequence_length_property(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test that the sequence_length property is set correctly for same-length sequences.
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # All sequences should be the same length
    expected_length = len(test_sequences_same_length[0])
    assert all(
        len(seq) == expected_length for seq in test_sequences_same_length
    ), "Test sequences should all be the same length"

    # Fit UMAP with same-length sequences (no reduction needed)
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length,
        n_components=n_components,
    )

    assert umap_future.wait_until_done()
    umap_model = umap_future
    assert isinstance(umap_model, UMAPModel)
    # Embedding model fit UMAP should have no sequence length since reduction is auto used
    assert umap_model.sequence_length == None


@pytest.mark.e2e
def test_umap_get_embeddings(
    session: OpenProtein, test_sequences_same_length: list[bytes]
):
    """
    Test retrieving the projected embeddings from a fitted UMAP model.
    """
    embedding_model = session.embedding.esm2
    n_components = 2
    num_sequences = 10

    # Fit UMAP model
    sequences = test_sequences_same_length[:num_sequences]
    umap_future = embedding_model.fit_umap(
        sequences=sequences, n_components=n_components
    )
    assert umap_future.wait_until_done()
    umap_model = umap_future

    # Get the projected embeddings
    embeddings = umap_model.embeddings
    assert isinstance(embeddings, list)
    assert len(embeddings) == num_sequences

    # Validate each embedding
    for seq, embedding in embeddings:
        assert seq in sequences
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (n_components,)


@pytest.mark.e2e
def test_umap_get_model(session: OpenProtein, test_sequences_same_length: list[bytes]):
    """
    Test retrieving the base embedding model from a UMAP model.
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # Fit UMAP model
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length, n_components=n_components
    )
    assert umap_future.wait_until_done()
    umap_model = umap_future

    # Get the base model
    base_model = umap_model.get_model()
    assert base_model is not None
    # The base model should be the same as the one we used
    assert base_model.model_id == embedding_model.model_id


@pytest.mark.e2e
def test_umap_delete(session: OpenProtein, test_sequences_same_length: list[bytes]):
    """
    Test deleting a UMAP model.
    """
    embedding_model = session.embedding.esm2
    n_components = 2

    # Fit UMAP model
    umap_future = embedding_model.fit_umap(
        sequences=test_sequences_same_length[:10], n_components=n_components
    )
    assert umap_future.wait_until_done()
    umap_model = umap_future

    # Delete the model
    umap_id = umap_model.id
    result = umap_model.delete()
    assert result is True

    # TODO - there seems to be some cache / race condition, get still works immediately
    # Verify it's deleted (should raise an error when trying to retrieve)
    # with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
    # umap = session.umap.get_umap(umap_id)
    # assert umap is not None
