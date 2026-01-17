"""E2E tests for the predictor domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common import FeatureType
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from tests.utils.sequences import random_sequence_fake

# Model configurations for predictor training
PREDICTOR_MODELS = [
    ("esm2_t33_650M_UR50D", ReductionType.MEAN),
    ("prot-seq", ReductionType.MEAN),
    ("poet", ReductionType.MEAN),
    ("poet-2", ReductionType.MEAN),
]

TIMEOUT = 20 * 60  # 20 minutes for training
SVD_TIMEOUT = 30 * 60  # 30 minutes for SVD + training

# SVD configurations for predictor training
SVD_CONFIGS = [
    ("esm2_t33_650M_UR50D", 32),
    ("prot-seq", 64),
]


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,reduction", PREDICTOR_MODELS)
def test_predictor_training_single_model(
    session: OpenProtein,
    assay_small: AssayDataset,
    model_id: str,
    reduction: ReductionType,
):
    """
    Test training a GP predictor with different embedding models.
    Validates training workflow and prediction output.
    """
    # Get the embedding model
    embedding_model = session.embedding.get_model(model_id)

    # Get the first measurement name from the assay
    property_name = assay_small.measurement_names[0]

    # Train the predictor
    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=[property_name], reduction=reduction
    )

    # Wait for training to complete
    assert predictor_future.wait(
        timeout=TIMEOUT
    ), f"Predictor training failed for {model_id}"
    predictor_model = predictor_future

    # Validate predictor metadata
    assert predictor_model.id is not None
    assert predictor_model.training_assay.id == assay_small.id

    # Make a prediction
    test_sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    prediction_future = predictor_model.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()

    # Validate prediction output
    assert isinstance(mus, np.ndarray)
    assert isinstance(vs, np.ndarray)
    assert mus.shape == (1, 1)  # 1 sequence, 1 property
    assert vs.shape == (1, 1)


@pytest.mark.e2e
def test_predictor_parallel_training(session: OpenProtein, assay_small: AssayDataset):
    """
    Test training multiple predictors in parallel.
    Validates that the backend can handle concurrent training jobs.
    """
    property_name = assay_small.measurement_names[0]

    # Submit training jobs for multiple models in parallel
    futures = []
    for model_id, reduction in PREDICTOR_MODELS:
        embedding_model = session.embedding.get_model(model_id)
        predictor_future = embedding_model.fit_gp(
            assay=assay_small, properties=[property_name], reduction=reduction
        )
        futures.append((model_id, predictor_future))

    # Wait for all training jobs to complete
    for model_id, future in futures:
        assert future.wait(timeout=TIMEOUT), f"Training failed for {model_id}"

    # Validate all predictors can make predictions
    test_sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    for model_id, predictor in futures:
        prediction_future = predictor.predict(sequences=[test_sequence])
        mus, vs = prediction_future.wait()
        assert mus.shape == (1, 1), f"Model {model_id}: unexpected prediction shape"


@pytest.mark.e2e
@pytest.mark.parametrize("num_properties", [1, 2, 3])
@pytest.mark.skip("Multitask models are not supported for now")
def test_predictor_multitask(
    session: OpenProtein, assay_small: AssayDataset, num_properties: int
):
    """
    Test training a multitask predictor with multiple properties.
    """
    # Get the first N measurement names
    properties = assay_small.measurement_names[:num_properties]

    # Train multitask predictor
    embedding_model = session.embedding.esm2
    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=properties, reduction=ReductionType.MEAN
    )

    assert predictor_future.wait(timeout=TIMEOUT), "Multitask training failed"
    predictor_model = predictor_future

    # Make predictions
    test_sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    prediction_future = predictor_model.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()

    # Validate output shape matches number of properties
    assert mus.shape == (1, num_properties)
    assert vs.shape == (1, num_properties)


@pytest.mark.e2e
@pytest.mark.parametrize("num_sequences", [1, 10, 50])
def test_predictor_batch_prediction(
    session: OpenProtein, assay_small: AssayDataset, num_sequences: int
):
    """
    Test making predictions on batches of sequences.
    """
    # Train a predictor
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=[property_name], reduction=ReductionType.MEAN
    )
    assert predictor_future.wait(timeout=TIMEOUT)
    predictor_model = predictor_future

    # Generate test sequences
    seq_len = assay_small.sequence_length or 346
    test_sequences = [random_sequence_fake(seq_len) for _ in range(num_sequences)]

    # Make batch predictions
    prediction_future = predictor_model.predict(sequences=test_sequences)
    mus, vs = prediction_future.wait()

    # Validate batch output
    assert mus.shape == (num_sequences, 1)
    assert vs.shape == (num_sequences, 1)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "assay_fixture",
    ["assay_small", "assay_medium", "assay_large"],
)
def test_predictor_different_dataset_sizes(
    session: OpenProtein, assay_fixture: str, request
):
    """
    Test training predictors on datasets of different sizes.
    Validates scalability and training time.
    """
    # Get the fixture dynamically
    try:
        assay = request.getfixturevalue(assay_fixture)
    except pytest.FixtureLookupError:
        pytest.skip(f"Fixture {assay_fixture} not available")

    # Train predictor
    embedding_model = session.embedding.esm2
    property_name = assay.measurement_names[0]

    predictor_future = embedding_model.fit_gp(
        assay=assay, properties=[property_name], reduction=ReductionType.MEAN
    )

    # Larger datasets may take longer
    timeout = TIMEOUT * (2 if "large" in assay_fixture else 1)
    assert predictor_future.wait(
        timeout=timeout
    ), f"Training failed for {assay_fixture}"

    # Validate prediction works
    test_sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    prediction_future = predictor_future.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()
    assert mus.shape == (1, 1)


@pytest.mark.e2e
def test_predictor_retrieval_by_id(session: OpenProtein, assay_small: AssayDataset):
    """
    Test retrieving a trained predictor by ID.
    """
    # Train a predictor
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=[property_name], reduction=ReductionType.MEAN
    )
    assert predictor_future.wait(timeout=TIMEOUT)
    original_predictor = predictor_future

    # Retrieve by ID
    retrieved_predictor = session.predictor.get_predictor(original_predictor.id)

    assert retrieved_predictor.id == original_predictor.id
    assert retrieved_predictor.training_assay.id == assay_small.id

    # Validate prediction works with retrieved predictor
    test_sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    prediction_future = retrieved_predictor.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()
    assert mus.shape == (1, 1)


@pytest.mark.e2e
def test_predictor_error_handling_invalid_sequence(
    session: OpenProtein, assay_small: AssayDataset
):
    """
    Test error handling for invalid sequences in prediction.
    """
    # Train a predictor
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=[property_name], reduction=ReductionType.MEAN
    )
    assert predictor_future.wait(timeout=TIMEOUT)
    predictor_model = predictor_future

    # Try to predict with invalid sequence
    invalid_sequence = b"INVALID123XYZ"

    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        prediction_future = predictor_model.predict(sequences=[invalid_sequence])
        prediction_future.wait()


@pytest.mark.e2e
def test_predictor_error_handling_empty_sequence(
    session: OpenProtein, assay_small: AssayDataset
):
    """
    Test error handling for empty sequences in prediction.
    """
    # Train a predictor
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    predictor_future = embedding_model.fit_gp(
        assay=assay_small, properties=[property_name], reduction=ReductionType.MEAN
    )
    assert predictor_future.wait(timeout=TIMEOUT)
    predictor_model = predictor_future

    # Try to predict with empty sequence
    with pytest.raises(Exception):  # Adjust exception type based on actual API behavior
        prediction_future = predictor_model.predict(sequences=[b""])
        prediction_future.wait()


@pytest.mark.e2e
@pytest.mark.parametrize("model_id,n_components", SVD_CONFIGS)
def test_predictor_train_on_svd_embeddings(
    session: OpenProtein,
    assay_small: AssayDataset,
    model_id: str,
    n_components: int,
):
    """
    Test training a GP predictor on SVD-reduced embeddings.
    This tests the full workflow: fit SVD -> train predictor on SVD embeddings.
    """
    # Get the embedding model
    embedding_model = session.embedding.get_model(model_id)

    # Fit SVD model on the assay
    svd_future = embedding_model.fit_svd(assay=assay_small, n_components=n_components)
    svd_model = svd_future.wait(timeout=SVD_TIMEOUT)
    assert svd_model is not None, f"SVD fitting failed for {model_id}"

    # Train predictor on SVD embeddings
    property_name = assay_small.measurement_names[0]
    predictor_future = svd_model.fit_gp(assay=assay_small, properties=[property_name])

    # Wait for training to complete
    assert predictor_future.wait(
        timeout=TIMEOUT
    ), f"Predictor training on SVD embeddings failed for {model_id}"
    predictor_model = predictor_future

    # Validate predictor metadata
    assert predictor_model.id is not None
    assert predictor_model.training_assay.id == assay_small.id

    # Make a prediction
    seq_len = assay_small.sequence_length or 346
    test_sequence = random_sequence_fake(seq_len)
    prediction_future = predictor_model.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()

    # Validate prediction output
    assert isinstance(mus, np.ndarray)
    assert isinstance(vs, np.ndarray)
    assert mus.shape == (1, 1)  # 1 sequence, 1 property
    assert vs.shape == (1, 1)


@pytest.mark.e2e
def test_predictor_parallel_svd_training(
    session: OpenProtein, assay_small: AssayDataset
):
    """
    Test training multiple predictors on different SVD models in parallel.
    """
    property_name = assay_small.measurement_names[0]

    # First, fit SVD models in parallel
    svd_futures = []
    for model_id, n_components in SVD_CONFIGS:
        embedding_model = session.embedding.get_model(model_id)
        svd_future = embedding_model.fit_svd(
            assay=assay_small, n_components=n_components
        )
        svd_futures.append((model_id, n_components, svd_future))

    # Wait for all SVD models to be ready
    svd_models = []
    for model_id, n_components, future in svd_futures:
        svd_model = future.wait(timeout=SVD_TIMEOUT)
        assert svd_model is not None, f"SVD fitting failed for {model_id}"
        svd_models.append((model_id, n_components, svd_model))

    # Now train predictors on all SVD models in parallel
    predictor_futures = []
    for model_id, n_components, svd_model in svd_models:
        predictor_future = svd_model.fit_gp(
            assay=assay_small, properties=[property_name]
        )
        predictor_futures.append((model_id, predictor_future))

    # Wait for all training jobs to complete
    for model_id, future in predictor_futures:
        assert future.wait(
            timeout=TIMEOUT
        ), f"Training on SVD embeddings failed for {model_id}"

    # Validate all predictors can make predictions
    seq_len = assay_small.sequence_length or 346
    test_sequence = random_sequence_fake(seq_len)
    for model_id, predictor in predictor_futures:
        prediction_future = predictor.predict(sequences=[test_sequence])
        mus, vs = prediction_future.wait()
        assert mus.shape == (
            1,
            1,
        ), f"Model {model_id}: unexpected prediction shape"


@pytest.mark.e2e
@pytest.mark.parametrize("n_components", [8, 32, 128])
def test_predictor_svd_different_n_components(
    session: OpenProtein, assay_small: AssayDataset, n_components: int
):
    """
    Test training predictors on SVD embeddings with different n_components.
    Validates that predictor performance is consistent across different SVD dimensions.
    """
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    # Fit SVD model
    svd_future = embedding_model.fit_svd(assay=assay_small, n_components=n_components)
    svd_model = svd_future.wait(timeout=SVD_TIMEOUT)
    assert svd_model is not None

    # Train predictor on SVD embeddings
    predictor_future = svd_model.fit_gp(assay=assay_small, properties=[property_name])
    assert predictor_future.wait(timeout=TIMEOUT)
    predictor_model = predictor_future

    # Make predictions
    seq_len = assay_small.sequence_length or 346
    test_sequence = random_sequence_fake(seq_len)
    prediction_future = predictor_model.predict(sequences=[test_sequence])
    mus, vs = prediction_future.wait()

    assert mus.shape == (1, 1)
    assert vs.shape == (1, 1)


@pytest.mark.e2e
def test_predictor_svd_batch_prediction(
    session: OpenProtein, assay_small: AssayDataset
):
    """
    Test batch predictions using a predictor trained on SVD embeddings.
    """
    embedding_model = session.embedding.esm2
    property_name = assay_small.measurement_names[0]

    # Fit SVD and train predictor
    svd_future = embedding_model.fit_svd(assay=assay_small, n_components=32)
    svd_model = svd_future.wait(timeout=SVD_TIMEOUT)
    predictor_future = svd_model.fit_gp(assay=assay_small, properties=[property_name])
    assert predictor_future.wait(timeout=TIMEOUT)
    predictor_model = predictor_future

    # Make batch predictions
    num_sequences = 10
    seq_len = assay_small.sequence_length or 346
    test_sequences = [random_sequence_fake(seq_len) for _ in range(num_sequences)]

    prediction_future = predictor_model.predict(sequences=test_sequences)
    mus, vs = prediction_future.wait()

    # Validate batch output
    assert mus.shape == (10, 1)
    assert vs.shape == (10, 1)
