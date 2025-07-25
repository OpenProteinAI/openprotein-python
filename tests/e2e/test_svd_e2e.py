"""E2E tests for the svd domain."""

import os

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.svd.models import SVDModel


@pytest.mark.e2e
def test_svd_workflow_e2e(api_session: OpenProtein):
    """
    Tests a basic SVD E2E workflow:
    1. Select a base embedding model.
    2. Fit an SVD model on a set of sequences.
    3. Wait for the SVD model to be ready.
    4. Use the SVD model to embed a sequence.
    5. Validate the output embedding's shape.
    """
    # 1. Select the base model
    embedding_model = api_session.embedding.esm2
    n_components = 32

    # 2. Fit the SVD model
    sequences = [
        b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
        b"MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
    ]
    svd_future = embedding_model.fit_svd(sequences=sequences, n_components=n_components)

    # 3. Wait for the model to be ready
    # The .wait() on the SVDModel future blocks until the fitting job is complete and returns the svd
    svd_model = svd_future.wait()
    assert isinstance(svd_model, SVDModel), "SVD model fitting failed or returned None"
    assert svd_model.n_components == n_components

    # 4. Use the SVD model to embed a sequence
    embedding_future = svd_model.embed(sequences=[sequences[0]])
    assert embedding_future.wait_until_done(), "SVD embed failed"
    results = embedding_future.get()

    # 5. Validate the output
    assert isinstance(results, list)
    assert len(results) == 1
    embedding = results[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (
        n_components,
    ), f"Expected SVD embedding shape of ({n_components},), but got {embedding.shape}"
