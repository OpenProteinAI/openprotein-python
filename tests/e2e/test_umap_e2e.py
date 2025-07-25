"""E2E tests for the umap domain."""

import os

import numpy as np
import pytest

from openprotein import OpenProtein


@pytest.mark.e2e
def test_umap_workflow_e2e(api_session: OpenProtein):
    """
    Tests a basic UMAP E2E workflow:
    1. Select a base embedding model.
    2. Fit a UMAP model on a set of sequences.
    3. Wait for the UMAP model to be ready.
    4. Use the UMAP model to embed a sequence.
    5. Validate the output embedding's shape.
    """
    # 1. Select the base model
    embedding_model = api_session.embedding.get_model("esm2_t33_650M_UR50D")
    n_components = 3  # UMAP can be any dimension, using 3 for this test

    # 2. Fit the UMAP model
    sequences = [
        b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
        b"MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
        b"MASGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
    ]
    umap_future = embedding_model.fit_umap(
        sequences=sequences, n_components=n_components
    )

    # 3. Wait for the model to be ready
    assert umap_future.wait_until_done(), "UMAP model fitting failed or returned None"
    umap_model = umap_future
    assert umap_model.n_components == n_components

    # 4. Use the UMAP model to embed a sequence
    embedding_future = umap_model.embed(sequences=[sequences[0]])
    assert embedding_future.wait_until_done(), "UMAP embed failed"
    results = embedding_future.get()

    # 5. Validate the output
    assert isinstance(results, list)
    assert len(results) == 1
    embedding = results[0]
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (
        n_components,
    ), f"Expected UMAP embedding shape of ({n_components},), but got {embedding.shape}"
