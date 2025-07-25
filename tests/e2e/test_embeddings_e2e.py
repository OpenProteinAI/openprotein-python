"""E2E tests for the embeddings domain."""

import os

import numpy as np
import pytest

from openprotein import OpenProtein


@pytest.mark.e2e
def test_embedding_workflow_e2e(api_session: OpenProtein):
    """
    Tests a basic E2E workflow:
    1. Connect to the session.
    2. Select an embedding model (ESM2).
    3. Embed a single sequence.
    4. Fetch the result and validate its structure.
    """
    # 1. Session is already connected via the api_session fixture.

    # 2. Select the model
    # Use a well-known, relatively small model
    model = api_session.embedding.get_model("esm2_t33_650M_UR50D")
    assert model is not None, "Failed to get model from session"

    # 3. Embed a sequence
    sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    future = model.embed(sequences=[sequence])

    # 4. Fetch the result and validate
    results = future.get()
    assert isinstance(results, list), f"Expected a list, but got {type(results)}"
    assert len(results) == 1, f"Expected 1 result, but got {len(results)}"

    embedding = results[0]
    assert isinstance(embedding, np.ndarray), "Embedding is not a numpy array"
    # ESM2 650M model has an embedding dimension of 1280
    assert embedding.shape == (
        1280,
    ), f"Expected embedding shape of (1280,), but got {embedding.shape}"
