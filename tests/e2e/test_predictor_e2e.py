"""E2E tests for the predictor domain."""

import os
import time

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common import FeatureType
from openprotein.common.reduction import ReductionType


@pytest.mark.e2e
@pytest.mark.skip(reason="Need to set up a testing dataset")
def test_predictor_workflow_e2e(api_session: OpenProtein):
    """
    Tests a basic predictor E2E workflow:
    1. Select a base embedding model.
    2. Select an assay.
    3. Train a GP predictor.
    4. Wait for the predictor to be ready.
    5. Use the predictor to make a prediction.
    6. Validate the output shape.
    """
    # 1. Select the base model
    embedding_model = api_session.embedding.esm2

    # 2. Select an assay (this assay is known to be in the test environment)
    assay = api_session.data.get("6155a68a5c378f01b13dd11c")

    # 3. Train the predictor
    predictor_future = embedding_model.fit_gp(
        assay=assay, properties=["yield"], reduction=ReductionType.MEAN
    )

    # 4. Wait for the predictor to be ready
    assert predictor_future.wait(), "Predictor training failed"
    predictor_model = predictor_future

    # 5. Use the predictor to make a prediction
    sequence = b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    prediction_future = predictor_model.predict(sequences=[sequence])
    mus, vs = prediction_future.get()

    # 6. Validate the output
    assert isinstance(mus, np.ndarray)
    assert isinstance(vs, np.ndarray)
    assert mus.shape == (1, 1)  # 1 sequence, 1 property
    assert vs.shape == (1, 1)
