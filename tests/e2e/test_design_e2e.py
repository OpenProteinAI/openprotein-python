"""E2E tests for the design domain."""

import os
import time

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.design import ModelCriterion


@pytest.mark.e2e
@pytest.mark.skip(reason="Long-running test, requires a trained predictor")
def test_design_workflow_e2e(api_session: OpenProtein):
    """
    Tests a basic design E2E workflow:
    1. Get a pre-trained predictor model.
    2. Create a design criterion from that model.
    3. Start a design job.
    4. Wait for the design job to complete.
    5. Fetch and validate the results.
    """
    # 1. Get a pre-trained predictor
    # This ID is for a predictor known to be in the test environment
    predictor_id = "your_predictor_id_here"
    predictor = api_session.predictor.get_predictor(predictor_id)
    assert predictor is not None, "Failed to get predictor"

    # 2. Create a design criterion
    # We want to maximize the "yield" property predicted by our model
    criterion = ModelCriterion(model_id=predictor.id, measurement_name="yield")

    # 3. Start a design job
    assay = predictor.training_assay
    design_future = api_session.design.create_genetic_algorithm_design(
        assay=assay, criteria=criterion, num_steps=5, pop_size=10
    )

    # 4. Wait for the design job to complete
    assert design_future.wait_until_done(), "Design job failed"
    results = design_future.get()

    # 5. Fetch and validate results
    assert len(results) > 0, "Design job produced no results"
    first_result = results[0]
    assert isinstance(first_result.sequence, str)
    assert isinstance(first_result.scores, np.ndarray)
