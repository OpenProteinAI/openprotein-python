"""E2E tests for the design domain."""

import os

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.design import ModelCriterion
from openprotein.errors import HTTPError
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(2.0)


@pytest.mark.e2e
def test_design_workflow_e2e(session: OpenProtein):
    """
    Tests a basic design E2E workflow:
    1. Get a pre-trained predictor model.
    2. Create a design criterion from that model.
    3. Start a design job.
    4. Wait for the design job to complete.
    5. Fetch and validate the results.
    """
    # This e2e flow requires a backend-stable, pre-trained predictor.
    # Keep this test enabled by default, but conditionally skip when the
    # dedicated predictor ID is not configured in this environment.
    predictor_id = os.getenv("OPENPROTEIN_E2E_DESIGN_PREDICTOR_ID")
    if not predictor_id:
        pytest.skip(
            "Set OPENPROTEIN_E2E_DESIGN_PREDICTOR_ID to run design workflow e2e"
        )
    try:
        predictor = session.predictor.get_predictor(predictor_id)
    except HTTPError as exc:
        pytest.skip(f"Configured design predictor is unavailable: {exc}")
    assert predictor is not None, "Failed to get predictor"

    # 2. Create a design criterion
    # We want to maximize the "yield" property predicted by our model
    criterion = ModelCriterion(model_id=predictor.id, measurement_name="yield")

    # 3. Start a design job
    assay = predictor.training_assay
    try:
        design_future = session.design.create_genetic_algorithm_design(
            assay=assay, criteria=criterion, num_steps=5, pop_size=10
        )
    except (HTTPError, NotImplementedError) as exc:
        pytest.skip(f"Design workflow unsupported in current backend: {exc}")

    # 4. Wait for the design job to complete
    assert design_future.wait_until_done(timeout=TIMEOUT), "Design job failed"
    results = design_future.get()

    # 5. Fetch and validate results
    assert len(results) > 0, "Design job produced no results"
    first_result = results[0]
    assert isinstance(first_result.sequence, str)
    assert len(first_result.sequence) > 0
    assert isinstance(first_result.scores, np.ndarray)
    assert first_result.scores.size > 0
    assert np.isfinite(first_result.scores).all()
