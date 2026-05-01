"""E2E tests for the design domain."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.design import ModelCriterion
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(2.0)


@pytest.mark.e2e
def test_design_workflow_e2e(session: OpenProtein, assay_small: AssayDataset):
    """
    Self-contained design GA workflow:
    1. Train an ESM2 GP predictor on `assay_small`.
    2. Build a ModelCriterion against that predictor's first measurement.
    3. Run a tiny GA (num_steps=2, pop_size=4).
    4. Validate the results.
    """
    property_name = assay_small.measurement_names[0]

    embedding_model = session.embedding.esm2
    predictor_future = embedding_model.fit_gp(
        assay=assay_small,
        properties=[property_name],
        reduction=ReductionType.MEAN,
    )
    assert predictor_future.wait(timeout=TIMEOUT), "Predictor training failed"
    predictor = predictor_future
    assert predictor.id is not None

    # The SDK's ModelCriterion requires direction and target to be set
    # before serialization (otherwise the inner `criterion` field is dropped
    # by Pydantic's union resolution and the backend rejects the request
    # with KeyError 'criterion'). `> 0` is a generic maximize-toward-positive
    # constraint suitable for a smoke test against any numeric assay.
    criterion = (
        ModelCriterion(model_id=predictor.id, measurement_name=property_name) > 0
    )

    design_future = session.design.create_genetic_algorithm_design(
        assay=assay_small,
        criteria=criterion,
        num_steps=2,
        pop_size=4,
    )
    assert design_future.wait_until_done(timeout=TIMEOUT), "Design job failed"
    results = design_future.get()

    assert len(results) > 0, "Design job produced no results"
    first_result = results[0]
    assert isinstance(first_result.sequence, str)
    assert len(first_result.sequence) > 0
    assert isinstance(first_result.scores, np.ndarray)
    assert first_result.scores.size > 0
    assert np.isfinite(first_result.scores).all()
