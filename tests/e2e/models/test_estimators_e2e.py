"""Estimators on session.models, chained end to end."""

from uuid import uuid4

import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.jobs import JobStatus, _dispatch
from openprotein.models import (
    GPModel,
    RBFKernel,
    SVDModel,
)
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(3.0)


@pytest.mark.e2e
def test_an_svd_then_a_gp_on_an_embedded_assay(session: OpenProtein, mutated_sequences):
    """Embedding an assay, fitting an SVD and training a GP on its projections chains."""
    sequences = mutated_sequences(num_sequences=5, mutation_rate=0.05)
    table = pd.DataFrame({"sequence": sequences, "fitness": [0.1, 0.4, 0.2, 0.9, 0.5]})
    dataset = session.data.Assay.create(
        table=table,
        name=f"e2e-estimators-{uuid4().hex[:8]}",
        description="estimators e2e",
    )

    embeddings = session.models.esm2_t6_8M_UR50D.embed(dataset)
    projected = session.models.SVD(2).fit_transform(embeddings)
    trained = (
        session.models.GP(kernel="rbf")
        .fit(projected, property="fitness")
        .wait(timeout=TIMEOUT)
    )
    fitted_svd = projected.wait(timeout=TIMEOUT)

    assert isinstance(trained, GPModel)
    assert isinstance(fitted_svd, SVDModel)
    assert trained.spec is not None
    assert trained.spec["features"]["type"] == "SVD"
    assert trained.spec["features"]["model_id"] == fitted_svd.id
    kernel = trained.kernel
    assert isinstance(kernel, RBFKernel)
    assert kernel.type == "rbf"
    assert kernel.lengthscale > 0
    assert kernel.outputscale > 0
    assert trained.mean is not None
    noise = trained.noise
    assert noise is not None and noise > 0

    predicted = trained.predict(sequences[:2])
    assert predicted.wait_until_done(timeout=TIMEOUT)
    assert predicted.status is JobStatus.SUCCESS
    if _dispatch.is_models_job(predicted.job):
        assert len(predicted.get()) == 2
    else:
        mus, _ = session.jobs.get(predicted.id).wait(timeout=TIMEOUT)
        assert len(mus) == 2

    trained.delete()
    fitted_svd.delete()
