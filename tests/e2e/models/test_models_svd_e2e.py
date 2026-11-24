"""SVDs served as svd/<id> models."""

from typing import Any, cast

import pytest

from openprotein import OpenProtein
from openprotein.errors import APIError
from openprotein.jobs import JobStatus, _dispatch
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(3.0)
MODEL_ID = "esm2_t6_8M_UR50D"


@pytest.mark.e2e
def test_svd_fits_projects_and_is_managed_as_a_model(
    session: OpenProtein, mutated_sequences
):
    """A 16-component fit over 8 sequences has rank 8."""
    sequences = mutated_sequences(num_sequences=8, mutation_rate=0.05)
    future = session.models.SVD(n_components=16, seed=11).fit(
        model=MODEL_ID, sequences=sequences, reduction="MEAN"
    )
    if not _dispatch.is_models_job(future.job):
        pytest.skip("the fit isn't a models API job")
    svd = future.wait(timeout=TIMEOUT)
    assert svd.id == f"svd/{future.id}"
    assert future.outputs == ["sequences"]
    assert svd.n_components == 8

    card = session.models.get_model(svd.id).metadata
    assert card.dimension == 8
    assert [method.path for method in card.methods] == ["batch/embeddings"]
    assert svd.id in [listed.id for listed in session.models.SVD.list()]

    projected = svd.embed(sequences[:2])
    values = cast(list[Any], projected.wait(timeout=TIMEOUT))
    assert projected.outputs == ["embeddings"]
    assert len(values) == 2 and all(value.shape == (8,) for value in values)
    assert projected.status is JobStatus.SUCCESS

    svd.delete()
    assert svd.id not in [listed.id for listed in session.models.SVD.list()]


@pytest.mark.e2e
def test_an_svd_fit_outside_the_models_api_reads_back_and_projects(
    session: OpenProtein, mutated_sequences
):
    """The svd reads back once its fit succeeds; projections come from their job."""
    sequences = mutated_sequences(num_sequences=8, mutation_rate=0.05)
    future = session.models.SVD(n_components=4).fit(
        model=MODEL_ID, sequences=sequences, reduction="MEAN"
    )
    if _dispatch.is_models_job(future.job):
        pytest.skip("the fit is a models API job")
    svd = future.wait(timeout=TIMEOUT)
    assert (svd.id, svd.n_components) == (f"svd/{future.id}", 4)
    assert svd.id in [listed.id for listed in session.models.SVD.list()]

    projected = svd.embed(sequences[:2])
    projected.wait_until_done(timeout=TIMEOUT)
    assert projected.status is JobStatus.SUCCESS
    with pytest.raises(APIError, match="session.jobs.get"):
        projected.get()
    values = session.jobs.get(projected.id).wait(timeout=TIMEOUT)
    assert len(values) == 2

    svd.delete()
    assert svd.id not in [listed.id for listed in session.models.SVD.list()]
