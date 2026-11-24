"""The model a fit takes, as an id, a handle or an SVD."""

from typing import Any

import pytest

from openprotein.models import (
    ModelsAPI,
    gp,
    svd,
    umap,
)
from openprotein.models.svd import SVDModel


def _job(job_id: str, status: str = "PENDING") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_type": "/models",
        "status": status,
        "created_date": "2026-10-21T08:00:00Z",
    }


@pytest.fixture
def models(models_session) -> ModelsAPI:
    return ModelsAPI(models_session.session)


SEQUENCES = ["MKT", "AC", "DE", "FG"]


class TestModelArgument:
    """Each fit takes its model as an id, a get_model handle or an SVD."""

    @pytest.mark.parametrize(
        "model, model_id",
        [
            ("esm2", "esm2"),
            (lambda m: m.get_model("esm2"), "esm2"),
            (lambda m: SVDModel._from_record(m.session, {"id": "s-1"}), "svd/s-1"),
        ],
        ids=["id", "handle", "svd"],
    )
    def test_umap_and_gp_fits_resolve_the_model(
        self, models, models_session, model, model_id
    ):
        model = model(models) if callable(model) else model
        models_session.post_routes["v1/models/umap"] = _job("fit-1")
        models_session.post_routes["v1/models/gp"] = _job("p-1")

        umap._fit(models.session, model, SEQUENCES)
        gp._fit(models.session, model, "assay-1", ["fitness"])

        umap_body, gp_body = (body for _, body in models_session.submissions)
        assert umap_body["model_id"] == gp_body["features"]["model_id"] == model_id

    @pytest.mark.parametrize(
        "fit",
        [
            lambda m: umap._fit(m.session, 3, assay="assay-1"),
            lambda m: gp._fit(m.session, 3, "assay-1", ["fitness"]),
            lambda m: svd._fit(m.session, 3, assay="assay-1"),
        ],
        ids=["umap", "gp", "svd"],
    )
    def test_a_wrong_model_type_raises_before_any_request(
        self, models, models_session, fit
    ):
        with pytest.raises(TypeError, match="model must be"):
            fit(models)
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()
