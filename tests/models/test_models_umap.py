"""UMAP fits, loads and lists on the models API."""

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data import AssayDataset
from openprotein.errors import APIError, JobFailedException
from openprotein.jobs.futures import UMAPFitFuture
from openprotein.models import (
    ModelsAPI,
    UMAPModel,
)
from openprotein.models.umap import _fit


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


class TestUmaps:
    def test_fit_submits_the_fit_with_its_defaults(self, models, models_session):
        models_session.post_routes["v1/models/umap"] = _job("fit-1")

        future = _fit(
            models.session,
            "svd/s-1",
            sequences=["MKT", "AC", "DE", "FG", "HI"],
            n_components=3,
        )

        assert isinstance(future, UMAPFitFuture)
        assert future.id == "fit-1"
        assert models_session.submitted == (
            "v1/models/umap",
            {
                "model_id": "svd/s-1",
                "sequences": ["MKT", "AC", "DE", "FG", "HI"],
                "n_components": 3,
            },
        )

    def test_fit_takes_sequences_or_an_assay(self, models, models_session):
        with pytest.raises(ValueError, match="one of sequences or assay"):
            _fit(models.session, "esm2", reduction="MEAN")
        models_session.session.post.assert_not_called()

    def test_wait_reads_the_fitted_umap(self, models, models_session, model_card):
        models_session.post_routes["v1/models/umap"] = _job("fit-1", "SUCCESS")
        models_session.get_routes["v1/jobs/fit-1"] = _job("fit-1", "SUCCESS")
        models_session.get_routes["v1/models/umap/fit-1"] = model_card("umap/fit-1") | {
            "owner": "alice",
            "spec": {"kind": "umap", "n_components": 2},
        }
        future = _fit(models.session, "esm2", sequences=SEQUENCES, reduction="MEAN")

        umap = future.wait(interval=0)

        assert isinstance(umap, UMAPModel)
        assert umap.id == "umap/fit-1"
        assert umap.spec == {"kind": "umap", "n_components": 2}

    def test_a_failed_fit_raises_on_wait(self, models, models_session):
        models_session.post_routes["v1/models/umap"] = _job("fit-1", "FAILURE")
        models_session.get_routes["v1/jobs/fit-1"] = _job("fit-1", "FAILURE")
        future = _fit(models.session, "esm2", sequences=SEQUENCES, reduction="MEAN")

        with pytest.raises(JobFailedException):
            future.wait(interval=0)

    def test_get_list_and_delete(self, models, models_session, model_card):
        models_session.get_routes["v1/models/umap/fit-1"] = model_card("umap/fit-1")
        models_session.get_routes["v1/models"] = [model_card("umap/fit-1")]

        umap = models.UMAP(id="umap/fit-1")
        listed = models.UMAP.list(limit=5, offset=10)
        models.get_model("umap/fit-1").delete()
        umap.delete()

        assert umap.id == "umap/fit-1"
        assert [m.id for m in listed] == ["umap/fit-1"]
        assert models_session.session.get.call_args_list[1].kwargs == {
            "params": {
                "scope": "mine",
                "type": "umap",
                "verbose": "true",
                "limit": 5,
                "offset": 10,
            }
        }
        assert [c.args for c in models_session.session.delete.call_args_list] == [
            ("v1/models/umap/fit-1",),
            ("v1/models/umap/fit-1",),
        ]

    def test_embed_projects_through_the_umap(self, models, models_session):
        models.get_model("umap/fit-1").embed(["MKT"])

        assert models_session.submitted == (
            "v1/models/umap/fit-1/batch/embeddings",
            {"sequences": ["MKT"]},
        )


class TestUmapAssay:
    def test_an_assay_is_read_page_by_page(self, models, models_session):
        rows = pd.DataFrame({"sequence": SEQUENCES + ["HI"]})
        assay = MagicMock(spec=AssayDataset, page_size=2)
        assay.get_slice.side_effect = lambda start, end: rows.iloc[start:end]
        models_session.post_routes["v1/models/umap"] = _job("fit-1")

        _fit(models.session, "svd/s-1", assay=assay)

        assert models_session.submitted[1]["sequences"] == SEQUENCES + ["HI"]

    def test_a_page_that_fails_raises_before_the_fit(self, models, models_session):
        assay = MagicMock(spec=AssayDataset, page_size=2)
        assay.get_slice.side_effect = [
            pd.DataFrame({"sequence": SEQUENCES[:2]}),
            APIError("page failed"),
        ]

        with pytest.raises(APIError, match="page failed"):
            _fit(models.session, "svd/s-1", assay=assay)
        models_session.session.post.assert_not_called()
