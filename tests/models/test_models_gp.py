"""GP fits, loads and lists on the models API."""

from typing import Any

import numpy as np
import pytest

from openprotein.data import AssayDataset
from openprotein.errors import InvalidParameterError
from openprotein.jobs.futures import (
    CrossvalidateFuture,
    CrossvalidateResults,
    GPFitFuture,
)
from openprotein.models import (
    GPModel,
    LinearKernel,
    ModelsAPI,
    PeriodicKernel,
    RBFKernel,
)
from openprotein.models.gp import _fit


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


class TestGPs:
    def test_fit_submits_the_training(self, models, models_session):
        models_session.post_routes["v1/models/gp"] = _job("p-1")

        future = _fit(
            models.session,
            "svd/s-1",
            "assay-1",
            ["fitness"],
            kernel="periodic",
            period=2.0,
            name="gp",
        )

        assert isinstance(future, GPFitFuture)
        assert models_session.submitted == (
            "v1/models/gp",
            {
                "features": {"model_id": "svd/s-1"},
                "kernel": {"type": "periodic", "period": 2.0},
                "training_dataset": {"assay_id": "assay-1", "properties": ["fitness"]},
                "name": "gp",
            },
        )

    def test_wait_reads_the_trained_gp(self, models, models_session, model_card):
        models_session.post_routes["v1/models/gp"] = _job("p-1", "SUCCESS")
        models_session.get_routes["v1/jobs/p-1"] = _job("p-1", "SUCCESS")
        models_session.get_routes["v1/models/gp/p-1"] = model_card("gp/p-1")

        gp = _fit(
            models.session, "esm2", "assay-1", ["fitness"], reduction="MEAN"
        ).wait(interval=0)

        assert isinstance(gp, GPModel)
        assert gp.id == "gp/p-1"

    def test_list_pages_and_delete(self, models, models_session, model_card):
        models_session.get_routes["v1/models"] = [model_card("gp/p-1")]

        listed = models.GP.list(limit=2)
        models.get_model("gp/p-1").delete()

        assert [type(p) for p in listed] == [GPModel]
        models_session.session.get.assert_called_once_with(
            "v1/models",
            params={
                "scope": "mine",
                "type": "gp",
                "verbose": "true",
                "limit": 2,
                "offset": 0,
            },
        )
        models_session.session.delete.assert_called_once_with("v1/models/gp/p-1")

    def test_predict_and_single_site_submit_predictions(self, models, models_session):
        gp = models.get_model("gp/p-1")

        gp.predict(["MKT"])
        gp.single_site("MKT")

        assert models_session.submissions == [
            ("v1/models/gp/p-1/batch/predictions", {"sequences": ["MKT"]}),
            ("v1/models/gp/p-1/single-site/predictions", {"base_sequence": "MKT"}),
        ]

    def test_crossvalidate_reads_its_rows_as_cv_results(self, models, models_session):
        gp = models.get_model("gp/p-1")
        assert isinstance(gp, GPModel)
        _serve_crossvalidation(models_session, folds=[1, -1])

        future = gp.crossvalidate(n_splits=3)

        assert isinstance(future, CrossvalidateFuture)
        assert models_session.submitted == (
            "v1/models/gp/p-1/crossvalidate",
            {"n_splits": 3},
        )
        results = future.wait(interval=0)
        assert isinstance(results, CrossvalidateResults)
        assert results.row_index.tolist() == [0, 1]
        assert results.sequence.tolist() == ["MKT", "AC"]
        assert results.measurement_name.tolist() == ["fitness", "fitness"]
        assert results.fold is not None and results.fold.tolist() == [1, -1]
        assert results.y.tolist() == [1.0, 2.0]
        assert results.y_mu[0] == 0.9 and np.isnan(results.y_mu[1])
        assert results.y_var[0] == 0.1 and np.isnan(results.y_var[1])

    def test_a_run_without_folds_reports_none(self, models, models_session):
        gp = models.get_model("gp/p-1")
        assert isinstance(gp, GPModel)
        _serve_crossvalidation(models_session, folds=[None, None])

        assert gp.crossvalidate().wait(interval=0).fold is None

    def test_crossvalidate_sends_folds_checked_against_the_training_assay(
        self, models, models_session, model_card, monkeypatch
    ):
        _serve_training_assay(models_session, model_card, monkeypatch, rows=4)
        _serve_crossvalidation(models_session, folds=[0, 1])
        gp = models.get_model("gp/p-1")
        assert isinstance(gp, GPModel)

        gp.crossvalidate(folds=np.array([0, 1, -1, 1]))

        assert models_session.submitted == (
            "v1/models/gp/p-1/crossvalidate",
            {"folds": [0, 1, -1, 1]},
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_splits": 3, "folds": [0, 1, 0, 1]}, "mutually exclusive"),
            ({"folds": [0, 1, 0]}, r"\(4\), got 3"),
            ({"folds": [0, 0, 0, 0]}, "between 2 and 10 folds"),
        ],
    )
    def test_crossvalidate_refuses_bad_folds_before_submitting(
        self, models, models_session, model_card, monkeypatch, kwargs, message
    ):
        _serve_training_assay(models_session, model_card, monkeypatch, rows=4)
        gp = models.get_model("gp/p-1")
        assert isinstance(gp, GPModel)

        with pytest.raises(InvalidParameterError, match=message):
            gp.crossvalidate(**kwargs)
        models_session.session.post.assert_not_called()

    def test_reads_of_the_training_record(self, models, models_session):
        gp = models.get_model("gp/p-1")
        assert isinstance(gp, GPModel)
        models_session.get_routes |= {
            "v1/models/gp/p-1/evaluations": [{"job_id": "cv-1"}],
            "v1/models/gp/p-1/training": [{"measurement_name": "fitness"}],
        }

        assert gp.evaluations() == [{"job_id": "cv-1"}]
        assert gp.training() == [{"measurement_name": "fitness"}]

    @pytest.mark.parametrize(
        "kernel,served,expected",
        [
            ("LINEAR", {"lengthscale": [None]}, LinearKernel(outputscale=2.0)),
            (
                "PERIODIC",
                {"lengthscale": [0.8], "period": [3.0]},
                PeriodicKernel(outputscale=2.0, lengthscale=0.8, period=3.0),
            ),
        ],
    )
    def test_trained_hyperparameters(
        self, models, models_session, model_card, kernel, served, expected
    ):
        _serve_trained(models_session, model_card, kernel, served)
        models_session.session.timeout = 7
        gp = models.get_model("gp/p-1")

        assert gp.kernel == expected
        assert type(gp.kernel) is type(expected)
        assert (gp.mean, gp.noise) == (0.5, 0.1)
        reads = [
            (c.args[0], c.kwargs.get("timeout"))
            for c in models_session.session.get.call_args_list
        ]
        assert reads == [
            ("v1/models/gp/p-1/hyperparameters", 7),
            ("v1/models/gp/p-1", 7),
        ]

    @pytest.mark.parametrize("lengthscale", [[None], []])
    def test_a_rejected_kernel_is_read_again(
        self, models, models_session, model_card, lengthscale
    ):
        _serve_trained(models_session, model_card, "RBF", {"lengthscale": lengthscale})
        gp = models.get_model("gp/p-1")

        assert gp.mean == 0.5
        with pytest.raises(ValueError):
            _ = gp.kernel
        _serve_trained(models_session, model_card, "RBF", {"lengthscale": [0.8]})

        assert gp.kernel == RBFKernel(outputscale=2.0, lengthscale=0.8)
        reads = [
            c
            for c in models_session.session.get.call_args_list
            if c.args[0] == "v1/models/gp/p-1/hyperparameters"
        ]
        assert len(reads) == 2

    def test_ipython_completes_the_attributes_of_the_trained_kernel(
        self, models, models_session, model_card
    ):
        pytest.importorskip("IPython")
        from IPython.core.completer import provisionalcompleter
        from IPython.core.interactiveshell import InteractiveShell

        _serve_trained(
            models_session,
            model_card,
            "PERIODIC",
            {"lengthscale": [0.8], "period": [3.0]},
        )
        shell = InteractiveShell.instance()
        shell.user_ns["trained"] = models.get_model("gp/p-1")
        text = "trained.kernel."
        with provisionalcompleter():
            completions = {
                c.text.rsplit(".", 1)[-1]
                for c in shell.Completer.completions(text, len(text))
            }

        assert completions & {"lengthscale", "outputscale", "period", "alpha"} == {
            "lengthscale",
            "outputscale",
            "period",
        }


def _serve_trained(models_session, model_card, kernel: str, served: dict) -> None:
    models_session.get_routes |= {
        "v1/models/gp/p-1": model_card("gp/p-1")
        | {"spec": {"kernel": {"type": kernel}}},
        "v1/models/gp/p-1/hyperparameters": {
            "fitness": {"mean": 0.5, "noise": 0.1, "outputscale": [2.0], **served}
        },
    }


def _serve_crossvalidation(models_session, folds: list[int | None]) -> None:
    rows = [
        {
            "row_index": 0,
            "sequence": "MKT",
            "property": "fitness",
            "y": 1.0,
            "mean": 0.9,
            "variance": 0.1,
            "fold": folds[0],
        },
        {
            "row_index": 1,
            "sequence": "AC",
            "property": "fitness",
            "y": 2.0,
            "mean": None,
            "variance": None,
            "fold": folds[1],
        },
    ]
    models_session.post_routes["v1/models/gp/p-1/crossvalidate"] = _job(
        "cv-1", "SUCCESS"
    )
    models_session.get_routes["v1/jobs/cv-1"] = _job("cv-1", "SUCCESS")
    models_session.get_routes["v1/models/gp/p-1/evaluations/cv-1"] = {
        "job_id": "cv-1",
        "rows": rows,
    }


def _serve_training_assay(models_session, model_card, monkeypatch, rows: int) -> None:
    models_session.get_routes["v1/models/gp/p-1"] = model_card("gp/p-1") | {
        "spec": {
            "kernel": {"type": "rbf"},
            "training_dataset": {"assay_id": "assay-1", "properties": ["fitness"]},
        }
    }
    sizes = {"assay-1": rows}
    monkeypatch.setattr(
        AssayDataset,
        "_for",
        classmethod(lambda cls, session: lambda id: [None] * sizes[id]),
    )
