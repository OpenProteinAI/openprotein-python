"""Estimators on ``session.models`` and the futures they fit on."""

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data import AssayDataset, DataAPI
from openprotein.embeddings.future import EmbeddingsResultFuture
from openprotein.errors import HTTPError, JobFailedException
from openprotein.jobs import Future, Job
from openprotein.jobs import futures as results
from openprotein.jobs.futures import (
    GPFitFuture,
    SVDFitFuture,
    UMAPFitFuture,
    _Submission,
)
from openprotein.models import (
    GPModel,
    ModelsAPI,
    SVDModel,
    UMAPModel,
    _inputs,
)
from openprotein.models.estimators import GP, SVD, UMAP, _dataset_of, _job_id
from openprotein.svd.models import SVDModel as SVDFit
from tests.models.support import SUBMITTED_JOB

ESM2 = "esm2_t33_650M_UR50D"
SEQUENCES = ["MKTA", "MKTC", "MKTD", "MKTE"]
ASSAY = "6f1c2c1e-0000-4000-a000-0000000000a1"


def _assay(*measurements: str) -> AssayDataset:
    rows = pd.DataFrame({"sequence": SEQUENCES})
    assay = MagicMock(
        spec=AssayDataset,
        id=ASSAY,
        page_size=3,
        measurement_names=list(measurements or ("fitness",)),
    )
    assay.get_slice.side_effect = lambda start, end: rows.iloc[start:end]
    return assay


def _job(job_id: str, status: str = "PENDING") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_type": "/models",
        "status": status,
        "created_date": "2026-10-29T08:00:00Z",
    }


@pytest.fixture
def models(models_session) -> ModelsAPI:
    return ModelsAPI(models_session.session)


class TestEmbedAnAssay:
    def test_it_is_named_by_id_and_stays_on_the_future(self, models, models_session):
        assay = _assay()

        future = models.esm2.embed(assay)

        body = {"assay_id": ASSAY, "reduction": "MEAN"}
        assert models_session.submitted == (
            f"v1/models/{ESM2}/batch/embeddings",
            body,
        )
        assert isinstance(future, results.EmbeddingFuture)
        assert future.dataset is assay
        assert future._submission == _Submission(ESM2, "batch/embeddings", body)
        assay.get_slice.assert_not_called()

    def test_a_projection_sends_its_sequences(self, models, models_session):
        assay = _assay()

        future = models.get_model("svd/s-1").transform(assay)

        assert models_session.submitted[1] == {"sequences": SEQUENCES}
        assert isinstance(future, results.EmbeddingFuture)
        assert future.dataset is assay

    def test_sequences_leave_no_dataset(self, models):
        future = models.esm2.embed(SEQUENCES, reduction=None)

        assert future.dataset is None
        assert future._submission == _Submission(
            ESM2, "batch/embeddings", {"sequences": SEQUENCES, "reduction": None}
        )


class TestFittedSVD:
    def test_get_model_returns_a_fitted_svd(self, models):
        svd = models.get_model("svd/s-1")

        assert isinstance(svd, SVDModel)
        assert svd.id == "svd/s-1"

    def test_a_bare_id_is_prefixed(self, models):
        assert SVDModel(models.session, "s-1").id == "svd/s-1"

    def test_transform_projects_an_assay_and_keeps_it(self, models, models_session):
        assay = _assay()

        future = models.get_model("svd/s-1").transform(assay)

        assert models_session.submitted == (
            "v1/models/svd/s-1/batch/embeddings",
            {"sequences": SEQUENCES},
        )
        assert future.dataset is assay

    def test_transform_of_sequences_keeps_no_dataset(self, models):
        assert models.get_model("svd/s-1").transform(SEQUENCES).dataset is None

    def test_embed_submits_what_transform_submits(self, models, models_session):
        svd = models.get_model("svd/s-1")
        assay = _assay()

        transformed = svd.transform(assay)
        embedded = svd.embed(assay)

        assert models_session.submissions[-1] == models_session.submissions[-2]
        assert isinstance(embedded, Future)
        assert embedded.outputs == transformed.outputs
        assert embedded.dataset is assay

    def test_it_fits_a_umap_and_a_gp_on_its_projections(self, models, models_session):
        svd = models.get_model("svd/s-1")
        assert isinstance(svd, SVDModel)

        umap = svd.fit_umap(sequences=SEQUENCES)
        gp = svd.fit_gp(ASSAY, ["fitness"])

        assert isinstance(umap, UMAPFitFuture)
        assert isinstance(gp, GPFitFuture)
        assert [path for path, _ in models_session.submissions] == [
            "v1/models/umap",
            "v1/models/gp",
        ]

    def test_its_record_is_read_from_its_card(self, models, models_session, model_card):
        models_session.get_routes["v1/models/svd/s-1"] = model_card("svd/s-1") | {
            "svd": {"id": "s-1", "n_components": 2}
        }

        assert SVDModel(models.session, "svd/s-1").n_components == 2


JOB_ID = SUBMITTED_JOB["job_id"]


@pytest.fixture
def distinct_ids(models_session):
    """Give SVD fits and projections ids of their own, apart from their embeddings."""
    models_session.post_routes["v1/models/svd"] = _job("fit-1")
    models_session.post_routes["v1/models/svd/s-1/batch/embeddings"] = _job("proj-1")


def _reloaded(job_id: str, **fields: Any) -> Job:
    return Job.model_validate(_job(job_id, "SUCCESS") | fields)


def _reloaded_embeddings(job_id: str) -> Job:
    return _reloaded(
        job_id,
        job_type=f"/models/{ESM2}/batch/embeddings",
    )


class TestJobId:
    def test_a_fitted_svd_goes_by_its_bare_id(self, models):
        assert _job_id(models.get_model("svd/s-1")) == "s-1"

    def test_an_earlier_api_svd_goes_by_its_id(self):
        assert _job_id(MagicMock(spec=SVDFit, id="s-1")) == "s-1"

    @pytest.mark.parametrize("given, sent", [("e-1", "e-1"), ("svd/s-1", "s-1")])
    def test_an_id_goes_bare(self, given, sent):
        assert _job_id(given) == sent

    @pytest.mark.parametrize(
        "x",
        [
            lambda m: m.esm2.embed(SEQUENCES),
            lambda m: m.get_model("svd/s-1").transform(SEQUENCES),
            lambda m: m.SVD().fit(m.esm2.embed(SEQUENCES)),
            lambda m: m.get_model("umap/u-1").embed(SEQUENCES),
            lambda m: Future(m.session, _reloaded("r-1")),
            lambda m: MagicMock(spec=EmbeddingsResultFuture, id=JOB_ID),
            lambda m: Future.create(
                session=m.session,
                response=_job(JOB_ID, "SUCCESS") | {"job_type": "/svd/embed"},
            ),
        ],
        ids=[
            "embeddings",
            "projection",
            "svd fit",
            "umap embeddings",
            "reloaded job",
            "earlier api embeddings",
            "earlier api projection",
        ],
    )
    def test_a_job_is_named_by_its_id_with_no_read(self, models, models_session, x):
        x = x(models)
        models_session.session.get.reset_mock()

        assert _job_id(x) == x.id
        models_session.session.get.assert_not_called()

    @pytest.mark.parametrize(
        "x, named", [(3, "int"), (_assay(), f"assay {ASSAY}")], ids=["int", "assay"]
    )
    def test_anything_else_is_refused(self, x, named):
        with pytest.raises(TypeError, match=f"{named} isn't an embeddings job"):
            _job_id(x)


class TestDatasetOf:
    def test_an_assay_is_its_own_dataset(self):
        assay = _assay()

        assert _dataset_of(assay) is assay

    def test_a_future_carries_the_assay_it_embedded(self, models):
        assay = _assay()

        assert _dataset_of(models.esm2.embed(assay)) is assay

    @pytest.mark.parametrize(
        "x",
        [
            lambda m: m.esm2.embed(SEQUENCES),
            lambda m: m.get_model("svd/s-1"),
            lambda m: 3,
        ],
        ids=["sequences", "fitted svd", "other"],
    )
    def test_anything_else_has_none(self, models, x):
        assert _dataset_of(x(models)) is None

    def test_a_reloaded_embedding_reads_its_assay_from_its_args(
        self, models, models_session, monkeypatch
    ):
        assay = _assay()
        loaded = []

        def as_dataset(session, assay_id):
            loaded.append(assay_id)
            return assay

        monkeypatch.setattr(_inputs, "as_dataset", as_dataset)
        models_session.get_routes["v1/jobs/e-1/args"] = {"assay_id": ASSAY}
        x = Future(models.session, _reloaded_embeddings("e-1"))

        assert _dataset_of(x) is assay
        assert _dataset_of(x) is assay
        assert loaded == [ASSAY]

    def test_a_reloaded_embedding_of_sequences_has_none(self, models, models_session):
        models_session.get_routes["v1/jobs/e-1/args"] = {"sequences": SEQUENCES}

        assert _dataset_of(Future(models.session, _reloaded_embeddings("e-1"))) is None


class TestAFitTheModelsApiRefuses:
    @pytest.mark.parametrize(
        "fit",
        [
            lambda m, x: m.SVD().fit(x),
            lambda m, x: m.UMAP().fit(x),
            lambda m, x: m.GP().fit(x),
        ],
        ids=["SVD", "UMAP", "GP"],
    )
    def test_its_422_reaches_the_caller(self, models, models_session, fit):
        x = models.esm2.embed(_assay(), reduction=None)
        models_session.session.post.reset_mock()
        models_session.session.post.side_effect = HTTPError(
            MagicMock(status_code=422, text="refused", url="v1/models")
        )

        with pytest.raises(HTTPError) as raised:
            fit(models, x)

        assert raised.value.status_code == 422
        models_session.session.post.assert_called_once()


def _svd_fit_succeeds(models_session, model_card, job_id: str) -> None:
    models_session.post_routes["v1/models/svd"] = _job(job_id, "SUCCESS")
    models_session.get_routes[f"v1/jobs/{job_id}"] = _job(job_id, "SUCCESS")
    models_session.get_routes[f"v1/models/svd/{job_id}"] = model_card(
        f"svd/{job_id}"
    ) | {"svd": {"id": job_id, "n_components": 2}}


class TestSVD:
    def test_fit_names_the_embeddings_job_with_no_read(self, models, models_session):
        embeddings = models.poet2.embed(
            SEQUENCES, prompt="prompt-1", decoder_type="clm"
        )
        models_session.session.get.reset_mock()
        models_session.post_routes["v1/models/svd"] = _job("fit-1")

        future = models.SVD(n_components=2, seed=7).fit(embeddings)

        assert isinstance(future, SVDFitFuture)
        assert models_session.submissions[-1] == (
            "v1/models/svd",
            {
                "embeddings_job_id": embeddings.id,
                "n_components": 2,
                "output": "embeddings",
                "seed": 7,
            },
        )
        models_session.session.get.assert_not_called()

    def test_fit_transform_is_fit_and_carries_the_assay(self, models):
        assay = _assay()
        svd = models.SVD(n_components=2)

        future = svd.fit_transform(models.esm2.embed(assay))

        assert future.dataset is assay

    def test_a_fit_on_a_reloaded_job_reads_its_assay_only_when_asked(
        self, models, models_session, monkeypatch
    ):
        assay = _assay()
        monkeypatch.setattr(_inputs, "as_dataset", lambda session, assay_id: assay)
        models_session.get_routes["v1/jobs/e-1/args"] = {"assay_id": ASSAY}
        x = Future(models.session, _reloaded_embeddings("e-1"))

        future = models.SVD(n_components=2).fit(x)

        models_session.session.get.assert_not_called()
        assert future.dataset is assay

    def test_waiting_with_an_output_reads_that_output_of_the_fit(
        self, models, models_session, model_card
    ):
        _svd_fit_succeeds(models_session, model_card, "fit-1")
        models_session.get_routes["v1/jobs/fit-1"] |= {"outputs": ["embeddings"]}
        models_session.outputs[("fit-1", "embeddings")] = [[0.5, 0.25]]

        future = models.SVD(n_components=2).fit(models.esm2.embed(SEQUENCES))

        assert future.wait(output="embeddings") == future.get(output="embeddings")
        assert isinstance(future.wait(), SVDModel)

    def test_a_fitted_svd_as_x_goes_by_its_id(self, models, models_session):
        models.SVD().fit(models.get_model("svd/s-1"))

        assert models_session.submissions[-1] == (
            "v1/models/svd",
            {"embeddings_job_id": "s-1", "output": "embeddings"},
        )

    def test_fit_on_a_models_embeddings_of_an_assay(self, models, models_session):
        assay = _assay()

        future = models.SVD(n_components=2).fit(
            model=models.esm2, assay=assay, reduction="MEAN"
        )

        assert models_session.submissions[-1] == (
            "v1/models/svd",
            {
                "model_id": ESM2,
                "output": "embeddings",
                "n_components": 2,
                "assay_id": ASSAY,
                "reduction": "MEAN",
            },
        )
        assert future.dataset is assay

    def test_fit_on_a_models_embeddings_of_sequences(self, models, models_session):
        models.SVD().fit(model="prot-seq", sequences=SEQUENCES)

        assert models_session.submissions[-1] == (
            "v1/models/svd",
            {"model_id": "prot-seq", "output": "embeddings", "sequences": SEQUENCES},
        )

    def test_waiting_on_the_fit_returns_the_fitted_svd(
        self, models, models_session, model_card
    ):
        _svd_fit_succeeds(models_session, model_card, "fit-1")

        fitted = models.SVD(n_components=2).fit(models.esm2.embed(SEQUENCES)).wait()

        assert isinstance(fitted, SVDModel)
        assert fitted.id == "svd/fit-1"

    def test_the_fit_future_projects_without_waiting_for_its_fit(
        self, models, models_session
    ):
        models_session.post_routes["v1/models/svd"] = _job("fit-1")
        future = models.SVD(n_components=2).fit(models.esm2.embed(SEQUENCES))
        assay = _assay()

        projected = future.transform(assay)

        assert models_session.submissions[-1] == (
            "v1/models/svd/fit-1/batch/embeddings",
            {"sequences": SEQUENCES},
        )
        assert projected.dataset is assay
        assert projected._path == "batch/embeddings"
        assert projected.outputs == ["embeddings"]
        assert all(
            "fit-1" not in str(call)
            for call in models_session.session.get.call_args_list
        )

    def test_the_svd_fit_future_embed_submits_what_transform_submits(
        self, models, models_session
    ):
        models_session.post_routes["v1/models/svd"] = _job("fit-1")
        future = models.SVD(n_components=2).fit(models.esm2.embed(SEQUENCES))

        transformed = future.transform(SEQUENCES)
        embedded = future.embed(SEQUENCES)

        assert models_session.submissions[-1] == models_session.submissions[-2]
        assert isinstance(embedded, Future)
        assert embedded._path == transformed._path
        assert embedded.outputs == transformed.outputs

    def test_waiting_on_a_failed_fit_raises(self, models, models_session):
        models_session.post_routes["v1/models/svd"] = _job("fit-1", "FAILURE")
        models_session.get_routes["v1/jobs/fit-1"] = _job("fit-1", "FAILURE")
        future = models.SVD(n_components=2).fit(models.esm2.embed(SEQUENCES))

        with pytest.raises(JobFailedException):
            future.wait()


class TestUMAP:
    @pytest.mark.parametrize(
        "x, sent",
        [
            (lambda m: m.esm2.embed(SEQUENCES), lambda x: {"embeddings_job_id": x.id}),
            (
                lambda m: m.get_model("svd/s-1").transform(SEQUENCES),
                lambda x: {"embeddings_job_id": x.id},
            ),
            (
                lambda m: m.SVD(n_components=2).fit(m.esm2.embed(SEQUENCES)),
                lambda x: {"embeddings_job_id": x.id},
            ),
            (
                lambda m: m.get_model("svd/s-1"),
                lambda x: {"embeddings_job_id": "s-1"},
            ),
        ],
        ids=["embeddings", "projection", "svd fit", "fitted svd"],
    )
    def test_fit_transform_names_its_source(
        self, models, models_session, distinct_ids, x, sent
    ):
        x = x(models)

        future = models.UMAP(n_components=2).fit_transform(x)

        assert isinstance(future, UMAPFitFuture)
        assert models_session.submissions[-1] == (
            "v1/models/umap",
            {"n_components": 2, **sent(x)},
        )

    def test_the_assay_stays_on_the_fit(self, models):
        assay = _assay()

        future = models.UMAP().fit(models.esm2.embed(assay))

        assert future.dataset is assay

    def test_fit_on_an_svds_features_of_sequences(self, models, models_session):
        models.UMAP(n_components=2).fit(
            model=models.get_model("svd/s-1"), sequences=SEQUENCES
        )

        assert models_session.submissions[-1] == (
            "v1/models/umap",
            {"model_id": "svd/s-1", "sequences": SEQUENCES, "n_components": 2},
        )

    def test_the_fit_future_maps_without_waiting_for_its_fit(
        self, models, models_session
    ):
        models_session.post_routes["v1/models/umap"] = _job("u-1")
        future = models.UMAP(n_components=2).fit(models.esm2.embed(SEQUENCES))
        assay = _assay()

        mapped = future.transform(assay)

        assert models_session.submissions[-1] == (
            "v1/models/umap/u-1/batch/embeddings",
            {"sequences": SEQUENCES},
        )
        assert mapped.dataset is assay
        assert mapped._path == "batch/embeddings"
        assert mapped.outputs == ["embeddings"]
        assert all(
            "u-1" not in str(call) for call in models_session.session.get.call_args_list
        )

    def test_the_umap_fit_future_embed_submits_what_transform_submits(
        self, models, models_session
    ):
        models_session.post_routes["v1/models/umap"] = _job("u-1")
        future = models.UMAP(n_components=2).fit(models.esm2.embed(SEQUENCES))

        transformed = future.transform(SEQUENCES)
        embedded = future.embed(SEQUENCES)

        assert models_session.submissions[-1] == models_session.submissions[-2]
        assert isinstance(embedded, Future)
        assert embedded._path == transformed._path
        assert embedded.outputs == transformed.outputs

    def test_a_fitted_umap_transform_submits_what_embed_submits(
        self, models, models_session
    ):
        umap = models.get_model("umap/u-1")
        assay = _assay()

        embedded = umap.embed(assay)
        transformed = umap.transform(assay)

        assert models_session.submissions[-1] == models_session.submissions[-2]
        assert isinstance(transformed, Future)
        assert transformed.outputs == embedded.outputs
        assert transformed.dataset is assay

    @pytest.mark.parametrize("by_id", [False, True], ids=["dataset", "id"])
    def test_a_fit_on_an_assay_keeps_the_assay(
        self, models, models_session, monkeypatch, by_id
    ):
        assay = _assay()
        monkeypatch.setattr(_inputs, "as_dataset", lambda session, given: assay)
        models_session.post_routes["v1/models/umap"] = _job("u-1")
        given = ASSAY if by_id else assay

        handle_fit = models.esm2.fit_umap(assay=given, reduction="MEAN")
        estimator_fit = models.UMAP().fit(
            model=models.esm2, assay=given, reduction="MEAN"
        )

        assert handle_fit.dataset is assay
        assert estimator_fit.dataset is assay

    def test_waiting_on_the_fit_returns_the_fitted_umap(
        self, models, models_session, model_card
    ):
        models_session.post_routes["v1/models/umap"] = _job("u-1", "SUCCESS")
        models_session.get_routes["v1/jobs/u-1"] = _job("u-1", "SUCCESS")
        models_session.get_routes["v1/models/umap/u-1"] = model_card("umap/u-1")

        fitted = models.UMAP().fit(models.esm2.embed(SEQUENCES)).wait()

        assert isinstance(fitted, UMAPModel)
        assert fitted.id == "umap/u-1"


class TestGP:
    def test_fit_trains_on_the_embeddings_job_and_its_assay(
        self, models, models_session
    ):
        x = models.esm2.embed(_assay())

        future = models.GP(kernel="rbf", name="gp", description="d").fit(x)

        assert isinstance(future, GPFitFuture)
        assert models_session.submissions[-1] == (
            "v1/models/gp",
            {
                "embeddings_job_id": x.id,
                "kernel": {"type": "rbf"},
                "training_dataset": {"assay_id": ASSAY, "properties": ["fitness"]},
                "name": "gp",
                "description": "d",
            },
        )

    @pytest.mark.parametrize(
        "kernel, given",
        [("periodic", {"period": 2.0}), ("rational_quadratic", {"alpha": 0.5})],
    )
    def test_a_kernel_param_is_sent_with_its_kernel(
        self, models, models_session, kernel, given
    ):
        models.GP(kernel=kernel, **given).fit(models.esm2.embed(_assay()))

        assert models_session.submissions[-1][1]["kernel"] == {"type": kernel, **given}

    @pytest.mark.parametrize(
        "kernel, given", [("rbf", {"period": 2.0}), ("periodic", {"alpha": 0.5})]
    )
    def test_a_kernel_param_is_sent_whatever_the_kernel(
        self, models, models_session, kernel, given
    ):
        models.GP(kernel=kernel, **given).fit(models.esm2.embed(_assay()))

        assert models_session.submissions[-1][1]["kernel"] == {
            "type": kernel,
            **given,
        }

    @pytest.mark.parametrize(
        "x",
        [
            lambda m: m.get_model("svd/s-1").transform(_assay()),
            lambda m: m.SVD(n_components=2).fit(m.esm2.embed(_assay())),
        ],
        ids=["projection", "svd fit"],
    )
    def test_a_job_is_named_by_id_with_no_wait(
        self, models, models_session, distinct_ids, x
    ):
        x = x(models)
        assert x.id != JOB_ID

        models.GP().fit(x)

        body = models_session.submissions[-1][1]
        assert body["embeddings_job_id"] == x.id
        assert "features" not in body
        reads = [c.args[0] for c in models_session.session.get.call_args_list]
        assert f"v1/jobs/{x.id}" not in reads

    def test_a_fitted_svd_as_model_featurizes_a_named_assay(
        self, models, models_session
    ):
        models.GP().fit(
            model=models.get_model("svd/s-1"), property="fitness", assay=ASSAY
        )

        assert models_session.submissions[-1][1] == {
            "features": {"model_id": "svd/s-1"},
            "kernel": {"type": "rbf"},
            "training_dataset": {"assay_id": ASSAY, "properties": ["fitness"]},
        }

    def test_a_fitted_svd_as_x_trains_on_its_embeddings(self, models, models_session):
        models.GP().fit(models.get_model("svd/s-1"), property="fitness", assay=ASSAY)

        body = models_session.submissions[-1][1]
        assert body["embeddings_job_id"] == "s-1"
        assert "features" not in body

    def test_a_system_model_featurizes_with_its_embedding_params(
        self, models, models_session
    ):
        models.GP().fit(
            model=models.esm2, assay=ASSAY, property="fitness", reduction="MEAN"
        )

        assert models_session.submissions[-1][1]["features"] == {
            "model_id": ESM2,
            "reduction": "MEAN",
        }

    def test_a_lone_measurement_is_the_property(self, models, models_session):
        models.GP().fit(models.esm2.embed(_assay("stability")))

        training = models_session.submissions[-1][1]["training_dataset"]
        assert training["properties"] == ["stability"]

    def test_several_measurements_need_a_property(self, models, models_session):
        x = models.esm2.embed(_assay("fitness", "stability"))
        models_session.session.get.reset_mock()
        models_session.session.post.reset_mock()

        with pytest.raises(ValueError, match="fitness, stability"):
            models.GP().fit(x)
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()

    def test_embeddings_of_sequences_need_an_assay(self, models, models_session):
        x = models.esm2.embed(SEQUENCES)
        models_session.session.get.reset_mock()
        models_session.session.post.reset_mock()

        with pytest.raises(ValueError, match="pass assay"):
            models.GP().fit(x, property="fitness")
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()

    def test_an_assay_named_with_a_job_is_the_one_trained_on(
        self, models, models_session
    ):
        x = models.esm2.embed(_assay())

        models.GP().fit(x, property="fitness", assay="assay-2")

        body = models_session.submissions[-1][1]
        assert body["training_dataset"] == {
            "assay_id": "assay-2",
            "properties": ["fitness"],
        }

    def test_an_svd_fit_of_an_assay_trains_on_that_assay(self, models, models_session):
        assay = _assay()
        fit = models.esm2.fit_svd(assay=assay, n_components=2, reduction="MEAN")

        models.GP().fit(fit)

        assert fit.dataset is assay
        assert models_session.submissions[-1][1]["training_dataset"] == {
            "assay_id": ASSAY,
            "properties": ["fitness"],
        }

    def test_a_submitted_fit_keeps_the_assay_it_trained_on(self, models):
        assay = _assay()

        assert models.GP().fit(assay=assay).dataset is assay

    def test_a_submitted_fit_on_a_projection_keeps_its_assay(self, models):
        assay = _assay()

        assert models.GP().fit(models.esm2.embed(assay)).dataset is assay

    def test_a_fit_from_a_model_handle_keeps_the_assay_it_trained_on(
        self, models, monkeypatch
    ):
        assay = _assay()

        def no_lookup(self, *args, **kwargs):
            raise AssertionError("the assay is not looked up")

        monkeypatch.setattr(DataAPI, "get", no_lookup)

        future = models.esm2.fit_gp(assay, ["fitness"], reduction="MEAN")

        assert future.dataset is assay

    def test_a_reloaded_fit_reads_its_assay_from_the_training_dataset(
        self, models, models_session, monkeypatch
    ):
        assay = _assay()

        def as_dataset(session, assay_id):
            assert assay_id == ASSAY
            return assay

        monkeypatch.setattr(_inputs, "as_dataset", as_dataset)
        models_session.get_routes["v1/jobs/gp-1/args"] = {
            "training_dataset": {"assay_id": ASSAY, "properties": ["fitness"]}
        }
        job = Job.create(
            SUBMITTED_JOB | {"job_id": "gp-1", "job_type": "/models/gp/fit"}
        )

        assert GPFitFuture(models.session, job).dataset is assay

    def test_an_assay_trains_on_empty_features(self, models, models_session):
        models.GP().fit(assay=_assay())

        body = models_session.submitted[1]
        assert body["features"] == {}
        assert body["training_dataset"] == {
            "assay_id": ASSAY,
            "properties": ["fitness"],
        }

    def test_an_assay_as_x_raises(self, models, models_session):
        with pytest.raises(TypeError, match="isn't an embeddings job"):
            models.GP().fit(_assay())
        models_session.session.post.assert_not_called()

    def test_a_reloaded_legacy_svd_is_named_by_id_with_no_read(
        self, models, models_session
    ):
        models.GP().fit(
            MagicMock(spec=SVDFit, id="s-1"), property="fitness", assay=ASSAY
        )

        assert models_session.submitted[1]["embeddings_job_id"] == "s-1"
        models_session.session.get.assert_not_called()

    def test_a_reloaded_embedding_trains_on_the_assay_it_embedded(
        self, models, models_session, monkeypatch
    ):
        monkeypatch.setattr(_inputs, "as_dataset", lambda session, assay_id: _assay())
        models_session.get_routes["v1/jobs/e-1/args"] = {"assay_id": ASSAY}
        x = Future(models.session, _reloaded_embeddings("e-1"))

        models.GP().fit(x)

        assert models_session.submitted[1]["training_dataset"] == {
            "assay_id": ASSAY,
            "properties": ["fitness"],
        }

    def test_a_reloaded_embedding_is_named_with_no_read(self, models, models_session):
        x = Future(models.session, _reloaded_embeddings("e-1"))

        models.GP().fit(x, property="fitness", assay=ASSAY)

        assert models_session.submitted[1]["embeddings_job_id"] == "e-1"
        models_session.session.get.assert_not_called()

    def test_waiting_on_the_fit_returns_the_trained_gp(
        self, models, models_session, model_card
    ):
        models_session.post_routes["v1/models/gp"] = _job("p-1", "SUCCESS")
        models_session.get_routes["v1/jobs/p-1"] = _job("p-1", "SUCCESS")
        models_session.get_routes["v1/models/gp/p-1"] = model_card("gp/p-1")

        trained = models.GP().fit(models.esm2.embed(_assay())).wait()

        assert isinstance(trained, GPModel)
        assert trained.id == "gp/p-1"


class TestAccessors:
    @pytest.mark.parametrize("name", ["SVD", "UMAP", "GP"])
    def test_each_is_its_estimator_class_bound_to_the_session(
        self, models, models_session, name
    ):
        bound = getattr(models, name)
        other = ModelsAPI(MagicMock())

        assert bound.__name__ == name
        assert issubclass(bound, {"SVD": SVD, "UMAP": UMAP, "GP": GP}[name])
        assert getattr(models, name) is bound
        assert bound.session is models_session.session
        assert getattr(other, name).session is other.session

    @pytest.mark.parametrize(
        "call",
        [lambda: GP(), lambda: SVD(256), lambda: UMAP.list()],
        ids=["gp", "svd", "umap list"],
    )
    def test_an_estimator_class_needs_a_session(self, call):
        with pytest.raises(TypeError, match="use session.models"):
            call()

    def test_a_call_constructs_an_estimator(self, models):
        assert isinstance(models.GP(kernel="matern52"), GP)
        assert models.SVD(256).n_components == 256
        assert models.UMAP().n_components is None

    @pytest.mark.parametrize("model_id", ["p-1", "gp/p-1"])
    def test_gp_by_id_reads_its_card(self, models, models_session, model_id):
        gp = models.GP(id=model_id)

        assert isinstance(gp, GPModel)
        assert gp.id == "gp/p-1"
        models_session.session.get.assert_called_once_with("v1/models/gp/p-1")

    @pytest.mark.parametrize("model_id", ["s-1", "svd/s-1"])
    def test_svd_by_id_reads_its_card(
        self, models, models_session, model_card, model_id
    ):
        models_session.get_routes["v1/models/svd/s-1"] = model_card("svd/s-1") | {
            "svd": {"id": "s-1"}
        }

        svd = models.SVD(id=model_id)

        assert isinstance(svd, SVDModel)
        assert svd.id == "svd/s-1"

    def test_umap_by_id_reads_its_card(self, models):
        umap = models.UMAP(id="u-1")

        assert isinstance(umap, UMAPModel)
        assert umap.id == "umap/u-1"

    @pytest.mark.parametrize(
        "kind, family", [("GP", "gp"), ("SVD", "svd"), ("UMAP", "umap")]
    )
    def test_list_follows_every_page(
        self, models, models_session, model_card, monkeypatch, kind, family
    ):
        monkeypatch.setattr("openprotein.models._api._LIST_PAGE", 2)
        cards = [model_card(f"{family}/x-{i}") for i in range(3)]
        if family == "svd":
            cards = [card | {"svd": {"id": f"x-{i}"}} for i, card in enumerate(cards)]
        pages = iter([cards[:2], cards[2:]])
        models_session.session.get.side_effect = lambda *a, **k: MagicMock(
            json=MagicMock(return_value=next(pages))
        )

        listed = getattr(models, kind).list()

        assert [m.id for m in listed] == [f"{family}/x-{i}" for i in range(3)]
        assert [
            (call.kwargs["params"]["limit"], call.kwargs["params"]["offset"])
            for call in models_session.session.get.call_args_list
        ] == [(2, 0), (2, 2)]

    @pytest.mark.parametrize(
        "kind, family", [("GP", "gp"), ("SVD", "svd"), ("UMAP", "umap")]
    )
    def test_list_reads_up_to_the_limit_from_the_offset(
        self, models, models_session, model_card, kind, family
    ):
        listed = model_card(f"{family}/x-1")
        if family == "svd":
            listed |= {"svd": {"id": "x-1"}}
        models_session.get_routes["v1/models"] = [listed]

        page = getattr(models, kind).list(limit=2, offset=4)

        assert [m.id for m in page] == [f"{family}/x-1"]
        models_session.session.get.assert_called_once_with(
            "v1/models",
            params={
                "scope": "mine",
                "type": family,
                "verbose": "true",
                "limit": 2,
                "offset": 4,
            },
        )

    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.GP(id="p-1", kernel="rbf"),
            lambda m: m.SVD(256, id="s-1"),
            lambda m: m.UMAP(id="u-1", n_neighbors=5),
            lambda m: m.GP(id="p-1", alpha=None),
        ],
        ids=["gp", "svd", "umap", "explicit none"],
    )
    def test_id_with_hyperparameters_raises(self, models, models_session, call):
        with pytest.raises(TypeError, match="pass id alone"):
            call(models)
        models_session.session.get.assert_not_called()


class TestPipelines:
    def test_an_svd_then_a_gp_on_an_embedded_assay(self, models, models_session):
        models_session.post_routes["v1/models/svd"] = _job("fit-1")
        dataset = _assay()

        embeddings = models.esm2.embed(dataset)
        svd = models.SVD(256)
        projected = svd.fit_transform(embeddings)
        gp = models.GP(kernel="rbf")
        gp.fit(projected, property="fitness")

        assert models_session.submissions[1:] == [
            (
                "v1/models/svd",
                {
                    "embeddings_job_id": embeddings.id,
                    "n_components": 256,
                    "output": "embeddings",
                },
            ),
            (
                "v1/models/gp",
                {
                    "embeddings_job_id": projected.id,
                    "kernel": {"type": "rbf"},
                    "training_dataset": {
                        "assay_id": ASSAY,
                        "properties": ["fitness"],
                    },
                },
            ),
        ]

    def test_a_gp_on_an_assay_projected_through_an_svd_of_other_sequences(
        self, models, models_session, model_card
    ):
        _svd_fit_succeeds(models_session, model_card, "fit-1")
        models_session.post_routes["v1/models/svd/fit-1/batch/embeddings"] = _job(
            "proj-2"
        )
        dataset = _assay()

        embeddings = models.esm2.embed(["MKTW", "MKTY", "MKTV"])
        svd = models.SVD(256).fit_transform(embeddings)
        dataset_embeddings = svd.transform(dataset)
        gp = models.GP(kernel="rbf")
        gp.fit(dataset_embeddings)

        assert models_session.submissions[-2:] == [
            ("v1/models/svd/fit-1/batch/embeddings", {"sequences": SEQUENCES}),
            (
                "v1/models/gp",
                {
                    "embeddings_job_id": dataset_embeddings.id,
                    "kernel": {"type": "rbf"},
                    "training_dataset": {
                        "assay_id": ASSAY,
                        "properties": ["fitness"],
                    },
                },
            ),
        ]

    def test_a_one_hot_gp_then_loading_and_listing(
        self, models, models_session, model_card
    ):
        models_session.get_routes["v1/models"] = [model_card("gp/p-1")]

        models.GP(kernel="rbf").fit(assay=_assay())
        gp = models.GP(id="p-1")
        gps = models.GP.list()

        assert models_session.submitted[1]["features"] == {}
        assert gp.id == "gp/p-1"
        assert [listed.id for listed in gps] == ["gp/p-1"]


def test_the_fitted_collections_are_gone(models):
    import openprotein.models as package

    assert not any(hasattr(models, name) for name in ("svds", "umaps", "gps"))
    assert not any(hasattr(package, name) for name in ("GPsAPI", "SVDsAPI", "UMAPsAPI"))


@pytest.mark.parametrize(
    "fit, unset",
    [
        (lambda m: m.SVD().fit(m.esm2.embed(SEQUENCES)), {"n_components"}),
        (
            lambda m: m.UMAP().fit(m.esm2.embed(SEQUENCES)),
            {"n_components", "n_neighbors", "min_dist"},
        ),
        (lambda m: m.esm2.fit_svd(SEQUENCES, reduction="MEAN"), {"n_components"}),
        (
            lambda m: m.esm2.fit_umap(SEQUENCES),
            {"n_components", "n_neighbors", "min_dist"},
        ),
    ],
    ids=["SVD", "UMAP", "fit_svd", "fit_umap"],
)
def test_a_fit_leaves_unset_hyperparameters_to_the_server(
    models, models_session, fit, unset
):
    fit(models)

    assert not unset & set(models_session.submissions[-1][1])


def test_a_fit_sends_the_hyperparameters_given(models, models_session):
    models.UMAP(n_neighbors=5).fit(models.esm2.embed(SEQUENCES))

    body = models_session.submissions[-1][1]
    assert body["n_neighbors"] == 5
    assert not {"n_components", "min_dist"} & set(body)


@pytest.mark.parametrize(
    "fit, error, match",
    [
        (lambda m: m.SVD().fit(), TypeError, "pass X"),
        (
            lambda m: m.SVD().fit(m.esm2.embed(SEQUENCES), model=m.esm2),
            TypeError,
            "pass X",
        ),
        (
            lambda m: m.UMAP().fit(m.esm2.embed(SEQUENCES), assay=ASSAY),
            TypeError,
            "go with model=",
        ),
        (
            lambda m: m.SVD().fit(m.esm2.embed(SEQUENCES), reduction="MEAN"),
            TypeError,
            "go with model=",
        ),
        (lambda m: m.SVD().fit(model=m.esm2), ValueError, "one of sequences or assay"),
        (
            lambda m: m.UMAP().fit(model=m.esm2, sequences=SEQUENCES, assay=ASSAY),
            ValueError,
            "one of sequences or assay",
        ),
        (
            lambda m: m.GP().fit(m.esm2.embed(_assay()), model=m.esm2),
            TypeError,
            "not both",
        ),
        (
            lambda m: m.GP().fit(m.esm2.embed(_assay()), reduction="MEAN"),
            TypeError,
            "go with model=",
        ),
        (lambda m: m.GP().fit(model=m.esm2), ValueError, "pass X, or assay"),
    ],
    ids=[
        "neither",
        "both",
        "x with assay",
        "x with params",
        "model alone",
        "model with both inputs",
        "gp both",
        "gp x with params",
        "gp model without assay",
    ],
)
def test_a_fit_with_the_wrong_sources_raises_before_any_request(
    models, models_session, fit, error, match
):
    models_session.session.post.reset_mock()

    with pytest.raises(error, match=match):
        fit(models)

    assert not [
        c
        for c in models_session.session.post.call_args_list
        if c.args and str(c.args[0]).rstrip("/").endswith(("svd", "umap", "gp"))
    ]


def _missing_card(models_session):
    served = models_session.session.get.side_effect

    def get(path: str, **kwargs: Any):
        if path == "v1/models/no-such-model":
            raise HTTPError(MagicMock(status_code=404, text="not found"))
        return served(path, **kwargs)

    return get


class TestFitFeatureModel:
    """``model=`` is the only feature-model selector of a fit."""

    @pytest.mark.parametrize("estimator", ["SVD", "UMAP", "GP"])
    def test_model_id_in_embedding_params_is_refused_before_any_request(
        self, models, models_session, estimator
    ):
        fit = getattr(models, estimator)().fit
        models_session.session.get.reset_mock()
        models_session.session.post.reset_mock()

        with pytest.raises(TypeError, match="pass the model as model="):
            fit(model=models.esm2, assay=ASSAY, property="fitness", model_id="prot-seq")

        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()

    def test_a_model_handle_features_the_gp_as_is(self, models, models_session):
        models.GP().fit(
            model=models.esm2, assay=ASSAY, property="fitness", reduction="MEAN"
        )

        assert models_session.submissions[-1][1]["features"]["model_id"] == ESM2

    def test_a_model_id_is_resolved_through_the_model_lookup(
        self, models, models_session
    ):
        models.GP().fit(model=ESM2, assay=ASSAY, property="fitness", reduction="MEAN")

        assert models_session.submissions[-1][1]["features"] == {
            "model_id": ESM2,
            "reduction": "MEAN",
        }

    @pytest.mark.parametrize("estimator", ["SVD", "UMAP", "GP"])
    def test_an_unknown_model_id_is_refused(self, models, models_session, estimator):
        models_session.session.get.side_effect = _missing_card(models_session)
        models_session.session.post.reset_mock()
        kwargs: dict[str, Any] = (
            {"assay": ASSAY, "property": "fitness"}
            if estimator == "GP"
            else {"sequences": SEQUENCES}
        )

        with pytest.raises(ValueError):
            getattr(models, estimator)().fit(model="no-such-model", **kwargs)

        models_session.session.post.assert_not_called()
