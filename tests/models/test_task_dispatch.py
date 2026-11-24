import logging
from typing import Literal

import pytest

from openprotein.embeddings.future import EmbeddingsResultFuture
from openprotein.embeddings.schemas import EmbeddingsJob
from openprotein.jobs import Future, Job, JobStatus, _dispatch
from openprotein.jobs import futures as results
from openprotein.jobs.futures import (
    GPFitFuture,
    SVDFitFuture,
    UMAPFitFuture,
)
from openprotein.models import _api
from openprotein.models.model import Model
from openprotein.models.schemas import ModelMetadata
from tests.models.support import PARAMS

CASES = [
    ("batch/embeddings", results.EmbeddingFuture),
    ("batch/logits", results.LogitsFuture),
    ("batch/attn", results.AttentionFuture),
    ("batch/loglikelihood", results.LogLikelihoodFuture),
    ("single-site/scores", results.SingleSiteScoreFuture),
    ("indel/scores", results.IndelScoreFuture),
    ("generate/sequences", results.SequenceGenerationFuture),
    ("generate/structures", results.StructureDesignFuture),
    ("batch/structures", results.StructurePredictionFuture),
    ("batch/predictions", results.PredictionFuture),
    ("single-site/predictions", results.SingleSitePredictionFuture),
]


def task_job(path, *, provider="esm2", status="PENDING"):
    return Job.model_validate(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": f"/models/{provider}/{path}",
            "status": status,
            "created_date": "2026-10-21T08:00:00Z",
        }
    )


def test_catalogued_methods_have_specialised_primary_outputs(mock_session):
    for key, method in PARAMS["methods"].items():
        _, first, last = key.rsplit("/", 2)
        path = f"{first}/{last}"
        future = Future.create(mock_session, job=task_job(path))

        assert type(future) is not Future, key
        assert future._primary_output == method["outputs"][0], key


@pytest.mark.parametrize(("path", "expected"), CASES)
@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "SUCCESS"])
def test_identified_tasks_select_their_future(mock_session, path, expected, status):
    future = Future.create(mock_session, job=task_job(path, status=status))
    assert type(future) is expected
    mock_session.get.assert_not_called()


def test_selection_uses_the_method(mock_session):
    future = Future.create(
        mock_session,
        job=task_job("batch/logits", provider="rfdiffusion"),
    )
    assert type(future) is results.LogitsFuture


def test_unknown_models_methods_keep_generic_access(mock_session):
    future = Future.create(mock_session, job=task_job("batch/new-output"))
    assert type(future) is Future
    assert future.status is JobStatus.PENDING


def test_unidentified_records_keep_tracking(mock_session):
    job = Job.create(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": "/future/new-task",
            "status": "PENDING",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    future = Future.create(mock_session, job=job)
    assert type(future).__name__ == "UnknownJobFuture"
    with pytest.raises(NotImplementedError):
        future.get()


def test_typed_embedding_records_keep_their_compatibility_future(mock_session):
    job = EmbeddingsJob(
        job_id="e0b1c3d4-0000-4000-8000-00000000000a",
        status="PENDING",
        created_date="2026-10-21T08:00:00Z",
    )

    future = EmbeddingsResultFuture.create(mock_session, job=job, sequences=["MKT"])

    assert type(future) is EmbeddingsResultFuture
    assert future.sequences == [b"MKT"]
    mock_session.get.assert_not_called()


def test_undeclared_subclasses_do_not_claim_the_parent_schema(mock_session):
    class DerivedEmbeddingFuture(EmbeddingsResultFuture):
        pass

    assert DerivedEmbeddingFuture.__annotations__.get("job") is None
    job = EmbeddingsJob(
        job_id="e0b1c3d4-0000-4000-8000-00000000000a",
        status="PENDING",
        created_date="2026-10-21T08:00:00Z",
    )
    future = Future.create(mock_session, job=job, sequences=["MKT"])
    assert type(future) is EmbeddingsResultFuture


def test_search_records_load_as_plain_futures(mock_session):
    job = Job.create(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": "/search/mmseqs2",
            "status": "PENDING",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    assert type(job) is Job
    assert type(Future.create(mock_session, job=job)) is Future
    mock_session.get.assert_not_called()


def test_job_type_alone_identifies_the_model_and_method(mock_session):
    job = task_job("batch/embeddings")
    assert type(job) is Job

    future = Future.create(mock_session, job=job)

    assert isinstance(future, results.EmbeddingFuture)
    future._require_outputs()
    assert (future._model_id, future._path) == ("esm2", "batch/embeddings")
    mock_session.get.assert_not_called()


def test_a_job_payload_naming_a_provider_does_not_identify_the_method(mock_session):
    job = Job.model_validate(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": "/models",
            "status": "SUCCESS",
            "created_date": "2026-10-21T08:00:00Z",
            "provider_id": "esm2",
            "path": "/batch/embeddings",
        }
    )

    assert _dispatch.identity(job) == (None, None)
    assert not _dispatch.recognizes(job)
    assert type(Future.create(mock_session, job=job)) is Future


def test_plain_job_encoded_only_in_job_type_loads_its_specialised_future(
    mock_session,
):
    job = Job(
        job_id="e0b1c3d4-0000-4000-8000-00000000000a",
        job_type="/models/gp/p-1/batch/predictions",
        status=JobStatus.SUCCESS,
        created_date="2026-10-21T08:00:00Z",
    )
    assert type(job) is Job

    future = Future.create(mock_session, job=job)

    assert type(future) is results.PredictionFuture
    assert future._model_id == "gp/p-1"
    assert future._path == "batch/predictions"
    future._require_outputs()
    mock_session.get.assert_not_called()
    mock_session.post.assert_not_called()


def test_duplicate_future_owners_select_the_first_in_name_order(mock_session, caplog):
    class DuplicateOwnerJob(Job):
        job_type: Literal["/test/duplicate-owner"]  # type: ignore[assignment]

    class BOwner(Future):
        job: DuplicateOwnerJob

        def _get(self, verbose=False, **kwargs):
            return None

    class AOwner(Future):
        job: DuplicateOwnerJob

        def _get(self, verbose=False, **kwargs):
            return None

    with caplog.at_level(logging.WARNING, logger="openprotein.jobs.futures"):
        future = Future.create(
            mock_session,
            job={
                "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
                "job_type": "/test/duplicate-owner",
                "status": "PENDING",
                "created_date": "2026-10-21T08:00:00Z",
            },
        )

    assert type(future) is AOwner
    assert any(
        "multiple future owners" in record.getMessage()
        and "BOwner" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(("path", "expected"), CASES)
def test_model_submission_uses_task_dispatch(mock_session, monkeypatch, path, expected):
    metadata = ModelMetadata.model_validate(
        {
            "model_id": "example",
            "description": {"summary": "Example"},
            "dimension": 1,
            "max_sequence_length": 100,
            "methods": [{"path": path, "outputs": [expected._primary_output]}],
        }
    )
    monkeypatch.setattr(
        _api,
        "run_method",
        lambda session, model_id, method, params: task_job(method, provider=model_id),
    )

    future = Model(mock_session, "example", metadata)._run("run", path)

    assert type(future) is expected
    assert future._submission.model_id == "example"
    assert future._submission.path == path


@pytest.mark.parametrize(
    ("family", "future_type"),
    [
        ("gp", GPFitFuture),
        ("svd", SVDFitFuture),
        ("umap", UMAPFitFuture),
    ],
)
def test_identified_fit_jobs_keep_their_fit_future(mock_session, family, future_type):
    future = Future.create(mock_session, job=task_job("fit", provider=family))
    assert type(future) is future_type


@pytest.mark.parametrize(
    ("future_type", "family"), [(SVDFitFuture, "svd"), (UMAPFitFuture, "umap")]
)
def test_pending_fit_projection_is_an_embedding_future(
    mock_session, monkeypatch, future_type, family
):
    fit = future_type(mock_session, task_job("fit", provider=family))
    sent = []

    def run(session, model_id, path, params):
        sent.append((model_id, path, params))
        return task_job(path, provider=family)

    monkeypatch.setattr(_api, "run_method", run)

    transformed = fit.transform(["MKT"])
    embedded = fit.embed(["MKT"])

    assert type(transformed) is results.EmbeddingFuture
    assert type(embedded) is results.EmbeddingFuture
    assert sent == [
        (f"{family}/{fit.id}", "batch/embeddings", {"sequences": ["MKT"]}),
        (f"{family}/{fit.id}", "batch/embeddings", {"sequences": ["MKT"]}),
    ]
    assert fit.status is JobStatus.PENDING
