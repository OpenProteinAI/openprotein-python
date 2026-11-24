"""SVD handles and their models API routes."""

import importlib
import inspect
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from openprotein.base import APISession
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.errors import HTTPError, JobFailedException
from openprotein.jobs import Job, _dispatch
from openprotein.jobs.futures import SVDFitFuture
from openprotein.models import ModelsAPI
from openprotein.models.model import Model
from openprotein.models.svd import (
    SVDModel,
    _fit,
    _SVDRecord,
)


@pytest.fixture
def session():
    session = MagicMock(spec=APISession)
    session.timeout = 180
    session.headers = {"Authorization": "Bearer caller"}
    return session


def response(body):
    return MagicMock(json=lambda: body)


def error(status, body=None):
    r = response(body or {})
    r.status_code = status
    return HTTPError(r)


def card(record):
    return {
        "model_id": f"svd/{record['id']}",
        "type": "svd",
        "family_id": "svd",
        "is_system": False,
        "description": {"summary": "2-component SVD of esm2 embeddings"},
        "dimension": record.get("n_components", 2),
        "max_sequence_length": 1024,
        "methods": [{"path": "batch/embeddings", "outputs": ["embeddings"]}],
        "svd": record,
    }


@pytest.fixture
def row():
    return {
        "id": "fit-1",
        "owner": "alice",
        "requested_n_components": 4,
        "n_components": 2,
        "embedding": {
            "model_id": "esm2",
            "version": "1",
            "reduction": "MEAN",
            "prompt_id": None,
        },
    }


FIT_JOB = {
    "job_id": "fit-1",
    "job_type": "/models/svd/fit",
    "status": "PENDING",
    "created_date": "2026-09-28T00:00:00Z",
    "metadata": {"svd": {"embedding": {"model_id": "esm2", "version": "1"}}},
}


def jobs(session, routes):
    session.get.side_effect = lambda path, **kwargs: response(routes[path])


def test_svds_live_on_the_models_api(session):
    from openprotein.data.data import DataAPI

    assert callable(ModelsAPI(session).SVD)
    assert not hasattr(DataAPI(session), "svds")
    with pytest.raises(ImportError):
        importlib.import_module("openprotein.data.svds")


def test_fit_posts_the_request_and_returns_the_fit_future(session):
    session.post.return_value = response(FIT_JOB)
    future = _fit(session, "esm2", ["ACD"], n_components=2, seed=7)
    session.post.assert_called_once()
    call = session.post.call_args
    assert call.args == ("v1/models/svd",)
    assert call.kwargs["json"] == {
        "model_id": "esm2",
        "sequences": ["ACD"],
        "n_components": 2,
        "seed": 7,
        "output": "embeddings",
    }
    assert UUID(call.kwargs["headers"]["Idempotency-Key"])
    assert (future.id, future.status.value) == ("fit-1", "PENDING")
    assert session.headers == {"Authorization": "Bearer caller"}


def test_fit_with_sequences_uses_the_fit_request_body(session):
    session.post.return_value = response(FIT_JOB)
    _fit(
        session,
        sequences=["ACD"],
        model="esm2",
        prompt_id="prompt-1",
        reduction="MEAN",
        n_components=2,
        seed=7,
    )
    assert session.post.call_args.args == ("v1/models/svd",)
    assert session.post.call_args.kwargs["json"] == {
        "sequences": ["ACD"],
        "model_id": "esm2",
        "prompt_id": "prompt-1",
        "reduction": "MEAN",
        "output": "embeddings",
        "n_components": 2,
        "seed": 7,
    }


def test_get_reads_the_card(session, row):
    session.get.return_value = response(card(row))
    svd = ModelsAPI(session).SVD(id="fit-1")
    session.get.assert_called_once_with("v1/models/svd/fit-1")
    assert (svd.requested_n_components, svd.n_components) == (4, 2)
    assert svd.embedding is not None
    assert (svd.embedding.model_id, svd.embedding.reduction) == ("esm2", "MEAN")
    assert session.headers == {"Authorization": "Bearer caller"}


def test_embed_projects_through_the_fitted_model(session, row):
    session.get.return_value = response(card(row))
    session.post.return_value = response(
        {
            "job_id": "projection-2",
            "job_type": "/models/svd/batch/embeddings",
            "status": "PENDING",
            "created_date": "2026-09-28T00:00:00Z",
        }
    )
    svd = SVDModel._from_record(session, row)
    future = svd.embed(["ACD"])
    svd.embed(["EFG"])
    assert session.post.call_args_list[0].args[0] == (
        "v1/models/svd/fit-1/batch/embeddings"
    )
    assert session.post.call_args_list[0].kwargs["json"] == {"sequences": ["ACD"]}
    assert (future.id, future.outputs) == ("projection-2", ["embeddings"])
    session.get.assert_called_once_with("v1/models/svd/fit-1", timeout=180)


def test_list_reads_the_callers_svds(session, row):
    session.get.return_value = response([card(row)])
    listed = ModelsAPI(session).SVD.list(limit=None)
    session.get.assert_called_once_with(
        "v1/models",
        params={
            "scope": "mine",
            "type": "svd",
            "verbose": "true",
            "limit": 1000,
            "offset": 0,
        },
    )
    assert [svd.id for svd in listed] == ["svd/fit-1"]


def test_list_pages_with_limit_and_offset(session, row):
    session.get.return_value = response([card(row)])
    ModelsAPI(session).SVD.list(limit=10, offset=20)
    session.get.assert_called_once_with(
        "v1/models",
        params={
            "scope": "mine",
            "type": "svd",
            "verbose": "true",
            "limit": 10,
            "offset": 20,
        },
    )


def test_fit_takes_a_model_handle_and_leaves_n_components_to_the_server(session):
    session.post.return_value = response(FIT_JOB)
    _fit(session, Model(session, "esm2"), ["ACD"])
    assert session.post.call_args.kwargs["json"] == {
        "sequences": ["ACD"],
        "model_id": "esm2",
        "output": "embeddings",
    }


def test_fit_refuses_a_model_of_the_wrong_type_before_any_request(session):
    with pytest.raises(TypeError, match="model must be"):
        _fit(session, 3, ["ACD"])
    session.post.assert_not_called()


def test_fit_passes_poet2_embedding_params(session):
    session.post.return_value = response(FIT_JOB)
    _fit(
        session,
        sequences=["ACD"],
        model="poet-2",
        prompt_id="prompt-1",
        query_id="query-1",
        use_query_structure_in_decoder=False,
        decoder_type="mlm",
        n_components=2,
    )
    body = session.post.call_args.kwargs["json"]
    assert body["query_id"] == "query-1"
    assert body["use_query_structure_in_decoder"] is False
    assert body["decoder_type"] == "mlm"


@pytest.mark.parametrize("svd_id", ["fit-1", "svd/fit-1"])
def test_get_and_delete_take_a_bare_or_prefixed_id(session, row, svd_id):
    session.get.return_value = response(card(row))
    svd = ModelsAPI(session).SVD(id=svd_id)
    SVDModel(session, svd_id).delete()
    session.get.assert_called_once_with("v1/models/svd/fit-1")
    session.delete.assert_called_once_with("v1/models/svd/fit-1")
    assert (svd.id, svd._svd_record.id) == ("svd/fit-1", "fit-1")


def test_delete(session, row):
    SVDModel._from_record(session, row).delete()
    session.delete.assert_called_once_with("v1/models/svd/fit-1")
    assert session.headers == {"Authorization": "Bearer caller"}


@pytest.mark.parametrize("status", [404, 410])
def test_deleted_or_other_owner_svd_does_not_fall_back(session, status):
    session.get.side_effect = error(status)
    with pytest.raises(HTTPError) as raised:
        ModelsAPI(session).SVD(id="fit-1")
    assert raised.value.status_code == status
    session.get.assert_called_once_with("v1/models/svd/fit-1")


@pytest.mark.parametrize("kind", ["id", "metadata", "dataset"])
def test_fit_from_an_assay_sends_its_id(session, kind):
    session.post.return_value = response(FIT_JOB)
    metadata = MagicMock(spec=AssayMetadata, assay_id="assay-1")
    assay = {
        "id": "assay-1",
        "metadata": metadata,
        "dataset": AssayDataset(session, metadata),
    }[kind]
    _fit(session, "esm2", assay=assay, n_components=2)
    assert session.post.call_args.kwargs["json"] == {
        "assay_id": "assay-1",
        "model_id": "esm2",
        "output": "embeddings",
        "n_components": 2,
    }


@pytest.mark.parametrize("status", ["PENDING", "SUCCESS", "FAILURE"])
def test_fit_returns_the_fit_future(session, status):
    session.post.return_value = response(FIT_JOB | {"status": status})
    future = _fit(session, "esm2", ["ACD"], n_components=2)
    assert isinstance(future, SVDFitFuture)
    assert (future.id, future.status.value) == ("fit-1", status)
    session.get.assert_not_called()


@pytest.mark.parametrize("status", ["PENDING", "SUCCESS", "FAILURE"])
def test_fit_reads_the_fit_job_when_the_operation_names_only_its_id(session, status):
    session.post.return_value = response({"id": "fit-1", "state": "pending"})
    session.get.return_value = response(FIT_JOB | {"status": status})
    future = _fit(session, "esm2", ["ACD"], n_components=2)
    assert (future.id, future.status.value) == ("fit-1", status)
    session.get.assert_called_once_with("v1/jobs/fit-1")


def test_wait_returns_the_svd_once_the_fit_succeeds(session, row):
    jobs(
        session,
        {
            "v1/jobs/fit-1": FIT_JOB | {"status": "SUCCESS"},
            "v1/models/svd/fit-1": card(row),
        },
    )
    future = SVDFitFuture(session, Job.model_validate(FIT_JOB))
    svd = future.wait(interval=0)
    assert isinstance(svd, SVDModel) and svd.n_components == 2


@pytest.mark.parametrize("status", ["FAILURE", "CANCELED"])
def test_wait_raises_naming_a_fit_that_did_not_succeed(session, row, status):
    jobs(session, {"v1/jobs/fit-1": FIT_JOB | {"status": status}})
    with pytest.raises(JobFailedException) as raised:
        SVDFitFuture(session, Job.model_validate(FIT_JOB)).wait(interval=0)
    assert raised.value.job_id == "fit-1"


def test_a_fit_future_reads_its_emitted_embeddings_by_name(session):
    rows = [[0.1, 0.2], [0.3, 0.4]]
    jobs(
        session,
        {
            "v1/data/results/fit-1/embeddings": {
                "output": "embeddings",
                "count": 2,
                "complete": True,
                "values": [
                    {"position": i, "unit_id": "u", "value": {"inline": row}}
                    for i, row in enumerate(rows)
                ],
                "next_cursor": None,
            }
        },
    )
    future = SVDFitFuture(session, Job.model_validate(FIT_JOB))
    assert future.get(output="embeddings") == rows


def test_records_parse_without_exposing_job_fields(row):
    record = _SVDRecord.model_validate(
        row
        | {
            "state": "ready",
            "job_id": "fit-1",
            "transform_job_id": "transform-1",
            "failure_summary": None,
        }
    )
    dumped = record.model_dump()
    assert {"state", "job_id", "transform_job_id", "failure_summary"}.isdisjoint(dumped)


def test_svd_handles_carry_no_projections_or_job_state(session, row):
    svd = SVDModel._from_record(session, row)
    assert not any(
        hasattr(svd, name)
        for name in ("state", "job", "transform_job_id", "wait", "embeddings")
    )


@pytest.mark.parametrize("status", [404, 425])
def test_get_raises_a_404_unless_the_svd_is_live(session, status):
    session.get.side_effect = error(status)
    with pytest.raises(HTTPError) as raised:
        ModelsAPI(session).SVD(id="fit-1")
    assert raised.value.status_code == 404
    session.get.assert_called_once_with("v1/models/svd/fit-1")


NON_MODELS_FIT = {
    "job_id": "fit-1",
    "job_type": "/svd/fit",
    "status": "PENDING",
    "created_date": "2026-09-28T00:00:00Z",
}


def test_fit_tracks_a_fit_job_outside_the_models_api(session):
    session.post.return_value = response({"id": "fit-1", "state": "pending"})
    session.get.return_value = response(NON_MODELS_FIT)
    future = _fit(session, "prot-seq", ["AA", "AC"], n_components=2)
    assert isinstance(future, SVDFitFuture)
    assert not _dispatch.is_models_job(future.job)
    assert future.id == "fit-1"


def test_wait_returns_the_svd_of_a_fit_job_outside_the_models_api(session, row):
    jobs(
        session,
        {
            "v1/jobs/fit-1": NON_MODELS_FIT | {"status": "SUCCESS"},
            "v1/models/svd/fit-1": card(row),
        },
    )
    future = SVDFitFuture(session, Job.create(NON_MODELS_FIT))
    svd = future.wait(interval=0)
    assert isinstance(svd, SVDModel) and svd.id == "svd/fit-1"
    assert [call.args[0] for call in session.get.call_args_list] == [
        "v1/jobs/fit-1",
        "v1/models/svd/fit-1",
    ]


@pytest.mark.parametrize("outputs", [["components", "embeddings"], None])
def test_an_svd_fit_future_lists_its_fixed_outputs(session, outputs):
    job = Job.model_validate(
        {
            "job_id": "fit-1",
            "job_type": "/models/svd/fit",
            "status": "SUCCESS",
            "created_date": "2026-10-30T00:00:00Z",
            "outputs": outputs,
        }
    )

    future = SVDFitFuture(session, job)

    assert future.outputs == ["sequences"]


def test_public_constructor_takes_session_model_id_and_metadata():
    """The public constructor has no record parameter."""
    assert list(inspect.signature(SVDModel).parameters) == [
        "session",
        "model_id",
        "metadata",
    ]
