import inspect
from typing import Any

import pytest

from openprotein.jobs import Job, _dispatch, futures
from openprotein.jobs.futures import (
    EmbeddingFuture,
    Future,
    GPFitFuture,
    StructureDesignFuture,
    SVDFitFuture,
    UMAPFitFuture,
    _ResultFuture,
    _Submission,
)
from openprotein.models.schemas import MethodDescriptor
from tests.models.test_future_results import (
    JOB,
    METADATA,
    UNITS,
    _get_side_effect,
)

_RESULT_FUTURES = [
    future_type
    for future_type in vars(futures).values()
    if isinstance(future_type, type)
    and issubclass(future_type, _ResultFuture)
    and future_type is not _ResultFuture
]

PAYLOAD_OUTPUTS = ["from-the-job-payload"]


def _fit_job(family: str) -> Job:
    return Job.model_validate(
        {
            "job_id": "fit-1",
            "job_type": f"/models/{family}/fit",
            "status": "SUCCESS",
            "created_date": "2026-10-30T00:00:00Z",
            "outputs": PAYLOAD_OUTPUTS,
        }
    )


def test_task_outputs_come_from_the_model_listing_not_the_job_payload(mock_session):
    mock_session.get.side_effect = _get_side_effect(UNITS, METADATA)
    job = Job.model_validate(JOB | {"outputs": PAYLOAD_OUTPUTS})

    assert Future(mock_session, job).outputs == ["embeddings", "annotations"]


def test_task_outputs_come_from_the_submitting_method_not_the_job_payload(
    mock_session,
):
    job = Job.model_validate(JOB | {"outputs": PAYLOAD_OUTPUTS})
    descriptor = MethodDescriptor(path="batch/embeddings", outputs=["embeddings"])
    submission = _Submission(job.job_type.split("/")[2], "batch/embeddings")

    future = _dispatch.create(
        mock_session, job, descriptor=descriptor, submission=submission
    )

    assert future.outputs == ["embeddings"]
    mock_session.get.assert_not_called()


def test_a_future_is_built_from_a_session_and_a_job():
    """The base future constructor accepts exactly a session and a job."""
    parameters = inspect.signature(Future.__init__).parameters

    assert list(parameters) == ["self", "session", "job"]


@pytest.mark.parametrize("future_type", [EmbeddingFuture, StructureDesignFuture])
def test_a_result_future_is_built_from_a_session_and_a_job(future_type):
    """A result future constructor accepts exactly a session and a job."""
    parameters = inspect.signature(future_type.__init__).parameters

    assert list(parameters) == ["self", "session", "job"]


@pytest.mark.parametrize("future_type", _RESULT_FUTURES)
def test_a_result_future_has_no_max_workers(mock_session, future_type):
    """A result future reads through the job's output stream and has no worker count."""
    future = future_type(mock_session, Job.model_validate(JOB))

    assert not hasattr(future, "max_workers")


@pytest.mark.parametrize(
    ("family", "future_type", "expected"),
    [
        ("gp", GPFitFuture, []),
        ("svd", SVDFitFuture, ["sequences"]),
        ("umap", UMAPFitFuture, ["sequences", "embeddings"]),
    ],
)
def test_each_fit_future_lists_its_fixed_outputs(
    mock_session, family, future_type, expected
):
    future = future_type(mock_session, _fit_job(family))

    assert future.outputs == expected
    mock_session.get.assert_not_called()


@pytest.mark.parametrize(
    ("future_type", "family", "default"),
    [(SVDFitFuture, "svd", "sequences"), (UMAPFitFuture, "umap", "sequences")],
)
def test_a_fit_future_defaults_to_its_first_output(
    mock_session, future_type, family, default
):
    assert future_type(mock_session, _fit_job(family))._default_output == default


def test_a_gp_fit_future_has_no_default_output(mock_session):
    with pytest.raises(ValueError, match="no outputs"):
        _ = GPFitFuture(mock_session, _fit_job("gp"))._default_output


def test_fit_job_payload_fields_do_not_change_fit_outputs(mock_session):
    payload: dict[str, Any] = _fit_job("svd").model_dump() | {
        "outputs": ["components"],
        "provider_id": "other",
        "path": "/other",
    }

    future = SVDFitFuture(mock_session, Job.model_validate(payload))

    assert future.outputs == ["sequences"]
