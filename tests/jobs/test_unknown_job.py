"""A job type this client has no typed support for stays usable.

The server can offer job types newer than this client, so resolving one
reports status and raw results, and fails only where typed data is genuinely
unavailable.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from openprotein.jobs import _dispatch
from openprotein.jobs.futures import Future, UnknownJobFuture
from openprotein.jobs.schemas import Job, JobStatus, JobType


@pytest.fixture
def unknown_job_dict() -> dict[str, Any]:
    """A job whose type no installed Future claims."""
    return {
        "job_id": "job-unknown-1",
        "job_type": "/some/type/added/later",
        "status": "SUCCESS",
        "created_date": datetime(2026, 9, 10, 12, 0, 0).isoformat(),
    }


def test_an_unclaimed_job_type_resolves_to_the_fallback(unknown_job_dict) -> None:
    future = Future.create(session=MagicMock(), job=Job(**unknown_job_dict))
    assert isinstance(future, UnknownJobFuture)


def test_the_fallback_reports_identity_and_status(unknown_job_dict) -> None:
    future = Future.create(session=MagicMock(), job=Job(**unknown_job_dict))
    assert future.id == "job-unknown-1"
    assert future.job_type == "/some/type/added/later"
    assert future.status is JobStatus.SUCCESS


def test_the_fallback_names_the_job_type_when_asked_for_typed_results(
    unknown_job_dict,
) -> None:
    """The error arrives where the capability is missing, not at resolution."""
    future = Future.create(session=MagicMock(), job=Job(**unknown_job_dict))
    with pytest.raises(NotImplementedError, match="/some/type/added/later"):
        future.get()


def test_a_claimed_job_type_still_resolves_to_its_own_future() -> None:
    """The fallback is last resort: a known type is unaffected by it."""
    from openprotein.align.schemas import MSAJob

    job = MSAJob(
        job_id="msa-1",
        job_type=JobType.align_align,
        status=JobStatus.SUCCESS,
        created_date=datetime(2026, 9, 10, 12, 0, 0),
    )
    future = Future.create(session=MagicMock(), job=job)
    assert not isinstance(future, UnknownJobFuture)
    assert type(future).__name__ == "MSAFuture"


def _models_payload(**overrides: Any) -> dict[str, Any]:
    """A `/models` job payload."""
    return {
        "job_id": "m-1",
        "job_type": "/models",
        "status": "SUCCESS",
        "created_date": datetime(2026, 9, 10, 12, 0, 0).isoformat(),
    } | overrides


def test_a_malformed_payload_for_a_known_type_is_not_silently_degraded() -> None:
    """A schema error for a typed job type surfaces as a validation error."""
    with pytest.raises(ValidationError):
        Job.create(_models_payload(job_type="/align/prompt", status="NOT_A_STATUS"))


def test_a_models_payload_loads_as_a_job_with_its_args() -> None:
    job = Job.create(_models_payload(args={"model_id": "prompt"}))
    assert type(job) is Job
    assert _dispatch.is_models_job(job)
    assert job.model_dump()["args"] == {"model_id": "prompt"}


def test_an_unreadable_response_reports_its_own_error() -> None:
    """A body that is not JSON surfaces the decode failure."""
    from requests import Response

    response = Response()
    response._content = b"<html>gateway error</html>"
    response.status_code = 502
    with pytest.raises(ValueError):
        Job.create(response)


def test_a_models_job_resolves_to_the_plain_future() -> None:
    """The live models path is unaffected by the fallback."""
    job = Job.create(_models_payload(args={"model_id": "prompt"}))
    future = Future.create(session=MagicMock(), job=job)
    assert type(future) is Future
    assert not isinstance(future, UnknownJobFuture)


@pytest.mark.parametrize("job_type", ["/models/boltzgen", "/models/rfdiffusion"])
def test_a_models_type_naming_no_method_resolves_to_the_fallback(job_type) -> None:
    """A ``/models/<model>`` type names no method, so it parses as a plain job."""
    job = Job.create(_models_payload(job_type=job_type))

    assert type(job) is Job
    assert isinstance(Future.create(session=MagicMock(), job=job), UnknownJobFuture)


@pytest.mark.parametrize(
    ("job_type", "future_name"),
    [
        ("/models/poet-2/batch/embeddings", "EmbeddingFuture"),
        ("/models/svd/fit", "SVDFitFuture"),
    ],
)
def test_a_per_model_job_type_loads_as_a_job_and_its_future(
    job_type, future_name
) -> None:
    """Task-addressed types load as a plain job, keeping type and args."""
    job = Job.create(_models_payload(job_type=job_type, args={"model_id": "poet-2"}))
    assert type(job) is Job
    assert _dispatch.is_models_job(job)
    assert job.job_type == job_type
    assert job.model_dump()["args"] == {"model_id": "poet-2"}
    future = Future.create(session=MagicMock(), job=job)
    assert type(future).__name__ == future_name
