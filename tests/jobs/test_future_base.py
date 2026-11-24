"""`Future` as the one job handle, with the positional protocol on result futures."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openprotein.align.msa import MSAFuture
from openprotein.errors import APIError, JobFailedException
from openprotein.jobs import Future, Job, JobsAPI, JobStatus
from openprotein.jobs import futures as results
from openprotein.prompt import Prompt, PromptJob
from openprotein.prompt.schemas import PromptMetadata

MODEL_ID = "esm2_t33_650M_UR50D"

METADATA: dict[str, Any] = {
    "model_id": MODEL_ID,
    "model_family": "esm2",
    "name": None,
    "version": None,
    "is_system": True,
    "base": True,
    "tags": None,
    "protein_types": None,
    "description": {"summary": "ESM-2 650M protein language model."},
    "dimension": 1280,
    "query": None,
    "multichain": None,
    "methods": [
        {
            "path": "batch/embeddings",
            "outputs": ["embeddings", "annotations"],
            "description": "Per-residue embeddings.",
        },
    ],
    "max_sequence_length": 4094,
}

MODELS_JOB: dict[str, Any] = {
    "job_id": "job-1",
    "job_type": f"/models/{MODEL_ID}/batch/embeddings",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
}

ALIGN_JOB: dict[str, Any] = {
    "job_id": "job-2",
    "job_type": "/align/mafft",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
}

UNITS = [{"position": 0, "unit_id": "u0", "state": "SUCCESS"}]


def _serve(session: MagicMock, args: dict | None = None) -> None:
    def handler(path: str, **kwargs: Any) -> MagicMock:
        response = MagicMock()
        if path.endswith("/units"):
            response.json.return_value = {"units": UNITS, "next_position": None}
        elif path == f"v1/models/{MODEL_ID}":
            response.json.return_value = METADATA
        elif path.endswith("/args"):
            response.json.return_value = args or {}
        else:
            raise AssertionError(f"unexpected GET {path}")
        return response

    session.get.side_effect = handler


def _prompt(session: MagicMock, job_type: str = "/prompt/sample_msa") -> Prompt:
    job = PromptJob(
        job_id="job-3",
        job_type=job_type,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
    )
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = job
    return Prompt(session, job=job, num_replicates=1)


def _legacy(session: MagicMock, payload: dict) -> MSAFuture:
    return MSAFuture(session, Job.model_validate(payload))


def _legacy_futures(session: MagicMock) -> list[Future]:
    return [_legacy(session, ALIGN_JOB), _prompt(session)]


def _prompt_with_own_id(session: MagicMock, job_id: str | None) -> Prompt:
    job = PromptJob(
        job_id="job-3",
        job_type="/prompt/sample_msa",
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
    )
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = job
    metadata = PromptMetadata(
        id="prompt-9",
        name="prompt",
        description=None,
        created_date=job.created_date,
        num_replicates=1,
        job_id=job_id,
        status=JobStatus.SUCCESS,
    )
    return Prompt(session, metadata=metadata)


def test_job_reads_address_the_job_id_not_the_resource_id(
    mock_session: MagicMock,
) -> None:
    """A future whose `id` is a resource id reads its job's args and units by job id."""
    prompt = _prompt_with_own_id(mock_session, "job-3")
    assert prompt.id == "prompt-9"
    _serve(mock_session, args={"replicates": 1})

    assert prompt.args == {"replicates": 1}
    assert [unit.position for unit in prompt.units] == [0]
    paths = [call.args[0] for call in mock_session.get.call_args_list]
    assert paths and all("prompt-9" not in path for path in paths)
    assert any("job-3" in path for path in paths)


def test_reading_args_of_a_future_without_a_job_raises_without_a_request(
    mock_session: MagicMock,
) -> None:
    """A future with no job has no args to read, and sends nothing."""
    prompt = _prompt_with_own_id(mock_session, None)
    mock_session.reset_mock()

    with pytest.raises(APIError, match="no job"):
        _ = prompt.args
    mock_session.get.assert_not_called()


def test_a_legacy_future_is_not_falsy_and_makes_no_request(
    mock_session: MagicMock,
) -> None:
    """`bool()` of a future that subclasses `Future` directly sends nothing."""
    futures = _legacy_futures(mock_session)
    mock_session.reset_mock()

    assert all(bool(future) for future in futures)
    assert mock_session.method_calls == []


@pytest.mark.parametrize(
    "future_type",
    [
        Future,
        results.FitFuture,
        results.GPFitFuture,
        results.CrossvalidateFuture,
        MSAFuture,
        Prompt,
    ],
)
def test_futures_other_than_result_futures_have_no_positional_protocol(
    future_type: type[Future],
) -> None:
    """`len()`, iteration and indexing belong to result futures alone."""
    for name in ("__len__", "__iter__", "__getitem__", "__keys__"):
        assert not hasattr(future_type, name), name
    assert issubclass(results.EmbeddingFuture, results._ResultFuture)
    assert hasattr(results.EmbeddingFuture, "__len__")


def test_the_args_of_a_legacy_future_are_fetched_once(mock_session: MagicMock) -> None:
    """A job without inline args reads them from the job once, then keeps them."""
    _serve(mock_session, args={"sequences": ["MKT"]})
    future = _legacy(mock_session, ALIGN_JOB)

    assert future.args == {"sequences": ["MKT"]}
    assert future.args == {"sequences": ["MKT"]}

    arg_reads = [c for c in mock_session.get.call_args_list if "args" in c.args[0]]
    assert len(arg_reads) == 1


def test_a_legacy_future_without_its_own_repr_prints_the_compact_form(
    mock_session: MagicMock,
) -> None:
    """The repr names the class, job id, status and args, loaded or not."""
    future = _legacy(mock_session, ALIGN_JOB)

    assert repr(future) == (
        "MSAFuture(job_id='job-2', status='SUCCESS', args=<not loaded>)"
    )
    assert str(future) == repr(future)

    inline = Job.model_validate(ALIGN_JOB | {"args": {"sequences": ["MKT" * 20]}})
    assert "args={'sequences': <1 items>}" in repr(MSAFuture(mock_session, inline))


def test_a_legacy_future_reads_units_and_outputs_when_its_job_type_serves_them(
    mock_session: MagicMock,
) -> None:
    """`units` and `outputs` read a stored job's records and method outputs."""
    _serve(mock_session)
    future = _legacy(mock_session, MODELS_JOB)

    assert [unit.position for unit in future.units] == [0]
    assert future.outputs == ["embeddings", "annotations"]


def test_a_legacy_future_without_stored_outputs_raises_on_every_output_read(
    mock_session: MagicMock,
) -> None:
    """A job type that serves no outputs raises the same error from each reader."""
    future = _legacy(mock_session, ALIGN_JOB)
    reads = [
        lambda: future.units,
        lambda: future.outputs,
        lambda: future.unit(0, output="embeddings"),
        lambda: future.get_item(0, output="embeddings"),
        lambda: future.stream(output="embeddings"),
    ]
    for read in reads:
        with pytest.raises(APIError, match="has no stored outputs"):
            read()
    mock_session.get.assert_not_called()


def test_an_uploaded_prompt_has_no_stored_outputs(mock_session: MagicMock) -> None:
    """A prompt with no job raises the stored-outputs error."""
    prompt = Prompt(
        mock_session,
        metadata=_prompt(mock_session).metadata.model_copy(update={"job_id": None}),
    )

    assert prompt.job is None
    with pytest.raises(APIError, match="has no stored outputs"):
        _ = prompt.units


def test_a_sampling_prompt_reads_its_units(mock_session: MagicMock) -> None:
    """A prompt sampled from a job stores outputs, so its units are readable."""
    prompt = _prompt(mock_session)
    _serve(mock_session)

    assert [unit.position for unit in prompt.units] == [0]


def _canceled(session: MagicMock, payload: dict, future_type: type[Future]) -> Future:
    session.get.return_value.json.return_value = payload | {"status": "CANCELED"}
    return future_type(session, Job.model_validate(payload | {"status": "RUNNING"}))


@pytest.mark.parametrize("future_type", [Future, results.EmbeddingFuture])
def test_waiting_on_a_canceled_job_with_stored_outputs_raises(
    mock_session: MagicMock, future_type: type[Future]
) -> None:
    """`wait()` raises for a canceled job that stores outputs."""
    future = _canceled(mock_session, MODELS_JOB, future_type)
    with (
        patch("time.sleep"),
        patch.object(future_type, "get") as get,
        pytest.raises(JobFailedException, match="canceled") as raised,
    ):
        future.wait(interval=0)
    assert raised.value.job_id == "job-1"
    get.assert_not_called()


@pytest.mark.parametrize(
    ("job_type", "make"),
    [
        (
            "/predictor/crossvalidate",
            lambda session, job: results.CrossvalidateFuture(session, job, MagicMock()),
        ),
    ],
)
def test_waiting_on_a_canceled_result_future_raises_for_any_job_type(
    mock_session: MagicMock, job_type: str, make: Any
) -> None:
    """A cross-validation future raises on a canceled job of any type."""
    payload = MODELS_JOB | {"job_type": job_type}
    mock_session.get.return_value.json.return_value = payload | {"status": "CANCELED"}
    future = make(mock_session, Job.model_validate(payload | {"status": "RUNNING"}))
    with (
        patch("time.sleep"),
        patch.object(type(future), "_get", return_value="read"),
        pytest.raises(JobFailedException, match="canceled"),
    ):
        future.wait(interval=0)


def test_waiting_on_a_canceled_job_without_stored_outputs_reads_its_results(
    mock_session: MagicMock,
) -> None:
    """A job type that stores no outputs reads its results whatever its status."""
    future = _canceled(mock_session, ALIGN_JOB, MSAFuture)
    with (
        patch("time.sleep"),
        patch.object(MSAFuture, "_get", return_value="results"),
    ):
        assert future.wait(interval=0) == "results"
    assert future.status is JobStatus.CANCELED


def test_a_read_passes_a_named_output_to_stream(
    mock_session: MagicMock,
) -> None:
    """A plain `Future` reads its job's units through `stream` with the named output."""
    future = Future(mock_session, Job.model_validate(MODELS_JOB))
    with (
        patch.object(Future, "stream", return_value=iter(["a", "b"])) as stream,
        patch.object(Future, "units", new=[MagicMock()]),
    ):
        assert future.get(verbose=True, output="annotations") == ["a", "b"]
    stream.assert_called_once_with(output="annotations")


def test_a_job_without_a_models_identity_is_tracked_by_a_plain_future(
    mock_session: MagicMock,
) -> None:
    """An unresolved models identity falls back to exactly `Future`."""
    job = Job.model_validate(MODELS_JOB | {"job_type": "/models"})

    assert type(Future.create(mock_session, job=job)) is Future
