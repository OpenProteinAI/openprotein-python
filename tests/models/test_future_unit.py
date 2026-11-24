"""One unit of a models job, handled on its own."""

import inspect
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from requests import Response

import openprotein.jobs as jobs_package
from openprotein.errors import APIError, HTTPError, JobFailedException, TimeoutException
from openprotein.jobs import Job, JobStatus
from openprotein.jobs.futures import (
    EmbeddingFuture,
    Future,
    FutureUnit,
)
from openprotein.prompt import PromptJob

MODEL_ID = "esm2_t33_650M_UR50D"

JOB: dict[str, Any] = {
    "job_id": "job-1",
    "job_type": f"/models/{MODEL_ID}/batch/embeddings",
    "status": "RUNNING",
    "created_date": "2026-09-05T08:00:00Z",
}

DEFAULT = "v1/data/results/job-1"
LISTING = "v1/jobs/job-1/units"


def _unit_record(position: int, state: str, **extra: Any) -> dict:
    return {
        "position": position,
        "unit_id": f"u{position}",
        "state": state,
        "retries": 0,
    } | extra


class _Recorder:
    """Serves the unit, listing and result endpoints, recording every path fetched."""

    def __init__(
        self,
        state: str = "SUCCESS",
        positions: int = 2,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.paths: list[str] = []
        self.state = state
        self.positions = positions
        self.error = error

    def __call__(self, path: str, **kwargs: Any) -> MagicMock:
        self.paths.append(path)
        response = MagicMock()
        if "/units/" in path:
            position = int(path.rsplit("/", 1)[1])
            response.json.return_value = _unit_record(
                position, self.state, error=self.error
            )
        elif path == LISTING:
            response.json.return_value = {
                "units": [
                    _unit_record(position, self.state, error={"message": "boom"})
                    for position in range(self.positions)
                ],
                "next_position": None,
            }
        elif path.startswith("v1/models/"):
            raise AssertionError("a read must not consult the model listing")
        elif path == DEFAULT:
            assert kwargs["params"] == {"limit": 1}
            response.json.return_value = {
                "values": [],
                "next_cursor": None,
                "output": "embeddings",
            }
        elif "/results" in path:
            output = "annotations" if "/annotations/" in path else "embeddings"
            response.json.return_value = {
                "position": int(path.rsplit("/", 1)[1]),
                "unit_id": "unit",
                "value": {
                    "inline": "some text" if output == "annotations" else [1.0, 2.0]
                },
            }
        else:
            raise AssertionError(f"unexpected GET {path}")
        return response

    def unit_reads(self, position: int) -> int:
        return self.paths.count(f"v1/jobs/job-1/units/{position}")


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture
def future(mock_session, recorder) -> Future:
    mock_session.get.side_effect = recorder
    return Future(mock_session, Job.model_validate(JOB))


def _future_in(mock_session, state: str, **kwargs: Any) -> Future:
    mock_session.get.side_effect = _Recorder(state=state, **kwargs)
    return Future(mock_session, Job.model_validate(JOB))


class TestSelection:
    def test_a_unit_defaults_to_the_output_the_bare_route_names(self, future, recorder):
        assert future.unit(0).output == "embeddings"
        assert future.unit(1).output == "embeddings"

        assert recorder.paths == [DEFAULT]

    def test_a_unit_can_name_any_output(self, future, recorder):
        assert future.unit(0, output="annotations").output == "annotations"

        assert recorder.paths == []


class TestLaziness:
    def test_creating_a_unit_fetches_nothing(self, future, recorder):
        unit = future.unit(0)
        future.unit(0, output="embeddings")
        future.unit(1, output="annotations")
        repr(unit)

        assert recorder.paths == []

    def test_the_snapshot_is_read_from_one_position(self, future, recorder):
        unit = future.unit(1, output="embeddings")

        assert unit.unit_id == "u1"
        assert recorder.paths == ["v1/jobs/job-1/units/1"]

    def test_the_snapshot_is_read_once(self, future, recorder):
        unit = future.unit(0)
        assert unit.unit_id == "u0"
        assert unit.status is JobStatus.SUCCESS
        assert unit.error is None

        assert recorder.unit_reads(0) == 1

    def test_refresh_reads_the_position_again(self, future, recorder):
        unit = future.unit(0)
        assert unit.status is JobStatus.SUCCESS
        unit.refresh()
        assert unit.status is JobStatus.SUCCESS

        assert recorder.unit_reads(0) == 2

    def test_a_unit_carries_its_coordinate_without_fetching(self, future, recorder):
        unit = future.unit(3, output="annotations")

        assert (unit.job_id, unit.position, unit.output) == ("job-1", 3, "annotations")
        assert recorder.paths == []


class TestListing:
    def test_listed_units_hold_their_record_as_the_snapshot(self, future, recorder):
        units = future.units

        assert all(isinstance(unit, FutureUnit) for unit in units)
        assert [(u.position, u.unit_id, u.status) for u in units] == [
            (0, "u0", JobStatus.SUCCESS),
            (1, "u1", JobStatus.SUCCESS),
        ]
        assert [unit.error for unit in units] == [{"message": "boom"}] * 2
        assert recorder.paths == [LISTING]

    def test_reading_units_does_not_request_the_default_output(self, future, recorder):
        _ = future.units

        assert recorder.paths == [LISTING]

    def test_a_listed_unit_and_a_requested_unit_agree(self, future):
        listed = future.units[1]
        requested = future.unit(1)

        assert type(listed) is type(requested) is FutureUnit
        assert (listed.job_id, listed.position, listed.output) == (
            requested.job_id,
            requested.position,
            requested.output,
        )

    def test_units_are_in_position_order(self, future, mock_session):
        def handler(path: str, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.json.return_value = {
                "units": [_unit_record(2, "RUNNING"), _unit_record(0, "RUNNING")],
                "next_position": None,
            }
            return response

        mock_session.get.side_effect = handler

        assert [unit.position for unit in future.units] == [0, 2]

    def test_a_result_future_selects_its_primary_output(self, mock_session, recorder):
        mock_session.get.side_effect = recorder
        future = EmbeddingFuture(mock_session, Job.model_validate(JOB))

        assert [unit.output for unit in future.units] == ["embeddings", "embeddings"]
        assert recorder.paths == [LISTING]


class TestListingLifetime:
    @pytest.fixture
    def listed(self, future, recorder, monkeypatch):
        monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
        done = Job.model_validate(JOB | {"status": "SUCCESS"})
        monkeypatch.setattr(Future, "_refresh_job", lambda self, timeout=None: done)
        monkeypatch.setattr(Future, "_wait_job", lambda self, **kwargs: done)
        monkeypatch.setattr(Future, "get", lambda self, **kwargs: None)
        return future, recorder

    @pytest.mark.parametrize("reload", ["refresh", "wait_until_done", "wait"])
    def test_units_are_read_again_after_the_job_is_polled(self, listed, reload):
        future, recorder = listed
        first = future.units
        assert future.units is first
        assert recorder.paths.count(LISTING) == 1

        getattr(future, reload)()
        second = future.units

        assert second is not first
        assert second is future.units
        assert recorder.paths.count(LISTING) == 2

    def test_a_handle_keeps_its_snapshot_across_a_poll(self, listed):
        future, recorder = listed
        unit = future.units[0]

        future.refresh()

        assert unit.status is JobStatus.SUCCESS
        assert recorder.unit_reads(0) == 0


class TestRetrieval:
    def test_get_decodes_only_the_selected_position(self, future, recorder):
        value = future.unit(1).get()

        np.testing.assert_array_equal(value, np.array([1.0, 2.0]))
        assert "v1/data/results/job-1/embeddings/1" in recorder.paths

    def test_inline_values_come_back_as_stored(self, future):
        assert future.unit(0, output="annotations").get() == "some text"


def _not_found(path: str) -> HTTPError:
    response = Response()
    response.status_code = 404
    response.url = path
    response._content = b'{"detail": "no output is stored"}'
    return HTTPError(response)


class TestFailedUnitRead:
    def test_get_raises_for_a_snapshot_that_is_failed(self, mock_session):
        future = _future_in(mock_session, "FAILED", error={"message": "boom"})
        unit = future.units[0]

        with pytest.raises(JobFailedException) as raised:
            unit.get()

        assert raised.value.failure_message == "unit 0 failed: boom"

    def test_get_raises_for_a_snapshot_that_is_canceled(self, mock_session):
        unit = _future_in(mock_session, "CANCELED").units[0]

        with pytest.raises(JobFailedException) as raised:
            unit.get()

        assert raised.value.failure_message == "unit 0 was canceled"

    def test_a_missing_output_refreshes_and_raises_for_a_failed_unit(
        self, mock_session
    ):
        recorder = _Recorder(state="FAILED", error={"message": "boom"})

        def handler(path: str, **kwargs: Any) -> MagicMock:
            if "/results" in path:
                raise _not_found(path)
            return recorder(path, **kwargs)

        mock_session.get.side_effect = handler
        unit = Future(mock_session, Job.model_validate(JOB)).unit(
            0, output="embeddings"
        )

        with pytest.raises(JobFailedException) as raised:
            unit.get()

        assert raised.value.failure_message == "unit 0 failed: boom"
        assert recorder.unit_reads(0) == 1

    def test_a_missing_output_of_a_live_unit_raises_the_not_found(self, mock_session):
        recorder = _Recorder(state="RUNNING")

        def handler(path: str, **kwargs: Any) -> MagicMock:
            if "/results" in path:
                raise _not_found(path)
            return recorder(path, **kwargs)

        mock_session.get.side_effect = handler
        unit = Future(mock_session, Job.model_validate(JOB)).unit(
            0, output="embeddings"
        )

        with pytest.raises(HTTPError) as raised:
            unit.get()

        assert raised.value.status_code == 404
        assert recorder.unit_reads(0) == 1

    def test_a_failing_refresh_keeps_the_not_found_as_the_error(self, mock_session):
        def handler(path: str, **kwargs: Any) -> MagicMock:
            if "/results" in path:
                raise _not_found(path)
            raise ConnectionError("refresh failed")

        mock_session.get.side_effect = handler
        unit = Future(mock_session, Job.model_validate(JOB)).unit(
            0, output="embeddings"
        )

        with pytest.raises(HTTPError) as raised:
            unit.get()

        assert raised.value.status_code == 404
        assert isinstance(raised.value.__cause__, ConnectionError)

    def test_a_successful_read_requests_no_unit_status(self, future, recorder):
        future.unit(0, output="embeddings").get()

        assert recorder.unit_reads(0) == 0


class TestStatus:
    @pytest.mark.parametrize(
        ("state", "status", "done"),
        [
            ("PENDING", JobStatus.PENDING, False),
            ("READY", JobStatus.PENDING, False),
            ("RUNNING", JobStatus.RUNNING, False),
            ("SUCCESS", JobStatus.SUCCESS, True),
            ("FAILED", JobStatus.FAILURE, True),
            ("CANCELED", JobStatus.CANCELED, True),
        ],
    )
    def test_done_reads_the_snapshot(self, mock_session, state, status, done):
        future = _future_in(mock_session, state)
        unit = future.units[0]
        reads = mock_session.get.call_count

        assert unit.status is status
        assert unit.done() is done
        assert mock_session.get.call_count == reads

    def test_a_stored_unit_is_done_while_its_job_runs(self, future):
        assert future.status is JobStatus.RUNNING
        assert future.unit(0).done() is True


class TestWait:
    def test_wait_returns_the_value_of_a_successful_unit(self, future):
        np.testing.assert_array_equal(future.unit(0).wait(), np.array([1.0, 2.0]))

    def test_wait_refreshes_until_the_unit_is_done(self, mock_session, monkeypatch):
        states = iter(["PENDING", "RUNNING", "SUCCESS"])
        recorder = _Recorder()

        def handler(path: str, **kwargs: Any) -> MagicMock:
            response = recorder(path, **kwargs)
            if "/units/" in path:
                response.json.return_value = _unit_record(0, next(states))
            return response

        mock_session.get.side_effect = handler
        monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
        future = Future(mock_session, Job.model_validate(JOB))

        future.unit(0).wait()

        assert recorder.unit_reads(0) == 3

    def test_wait_times_out_on_an_unfinished_unit(self, mock_session):
        future = _future_in(mock_session, "PENDING")

        with pytest.raises(TimeoutException):
            future.unit(0).wait(timeout=0)

    def test_a_failed_unit_raises_job_failed_with_its_error_message(self, mock_session):
        future = _future_in(mock_session, "FAILED", error={"message": "out of memory"})

        with pytest.raises(JobFailedException) as raised:
            future.unit(0).wait()

        assert raised.value.job_id == "job-1"
        assert raised.value.failure_message == "unit 0 failed: out of memory"

    def test_a_failed_unit_without_a_message_raises_job_failed(self, mock_session):
        future = _future_in(mock_session, "FAILED")

        with pytest.raises(JobFailedException) as raised:
            future.unit(1).wait()

        assert raised.value.job_id == "job-1"
        assert raised.value.failure_message == "unit 1 failed"

    def test_a_canceled_unit_raises_job_failed(self, mock_session):
        future = _future_in(mock_session, "CANCELED")

        with pytest.raises(JobFailedException) as raised:
            future.unit(0).wait()

        assert raised.value.job_id == "job-1"
        assert raised.value.failure_message == "unit 0 was canceled"


class TestFailedPosition:
    """A position whose entry reports an error and no value raises on every read."""

    @pytest.fixture
    def failed(self, mock_session) -> Future:
        def handler(path: str, **kwargs: Any) -> MagicMock:
            response = MagicMock(status_code=200)
            if path == DEFAULT:
                response.json.return_value = {"values": [], "output": "embeddings"}
            elif path == f"{DEFAULT}/embeddings":
                response.json.return_value = {
                    "values": [{"position": 0, "unit_id": "u0", "error": "boom"}],
                    "next_cursor": None,
                }
            elif path == f"{DEFAULT}/embeddings/0":
                response.json.return_value = {
                    "position": 0,
                    "unit_id": "u0",
                    "error": "boom",
                }
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler
        return Future(mock_session, Job.model_validate(JOB))

    @pytest.mark.parametrize(
        "read",
        [
            lambda future: future.get_item(0),
            lambda future: list(future.stream()),
            lambda future: future.unit(0).get(),
            lambda future: future.get(),
        ],
        ids=["get_item", "stream", "unit.get", "get"],
    )
    def test_a_read_raises_job_failed_with_the_job_id(self, failed, read):
        with pytest.raises(JobFailedException) as raised:
            read(failed)

        assert raised.value.job_id == "job-1"
        assert raised.value.failure_message == "output at position 0 failed: boom"


class TestSurface:
    def test_a_unit_exposes_its_protocol_fields(self, future):
        unit = future.unit(0)

        for name in (
            "job_id",
            "position",
            "output",
            "unit_id",
        ):
            assert hasattr(unit, name), name

    def test_a_unit_has_no_state(self):
        assert not hasattr(FutureUnit, "state")

    def test_a_future_has_no_iter_units(self):
        assert not hasattr(Future, "iter_units")

    @pytest.mark.parametrize("name", ["Unit", "UnitState"])
    def test_the_jobs_package_has_no_unit_record(self, name):
        assert not hasattr(jobs_package, name)
        assert name not in jobs_package.__all__


def test_a_sampling_job_reads_its_prompt_without_the_model_listing(mock_session):
    """The sampling provider is not in the model listing; reads must not need it."""

    def handler(path: str, **kwargs: Any) -> MagicMock:
        response = MagicMock()
        if path == "v1/data/results/sample-1/prompt/0":
            response.json.return_value = {
                "position": 0,
                "unit_id": "u0",
                "state": "SUCCESS",
                "value": {"inline": "prompt"},
            }
        elif path == "v1/jobs/sample-1/units/0":
            response.json.return_value = _unit_record(0, "SUCCESS")
        else:
            raise AssertionError(f"unexpected GET {path}")
        return response

    mock_session.get.side_effect = handler
    job = Job.create(
        JOB
        | {
            "job_id": "sample-1",
            "job_type": "/prompt/sample_msa",
        }
    )
    assert isinstance(job, PromptJob)
    unit = Future(mock_session, job).unit(0, output="prompt")

    assert unit.wait(timeout=0) == "prompt"
    assert all(
        not c.args[0].startswith("v1/models") for c in mock_session.get.call_args_list
    )


def test_a_job_outside_the_models_api_has_no_outputs(mock_session):
    job = Job.create(JOB | {"job_type": "/align/prompt"})
    future = Future(mock_session, job)

    for read in (lambda: future.units, lambda: future.unit(0)):
        with pytest.raises(APIError, match="has no stored outputs"):
            read()
    mock_session.get.assert_not_called()


def test_a_unit_is_built_from_a_future_a_position_and_an_output():
    parameters = list(inspect.signature(FutureUnit).parameters)

    assert parameters == ["future", "position", "output"]
