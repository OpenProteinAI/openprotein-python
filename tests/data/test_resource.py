"""The behaviour every stored-data handle shares."""

import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from openprotein.data._resource import _Resource
from openprotein.errors import JobFailedException, JobOutputsNotReady, TimeoutException
from openprotein.jobs.schemas import JobStatus


@dataclass
class _Row:
    id: str
    status: JobStatus = JobStatus.SUCCESS
    job_id: str | None = None


class _Thing(_Resource[_Row]):
    """A stored thing."""

    loads: ClassVar[list[_Row]] = []

    @classmethod
    def _load(cls, session, id):
        return cls.loads.pop(0)

    @property
    def id(self):
        return self.metadata.id

    @property
    def status(self):
        return self.metadata.status

    @property
    def job_id(self):
        return self.metadata.job_id


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)


def _job(status: str, **extra) -> dict:
    return {
        "job_id": "j1",
        "job_type": "/align/align",
        "status": status,
        "created_date": "2026-09-05T08:00:00Z",
        **extra,
    }


def _polls(session: MagicMock, *jobs: dict) -> None:
    session.get.side_effect = [MagicMock(json=MagicMock(return_value=j)) for j in jobs]


def test_a_bound_class_keeps_its_name_and_documentation():
    bound = _Thing._bind(MagicMock())
    assert issubclass(bound, _Thing)
    assert (bound.__name__, bound.__qualname__, bound.__doc__) == (
        "_Thing",
        _Thing.__qualname__,
        "A stored thing.",
    )


def test_id_loads_through_the_bound_session():
    session = MagicMock()
    _Thing.loads = [_Row("a")]
    thing = _Thing._bind(session)(id="a")
    assert (thing.id, thing.session) == ("a", session)
    assert isinstance(thing, _Thing)


def test_id_on_an_unbound_class_names_the_session_accessor():
    with pytest.raises(TypeError, match=r"use session\.data\._Thing"):
        _Thing(id="a")


def test_a_session_and_metadata_construct_without_a_request():
    session = MagicMock()
    thing = _Thing(session, _Row("a"))
    assert thing.metadata == _Row("a")
    session.get.assert_not_called()


@pytest.mark.parametrize("args", [(), (MagicMock(),)])
def test_construction_needs_an_id_or_a_session_with_metadata(args):
    with pytest.raises(TypeError, match="takes id=, or a session and its metadata"):
        _Thing(*args)


def test_id_with_metadata_is_refused():
    with pytest.raises(TypeError, match="pass id alone"):
        _Thing(MagicMock(), _Row("a"), id="a")


def test_a_handle_is_truthy():
    assert _Thing(MagicMock(), _Row("a"))


def test_refresh_reads_the_metadata_again_and_returns_the_handle():
    thing = _Thing(MagicMock(), _Row("a", JobStatus.PENDING, "j1"))
    _Thing.loads = [_Row("a", JobStatus.SUCCESS, "j1")]
    assert thing.refresh() is thing
    assert thing.status is JobStatus.SUCCESS


def test_wait_on_a_ready_row_makes_no_request():
    session = MagicMock()
    thing = _Thing(session, _Row("a"))
    assert thing.wait() is thing
    session.get.assert_not_called()


def test_wait_polls_the_job_then_reads_the_metadata():
    session = MagicMock()
    _polls(session, _job("RUNNING"), _job("RUNNING"), _job("SUCCESS"))
    _Thing.loads = [_Row("a", JobStatus.SUCCESS, "j1")]
    thing = _Thing(session, _Row("a", JobStatus.PENDING, "j1"))
    assert thing.wait() is thing
    assert thing.status is JobStatus.SUCCESS
    assert [call.args[0] for call in session.get.call_args_list] == ["v1/jobs/j1"] * 3


def test_wait_raises_the_failure_of_the_job():
    session = MagicMock()
    _polls(session, _job("FAILURE", failure_message="no hits"))
    _Thing.loads = [_Row("a", JobStatus.FAILURE, "j1")]
    thing = _Thing(session, _Row("a", JobStatus.PENDING, "j1"))
    with pytest.raises(JobFailedException, match="Job j1 failed: no hits"):
        thing.wait()


def test_wait_raises_for_a_canceled_job():
    session = MagicMock()
    _polls(session, _job("CANCELED"))
    _Thing.loads = [_Row("a", JobStatus.CANCELED, "j1")]
    thing = _Thing(session, _Row("a", JobStatus.PENDING, "j1"))
    with pytest.raises(JobFailedException, match="the job was canceled"):
        thing.wait()


def test_wait_raises_once_the_timeout_is_exceeded():
    session = MagicMock()
    session.timeout = 1
    session.get.return_value.json.return_value = _job("RUNNING")
    thing = _Thing(session, _Row("a", JobStatus.PENDING, "j1"))
    with pytest.raises(TimeoutException):
        thing.wait(timeout=0)


def test_content_of_a_row_that_is_not_ready_names_its_status():
    thing = _Thing(MagicMock(), _Row("a", JobStatus.PENDING, "j1"))
    _Thing.loads = [_Row("a", JobStatus.RUNNING, "j1")]
    with pytest.raises(JobOutputsNotReady, match="_Thing a is RUNNING"):
        thing._require_ready()


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (JobStatus.FAILURE, "Job j1 failed"),
        (JobStatus.CANCELED, "the job was canceled"),
    ],
)
def test_content_of_a_failed_or_canceled_row_raises_the_job_failure(status, message):
    thing = _Thing(MagicMock(), _Row("a", status, "j1"))
    _Thing.loads = [_Row("a", status, "j1")]
    with pytest.raises(JobFailedException, match=message):
        thing._require_ready()


def test_a_handle_from_a_bound_class_pickles():
    session = SimpleNamespace(name="session")
    _Thing.loads = [_Row("a")]
    thing = _Thing._bind(session)(id="a")  # ty: ignore[invalid-argument-type]
    restored = pickle.loads(pickle.dumps(thing))
    assert type(restored) is _Thing
    assert restored.metadata == _Row("a")


def test_a_pickled_handle_keeps_content_read_earlier():
    thing = _Thing(SimpleNamespace(name="session"), _Row("a"))  # ty: ignore[invalid-argument-type]
    vars(thing)["cached"] = ["row"]
    assert vars(pickle.loads(pickle.dumps(thing)))["cached"] == ["row"]
