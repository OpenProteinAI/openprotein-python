import pytest
from pydantic import ValidationError

from openprotein.jobs import JobStatus
from openprotein.jobs.schemas import _UnitRecord


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("PENDING", JobStatus.PENDING),
        ("READY", JobStatus.PENDING),
        ("RUNNING", JobStatus.RUNNING),
        ("SUCCESS", JobStatus.SUCCESS),
        ("FAILED", JobStatus.FAILURE),
        ("CANCELED", JobStatus.CANCELED),
    ],
)
def test_a_record_reads_its_state_as_a_job_status(state, status):
    record = _UnitRecord.model_validate(
        {"position": 0, "unit_id": "a" * 64, "state": state}
    )

    assert record.status is status


def test_a_record_with_a_status_key_reads_it_before_state():
    record = _UnitRecord.model_validate(
        {"position": 0, "unit_id": "u", "status": "FAILURE", "state": "SUCCESS"}
    )

    assert record.status is JobStatus.FAILURE


def test_a_record_holds_position_unit_id_status_and_error_only():
    record = _UnitRecord.model_validate(
        {
            "position": 3,
            "unit_id": "a" * 64,
            "state": "RUNNING",
            "retries": 1,
            "chunk_id": "c1",
            "cached": True,
            "error": {"message": "boom"},
        }
    )

    assert record.model_dump() == {
        "position": 3,
        "unit_id": "a" * 64,
        "status": JobStatus.RUNNING,
        "error": {"message": "boom"},
    }


def test_a_record_without_an_error_has_none():
    record = _UnitRecord.model_validate(
        {"position": 0, "unit_id": "u", "state": "SUCCESS"}
    )

    assert record.error is None


@pytest.mark.parametrize(
    "record",
    [
        {"position": 0, "unit_id": "u"},
        {"position": 0, "unit_id": "u", "state": "UNKNOWN"},
    ],
)
def test_a_record_needs_a_known_status(record):
    with pytest.raises(ValidationError):
        _UnitRecord.model_validate(record)


def test_a_null_status_reads_the_state_key():
    record = _UnitRecord.model_validate(
        {"position": 0, "unit_id": "u", "status": None, "state": "RUNNING"}
    )

    assert record.status is JobStatus.RUNNING
