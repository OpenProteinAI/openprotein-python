"""Reads of stored rows and the jobs that operations submit."""

from unittest.mock import MagicMock

import pytest

from openprotein.data._reads import read
from openprotein.errors import HTTPError
from openprotein.jobs._api import operation_job

JOB = {
    "job_id": "job-1",
    "job_type": "/search/mmseqs2",
    "status": "PENDING",
    "created_date": "2026-09-05T08:00:00Z",
}


def _error(status: int) -> HTTPError:
    response = MagicMock()
    response.status_code = status
    return HTTPError(response)


def test_a_live_row_is_returned():
    session = MagicMock()
    assert read(session, "v1/data/msa/a", params={"format": "a3m"}) is (
        session.get.return_value
    )
    session.get.assert_called_once_with("v1/data/msa/a", params={"format": "a3m"})


@pytest.mark.parametrize("status", [404, 425])
def test_a_row_that_is_not_live_raises_a_404(status):
    session = MagicMock()
    original = _error(status)
    session.get.side_effect = original
    with pytest.raises(HTTPError) as raised:
        read(session, "v1/data/msa/a")
    assert raised.value.status_code == raised.value.response.status_code == 404
    assert original.response.status_code == status


@pytest.mark.parametrize("status", [403, 410, 500])
def test_other_errors_pass_through(status):
    session = MagicMock()
    session.get.side_effect = _error(status)
    with pytest.raises(HTTPError) as raised:
        read(session, "v1/data/msa/a")
    assert raised.value.status_code == status


def test_a_job_body_is_the_submitted_job():
    session = MagicMock()
    assert operation_job(session, JOB, "msa_id") == JOB
    session.get.assert_not_called()


def test_a_body_naming_only_an_id_reads_the_job():
    session = MagicMock()
    session.get.return_value.json.return_value = JOB
    job = operation_job(session, {"msa_id": "job-1", "state": "pending"}, "msa_id")
    assert job["job_id"] == "job-1"
    session.get.assert_called_once_with("v1/jobs/job-1")
