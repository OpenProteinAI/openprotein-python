"""Stored rows answer the ``Future`` calls, through the job that made them."""

from unittest.mock import MagicMock

import pytest

from openprotein.data import MSA, Prompt
from openprotein.data.msa import MSAMetadata
from openprotein.data.prompts import PromptSummary
from openprotein.errors import (
    HTTPError,
    JobFailedException,
    JobOutputsNotReady,
    NoJobError,
)
from openprotein.jobs import Future, Job, JobStatus

PROMPT_ID = "e0b1c3d4-0000-4000-8000-000000000002"

JOB = {
    "job_id": PROMPT_ID,
    "job_type": "/prompt/sample_msa",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
    "start_date": "2026-09-05T08:00:01Z",
    "end_date": "2026-09-05T08:00:09Z",
}


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def _route(session: MagicMock, routes: dict[str, list[dict]]) -> None:
    """Answer each GET by its path, in order; the last answer repeats."""

    def get(path, **kwargs):
        answers = routes[path]
        return _response(answers.pop(0) if len(answers) > 1 else answers[0])

    session.get.side_effect = get


def _sampled(session: MagicMock, status: str = "SUCCESS") -> Prompt:
    return Prompt(
        session,
        PromptSummary(id=PROMPT_ID, status=JobStatus(status), job_id=PROMPT_ID),
    )


def _uploaded_msa(session: MagicMock) -> MSA:
    return MSA(
        session,
        MSAMetadata(
            msa_id="m1",
            source_format="a3m",
            num_sequences=2,
            num_match_columns=5,
            has_insertions=False,
            created_at="2026-09-05T08:00:00Z",
            updated_at="2026-09-05T08:00:00Z",
        ),
    )


def _missing_row(session: MagicMock, job: dict) -> None:
    def get(path, **kwargs):
        if path == f"v1/jobs/{PROMPT_ID}":
            return _response(job)
        raise HTTPError(MagicMock(status_code=404))

    session.get.side_effect = get


def test_a_finished_sampling_whose_prompt_is_gone_is_not_ready(mock_session):
    _missing_row(mock_session, JOB)

    prompt = Future.create(mock_session, job=Job.create(JOB))

    assert isinstance(prompt, Prompt) and prompt.status is JobStatus.SUCCESS
    for read in (prompt.get, prompt.wait, lambda: list(prompt)):
        with pytest.raises(HTTPError) as caught:
            read()
        assert caught.value.status_code == 404


@pytest.mark.parametrize(
    ("status", "message"),
    [("FAILURE", "sampler crashed"), ("CANCELED", "the job was canceled")],
)
def test_an_unsuccessful_sampling_whose_prompt_is_gone_raises_its_failure(
    mock_session, status, message
):
    job = JOB | {"status": status, "failure_message": "sampler crashed"}
    _missing_row(mock_session, job)

    prompt = Future.create(mock_session, job=Job.create(job))

    for read in (prompt.get, prompt.wait):
        with pytest.raises(JobFailedException, match=message):
            read()


def _job_gone(session: MagicMock, code: int, rows: list[dict]) -> None:
    def get(path, **kwargs):
        if path == f"v1/jobs/{PROMPT_ID}":
            raise HTTPError(MagicMock(status_code=code))
        return _response(rows.pop(0) if len(rows) > 1 else rows[0])

    session.get.side_effect = get


@pytest.mark.parametrize("code", [404, 410])
def test_a_failed_row_whose_job_is_gone_raises_the_failure(mock_session, code):
    failed = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "FAILURE"}
    _job_gone(mock_session, code, [failed])
    prompt = _sampled(mock_session, "FAILURE")

    for read in (prompt.get, prompt.wait):
        with pytest.raises(JobFailedException):
            read()


@pytest.mark.parametrize("code", [404, 410])
def test_waiting_on_a_row_whose_job_is_gone_reads_the_row(mock_session, code):
    failed = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "FAILURE"}
    _job_gone(mock_session, code, [failed])
    prompt = _sampled(mock_session, "RUNNING")

    assert prompt.wait_until_done() is True
    assert prompt.status is JobStatus.FAILURE


def test_a_failed_row_raises_other_job_read_errors(mock_session):
    failed = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "FAILURE"}
    _job_gone(mock_session, 500, [failed])

    with pytest.raises(HTTPError) as caught:
        _sampled(mock_session, "FAILURE").get()
    assert caught.value.status_code == 500


def test_a_sampled_prompt_reads_its_job_once(mock_session):
    _route(mock_session, {f"v1/jobs/{PROMPT_ID}": [JOB]})
    prompt = _sampled(mock_session)

    assert prompt.job is not None and prompt.job.job_id == PROMPT_ID
    assert prompt.job_type == "/prompt/sample_msa"
    assert prompt.created_date.year == 2026
    assert prompt.start_date is not None and prompt.end_date is not None
    assert mock_session.get.call_count == 1


def test_refresh_reads_the_job_again(mock_session):
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "SUCCESS"}
    _route(
        mock_session,
        {f"v1/jobs/{PROMPT_ID}": [JOB], f"v1/data/prompts/{PROMPT_ID}": [stored]},
    )
    prompt = _sampled(mock_session)
    assert prompt.job is not None

    prompt.refresh()
    assert prompt.job is not None

    paths = [call.args[0] for call in mock_session.get.call_args_list]
    assert paths.count(f"v1/jobs/{PROMPT_ID}") == 2


@pytest.mark.parametrize(
    "field", ["job_type", "created_date", "start_date", "end_date"]
)
def test_an_upload_has_no_job_fields(mock_session, field):
    msa = _uploaded_msa(mock_session)

    assert msa.job is None
    with pytest.raises(NoJobError, match="m1 was not made by a job"):
        getattr(msa, field)
    assert not hasattr(msa, field)
    mock_session.get.assert_not_called()


def test_an_upload_is_done_and_gets_itself(mock_session):
    msa = _uploaded_msa(mock_session)

    assert msa.done() is True
    assert msa.wait_until_done() is True
    assert msa.get() is msa
    assert msa.wait() is msa
    mock_session.get.assert_not_called()


@pytest.mark.parametrize(
    ("status", "done"),
    [("PENDING", False), ("RUNNING", False), ("SUCCESS", True), ("FAILURE", True)],
)
def test_done_reads_the_status_it_has(mock_session, status, done):
    assert _sampled(mock_session, status).done() is done
    mock_session.get.assert_not_called()


def test_wait_until_done_polls_the_job_and_reloads(mock_session, monkeypatch):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "SUCCESS"}
    _route(
        mock_session,
        {
            f"v1/jobs/{PROMPT_ID}": [JOB | {"status": "RUNNING"}, JOB],
            f"v1/data/prompts/{PROMPT_ID}": [stored],
        },
    )
    prompt = _sampled(mock_session, "PENDING")

    assert prompt.wait_until_done() is True
    assert prompt.status is JobStatus.SUCCESS
    assert prompt.job is not None and prompt.job.status is JobStatus.SUCCESS


def test_wait_until_done_returns_on_a_failure_and_get_raises_it(
    mock_session, monkeypatch
):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    failed = JOB | {"status": "FAILURE", "failure_message": "sampler crashed"}
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "FAILURE"}
    _route(
        mock_session,
        {
            f"v1/jobs/{PROMPT_ID}": [failed],
            f"v1/data/prompts/{PROMPT_ID}": [stored],
        },
    )
    prompt = _sampled(mock_session, "PENDING")

    assert prompt.wait_until_done() is True
    with pytest.raises(JobFailedException, match="sampler crashed"):
        prompt.get()
    with pytest.raises(JobFailedException, match="sampler crashed"):
        prompt.wait()


def test_get_returns_the_ready_handle(mock_session):
    prompt = _sampled(mock_session)
    assert prompt.get() is prompt
    mock_session.get.assert_not_called()


def test_get_on_a_row_that_is_not_ready_raises(mock_session):
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "RUNNING"}
    _route(mock_session, {f"v1/data/prompts/{PROMPT_ID}": [stored]})

    with pytest.raises(JobOutputsNotReady, match="is RUNNING"):
        _sampled(mock_session, "RUNNING").get()


def test_wait_returns_get(mock_session, monkeypatch):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "SUCCESS"}
    _route(
        mock_session,
        {f"v1/jobs/{PROMPT_ID}": [JOB], f"v1/data/prompts/{PROMPT_ID}": [stored]},
    )
    prompt = _sampled(mock_session, "PENDING")

    assert prompt.wait() is prompt
    assert prompt.status is JobStatus.SUCCESS
