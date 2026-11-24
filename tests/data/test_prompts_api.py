"""Tests for stored prompts (``session.data.Prompt``) and their wire calls."""

from unittest.mock import MagicMock

import pytest

from openprotein.data.prompts import Prompt, PromptRecord, PromptSummary
from openprotein.data.prompts._api import (
    prompts_delete,
    prompts_download,
    prompts_get,
    prompts_list,
    prompts_post,
    prompts_put,
)
from openprotein.errors import HTTPError
from openprotein.jobs.schemas import JobStatus


@pytest.fixture
def sample_summary() -> dict:
    return {
        "id": "prompt123",
        "owner": "user-1",
        "name": None,
        "description": None,
        "created_at": "2026-07-20T00:00:00Z",
    }


@pytest.fixture
def sample_record(sample_summary) -> dict:
    return {
        **sample_summary,
        "value": [
            {
                "position": 0,
                "value": {
                    "ref": {"download_url": "https://example.test/uploads/deadbeef"}
                },
            }
        ],
    }


@pytest.fixture
def mock_session(sample_record) -> MagicMock:
    session = MagicMock()
    for verb in ("get", "post", "patch"):
        getattr(session, verb).return_value.json.return_value = sample_record
    return session


def _zip_of(replicates: list[bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for position, content in enumerate(replicates):
            archive.writestr(f"replicate-{position}.fasta", content)
    return buf.getvalue()


def test_prompts_list_passes_pagination(mock_session, sample_summary):
    mock_session.get.return_value.json.return_value = {
        "items": [sample_summary],
        "next": None,
    }
    page = prompts_list(mock_session)
    assert mock_session.get.call_args[0][0] == "v1/data/prompts"
    assert mock_session.get.call_args.kwargs["params"] is None

    prompts_list(mock_session, after="cursor-1", limit=10)
    assert mock_session.get.call_args.kwargs["params"] == {
        "after": "cursor-1",
        "limit": 10,
    }
    assert page.items[0].id == "prompt123"


def test_prompts_get_reads_the_row(mock_session):
    record = prompts_get(mock_session, "prompt123")
    assert mock_session.get.call_args[0][0] == "v1/data/prompts/prompt123"
    assert isinstance(record, PromptRecord)
    assert record.value[0].position == 0


def test_prompts_put_sends_only_changed_fields(mock_session):
    prompts_put(mock_session, "prompt123", name="new")
    args, kwargs = mock_session.patch.call_args
    assert args[0] == "v1/data/prompts/prompt123"
    assert kwargs["json"] == {"name": "new"}


def test_prompts_delete_calls_delete_endpoint(mock_session):
    assert prompts_delete(mock_session, "prompt123") is None
    assert mock_session.delete.call_args[0][0] == "v1/data/prompts/prompt123"


def test_prompts_post_sends_one_file_per_replicate_under_one_field(mock_session):
    prompts_post(mock_session, [b">c1\nACDE\n", b">c1\nMKVL\n"], name="n")
    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/data/prompts"
    assert kwargs["data"] == {"name": "n"}
    assert [field for field, _ in kwargs["files"]] == ["replicates", "replicates"]
    names_and_contents = [value for _, value in kwargs["files"]]
    assert names_and_contents == [
        ("replicate-0.fasta", b">c1\nACDE\n"),
        ("replicate-1.fasta", b">c1\nMKVL\n"),
    ]


def test_prompts_download_reads_the_zip(mock_session):
    payload = _zip_of([b">c1\nACDE\n"])
    mock_session.get.return_value.content = payload
    assert prompts_download(mock_session, "prompt123") == payload
    assert mock_session.get.call_args[0][0] == "v1/data/prompts/prompt123/download"


def _http_error(status_code: int, payload: dict) -> HTTPError:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return HTTPError(response)


def test_prompts_get_reraises_other_statuses(mock_session):
    mock_session.get.side_effect = _http_error(404, {})

    with pytest.raises(HTTPError):
        prompts_get(mock_session, "prompt123")


def test_upload_encodes_str_and_wraps_result(mock_session):
    api = Prompt._bind(mock_session)
    prompt = api.create([">c1\nACDE\n"])
    _, kwargs = mock_session.post.call_args
    assert kwargs["files"][0][1] == ("replicate-0.fasta", b">c1\nACDE\n")
    assert isinstance(prompt, Prompt) and prompt.id == "prompt123"


def test_list_wraps_items(mock_session, sample_summary):
    mock_session.get.return_value.json.return_value = {
        "items": [sample_summary],
        "next": None,
    }
    api = Prompt._bind(mock_session)
    out = api.list()
    assert len(out) == 1 and isinstance(out[0], Prompt)
    assert out[0].owner == "user-1"


def test_list_follows_next_across_two_pages(mock_session, sample_summary):
    second_summary = {**sample_summary, "id": "prompt456"}
    mock_session.get.side_effect = [
        MagicMock(json=lambda: {"items": [sample_summary], "next": "cursor-1"}),
        MagicMock(json=lambda: {"items": [second_summary], "next": None}),
    ]
    api = Prompt._bind(mock_session)

    out = api.list()

    assert [prompt.id for prompt in out] == ["prompt123", "prompt456"]
    assert mock_session.get.call_args_list[0].kwargs["params"] == {"limit": 200}
    assert mock_session.get.call_args_list[1].kwargs["params"] == {
        "after": "cursor-1",
        "limit": 200,
    }


def test_list_stops_at_limit(mock_session, sample_summary):
    second_summary = {**sample_summary, "id": "prompt456"}
    mock_session.get.side_effect = [
        MagicMock(json=lambda: {"items": [sample_summary], "next": "cursor-1"}),
        MagicMock(json=lambda: {"items": [second_summary], "next": "cursor-2"}),
    ]

    out = Prompt._bind(mock_session).list(limit=2)

    assert [prompt.id for prompt in out] == ["prompt123", "prompt456"]
    assert [c.kwargs["params"] for c in mock_session.get.call_args_list] == [
        {"limit": 2},
        {"after": "cursor-1", "limit": 1},
    ]


def test_get_wraps_the_record(mock_session):
    api = Prompt._bind(mock_session)
    prompt = api(id="prompt123")
    assert isinstance(prompt, Prompt)
    assert prompt.id == "prompt123"


def test_prompt_wrapper_methods_route_to_its_id(mock_session, sample_record):
    payload = _zip_of([b">c1\nACDE\n", b">c1\nMKVL\n"])
    mock_session.get.return_value.content = payload
    prompt = Prompt(mock_session, PromptSummary.model_validate(sample_record))

    assert list(prompt) == [["ACDE"], ["MKVL"]]
    assert mock_session.get.call_args[0][0] == "v1/data/prompts/prompt123/download"

    prompt.update(description="d")
    assert mock_session.patch.call_args.kwargs["json"] == {"description": "d"}

    prompt.delete()
    assert mock_session.delete.call_args[0][0] == "v1/data/prompts/prompt123"


@pytest.mark.parametrize("status", [404, 425])
def test_prompts_get_raises_a_404_for_a_404_or_425(mock_session, status):
    mock_session.get.side_effect = _http_error(status, {})
    with pytest.raises(HTTPError) as raised:
        prompts_get(mock_session, "prompt123")
    assert raised.value.status_code == 404
    assert mock_session.get.call_count == 1


@pytest.fixture
def pending_summary(sample_summary) -> dict:
    return {**sample_summary, "status": "RUNNING", "job_id": "job1"}


def test_pending_summary_parses_with_its_job_status(mock_session, pending_summary):
    summary = PromptSummary.model_validate(pending_summary)
    prompt = Prompt(mock_session, summary)
    assert summary.status == JobStatus.RUNNING
    assert prompt.status == JobStatus.RUNNING
    assert prompt.job_id == "job1"


def test_summary_without_status_reads_success_and_no_job(mock_session, sample_summary):
    prompt = Prompt(mock_session, PromptSummary.model_validate(sample_summary))
    assert prompt.status == JobStatus.SUCCESS
    assert prompt.job_id is None


@pytest.mark.parametrize("status", ["FAILURE", "CANCELED"])
def test_summary_parses_terminal_job_statuses(sample_summary, status):
    summary = PromptSummary.model_validate({**sample_summary, "status": status})
    assert summary.status == JobStatus(status)


def test_prompts_list_returns_pending_and_ready_rows(
    mock_session, sample_summary, pending_summary
):
    mock_session.get.return_value.json.return_value = {
        "items": [pending_summary, {**sample_summary, "id": "prompt456"}],
        "next": None,
    }
    prompts = Prompt._bind(mock_session).list()
    assert [p.status for p in prompts] == [JobStatus.RUNNING, JobStatus.SUCCESS]
    assert [p.job_id for p in prompts] == ["job1", None]


def test_prompts_get_returns_a_pending_record(mock_session, pending_summary):
    mock_session.get.return_value.json.return_value = {
        **pending_summary,
        "value": None,
    }
    record = prompts_get(mock_session, "prompt123")
    assert record.status == JobStatus.RUNNING
    assert record.job_id == "job1"
    assert record.value is None


def test_prompts_download_raises_a_409_for_a_prompt_that_is_not_ready(mock_session):
    mock_session.get.side_effect = _http_error(409, {})
    with pytest.raises(HTTPError) as raised:
        prompts_download(mock_session, "prompt123")
    assert raised.value.status_code == 409


@pytest.mark.parametrize("contents", [">q\nAC\n", b">q\nAC\n"], ids=["str", "bytes"])
def test_upload_refuses_a_bare_file_before_any_request(mock_session, contents):
    api = Prompt._bind(mock_session)

    with pytest.raises(TypeError, match=r"create\(\[contents\]\)"):
        api.create(contents)

    assert mock_session.mock_calls == []


@pytest.mark.parametrize("contents", [">q\nAC\n", b">q\nAC\n"], ids=["str", "bytes"])
def test_upload_sends_one_file_for_a_one_entry_list(mock_session, contents):
    Prompt._bind(mock_session).create([contents])

    _, kwargs = mock_session.post.call_args
    assert [file[1] for file in kwargs["files"]] == [("replicate-0.fasta", b">q\nAC\n")]
