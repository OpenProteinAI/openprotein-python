"""Stored prompts through ``session.data.Prompt``."""

import inspect
import io
import zipfile
from unittest.mock import MagicMock

import pytest

from openprotein.data import DataAPI, Prompt
from openprotein.data.prompts import PromptSummary
from openprotein.errors import HTTPError, JobOutputsNotReady
from openprotein.jobs.schemas import JobStatus
from openprotein.prompt.schemas import PromptMetadata

SUMMARY = {
    "id": "p1",
    "owner": "user-1",
    "digest": "d" * 64,
    "name": "n",
    "description": None,
    "created_at": "2026-07-20T00:00:00Z",
}
SAMPLED = PromptMetadata(
    id="s1",
    name="prompt-abc",
    description=None,
    created_date="2026-09-05T08:00:00Z",
    num_replicates=1,
    job_id="s1",
    status=JobStatus.SUCCESS,
)


def _zip(*replicates: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for position, content in enumerate(replicates):
            archive.writestr(f"replicate-{position}.fasta", content)
    return buffer.getvalue()


def _response(payload=None, content=b"", status=200):
    response = MagicMock(status_code=status, content=content)
    response.json.return_value = payload
    return response


def _not_found() -> HTTPError:
    return HTTPError(_response({"detail": "prompt not found"}, status=404))


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def prompt(session) -> Prompt:
    session.get.return_value = _response(
        content=_zip(b">a\nACDE\n>b\nAC:DE\n", b">a\nMKVL\n")
    )
    return Prompt(session, PromptSummary.model_validate(SUMMARY))


def test_session_data_hands_out_one_bound_prompt_class(session):
    data = DataAPI(session)
    assert data.Prompt is data.Prompt
    assert issubclass(data.Prompt, Prompt)


def test_create_uploads_one_file_per_replicate(session):
    session.post.return_value = _response(SUMMARY)
    prompt = DataAPI(session).Prompt.create([">a\nACDE\n", b">a\nMKVL\n"], name="n")
    assert isinstance(prompt, Prompt) and prompt.id == "p1"
    _, kwargs = session.post.call_args
    assert kwargs["data"] == {"name": "n"}
    assert kwargs["files"] == [
        ("replicates", ("replicate-0.fasta", b">a\nACDE\n")),
        ("replicates", ("replicate-1.fasta", b">a\nMKVL\n")),
    ]


@pytest.mark.parametrize("contents", [">a\nAC\n", b">a\nAC\n"], ids=["str", "bytes"])
def test_create_refuses_a_bare_file_before_any_request(session, contents):
    with pytest.raises(TypeError, match=r"create\(\[contents\]\)"):
        DataAPI(session).Prompt.create(contents)
    assert session.mock_calls == []


def test_list_follows_every_page_up_to_the_limit(session):
    rows = [SUMMARY | {"id": f"p{i}"} for i in range(3)]
    session.get.side_effect = [
        _response({"items": rows[:2], "next": "cursor"}),
        _response({"items": rows[2:], "next": None}),
    ]
    assert [p.id for p in DataAPI(session).Prompt.list()] == ["p0", "p1", "p2"]
    assert session.get.call_args_list[1].kwargs["params"] == {
        "after": "cursor",
        "limit": 200,
    }
    assert {call.args[0] for call in session.get.call_args_list} == {"v1/data/prompts"}


def test_list_asks_for_at_most_the_routes_page_limit(session):
    rows = [SUMMARY | {"id": f"p{i}"} for i in range(201)]
    session.get.side_effect = [
        _response({"items": rows[:200], "next": "cursor"}),
        _response({"items": rows[200:], "next": "more"}),
    ]

    prompts = DataAPI(session).Prompt.list(limit=201)

    assert len(prompts) == 201
    assert [call.kwargs["params"] for call in session.get.call_args_list] == [
        {"limit": 200},
        {"after": "cursor", "limit": 1},
    ]


def test_list_takes_only_a_limit():
    assert list(inspect.signature(Prompt.list).parameters) == ["limit"]


def test_id_loads_a_stored_prompt(session):
    session.get.return_value = _response(SUMMARY)
    prompt = DataAPI(session).Prompt(id="p1")
    assert (prompt.id, prompt.name, prompt.owner) == ("p1", "n", "user-1")
    assert (prompt.status, prompt.job_id) == (JobStatus.SUCCESS, None)
    session.get.assert_called_once_with("v1/data/prompts/p1")


def test_an_unknown_id_raises_the_404(session):
    session.get.side_effect = _not_found()
    with pytest.raises(HTTPError) as error:
        DataAPI(session).Prompt(id="nope")
    assert error.value.status_code == 404
    session.get.assert_called_once_with("v1/data/prompts/nope")


def test_reading_a_prompt_the_data_routes_do_not_hold_raises_the_404(session):
    session.get.side_effect = _not_found()
    prompt = Prompt(session, PromptSummary(id="s1", status=JobStatus.SUCCESS))
    with pytest.raises(HTTPError) as error:
        list(prompt)
    assert error.value.status_code == 404
    session.get.assert_called_once_with("v1/data/prompts/s1/download")


def test_iterating_yields_each_replicate(prompt, session):
    assert list(prompt) == [["ACDE", "AC:DE"], ["MKVL"]]
    session.get.assert_called_once_with("v1/data/prompts/p1/download")


def test_len_and_indexing_read_the_replicates(prompt):
    assert len(prompt) == 2
    assert prompt[1] == ["MKVL"]
    assert prompt[:1] == [["ACDE", "AC:DE"]]


def test_replicates_are_read_once_per_handle(prompt, session):
    assert list(prompt) == list(prompt)
    assert session.get.call_count == 1


@pytest.mark.parametrize("read", [list, len, lambda p: p[0]])
def test_reading_a_pending_prompt_names_its_status(session, read):
    pending = SUMMARY | {"status": "PENDING", "job_id": "p1"}
    session.get.return_value = _response(pending)
    prompt = Prompt(session, PromptSummary.model_validate(pending))
    with pytest.raises(JobOutputsNotReady, match="Prompt p1 is PENDING"):
        read(prompt)
    session.get.assert_called_once_with("v1/data/prompts/p1")


def test_update_sends_only_the_given_fields_and_returns_the_handle(prompt, session):
    session.patch.return_value = _response(SUMMARY | {"name": "new"})
    assert prompt.update(name="new") is prompt
    assert session.patch.call_args.kwargs["json"] == {"name": "new"}
    assert prompt.name == "new"


def test_delete_deletes_the_row(prompt, session):
    prompt.delete()
    session.delete.assert_called_once_with("v1/data/prompts/p1")


@pytest.mark.parametrize("name", ["download", "replicates", "record", "upload"])
def test_a_prompt_has_no_download_replicates_record_or_upload(name):
    assert not hasattr(Prompt, name)
    assert not hasattr(Prompt(MagicMock(), PromptSummary(id="p1")), name)
