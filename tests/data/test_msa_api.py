"""Tests for the MSA data client."""

import inspect
from unittest.mock import MagicMock

import pytest

from openprotein.data.msa import MSA, MSAMetadata
from openprotein.data.msa._api import (
    MSAPage,
    msa_delete,
    msa_download,
    msa_get_metadata,
    msa_list,
    msa_post,
    msa_put,
)
from openprotein.errors import HTTPError
from openprotein.jobs.schemas import JobStatus


@pytest.fixture
def sample_metadata() -> dict:
    """A stored-MSA metadata record as returned."""
    return {
        "msa_id": "job123",
        "msa_hash": "deadbeef",
        "name": None,
        "description": None,
        "tags": [],
        "project_uuid": "project-1",
        "source_format": "a3m",
        "num_sequences": 3,
        "num_match_columns": 60,
        "has_insertions": False,
        "original_filename": "aln.a3m",
        "digest": "d" * 64,
        "created_at": "2026-07-20T00:00:00Z",
        "updated_at": "2026-07-20T00:00:00Z",
    }


@pytest.fixture
def search_metadata(sample_metadata) -> dict:
    """A search row: no uploaded original, so no msa_hash or filename."""
    return sample_metadata | {"msa_hash": None, "original_filename": None}


@pytest.fixture
def mock_session(sample_metadata) -> MagicMock:
    session = MagicMock()
    for verb in ("get", "post", "put"):
        getattr(session, verb).return_value.json.return_value = sample_metadata
    return session


def test_msa_post_sends_multipart_file_and_form_fields(mock_session):
    """The upload posts the file as multipart and format/name as form data."""
    meta = msa_post(mock_session, b">q\nACDE\n", format="a3m", name="n")
    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/data/msa"
    assert kwargs["data"] == {"format": "a3m", "name": "n"}
    filename, content = kwargs["files"]["file"]
    assert content == b">q\nACDE\n" and filename == "msa.a3m"
    assert isinstance(meta, MSAMetadata) and meta.msa_id == "job123"


def test_msa_post_omits_unset_optional_fields(mock_session):
    """Only the fields the caller set are sent (no null name/description)."""
    msa_post(mock_session, b">q\nACDE\n", format="fasta")
    assert mock_session.post.call_args.kwargs["data"] == {"format": "fasta"}


def test_msa_get_metadata_reads_metadata_endpoint(mock_session):
    """Metadata is read from the per-MSA metadata path."""
    msa_get_metadata(mock_session, "job123")
    assert mock_session.get.call_args[0][0] == "v1/data/msa/job123/metadata"


def test_msa_list_passes_pagination(mock_session, sample_metadata):
    """list omits limit by default and forwards offset/limit when set."""
    mock_session.get.return_value.json.return_value = {
        "items": [sample_metadata],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    page = msa_list(mock_session)
    assert mock_session.get.call_args[0][0] == "v1/data/msa"
    assert mock_session.get.call_args.kwargs["params"] == {"offset": 0}

    msa_list(mock_session, limit=50, offset=10)
    assert mock_session.get.call_args.kwargs["params"] == {"offset": 10, "limit": 50}
    assert page.total == 1 and page.items[0].msa_id == "job123"


def test_msa_download_defaults_to_stored_format(mock_session):
    """download with no format sends no format param and returns raw bytes."""
    mock_session.get.return_value.content = b"A3M-BYTES"
    assert msa_download(mock_session, "job123") == b"A3M-BYTES"
    args, kwargs = mock_session.get.call_args
    assert args[0] == "v1/data/msa/job123"
    assert kwargs["params"] is None


def test_msa_download_with_format_converts(mock_session):
    """download passes the requested format through as a query param."""
    mock_session.get.return_value.content = b"FASTA"
    msa_download(mock_session, "job123", format="fasta")
    assert mock_session.get.call_args.kwargs["params"] == {"format": "fasta"}


def test_msa_put_sends_only_the_given_fields(mock_session):
    msa_put(mock_session, "job123", name="new", tags=["x"])
    args, kwargs = mock_session.put.call_args
    assert args[0] == "v1/data/msa/job123"
    assert kwargs["json"] == {"name": "new", "tags": ["x"]}


def test_msa_delete_calls_delete_endpoint(mock_session):
    """delete issues a DELETE to the MSA path and returns None."""
    assert msa_delete(mock_session, "job123") is None
    assert mock_session.delete.call_args[0][0] == "v1/data/msa/job123"


def test_create_encodes_str_and_wraps_result(mock_session):
    """create encodes a str body to bytes and returns an MSA wrapper."""
    msa = MSA._bind(mock_session).create(">q\nACDE\n", format="a3m")
    _, kwargs = mock_session.post.call_args
    assert kwargs["files"]["file"][1] == b">q\nACDE\n"
    assert isinstance(msa, MSA) and msa.id == "job123"


def _page(items, total, offset=0, limit=50):
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def test_list_follows_every_page(mock_session, sample_metadata):
    second = sample_metadata | {"msa_id": "job456"}
    mock_session.get.return_value.json.side_effect = [
        _page([sample_metadata], total=2),
        _page([second], total=2, offset=1),
    ]

    out = MSA._bind(mock_session).list()

    assert [msa.id for msa in out] == ["job123", "job456"]
    assert [c.kwargs["params"] for c in mock_session.get.call_args_list] == [
        {"offset": 0, "limit": 200},
        {"offset": 1, "limit": 200},
    ]


def test_list_stops_at_limit(mock_session, sample_metadata):
    second = sample_metadata | {"msa_id": "job456"}
    mock_session.get.return_value.json.side_effect = [
        _page([sample_metadata], total=3, limit=1),
        _page([second], total=3, offset=1, limit=1),
    ]

    out = MSA._bind(mock_session).list(limit=2)

    assert [msa.id for msa in out] == ["job123", "job456"]
    assert [c.kwargs["params"] for c in mock_session.get.call_args_list] == [
        {"offset": 0, "limit": 2},
        {"offset": 1, "limit": 1},
    ]


def test_msa_wrapper_methods_route_to_its_id(mock_session, sample_metadata):
    """MSA.to_string/update/delete act on the wrapped msa_id."""
    mock_session.get.return_value.content = b"body"
    msa = MSA(mock_session, MSAMetadata.model_validate(sample_metadata))

    assert msa.to_string() == "body"
    assert mock_session.get.call_args[0][0] == "v1/data/msa/job123"

    msa.update(description="d")
    assert mock_session.put.call_args.kwargs["json"] == {"description": "d"}

    msa.delete()
    assert mock_session.delete.call_args[0][0] == "v1/data/msa/job123"


def test_search_rows_read_with_null_hash_and_filename(search_metadata):
    meta = MSAMetadata.model_validate(search_metadata)
    assert meta.msa_hash is None and meta.original_filename is None
    assert meta.digest == "d" * 64


def test_session_data_reaches_msas_only_through_its_class():
    from openprotein.data import DataAPI

    assert not hasattr(DataAPI, "msa")
    assert not hasattr(DataAPI, "msas")


def _error(status: int) -> HTTPError:
    response = MagicMock()
    response.status_code = status
    return HTTPError(response)


@pytest.fixture
def pending_metadata(sample_metadata) -> dict:
    """A row whose search hasn't finished: no content, so no stats."""
    return sample_metadata | {
        "status": "RUNNING",
        "job_id": "job123",
        "msa_hash": None,
        "source_format": None,
        "num_sequences": None,
        "num_match_columns": None,
        "has_insertions": None,
        "original_filename": None,
        "digest": None,
    }


def test_a_pending_row_parses_with_its_status_and_job(mock_session, pending_metadata):
    msa = MSA(mock_session, MSAMetadata.model_validate(pending_metadata))
    assert (msa.status, msa.job_id) == (JobStatus.RUNNING, "job123")
    assert (msa.source_format, msa.num_sequences, msa.has_insertions) == (
        None,
        None,
        None,
    )


def test_a_row_without_a_status_is_ready(sample_metadata):
    meta = MSAMetadata.model_validate(sample_metadata)
    assert (meta.status, meta.job_id) == (JobStatus.SUCCESS, None)


def test_a_page_mixing_pending_and_ready_rows_parses(
    mock_session, sample_metadata, pending_metadata
):
    mock_session.get.return_value.json.return_value = {
        "items": [pending_metadata, sample_metadata],
        "total": 2,
        "limit": 50,
        "offset": 0,
    }
    page = msa_list(mock_session)
    assert isinstance(page, MSAPage)
    assert [item.status for item in page.items] == [
        JobStatus.RUNNING,
        JobStatus.SUCCESS,
    ]


def test_get_returns_a_pending_row(mock_session, pending_metadata):
    mock_session.get.return_value.json.return_value = pending_metadata
    msa = MSA._bind(mock_session)(id="job123")
    assert msa.status == JobStatus.RUNNING
    mock_session.get.assert_called_once_with("v1/data/msa/job123/metadata")


@pytest.mark.parametrize("status", [404, 425])
def test_get_raises_a_404_for_a_404_or_425(mock_session, status):
    mock_session.get.side_effect = _error(status)
    with pytest.raises(HTTPError) as raised:
        MSA._bind(mock_session)(id="job123")
    assert raised.value.status_code == 404
    assert mock_session.get.call_count == 1


def test_get_returns_a_live_row(mock_session):
    msa = MSA._bind(mock_session)(id="job123")
    assert msa.metadata.msa_hash == "deadbeef"
    mock_session.get.assert_called_once_with("v1/data/msa/job123/metadata")


def test_download_reads_only_the_data_route(mock_session):
    mock_session.get.return_value.content = b"PAR1...PAR1"

    assert msa_download(mock_session, "job123", format="parquet") == b"PAR1...PAR1"
    mock_session.get.assert_called_once_with(
        "v1/data/msa/job123", params={"format": "parquet"}
    )


def test_an_msa_names_its_row_by_id_and_digest(mock_session, sample_metadata):
    msa = MSA(mock_session, MSAMetadata.model_validate(sample_metadata))
    assert (msa.id, msa.digest) == ("job123", "d" * 64)
    assert msa.metadata.msa_hash == "deadbeef"
    assert not any(hasattr(msa, n) for n in ("msa_id", "msa_hash", "project_uuid"))


def test_update_takes_name_description_and_tags():
    params = list(inspect.signature(MSA.update).parameters)
    assert params[-3:] == ["name", "description", "tags"]
