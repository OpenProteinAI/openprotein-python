"""Stored MSAs through ``session.data.MSA``."""

import inspect
from unittest.mock import MagicMock

import pytest

from openprotein.data import MSA, DataAPI
from openprotein.data.msa import MSAMetadata
from openprotein.errors import HTTPError, JobOutputsNotReady
from openprotein.jobs.schemas import JobStatus

METADATA = {
    "msa_id": "m1",
    "name": "n",
    "description": None,
    "tags": ["t"],
    "source_format": "a3m",
    "num_sequences": 3,
    "num_match_columns": 4,
    "has_insertions": True,
    "digest": "d" * 64,
    "created_at": "2026-07-20T00:00:00Z",
    "updated_at": "2026-07-21T00:00:00Z",
}
FASTA = b">query\nACDE\n>h1 desc\nAC-E\n>h2\n-CDE\n"
ROWS = [("query", "ACDE"), ("h1 desc", "AC-E"), ("h2", "-CDE")]


def _response(payload=None, content=b"", status=200):
    response = MagicMock(status_code=status, content=content)
    response.json.return_value = payload
    return response


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def msa(session) -> MSA:
    session.get.return_value = _response(content=FASTA)
    return MSA(session, MSAMetadata.model_validate(METADATA))


def test_session_data_hands_out_one_bound_msa_class(session):
    data = DataAPI(session)
    assert data.MSA is data.MSA
    assert issubclass(data.MSA, MSA)
    assert data.MSA._bound() is session


def test_create_uploads_and_returns_the_handle(session):
    session.post.return_value = _response(METADATA)
    msa = DataAPI(session).MSA.create(">q\nACDE\n", format="a3m", name="n")
    assert isinstance(msa, MSA) and msa.id == "m1"
    _, kwargs = session.post.call_args
    assert kwargs["data"] == {"format": "a3m", "name": "n"}
    assert kwargs["files"]["file"] == ("msa.a3m", b">q\nACDE\n")


def test_list_follows_every_page_up_to_the_limit(session):
    rows = [METADATA | {"msa_id": f"m{i}"} for i in range(3)]
    session.get.side_effect = [
        _response({"items": rows[:2], "total": 3, "limit": 2, "offset": 0}),
        _response({"items": rows[2:], "total": 3, "limit": 2, "offset": 2}),
    ]
    assert [m.id for m in DataAPI(session).MSA.list()] == ["m0", "m1", "m2"]
    assert [c.kwargs["params"]["offset"] for c in session.get.call_args_list] == [0, 2]


def test_list_asks_for_at_most_the_routes_page_limit(session):
    rows = [METADATA | {"msa_id": f"m{i}"} for i in range(201)]
    session.get.side_effect = [
        _response({"items": rows[:200], "total": 300, "limit": 200, "offset": 0}),
        _response({"items": rows[200:], "total": 300, "limit": 1, "offset": 200}),
    ]

    msas = DataAPI(session).MSA.list(limit=201)

    assert len(msas) == 201
    assert [call.kwargs["params"] for call in session.get.call_args_list] == [
        {"offset": 0, "limit": 200},
        {"offset": 200, "limit": 1},
    ]


def test_list_takes_only_a_limit():
    assert list(inspect.signature(MSA.list).parameters) == ["limit"]


def test_id_loads_one_msa(session):
    session.get.return_value = _response(METADATA)
    msa = DataAPI(session).MSA(id="m1")
    assert (msa.id, msa.name, msa.tags) == ("m1", "n", ["t"])
    session.get.assert_called_once_with("v1/data/msa/m1/metadata")


def test_an_unknown_id_raises_the_404(session):
    session.get.side_effect = HTTPError(_response(status=404))
    with pytest.raises(HTTPError) as error:
        DataAPI(session).MSA(id="nope")
    assert error.value.status_code == 404


def test_the_handle_names_the_shared_fields(msa):
    assert (msa.id, msa.name, msa.description) == ("m1", "n", None)
    assert (msa.status, msa.job_id) == (JobStatus.SUCCESS, None)
    assert msa.created_at.year == 2026 and msa.updated_at.day == 21
    assert msa.metadata.num_match_columns == 4


def test_len_is_the_sequence_count_without_a_request(msa, session):
    assert len(msa) == 3
    session.get.assert_not_called()


def test_an_a3m_msa_iterates_its_fasta_rendering(msa, session):
    assert list(msa) == ROWS
    session.get.assert_called_once_with("v1/data/msa/m1", params={"format": "fasta"})


def _stored(session, source_format, match_columns, *bodies):
    session.get.side_effect = [_response(content=body) for body in bodies]
    metadata = METADATA | {
        "source_format": source_format,
        "num_match_columns": match_columns,
    }
    return MSA(session, MSAMetadata.model_validate(metadata))


def _formats(session):
    return [call.kwargs["params"]["format"] for call in session.get.call_args_list]


def test_an_a2m_msa_iterates_its_match_columns(session):
    msa = _stored(session, "a2m", 4, b">query\nAC.DE\n>h1 desc\nACg-E\n>h2\n-C.DE\n")
    assert list(msa) == ROWS
    assert _formats(session) == ["a2m"]


def test_a_complex_keeps_its_chain_breaks_without_its_header(session):
    msa = _stored(session, "a2m", 5, b"#4,1\t1,1\n>q\nAC.DE:F\n>h\nACgDE:F\n")
    assert list(msa) == [("q", "ACDE:F"), ("h", "ACDE:F")]


def test_a_stockholm_msa_with_lowercase_inserts_iterates_its_match_columns(session):
    msa = _stored(session, "sto", 4, b">q\nAC..DE\n>s1\nACkkDE\n")
    assert list(msa) == [("q", "ACDE"), ("s1", "ACDE")]
    assert _formats(session) == ["a2m"]


def test_a_stockholm_msa_whose_inserts_are_unmarked_iterates_every_column(session):
    msa = _stored(session, "sto", 4, b">q\nACDE\n>s1\nAC.E\n", b">q\nACDE\n>s1\nAC-E\n")
    assert list(msa) == [("q", "ACDE"), ("s1", "AC-E")]
    assert _formats(session) == ["a2m", "fasta"]


@pytest.mark.parametrize(
    ("source_format", "match_columns", "a2m", "fasta"),
    [
        pytest.param(
            "sto",
            9,
            b">q\nMKV.LAAGIT\n>s1\nMKVQLA.GIT\n>s2\nMK.ALAAGIT\n",
            b">q\nMKV-LAAGIT\n>s1\nMKVQLA-GIT\n>s2\nMK-ALAAGIT\n",
            id="sto-dot-gaps",
        ),
        pytest.param(
            "sto",
            7,
            b">q\nMKVlaAGIT\n>s1\nMKvqLAGIT\n",
            b">q\nMKVLAAGIT\n>s1\nMKVQLAGIT\n",
            id="sto-soft-masked",
        ),
        pytest.param(
            "a2m",
            6,
            b">q\nACD.EFG\n>s1\nAC.DEFG\n",
            b">q\nACD-EFG\n>s1\nAC-DEFG\n",
            id="a2m-disagreeing-inserts",
        ),
    ],
)
def test_rows_that_mark_different_insert_columns_iterate_every_column(
    session, source_format, match_columns, a2m, fasta
):
    msa = _stored(session, source_format, match_columns, a2m, fasta)
    assert [sequence for _, sequence in msa] == fasta.decode().split()[1::2]
    assert _formats(session) == ["a2m", "fasta"]


def test_an_msa_with_no_match_columns_iterates_every_column(session):
    msa = _stored(session, "sto", 0, b">q\nMKVLAAGIT\n>s1\nMKVQLAGIT\n")
    assert list(msa) == [("q", "MKVLAAGIT"), ("s1", "MKVQLAGIT")]
    assert _formats(session) == ["fasta"]


def test_indexing_reads_one_row_or_a_slice(msa):
    assert msa[0] == ROWS[0]
    assert msa[-1] == ROWS[2]
    assert msa[1:] == ROWS[1:]


def test_rows_are_downloaded_once_per_handle(msa, session):
    assert list(msa) == list(msa) == ROWS
    assert dict(msa)["h2"] == "-CDE"
    assert session.get.call_count == 1


def test_refresh_makes_the_next_read_download_again(msa, session):
    list(msa)
    session.get.side_effect = [_response(METADATA), _response(content=FASTA)]
    assert msa.refresh() is msa
    list(msa)
    assert session.get.call_count == 3


def test_to_string_defaults_to_the_stored_format(msa, session):
    session.get.return_value = _response(content=b">q\nACaDE\n")
    assert msa.to_string() == ">q\nACaDE\n"
    session.get.assert_called_once_with("v1/data/msa/m1", params=None)


def test_to_string_renders_the_named_format(msa, session):
    msa.to_string("sto")
    session.get.assert_called_once_with("v1/data/msa/m1", params={"format": "sto"})


def test_to_string_refuses_a_format_it_cannot_render(msa, session):
    with pytest.raises(ValueError, match="a2m, a3m, csv, fasta, sto.*'parquet'"):
        msa.to_string("parquet")
    session.get.assert_not_called()


@pytest.mark.parametrize("read", [list, len, lambda m: m[0], lambda m: m.to_string()])
def test_reading_a_pending_msa_names_its_status_and_downloads_nothing(session, read):
    pending = METADATA | {"status": "RUNNING", "job_id": "m1", "num_sequences": None}
    session.get.return_value = _response(pending)
    msa = MSA(session, MSAMetadata.model_validate(pending))
    with pytest.raises(JobOutputsNotReady, match="MSA m1 is RUNNING"):
        read(msa)
    session.get.assert_called_once_with("v1/data/msa/m1/metadata")


def test_update_sends_only_the_given_fields_and_returns_the_handle(msa, session):
    session.put.return_value = _response(METADATA | {"tags": ["x"]})
    assert msa.update(tags=["x"]) is msa
    assert session.put.call_args.kwargs["json"] == {"tags": ["x"]}
    assert msa.tags == ["x"]


def test_delete_deletes_the_row(msa, session):
    msa.delete()
    session.delete.assert_called_once_with("v1/data/msa/m1")


@pytest.mark.parametrize("name", ["download", "upload"])
def test_an_msa_has_no_download_or_upload(name):
    assert not hasattr(MSA, name)
