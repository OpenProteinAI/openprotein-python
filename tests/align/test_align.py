"""L2 integration tests for the align domain."""

import io
from unittest.mock import MagicMock, patch

import pytest

from openprotein.align.align import AlignAPI
from openprotein.align.msa import MSAFuture
from openprotein.align.schemas import AbNumberScheme, UnnumberedPolicy
from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobType


@pytest.fixture
def align_api(mock_session: MagicMock) -> AlignAPI:
    return AlignAPI(mock_session)


@pytest.mark.parametrize("method", ["mafft", "clustalo", "abnumber"])
def test_names_of_another_length_than_sequences_are_refused_before_any_request(
    align_api: AlignAPI, mock_session: MagicMock, method
):
    with pytest.raises(InvalidParameterError, match="same length, not 1 and 2"):
        getattr(align_api, method)(["ACD", "AAA"], names=["a"])
    mock_session.post.assert_not_called()


def test_align_api_mafft(align_api: AlignAPI, mock_session: MagicMock):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.mafft.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.mafft(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)


def test_upload_msa_posts_the_file_as_a_ready_made_msa(
    align_api: AlignAPI, mock_session: MagicMock
):
    mock_session.post.return_value.json.return_value = {
        "job_id": "msa-1",
        "status": "PENDING",
        "job_type": JobType.align_align.value,
        "created_date": "2023-01-01T00:00:00",
    }
    msa_file = io.BytesIO(b">query\nACGT\n>homolog\nACGA\n")

    with pytest.warns(FutureWarning) as record:
        future = align_api.upload_msa(msa_file)

    expected = (
        "`AlignAPI.upload_msa` is deprecated since 0.18.0 and will be removed in "
        "0.19.0. Use `session.data.MSA.create` and `MSA.sample_prompt`."
    )
    assert [str(w.message) for w in record] == [expected]
    mock_session.post.assert_called_once_with(
        "v1/align/msa", files={"msa_file": msa_file}, params={"is_seed": False}
    )
    assert isinstance(future, MSAFuture)
    assert future.id == "msa-1"


def test_align_api_clustalo(align_api: AlignAPI, mock_session: MagicMock):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.clustalo.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.clustalo(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)


def test_align_api_abnumber(align_api: AlignAPI, mock_session: MagicMock):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.abnumber.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.abnumber(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)


@pytest.mark.parametrize(
    ("content", "sequences", "names"),
    [
        (b">a\nAC\nGT\n>b\nMKV\n", ["ACGT", "MKV"], ["a", "b"]),
        (b"ACGT\nMKV\n\n", ["ACGT", "MKV"], None),
        (b"a,ACGT\nb,MKV,\n,\n", ["ACGT", "MKV"], ["a", "b"]),
        (
            b"ok,ACGT\nmissing,\nspaces,  , ,\nextra,MKV,,\nsolo\n",
            ["ACGT", "MKV", "solo"],
            ["ok", "extra", ""],
        ),
        (
            b'"first\nsecond",ACGT\n"comma,label",MKV\n',
            ["ACGT", "MKV"],
            ["first\nsecond", "comma,label"],
        ),
        (b"ACGT,\nMKV,,\n", ["ACGT", "MKV"], None),
    ],
    ids=[
        "fasta",
        "csv-one-column",
        "csv-two-columns",
        "csv-blank-sequences",
        "csv-quoted-labels",
        "csv-one-column-trailing-fields",
    ],
)
@pytest.mark.parametrize(
    ("method", "options"),
    [
        ("mafft", {"auto": False, "ep": 0.1, "op": 1.5}),
        ("clustalo", {"clustersize": 10, "iterations": 3}),
        (
            "abnumber",
            {
                "scheme": AbNumberScheme.KABAT,
                "drop_minority_chains": True,
                "unnumbered_policy": UnnumberedPolicy.DROP,
            },
        ),
    ],
)
def test_align_api_file_methods_read_sequences_and_pass_options(
    align_api: AlignAPI,
    method: str,
    options: dict,
    content: bytes,
    sequences: list[str],
    names: list[str] | None,
):
    with patch.object(AlignAPI, method) as target, pytest.warns(FutureWarning):
        result = getattr(align_api, f"{method}_file")(io.BytesIO(content), **options)
    target.assert_called_once_with(sequences, names, **options)
    assert result is target.return_value


def test_align_api_file_methods_read_text_files(align_api: AlignAPI):
    with patch.object(AlignAPI, "mafft") as target, pytest.warns(FutureWarning):
        align_api.mafft_file(io.StringIO(">a\nACGT\n"))
    target.assert_called_once_with(["ACGT"], ["a"], auto=True, ep=None, op=None)


@pytest.mark.parametrize("method", ["mafft_file", "clustalo_file", "abnumber_file"])
@pytest.mark.parametrize("content", [b"a,b,ACGT\n", b"", b"\n\n", b">a\n\n"])
def test_align_api_file_methods_reject_unusable_files(
    align_api: AlignAPI, method: str, content: bytes
):
    with pytest.warns(FutureWarning), pytest.raises(InvalidParameterError):
        getattr(align_api, method)(io.BytesIO(content))


@pytest.mark.parametrize("method", ["mafft", "clustalo", "abnumber"])
def test_align_api_file_methods_warn_with_the_sequence_based_replacement(
    align_api: AlignAPI, method: str
):
    with patch.object(AlignAPI, method), pytest.warns(FutureWarning) as record:
        getattr(align_api, f"{method}_file")(io.BytesIO(b">a\nACGT\n"))
    expected = (
        f"`AlignAPI.{method}_file` is deprecated since 0.18.0 and will be removed "
        f"in 0.19.0. Use `AlignAPI.{method}` with sequences and names."
    )
    assert [str(w.message) for w in record] == [expected]
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "make_file",
    [
        lambda: b">a\nACGT\n",
        lambda: ">a\nACGT\n",
        lambda: io.BytesIO(b">a\nACGT\n"),
        lambda: ("seqs.fasta", io.BytesIO(b">a\nACGT\n")),
        lambda: ("seqs.fasta", io.StringIO(">a\nACGT\n"), "text/plain"),
        lambda: ["seqs.fasta", io.BytesIO(b">a\nACGT\n")],
        lambda: ["seqs.fasta", io.StringIO(">a\nACGT\n"), "text/plain"],
        lambda: bytearray(b">a\nACGT\n"),
    ],
)
@pytest.mark.parametrize("method", ["mafft", "clustalo", "abnumber"])
def test_align_api_file_methods_accept_bytes_bytearray_str_file_objects_tuples_and_lists(
    align_api: AlignAPI, method: str, make_file
):
    with patch.object(AlignAPI, method) as target, pytest.warns(FutureWarning):
        getattr(align_api, f"{method}_file")(make_file())
    assert target.call_args.args == (["ACGT"], ["a"])


def test_align_api_file_methods_keep_every_record_after_a_utf8_bom(
    align_api: AlignAPI,
):
    content = b"\xef\xbb\xbf>a\nACGT\n>b\nACGA\n"
    with patch.object(AlignAPI, "mafft") as target, pytest.warns(FutureWarning):
        align_api.mafft_file(io.BytesIO(content))
    assert target.call_args.args == (["ACGT", "ACGA"], ["a", "b"])


def test_align_api_file_methods_keep_every_record_after_a_bom_in_text(
    align_api: AlignAPI,
):
    content = "\ufeff>a\nACGT\n>b\nACGA\n"
    with patch.object(AlignAPI, "mafft") as target, pytest.warns(FutureWarning):
        align_api.mafft_file(content)
    assert target.call_args.args == (["ACGT", "ACGA"], ["a", "b"])
