"""A generated MSA is read from the stored MSA in the format its job type names."""

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from openprotein.align import api
from openprotein.align.msa import MSAFuture
from openprotein.align.schemas import AbNumberJob, ClustalOJob, MafftJob, MSAJob
from openprotein.errors import HTTPError

A3M = b"#12\t1\n>query\nMKTAYIAKQR\n>hit_1\nMKT-YIAkqR\n>hit_2\nMKTAYIA\nKQR\n"
FASTA = b">query\nMKTAYIAKQR\n>hit_1\nMKT-YIAKQR\n"


def _job_json(job_type: str) -> dict:
    return {
        "job_id": "msa1",
        "job_type": job_type,
        "status": "SUCCESS",
        "created_date": datetime(2023, 1, 1).isoformat(),
    }


def _stored(session: MagicMock, body: bytes, job_type: str = "/align/align") -> None:
    """Serve the job record at ``v1/jobs/...`` and the MSA bytes elsewhere."""

    def get(endpoint, **kwargs):
        if endpoint.startswith("v1/jobs/"):
            return MagicMock(status_code=200, json=lambda: _job_json(job_type))
        return MagicMock(status_code=200, content=body)

    session.get.side_effect = get


def _msa_request(session: MagicMock):
    return next(c for c in session.get.call_args_list if "data/msa" in c.args[0])


def test_get_msa_reads_the_stored_msa_as_names_and_sequences(mock_session):
    """The a3m header is skipped and insertions and wrapped lines are kept."""
    _stored(mock_session, A3M)
    pairs = list(api.get_msa(mock_session, "msa1"))
    assert pairs == [
        ("query", "MKTAYIAKQR"),
        ("hit_1", "MKT-YIAkqR"),
        ("hit_2", "MKTAYIAKQR"),
    ]
    assert _msa_request(mock_session).kwargs["params"] == {"format": "a3m"}


def test_get_msa_reads_a_fasta_body(mock_session):
    """A FASTA body reads as the same pairs."""
    _stored(mock_session, FASTA)
    assert list(api.get_msa(mock_session, "msa1")) == [
        ("query", "MKTAYIAKQR"),
        ("hit_1", "MKT-YIAKQR"),
    ]


def test_get_msa_raises_the_session_error_for_a_missing_msa(mock_session):
    """A 404 from the stored MSA raises `HTTPError` with that status."""
    _stored(mock_session, FASTA)
    stored = mock_session.get.side_effect
    missing = MagicMock(status_code=404, text="not found")
    missing.url = "v1/data/msa/msa1"

    def get(endpoint, **kwargs):
        if endpoint.startswith("v1/data/msa/"):
            raise HTTPError(missing)
        return stored(endpoint, **kwargs)

    mock_session.get.side_effect = get
    with pytest.raises(HTTPError) as raised:
        api.get_msa(mock_session, "msa1")
    assert raised.value.status_code == 404
    assert "data/msa/msa1" in _msa_request(mock_session).args[0]


def test_an_msa_future_reads_its_msa_id_from_the_stored_msa(mock_session):
    """`MSAFuture.get()` reads `v1/data/msa/<msa_id>` and parses it."""
    job = MSAJob.model_validate(
        {
            "job_id": "job1",
            "job_type": "/align/align",
            "status": "SUCCESS",
            "created_date": datetime(2023, 1, 1).isoformat(),
        }
    )
    _stored(mock_session, A3M)
    future = MSAFuture(mock_session, job)
    assert next(iter(future.get())) == ("query", "MKTAYIAKQR")
    assert mock_session.get.call_args.args[0] == f"v1/data/msa/{job.msa_id}"


@pytest.mark.parametrize(
    ("job_type", "job_class", "expected"),
    [
        ("/align/align", MSAJob, "a3m"),
        ("/align/mafft", MafftJob, "fasta"),
        ("/align/clustalo", ClustalOJob, "fasta"),
        ("/align/abnumber", AbNumberJob, "fasta"),
    ],
)
def test_an_msa_is_requested_in_the_format_its_job_type_names(
    mock_session, job_type, job_class, expected
):
    """`MSAFuture.get()` and `get_msa(job_id)` request the same format per job type."""
    body = A3M if expected == "a3m" else FASTA
    _stored(mock_session, body, job_type)
    job = job_class.model_validate(_job_json(job_type))

    from_future = list(MSAFuture(mock_session, job).get())
    assert _msa_request(mock_session).kwargs["params"] == {"format": expected}

    mock_session.get.reset_mock()
    from_id = list(api.get_msa(mock_session, "msa1"))
    assert _msa_request(mock_session).kwargs["params"] == {"format": expected}
    assert from_id == from_future
    assert from_id[0] == ("query", "MKTAYIAKQR")


def test_reading_an_msa_with_its_format_makes_no_job_lookup(mock_session):
    """`_read_msa` requests the MSA in the given format and nothing else."""
    _stored(mock_session, FASTA)
    pairs = list(api._read_msa(mock_session, "msa1", "fasta"))
    mock_session.get.assert_called_once()
    assert _msa_request(mock_session).kwargs["params"] == {"format": "fasta"}
    assert pairs[0] == ("query", "MKTAYIAKQR")


def test_get_msa_takes_a_session_and_a_job_id_only():
    """`get_msa` has the parameters `session` and `job_id`."""
    assert list(inspect.signature(api.get_msa).parameters) == ["session", "job_id"]
