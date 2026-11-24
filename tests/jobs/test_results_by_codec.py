"""`/data/results` values decode by the codec their `Ref` names."""

import hashlib
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein import codecs
from openprotein.codecs import (
    MSA,
    Codec,
    MSARecord,
    PromptComplex,
    PromptReplicate,
    fasta_utf8_v1,
    mmcif_utf8_v1,
    msa_parquet_v1,
    ndarray_v1,
    prompt_npz_v1,
)
from openprotein.jobs.jobs import JobsAPI


def _jobs(session, storage):
    jobs = JobsAPI(session)
    jobs._storage = storage
    return jobs


class MemoryStorage:
    def __init__(self, payload: bytes):
        self.payload = payload

    def iter_bytes(self, reference):
        yield self.payload


def _read(payload: bytes, codec: str | None):
    """Read position 0 of a mocked `/data/results/job/out/0` holding `payload`."""
    session = MagicMock()
    session.get.return_value.json.return_value = {
        "position": 0,
        "unit_id": "unit",
        "state": "SUCCESS",
        "value": {
            "ref": {
                "uri": "s3://content/abc",
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "s3_checksum": "checksum",
                "codec": codec,
                "download_url": "https://storage.example/signed",
            }
        },
    }
    jobs = _jobs(session, MemoryStorage(payload))
    output = jobs._get_job_output("job", 0, "out")
    session.get.assert_called_once_with("v1/data/results/job/out/0")
    return output.value


def test_ndarray_v1_decodes_to_an_array():
    expected = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    actual = _read(ndarray_v1.encode(expected), "ndarray-v1")
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name,shape,values",
    [
        ("none", (2,), [1, -2]),
        ("one", (2,), [1, -2]),
        ("two", (2, 2), [[1, -2], [3, 4]]),
        ("selected", (2,), [3, 4]),
    ],
)
def test_ndarray_v1_preserves_replicate_array(name, shape, values):
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "model_result_contracts"
        / f"replicate_{name}.npy"
    )
    actual = _read(path.read_bytes(), "ndarray-v1")
    assert actual.dtype.str == "<f2"
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, values)


def test_ndarray_v1_rejects_non_numeric_arrays():
    buffer = io.BytesIO()
    np.save(buffer, np.array(["A"]), allow_pickle=False)
    with pytest.raises(ValueError, match="numeric array"):
        _read(buffer.getvalue(), "ndarray-v1")


def test_ndarray_v1_rejects_trailing_bytes():
    payload = ndarray_v1.encode(np.ones(3)) + b"trailing"
    with pytest.raises(ValueError, match="trailing"):
        _read(payload, "ndarray-v1")


@pytest.mark.parametrize("codec", [mmcif_utf8_v1, fasta_utf8_v1], ids=str)
def test_text_codecs_decode_to_str(codec):
    assert _read(codec.encode(">q\nMKV\n"), codec.name) == ">q\nMKV\n"


def test_msa_parquet_v1_decodes_to_an_msa():
    pytest.importorskip("pyarrow")
    msa = MSA(
        records=[MSARecord("query", "MKV"), MSARecord("hit", "MK-")],
        source_format="a3m",
        num_match_columns=3,
    )
    assert _read(msa_parquet_v1.encode(msa), "msa-parquet-v1") == msa


def test_msa_parquet_v1_without_pyarrow_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError, match=r"openprotein-python\[msa\]"):
        _read(b"PAR1", "msa-parquet-v1")


def test_prompt_npz_v1_decodes_to_a_prompt_replicate():
    payload = prompt_npz_v1.encode(
        PromptReplicate([PromptComplex("MK:LV"), PromptComplex("ACDE")])
    )
    replicate = _read(payload, "prompt-npz-v1")
    assert isinstance(replicate, PromptReplicate)
    assert [c.sequence for c in replicate.complexes] == ["MK:LV", "ACDE"]


@pytest.mark.parametrize("codec", [None, "tests/unknown-v1"])
def test_null_or_unknown_codec_returns_bytes(codec):
    assert _read(b"\x00raw", codec) == b"\x00raw"


def test_registered_codec_decodes_its_refs():
    codecs._register(
        Codec(
            name="tests/measurement-v1",
            encode=lambda value: str(value).encode(),
            decode=lambda data: float(data) * 2,
        )
    )
    assert _read(b"1.25", "tests/measurement-v1") == 2.5


def test_ref_bytes_are_verified_before_decoding():
    session = MagicMock()
    session.get.return_value.json.return_value = {
        "position": 0,
        "unit_id": "unit",
        "state": "SUCCESS",
        "value": {
            "ref": {
                "uri": "s3://content/abc",
                "byte_length": 3,
                "sha256": "0" * 64,
                "codec": "fasta-utf8-v1",
            }
        },
    }
    with pytest.raises(ValueError, match="digest"):
        _jobs(session, MemoryStorage(b"abc"))._get_job_output("job", 0, "out")


@pytest.mark.parametrize("value", [{"inline": 1, "ref": {}}, {}, {"ref": "uri"}])
def test_malformed_values_are_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        JobsAPI(MagicMock())._decode_job_output(
            "job", {"position": 0, "unit_id": "u", "value": value}
        )
