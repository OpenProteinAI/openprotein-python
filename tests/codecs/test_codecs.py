import subprocess
import sys
import textwrap

import numpy as np
import pytest

from openprotein import codecs
from openprotein.codecs import (
    MSA,
    Codec,
    InvalidPrompt,
    MSARecord,
    PromptComplex,
    PromptReplicate,
    fasta_utf8_v1,
    json_utf8_v1,
    mmcif_utf8_v1,
    msa_parquet_v1,
    ndarray_v1,
    prompt_npz_v1,
)


@pytest.mark.parametrize(
    "codec",
    [
        ndarray_v1,
        mmcif_utf8_v1,
        fasta_utf8_v1,
        json_utf8_v1,
        msa_parquet_v1,
        prompt_npz_v1,
    ],
    ids=lambda codec: codec.name,
)
def test_builtins_are_registered_by_name(codec):
    assert codecs.get(codec.name) is codec


def test_get_returns_none_for_unknown_name():
    assert codecs.get("unknown-v1") is None


def test_register_binds_a_name_to_one_codec():
    codec = Codec(name="test/register@v1", encode=bytes, decode=bytes)
    codecs._register(codec)
    codecs._register(codec)
    assert codecs.get("test/register@v1") is codec
    with pytest.raises(ValueError, match="already registered"):
        codecs._register(Codec(name="test/register@v1", encode=bytes, decode=bytes))
    with pytest.raises(ValueError, match="already registered"):
        codecs._register(Codec(name="ndarray-v1", encode=bytes, decode=bytes))


def test_ndarray_v1_encodes_npy_bytes():
    header = b"{'descr': '<i2', 'fortran_order': False, 'shape': (3,), }"
    expected = (
        b"\x93NUMPY\x01\x00v\x00" + header + b" " * 60 + b"\n\x01\x00\x02\x00\x03\x00"
    )
    assert ndarray_v1.encode(np.array([1, 2, 3], dtype="<i2")) == expected


@pytest.mark.parametrize(
    "value",
    [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.array([np.nan, np.inf, -np.inf]),
        np.array([1 + 2j], dtype=np.complex64),
        np.array([True, False]),
        np.array([0, 2**63], dtype=np.uint64),
    ],
)
def test_ndarray_v1_round_trips(value):
    actual = ndarray_v1.decode(ndarray_v1.encode(value))
    assert actual.dtype == value.dtype
    assert np.array_equal(actual, value, equal_nan=True)


def test_ndarray_v1_rejects_non_numeric_arrays():
    with pytest.raises(ValueError, match="numeric array"):
        ndarray_v1.decode(ndarray_v1.encode(np.array(["A"])))


def test_ndarray_v1_rejects_pickled_arrays():
    import io

    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        ndarray_v1.decode(buf.getvalue())


def test_ndarray_v1_rejects_trailing_bytes():
    with pytest.raises(ValueError, match="trailing bytes"):
        ndarray_v1.decode(ndarray_v1.encode(np.arange(3)) + b"\x00")


@pytest.mark.parametrize(
    "codec", [mmcif_utf8_v1, fasta_utf8_v1], ids=lambda codec: codec.name
)
def test_text_codecs_encode_utf8(codec):
    assert codec.encode(">q é\nMK\n") == b">q \xc3\xa9\nMK\n"
    assert codec.decode(b">q \xc3\xa9\nMK\n") == ">q é\nMK\n"


@pytest.mark.parametrize(
    "codec", [mmcif_utf8_v1, fasta_utf8_v1], ids=lambda codec: codec.name
)
def test_text_codecs_reject_invalid_utf8(codec):
    with pytest.raises(UnicodeDecodeError):
        codec.decode(b"\xff")


def test_json_utf8_v1_encodes_compact_utf8():
    assert json_utf8_v1.encode({"a": [1, 2.5, None], "é": True}) == (
        b'{"a":[1,2.5,null],"\\u00e9":true}'
    )


@pytest.mark.parametrize(
    "value",
    [
        [{"chain_ids": ["A", "B"], "ptm": 0.5, "iptm": None}],
        {"nested": {"list": [[1, 2], [3.25, None]]}},
        None,
        "text",
    ],
)
def test_json_utf8_v1_round_trips(value):
    assert json_utf8_v1.decode(json_utf8_v1.encode(value)) == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1.0, -float("inf")]])
def test_json_utf8_v1_refuses_non_finite_numbers(value):
    with pytest.raises(ValueError):
        json_utf8_v1.encode(value)


def test_json_utf8_v1_rejects_invalid_utf8():
    with pytest.raises(UnicodeDecodeError):
        json_utf8_v1.decode(b'"\xff"')


def test_msa_parquet_v1_round_trips():
    pytest.importorskip("pyarrow")
    msa = MSA(
        records=[
            MSARecord("q", "MKLV", [("DE", "x")], {"SS": "HHEE"}),
            MSARecord("h", "MK-v"),
        ],
        source_format="sto",
        num_match_columns=4,
        annotations=[("ID", "a")],
        column_annotations={"RF": "xxxx"},
        chain_copies=[1],
    )
    assert msa_parquet_v1.decode(msa_parquet_v1.encode(msa)) == msa


def test_prompt_npz_v1_round_trips():
    coordinates = np.ones((4, 3, 3), dtype=np.float32)
    plddt = np.full(4, 70.0, dtype=np.float32)
    replicate = prompt_npz_v1.decode(
        prompt_npz_v1.encode(
            PromptReplicate(
                [PromptComplex("MK:L", coordinates, plddt), PromptComplex("AC")]
            )
        )
    )
    first, second = replicate.complexes
    assert (first.sequence, second.sequence) == ("MK:L", "AC")
    np.testing.assert_array_equal(first.coordinates, coordinates)
    np.testing.assert_array_equal(first.plddt, plddt)
    assert second.coordinates is not None and np.isnan(second.coordinates).all()
    assert second.plddt is not None and np.isnan(second.plddt).all()


def test_prompt_npz_v1_round_trips_names():
    replicate = prompt_npz_v1.decode(
        prompt_npz_v1.encode(
            PromptReplicate(
                [
                    PromptComplex("MK:L", name="UniRef100_A0A8B9J8Y4"),
                    PromptComplex("AC"),
                ]
            )
        )
    )

    assert [c.name for c in replicate.complexes] == ["UniRef100_A0A8B9J8Y4", None]


def test_prompt_npz_v1_writes_no_names_for_unnamed_complexes():
    import io

    data = prompt_npz_v1.encode(PromptReplicate([PromptComplex("MK")]))

    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        assert "context_0_names" not in npz.files


def test_prompt_npz_v1_rejects_names_that_do_not_match_the_complexes():
    import io

    buf = io.BytesIO()
    np.savez(
        buf,
        context_0_sequence=np.frombuffer(b"MKLV", dtype=np.uint8),
        context_0_seqlens=np.array([2, 2], dtype=np.int64),
        context_0_names=np.array(["only-one"], dtype=np.str_),
    )
    with pytest.raises(InvalidPrompt, match="names"):
        prompt_npz_v1.decode(buf.getvalue())


def test_prompt_npz_v1_rejects_an_empty_replicate():
    with pytest.raises(InvalidPrompt):
        prompt_npz_v1.encode(PromptReplicate([]))


def test_prompt_npz_v1_rejects_seqlens_that_do_not_sum_to_the_sequence():
    import io

    buf = io.BytesIO()
    np.savez(
        buf,
        context_0_sequence=np.frombuffer(b"MKLV", dtype=np.uint8),
        context_0_seqlens=np.array([3], dtype=np.int64),
    )
    with pytest.raises(InvalidPrompt, match="sum"):
        prompt_npz_v1.decode(buf.getvalue())


def test_import_does_not_require_pyarrow():
    script = textwrap.dedent(
        """
        import sys
        sys.modules["pyarrow"] = None
        import openprotein.codecs as codecs
        msa = codecs.MSA(records=[], source_format="a3m", num_match_columns=0)
        try:
            codecs.msa_parquet_v1.encode(msa)
        except ImportError as exc:
            assert "openprotein-python[msa]" in str(exc), exc
        else:
            raise AssertionError("expected ImportError")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
