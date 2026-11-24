"""Fixture files written by the server's codecs decode identically here."""

import io
from pathlib import Path

import numpy as np
import pytest

from openprotein.codecs import MSA, MSARecord, msa_parquet_v1, prompt_npz_v1

pq = pytest.importorskip("pyarrow.parquet")

FIXTURES = Path(__file__).parent / "fixtures"

SINGLE_CHAIN = MSA(
    records=[
        MSARecord(
            name="query desc\twith tab",
            sequence="MK-LV",
            annotations=[("DE", "query protein"), ("AC", "P12345")],
            column_annotations={"SS": "HH-EE", "PP": "99*87"},
        ),
        MSARecord(name="hit1", sequence="MKALV"),
        MSARecord(name="hit2", sequence="-KAL-", annotations=[("DE", "second hit")]),
    ],
    source_format="sto",
    num_match_columns=5,
    annotations=[("ID", "fixture"), ("AU", "openprotein")],
    column_annotations={"RF": "xx.xx", "SS_cons": "HH-EE"},
)
MULTICHAIN = MSA(
    records=[
        MSARecord(name="101\t102", sequence="MKLV:ACDE"),
        MSARecord(name="paired", sequence="MKaLV:AC-E"),
        MSARecord(name="unpaired", sequence="----:ACDE"),
    ],
    source_format="a3m",
    num_match_columns=8,
    chain_copies=[2, 1],
)


@pytest.mark.parametrize(
    "filename, expected, footer_keys",
    [
        (
            "msa_single_chain_annotated.parquet",
            SINGLE_CHAIN,
            {"gf", "gc"},
        ),
        (
            "msa_multichain_chain_copies.parquet",
            MULTICHAIN,
            {"chain_copies"},
        ),
    ],
)
def test_msa_parquet_fixture_decodes_and_reencodes(filename, expected, footer_keys):
    data = (FIXTURES / filename).read_bytes()
    decoded = msa_parquet_v1.decode(data)
    assert decoded == expected
    assert msa_parquet_v1.decode(msa_parquet_v1.encode(decoded)) == expected

    for payload in (data, msa_parquet_v1.encode(decoded)):
        parquet = pq.ParquetFile(io.BytesIO(payload))
        metadata = parquet.schema_arrow.metadata
        assert set(metadata) == {
            b"schema_version",
            b"source_format",
            b"num_match_columns",
            b"num_sequences",
        } | {key.encode() for key in footer_keys}
        assert metadata[b"schema_version"] == b"1.2"
        assert metadata[b"num_sequences"] == str(expected.num_sequences).encode()
        assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"


def test_msa_parquet_fixture_columns():
    single = pq.read_table(FIXTURES / "msa_single_chain_annotated.parquet")
    multi = pq.read_table(FIXTURES / "msa_multichain_chain_copies.parquet")
    assert single.column_names == ["name", "sequence", "gs", "gr"]
    assert multi.column_names == ["name", "sequence"]


def _assert_prompt_round_trips(data):
    decoded = prompt_npz_v1.decode(data)
    again = prompt_npz_v1.decode(prompt_npz_v1.encode(decoded))
    for left, right in zip(decoded.complexes, again.complexes, strict=True):
        assert left.sequence == right.sequence
        for a, b in ((left.coordinates, right.coordinates), (left.plddt, right.plddt)):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)
    return decoded


def test_prompt_npz_fixture_with_coordinates():
    data = (FIXTURES / "prompt_with_coordinates.npz").read_bytes()
    decoded = _assert_prompt_round_trips(data)
    first, second = decoded.complexes
    assert (first.sequence, second.sequence) == ("MK:LVACDE", "GHIK")

    coordinates = np.arange(81, dtype=np.float32).reshape(9, 3, 3)
    coordinates[2] = np.nan
    np.testing.assert_array_equal(first.coordinates, coordinates)
    np.testing.assert_array_equal(first.plddt, np.linspace(50, 90, 9, dtype=np.float32))
    assert second.coordinates is not None and second.coordinates.shape == (4, 3, 3)
    assert np.isnan(second.coordinates).all()
    assert second.plddt is not None and np.isnan(second.plddt).all()

    with np.load(FIXTURES / "prompt_with_coordinates.npz") as npz:
        assert sorted(npz.files) == [
            "context_0_coordinates",
            "context_0_plddt",
            "context_0_seqlens",
            "context_0_sequence",
        ]
        np.testing.assert_array_equal(npz["context_0_seqlens"], [9, 4])


def test_prompt_npz_fixture_sequences_only():
    data = (FIXTURES / "prompt_sequences_only.npz").read_bytes()
    decoded = _assert_prompt_round_trips(data)
    assert [c.sequence for c in decoded.complexes] == ["MKLV", "AC:DE"]
    assert all(c.coordinates is None and c.plddt is None for c in decoded.complexes)

    with np.load(FIXTURES / "prompt_sequences_only.npz") as npz:
        assert sorted(npz.files) == ["context_0_seqlens", "context_0_sequence"]
        assert npz["context_0_sequence"].dtype == np.uint8
