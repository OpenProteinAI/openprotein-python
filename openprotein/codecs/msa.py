"""The MSA model and the ``msa-parquet-v1`` codec: ``MSA`` <-> canonical Parquet.

One row per sequence record; MSA-level metadata is stored in the Parquet
file-level (footer) key/value metadata so it can be read without scanning the
row groups.

Stockholm annotations are preserved: per-sequence ``#=GS`` / ``#=GR`` ride in
optional ``gs`` / ``gr`` JSON columns, and alignment-level ``#=GF`` / ``#=GC`` in
the footer. All are written only when present, so a3m/fasta/csv blobs keep the
plain ``name`` / ``sequence`` schema.

``MSA.chain_copies`` rides in an optional ``chain_copies`` JSON-list footer key
(schema 1.2), written only when the MSA carries it; a schema 1.1 file has no such
key and reads back with ``chain_copies`` ``None``.

pyarrow is imported on first use; install it with ``openprotein-python[msa]``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

from .codec import Codec

__all__ = ["MSA", "SCHEMA_VERSION", "MSARecord", "decode", "encode", "msa_parquet_v1"]


SCHEMA_VERSION = "1.2"


@dataclass(slots=True)
class MSARecord:
    """One aligned sequence in an MSA.

    ``name`` is the record's header stored **verbatim** — the entire string after
    ``>`` exactly as uploaded, with its original whitespace (spaces or tabs)
    preserved. It is an opaque label; we never split it into id/description, so
    headers round-trip byte-for-byte.

    ``sequence`` preserves case: uppercase / ``-`` are match columns, lowercase
    are insert residues (a3m convention).

    ``annotations`` / ``column_annotations`` carry this record's own metadata
    (``None`` for formats that have no such concept): ``annotations`` is an ordered
    list of ``(feature, value)`` (Stockholm ``#=GS``); ``column_annotations`` maps
    ``feature -> per-column string`` (Stockholm ``#=GR``, e.g. ``PP``, ``SS``), one
    character per alignment column.
    """

    name: str
    sequence: str
    annotations: list[tuple[str, str]] | None = None
    column_annotations: dict[str, str] | None = None


@dataclass(slots=True)
class MSA:
    """An MSA as parsed from any supported format.

    ``annotations`` / ``column_annotations`` describe the alignment as a whole
    rather than any one record (``None`` when the file carried none):
    ``annotations`` is an ordered list of ``(feature, value)`` (Stockholm ``#=GF``,
    and the ``#<KEY> <value>`` comment lines the FASTA-family codecs read);
    ``column_annotations`` maps ``feature -> per-column string`` (Stockholm
    ``#=GC``, e.g. ``RF``, ``SS_cons``).

    The per-record counterparts live on ``MSARecord`` under the same two names.

    ``chain_copies`` is a complex's copy count per chain of the query row, in query
    order (``None`` when the source named no more than one copy in total).
    """

    records: list[MSARecord]
    source_format: str
    num_match_columns: int
    annotations: list[tuple[str, str]] | None = None
    column_annotations: dict[str, str] | None = None
    chain_copies: list[int] | None = None

    @property
    def num_sequences(self) -> int:
        """Number of sequence records."""
        return len(self.records)

    @property
    def has_insertions(self) -> bool:
        """True if any record carries lowercase insert residues."""
        return any(
            char.islower() for record in self.records for char in record.sequence
        )


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "msa-parquet-v1 requires pyarrow; install openprotein-python[msa]"
        ) from exc
    return pa, pq


def encode(msa: MSA) -> bytes:
    """Serialize an ``MSA`` to Parquet bytes."""
    pa, pq = _pyarrow()
    columns = {
        "name": pa.array([r.name for r in msa.records], type=pa.string()),
        "sequence": pa.array([r.sequence for r in msa.records], type=pa.string()),
    }
    # Per-record annotations, only added when any record carries them. Keys use
    # the Stockholm short names (gs/gr/gf/gc), which are part of the stored format.
    if any(r.annotations for r in msa.records):
        columns["gs"] = pa.array(
            [json.dumps(r.annotations) if r.annotations else None for r in msa.records],
            type=pa.string(),
        )
    if any(r.column_annotations for r in msa.records):
        columns["gr"] = pa.array(
            [
                json.dumps(r.column_annotations) if r.column_annotations else None
                for r in msa.records
            ],
            type=pa.string(),
        )
    table = pa.table(columns)

    metadata = {
        b"schema_version": SCHEMA_VERSION.encode(),
        b"source_format": msa.source_format.encode(),
        b"num_match_columns": str(msa.num_match_columns).encode(),
        b"num_sequences": str(msa.num_sequences).encode(),
    }
    if msa.annotations:
        metadata[b"gf"] = json.dumps(msa.annotations).encode()
    if msa.column_annotations:
        metadata[b"gc"] = json.dumps(msa.column_annotations).encode()
    if msa.chain_copies is not None:
        metadata[b"chain_copies"] = json.dumps(msa.chain_copies).encode()
    table = table.replace_schema_metadata(metadata)

    buf = io.BytesIO()
    # zstd compresses repetitive (gapped, annotation-heavy) columns far
    # better than the snappy default
    pq.write_table(table, buf, compression="zstd", compression_level=3)
    return buf.getvalue()


def decode(data: bytes) -> MSA:
    """Deserialize Parquet bytes back into an ``MSA``."""
    _, pq = _pyarrow()
    table = pq.read_table(io.BytesIO(data))
    meta = _decode_metadata(table.schema.metadata)
    cols = table.to_pydict()
    gs_col = cols.get("gs")
    gr_col = cols.get("gr")

    records = []
    for i in range(table.num_rows):
        gs = json.loads(gs_col[i]) if gs_col and gs_col[i] else None
        gr = json.loads(gr_col[i]) if gr_col and gr_col[i] else None
        records.append(
            MSARecord(
                name=cols["name"][i],
                sequence=cols["sequence"][i],
                annotations=[tuple(pair) for pair in gs] if gs else None,
                column_annotations=gr,
            )
        )

    gf = json.loads(meta["gf"]) if "gf" in meta else None
    return MSA(
        records=records,
        source_format=meta.get("source_format", "a3m"),
        num_match_columns=int(meta.get("num_match_columns", "0")),
        annotations=[tuple(pair) for pair in gf] if gf else None,
        column_annotations=json.loads(meta["gc"]) if "gc" in meta else None,
        chain_copies=json.loads(meta["chain_copies"])
        if "chain_copies" in meta
        else None,
    )


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    if not metadata:
        return {}
    return {k.decode(): v.decode() for k, v in metadata.items()}


msa_parquet_v1 = Codec(name="msa-parquet-v1", encode=encode, decode=decode)
