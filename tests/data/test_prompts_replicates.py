"""Tests for parsing the stored-prompt download zip into replicate sequences."""

import io
import random
import zipfile

from openprotein.data.prompts._replicates import parse_replicates


def _zip_of(replicates_by_position: dict[int, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for position, content in replicates_by_position.items():
            archive.writestr(f"replicate-{position}.fasta", content)
    return buf.getvalue()


def test_parse_replicates_orders_a_shuffled_zip_by_position():
    positions = list(range(11))
    shuffled = positions[:]
    random.Random(0).shuffle(shuffled)
    payload = _zip_of(
        {position: f">c1\nSEQ{position}\n".encode() for position in shuffled}
    )

    replicates = parse_replicates(payload)

    assert [replicate[0] for replicate in replicates] == [
        f"SEQ{position}" for position in positions
    ]


def test_parse_replicates_keeps_chain_separators_within_one_sequence():
    payload = _zip_of({0: b">c1\nMK:LV\n"})

    assert parse_replicates(payload) == [["MK:LV"]]
