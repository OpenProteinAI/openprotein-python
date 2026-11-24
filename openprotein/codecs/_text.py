"""UTF-8 text codecs: ``mmcif-utf8-v1``, ``fasta-utf8-v1`` and ``json-utf8-v1``."""

from __future__ import annotations

import json
from typing import Any

from .codec import Codec


def encode(value: str) -> bytes:
    return value.encode("utf-8")


def decode(data: bytes) -> str:
    return data.decode("utf-8")


mmcif_utf8_v1 = Codec(name="mmcif-utf8-v1", encode=encode, decode=decode)
fasta_utf8_v1 = Codec(name="fasta-utf8-v1", encode=encode, decode=decode)


def encode_json(value: Any) -> bytes:
    """Compact JSON; NaN and infinities raise ``ValueError``."""
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode_json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


json_utf8_v1 = Codec(name="json-utf8-v1", encode=encode_json, decode=decode_json)
