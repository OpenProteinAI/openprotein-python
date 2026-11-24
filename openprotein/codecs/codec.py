"""The ``Codec`` record: a named, stable mapping between values and bytes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["Codec"]


@dataclass(frozen=True)
class Codec:
    """Encode a value to bytes and decode it back.

    ``name`` identifies the on-disk format. Names are immutable and
    append-only: a format change is a new name.
    """

    name: str
    encode: Callable[[Any], bytes]
    decode: Callable[[bytes], Any]
