"""The ``ndarray-v1`` codec: a numeric numpy array as ``.npy`` bytes."""

from __future__ import annotations

import io

import numpy as np

from .codec import Codec

__all__ = ["decode", "encode", "ndarray_v1"]


def encode(value: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    source = io.BytesIO(data)
    value = np.load(source, allow_pickle=False)
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or value.dtype.kind not in "iufcb"
    ):
        raise ValueError("ndarray artifact must contain a numeric array")
    if source.read(1):
        raise ValueError("ndarray artifact contains trailing bytes")
    return value


ndarray_v1 = Codec(name="ndarray-v1", encode=encode, decode=decode)
