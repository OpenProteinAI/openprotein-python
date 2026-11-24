"""Decode a stored ``Value`` by the codec its ``Ref`` names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openprotein import codecs

from ._storage import ArtifactStorage, verified_artifact


def decode_value(value: Any, storage: ArtifactStorage) -> Any:
    """Return an inline value as stored, or a ref's verified bytes decoded by its codec.

    A null codec, or one this process has not registered, returns the bytes.
    """
    if not isinstance(value, Mapping) or set(value) not in ({"inline"}, {"ref"}):
        raise ValueError("stored value must contain exactly one of inline or ref")
    if "inline" in value:
        return value["inline"]
    reference = value["ref"]
    if not isinstance(reference, Mapping):
        raise TypeError("stored value ref must be an object")
    with verified_artifact(storage, reference) as source:
        data = source.read()
    name = reference.get("codec")
    codec = codecs.get(name) if isinstance(name, str) else None
    return data if codec is None else codec.decode(data)
