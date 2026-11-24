"""Codecs for stored result files.

A ``Codec`` is looked up by its immutable ``name`` with ``get``. The built-ins
are registered at import.
"""

from __future__ import annotations

from ._text import fasta_utf8_v1, json_utf8_v1, mmcif_utf8_v1
from .codec import Codec
from .msa import MSA, MSARecord, msa_parquet_v1
from .ndarray import ndarray_v1
from .prompt import InvalidPrompt, PromptComplex, PromptReplicate, prompt_npz_v1

_REGISTRY: dict[str, Codec] = {}


def _register(codec: Codec) -> None:
    """Add ``codec`` under its name; a name is bound to one codec for good."""
    existing = _REGISTRY.get(codec.name)
    if existing is not None and existing is not codec:
        raise ValueError(f"codec {codec.name!r} is already registered")
    _REGISTRY[codec.name] = codec


def get(name: str) -> Codec | None:
    return _REGISTRY.get(name)


for _builtin in (
    ndarray_v1,
    mmcif_utf8_v1,
    fasta_utf8_v1,
    json_utf8_v1,
    msa_parquet_v1,
    prompt_npz_v1,
):
    _register(_builtin)
del _builtin

__all__ = [
    "MSA",
    "Codec",
    "InvalidPrompt",
    "MSARecord",
    "PromptComplex",
    "PromptReplicate",
    "fasta_utf8_v1",
    "get",
    "json_utf8_v1",
    "mmcif_utf8_v1",
    "msa_parquet_v1",
    "ndarray_v1",
    "prompt_npz_v1",
]
