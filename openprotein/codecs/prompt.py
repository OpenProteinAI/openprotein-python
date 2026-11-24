"""The ``prompt-npz-v1`` codec: one file per replicate, a one-context ``prompt.npz``.

``context_0_sequence`` is every complex's ``:``-joined sequence concatenated
(uint8 ASCII); ``context_0_seqlens`` is each complex's length, so the
concatenation splits back apart unambiguously. ``context_0_coordinates`` is
written when at least one complex carries real coordinates, NaN-filled for
any complex that doesn't; ``context_0_plddt`` is written alongside it,
NaN-filled wherever a complex carries no pLDDT of its own.
``context_0_names`` is written when at least one complex has a name: one
string per complex, empty for a complex without one.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from zipfile import BadZipFile

import numpy as np

from .codec import Codec

__all__ = [
    "InvalidPrompt",
    "PromptComplex",
    "PromptReplicate",
    "decode",
    "encode",
    "prompt_npz_v1",
]


class InvalidPrompt(ValueError):
    """Raised when a prompt replicate or its ``prompt-npz-v1`` bytes are malformed."""


@dataclass(slots=True, eq=False)
class PromptComplex:
    """One complex of a replicate: its chains joined by ``:`` in ``sequence``.

    ``coordinates``, when given, is a per-residue backbone array whose
    leading axis has length ``len(sequence)`` (the ``:`` delimiter positions
    carry NaN rows); ``plddt``, when given, is a matching 1-D array. Either
    may be absent independently of the other. ``name`` is the complex's record
    name, such as the accession of the MSA row it was sampled from.
    """

    sequence: str
    coordinates: np.ndarray | None = None
    plddt: np.ndarray | None = None
    name: str | None = None


@dataclass(slots=True, eq=False)
class PromptReplicate:
    """One replicate: its complexes, concatenated in this order at rest."""

    complexes: list[PromptComplex]


def encode(replicate: PromptReplicate) -> bytes:
    """Render one replicate to ``prompt-npz-v1`` bytes.

    Raises
    ------
    InvalidPrompt
        If the replicate has no complexes, a sequence is not ASCII, a
        ``coordinates``/``plddt`` array's length does not match its
        complex's sequence length, or ``coordinates`` or ``plddt`` arrays
        across complexes disagree on trailing shape.
    """
    complexes = replicate.complexes
    if not complexes:
        raise InvalidPrompt("a prompt replicate needs at least one complex")

    sequence_arrays = []
    seqlens = []
    for complex_ in complexes:
        try:
            encoded = complex_.sequence.encode("ascii")
        except UnicodeEncodeError as exc:
            raise InvalidPrompt(f"non-ASCII sequence: {complex_.sequence!r}") from exc
        sequence_arrays.append(np.frombuffer(encoded, dtype=np.uint8))
        seqlens.append(len(encoded))
        if complex_.coordinates is not None and len(complex_.coordinates) != len(
            encoded
        ):
            raise InvalidPrompt(
                f"coordinates length {len(complex_.coordinates)} does not match "
                f"sequence length {len(encoded)}"
            )
        if complex_.plddt is not None and len(complex_.plddt) != len(encoded):
            raise InvalidPrompt(
                f"plddt length {len(complex_.plddt)} does not match sequence "
                f"length {len(encoded)}"
            )

    files: dict[str, np.ndarray] = {
        "context_0_sequence": np.concatenate(sequence_arrays),
        "context_0_seqlens": np.array(seqlens, dtype=np.int64),
    }
    coordinates = _stack_optional(complexes, seqlens, "coordinates")
    if coordinates is not None and np.isnan(coordinates).all():
        coordinates = None
    plddt = _stack_optional(complexes, seqlens, "plddt")
    if plddt is not None and np.isnan(plddt).all():
        plddt = None
    if coordinates is not None:
        files["context_0_coordinates"] = coordinates
        if plddt is None:
            plddt = np.full(len(coordinates), np.nan, dtype=np.float32)
    if plddt is not None:
        files["context_0_plddt"] = plddt
    names = [complex_.name for complex_ in complexes]
    if any(name is not None for name in names):
        files["context_0_names"] = np.array(
            [name or "" for name in names], dtype=np.str_
        )

    buf = io.BytesIO()
    np.savez_compressed(buf, allow_pickle=False, **files)
    return buf.getvalue()


def _stack_optional(
    complexes: Sequence[PromptComplex], seqlens: list[int], field: str
) -> np.ndarray | None:
    present = [getattr(complex_, field) for complex_ in complexes]
    if all(value is None for value in present):
        return None
    trailing_shape = next(
        np.asarray(value).shape[1:] for value in present if value is not None
    )
    parts = [
        np.full((length, *trailing_shape), np.nan, dtype=np.float32)
        if value is None
        else np.asarray(value, dtype=np.float32)
        for value, length in zip(present, seqlens, strict=True)
    ]
    try:
        return np.concatenate(parts)
    except ValueError as exc:
        raise InvalidPrompt(
            f"inconsistent {field} shape across complexes: {exc}"
        ) from exc


def decode(data: bytes) -> PromptReplicate:
    """Parse ``prompt-npz-v1`` bytes back into one replicate's complexes.

    Splits ``context_0_sequence`` by the cumulative ``context_0_seqlens``,
    matching Gen 1 PoET's per-context decode exactly, so a file this codec
    writes round-trips through either reader.

    Raises
    ------
    InvalidPrompt
        If the bytes are not a well-formed ``prompt-npz-v1`` file, a
        required key is missing, the sequence is not uint8, seqlens is not a
        1-D integer array, the sequence, coordinates or pLDDT lengths are
        inconsistent, the names are not one string per complex, or a sequence
        is not ASCII.
    """
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            sequence = npz["context_0_sequence"]
            seqlens = npz["context_0_seqlens"]
            coordinates = npz.get("context_0_coordinates")
            plddt = npz.get("context_0_plddt")
            names = npz.get("context_0_names")
    except (BadZipFile, KeyError, ValueError) as exc:
        raise InvalidPrompt(f"malformed prompt-npz-v1 bytes: {exc}") from exc

    if sequence.dtype != np.uint8:
        raise InvalidPrompt("prompt sequence must be uint8")
    if seqlens.ndim != 1 or seqlens.dtype.kind not in "iu":
        raise InvalidPrompt("prompt seqlens must be a 1-D integer array")
    if len(sequence) != seqlens.sum():
        raise InvalidPrompt("sequence length does not match the sum of seqlens")
    if coordinates is not None and len(coordinates) != len(sequence):
        raise InvalidPrompt("coordinates length does not match the sequence length")
    if plddt is not None and len(plddt) != len(sequence):
        raise InvalidPrompt("plddt length does not match the sequence length")
    if names is not None and (
        names.ndim != 1 or names.dtype.kind != "U" or len(names) != len(seqlens)
    ):
        raise InvalidPrompt("prompt names must be one string per complex")

    breaks = np.cumsum(seqlens)[:-1]
    sequence_splits = np.split(sequence, breaks)
    coordinate_splits = (
        np.split(coordinates, breaks)
        if coordinates is not None
        else [None] * len(seqlens)
    )
    plddt_splits = (
        np.split(plddt, breaks) if plddt is not None else [None] * len(seqlens)
    )
    name_list = (
        [str(name) or None for name in names]
        if names is not None
        else [None] * len(seqlens)
    )
    try:
        complexes = [
            PromptComplex(
                sequence=seq.tobytes().decode("ascii"),
                coordinates=coord,
                plddt=p,
                name=name,
            )
            for seq, coord, p, name in zip(
                sequence_splits, coordinate_splits, plddt_splits, name_list, strict=True
            )
        ]
    except UnicodeDecodeError as exc:
        raise InvalidPrompt(f"non-ASCII sequence bytes: {exc}") from exc
    except ValueError as exc:
        raise InvalidPrompt(
            f"seqlens does not match the encoded complexes: {exc}"
        ) from exc
    return PromptReplicate(complexes=complexes)


prompt_npz_v1 = Codec(name="prompt-npz-v1", encode=encode, decode=decode)
