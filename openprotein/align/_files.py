"""Sequence files read by the deprecated align methods."""

import csv

from openprotein import fasta
from openprotein.errors import InvalidParameterError


def read_file_content(file) -> bytes | bytearray | str:
    """Read the sequence file a deprecated align method was given.

    Accepts a file-like object, ``bytes``, ``bytearray``, ``str`` or a
    ``(filename, fileobj)`` / ``(filename, fileobj, content_type)`` tuple or list.
    """
    if isinstance(file, (tuple, list)):
        file = file[1]
    return file.read() if hasattr(file, "read") else file


def read_sequences(
    content: bytes | bytearray | str,
) -> tuple[list[str], list[str] | None]:
    """Parse FASTA or headerless CSV text into sequences and optional names."""
    if isinstance(content, (bytes, bytearray)):
        content = content.decode("utf-8-sig")
    text = content.removeprefix("\ufeff")
    lines = text.splitlines()
    if any(line.lstrip().startswith(">") for line in lines):
        parsed_names, parsed_sequences = fasta.parse(lines)
        names: list[str] | None = []
        sequences: list[str] = []
        for name, sequence in zip(parsed_names, parsed_sequences):
            if sequence.strip():
                names.append(str(name))
                sequences.append(str(sequence))
    else:
        rows = []
        width = 1
        for row in csv.reader(text.splitlines(keepends=True)):
            trimmed = row.copy()
            while trimmed and not trimmed[-1].strip():
                trimmed.pop()
            if trimmed:
                rows.append(row)
                width = max(width, len(trimmed))
        if width > 2:
            raise InvalidParameterError(
                "CSV must have one sequence column or name and sequence columns."
            )
        names = None if width == 1 else []
        sequences = []
        for row in rows:
            sequence = row[1] if names is not None and len(row) > 1 else row[0]
            if not sequence.strip():
                continue
            sequences.append(sequence.strip())
            if names is not None:
                names.append(row[0].strip() if len(row) > 1 else "")
    if not sequences:
        raise InvalidParameterError("The file contains no sequences.")
    return sequences, names
