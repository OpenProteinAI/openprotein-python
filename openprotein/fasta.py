from typing import Iterator, Sequence, overload


@overload
def parse_stream(
    lines: Iterator[str], comment: str = "#"
) -> Iterator[tuple[str, str]]: ...


@overload
def parse_stream(
    lines: Iterator[bytes], comment: str = "#"
) -> Iterator[tuple[bytes, bytes]]: ...


def parse_stream(
    lines: Iterator[str] | Iterator[bytes], comment: str = "#"
) -> Iterator[tuple[str, str]] | Iterator[tuple[bytes, bytes]]:
    is_bytes: bool | None = None
    name = None
    sequence = []

    for line in lines:
        if not line:
            continue  # skip empty lines
        if is_bytes := isinstance(line, bytes):
            line = line.decode()
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                sequence = "".join(sequence)
                if is_bytes:
                    name = name.encode()
                    sequence = sequence.encode()
                    yield name, sequence
                else:
                    yield name, sequence
            name = line[1:].strip()
            sequence = []
        else:
            sequence.append(line.strip())

    if name is not None:
        sequence = "".join(sequence)
        if is_bytes:
            name = name.encode()
            sequence = sequence.encode()
            yield name, sequence
        else:
            yield name, sequence


def parse(
    f: Sequence[str] | Sequence[bytes], comment: str = "#"
) -> tuple[list[str], list[str]] | tuple[list[bytes], list[bytes]]:
    is_bytes: bool | None = None
    names = []
    sequences = []
    name = None
    sequence = []
    for line in f:
        if is_bytes := isinstance(line, bytes):
            line = line.decode()
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(">"):
            # its a new entry
            if name is not None:
                sequence = "".join(sequence)
                if is_bytes:
                    name = name.encode()
                    sequence = sequence.encode()
                names.append(name)
                sequences.append(sequence)
            # reset the reading
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        # last entry
        sequence = "".join(sequence)
        if is_bytes:
            name = name.encode()
            sequence = sequence.encode()
        names.append(name)
        sequences.append(sequence)

    return names, sequences
