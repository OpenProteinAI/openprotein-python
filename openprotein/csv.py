import csv
from typing import Iterator

__all__ = ["ensure_str_lines", "parse_stream"]


def ensure_str_lines(
    lines: Iterator[str] | Iterator[bytes], encoding="utf-8"
) -> Iterator[str]:
    """
    Decode bytes lines, passing str lines through unchanged.

    Parameters
    ----------
    lines : Iterator[str] | Iterator[bytes]
        Lines to normalize. A mix of str and bytes lines is permitted.
    encoding : str, optional
        Encoding used to decode bytes lines. Defaults to "utf-8".

    Yields
    ------
    str
        Each line as text.
    """
    for line in lines:
        if isinstance(line, bytes):
            yield line.decode(encoding)
        else:
            yield line


def parse_stream(lines: Iterator[str] | Iterator[bytes]) -> Iterator[list[str]]:
    """
    Parse lines of CSV text, yielding one list of fields per row.

    Parameters
    ----------
    lines : Iterator[str] | Iterator[bytes]
        Lines of CSV text. Bytes lines are decoded as UTF-8.

    Yields
    ------
    list[str]
        The fields of one CSV row.
    """
    reader = csv.reader(ensure_str_lines(lines))
    for row in reader:
        yield row
