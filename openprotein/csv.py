import csv
from typing import Iterator


def ensure_str_lines(
    lines: Iterator[str] | Iterator[bytes], encoding="utf-8"
) -> Iterator[str]:
    for line in lines:
        if isinstance(line, bytes):
            yield line.decode(encoding)
        else:
            yield line


def parse_stream(lines: Iterator[str] | Iterator[bytes]) -> Iterator[list[str]]:
    """
    Returns a CSV reader from a requests.Response object.

    Parameters
    ----------
    response : requests.Response
        The response object to parse.

    Returns
    -------
    csv.reader
        A csv reader object for the response.
    """
    reader = csv.reader(ensure_str_lines(lines))
    for row in reader:
        yield row
