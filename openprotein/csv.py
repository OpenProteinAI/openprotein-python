import codecs
import csv
from typing import Iterator

import requests


def csv_stream(response: requests.Response) -> Iterator[list[str]]:
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
    # get raw bytes stream
    raw_content = response.raw
    # force the response to be encoded as utf-8
    content = codecs.getreader("utf-8")(raw_content)
    return csv.reader(content)
