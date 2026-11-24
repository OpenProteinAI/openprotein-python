"""Shared submission retry: a 425 answer is resubmitted, body and key unchanged."""

import time
from typing import Any

from requests import Response

from openprotein import config
from openprotein.base import APISession
from openprotein.errors import HTTPError

MAX_425_RETRIES = 5


def post_with_retry(
    session: APISession,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = MAX_425_RETRIES,
) -> Response:
    """POST once, then resubmit the identical body and headers while the answer is 425."""
    attempt = 0
    while True:
        try:
            return session.post(url, json=json, headers=headers)
        except HTTPError as error:
            if error.status_code != 425 or attempt >= max_retries:
                raise
            time.sleep(_retry_after(error))
            attempt += 1


def _retry_after(error: HTTPError) -> float:
    header = error.response.headers.get("Retry-After")
    if header is None:
        return config.POLLING_INTERVAL
    try:
        return float(header)
    except ValueError:
        return config.POLLING_INTERVAL
