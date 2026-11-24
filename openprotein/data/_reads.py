"""Reads of stored data rows."""

import copy

from requests import Response

from openprotein.base import APISession
from openprotein.errors import HTTPError


def read(session: APISession, path: str, **kwargs) -> Response:
    """GET a stored row and return the response.

    Errors are raised as the session raises them, except that a 425 is raised
    as a 404.
    """
    try:
        return session.get(path, **kwargs)
    except HTTPError as error:
        if error.status_code != 425:
            raise
        response = copy.copy(error.response)
        response.status_code = 404
        raise HTTPError(response) from None
