"""Wire calls for job updates, lookups, outputs and units."""

from typing import Any

from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.errors import HTTPError, JobOutputsNotReady

from .api import job_get
from .schemas import Job, _UnitRecord


def operation_job(
    session: APISession, body: dict[str, Any], id_key: str
) -> dict[str, Any]:
    """The job an operation submitted: its body, or a read of the id it names."""
    if "job_type" in body:
        return body
    return job_get(session, job_id=body[id_key]).model_dump()


def job_update(
    session: APISession,
    job_id: str,
    name: str | None = None,
    description: str | None = None,
) -> Job:
    """
    Update a job's name and description.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    job_id : str
        The job to update.
    name : str, optional
        New name for the job. Left unchanged when None.
    description : str, optional
        New description for the job. Left unchanged when None.

    Returns
    -------
    Job
        The updated job.
    """
    endpoint = f"v1/jobs/{job_id}"

    body = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description

    response = session.put(endpoint, json=body)
    return TypeAdapter(Job).validate_python(response.json())


def job_output_page(
    session: APISession,
    job_id: str,
    output_name: str | None,
    cursor: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fetch one page of an output collection in position order.

    ``output_name=None`` reads the bare route, which serves the method's
    default output and names it in the page's ``output``.
    """
    endpoint = f"v1/data/results/{job_id}"
    if output_name is not None:
        endpoint = f"{endpoint}/{output_name}"
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    kwargs: dict[str, Any] = {"params": params} if params else {}
    try:
        response = session.get(endpoint, **kwargs)
    except HTTPError as exc:
        if exc.status_code != 409:
            raise
        raise JobOutputsNotReady(
            f"output {output_name or '<default>'!r} of job {job_id} is not complete yet"
        ) from exc
    if response.status_code == 409:
        raise JobOutputsNotReady(
            f"output {output_name or '<default>'!r} of job {job_id} is not complete yet"
        )
    return dict(response.json())


def job_default_output(session: APISession, job_id: str) -> str:
    """The name of the job's default output, from the bare results route."""
    page = job_output_page(session, job_id, None, limit=1)
    name = page.get("output")
    if not isinstance(name, str) or not name:
        raise ValueError(f"results of job {job_id} do not name their output")
    return name


def job_output_get(
    session: APISession, job_id: str, output_name: str, position: int
) -> dict[str, Any]:
    """Resolve an authorized logical output handle to metadata, never bytes."""
    response = session.get(f"v1/data/results/{job_id}/{output_name}/{position}")
    return dict(response.json())


def job_unit_page(
    session: APISession, job_id: str, *, position: int | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = (
        {} if position is None else {"params": {"after_position": position}}
    )
    return dict(session.get(f"v1/jobs/{job_id}/units", **kwargs).json())


def get_unit(session: APISession, job_id: str, position: int) -> _UnitRecord:
    """Read one unit's record."""
    response = session.get(f"v1/jobs/{job_id}/units/{position}")
    return _UnitRecord.model_validate(response.json())
