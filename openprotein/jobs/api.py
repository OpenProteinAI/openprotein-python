"""Jobs and job-centric flows."""

from pydantic import TypeAdapter

from openprotein.base import APISession

from .schemas import Job

__all__ = ["job_args_get", "job_get", "jobs_list"]


def job_args_get(session: APISession, job_id: str) -> dict:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}/args"
    response = session.get(endpoint)
    return dict(response.json() or {})


def job_get(session: APISession, job_id: str, timeout: float | None = None) -> Job:
    """
    Get a job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.
    timeout : float | None, optional
        Seconds the request may take. When omitted, the session timeout applies.

    Returns
    -------
    Job
        The job record.
    """
    endpoint = f"v1/jobs/{job_id}"
    if timeout is None:
        response = session.get(endpoint)
    else:
        response = session.get(endpoint, timeout=timeout)
    return TypeAdapter(Job).validate_python(response.json())


def jobs_list(
    session: APISession,
    status: str | None = None,
    job_type: str | None = None,
    assay_id: str | None = None,
    more_recent_than: str | None = None,
    limit: int | None = None,
    page_size: int | None = None,
    page_offset: int | None = None,
    *,
    name: str | None = None,
) -> list[Job]:
    """
    Retrieve a list of jobs filtered by specific criteria.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    status : str, optional
        Filter by job status. If None, jobs of all statuses are retrieved. Default is None.
    job_type : str, optional
        Filter by Filter. If None, jobs of all types are retrieved. Default is None.
    assay_id : str, optional
        Filter by assay. If None, jobs for all assays are retrieved. Default is None.
    more_recent_than : str, optional
        Retrieve jobs that are more recent than a specified date. If None, no date filtering is applied. Default is None.
    limit : int, optional
        Deprecated alias for `page_size`. Forwarded to the server as `limit` for backwards compat.
    page_size : int, optional
        Maximum number of jobs to return per page.
    page_offset : int, optional
        Number of jobs to skip from the start of the result set.
    name : str, optional
        Filter by the job's name, matched exactly. If None, jobs of all names are retrieved. Default is None.

    Returns
    -------
    List[Job]
        A list of Job instances that match the specified criteria.
    """
    endpoint = "v1/jobs"

    params = {}
    if status is not None:
        params["status"] = status
    if job_type is not None:
        params["job_type"] = job_type
    if assay_id is not None:
        params["assay_id"] = assay_id
    if name is not None:
        params["name"] = name
    if more_recent_than is not None:
        params["more_recent_than"] = more_recent_than
    if limit is not None:
        params["limit"] = limit
    if page_size is not None:
        params["page_size"] = page_size
    if page_offset is not None:
        params["page_offset"] = page_offset

    response = session.get(endpoint, params=params)
    # return jobs, not futures
    return TypeAdapter(list[Job]).validate_python(response.json())
