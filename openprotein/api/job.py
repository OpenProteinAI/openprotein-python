"""Jobs and job-centric flows."""

from typing import List

from openprotein.base import APISession
from openprotein.schemas import Job
from pydantic import TypeAdapter

# def load_job(session: APISession, job_id: str) -> Future:
#     """
#     Reload a Submitted job to resume from where you left off!


#     Parameters
#     ----------
#     session : APISession
#         The current API session for communication with the server.
#     job_id : str
#         The identifier of the job whose details are to be loaded.

#     Returns
#     -------
#     Job
#         Job

#     Raises
#     ------
#     HTTPError
#         If the request to the server fails.

#     """
#     return Future.create(session=session, job_id=job_id)


def job_args_get(session: APISession, job_id: str) -> dict:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}/args"
    response = session.get(endpoint)
    return dict(**response.json())


def job_get(session: APISession, job_id: str) -> Job:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}"
    response = session.get(endpoint)
    return TypeAdapter(Job).validate_python(response.json())


def jobs_list(
    session: APISession,
    status: str | None = None,
    job_type: str | None = None,
    assay_id: str | None = None,
    more_recent_than: str | None = None,
) -> List[Job]:
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
    if more_recent_than is not None:
        params["more_recent_than"] = more_recent_than

    response = session.get(endpoint, params=params)
    # return jobs, not futures
    return TypeAdapter(List[Job]).validate_python(response.json())
