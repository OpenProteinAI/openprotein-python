from datetime import datetime

from openprotein import config
from openprotein.base import APISession

from . import api
from .futures import Future
from .schemas import Job, JobStatus, JobType


class JobsAPI:
    """API interface to get jobs."""

    def __init__(self, session: APISession):
        self.session = session

    def list(
        self,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        assay_id: str | None = None,
        more_recent_than: datetime | str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs."""
        more_recent_than_str = (
            more_recent_than.isoformat()
            if isinstance(more_recent_than, datetime)
            else more_recent_than
        )
        return [
            Job.create(j)
            for j in api.jobs_list(
                self.session,
                status=status,
                job_type=job_type,
                assay_id=assay_id,
                more_recent_than=more_recent_than_str,
                limit=limit,
            )
        ]

    def get_job(self, job_id: str) -> Job:
        return api.job_get(session=self.session, job_id=job_id)

    def get(self, job_id: str, verbose: bool = False) -> Future:  # Job:
        """
        Get job by ID.

        Notes
        -----
        This retrieves the job and loads it as a future so you can do `wait` and `get`.
        """
        return self.__load(job_id=job_id)
        # return Job.create(job.job_get(session=self.session, job_id=job_id))

    def __load(self, job_id: str) -> Future:
        """Loads a job by ID and returns the future."""
        return Future.create(session=self.session, job_id=job_id)

    def wait(
        self,
        future: Future,
        interval=config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ):
        """Waits on a job result."""
        return future.wait(interval=interval, timeout=timeout, verbose=verbose)
