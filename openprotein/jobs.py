from datetime import datetime
from typing import Optional, Literal
import time
from openprotein.pydantic import BaseModel, Field
from openprotein.errors import TimeoutException
from openprotein.base import APISession
import openprotein.config as config
import tqdm
from requests import Response
from openprotein.config import JOB_REGISTRY
from openprotein.schemas import JobStatus, JobType


class Job(BaseModel):
    status: JobStatus
    job_id: Optional[str]  # must be optional as predict can return None
    # new emb service get doesnt have job_type
    job_type: Optional[Literal[tuple(member.value for member in JobType.__members__.values())]]  # type: ignore
    created_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    prerequisite_job_id: Optional[str] = None
    progress_message: Optional[str] = None
    progress_counter: Optional[int] = 0
    num_records: Optional[int] = None
    sequence_length: Optional[int] = None

    def refresh(self, session: APISession):
        """refresh job status"""
        return job_get(session, self.job_id)

    def done(self) -> bool:
        """Check if job is complete"""
        return self.status.done()

    def cancelled(self) -> bool:
        """check if job is cancelled"""
        return self.status.cancelled()

    def _update_progress(self, job) -> int:
        """update rules for jobs without counters"""
        progress = job.progress_counter
        # if progress is not None:  # Check None before comparison
        if progress is None:
            if job.status == JobStatus.PENDING:
                progress = 5
            if job.status == JobStatus.RUNNING:
                progress = 25
        if job.status in [JobStatus.SUCCESS, JobStatus.FAILURE]:
            progress = 100
        return progress or 0  # never None

    def wait(
        self,
        session: APISession,
        interval: int = config.POLLING_INTERVAL,
        timeout: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Wait for a job to finish, and then get the results.

        Args:
            session (APISession): Auth'd APIsession
            interval (int): Wait between polls (secs). Defaults to POLLING_INTERVAL
            timeout (int): Max. time to wait before raising error. Defaults to unlimited.
            verbose (bool, optional): print status updates. Defaults to False.

        Raises:
            TimeoutException: _description_

        Returns:
            _type_: _description_
        """
        start_time = time.time()

        def is_done(job: Job):
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutException(
                        f"Wait time exceeded timeout {timeout}, waited {elapsed_time}"
                    )
            return job.done()

        pbar = None
        if verbose:
            pbar = tqdm.tqdm(total=100, desc="Waiting", position=0)

        job = self.refresh(session)
        while not is_done(job):
            if verbose:
                # pbar.update(1)
                # pbar.set_postfix({"status": job.status})
                progress = self._update_progress(job)
                pbar.n = progress
                pbar.set_postfix({"status": job.status})
                # pbar.refresh()
                # print(f'Retry {retries}, status={self.job.status}, time elapsed {time.time() - start_time:.2f}')
            time.sleep(interval)
            job = job.refresh(session)

        if verbose:
            # pbar.update(1)
            # pbar.set_postfix({"status": job.status})

            progress = self._update_progress(job)
            pbar.n = progress
            pbar.set_postfix({"status": job.status})
            # pbar.refresh()

        return job

    wait_until_done = wait


class JobDetails(BaseModel):
    job_id: str
    job_type: str
    status: str


def register_job_type(job_type: str):
    def decorator(cls):
        JOB_REGISTRY[job_type] = cls
        return cls

    return decorator


@register_job_type(JobType.workflow_design)
class DesignJob(Job):
    job_id: Optional[str] = None
    job_type: Literal[JobType.workflow_design] = JobType.workflow_design


# old and new style names
@register_job_type(JobType.embeddings_svd)
@register_job_type(JobType.svd_fit)
@register_job_type(JobType.svd_embed)
class SVDJob(Job):
    job_type: Literal[JobType.embeddings_svd, JobType.svd_fit, JobType.svd_embed] = (
        JobType.embeddings_svd
    )


class ResultsParser(BaseModel):
    """Polymorphic class to parse results from GET correctly"""

    __root__: Job = Field(...)

    @classmethod
    def parse_obj(cls, obj, **kwargs):
        try:
            if isinstance(obj, Response):
                obj = obj.json()
            # Determine the correct job class based on the job_type field
            job_type = obj.get("job_type")
            job_class = JOB_REGISTRY.get(job_type)
            if job_class:
                return job_class.parse_obj(obj, **kwargs)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
        except Exception as e:
            # default to Job class
            return Job.parse_obj(obj, **kwargs)


class SpecialPredictJob(Job):
    """special case of Job for predict that doesnt require job_id"""

    job_id: Optional[str] = None


def job_args_get(session: APISession, job_id) -> dict:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}/args"
    response = session.get(endpoint)
    return dict(**response.json())


def job_get(session: APISession, job_id) -> Job:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}"
    response = session.get(endpoint)
    return ResultsParser.parse_obj(response)
