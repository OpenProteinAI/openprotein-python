from openprotein.base import APISession
import openprotein.config as config

from enum import Enum
import pydantic
from datetime import datetime
from typing import List, Optional, Dict
import time
import tqdm


class JobStatus(str, Enum):
    PENDING: str = 'PENDING'
    RUNNING: str = 'RUNNING'
    SUCCESS: str = 'SUCCESS'
    FAILURE: str = 'FAILURE'
    RETRYING: str = 'RETRYING'
    CANCELED: str = 'CANCELED'

    def done(self):
        return (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)

    def cancelled(self):
        return self is self.CANCELED


class Job(pydantic.BaseModel):
    status: JobStatus
    job_id: str
    job_type: str
    created_date: Optional[datetime]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    prerequisite_job_id: Optional[str]
    progress_message: Optional[str]
    progress_counter: Optional[int]

    def refresh(self, session: APISession):
        return job_get(session, self.job_id)

    def done(self):
        return self.status.done()

    def cancelled(self):
        return self.status.cancelled()

    def wait(self, session: APISession, interval=config.POLLING_INTERVAL, timeout=None, verbose=False):
        start_time = time.time()
        
        def is_done(job: Job):
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutException(f'Wait time exceeded timeout {timeout}, waited {elapsed_time}')
            return job.done()
        
        pbar = None
        if verbose:
            pbar = tqdm.tqdm()

        job = self.refresh(session)
        while not is_done(job):
            if verbose:
                pbar.update(1)
                pbar.set_postfix({'status': job.status})
                #print(f'Retry {retries}, status={self.job.status}, time elapsed {time.time() - start_time:.2f}')
            time.sleep(interval)
            job = job.refresh(session)
        
        if verbose:
            pbar.update(1)
            pbar.set_postfix({'status': job.status})

        return job


def jobs_list(
        session: APISession,
        status=None,
        job_type=None,
        assay_id=None,
        more_recent_than=None
    ) -> List[Job]:
    endpoint = 'v1/jobs'

    params = {}
    if status is not None:
        params['status'] = status
    if job_type is not None:
        params['job_type'] = job_type
    if assay_id is not None:
        params['assay_id'] = assay_id
    if more_recent_than is not None:
        params['more_recent_than'] = more_recent_than
    
    response = session.get(endpoint, params=params)
    return pydantic.parse_obj_as(List[Job], response.json())


def job_get(session: APISession, job_id) -> Job:
    endpoint = f'v1/jobs/{job_id}'
    response = session.get(endpoint)
    return Job(**response.json())


class JobsAPI:
    def __init__(self, session: APISession):
        self.session = session

    def list(self, status=None, job_type=None, assay_id=None, more_recent_than=None) -> List[Job]:
        return jobs_list(self.session, status=status, job_type=job_type, assay_id=assay_id, more_recent_than=more_recent_than)

    def get(self, job_id) -> Job:
        return job_get(self.session, job_id)

    def wait(self, job: Job, interval=config.POLLING_INTERVAL, timeout=None, verbose=False):
        return job.wait(self.session, interval=interval, timeout=timeout, verbose=verbose)


class TimeoutException(Exception):
    pass


class AsyncJobFuture:
    def __init__(self, session: APISession, job: Job):
        self.session = session
        self.job = job

    def refresh(self):
        self.job = self.job.refresh(self.session)

    @property
    def status(self):
        return self.job.status

    def done(self):
        return self.job.done()

    def cancelled(self):
        return self.job.cancelled()

    def get(self):
        raise NotImplementedError()

    def wait(self, interval=config.POLLING_INTERVAL, timeout=None, verbose=False):
        job = self.job.wait(self.session, interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.get()


class StreamingAsyncJobFuture(AsyncJobFuture):
    def get(self):
        return [entry for entry in self.stream()]