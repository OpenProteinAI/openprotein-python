# Jobs and job centric flows


from datetime import datetime
from typing import List, Optional, Union
import concurrent.futures
import time
from enum import Enum

import tqdm
import pydantic
from pydantic import BaseModel, ConfigDict

from openprotein.errors import TimeoutException
from openprotein.base import APISession
import openprotein.config as config



class NewModel(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )

class JobStatus(str, Enum):
    PENDING: str = "PENDING"
    RUNNING: str = "RUNNING"
    SUCCESS: str = "SUCCESS"
    FAILURE: str = "FAILURE"
    RETRYING: str = "RETRYING"
    CANCELED: str = "CANCELED"

    def done(self):
        return (
            (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)
        )  # noqa: E501

    def cancelled(self):
        return self is self.CANCELED



class Job(NewModel):
    status: JobStatus
    job_id: str
    job_type: str
    created_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    prerequisite_job_id: Optional[str]  = None
    progress_message: Optional[str] = None
    progress_counter: Optional[int] = 0
    num_records: Optional[int] = None

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
        #if progress is not None:  # Check None before comparison
        if progress is None:
            if job.status == JobStatus.PENDING:
                progress = 5
            if job.status == JobStatus.RUNNING:
                progress = 25
        if job.status in [JobStatus.SUCCESS, JobStatus.FAILURE]:
            progress = 100
        return progress or 0 # never None

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
                #pbar.update(1)
                #pbar.set_postfix({"status": job.status})
                progress = self._update_progress(job)
                pbar.n = progress
                pbar.set_postfix({"status": job.status})
                #pbar.refresh()
                # print(f'Retry {retries}, status={self.job.status}, time elapsed {time.time() - start_time:.2f}')
            time.sleep(interval)
            job = job.refresh(session)

        if verbose:
            #pbar.update(1)
            #pbar.set_postfix({"status": job.status})
            
            progress = self._update_progress(job)
            pbar.n = progress
            pbar.set_postfix({"status": job.status})
            #pbar.refresh()

        return job

    wait_until_done = wait


def load_job(session: APISession, job_id: str) -> Job:
    """
    Reload a Submitted job to resume from where you left off!


    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    job_id : str
        The identifier of the job whose details are to be loaded.

    Returns
    -------
    Job
        Job

    Raises
    ------
    HTTPError
        If the request to the server fails.

    """
    endpoint = f"v1/jobs/{job_id}"
    response = session.get(endpoint)
    return pydantic.parse_obj_as(Job, response.json())


def job_get(session: APISession, job_id) -> Job:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}"
    response = session.get(endpoint)
    return Job(**response.json())

def job_args_get(session: APISession, job_id) -> dict:
    """Get job."""
    endpoint = f"v1/jobs/{job_id}/args"
    response = session.get(endpoint)
    return dict(**response.json())

def jobs_list(
    session: APISession,
    status=None,
    job_type=None,
    assay_id=None,
    more_recent_than=None,
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
    return pydantic.parse_obj_as(List[Job], response.json())


class JobsAPI:
    """API wrapper to get jobs."""

    def __init__(self, session: APISession):
        self.session = session

    def list(
        self, status=None, job_type=None, assay_id=None, more_recent_than=None
    ) -> List[Job]:
        """ List jobs"""
        return jobs_list(
            self.session,
            status=status,
            job_type=job_type,
            assay_id=assay_id,
            more_recent_than=more_recent_than,
        )

    def get(self, job_id) -> Job:
        """get Job by ID"""
        return job_get(self.session, job_id)

    def wait(
        self, job: Job, interval=config.POLLING_INTERVAL, timeout=None, verbose=False
    ):
        return job.wait(
            self.session, interval=interval, timeout=timeout, verbose=verbose
        )


class AsyncJobFuture:
    def __init__(self, session: APISession, job: Union[Job, str]):
        if isinstance(job, str):
            job = job_get(session, job)
        self.session = session
        self.job = job

    def refresh(self):
        """ refresh job status"""
        self.job = self.job.refresh(self.session)

    @property
    def status(self):
        return self.job.status

    @property
    def progress(self):
        return self.job.progress_counter or 0
    @property
    def num_records(self):
        return self.job.num_records

    def done(self):
        return self.job.done()

    def cancelled(self):
        return self.job.cancelled()

    def get(self, verbose=False):
        raise NotImplementedError()

    def wait_until_done(
        self, interval=config.POLLING_INTERVAL, timeout=None, verbose=False
    ):
        """
        Wait for job to complete. Do not fetch results (unlike wait())

        Args:
            interval (int, optional): time between polling. Defaults to config.POLLING_INTERVAL.
            timeout (int, optional): max time to wait. Defaults to None.
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            results: results of job
        """
        job = self.job.wait(
            self.session, interval=interval, timeout=timeout, verbose=verbose
        )
        self.job = job
        return self.done()

    def wait(self, interval: int=config.POLLING_INTERVAL, timeout: int=None, verbose:bool=False):
        """
        Wait for job to complete, then fetch results.

        Args:
            interval (int, optional): time between polling. Defaults to config.POLLING_INTERVAL.
            timeout (int, optional): max time to wait. Defaults to None.
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            results: results of job
        """
        time.sleep(1) # buffer for BE to register job
        job = self.job.wait(
            self.session, interval=interval, timeout=timeout, verbose=verbose
        )
        self.job = job
        return self.get(verbose=verbose)


class StreamingAsyncJobFuture(AsyncJobFuture):
    def stream(self):
        raise NotImplementedError()

    def get(self, verbose=False):
        generator = self.stream()
        if verbose:
            total = None
            if hasattr(self, "__len__"):
                total = len(self)
            generator = tqdm.tqdm(
                generator, desc="Retrieving", total=total, position=0, mininterval=1.0
            )
        return [entry for entry in generator]


class MappedAsyncJobFuture(StreamingAsyncJobFuture):
    def __init__(
        self, session: APISession, job: Job, max_workers=config.MAX_CONCURRENT_WORKERS
    ):
        """
        Retrieve results from asynchronous, mapped endpoints. Use `max_workers` > 0 to enable concurrent retrieval of multiple pages.
        """
        super().__init__(session, job)
        self.max_workers = max_workers
        self._cache = {}

    def keys(self):
        raise NotImplementedError()

    def get_item(self, k):
        raise NotImplementedError()

    def stream_sync(self):
        for k in self.keys():
            v = self[k]
            yield k, v

    def stream_parallel(self):
        num_workers = self.max_workers

        def process(k):
            v = self[k]
            return k, v

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for k in self.keys():
                if k in self._cache:
                    yield k, self._cache[k]
                else:
                    f = executor.submit(process, k)
                    futures.append(f)

            for f in concurrent.futures.as_completed(futures):
                yield f.result()

    def stream(self):
        if self.max_workers > 0:
            return self.stream_parallel()
        return self.stream_sync()

    def __getitem__(self, k):
        if k in self._cache:
            return self._cache[k]
        v = self.get_item(k)
        self._cache[k] = v
        return v

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return self.stream()


class PagedAsyncJobFuture(StreamingAsyncJobFuture):
    DEFAULT_PAGE_SIZE = 1024

    def __init__(
        self,
        session: APISession,
        job: Job,
        page_size=None,
        num_records=None,
        max_workers=config.MAX_CONCURRENT_WORKERS,
    ):
        """
        Retrieve results from asynchronous, paged endpoints. Use `max_workers` > 0 to enable concurrent retrieval of multiple pages.
        """
        if page_size is None:
            page_size = self.DEFAULT_PAGE_SIZE
        super().__init__(session, job)
        self.page_size = page_size
        self.max_workers = max_workers
        self._num_records = num_records

    def get_slice(self, start, end):
        raise NotImplementedError()

    def stream_sync(self):
        step = self.page_size
        num_returned = step
        offset = 0
        while num_returned >= step:
            result_page = self.get_slice(offset, offset + step)
            for result in result_page:
                yield result
            num_returned = len(result_page)
            offset += num_returned

    # TODO - check the number of results, or store it somehow, so that we don't need
    # to check the number of returned entries to see if we're finished (very awkward when using concurrency)
    def stream_parallel(self):
        step = self.page_size
        offset = 0

        num_workers = self.max_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # submit the paged requests
            futures = []
            for _ in range(num_workers * 2):
                f = executor.submit(self.get_slice, offset, offset + step)
                futures.append(f)
                offset += step

            # until we've retrieved all pages (known by retrieving a page with less than the requested number of records)
            done = False
            while not done:
                futures_next = []
                # iterate the futures and submit new requests as needed
                for f in concurrent.futures.as_completed(futures):
                    result_page = f.result()
                    # check if we're done, meaning the result page is not full
                    done = done or len(result_page) < step
                    # if we aren't done, submit another request
                    if not done:
                        f = executor.submit(self.get_slice, offset, offset + step)
                        futures_next.append(f)
                        offset += step
                    # yield the results from this page
                    for result in result_page:
                        yield result
                # update the list of futures and wait on them again
                futures = futures_next

    def stream(self):
        if self.max_workers > 0:
            return self.stream_parallel()
        return self.stream_sync()
