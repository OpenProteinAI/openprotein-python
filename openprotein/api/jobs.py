from openprotein.base import APISession
import openprotein.config as config

from enum import Enum
import pydantic
from datetime import datetime
from typing import List, Optional, Dict, Union
import tqdm
import concurrent.futures
from requests import HTTPError
from ..errors import TimeoutException
from ..models import Job, JobStatus


def jobs_list(
        session: APISession,
        status=None,
        job_type=None,
        assay_id=None,
        more_recent_than=None
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



class AsyncJobFuture:
    def __init__(self, session: APISession, job: Union[Job, str]):
        if type(job) is str:
            job = job_get(session, job)
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

    def get(self, verbose=False):
        raise NotImplementedError()
    
    def wait_until_done(self, interval=config.POLLING_INTERVAL, timeout=None, verbose=False):
        job = self.job.wait(self.session, interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.done()

    def wait(self, interval=config.POLLING_INTERVAL, timeout=None, verbose=False):
        job = self.job.wait(self.session, interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.get(verbose=verbose)


class StreamingAsyncJobFuture(AsyncJobFuture):
    def stream(self):
        raise NotImplementedError()

    def get(self, verbose=False):
        generator = self.stream()
        if verbose:
            generator = tqdm.tqdm(generator, desc='Retrieving')
        return [entry for entry in generator]


class PagedAsyncJobFuture(StreamingAsyncJobFuture):
    DEFAULT_PAGE_SIZE = 1024

    def __init__(self, session: APISession, job: Job, page_size=None, max_workers=config.MAX_CONCURRENT_WORKERS):
        """
        Retrieve results from asynchronous, paged endpoints. Use `max_workers` > 0 to enable concurrent retrieval of multiple pages.
        """
        if page_size is None:
            page_size = self.DEFAULT_PAGE_SIZE
        super().__init__(session, job)
        self.page_size = page_size
        self.max_workers = max_workers

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
            for _ in range(num_workers*2):
                f = executor.submit(self.get_slice, offset, offset+step)
                futures.append(f)
                offset += step
            
            # until we've retrieved all pages (known by retrieving a page with less than the requested number of records)
            done = False
            while not done:
                futures_next = []
                # iterate the futures and submit new requests as needed
                for f in concurrent.futures.as_completed(futures):
                    try:
                        result_page = f.result()
                    except HTTPError:
                        # if getting the page failed with an HTTP error, it means the index was out of bounds
                        # TODO - this is a problem, because the request could have failed for other reasons
                        result_page = []
                    # check if we're done, meaning the result page is not full
                    done = (done or len(result_page) < step)
                    # if we aren't done, submit another request
                    if not done:
                        f = executor.submit(self.get_slice, offset, offset+step)
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
