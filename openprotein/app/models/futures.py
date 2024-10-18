"""Application futures for waiting for results from jobs."""

import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from types import UnionType
from typing import Collection, Generator

import tqdm
from openprotein import config
from openprotein.api import job as job_api
from openprotein.base import APISession
from openprotein.errors import TimeoutException
from openprotein.schemas import Job, JobStatus, JobType
from requests import Response
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Future(ABC):
    """
    Base class for all Futures returning results from a job.

    This base class should be directly inherited for class discovery for factory create.
    """

    session: APISession
    job: Job

    def __init__(self, session: APISession, job: Job):
        self.session = session
        self.job = job

    @classmethod
    def create(
        cls: type[Self],
        session: APISession,
        job_id: str | None = None,
        job: Job | None = None,
        response: Response | dict | None = None,
        **kwargs,
    ) -> Self:
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Parameters:
        - session: Session for API interactions.
        - job_id: The optional job_id of the Job to initialize this future with.
        - job: The optional Job to initialize this future with.
        - response: The optional response from a job request returning a job-like object.
        - **kwargs: Additional keyword arguments to pass to the Future class constructor.

        Returns:
        - An instance of the appropriate Future class.
        """

        # parse job
        # default to use job_id first
        if job_id is not None:
            # get job
            job = job_api.job_get(session=session, job_id=job_id)
        # set obj to parse using job or response
        obj = job or response
        if obj is None:
            raise ValueError("Expected job_id, job or response")

        # parse specific job
        job = Job.create(obj, **kwargs)

        # Dynamically discover all subclasses of FutureBase
        future_classes = Future.__subclasses__()

        # Find the Future class that matches the job
        for future_class in future_classes:
            if (
                type(job) == (future_type := future_class.__annotations__.get("job"))
                or isinstance(future_type, UnionType)
                and type(job) in future_type.__args__
            ):
                future = future_class(session=session, job=job, **kwargs)
                return future  # type: ignore - needed since type checker doesnt know subclass

        raise ValueError(f"Unsupported job type: {job.job_type}")

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self):
        return repr(self.job)

    @property
    def id(self) -> str:
        return self.job.job_id

    job_id = id

    @property
    def job_type(self) -> JobType:
        return self.job.job_type

    @property
    def status(self) -> JobStatus:
        return self.job.status

    @property
    def created_date(self) -> datetime:
        return self.job.created_date

    @property
    def start_date(self) -> datetime | None:
        return self.job.start_date

    @property
    def end_date(self) -> datetime | None:
        return self.job.end_date

    @property
    def progress_counter(self) -> int:
        return self.job.progress_counter or 0

    def done(self) -> bool:
        """Check if job is complete"""
        return self.status.done()

    def cancelled(self) -> bool:
        """check if job is cancelled"""
        return self.status.cancelled()

    def _update_progress(self, job: Job) -> int:
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

    def _refresh_job(self) -> Job:
        """Refresh and return internal specific job."""
        # dump extra kwargs to keep on refresh
        kwargs = {
            k: v for k, v in self.job.model_dump().items() if k not in Job.model_fields
        }
        job = Job.create(
            job_api.job_get(session=self.session, job_id=self.job_id), **kwargs
        )
        return job

    def refresh(self):
        """Refresh job status."""
        self.job = self._refresh_job()

    @abstractmethod
    def get(self, verbose: bool = False):
        raise NotImplementedError()

    def _wait_job(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Job:
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
            return job.status.done()

        pbar = None
        if verbose:
            pbar = tqdm.tqdm(total=100, desc="Waiting", position=0)

        job = self._refresh_job()
        while not is_done(job):
            if pbar is not None:
                # pbar.update(1)
                # pbar.set_postfix({"status": job.status})
                progress = self._update_progress(job)
                pbar.n = progress
                pbar.set_postfix({"status": job.status})
                # pbar.refresh()
                # print(f'Retry {retries}, status={self.job.status}, time elapsed {time.time() - start_time:.2f}')
            time.sleep(interval)
            job = self._refresh_job()

        if pbar is not None:
            # pbar.update(1)
            # pbar.set_postfix({"status": job.status})

            progress = self._update_progress(job)
            pbar.n = progress
            pbar.set_postfix({"status": job.status})
            # pbar.refresh()

        return job

    def wait_until_done(
        self, interval: int = config.POLLING_INTERVAL, timeout=None, verbose=False
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
        job = self._wait_job(interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.done()

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ):
        """
        Wait for job to complete, then fetch results.

        Args:
            interval (int, optional): time between polling. Defaults to config.POLLING_INTERVAL.
            timeout (int, optional): max time to wait. Defaults to None.
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            results: results of job
        """
        time.sleep(1)  # buffer for BE to register job
        job = self._wait_job(interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.get()


class StreamingFuture(ABC):
    @abstractmethod
    def stream(self) -> Generator:
        raise NotImplementedError()

    def get(self, verbose: bool = False) -> list:
        generator = self.stream()
        if verbose:
            total = None
            if hasattr(self, "__len__"):
                total = len(self)  # type: ignore - static type checker doesnt know
            generator = tqdm.tqdm(
                generator, desc="Retrieving", total=total, position=0, mininterval=1.0
            )
        return [entry for entry in generator]


class MappedFuture(StreamingFuture, ABC):
    """Base future class for returning results from jobs with a mapping for keys (e.g. sequence) to results (e.g. embeddings)."""

    def __init__(
        self,
        session: APISession,
        job: Job,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        """
        Retrieve results from asynchronous, mapped endpoints.

        Use `max_workers` > 0 to enable concurrent retrieval of multiple pages.
        """
        self.session = session
        self.job = job
        self.max_workers = max_workers
        self._cache = {}

    @abstractmethod
    def keys(self):
        raise NotImplementedError()

    @abstractmethod
    def get_item(self, k):
        raise NotImplementedError()

    def stream_sync(self):
        """Stream the results back in-sync."""
        for k in self.keys():
            v = self[k]
            yield k, v

    def stream_parallel(self):
        """Stream the results back in parallel."""
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
        """Stream results."""
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


class PagedFuture(StreamingFuture, ABC):
    """Base future class for returning results from jobs which have paged results."""

    DEFAULT_PAGE_SIZE = 1024

    def __init__(
        self,
        session: APISession,
        job: Job,
        page_size: int | None = None,
        num_records: int | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        """
        Retrieve results from asynchronous, paged endpoints.

        Use `max_workers` > 0 to enable concurrent retrieval of multiple pages.
        """
        if page_size is None:
            page_size = self.DEFAULT_PAGE_SIZE
        self.session = session
        self.job = job
        self.page_size = page_size
        self.max_workers = max_workers
        self._num_records = num_records

    @abstractmethod
    def get_slice(self, start: int, end: int, **kwargs) -> Collection:
        raise NotImplementedError()

    def stream_sync(self):
        step = self.page_size
        num_returned = step
        offset = 0
        while num_returned >= step:
            result_page = self.get_slice(start=offset, end=offset + step)
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
            futures: dict[concurrent.futures.Future, int] = {}
            index: int = 0
            for _ in range(num_workers * 2):
                f = executor.submit(self.get_slice, offset, offset + step)
                futures[f] = index
                index += 1
                offset += step

            # until we've retrieved all pages (known by retrieving a page with less than the requested number of records)
            done = False
            while not done:
                results: list[list | None] = [None] * len(futures)
                futures_next: dict[concurrent.futures.Future, int] = {}
                index_next: int = 0
                next_result_index = 0
                # iterate the futures and submit new requests as needed
                for f in concurrent.futures.as_completed(futures):
                    index = futures[f]
                    result_page = f.result()
                    results[index] = result_page
                    # check if we're done, meaning the result page is not full
                    done = done or len(result_page) < step
                    # if we aren't done, submit another request
                    if not done:
                        f = executor.submit(self.get_slice, offset, offset + step)
                        futures_next[f] = index_next
                        index_next += 1
                        offset += step
                    # yield the results from this page
                    while (
                        next_result_index < len(results)
                        and results[next_result_index] is not None
                    ):
                        result_page = results[next_result_index]
                        assert result_page is not None  # checked above
                        for result in result_page:
                            yield result
                        next_result_index += 1
                # update the list of futures and wait on them again
                futures = futures_next

    def stream(self):
        if self.max_workers > 0:
            return self.stream_parallel()
        return self.stream_sync()


class InvalidFutureError(Exception):
    """Error thrown if unexpected future is created from job."""

    def __init__(self, future: Future, expected: type[Future]):
        self.future = future
        self.expected = future
        self.message = f"Expected future of type {expected}, got {type(future)}"
        super().__init__(self.message)
