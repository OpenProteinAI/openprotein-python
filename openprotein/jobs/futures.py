"""Application futures for waiting for results from jobs."""

import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from types import UnionType
from typing import Collection, Generator

import tqdm
from requests import Response
from typing_extensions import Self

from openprotein import config
from openprotein.base import APISession
from openprotein.errors import TimeoutException
from openprotein.jobs.schemas import Job, JobStatus, JobType

from . import api

logger = logging.getLogger(__name__)


class Future(ABC):
    """
    Base class for all Futures returning results from a job.
    """

    # NOTE: This base class should be directly inherited for class discovery by the factory `create` method.
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
        """Create an instance of the appropriate Future class based on the job type.

        Parameters
        ----------
        session : APISession
            Session for API interactions.
        job_id : str | None, optional
            The ID of the Job to initialize this future with.
        job : Job | None, optional
            The Job object to initialize this future with.
        response : Response | dict | None, optional
            The response from a job request returning a job-like object.
        **kwargs
            Additional keyword arguments to pass to the Future class constructor.

        Returns
        -------
        Self
            An instance of the appropriate Future class.

        Raises
        ------
        ValueError
            If `job_id`, `job`, and `response` are all None.
        ValueError
            If an appropriate Future subclass cannot be found for the job type.

        :meta private:
        """
        # parse job
        # default to use job_id first
        if job_id is not None:
            # get job
            job = api.job_get(session=session, job_id=job_id)
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
                if isinstance(future_class.__dict__.get("create"), classmethod):
                    future = future_class.create(session=session, job=job, **kwargs)
                else:
                    future = future_class(session=session, job=job, **kwargs)
                return future  # type: ignore - needed since type checker doesnt know subclass

        raise ValueError(f"Unsupported job type: {job.job_type}")

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self):
        return repr(self.job)

    @property
    def id(self) -> str:
        """The unique identifier of the job."""
        return self.job.job_id

    job_id = id

    @property
    def job_type(self) -> str:
        """The type of the job."""
        return self.job.job_type

    @property
    def status(self) -> JobStatus:
        """The current status of the job."""
        return self.job.status

    @property
    def created_date(self) -> datetime:
        """The creation timestamp of the job."""
        return self.job.created_date

    @property
    def start_date(self) -> datetime | None:
        """The start timestamp of the job."""
        return self.job.start_date

    @property
    def end_date(self) -> datetime | None:
        """The end timestamp of the job."""
        return self.job.end_date

    @property
    def progress_counter(self) -> int:
        """The progress counter of the job."""
        return self.job.progress_counter or 0

    def done(self) -> bool:
        """Check if the job has completed.

        Returns
        -------
        bool
            True if the job is done, False otherwise.

        """
        return self.status.done()

    def cancelled(self) -> bool:
        """Check if the job has been cancelled.

        Returns
        -------
        bool
            True if the job is cancelled, False otherwise.

        """
        return self.status.cancelled()

    def _update_progress(self, job: Job) -> int:
        """Update progress for jobs that may not have explicit counters.

        Parameters
        ----------
        job : Job
            The job object to update progress from.

        Returns
        -------
        int
            The calculated progress value (0-100).

        """
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
        """Refresh and return the internal job object.

        Returns
        -------
        Job
            The refreshed job object.

        """
        # dump extra kwargs to keep on refresh
        kwargs = {
            k: v for k, v in self.job.model_dump().items() if k not in Job.model_fields
        }
        job = Job.create(
            api.job_get(session=self.session, job_id=self.job_id), **kwargs
        )
        return job

    def refresh(self):
        """Refresh the job status and internal job object."""
        self.job = self._refresh_job()

    @abstractmethod
    def get(self, verbose: bool = False, **kwargs):
        """
        Return the results from this job.

        Parameters
        ----------
        verbose : bool, optional
            Flag to enable verbose output, by default False.
        **kwargs
            Additional keyword arguments.
        """
        raise NotImplementedError()

    def _wait_job(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Job:
        """Wait for a job to finish and return the final job object.

        Parameters
        ----------
        interval : float, optional
            Time in seconds to wait between polls.
            Defaults to `config.POLLING_INTERVAL`.
        timeout : int | None, optional
            Maximum time in seconds to wait before raising an error.
            Defaults to None (unlimited).
        verbose : bool, optional
            If True, print status updates. Defaults to False.

        Returns
        -------
        Job
            The completed job object.

        Raises
        ------
        TimeoutException
            If the wait time exceeds the specified timeout.

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
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ):
        """Wait for the job to complete.

        Parameters
        ----------
        interval : float, optional
            Time in seconds between polling. Defaults to `config.POLLING_INTERVAL`.
        timeout : int, optional
            Maximum time in seconds to wait. Defaults to None.
        verbose : bool, optional
            Verbosity flag. Defaults to False.

        Returns
        -------
        bool
            True if the job completed successfully.

        Notes
        -----
        This method does not fetch the job results, unlike `wait()`.

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
        """Wait for the job to complete, then fetch results.

        Parameters
        ----------
        interval : int, optional
            Time in seconds between polling. Defaults to `config.POLLING_INTERVAL`.
        timeout : int | None, optional
            Maximum time in seconds to wait. Defaults to None.
        verbose : bool, optional
            Verbosity flag. Defaults to False.

        Returns
        -------
        Any
            The results of the job.

        """
        time.sleep(1)  # buffer for BE to register job
        job = self._wait_job(interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        return self.get()


class StreamingFuture(ABC):
    """Abstract base class for Futures that support streaming results."""

    @abstractmethod
    def stream(self, **kwargs) -> Generator:
        """Return the results from this job as a generator.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the streaming implementation.

        Returns
        -------
        Generator
            A generator that yields job results.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by a subclass.

        """
        raise NotImplementedError()

    def get(self, verbose: bool = False, **kwargs) -> list:
        """Return all results from the job by consuming the stream.

        Parameters
        ----------
        verbose : bool, optional
            If True, display a progress bar. Defaults to False.
        **kwargs
            Keyword arguments passed to the `stream` method.

        Returns
        -------
        list
            A list containing all results from the job.

        """
        generator = self.stream(**kwargs)
        if verbose:
            total = None
            if hasattr(self, "__len__"):
                total = len(self)  # type: ignore - static type checker doesnt know
            generator = tqdm.tqdm(
                generator, desc="Retrieving", total=total, position=0, mininterval=1.0
            )
        return [entry for entry in generator]


class MappedFuture(StreamingFuture, ABC):
    """Base future for jobs with a key-to-result mapping.

    This class provides methods to retrieve results from jobs where each result
    is associated with a unique key (e.g., sequence to embedding).

    """

    def __init__(
        self,
        session: APISession,
        job: Job,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        """Initialize the MappedFuture.

        Parameters
        ----------
        session : APISession
            The session for API interactions.
        job : Job
            The job to retrieve results from.
        max_workers : int, optional
            The number of workers for concurrent result retrieval.
            Defaults to `config.MAX_CONCURRENT_WORKERS`.

        Notes
        -----
        Use `max_workers` > 0 to enable concurrent retrieval.

        """
        self.session = session
        self.job = job
        self.max_workers = max_workers
        self._cache = {}

    @abstractmethod
    def __keys__(self):
        """Return the keys for the mapped results.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by a subclass.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_item(self, k):
        """Retrieve a single item by its key.

        Parameters
        ----------
        k
            The key of the item to retrieve.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by a subclass.

        """
        raise NotImplementedError()

    def stream_sync(self):
        """Stream the results synchronously.

        Yields
        ------
        tuple
            A tuple of (key, value) for each result.

        :meta private:
        """
        for k in self.__keys__():
            v = self[k]
            yield k, v

    def stream_parallel(self):
        """Stream the results in parallel using a thread pool.

        Yields
        ------
        tuple
            A tuple of (key, value) for each result.

        :meta private:
        """
        num_workers = self.max_workers

        def process(k):
            v = self[k]
            return k, v

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for k in self.__keys__():
                if k in self._cache:
                    yield k, self._cache[k]
                else:
                    f = executor.submit(process, k)
                    futures.append(f)

            for f in futures:
                yield f.result()

    def stream(self):
        """Retrieve results for this job as a stream.

        Returns
        -------
        Generator
            A generator that yields (key, value) tuples.

        """
        if self.max_workers > 0:
            return self.stream_parallel()
        return self.stream_sync()

    def __getitem__(self, k):
        """Get an item by key, using the cache if available.

        Parameters
        ----------
        k
            The key of the item to retrieve.

        Returns
        -------
        Any
            The value associated with the key.

        """
        if k in self._cache:
            return self._cache[k]
        v = self.get_item(k)
        self._cache[k] = v
        return v

    def __len__(self):
        """Return the total number of items."""
        return len(self.__keys__())

    def __iter__(self):
        """Return an iterator over the results."""
        return self.stream()


class PagedFuture(StreamingFuture, ABC):
    """Base future class for jobs which have paged results."""

    DEFAULT_PAGE_SIZE = 1024

    def __init__(
        self,
        session: APISession,
        job: Job,
        page_size: int | None = None,
        num_records: int | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        """Initialize the PagedFuture.

        Parameters
        ----------
        session : APISession
            The session for API interactions.
        job : Job
            The job to retrieve results from.
        page_size : int | None, optional
            The number of records per page. Defaults to `DEFAULT_PAGE_SIZE`.
        num_records : int | None, optional
            The total number of records expected.
        max_workers : int, optional
            Number of workers for concurrent page retrieval.
            Defaults to `config.MAX_CONCURRENT_WORKERS`.

        Notes
        -----
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
        """Retrieve a slice of results.

        Parameters
        ----------
        start : int
            The starting index of the slice.
        end : int
            The ending index of the slice.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Collection
            A collection of results for the specified slice.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by a subclass.

        """
        raise NotImplementedError()

    def stream_sync(self):
        """Stream results by fetching pages synchronously.

        Yields
        ------
        Any
            Individual results from the paged endpoint.

        :meta private:
        """
        step = self.page_size
        num_returned = step
        offset = 0
        while num_returned >= step:
            result_page = self.get_slice(start=offset, end=offset + step)
            for result in result_page:
                yield result
            num_returned = len(result_page)
            offset += num_returned

    def stream_parallel(self):
        """Stream results by fetching pages in parallel.

        Yields
        ------
        Any
            Individual results from the paged endpoint.

        Notes
        -----
        The number of results should be checked, or stored somehow, so that
        we don't need to check the number of returned entries to see if we're
        finished (very awkward when using concurrency).

        :meta private:
        """
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
        """Retrieve results for this job as a stream.

        Returns
        -------
        Generator
            A generator that yields job results.

        """
        if self.max_workers > 0:
            return self.stream_parallel()
        return self.stream_sync()


class InvalidFutureError(Exception):
    """Error for when an unexpected future is created from a job."""

    def __init__(self, future: Future, expected: type[Future]):
        """Initialize the InvalidFutureError.

        Parameters
        ----------
        future : Future
            The future instance that was created.
        expected : type[Future]
            The type of future that was expected.

        """
        self.future = future
        self.expected = future
        self.message = f"Expected future of type {expected}, got {type(future)}"
        super().__init__(self.message)
