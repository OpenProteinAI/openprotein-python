"""Application futures for waiting for results from jobs."""

import concurrent.futures
import inspect
import io
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Collection,
    Generator,
    Generic,
    Iterator,
    KeysView,
    Set,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import requests
import tqdm
from requests import Response
from typing_extensions import Self

from openprotein import config
from openprotein.base import APISession
from openprotein.errors import (
    APIError,
    HTTPError,
    JobFailedException,
    TimeoutException,
)
from openprotein.jobs.schemas import Job, JobStatus, _UnitRecord
from openprotein.molecules import Complex, Structure

from . import _api, api
from .api import job_args_get

if TYPE_CHECKING:
    from openprotein.data.assaydataset import AssayDataset
    from openprotein.models.gp import GPModel
    from openprotein.models.schemas import MethodDescriptor
    from openprotein.models.svd import SVDModel
    from openprotein.models.umap import UMAPModel

    from .jobs import JobsAPI

__all__ = [
    "AttentionFuture",
    "CrossvalidateFuture",
    "CrossvalidateResults",
    "EmbeddingFuture",
    "F",
    "FitFuture",
    "Future",
    "FutureUnit",
    "GPFitFuture",
    "IndelScoreFuture",
    "InvalidFutureError",
    "K",
    "LogLikelihoodFuture",
    "LogitsFuture",
    "MappedFuture",
    "PagedFuture",
    "PredictionFuture",
    "SVDFitFuture",
    "SequenceGenerationFuture",
    "SingleSitePredictionFuture",
    "SingleSiteScoreFuture",
    "StreamingFuture",
    "StructureDesignFuture",
    "StructurePredictionFuture",
    "UMAPFitFuture",
    "UnknownJobFuture",
    "V",
]


def _jobs_api_for(session: APISession) -> "JobsAPI":
    from .jobs import JobsAPI

    return JobsAPI(session)


logger = logging.getLogger(__name__)

F = TypeVar("F")  # Future type
V = TypeVar("V")  # Concrete unit value returned by the future
K = TypeVar("K")  # Key that is connected to the value

_UNREAD = object()


class Future(ABC, Generic[V]):
    """Handle on a job: its identity, status and waiting, and its stored outputs.

    A job submitted through the models API, and any job whose type names a
    models task or a prompt sampling, stores outputs by unit position.
    ``units``, ``outputs``, ``unit``, ``get_item`` and ``stream`` read them,
    decoded by the codec each stored value names. Reads go to the data routes by
    output name; an unnamed output is the default the bare results route names.
    ``outputs`` come from the submitting method, else the model listing. For any
    other job these raise an `APIError`.

    ``dataset`` is the assay its sequences came from: as set, the one a future it
    is set to carries, or else the one its args name.
    """

    # NOTE: This base class should be directly inherited for class discovery by the factory `create` method.
    session: APISession
    job: Job

    _units: "list[FutureUnit] | None" = None
    _resolved_descriptor: "MethodDescriptor | None" = None
    _resolved_output: str | None = None
    _fetched_args: dict[str, Any] | None = None
    _submission: "_Submission | None" = None
    _dataset: "AssayDataset | Future | None | object" = _UNREAD
    _raises_when_canceled: ClassVar[bool] = False

    def __init__(self, session: APISession, job: Job):
        self.session = session
        self.job = job

    @classmethod
    def create(
        cls,
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

        :meta private:
        """
        if job_id is not None:
            job = api.job_get(session=session, job_id=job_id)
        obj = job or response
        if obj is None:
            raise ValueError("Expected job_id, job or response")

        job = Job.create(obj, **kwargs)

        from . import _dispatch

        if type(job) is Job and (
            _dispatch.is_models_job(job) or _dispatch.recognizes(job)
        ):
            return cast(Self, _dispatch.create(session, job, **kwargs))

        if _dispatch.is_prompt_sampling(job):
            from openprotein.data.prompts import Prompt

            return cast(Self, Prompt._from_job(session, job))

        def all_subclasses(cls: Type[F]) -> Set[Type[F]]:
            result = set()
            for sub in cls.__subclasses__():
                result.add(sub)
                result.update(all_subclasses(sub))
            return result

        future_classes = all_subclasses(Future) - {UnknownJobFuture}

        matches = []
        for future_class in sorted(
            future_classes, key=lambda item: (item.__module__, item.__qualname__)
        ):
            future_type = future_class.__annotations__.get("job")
            if (
                type(job) is future_type
                or isinstance(future_type, UnionType)
                and type(job) in future_type.__args__
            ):
                matches.append(future_class)

        if len(matches) > 1:
            logger.warning(
                "multiple future owners for %r: %s; selecting %s",
                job.job_type,
                ", ".join(f"{item.__module__}.{item.__qualname__}" for item in matches),
                f"{matches[0].__module__}.{matches[0].__qualname__}",
            )

        if matches:
            future_class = matches[0]
            assert not inspect.isabstract(future_class), (
                "Expected concrete class implementation!"
            )
            if isinstance(future_class.__dict__.get("create"), classmethod):
                future = future_class.create(session=session, job=job, **kwargs)
            else:
                future = future_class(session=session, job=job, **kwargs)
            return future  # type: ignore - needed since type checker doesnt know subclass

        # A job type with no typed Future resolves to one that reports its status.
        logger.info(
            "no typed future for job type %s; returning %s",
            job.job_type,
            UnknownJobFuture.__name__,
        )
        return UnknownJobFuture(session=session, job=job)  # type: ignore[return-value]

    def __repr__(self) -> str:
        inline = getattr(self.job, "args", None)
        args = inline if inline is not None else self._fetched_args
        shown = _compact(args) if args is not None else "<not loaded>"
        return (
            f"{type(self).__name__}(job_id={self.job_id!r}, status={self.status.value!r}, "
            f"args={shown})"
        )

    def __str__(self) -> str:
        return repr(self)

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
    def name(self) -> str | None:
        """The job's name, or its job id if the job has no name."""
        return self.job.name

    @property
    def description(self) -> str | None:
        """The job's description, or an empty string if the job has none."""
        return self.job.description

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

    @property
    def args(self) -> dict[str, Any]:
        """The job's validated request, loaded on first access."""
        inline = getattr(self.job, "args", None)
        if inline is not None:
            return inline
        if self._fetched_args is None:
            if self.job is None:
                raise APIError(f"{self.id} has no job to read args from")
            self._fetched_args = job_args_get(self.session, job_id=self.job_id) or {}
        return self._fetched_args

    @property
    def dataset(self) -> "AssayDataset | None":
        """The assay the job's sequences came from, or None; read on first access."""
        if isinstance(self._dataset, Future):
            self._dataset = self._dataset.dataset
        elif self._dataset is _UNREAD:
            from openprotein.models._inputs import as_dataset

            args = self.args
            assay_id = args.get("assay_id") or (args.get("training_dataset") or {}).get(
                "assay_id"
            )
            self._dataset = as_dataset(self.session, assay_id) if assay_id else None
        return cast("AssayDataset | None", self._dataset)

    @dataset.setter
    def dataset(self, dataset: "AssayDataset | Future | None") -> None:
        self._dataset = dataset

    def _require_outputs(self) -> None:
        """Raise unless this job's outputs are stored and read by position.

        A job submitted through the models API has them, as does a job whose
        type names a models job, a supported models task or a prompt sampling.
        """
        from . import _dispatch

        if self.job is None or not _dispatch.serves_outputs(self.job):
            raise APIError(f"job {self.id} has no stored outputs to read")

    def _method(self) -> tuple[str, str]:
        self._require_outputs()
        if self._submission is not None:
            return self._submission.model_id, self._submission.path.strip("/")

        from . import _dispatch

        provider, path = _dispatch.identity(self.job)
        if not provider or not path:
            self.refresh()
            provider, path = _dispatch.identity(self.job)
        if not provider or not path:
            raise ValueError(
                f"job {self.id} does not name a model and method in its job type"
            )
        return provider, path

    @property
    def _model_id(self) -> str:
        return self._method()[0]

    @property
    def _path(self) -> str:
        """The method's path on its model's card, e.g. ``batch/embeddings``."""
        if self._resolved_descriptor is not None:
            return self._resolved_descriptor.path
        return self._method()[1].strip("/")

    @property
    def units(self) -> "list[FutureUnit]":
        """One handle per submitted unit, in position order.

        Each handle holds the record the listing returned as its snapshot and
        selects the default output. The list is loaded on first access and read
        again after the job is polled.
        """
        self._require_outputs()
        if self._units is None:
            records = sorted(
                _jobs_api_for(self.session)._iter_job_units(self.job_id),
                key=lambda record: record.position,
            )
            self._units = []
            for record in records:
                unit = FutureUnit(self, record.position)
                unit._record = record
                self._units.append(unit)
        return self._units

    @property
    def _descriptor(self) -> "MethodDescriptor":
        """This job's method, from the submitting method or else the model listing."""
        if self._resolved_descriptor is None:
            model_id, path = self._method()
            from openprotein.models import _api as _models_api

            metadata = _models_api.get_model(self.session, model_id)
            for method in metadata.methods:
                if method.path == path.strip("/"):
                    self._resolved_descriptor = method
                    break
            else:
                raise ValueError(f"Model {model_id!r} has no method {path}")
        return self._resolved_descriptor

    @property
    def outputs(self) -> list[str]:
        """Output names of the submitted method, else of the model listing.

        A prompt sampling job lists the one output its results route names.
        """
        self._require_outputs()
        if self._resolved_descriptor is not None:
            return list(self._resolved_descriptor.outputs)
        from . import _dispatch

        assert self.job is not None
        if _dispatch.is_prompt_sampling(self.job):
            return [self._output_name(None)]
        return list(self._descriptor.outputs)

    @property
    def _default_output(self) -> str:
        """Default output of this job's method."""
        return self._descriptor.default_output

    def unit(self, position: int, output: str | None = None) -> "FutureUnit":
        """Handle on one unit's selected output.

        Creating the handle makes no request; ``None`` selects the default
        output.
        """
        self._require_outputs()
        return FutureUnit(self, position, output)

    def _positions(self) -> list[int]:
        return list(dict.fromkeys(sorted(unit.position for unit in self.units)))

    def _output_name(self, output: str | None) -> str:
        self._require_outputs()
        if output is not None:
            return output
        if self._resolved_output is None:
            self._resolved_output = _api.job_default_output(self.session, self.job_id)
        return self._resolved_output

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return value

    def get_item(self, position: int, output: str | None = None) -> Any:
        """Read one submitted position of ``output``, default the method's own."""
        self._require_outputs()
        name = self._output_name(output)
        entry = _jobs_api_for(self.session)._get_job_output(self.job_id, position, name)
        return self._convert_output(name, entry.position, entry.value)

    def stream(
        self,
        output: str | None = None,
        *,
        follow: bool = False,
        interval: float = config.POLLING_INTERVAL,
    ) -> Iterator[Any]:
        """Stream decoded outputs in position order; ``None`` is the default output.

        An emitted output grows while the job runs. ``follow`` keeps reading,
        every ``interval`` seconds, until it is complete.
        """
        self._require_outputs()

        def _stream() -> Iterator[Any]:
            name = self._output_name(output)
            entries = _jobs_api_for(self.session)._iter_job_outputs(
                self.job_id, name, follow=follow, interval=interval
            )
            for entry in entries:
                yield self._convert_output(name, entry.position, entry.value)

        return _stream()

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
        if progress is None:
            if job.status == JobStatus.PENDING:
                progress = 5
            if job.status == JobStatus.RUNNING:
                progress = 25
        if job.status in [JobStatus.SUCCESS, JobStatus.FAILURE]:
            progress = 100
        return progress or 0

    def _refresh_job(self, timeout: float | None = None) -> Job:
        """Refresh and return the internal job object.

        Parameters
        ----------
        timeout : float | None, optional
            Seconds the request may take. Defaults to the session timeout.

        Returns
        -------
        Job
            The refreshed job object.

        """
        # dump extra kwargs to keep on refresh
        kwargs = {
            k: v for k, v in self.job.model_dump().items() if k not in Job.model_fields
        }
        poll = {} if timeout is None else {"timeout": timeout}
        job = Job.create(
            api.job_get(session=self.session, job_id=self.job_id, **poll), **kwargs
        )
        return job

    def refresh(self):
        """Refresh the job status and internal job object."""
        self.job = self._refresh_job()
        self._units = None

    def rename(self, name: str | None = None, description: str | None = None):
        """Set this job's name and description.

        Only the arguments given are sent, so renaming leaves the description
        alone. Extra fields carried on the job are preserved, as on refresh.
        """
        # dump extra kwargs to keep on rename
        kwargs = {
            k: v for k, v in self.job.model_dump().items() if k not in Job.model_fields
        }
        self.job = Job.create(
            _api.job_update(
                session=self.session,
                job_id=self.job_id,
                name=name,
                description=description,
            ),
            **kwargs,
        )

    def _get(self, verbose: bool = False, **kwargs) -> V:
        """Return every position of the selected output.

        Parameters
        ----------
        verbose : bool, optional
            If True, display a progress bar. Defaults to False.
        **kwargs
            Keyword arguments passed to `stream`, such as ``output``.
        """
        generator = self.stream(**kwargs)
        if verbose:
            generator = tqdm.tqdm(
                generator,
                desc="Retrieving",
                total=len(self.units),
                position=0,
                mininterval=1.0,
            )
        return cast(V, list(generator))

    def get(self, verbose: bool = False, **kwargs) -> V:
        if self.status == JobStatus.FAILURE:
            raise JobFailedException(
                job_id=self.job_id, failure_message=self.job.failure_message
            )
        try:
            return self._get(verbose=verbose, **kwargs)
        except (APIError, JobFailedException):
            raise
        except Exception as e:
            raise APIError(f"Failed to retrieve results: {e}") from e

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
            Defaults to None (unlimited). The timeout is checked between
            polls, and a stalled poll is retried before it fails, so the
            wait can run past it.
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

        def remaining() -> float | None:
            if timeout is None:
                return None
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutException(
                    f"Wait time exceeded timeout {timeout}, waited {elapsed_time}"
                )
            return timeout - elapsed_time

        def poll() -> Job:
            budget = remaining()
            if budget is None:
                return self._refresh_job()
            session_timeout = self.session.timeout
            if isinstance(session_timeout, (int, float)):
                budget = min(budget, session_timeout)
            try:
                return self._refresh_job(timeout=budget)
            except requests.exceptions.RequestException as e:
                if timeout is None or time.time() - start_time < timeout:
                    raise
                raise TimeoutException(
                    f"Wait time exceeded timeout {timeout} while polling job {self.job_id}"
                ) from e

        def is_done(job: Job):
            remaining()
            return job.status.done()

        pbar = None
        if verbose:
            pbar = tqdm.tqdm(total=100, desc="Waiting", position=0)

        job = poll()
        while not is_done(job):
            if pbar is not None:
                progress = self._update_progress(job)
                pbar.n = progress
                pbar.set_postfix({"status": job.status})
            budget = remaining()
            time.sleep(interval if budget is None else min(interval, budget))
            job = poll()

        if pbar is not None:
            progress = self._update_progress(job)
            pbar.n = progress
            pbar.set_postfix({"status": job.status})

        return job

    def wait_until_done(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        """Wait for the job to complete.

        Parameters
        ----------
        interval : float, optional
            Time in seconds between polling. Defaults to `config.POLLING_INTERVAL`.
        timeout : int, optional
            Maximum time in seconds to wait. Defaults to None. The timeout is
            checked between polls, and a stalled poll is retried before it
            fails, so the wait can run past it.
        verbose : bool, optional
            Verbosity flag. Defaults to False.

        Returns
        -------
        bool
            True if the job reached a terminal status (success, failure or cancellation).

        Notes
        -----
        This method refreshes the stored job but does not fetch results, unlike
        `wait()`. Check `status` for success or failure. A polling timeout raises
        `TimeoutException`.

        """
        job = self._wait_job(interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        self._units = None
        return self.done()

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> V:
        """Wait for the job to complete, then fetch results.

        Parameters
        ----------
        interval : int, optional
            Time in seconds between polling. Defaults to `config.POLLING_INTERVAL`.
        timeout : int | None, optional
            Maximum time in seconds to wait. Defaults to None. The timeout is
            checked between polls, and a stalled poll is retried before it
            fails, so the wait can run past it.
        verbose : bool, optional
            Verbosity flag. Defaults to False.
        **kwargs
            Keyword arguments passed to `get`.

        Returns
        -------
        Any
            The results of the job.

        Raises
        ------
        TimeoutException
            If polling exceeds the timeout.
        JobFailedException
            If the job was canceled and is a result future or stores outputs.
            A failed job raises on the read.

        Notes
        -----
        A terminal status does not necessarily mean success. This method
        updates the stored job and calls ``get``, whose errors depend on the
        result reader.
        """
        time.sleep(1)  # allow the job to register before polling
        job = self._wait_job(interval=interval, timeout=timeout, verbose=verbose)
        self.job = job
        self._units = None
        if self.job is not None and self.status is JobStatus.CANCELED:
            from . import _dispatch

            if self._raises_when_canceled or _dispatch.serves_outputs(self.job):
                raise JobFailedException(self.job_id, "the job was canceled")
        return self.get(**kwargs)


class UnknownJobFuture(Future[None]):
    """A job whose type no installed Future claims.

    Identity, status and waiting work, so a caller can track the job. ``get``
    and ``wait`` report that this client has no typed support for the job type.
    The reads of stored outputs (``units``, ``get_item``, ``stream`` and the
    like) work when the job type stores outputs and raise an `APIError`
    otherwise.
    """

    job: Job

    def get(self, verbose: bool = False, **kwargs) -> None:
        """Report that this client cannot type this job's result."""
        raise NotImplementedError(
            f"no typed result support for job type {self.job.job_type!r}; "
            "upgrade openprotein-python to read it"
        )

    def _get(self, verbose: bool = False, **kwargs) -> None:
        return self.get(verbose=verbose, **kwargs)


class StreamingFuture(Future[list[V]], ABC, Generic[V]):
    """Abstract base class for Futures that support streaming results."""

    @abstractmethod
    def stream(  # ty: ignore[invalid-method-override]
        self, **kwargs
    ) -> Iterator[V]:
        """Return the results from this job as a generator.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the streaming implementation.

        Returns
        -------
        Generator
            A generator that yields job results.

        """
        raise NotImplementedError()

    def _get(self, verbose: bool = False, **kwargs) -> list[V]:
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

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[V]:
        return super().wait(interval, timeout, verbose, **kwargs)


class MappedFuture(StreamingFuture[tuple[K, V]], ABC, Generic[K, V]):
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
        super().__init__(session, job)
        self.max_workers = max_workers
        self._cache = {}

    @abstractmethod
    def __keys__(self) -> KeysView[K]:
        """Return the keys for the mapped results."""
        raise NotImplementedError()

    @abstractmethod
    def get_item(  # ty: ignore[invalid-method-override]
        self, k: K, **kwargs
    ) -> V:
        """Retrieve a single item by its key.

        Parameters
        ----------
        k
            The key of the item to retrieve.

        """
        raise NotImplementedError()

    def stream_sync(self, **kwargs) -> Iterator[tuple[K, V]]:
        """Stream the results synchronously.

        Yields
        ------
        tuple
            A tuple of (key, value) for each result.

        :meta private:
        """
        for k in self.__keys__():
            v = self.__getitem__(k, **kwargs)
            yield k, v

    def stream_parallel(self, **kwargs) -> Iterator[tuple[K, V]]:
        """Stream the results in parallel using a thread pool.

        Yields
        ------
        tuple
            A tuple of (key, value) for each result.

        :meta private:
        """
        num_workers = self.max_workers

        def process(k, **kwargs):
            v = self.__getitem__(k, **kwargs)
            return k, v

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for k in self.__keys__():
                f = executor.submit(process, k, **kwargs)
                futures.append(f)

            for f in futures:
                yield f.result()

    def stream(self, **kwargs) -> Iterator[tuple[K, V]]:
        """Retrieve results for this job as a stream.

        Returns
        -------
        Generator
            A generator that yields (key, value) tuples.

        """
        if self.max_workers > 0:
            return self.stream_parallel(**kwargs)
        return self.stream_sync(**kwargs)

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[tuple[K, V]]:
        return super().wait(interval, timeout, verbose, **kwargs)

    def __getitem__(self, k: K, **kwargs) -> V:
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
        if len(kwargs) > 0:
            _k = tuple([k] + [f"{k}={v}" for k, v in kwargs.items()])
        else:
            _k = k
        if _k in self._cache:
            return self._cache[_k]
        v = self.get_item(k, **kwargs)
        self._cache[_k] = v
        return v

    def __len__(self):
        """Return the total number of items."""
        return len(self.__keys__())

    def __iter__(self):
        """Return an iterator over the results."""
        return self.stream()


class PagedFuture(StreamingFuture[V], ABC):
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
    def get_slice(self, start: int, end: int, **kwargs) -> Collection[V]:
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

        """
        raise NotImplementedError()

    def stream_sync(self) -> Generator[V, None, None]:
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

    def stream_parallel(self) -> Generator[V, None, None]:
        """Stream results by fetching pages in parallel.

        Yields
        ------
        Any
            Individual results from the paged endpoint.

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

    def stream(self, **kwargs) -> Iterator[V]:
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


class FutureUnit(Generic[V]):
    """One ``(job, position, output)`` of a job that stores its outputs by position.

    A unit finishes on its own: its output is readable as soon as it is stored,
    whether or not its job's other units are done. ``job_id`` and ``position``
    need no request, nor does a named ``output``; the default output is resolved
    on first access.

    ``unit_id``, ``status`` and ``error`` are a snapshot: the record a listing
    returned, or one read of this position on first access. `refresh` reads the
    record again; reading the snapshot makes no request. A unit reports its
    ``status`` as a `JobStatus`.
    """

    def __init__(
        self,
        future: "Future",
        position: int,
        output: str | None = None,
    ) -> None:
        self._future = future
        self._position = position
        self._output = output
        self._record: _UnitRecord | None = None

    def __repr__(self) -> str:
        output = "<default>" if self._output is None else self._output
        return (
            f"FutureUnit(job_id={self.job_id!r}, position={self._position}, "
            f"output={output!r})"
        )

    @property
    def job_id(self) -> str:
        """Identifier of the job containing this unit."""
        return self._future.job_id

    @property
    def position(self) -> int:
        """Zero-based position of this unit within the job."""
        return self._position

    @property
    def output(self) -> str:
        """Name of the output selected for this unit."""
        if self._output is None:
            self._output = self._future._output_name(None)
        return self._output

    @property
    def unit_id(self) -> str:
        """The immutable authored identity of this submitted unit."""
        return self._snapshot().unit_id

    @property
    def status(self) -> JobStatus:
        """The unit's status in the snapshot."""
        return self._snapshot().status

    @property
    def error(self) -> dict[str, Any] | None:
        """The unit's error in the snapshot, or None."""
        return self._snapshot().error

    def _snapshot(self) -> _UnitRecord:
        if self._record is None:
            self.refresh()
        assert self._record is not None
        return self._record

    def refresh(self) -> None:
        """Read this position's record again."""
        self._record = _api.get_unit(self._future.session, self.job_id, self._position)

    def done(self) -> bool:
        """Whether the unit has finished, as the snapshot reports it.

        Returns
        -------
        bool
            True for a unit that succeeded, failed or was canceled.
        """
        return self.status.done()

    def _raise_if_unsuccessful(self) -> None:
        """Raise `JobFailedException` when the snapshot is failed or canceled."""
        record = self._record
        if record is None:
            return
        if record.status is JobStatus.FAILURE:
            message = (record.error or {}).get("message")
            raise JobFailedException(
                self.job_id,
                f"unit {self._position} failed: {message}"
                if message
                else f"unit {self._position} failed",
            )
        if record.status is JobStatus.CANCELED:
            raise JobFailedException(self.job_id, f"unit {self._position} was canceled")

    def get(self) -> V:
        """Fetch and decode this unit's selected output.

        Raises
        ------
        JobFailedException
            If the unit failed or was canceled, as the snapshot reports or as a
            refresh after a missing output shows.
        """
        self._raise_if_unsuccessful()
        try:
            return self._future.get_item(self._position, output=self.output)
        except HTTPError as error:
            if error.status_code != 404:
                raise
            try:
                self.refresh()
            except Exception as refresh_error:
                raise error from refresh_error
            self._raise_if_unsuccessful()
            raise

    def wait(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> V:
        """Block until this unit is done, then return its output.

        Parameters
        ----------
        interval : float, optional
            Time in seconds between reads. Defaults to `config.POLLING_INTERVAL`.
        timeout : int | None, optional
            Maximum time in seconds to wait. Defaults to None (unlimited).
        verbose : bool, optional
            If True, print the unit's status on each read. Defaults to False.

        Returns
        -------
        Any
            The unit's decoded output.

        Raises
        ------
        TimeoutException
            If the unit is unfinished after ``timeout`` seconds.
        JobFailedException
            If the unit failed or was canceled.
        """
        deadline = None if timeout is None else time.time() + timeout
        while True:
            self.refresh()
            status = self.status
            self._raise_if_unsuccessful()
            if status is JobStatus.SUCCESS:
                return self.get()
            if verbose:
                print(f"Waiting on unit {self._position}: {status.value}")
            remaining = None if deadline is None else deadline - time.time()
            if remaining is not None and remaining <= 0:
                raise TimeoutException(
                    f"unit {self._position} of job {self.job_id} is "
                    f"{status.value} after {timeout}s"
                )
            time.sleep(interval if remaining is None else min(interval, remaining))


_M = TypeVar("_M")


@dataclass(frozen=True)
class _Submission:
    """The model, method path and params a job was submitted with."""

    model_id: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)


class FitFuture(Future, Generic[_M]):
    """Handle on a job that fits a model; ``wait`` returns the fitted model.

    A fit has no model card method; each fit future lists its own outputs.
    """

    _family: ClassVar[str]
    _fit_outputs: ClassVar[tuple[str, ...]] = ()
    _raises_when_canceled = True

    @property
    def _descriptor(self) -> "MethodDescriptor":
        raise ValueError(f"fit job {self.id} has no model method; see outputs")

    @property
    def outputs(self) -> list[str]:
        """Output names of this kind of fit."""
        self._require_outputs()
        return list(self._fit_outputs)

    @property
    def _default_output(self) -> str:
        """The fit's first output."""
        if not self.outputs:
            raise ValueError(f"fit job {self.id} has no outputs")
        return self.outputs[0]

    def _fitted(self) -> _M:
        raise NotImplementedError

    @overload
    def get(self, verbose: bool = False, output: None = None, **kwargs: Any) -> _M: ...

    @overload
    def get(
        self, verbose: bool = False, output: str | None = None, **kwargs: Any
    ) -> list[Any]: ...

    def get(self, verbose: bool = False, output: str | None = None, **kwargs) -> Any:
        """Read the fitted model, or a named output of its fit."""
        return super().get(verbose=verbose, output=output, **kwargs)

    @overload
    def wait(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        output: None = None,
        **kwargs: Any,
    ) -> _M: ...

    @overload
    def wait(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        output: str | None = None,
        **kwargs: Any,
    ) -> list[Any]: ...

    def wait(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        output: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wait for the fit, then read the fitted model, or a named output of its fit."""
        self.wait_until_done(interval=interval, timeout=timeout, verbose=verbose)
        if self.status is not JobStatus.SUCCESS:
            raise JobFailedException(self.job_id, self.job.failure_message)
        return self.get(verbose=verbose, output=output, **kwargs)

    def _get(self, verbose: bool = False, **kwargs) -> Any:
        """The fitted model, or with ``output`` that output of the fit job."""
        if kwargs.get("output") is not None:
            return super()._get(verbose=verbose, **kwargs)
        return self._fitted()


def _compact(value: Any) -> str:
    """A one-line view of job args that counts lists and elides long strings."""
    if isinstance(value, dict):
        items = ", ".join(f"{key!r}: {_compact(item)}" for key, item in value.items())
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        return f"<{len(value)} items>"
    if isinstance(value, str) and len(value) > 32:
        return f"<str of {len(value)} chars>"
    return repr(value)


_V = TypeVar("_V")
_Scores = float | list[float] | np.ndarray
_Predictions = dict[str, Any]
#: JSON has no non-finite numbers, so the API spells them out.
_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _numbers(value: Any) -> Any:
    """``value`` with the spelt-out non-finite numbers read as floats."""
    if isinstance(value, str):
        return _NON_FINITE.get(value, value)
    if isinstance(value, list):
        return [_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _numbers(item) for key, item in value.items()}
    return value


#: structure outputs stored as CSV text, read as one table per position
_TABLE_OUTPUTS = frozenset({"score", "metrics"})


def _as_structure(value: Any) -> Structure:
    if isinstance(value, Structure):
        structure = value
    elif isinstance(value, Complex):
        structure = Structure([value])
    elif isinstance(value, (str, bytes)):
        structure = Structure.from_string(value, format="cif", verbose=False)
    else:
        raise TypeError(
            "structure output must be mmCIF, Complex or Structure; "
            f"received {type(value).__name__}"
        )

    if not len(structure) or any(
        not structure[i].get_chains() for i in range(len(structure))
    ):
        raise ValueError("structure output contains no molecular chains")
    return structure


def _as_table(value: str | bytes) -> pd.DataFrame:
    text = value.decode() if isinstance(value, bytes) else value
    return pd.read_csv(io.StringIO(text))


def _as_complex(value: Any) -> Complex:
    if isinstance(value, Complex):
        if not value.get_chains():
            raise ValueError("structure output contains no molecular chains")
        return value
    structure = _as_structure(value)
    if len(structure) != 1:
        raise ValueError("a design position must contain exactly one complex")
    return structure[0]


class _ResultFuture(Future[list[_V]], Generic[_V]):
    """Positional results whose bare reads select the task's primary output.

    Bare reads use the task's named output rather than the job's default
    output; pass ``output=`` to read another name in ``outputs``.
    """

    _primary_output: ClassVar[str]
    _raises_when_canceled = True

    def __init__(self, session: APISession, job: Job):
        super().__init__(session, job)
        self._cache: dict[Any, Any] = {}

    def __keys__(self) -> KeysView[int]:
        return dict.fromkeys(self._positions()).keys()

    def __len__(self) -> int:
        return len(self.__keys__())

    def _output_name(self, output: str | None) -> str:
        return super()._output_name(self._primary_output if output is None else output)

    @overload
    def get(
        self, verbose: bool = False, output: None = None, **kwargs: Any
    ) -> list[_V]: ...

    @overload
    def get(
        self, verbose: bool = False, output: str | None = None, **kwargs: Any
    ) -> list[Any]: ...

    def get(
        self, verbose: bool = False, output: str | None = None, **kwargs: Any
    ) -> Any:
        """Read every position of the selected output."""
        return super().get(verbose=verbose, output=output, **kwargs)

    @overload
    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        *,
        output: None = None,
        **kwargs: Any,
    ) -> list[_V]: ...

    @overload
    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        *,
        output: str | None = None,
        **kwargs: Any,
    ) -> list[Any]: ...

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        *,
        output: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wait for completion and read the selected output."""
        return super().wait(
            interval=interval,
            timeout=timeout,
            verbose=verbose,
            output=output,
            **kwargs,
        )

    @overload
    def get_item(self, position: int, output: None = None) -> _V: ...

    @overload
    def get_item(self, position: int, output: str | None = None) -> Any: ...

    def get_item(self, position: int, output: str | None = None) -> Any:
        """Read one submitted position of the selected output."""
        return super().get_item(position, output=output)

    @overload
    def stream(
        self,
        output: None = None,
        *,
        follow: bool = False,
        interval: float = config.POLLING_INTERVAL,
    ) -> Iterator[_V]: ...

    @overload
    def stream(
        self,
        output: str | None = None,
        *,
        follow: bool = False,
        interval: float = config.POLLING_INTERVAL,
    ) -> Iterator[Any]: ...

    def stream(
        self,
        output: str | None = None,
        *,
        follow: bool = False,
        interval: float = config.POLLING_INTERVAL,
    ) -> Iterator[Any]:
        """Yield selected values in submitted position order."""
        return super().stream(output=output, follow=follow, interval=interval)

    @property
    def units(self) -> list[FutureUnit[_V]]:
        """One handle per submitted unit, in position order."""
        return super().units

    @overload
    def unit(self, position: int, output: None = None) -> FutureUnit[_V]: ...

    @overload
    def unit(self, position: int, output: str | None = None) -> FutureUnit[Any]: ...

    def unit(self, position: int, output: str | None = None) -> FutureUnit[Any]:
        """Handle on one unit's selected output."""
        return super().unit(position, output=output)

    @overload
    def __getitem__(self, k: int) -> _V: ...

    @overload
    def __getitem__(self, k: int, **kwargs: Any) -> Any: ...

    def __getitem__(self, k: int, **kwargs: Any) -> Any:
        """Read one position, caching the value per position and keyword arguments."""
        key = (
            (k, *(f"{name}={value}" for name, value in kwargs.items())) if kwargs else k
        )
        if key not in self._cache:
            self._cache[key] = self.get_item(k, **kwargs)
        return self._cache[key]

    def __iter__(self) -> Iterator[_V]:
        return self.stream()


class EmbeddingFuture(_ResultFuture[np.ndarray]):
    """Embedding arrays, one per submitted position.

    Bare reads return the ``embeddings`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "embeddings"


class LogitsFuture(_ResultFuture[np.ndarray]):
    """Per-position logit arrays.

    Bare reads return the ``logits`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "logits"


class AttentionFuture(_ResultFuture[np.ndarray]):
    """Attention arrays with the model's head and residue dimensions.

    Bare reads return the ``attn`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "attn"


class LogLikelihoodFuture(_ResultFuture[_Scores]):
    """Sequence log-likelihoods, one per submitted sequence in the order submitted.

    A score is a float, or one float per prompt replicate when the prompt has
    several: ``np.asarray(future.get(), dtype=float)`` has shape ``(n,)`` or
    ``(n, replicates)``.

    Bare reads return the ``loglikelihood`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "loglikelihood"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value)


class SingleSiteScoreFuture(_ResultFuture[_Scores]):
    """Scores of every single-site substitution of the base sequence.

    Positions run in sequence order. At each position the substitutions run in
    the order of ``openprotein.utils.sequence.AMINO_ACIDS``, the position's own
    residue included, so a scanned position has 20 scores. Chain breaks (``:``)
    and gaps (``-``) are not scanned; with PoET's ``include_gaps``, gaps are
    scanned too and ``-`` is a 21st substitution. A score is a float, or one
    float per prompt replicate.

    Bare reads return the ``score`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "score"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value)


class IndelScoreFuture(_ResultFuture[_Scores]):
    """Scores of the base sequence, then of each insertion or deletion.

    The first score is the base sequence's. With ``delete``, each size follows
    in the order given, at every start from the first residue, with ``insert``
    in place of the deleted fragment when both are given. With ``insert``
    alone, the insertion follows at every site, from before the first residue
    to after the last. A score is a float, or one float per prompt replicate.

    Bare reads return the ``score`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "score"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value)


class SequenceGenerationFuture(_ResultFuture[str]):
    """Generated sequence strings in sample order.

    Bare reads return the ``sequences`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "sequences"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value) if output == "loglikelihood" else value


class PredictionFuture(_ResultFuture[_Predictions]):
    """Property-keyed predictions, one dictionary per position.

    Bare reads return the ``predictions`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "predictions"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value)


class SingleSitePredictionFuture(_ResultFuture[_Predictions]):
    """Property-keyed predictions in single-site mutation order.

    Bare reads return the ``predictions`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "predictions"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        return _numbers(value)


class StructureDesignFuture(_ResultFuture[Complex]):
    """Generated complexes, one per design position.

    Bare reads return the ``structures`` output; pass ``output=`` to read another
    name in ``outputs``.
    """

    _primary_output = "structures"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        if output != "structures":
            return value
        try:
            return _as_complex(value)
        except (TypeError, ValueError) as error:
            raise type(error)(f"job {self.id} position {position}: {error}") from error


class StructurePredictionFuture(_ResultFuture[Structure]):
    """Predicted structure collections, one per submitted input.

    Bare reads return the ``structures`` output; pass ``output=`` to read another
    name in ``outputs``. RoseTTAFold3's ``score`` and ``metrics`` read as one
    ``pandas.DataFrame`` per input.
    """

    _primary_output = "structures"

    def _convert_output(self, output: str, position: int, value: Any) -> Any:
        try:
            if output == "structures":
                return _as_structure(value)
            if output in _TABLE_OUTPUTS and isinstance(value, (str, bytes)):
                return _as_table(value)
        except (TypeError, ValueError) as error:
            raise type(error)(f"job {self.id} position {position}: {error}") from error
        return value


@dataclass
class CrossvalidateResults:
    """Cross-validation results as parallel arrays, one per sequence x measurement.

    Attributes
    ----------
    row_index : np.ndarray
        Position in the uploaded assay of the row each evaluation came from. This is
        the index the ``folds`` argument of `crossvalidate` is given in.
    sequence : np.ndarray
        Sequence for each evaluation.
    measurement_name : np.ndarray
        Measurement/property name for each evaluation.
    fold : np.ndarray or None
        Held-out test fold the row was scored in, or -1 if the row was excluded, in
        which case ``y_mu`` and ``y_var`` are NaN. None for a crossvalidation the
        service reported without fold assignments.
    y : np.ndarray
        Observed value.
    y_mu : np.ndarray
        Cross-validated predicted mean.
    y_var : np.ndarray
        Cross-validated predicted variance.
    """

    row_index: np.ndarray
    sequence: np.ndarray
    measurement_name: np.ndarray
    fold: np.ndarray | None
    y: np.ndarray
    y_mu: np.ndarray
    y_var: np.ndarray


def _reject_output(future: Future, output: str | None) -> None:
    """Raise for an output name on a future that has one result."""
    if output is not None:
        raise ValueError(
            f"{type(future).__name__} has one result; call get() without output"
        )


class CrossvalidateFuture(Future):
    """Handle on a cross-validation job; its id is the evaluation's.

    ``get`` and ``wait`` return its `CrossvalidateResults`. The run's statistics are read
    with ``GPModel.evaluations``. A reloaded cross-validation job loads as
    ``CVResultFuture`` instead.
    """

    _raises_when_canceled = True

    def __init__(self, session: APISession, job: Job, gp: "GPModel"):
        super().__init__(session, job)
        self._gp = gp

    def get(
        self, verbose: bool = False, output: str | None = None, **kwargs: Any
    ) -> CrossvalidateResults:
        """Read the cross-validated prediction of each assay row.

        Raises
        ------
        ValueError
            If ``output`` is given; the future has one result.
        """
        _reject_output(self, output)
        return super().get(verbose=verbose, output=output, **kwargs)

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> CrossvalidateResults:
        """Wait for cross-validation and read its predictions.

        Raises
        ------
        ValueError
            If ``output`` is given; the future has one result.
        """
        _reject_output(self, kwargs.get("output"))
        return super().wait(
            interval=interval, timeout=timeout, verbose=verbose, **kwargs
        )

    def _get(self, verbose: bool = False, **kwargs: Any) -> Any:
        rows = self._gp.evaluations(self.id)["rows"]
        folds = [row.get("fold") for row in rows]

        def measured(name: str) -> np.ndarray:
            return np.array(
                [np.nan if row[name] is None else row[name] for row in rows],
                dtype=float,
            )

        return CrossvalidateResults(
            row_index=np.array([row["row_index"] for row in rows], dtype=int),
            sequence=np.array([row["sequence"] for row in rows], dtype=object),
            measurement_name=np.array([row["property"] for row in rows], dtype=object),
            fold=None
            if any(fold is None for fold in folds)
            else np.array(folds, dtype=int),
            y=measured("y"),
            y_mu=measured("mean"),
            y_var=measured("variance"),
        )


class GPFitFuture(FitFuture["GPModel"]):
    """Handle on a GP's training job.

    ``wait`` returns the fitted ``GPModel``. A reloaded training job loads as
    ``PredictorModel`` instead.
    """

    _family = "gp"

    def _fitted(self) -> "GPModel":
        from openprotein.models.gp import _load

        return _load(self.session, self.id)


class SVDFitFuture(FitFuture["SVDModel"]):
    """Handle on an SVD fit job.

    Its output is the fit's source sequences.
    """

    _family = "svd"
    _fit_outputs = ("sequences",)

    def _fitted(self) -> "SVDModel":
        from openprotein.models.svd import _load

        return _load(self.session, self.id)

    def transform(
        self, data: "list[str] | list[bytes] | AssayDataset"
    ) -> EmbeddingFuture:
        """Project sequences, or an assay's, through this fit.

        The projection is submitted now and runs once the fit has finished.
        """
        from openprotein.models.model import _project

        return _project(self, data)

    def embed(self, data: "list[str] | list[bytes] | AssayDataset") -> EmbeddingFuture:
        """Same as ``transform``."""
        return self.transform(data)


class UMAPFitFuture(FitFuture["UMAPModel"]):
    """Handle on a UMAP fit job; its outputs are the fit set's projections."""

    _family = "umap"
    _fit_outputs = ("sequences", "embeddings")

    def _fitted(self) -> "UMAPModel":
        from openprotein.models.umap import _load

        return _load(self.session, self.id)

    def transform(
        self, data: "list[str] | list[bytes] | AssayDataset"
    ) -> EmbeddingFuture:
        """Map sequences, or an assay's, through this fit.

        The projection is submitted now and runs once the fit has finished.
        """
        from openprotein.models.model import _project

        return _project(self, data)

    def embed(self, data: "list[str] | list[bytes] | AssayDataset") -> EmbeddingFuture:
        """Same as ``transform``."""
        return self.transform(data)
