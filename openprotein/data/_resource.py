"""Behaviour shared by the stored-data handles."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from typing_extensions import Self

from openprotein import config
from openprotein.base import APISession
from openprotein.errors import (
    HTTPError,
    JobFailedException,
    JobOutputsNotReady,
    NoJobError,
)
from openprotein.jobs.schemas import JobStatus

if TYPE_CHECKING:
    from openprotein.jobs.schemas import Job

M = TypeVar("M")


class _Resource(Generic[M]):
    """A stored row: its metadata, and the job that produces it.

    ``session.data`` hands out each class bound to its session. ``id=`` alone
    loads a row. A row answers ``Future``'s calls: ``done``, ``wait_until_done``,
    ``get`` and ``wait`` return once it is ready, and ``get`` and ``wait``
    return the row itself. A row no job made, such as an upload, is always
    ready, and its job fields raise `NoJobError`.
    """

    _session: ClassVar[APISession | None] = None
    _unbound: ClassVar[type | None] = None
    _job: Job | None = None
    # The metadata came from the job, as the row couldn't be read; the job's
    # status says nothing about whether the row exists.
    _from_job_only: bool = False

    session: APISession
    metadata: M

    def __init__(
        self,
        session: APISession | None = None,
        metadata: M | None = None,
        *,
        id: str | None = None,
    ):
        if id is not None:
            if session is not None or metadata is not None:
                raise TypeError("pass id alone to load a stored row")
            session = self._bound()
            metadata = self._load(session, id)
        elif session is None or metadata is None:
            raise TypeError(
                f"{type(self).__name__} takes id=, or a session and its metadata"
            )
        self.session = session
        self.metadata = metadata

    @classmethod
    def _bind(cls, session: APISession) -> type[Self]:
        """This class, creating, listing and loading through ``session``."""
        attrs = {
            "_session": session,
            "_unbound": cls._unbound or cls,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__doc__": cls.__doc__,
        }
        return cast("type[Self]", type(cls.__name__, (cls,), attrs))

    @classmethod
    def _for(cls, session: APISession) -> type[Self]:
        """``session.data``'s own class, else this class bound to ``session``."""
        from openprotein.data.data import DataAPI

        data = getattr(session, "data", None)
        if isinstance(data, DataAPI):
            return getattr(data, cls.__name__)
        return cls._bind(session)

    @classmethod
    def _bound(cls) -> APISession:
        if cls._session is None:
            raise TypeError(
                f"{cls.__name__} needs a session; use session.data.{cls.__name__}"
            )
        return cls._session

    @classmethod
    def _load(cls, session: APISession, id: str) -> M:
        """Read one row's metadata by id."""
        raise NotImplementedError

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, (self._unbound or type(self), vars(self).copy()))

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    def __bool__(self) -> bool:
        return True

    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def status(self) -> JobStatus:
        raise NotImplementedError

    @property
    def job_id(self) -> str | None:
        raise NotImplementedError

    @property
    def job(self) -> Job | None:
        """The job that made the row, read on first use; ``None`` if no job did."""
        job_id = self.job_id
        if job_id is None:
            return None
        if self._job is None:
            from openprotein.jobs import api
            from openprotein.jobs.schemas import Job

            self._job = Job.create(api.job_get(self.session, job_id))
        return self._job

    def _job_if_present(self) -> Job | None:
        """The row's job, or ``None`` if no job made it or its job is gone."""
        try:
            return self.job
        except HTTPError as error:
            if error.status_code in (404, 410):
                return None
            raise

    def _job_field(self, name: str) -> Any:
        job = self.job
        if job is None:
            raise NoJobError(
                f"{type(self).__name__} {self.id} was not made by a job, "
                f"so it has no {name}"
            )
        return getattr(job, name)

    @property
    def job_type(self) -> str:
        """The type of the job that made the row."""
        return self._job_field("job_type")

    @property
    def created_date(self) -> datetime:
        """When the job that made the row was created."""
        return self._job_field("created_date")

    @property
    def start_date(self) -> datetime | None:
        """When the job that made the row started."""
        return self._job_field("start_date")

    @property
    def end_date(self) -> datetime | None:
        """When the job that made the row finished."""
        return self._job_field("end_date")

    def _forget(self) -> None:
        """Drop content read earlier."""

    def refresh(self) -> Self:
        """Read the metadata again, and drop the job and content read earlier."""
        self.metadata = self._load(self.session, self.id)
        self._from_job_only = False
        self._job = None
        self._forget()
        return self

    def done(self) -> bool:
        """Whether the row is ready, failed or canceled, as last read."""
        return self.status.done()

    def wait_until_done(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> bool:
        """Wait for the job that makes the row to finish, then read the row again.

        A row that is already done returns at once. A failure does not raise;
        ``get`` raises it.

        Parameters
        ----------
        interval : float, optional
            Seconds between polls. Defaults to `config.POLLING_INTERVAL`.
        timeout : int, optional
            Seconds to wait before raising. Defaults to no limit.
        verbose : bool, optional
            Show a progress bar. Defaults to False.

        Returns
        -------
        bool
            Whether the row is done.

        Raises
        ------
        TimeoutException
            If the wait exceeds ``timeout``.
        """
        if self.done():
            return True
        job = self._job_if_present()
        if job is not None:
            from openprotein.jobs.futures import Future

            tracker: Future[Any] = Future(self.session, job)
            if not tracker.done():
                tracker.wait_until_done(
                    interval=interval, timeout=timeout, verbose=verbose
                )
            job = tracker.job
        self.refresh()
        self._job = job
        return self.done()

    def get(self, verbose: bool = False, **kwargs: Any) -> Self:
        """The row, once it is ready.

        Raises
        ------
        JobOutputsNotReady
            If the row isn't ready yet.
        JobFailedException
            If the job that makes the row failed or was canceled.
        """
        if not self.done() or self._from_job_only:
            self.refresh()
        failure = None
        if self.status is JobStatus.FAILURE:
            job = self._job_if_present()
            failure = None if job is None else job.failure_message
        self._check(failure)
        return self

    def wait(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> Self:
        """Wait until the row is ready and return it, as ``get`` does.

        Parameters
        ----------
        interval : float, optional
            Seconds between polls. Defaults to `config.POLLING_INTERVAL`.
        timeout : int, optional
            Seconds to wait before raising. Defaults to no limit.
        verbose : bool, optional
            Show a progress bar. Defaults to False.

        Raises
        ------
        TimeoutException
            If the wait exceeds ``timeout``.
        JobFailedException
            If the job that makes the row failed or was canceled.
        JobOutputsNotReady
            If the job finished but the row isn't ready yet.
        """
        self.wait_until_done(interval=interval, timeout=timeout, verbose=verbose)
        return self.get()

    def _require_ready(self) -> None:
        """Raise unless the row's content can be read."""
        if self.status is not JobStatus.SUCCESS or self._from_job_only:
            self.refresh()
        self._check()

    def _check(self, failure: str | None = None) -> None:
        status = self.status
        if status is JobStatus.CANCELED:
            raise JobFailedException(self.job_id or self.id, "the job was canceled")
        if status is JobStatus.FAILURE:
            raise JobFailedException(self.job_id or self.id, failure)
        if status is not JobStatus.SUCCESS:
            raise JobOutputsNotReady(
                f"{type(self).__name__} {self.id} is {status.value}; "
                "wait() returns it once it is ready"
            )


def _restore(cls: type, state: dict[str, Any]) -> Any:
    """A handle of ``cls`` holding ``state``, as pickled."""
    handle = object.__new__(cls)
    vars(handle).update(state)
    return handle
