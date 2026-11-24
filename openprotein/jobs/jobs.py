from __future__ import annotations

import builtins
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future as Pending
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openprotein import config
from openprotein.base import APISession
from openprotein.errors import JobFailedException

from . import _api, _decode, _storage, api
from .futures import Future
from .schemas import Job, JobStatus, JobType, _UnitRecord

__all__ = ["JobsAPI"]


@dataclass(frozen=True, slots=True)
class _JobOutput:
    """One decoded canonical output at its original position."""

    position: int
    unit_id: str
    value: Any


class JobsAPI:
    """API interface to get jobs."""

    def __init__(self, session: APISession):
        self.session = session
        self._storage = _storage.DirectArtifactStorage()

    # Qualify `list` because the method name shadows the builtin in annotations.
    def list(
        self,
        status: JobStatus | None = None,
        job_type: str | JobType | None = None,
        assay_id: str | None = None,
        more_recent_than: datetime | str | None = None,
        page_size: int | None = None,
        page_offset: int | None = None,
        limit: int | None = None,
        *,
        name: str | None = None,
    ) -> builtins.list[Job]:
        """List jobs.

        `name` matches a job's name exactly; set one with `update`. Names are
        not unique, so the result stays a list.

        Pass `page_size` / `page_offset` to paginate. `limit` is a deprecated
        alias for `page_size` — pass one, not both.
        """
        if limit is not None and page_size is not None:
            raise ValueError("Pass either page_size or limit (deprecated), not both")
        if page_size is None:
            page_size = limit if limit is not None else 100
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
                name=name,
                more_recent_than=more_recent_than_str,
                page_size=page_size,
                page_offset=page_offset,
            )
        ]

    def get_job(self, job_id: str) -> Job:
        return api.job_get(session=self.session, job_id=job_id)

    def _iter_job_outputs(
        self,
        job_id: str,
        output_name: str | None,
        *,
        follow: bool = False,
        interval: float = config.POLLING_INTERVAL,
    ) -> Iterator[_JobOutput]:
        """Stream decoded outputs in submission order, retaining duplicate positions.

        HTTP resolves authorized metadata; the storage adapter downloads and
        verifies artifact bytes directly, a few downloads at a time and at most
        twice that many ahead of the caller. ``output_name=None`` reads the
        method's default output.

        An emitted output grows while its job runs. The stream ends at the items
        registered now, or with ``follow`` reads again every ``interval`` seconds
        until the output is complete.
        """
        pool = ThreadPoolExecutor(max_workers=_storage.DOWNLOAD_WORKERS)
        pending: deque[Pending[_JobOutput]] = deque()
        try:
            for values in self._iter_output_pages(
                job_id, output_name, follow, interval
            ):
                for entry in values:
                    pending.append(pool.submit(self._decode_job_output, job_id, entry))
                    if len(pending) >= 2 * _storage.DOWNLOAD_WORKERS:
                        yield pending.popleft().result()
                if not values:
                    while pending:
                        yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            pool.shutdown(cancel_futures=True)

    def _iter_output_pages(
        self,
        job_id: str,
        output_name: str | None,
        follow: bool,
        interval: float,
    ) -> Iterator[list[dict[str, Any]]]:
        """Each page's entries. An empty page that keeps the cursor is caught up."""
        cursor: str | None = None
        while True:
            page = _api.job_output_page(
                self.session, job_id, output_name, cursor=cursor
            )
            values = page.get("values", [])
            yield values
            next_cursor = page.get("next_cursor")
            if not next_cursor:
                return
            if not values and next_cursor == cursor:
                if not follow:
                    return
                time.sleep(interval)
            cursor = next_cursor

    def _decode_job_output(self, job_id: str, entry: dict[str, Any]) -> _JobOutput:
        position = entry["position"]
        if type(position) is not int or position < 0:
            raise ValueError("output position must be a nonnegative integer")
        if entry.get("value") is None and entry.get("error") is not None:
            raise JobFailedException(
                job_id, f"output at position {position} failed: {entry['error']}"
            )
        return _JobOutput(
            position=position,
            unit_id=str(entry["unit_id"]),
            value=_decode.decode_value(entry.get("value"), self._storage),
        )

    def _get_job_output(
        self, job_id: str, position: int, output_name: str
    ) -> _JobOutput:
        """Read the stable public (job, position, output) handle."""
        return self._decode_job_output(
            job_id, _api.job_output_get(self.session, job_id, output_name, position)
        )

    def _iter_job_units(self, job_id: str) -> Iterator[_UnitRecord]:
        """Read the job's unit records in position order, without waiting for outputs."""
        position: int | None = None
        while True:
            page = _api.job_unit_page(self.session, job_id, position=position)
            for item in page.get("units", []):
                yield _UnitRecord.model_validate(item)
            next_position = page.get("next_position")
            if next_position is None:
                return
            if (
                type(next_position) is not int
                or next_position < 0
                or (position is not None and next_position <= position)
            ):
                raise ValueError("invalid unit metadata pagination position")
            position = next_position

    def get_job_args(self, job_id: str) -> dict[str, Any]:
        return api.job_args_get(session=self.session, job_id=job_id)

    def update(
        self,
        job_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Job:
        """Set a job's name and description.

        Only the arguments given are sent, so renaming a job leaves its
        description alone.
        """
        return _api.job_update(
            session=self.session,
            job_id=job_id,
            name=name,
            description=description,
        )

    def get(self, job_id: str, verbose: bool = False) -> Future:
        """
        Get job by ID.

        Notes
        -----
        This retrieves the job and loads it as a future so you can do `wait` and `get`.

        Supported models-task records load their typed result future. Records
        owned by compatibility schemas retain their compatibility future, including
        when they carry provider and method metadata. GP training loads
        `PredictorModel`, GP cross-validation loads `CVResultFuture`, and prompt
        sampling loads the `Prompt` it samples, which is a stored prompt and not
        a `Future`. Each class documents its result contract. An unresolved
        models identity remains a generic, trackable `Future`.
        """
        return self.__load(job_id=job_id)

    def __load(self, job_id: str) -> Future:
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
