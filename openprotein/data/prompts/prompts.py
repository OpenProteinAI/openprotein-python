"""Stored prompts (``session.data.Prompt``).

Upload a prompt or sample one from a stored MSA, and address it by its ``id``.
Iterating a prompt reads its replicates.

>>> prompt = session.data.Prompt.create([open("r0.fasta", "rb").read()])
>>> replicate = prompt[0]
>>> sampled = msa.sample_prompt(replicates=2).wait()
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, overload

from typing_extensions import Self

from openprotein.base import APISession
from openprotein.data._resource import _Resource
from openprotein.errors import HTTPError
from openprotein.jobs.schemas import Job, JobStatus

from . import _api, _replicates
from .schemas import PromptMetadata

if TYPE_CHECKING:
    from openprotein.align.schemas import MSASamplingMethod

__all__ = ["Prompt"]

_LIST_PAGE = 200


class Prompt(_Resource[PromptMetadata]):
    """A stored prompt.

    ``session.data.Prompt.create`` uploads one, ``session.data.Prompt.list``
    lists the caller's and ``session.data.Prompt(id=...)`` loads one. Iterating
    it yields its replicates, read on first use.

    See Also
    --------
    openprotein.prompt.Prompt : A prompt created from sequences and structures
        with ``session.prompt.create_prompt``.
    """

    _replicates: builtins.list[builtins.list[str]] | None = None

    @classmethod
    def _load(cls, session: APISession, id: str) -> PromptMetadata:
        return _api.prompts_get(session, id)

    @classmethod
    def _from_job(cls, session: APISession, job: Job) -> Self:
        """The prompt a sampling job produces, as stored or else as the job reports it."""
        bound = cls._for(session)
        try:
            return bound(session, cls._load(session, job.job_id))
        except HTTPError as error:
            if error.status_code != 404:
                raise
        prompt = bound(
            session,
            PromptMetadata(
                id=job.job_id,
                status=job.status,
                job_id=job.job_id,
                created_at=job.created_date,
            ),
        )
        prompt._job = job
        prompt._from_job_only = job.status is JobStatus.SUCCESS
        return prompt

    @classmethod
    def create(
        cls,
        replicates: builtins.list[bytes | str],
        name: str | None = None,
        description: str | None = None,
    ) -> Self:
        """Upload a prompt, one FASTA file per replicate, in position order.

        Parameters
        ----------
        replicates : list of bytes or str
            One FASTA file per replicate; a ``str`` is encoded as UTF-8.
        name, description : str, optional
            Human-readable metadata to store alongside the prompt.

        Returns
        -------
        Prompt
            The stored prompt.

        Raises
        ------
        TypeError
            If ``replicates`` is a bare ``str`` or ``bytes``.
        """
        if isinstance(replicates, (str, bytes)):
            raise TypeError(
                "replicates must be a list of FASTA files, "
                f"not a {type(replicates).__name__}; "
                "pass a list of files, e.g. create([contents])"
            )
        session = cls._bound()
        contents = [
            replicate.encode() if isinstance(replicate, str) else replicate
            for replicate in replicates
        ]
        return cls(
            session,
            _api.prompts_post(session, contents, name=name, description=description),
        )

    @classmethod
    def list(cls, limit: int | None = None) -> builtins.list[Self]:
        """The caller's stored prompts from every page, newest first, at most ``limit``."""
        session = cls._bound()
        prompts: builtins.list[Self] = []
        after: str | None = None
        while limit is None or len(prompts) < limit:
            size = (
                _LIST_PAGE if limit is None else min(_LIST_PAGE, limit - len(prompts))
            )
            page = _api.prompts_list(session, after=after, limit=size)
            prompts.extend(cls(session, item) for item in page.items)
            after = page.next
            if after is None:
                break
        return prompts if limit is None else prompts[:limit]

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def status(self) -> JobStatus:
        """``SUCCESS`` once the prompt is ready, else the status of the job producing it."""
        return self.metadata.status

    @property
    def job_id(self) -> str | None:
        """The job that produces the prompt; ``None`` for an upload."""
        return self.metadata.job_id

    @property
    def owner(self) -> str | None:
        return self.metadata.owner

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @property
    def description(self) -> str | None:
        return self.metadata.description

    @property
    def created_at(self) -> datetime | None:
        return self.metadata.created_at

    @property
    def digest(self) -> str | None:
        """The row's content identity."""
        return self.metadata.digest

    def _forget(self) -> None:
        self._replicates = None

    def _read(self) -> builtins.list[builtins.list[str]]:
        if self._replicates is None:
            self._require_ready()
            self._replicates = _replicates.parse_replicates(
                _api.prompts_download(self.session, self.id)
            )
        return self._replicates

    def __len__(self) -> int:
        return len(self._read())

    def __iter__(self) -> Iterator[builtins.list[str]]:
        """Each replicate's sequences, a complex's chains joined by ``:``.

        The replicates are read on first use.
        """
        return iter(self._read())

    @overload
    def __getitem__(self, index: int) -> builtins.list[str]: ...

    @overload
    def __getitem__(self, index: slice) -> builtins.list[builtins.list[str]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> builtins.list[str] | builtins.list[builtins.list[str]]:
        return self._read()[index]

    def update(self, name: str | None = None, description: str | None = None) -> Self:
        """Update the name or description; only the ones given change."""
        self.metadata = _api.prompts_put(
            self.session, self.id, name=name, description=description
        )
        return self

    def delete(self) -> None:
        """Delete this prompt."""
        _api.prompts_delete(self.session, self.id)


def _sample(
    session: APISession,
    msa_id: str,
    replicates: int = 1,
    seed: int | None = None,
    method: MSASamplingMethod | None = None,
    homology_level: float | None = None,
    max_similarity: float | None = None,
    min_similarity: float | None = None,
    max_msa_sequences: int | None = None,
    max_msa_tokens: int | None = None,
    force_include_first: bool | None = None,
) -> Prompt:
    """Submit one sampling of a stored MSA and return the prompt it produces."""
    job = _api.prompts_sample_post(
        session,
        msa_id=msa_id,
        replicates=replicates,
        seed=seed,
        method=method,
        homology_level=homology_level,
        max_similarity=max_similarity,
        min_similarity=min_similarity,
        max_msa_sequences=max_msa_sequences,
        max_msa_tokens=max_msa_tokens,
        force_include_first=force_include_first,
    )
    return Prompt._for(session)(
        session,
        PromptMetadata(
            id=job["job_id"],
            status=job["status"],
            job_id=job["job_id"],
            created_at=job.get("created_date"),
        ),
    )
