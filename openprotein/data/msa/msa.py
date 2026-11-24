"""Stored MSAs (``session.data.MSA``).

Upload an MSA, or build one with ``session.align.create_msa``, and address it by
its ``id``. Iterating a stored MSA reads its rows, ``to_string`` renders it in
any supported format, and ``sample_prompt`` samples it into a prompt.

>>> msa = session.data.MSA.create(open("aln.a3m", "rb").read(), format="a3m")
>>> len(msa)
128
>>> name, sequence = msa[0]
>>> fasta = msa.to_string("fasta")
>>> prompt = msa.sample_prompt(replicates=2).wait()
"""

from __future__ import annotations

import builtins
import re
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, get_args, overload

from typing_extensions import Self

from openprotein import fasta
from openprotein.base import APISession
from openprotein.data._resource import _Resource
from openprotein.jobs.schemas import JobStatus

from . import _api
from .schemas import MSAFormat, MSAMetadata

if TYPE_CHECKING:
    from openprotein.align.schemas import MSASamplingMethod
    from openprotein.data.prompts import Prompt

__all__ = ["MSA"]

_LIST_PAGE = 200

# Rectangular formats that can mark inserts. Their FASTA rendering keeps every
# column, so their rows come from the a2m rendering with the inserts dropped.
_INSERTS_KEPT_BY_FASTA = frozenset({"a2m", "sto"})


class MSA(_Resource[MSAMetadata]):
    """A stored MSA.

    ``session.data.MSA.create`` uploads one, ``session.data.MSA.list`` lists the
    caller's and ``session.data.MSA(id=...)`` loads one. Iterating it yields its
    ``(name, sequence)`` rows, read on first use.

    See Also
    --------
    openprotein.align.MSAFuture : The job that builds an MSA. Its id is the id
        of the MSA it stores.
    """

    _rows: builtins.list[tuple[str, str]] | None = None

    @classmethod
    def _load(cls, session: APISession, id: str) -> MSAMetadata:
        return _api.msa_get_metadata(session, id)

    @classmethod
    def create(
        cls,
        file: bytes | str,
        format: MSAFormat,
        name: str | None = None,
        description: str | None = None,
        filename: str | None = None,
    ) -> Self:
        """Upload an MSA, declaring its ``format`` (e.g. ``"a3m"``, ``"fasta"``).

        Parameters
        ----------
        file : bytes | str
            The MSA contents; a ``str`` is encoded as UTF-8 (read a file first,
            e.g. ``open(path, "rb").read()``).
        format : str
            The upload's format, naming the codec used to parse it.
        name, description : str, optional
            Human-readable metadata to store alongside the MSA.
        filename : str, optional
            Original filename to record; defaults to a name derived from ``format``.

        Returns
        -------
        MSA
            The stored MSA.
        """
        session = cls._bound()
        content = file.encode() if isinstance(file, str) else file
        metadata = _api.msa_post(
            session,
            file=content,
            format=format,
            name=name,
            description=description,
            filename=filename,
        )
        return cls(session, metadata)

    @classmethod
    def list(cls, limit: int | None = None) -> builtins.list[Self]:
        """The caller's stored MSAs from every page, at most ``limit``."""
        session = cls._bound()
        msas: builtins.list[Self] = []
        while limit is None or len(msas) < limit:
            size = _LIST_PAGE if limit is None else min(_LIST_PAGE, limit - len(msas))
            page = _api.msa_list(session, limit=size, offset=len(msas))
            msas.extend(cls(session, metadata) for metadata in page.items)
            if not page.items or len(msas) >= page.total:
                break
        return msas if limit is None else msas[:limit]

    @property
    def id(self) -> str:
        """The MSA's id."""
        return self.metadata.msa_id

    @property
    def digest(self) -> str | None:
        """The row's content digest."""
        return self.metadata.digest

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @property
    def description(self) -> str | None:
        return self.metadata.description

    @property
    def status(self) -> JobStatus:
        """``SUCCESS`` once the MSA is ready, else the status of the job producing it."""
        return self.metadata.status

    @property
    def job_id(self) -> str | None:
        """The job that produces the MSA; ``None`` for an upload."""
        return self.metadata.job_id

    @property
    def source_format(self) -> MSAFormat | None:
        return self.metadata.source_format

    @property
    def num_sequences(self) -> int | None:
        return self.metadata.num_sequences

    @property
    def num_match_columns(self) -> int | None:
        return self.metadata.num_match_columns

    @property
    def has_insertions(self) -> bool | None:
        return self.metadata.has_insertions

    @property
    def tags(self) -> builtins.list[str]:
        return self.metadata.tags

    @property
    def original_filename(self) -> str | None:
        return self.metadata.original_filename

    @property
    def created_at(self) -> datetime:
        return self.metadata.created_at

    @property
    def updated_at(self) -> datetime:
        return self.metadata.updated_at

    def _forget(self) -> None:
        self._rows = None

    def _read(self) -> builtins.list[tuple[str, str]]:
        if self._rows is None:
            self._require_ready()
            rows = None
            if self.source_format in _INSERTS_KEPT_BY_FASTA and self.num_match_columns:
                rows = _match_column_rows(self._download("a2m"), self.num_match_columns)
            self._rows = rows if rows is not None else self._download("fasta")
        return self._rows

    def _download(self, format: MSAFormat) -> builtins.list[tuple[str, str]]:
        text = _api.msa_download(self.session, self.id, format).decode()
        return builtins.list(fasta.parse_stream(iter(text.splitlines())))

    def __len__(self) -> int:
        if self._rows is None and self.metadata.num_sequences is not None:
            return self.metadata.num_sequences
        return len(self._read())

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """The ``(name, sequence)`` rows, aligned to one length.

        Insertions relative to the query are left out; ``to_string`` keeps
        them. An a2m or Stockholm upload whose rows don't all mark the same
        insert columns, by lowercase and ``.``, keeps every column. The rows
        are read on first use.
        """
        return iter(self._read())

    @overload
    def __getitem__(self, index: int) -> tuple[str, str]: ...

    @overload
    def __getitem__(self, index: slice) -> builtins.list[tuple[str, str]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[str, str] | builtins.list[tuple[str, str]]:
        return self._read()[index]

    def to_string(self, format: MSAFormat | None = None) -> str:
        """The alignment as text in ``format``, by default the stored format.

        Parameters
        ----------
        format : str, optional
            One of ``"a2m"``, ``"a3m"``, ``"csv"``, ``"fasta"`` and ``"sto"``.

        Raises
        ------
        ValueError
            If ``format`` is not one of the five.
        """
        if format is not None and format not in get_args(MSAFormat):
            raise ValueError(
                f"format must be one of {', '.join(get_args(MSAFormat))}, "
                f"not {format!r}"
            )
        self._require_ready()
        return _api.msa_download(self.session, self.id, format).decode()

    def sample_prompt(
        self,
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
        """Sample this MSA into a prompt, one sample per replicate.

        Every option but ``replicates`` and ``seed`` is omitted when unset so
        the sampling default applies. Sampling stops at whichever of ``max_msa_sequences``
        and ``max_msa_tokens`` is set; leave both unset to take the default
        budget.

        Parameters
        ----------
        replicates : int, optional
            Number of prompts to sample. Defaults to 1.
        seed : int, optional
            Seed for the sampling. Each replicate draws its own from it. A
            missing seed is drawn client-side.
        method : MSASamplingMethod, optional
            Sampling method.
        homology_level : float, optional
            Level of homology for sequences in the MSA, between 0 and 1
            (neighbors methods only).
        max_similarity : float, optional
            Maximum similarity between sequences in the MSA and the seed,
            between 0 and 1.
        min_similarity : float, optional
            Minimum similarity between sequences in the MSA and the seed,
            between 0 and 1.
        max_msa_sequences : int, optional
            Cap on the number of sequences sampled.
        max_msa_tokens : int, optional
            Cap on the number of residues sampled.
        force_include_first : bool, optional
            Whether to always include the seed sequence.

        Returns
        -------
        Prompt
            The prompt being sampled. ``wait`` returns it once it is ready.

        Raises
        ------
        MissingParameterError
            If both ``max_msa_sequences`` and ``max_msa_tokens`` are given.
        """
        from openprotein.data.prompts.prompts import _sample

        return _sample(
            self.session,
            self.id,
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

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        tags: builtins.list[str] | None = None,
    ) -> Self:
        """Update the name, description or tags; only the ones given change."""
        self.metadata = _api.msa_put(
            self.session, self.id, name=name, description=description, tags=tags
        )
        return self

    def delete(self) -> None:
        """Delete this MSA."""
        _api.msa_delete(self.session, self.id)


_INSERT = re.compile(r"[a-z.]")


def _match_column_rows(
    rows: builtins.list[tuple[str, str]], num_match_columns: int
) -> builtins.list[tuple[str, str]] | None:
    """The a2m rows without their insert columns, if every row marks the same ones.

    Stockholm need not mark inserts by lowercase and ``.``: it can use ``.`` as a
    gap or lowercase for a masked residue. Rows that mark different columns, or
    leave other than ``num_match_columns``, give ``None``.
    """
    inserts: builtins.list[int] | None = None
    stripped = []
    for name, sequence in rows:
        marks = [match.start() for match in _INSERT.finditer(sequence)]
        if inserts is None:
            inserts = marks
        elif marks != inserts:
            return None
        kept = _INSERT.sub("", sequence)
        if len(kept) - kept.count(":") != num_match_columns:
            return None
        stripped.append((name, kept))
    return stripped
