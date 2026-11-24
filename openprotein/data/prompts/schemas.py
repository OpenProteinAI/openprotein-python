"""Schemas for stored prompts (``session.data.Prompt``).

.. deprecated:: 0.18.2
    ``PromptSummary``: use ``openprotein.data.PromptMetadata``.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from openprotein import _deprecation
from openprotein.jobs.schemas import JobStatus

__all__ = ["PromptMetadata", "PromptRecord", "PromptUnitValue"]


class PromptMetadata(BaseModel):
    """One row as listed, without its replicates.

    A row whose job hasn't finished has a ``status`` other than ``SUCCESS`` and
    no replicates.
    """

    id: str
    status: JobStatus = JobStatus.SUCCESS
    job_id: str | None = None
    owner: str | None = None
    digest: str | None = None
    name: str | None = None
    description: str | None = None
    created_at: datetime | None = None


class PromptUnitValue(BaseModel):
    """One replicate's presigned value."""

    position: int
    value: dict[str, Any]


class PromptRecord(PromptMetadata):
    """One row read in full, carrying each replicate's presigned value.

    ``value`` is ``None`` until the row's job has produced the replicates.
    """

    value: list[PromptUnitValue] | None = None


def __getattr__(name: str) -> Any:
    if name == "PromptSummary":
        _deprecation.warn_deprecated(
            f"{__name__}.PromptSummary",
            "`openprotein.data.PromptMetadata`",
            since="0.18.2",
            removed_in="0.19.0",
        )
        return PromptMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
