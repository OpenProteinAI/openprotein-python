from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from openprotein.jobs.schemas import JobStatus

__all__ = ["MSADownloadFormat", "MSAFormat", "MSAMetadata"]


MSAFormat = Literal["a2m", "a3m", "csv", "fasta", "sto"]
MSADownloadFormat = Literal["a2m", "a3m", "csv", "fasta", "sto", "parquet"]


class MSAMetadata(BaseModel):
    """Metadata for one stored MSA (never carries the sequence body).

    A search row has no uploaded original, so its ``msa_hash`` and
    ``original_filename`` are ``None``. Until its job produces it, a row has a
    ``status`` other than ``SUCCESS`` and no stats.
    """

    msa_id: str
    status: JobStatus = JobStatus.SUCCESS
    job_id: str | None = None
    msa_hash: str | None = None
    name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    project_uuid: str | None = None
    source_format: MSAFormat | None = None
    num_sequences: int | None = None
    num_match_columns: int | None = None
    has_insertions: bool | None = None
    original_filename: str | None = None
    digest: str | None = None
    created_at: datetime
    updated_at: datetime
