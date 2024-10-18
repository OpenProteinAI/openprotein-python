"""Schemas for OpenProtein SVD system."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import BatchJob, Job, JobStatus, JobType


class SVDMetadata(BaseModel):
    id: str
    status: JobStatus
    created_date: datetime | None = None
    model_id: str
    n_components: int
    reduction: str | None = None
    sequence_length: int | None = None

    def is_done(self):
        return self.status.done()

    class Config:
        protected_namespaces = ()


class FitJob(Job):
    job_type: Literal[JobType.svd_fit]


class EmbeddingsJob(Job, BatchJob):
    job_type: Literal[JobType.svd_embed]
