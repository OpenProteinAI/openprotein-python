"""Schemas for OpenProtein UMAP system."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from openprotein.common import FeatureType
from openprotein.jobs import BatchJob, Job, JobStatus, JobType


class UMAPMetadata(BaseModel):
    id: str
    status: JobStatus
    created_date: datetime | None = None
    model_id: str
    feature_type: FeatureType
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    reduction: str | None = None
    sequence_length: int | None = None

    def is_done(self):
        return self.status.done()

    model_config = ConfigDict(protected_namespaces=())


class UMAPFitJob(Job):
    job_type: Literal[JobType.umap_fit]


class UMAPEmbeddingsJob(Job, BatchJob):
    job_type: Literal[JobType.umap_embed]
