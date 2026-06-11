"""Schemas for OpenProtein clustering."""

from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from openprotein.common import FeatureType
from openprotein.jobs import Job, JobStatus, JobType


class LinkageMethod(str, Enum):
    """Hierarchical clustering linkage methods (scipy.cluster.hierarchy)."""

    WARD = "ward"
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"


class Metric(str, Enum):
    """Pairwise distance metrics supported by scipy.spatial.distance.pdist."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    CORRELATION = "correlation"
    HAMMING = "hamming"
    CHEBYSHEV = "chebyshev"
    CITYBLOCK = "cityblock"
    SQEUCLIDEAN = "sqeuclidean"
    CANBERRA = "canberra"
    BRAYCURTIS = "braycurtis"


class ClusteringMetadata(BaseModel):
    id: str
    status: JobStatus
    created_date: datetime | None = None
    method: str
    linkage_method: LinkageMethod | None = None
    metric: Metric | None = None
    model_id: str
    feature_type: FeatureType
    reduction: str | None = None
    svd_id: str | None = None

    def is_done(self) -> bool:
        return self.status.done()

    model_config = ConfigDict(protected_namespaces=())


class HierarchicalFitJob(Job):
    job_type: Literal[JobType.clustering_hierarchical]


class HierarchicalClusteringResult(BaseModel):
    """Result of a hierarchical clustering job.

    `linkage` is the scipy linkage matrix with shape (N-1, 4); pass it
    directly to `scipy.cluster.hierarchy.dendrogram` or `fcluster`.
    `sequences` is filled by `HierarchicalClusteringFuture._get` after the API fetch.

    Note: `linkage` is a numpy array, so this model is not JSON-serializable via
    `model_dump_json()` (use `linkage.tolist()` first if you need to serialize).
    """

    n_leaves: int
    linkage: np.ndarray
    leaf_order: list[int]
    sequences: list[bytes] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("linkage", mode="before")
    @classmethod
    def _to_ndarray(cls, v):
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v, dtype=float)
