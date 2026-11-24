"""
Jobs module for OpenProtein.

isort:skip_file
"""

from .futures import (
    AttentionFuture,
    CrossvalidateFuture,
    CrossvalidateResults,
    EmbeddingFuture,
    FitFuture,
    Future,
    FutureUnit,
    GPFitFuture,
    IndelScoreFuture,
    LogitsFuture,
    LogLikelihoodFuture,
    MappedFuture,
    PagedFuture,
    PredictionFuture,
    SequenceGenerationFuture,
    SingleSitePredictionFuture,
    SingleSiteScoreFuture,
    StreamingFuture,
    StructureDesignFuture,
    StructurePredictionFuture,
    SVDFitFuture,
    UMAPFitFuture,
    UnknownJobFuture,
)
from .jobs import JobsAPI
from .schemas import BatchJob, Job, JobStatus, JobType

__all__ = [
    "AttentionFuture",
    "BatchJob",
    "CrossvalidateFuture",
    "CrossvalidateResults",
    "EmbeddingFuture",
    "FitFuture",
    "Future",
    "FutureUnit",
    "GPFitFuture",
    "IndelScoreFuture",
    "Job",
    "JobStatus",
    "JobType",
    "JobsAPI",
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
]
