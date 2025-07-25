"""Schema for OpenProtein fold system."""

from typing import Literal

from pydantic import BaseModel

from openprotein.jobs import BatchJob, Job, JobType


class FoldMetadata(BaseModel):
    """
    Metadata for a folding job.

    Attributes
    ----------
    job_id : str
        Unique identifier for the job.
    model_id : str
        Identifier for the model used in the job.
    args : dict or None, optional
        Additional arguments for the job.
    """

    job_id: str
    model_id: str
    args: dict | None = None


class FoldJob(Job, BatchJob):
    """
    Folding job class.

    Attributes
    ----------
    job_type : Literal[JobType.embeddings_fold]
        The type of job, set to embeddings_fold.
    """

    job_type: Literal[JobType.embeddings_fold]
