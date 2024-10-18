"""OpenProtein schemas for Fold."""

from typing import Literal

from .job import BatchJob, Job, JobType


class FoldJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_fold]
