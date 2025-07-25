"""Schemas for OpenProtein embeddings system."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from openprotein.jobs import BatchJob, Job, JobType


class EmbeddedSequence(BaseModel):
    """
    Representation of an embedded sequence created from our models.

    Represented as an iterable yielding the sequence followed by the embedding.
    """

    sequence: bytes
    embedding: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self):
        yield self.sequence
        yield self.embedding

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return self.sequence
        elif i == 1:
            return self.embedding
        raise IndexError("Index out of range")


class EmbeddingsJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_embed, JobType.embeddings_embed_reduced] = Field(
        default=JobType.embeddings_embed
    )


class AttnJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_attn] = Field(default=JobType.embeddings_attn)


class LogitsJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_logits] = Field(
        default=JobType.embeddings_logits
    )


class ScoreJob(Job, BatchJob):

    job_type: Literal[JobType.poet_score] = Field(default=JobType.poet_score)


class ScoreIndelJob(Job, BatchJob):

    job_type: Literal[JobType.poet_score_indel] = Field(
        default=JobType.poet_score_indel
    )


class ScoreSingleSiteJob(Job, BatchJob):

    job_type: Literal[JobType.poet_single_site] = Field(
        default=JobType.poet_single_site
    )


class GenerateJob(Job, BatchJob):

    job_type: Literal[JobType.poet_generate] = Field(default=JobType.poet_generate)
