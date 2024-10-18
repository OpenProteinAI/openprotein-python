"""Schemas for embeddings."""

from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .job import BatchJob, Job, JobType


class ModelDescription(BaseModel):
    """Description of available protein embedding models."""

    citation_title: str | None = None
    doi: str | None = None
    summary: str = "Protein language model for embeddings"


class TokenInfo(BaseModel):
    """Information about the tokens used in the embedding model."""

    id: int
    token: str
    primary: bool
    description: str


class ModelMetadata(BaseModel):
    """Metadata about available protein embedding models."""

    id: str = Field(..., alias="model_id")
    description: ModelDescription
    max_sequence_length: int | None = None
    dimension: int
    output_types: list[str]
    input_tokens: list[str]
    output_tokens: list[str] | None = None
    token_descriptions: list[list[TokenInfo]]


class ReductionType(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"


class EmbeddedSequence(BaseModel):
    """
    Representation of an embedded sequence created from our models.

    Represented as an iterable yielding the sequence followed by the embedding.
    """

    class Config:
        arbitrary_types_allowed = True

    sequence: bytes
    embedding: np.ndarray

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


class EmbeddingsJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_embed, JobType.embeddings_embed_reduced] = (
        JobType.embeddings_embed
    )


class AttnJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_attn]


class LogitsJob(Job, BatchJob):

    job_type: Literal[JobType.embeddings_logits]


class ScoreJob(Job, BatchJob):

    job_type: Literal[JobType.poet_score]


class ScoreSingleSiteJob(Job, BatchJob):

    job_type: Literal[JobType.poet_single_site]


class GenerateJob(Job, BatchJob):

    job_type: Literal[JobType.poet_generate]
