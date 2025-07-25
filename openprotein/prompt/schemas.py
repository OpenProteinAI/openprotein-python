from datetime import datetime
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from openprotein.jobs import Job, JobStatus, JobType
from openprotein.protein import Protein

Context = Sequence[bytes | str | Protein]


class PromptJob(Job):
    """A representation of a prompt job."""

    job_type: Literal[JobType.align_prompt]

    @property
    def msa_id(self):
        """ID of the underlying MSA."""
        return self.msa_id

    @property
    def prompt_id(self):
        """Prompt ID."""
        return self.job_id


class PromptMetadata(BaseModel):
    """Metadata about a prompt."""

    id: str = Field(description="Prompt unique identifier.")
    name: str = Field(description="Name of the prompt")
    description: str | None = Field(
        None,
        description="Description of the prompt",
    )
    created_date: datetime = Field(description="The date the prompt was created.")
    num_replicates: int = Field(description="Number of replicates provided as context.")
    job_id: str | None = Field(
        None, description="The job_id of the sampling job, if it exists."
    )
    status: JobStatus = Field(description="The status of the prompt.")


class QueryMetadata(BaseModel):
    """Metadata about a query."""

    id: str = Field(description="Query unique identifier.")
    created_date: datetime = Field(description="The date the query was created.")
