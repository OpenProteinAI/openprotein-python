from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .job import JobStatus


class PromptMetadata(BaseModel):
    id: str = Field(description="Prompt unique identifier.")
    name: str = Field(description="Name of the prompt")
    description: Optional[str] = Field(
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
    id: str = Field(description="Query unique identifier.")
    created_date: datetime = Field(description="The date the query was created.")
