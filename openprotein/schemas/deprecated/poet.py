from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..job import Job, JobType


class PoetSSPResult(BaseModel):
    sequence: bytes = Field(validate_default=True)
    score: list[float]
    name: str | None = Field(default=None, validate_default=True)
    _n: int = 0

    @field_validator("sequence", mode="before")
    def replace_sequence(cls, value):
        """Rename X0X which refers to base sequence."""
        if "X0X" in str(value):
            return b"WT"
        return value

    @field_validator("name", mode="before")
    def increment_name(cls, value):
        if value is None:
            cls._n += 1
            return f"Mutant{cls._n}"
        return value


class PoetScoreResult(BaseModel):
    sequence: bytes
    score: list[float]
    name: str | None = None


class PoetScoreJob(Job):
    parent_id: str | None = None
    s3prefix: str | None = None
    page_size: int | None = None
    page_offset: int | None = None
    num_rows: int | None = None
    result: list[PoetScoreResult] | None = None
    n_completed: int | None = None

    job_type: Literal[JobType.poet]


# HACK - dont inherit directly so auto-parser doesnt find it
class PoetSSPJob(PoetScoreJob):
    parent_id: str | None = None
    s3prefix: str | None = None
    page_size: int | None = None
    page_offset: int | None = None
    num_rows: int | None = None
    result: list[PoetSSPResult] | None = None
    n_completed: int | None = None

    job_type: Literal[JobType.poet_single_site]


# HACK - dont inherit directly so auto-parser doesnt find it
class PoetGenerateJob(PoetScoreJob):
    parent_id: str | None = None
    s3prefix: str | None = None
    page_size: int | None = None
    page_offset: int | None = None
    num_rows: int | None = None
    result: list[PoetScoreResult] | None = None
    n_completed: int | None = None

    job_type: Literal[JobType.poet_generate]
