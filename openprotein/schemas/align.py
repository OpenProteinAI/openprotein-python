from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .job import Job, JobType


class PoetInputType(str, Enum):
    INPUT = "RAW"
    MSA = "GENERATED"
    PROMPT = "PROMPT"


class MSASamplingMethod(str, Enum):
    RANDOM = "RANDOM"
    NEIGHBORS = "NEIGHBORS"
    NEIGHBORS_NO_LIMIT = "NEIGHBORS_NO_LIMIT"
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = "NEIGHBORS_NONGAP_NORM_NO_LIMIT"
    TOP = "TOP"


class PromptPostParams(BaseModel):
    msa_id: str
    num_sequences: int | None = Field(None, ge=0, lt=100)
    num_residues: int | None = Field(None, ge=0, lt=24577)
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    homology_level: float = Field(0.8, ge=0, le=1)
    max_similarity: float = Field(1.0, ge=0, le=1)
    min_similarity: float = Field(0.0, ge=0, le=1)
    always_include_seed_sequence: bool = False
    num_ensemble_prompts: int = 1
    random_seed: int | None = None


class MSAJob(Job):
    msa_id: str | None = None
    job_type: Literal[JobType.align_align]

    @model_validator(mode="before")
    def set_msa_id(cls, values):
        if not values.get("msa_id"):
            values["msa_id"] = values.get("job_id")
        return values


class PromptJob(MSAJob, Job):
    prompt_id: str | None = None
    job_type: Literal[JobType.align_prompt]

    @model_validator(mode="before")
    def set_prompt_id(cls, values):
        if not values.get("prompt_id"):
            values["prompt_id"] = values.get("job_id")
        return values
