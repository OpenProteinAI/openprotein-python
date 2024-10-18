from typing import Literal

from pydantic import BaseModel

from .job import Job, JobType


class CVItem(BaseModel):
    row_index: int
    sequence: str
    measurement_name: str
    y: float
    y_mu: float
    y_var: float


class CVJob(Job):
    job_type: Literal[JobType.workflow_crossvalidate]
    num_rows: int | None = None
    page_size: int | None = None
    page_offset: int | None = None
    result: list[CVItem] | None = None


class TrainStep(BaseModel):
    step: int
    loss: float
    tag: str
    tags: dict


class TrainJob(Job):
    job_type: Literal[JobType.workflow_train]
    traingraph: list[TrainStep] | None = None
