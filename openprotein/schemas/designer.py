from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from .design import Criteria, DesignConstraint
from .job import Job, JobStatus, JobType


class DesignAlgorithm(str, Enum):
    genetic_algorithm = "genetic-algorithm"


class Design(BaseModel):
    id: str
    status: JobStatus
    progress_counter: int
    created_date: datetime
    algorithm: DesignAlgorithm
    num_rows: int
    num_steps: int
    assay_id: str
    criteria: Criteria
    allowed_tokens: dict[str, list[str]] | None
    pop_size: int
    # ga params
    n_offsprings: int
    crossover_prob: float
    crossover_prob_pointwise: float
    mutation_average_mutations_per_seq: int

    def is_done(self):
        return self.status.done()


class DesignJob(Job):
    job_type: Literal[JobType.designer]
