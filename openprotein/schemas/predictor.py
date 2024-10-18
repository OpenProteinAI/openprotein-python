from enum import Enum
from typing import Literal

from pydantic import BaseModel

from .job import Job, JobStatus, JobType


class Kernel(BaseModel):
    type: str
    multitask: bool = False


class Constraints(BaseModel):
    sequence_length: int | None = None


class FeatureType(str, Enum):

    PLM = "PLM"
    SVD = "SVD"


class Features(BaseModel):
    type: FeatureType
    model_id: str | None = None
    reduction: str | None = None

    class Config:
        protected_namespaces = ()


class PredictorArgs(BaseModel):
    kernel: Kernel


class ModelSpec(PredictorArgs, BaseModel):
    constraints: Constraints | None = None
    features: Features


class Dataset(BaseModel):
    assay_id: str
    properties: list[str]


class PredictorMetadata(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: JobStatus
    model_spec: ModelSpec
    training_dataset: Dataset

    def is_done(self):
        return self.status.done()

    class Config:
        protected_namespaces = ()


class TrainJob(Job):
    job_type: Literal[JobType.predictor_train]


class PredictJob(Job):
    job_type: Literal[JobType.predictor_predict]


class PredictSingleSiteJob(Job):
    job_type: Literal[JobType.predictor_predict_single_site]


class PredictMultiJob(Job):
    job_type: Literal[JobType.predictor_predict_multi]


class PredictMultiSingleSiteJob(Job):
    job_type: Literal[JobType.predictor_predict_multi_single_site]


class CVJob(Job):
    job_type: Literal[JobType.predictor_crossvalidate]
