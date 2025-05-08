from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .features import FeatureType
from .job import Job, JobStatus, JobType


class Kernel(BaseModel):
    type: str
    multitask: bool = False


class Constraints(BaseModel):
    sequence_length: int | None = None


class PredictorType(str, Enum):
    GP = "GP"
    ENSEMBLE = "ENSEMBLE"


class Features(BaseModel):
    type: FeatureType
    model_id: str | None = None
    reduction: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class PredictorArgs(BaseModel):
    kernel: Kernel | None = None


class ModelSpec(PredictorArgs, BaseModel):
    type: PredictorType
    constraints: Constraints | None = None
    features: Features | None = None


class Dataset(BaseModel):
    assay_id: str
    properties: list[str]


class PredictorMetadata(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: JobStatus
    created_date: datetime
    model_spec: ModelSpec
    ensemble_model_ids: list[str] | None = None
    training_dataset: Dataset
    traingraphs: list["TrainGraph"] | None = None

    def is_done(self):
        return self.status.done()

    model_config = ConfigDict(protected_namespaces=())

    class TrainGraph(BaseModel):
        measurement_name: str
        hyperparam_search_step: int
        losses: list[float]


class EnsembleJob(Job):
    job_id: None = None
    progress_counter: None = None


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
