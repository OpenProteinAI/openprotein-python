from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from .job import Job, JobType


class SequenceData(BaseModel):
    sequence: str


class SequenceDataset(BaseModel):
    sequences: list[str]


# class _Prediction(BaseModel):
#     """Prediction details."""

#     @root_validator(pre=True)
#     def extract_pred(cls, values):
#         p = values.pop("properties")
#         name = list(p.keys())[0]
#         ymu = p[name]["y_mu"]
#         yvar = p[name]["y_var"]
#         p["name"] = name
#         p["y_mu"] = ymu
#         p["y_var"] = yvar

#         values.update(p)
#         return values

#     model_id: str
#     model_name: str
#     y_mu: Optional[float] = None
#     y_var: Optional[float] = None
#     name: Optional[str]


class Prediction(BaseModel):
    """Prediction details."""

    model_id: str
    model_name: str
    properties: dict[str, dict[str, float]]

    class Config:
        protected_namespaces = ()


class PredictJobBase(BaseModel):
    # might be none if just fetching
    job_id: str | None = None
    # doesn't have created date
    created_date: datetime | None = None


class PredictJob(PredictJobBase, Job):
    """Properties about predict job returned via API."""

    class SequencePrediction(BaseModel):
        """Sequence prediction."""

        sequence: str
        predictions: list[Prediction] = []

    job_type: Literal[JobType.workflow_predict]
    result: list[SequencePrediction] | None = None


class PredictSingleSiteJob(PredictJobBase, Job):
    """Properties about single-site prediction job returned via API."""

    class MutantPrediction(BaseModel):
        """Sequence prediction."""

        position: int
        amino_acid: str
        # sequence: str
        predictions: list[Prediction] = []

    result: list[MutantPrediction] | None = None
    job_type: Literal[JobType.workflow_predict_single_site]
