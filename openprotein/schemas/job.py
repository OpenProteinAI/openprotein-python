import logging
from datetime import datetime
from enum import Enum
from typing import Union

from pydantic import BaseModel, ConfigDict, TypeAdapter
from requests import Response
from typing_extensions import Self

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    """
    Type of job.

    Describes the types of jobs that can be done.
    """

    stub = "stub"

    workflow_preprocess = "/workflow/preprocess"
    workflow_train = "/workflow/train"
    workflow_embed_umap = "/workflow/embed/umap"
    workflow_predict = "/workflow/predict"
    workflow_predict_single_site = "/workflow/predict/single_site"
    workflow_crossvalidate = "/workflow/crossvalidate"
    workflow_evaluate = "/workflow/evaluate"
    workflow_design = "/workflow/design"

    align_align = "/align/align"
    align_prompt = "/align/prompt"
    poet = "/poet"
    poet_score = "/poet/score"
    poet_single_site = "/poet/single_site"
    poet_generate = "/poet/generate"

    embeddings_embed = "/embeddings/embed"
    embeddings_svd = "/embeddings/svd"
    embeddings_attn = "/embeddings/attn"
    embeddings_logits = "/embeddings/logits"
    embeddings_embed_reduced = "/embeddings/embed_reduced"

    svd_fit = "/svd/fit"
    svd_embed = "/svd/embed"

    embeddings_fold = "/embeddings/fold"

    # predictor jobs
    predictor_train = "/predictor/train"
    predictor_predict = "/predictor/predict"
    predictor_crossvalidate = "/predictor/crossvalidate"
    predictor_predict_single_site = "/predictor/predict_single_site"
    predictor_predict_multi = "/predictor/predict_multi"
    predictor_predict_multi_single_site = "/predictor/predict_multi_single_site"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRYING = "RETRYING"
    CANCELED = "CANCELED"

    def done(self):
        return (
            (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)
        )  # noqa: E501

    def cancelled(self):
        return self is self.CANCELED


class Job(BaseModel):
    job_id: str
    # new emb service get doesnt have job_type
    job_type: JobType
    status: JobStatus
    created_date: datetime
    start_date: datetime | None = None
    end_date: datetime | None = None
    prerequisite_job_id: str | None = None
    progress_message: str | None = None
    progress_counter: int | None = None
    sequence_length: int | None = None

    @classmethod
    def create(cls, obj: "Job | Response | dict", **kwargs) -> Self:
        # parse specific child Job from base Job or Response
        try:
            # try to parse as subclass job
            # get dict form
            d = (
                obj.json()
                if isinstance(obj, Response)
                else obj.model_dump() if isinstance(obj, Job) else obj
            )
            job_classes = Job.__subclasses__()
            job = TypeAdapter(Union[tuple(job_classes)]).validate_python(d | kwargs)  # type: ignore
        except Exception as e:
            raise ValueError(f"Error parsing job from obj: {obj}: {e}")
        return job  # type: ignore - static checker cannot know runtime type

    # hide extra allowed fields
    def __repr_args__(self):
        for k, v in self.__dict__.items():
            field = self.model_fields.get(k)
            if field and field.repr:
                yield k, v

        yield from (
            (k, getattr(self, k))
            for k, v in self.model_computed_fields.items()
            if v.repr
        )

    # allows to carry over subclassed job fields when factory creating
    model_config = ConfigDict(extra="allow")


class BatchJob(BaseModel):
    num_records: int | None = None
