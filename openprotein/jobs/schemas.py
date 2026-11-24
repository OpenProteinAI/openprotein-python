import logging
from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from requests import Response
from typing_extensions import Self

__all__ = ["BatchJob", "Job", "JobStatus", "JobType"]


logger = logging.getLogger(__name__)


class JobType(str, Enum):
    """
    Type of job.

    The job types this client has names for. ``Job.job_type`` is a plain string
    and the server adds types on its own release cycle, so this is a vocabulary
    for readability, for filtering, and for the ``Literal`` discriminators on
    typed ``Job`` subclasses. Membership is not a validity check.
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

    clustalo = "/align/clustalo"
    mafft = "/align/mafft"
    abnumber = "/align/abnumber"

    poet = "/poet"
    poet_score = "/poet/score"
    poet_single_site = "/poet/single_site"
    poet_generate = "/poet/generate"
    poet_score_indel = "/poet/score/indel"

    embeddings_embed = "/embeddings/embed"
    embeddings_svd = "/embeddings/svd"
    embeddings_attn = "/embeddings/attn"
    embeddings_logits = "/embeddings/logits"
    embeddings_embed_reduced = "/embeddings/embed_reduced"
    embeddings_generate = "/embeddings/generate"

    svd_fit = "/svd/fit"
    svd_embed = "/svd/embed"

    umap_fit = "/umap/fit"
    umap_embed = "/umap/embed"

    clustering_hierarchical = "/clustering/hierarchical"

    embeddings_fold = "/embeddings/fold"

    # predictor jobs
    predictor_train = "/predictor/train"
    predictor_predict = "/predictor/predict"
    predictor_crossvalidate = "/predictor/crossvalidate"
    predictor_predict_single_site = "/predictor/predict_single_site"
    predictor_predict_multi = "/predictor/predict_multi"
    predictor_predict_multi_single_site = "/predictor/predict_multi_single_site"

    # designer
    designer = "/design"

    models = "/models"


class JobStatus(str, Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRYING = "RETRYING"
    CANCELED = "CANCELED"

    def done(self):
        return (
            (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)
        )

    def cancelled(self):
        return self is self.CANCELED


class Job(BaseModel):
    job_id: str
    job_type: str
    status: JobStatus
    created_date: datetime
    start_date: datetime | None = None
    end_date: datetime | None = None
    prerequisite_job_id: str | None = None
    progress_message: str | None = None
    progress_counter: int | None = None
    sequence_length: int | None = None
    failure_message: str | None = None
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def _typed_job_types(cls) -> set[str]:
        """Job type values that a typed subclass declares a ``Literal`` for."""
        claimed: set[str] = set()
        for sub in Job.__subclasses__():
            field = sub.model_fields.get("job_type")
            if field is None:
                continue
            annotation = field.annotation
            members = (
                get_args(annotation)
                if get_origin(annotation) in (Union, UnionType)
                else (annotation,)
            )
            for member in members:
                if get_origin(member) is Literal:
                    claimed.update(
                        arg.value if isinstance(arg, Enum) else str(arg)
                        for arg in get_args(member)
                    )
        return claimed

    @classmethod
    def create(cls, obj: "Job | Response | dict", **kwargs) -> Self:
        d = (
            obj.json()
            if isinstance(obj, Response)
            else obj.model_dump()
            if isinstance(obj, Job)
            else obj
        )
        payload = d | kwargs
        job_classes = Job.__subclasses__()
        try:
            job = TypeAdapter(Union[tuple(job_classes)]).validate_python(payload)  # noqa: UP007
        except ValidationError:
            # A subclass declaring this job type means the payload disagrees with
            # a schema this client has, which is a bug worth surfacing. Any other
            # type is one the server added, and parses as a plain job.
            if payload.get("job_type") in cls._typed_job_types():
                raise
            logger.info("no typed schema for job type %s", payload.get("job_type"))
            job = Job.model_validate(payload)
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


_UNIT_STATUS_ALIASES = {"FAILED": "FAILURE", "READY": "PENDING"}


class _UnitRecord(BaseModel):
    """A unit's position, identity, status and error, as one unit read returns them.

    The status comes from the record's ``status`` key when it has a value, else
    its ``state`` key, in the vocabulary of `JobStatus`: ``FAILED`` reads as
    ``FAILURE`` and a unit ready to run reads as ``PENDING``. Other keys are
    ignored.
    """

    position: int
    unit_id: str
    status: JobStatus
    error: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_status(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        status = data.get("status")
        if status is None:
            status = data.get("state")
        if isinstance(status, str):
            status = _UNIT_STATUS_ALIASES.get(status, status)
        return {
            "position": data.get("position"),
            "unit_id": data.get("unit_id"),
            "status": status,
            "error": data.get("error"),
        }
