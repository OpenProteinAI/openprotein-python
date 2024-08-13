from openprotein.pydantic import BaseModel, ConfigDict
from enum import Enum


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
    worflow_predict_single_site = "/workflow/predict/single_site"
    workflow_crossvalidate = "/workflow/crossvalidate"
    workflow_evaluate = "/workflow/evaluate"
    workflow_design = "/workflow/design"

    align_align = "/align/align"
    align_prompt = "/align/prompt"
    poet_score = "/poet"
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


class JobStatus(str, Enum):
    PENDING: str = "PENDING"
    RUNNING: str = "RUNNING"
    SUCCESS: str = "SUCCESS"
    FAILURE: str = "FAILURE"
    RETRYING: str = "RETRYING"
    CANCELED: str = "CANCELED"

    def done(self):
        return (
            (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)
        )  # noqa: E501

    def cancelled(self):
        return self is self.CANCELED
