import pydantic
from pydantic import Field
from enum import Enum
from typing import Optional, List, Union, Dict

from io import BytesIO
from datetime import datetime 

from openprotein.api.jobs import Job
import openprotein.config as config



class JobType(str, Enum):
    """
    Type of job.

    Describes the types of jobs that can be done.
    """

    stub = "stub"

    preprocess = "/workflow/preprocess"
    train = "/workflow/train"
    embed_umap = "/workflow/embed/umap"
    predict = "/workflow/predict"
    predict_single_site = "/workflow/predict/single_site"
    crossvalidate = "/workflow/crossvalidate"
    evaluate = "/workflow/evaluate"
    design = "/workflow/design"

    align = "/align/align"
    align_prompt = "/align/prompt"
    prots2prot = "/poet"
    prots2prot_single_site = "/poet/single_site"
    prots2prot_generate = "/poet/generate"

    embeddings = "/embeddings/embed"
    svd = "/embeddings/svd"
    attn = "/embeddings/attn"
    logits = "/embeddings/logits"
    
class TrainStep(pydantic.BaseModel):
    step: int
    loss: float
    tag: str
    tags: dict

class TrainGraph(pydantic.BaseModel):
    traingraph: List[TrainStep]
    created_date: datetime
    job_id: str

class SequenceData(pydantic.BaseModel):
    sequence: str
class SequenceDataset(pydantic.BaseModel):
    sequences: List[str]
class JobDetails(pydantic.BaseModel):
    job_id: str
    job_type: str
    status: str
class AssayMetadata(pydantic.BaseModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: List[str]
    sequence_length: Optional[int] = None

class AssayDataRow(pydantic.BaseModel):
    mut_sequence: str
    measurement_values: List[Union[float, None]]


class AssayDataPage(pydantic.BaseModel):
    assaymetadata: AssayMetadata
    page_size: int
    page_offset: int
    assaydata: List[AssayDataRow]

class MSAJob(Job):
    msa_id: str

class PromptJob(Job):
    prompt_id: str

class MSASamplingMethod(str, Enum):
    RANDOM = 'RANDOM'
    NEIGHBORS = 'NEIGHBORS'
    NEIGHBORS_NO_LIMIT = 'NEIGHBORS_NO_LIMIT'
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = 'NEIGHBORS_NONGAP_NORM_NO_LIMIT'
    TOP = 'TOP'

class PoetSiteResult(pydantic.BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str]

class PromptPostParams(pydantic.BaseModel):
    msa_id: str
    num_sequences: Optional[int] = Field(None, ge=0, lt=100)
    num_residues: Optional[int] = Field(None, ge=0, lt=24577)
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    homology_level: float = Field(0.8, ge=0, le=1)
    max_similarity: float = Field(1.0, ge=0, le=1)
    min_similarity: float = Field(0.0, ge=0, le=1)
    always_include_seed_sequence: bool = False
    num_ensemble_prompts: int = 1
    random_seed: Optional[int] = None

class PoetSingleSiteJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[PoetSiteResult]]
    #n_completed: Optional[int]

class PoetInputType(str, Enum):
    INPUT = 'RAW'
    MSA = 'GENERATED'
    PROMPT = 'PROMPT'

class PoetScoreResult(pydantic.BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str]

class PoetScoreJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[PoetScoreResult]]
    n_completed: Optional[int]

class Prediction(pydantic.BaseModel):
    """Prediction details."""

    model_id: str
    model_name: str
    properties: Dict[str, Dict[str, float]]

class PredictJobBase(pydantic.BaseModel):
    """Shared properties for predict job outputs."""

    # might be none if just fetching
    job_id: Optional[str] = None
    job_type: str
    status: str


class PredictJob(PredictJobBase):
    """Properties about predict job returned via API."""

    class SequencePrediction(pydantic.BaseModel):
        """Sequence prediction."""

        sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None

class PredictSingleSiteJob(PredictJobBase):
    """Properties about single-site prediction job returned via API."""

    class SequencePrediction(pydantic.BaseModel):
        """Sequence prediction."""

        position: int
        amino_acid: str
        # sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None