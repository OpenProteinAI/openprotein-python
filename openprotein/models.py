
from typing import Optional, List, Union, Dict, Literal
from datetime import datetime 
import time 
from enum import Enum
from pydantic import BaseModel, Field
import tqdm
from openprotein.errors import TimeoutException
import openprotein.config as config
from openprotein.base import APISession
from openprotein.api.jobs import Job, JobStatus


class DesignMetadata(BaseModel):
    y_mu: Optional[float]
    y_var: Optional[float]

class DesignSubscore(BaseModel):
    score: int
    metadata: DesignMetadata

class DesignStep(BaseModel):
    step: int
    sample_index: int
    sequence: str
    #scores: List[int]
    #subscores_metadata: List[List[DesignSubscore]]
    initial_scores: List[int] = Field(..., alias='scores')  # renaming 'scores' to 'initial_scores'  # noqa: E501
    scores: List[List[DesignSubscore]] = Field(..., alias='subscores_metadata')  # renaming 'subscores_metadata' to 'scores'  # noqa: E501
    umap1: float
    umap2: float


class DesignResults(BaseModel):
    status: str
    job_id: str
    created_date: str
    job_type: str
    start_date: str
    end_date: str
    assay_id: str
    num_rows: int
    result: List[DesignStep]


class DirectionEnum(str, Enum):
    gt = '>'
    lt = '<'
    eq = '='

class Criterion(BaseModel):
    target: float
    weight: float
    direction: str

class ModelCriterion(BaseModel):
    criterion_type: Literal["model"]
    model_id: str
    measurement_name: str
    criterion: Criterion

class NMutationCriterion(BaseModel):
    criterion_type: Literal["n_mutations"]
    #sequences: Optional[List[str]]

CriterionItem = Union[ModelCriterion, NMutationCriterion]

class DesignJobCreate(BaseModel):
    assay_id: str
    criteria: List[List[CriterionItem]]
    num_steps: Optional[int] = 8
    pop_size: Optional[int] = None
    n_offsprings:  Optional[int] = None
    crossover_prob: Optional[float] = None
    crossover_prob_pointwise: Optional[float] = None
    mutation_average_mutations_per_seq: Optional[int] =None
    mutation_positions: Optional[List[int]] = None

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

class Jobplus(Job):
    sequence_length: Optional[int]
 
class TrainStep(BaseModel):
    step: int
    loss: float
    tag: str
    tags: dict

class TrainGraph(BaseModel):
    traingraph: List[TrainStep]
    created_date: datetime
    job_id: str

class SequenceData(BaseModel):
    sequence: str
class SequenceDataset(BaseModel):
    sequences: List[str]
class JobDetails(BaseModel):
    job_id: str
    job_type: str
    status: str
class AssayMetadata(BaseModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: List[str]
    sequence_length: Optional[int] = None

class AssayDataRow(BaseModel):
    mut_sequence: str
    measurement_values: List[Union[float, None]]


class AssayDataPage(BaseModel):
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

class PoetSiteResult(BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str]

class PromptPostParams(BaseModel):
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

class PoetScoreResult(BaseModel):
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

class Prediction(BaseModel):
    """Prediction details."""

    model_id: str
    model_name: str
    properties: Dict[str, Dict[str, float]]

class PredictJobBase(BaseModel):
    """Shared properties for predict job outputs."""

    # might be none if just fetching
    job_id: Optional[str] = None
    job_type: str
    status: str

class DesignJob(BaseModel):
    job_id: Optional[str] = None
    job_type: str
    status: str

class PredictJob(PredictJobBase):
    """Properties about predict job returned via API."""

    class SequencePrediction(BaseModel):
        """Sequence prediction."""

        sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None

class PredictSingleSiteJob(PredictJobBase):
    """Properties about single-site prediction job returned via API."""

    class SequencePrediction(BaseModel):
        """Sequence prediction."""

        position: int
        amino_acid: str
        # sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None


class CVItem(BaseModel):
    row_index: int
    sequence: str
    measurement_name: str
    y: float
    y_mu: float
    y_var: float

class CVResults(Job):
    num_rows: int
    page_size: int
    page_offset: int
    result: List[CVItem]

