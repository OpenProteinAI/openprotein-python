# pydantic models
from typing import Optional, List, Union, Dict, Literal
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator, ConfigDict
from openprotein.api.jobs import Job, JobStatus, NewModel

class ModelDescription(NewModel):
    citation_title: Optional[str] = None
    doi: Optional[str] = None
    summary: str

class TokenInfo(NewModel):
    id: int
    token: str
    primary: bool
    description: str

class ModelMetadata(NewModel):
    model_id: str
    description: ModelDescription
    max_sequence_length: Optional[int] = None
    dimension: int
    output_types: List[str]
    input_tokens: List[str]
    output_tokens: List[str]
    token_descriptions: List[List[TokenInfo]]

class EmbeddedSequence(NewModel):
    class Config:
        arbitrary_types_allowed = True

    sequence: bytes
    embedding: np.ndarray

    def __iter__(self):
        yield self.sequence
        yield self.embedding

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return self.sequence
        elif i == 1:
            return self.embedding


class SVDJob(Job):
    pass


class SVDMetadata(NewModel):
    id: str
    status: JobStatus
    created_date: Optional[datetime] = None
    model_id: str
    n_components: int
    reduction: Optional[str] = None
    sequence_length: Optional[int] = None

    def is_done(self):
        return self.status.done()
    
class DesignMetadata(NewModel):
    y_mu: Optional[float] = None
    y_var: Optional[float] = None

class DesignSubscore(NewModel):
    score: float
    metadata: DesignMetadata

class DesignStep(NewModel):
    step: int
    sample_index: int
    sequence: str
    # scores: List[int]
    # subscores_metadata: List[List[DesignSubscore]]
    initial_scores: List[float] = Field(
        ..., alias="scores"
    )  # renaming 'scores' to 'initial_scores'  # noqa: E501
    scores: List[List[DesignSubscore]] = Field(
        ..., alias="subscores_metadata"
    )  # renaming 'subscores_metadata' to 'scores'  # noqa: E501
    umap1: float
    umap2: float


class DesignResults(NewModel):
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
    gt = ">"
    lt = "<"
    eq = "="


class Criterion(NewModel):
    target: float
    weight: float
    direction: str


class ModelCriterion(NewModel):
    criterion_type: Literal["model"]
    model_id: str
    measurement_name: str
    criterion: Criterion


class NMutationCriterion(NewModel):
    criterion_type: Literal["n_mutations"]
    # sequences: Optional[List[str]]


CriterionItem = Union[ModelCriterion, NMutationCriterion]


class DesignJobCreate(NewModel):
    assay_id: str
    criteria: List[List[CriterionItem]]
    num_steps: Optional[int] = 8
    pop_size: Optional[int] = None
    n_offsprings: Optional[int] = None
    crossover_prob: Optional[float] = None
    crossover_prob_pointwise: Optional[float] = None
    mutation_average_mutations_per_seq: Optional[int] = None
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
    sequence_length: Optional[int] = None


class TrainStep(NewModel):
    step: int
    loss: float
    tag: str
    tags: dict


class TrainGraph(NewModel):
    traingraph: List[TrainStep]
    created_date: datetime
    job_id: str


class SequenceData(NewModel):
    sequence: str


class SequenceDataset(NewModel):
    sequences: List[str]


class JobDetails(NewModel):
    job_id: str
    job_type: str
    status: str


class AssayMetadata(NewModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: List[str]
    sequence_length: Optional[int] = None


class AssayDataRow(NewModel):
    mut_sequence: str
    measurement_values: List[Union[float, None]]


class AssayDataPage(NewModel):
    assaymetadata: AssayMetadata
    page_size: int
    page_offset: int
    assaydata: List[AssayDataRow]


class MSAJob(Job):
    msa_id: Optional[str] = None


class PromptJob(MSAJob):
    prompt_id: Optional[str] = None


class MSASamplingMethod(str, Enum):
    RANDOM = "RANDOM"
    NEIGHBORS = "NEIGHBORS"
    NEIGHBORS_NO_LIMIT = "NEIGHBORS_NO_LIMIT"
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = "NEIGHBORS_NONGAP_NORM_NO_LIMIT"
    TOP = "TOP"





class PromptPostParams(NewModel):
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

#class PoetSiteResult(NewModel):
#    # obsolete
#    sequence: bytes
#    score: List[float]
#    name: Optional[str]

#class PoetSingleSiteJob(Job):
#    # obsolete
#    parent_id: Optional[str]
#    s3prefix: Optional[str]
#    page_size: Optional[int]
#    page_offset: Optional[int]
#    num_rows: Optional[int]
#    result: Optional[List[PoetSiteResult]]
#    # n_completed: Optional[int]

class PoetInputType(str, Enum):
    INPUT = "RAW"
    MSA = "GENERATED"
    PROMPT = "PROMPT"


class PoetScoreResult(NewModel):
    sequence: bytes
    score: List[float]
    name: Optional[str] = None

class PoetScoreJob(Job):
    parent_id: Optional[str] = None
    s3prefix: Optional[str] = None
    page_size: Optional[int] = None
    page_offset: Optional[int] = None
    num_rows: Optional[int] = None
    result: Optional[List[PoetScoreResult]] = None
    n_completed: Optional[int] = None

class PoetSSPResult(NewModel):
    sequence: bytes
    score: List[float]
    name: Optional[str] = None

    @validator('sequence', pre=True, always=True)
    def replacename(cls, value):
        """rename X0X"""
        if "X0X" in str(value):
            return b'input'
        return value

class PoetSSPJob(PoetScoreJob):
    parent_id: Optional[str] = None
    s3prefix: Optional[str] = None
    page_size: Optional[int] = None
    page_offset: Optional[int] = None
    num_rows: Optional[int] = None
    result: Optional[List[PoetSSPResult]] = None
    n_completed: Optional[int] = None

class Prediction(NewModel):
    """Prediction details."""

    model_id: str
    model_name: str
    properties: Dict[str, Dict[str, float]]


class PredictJobBase(NewModel):
    """Shared properties for predict job outputs."""

    # might be none if just fetching
    job_id: Optional[str] = None
    job_type: str
    status: str


class DesignJob(NewModel):
    job_id: Optional[str] = None
    job_type: str
    status: str


class PredictJob(PredictJobBase):
    """Properties about predict job returned via API."""

    class SequencePrediction(NewModel):
        """Sequence prediction."""

        sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None


class PredictSingleSiteJob(PredictJobBase):
    """Properties about single-site prediction job returned via API."""

    class SequencePrediction(NewModel):
        """Sequence prediction."""

        position: int
        amino_acid: str
        # sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None


class CVItem(NewModel):
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
