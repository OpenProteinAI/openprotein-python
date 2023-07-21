
from typing import Optional, List, Union, Dict, Literal

from io import BytesIO
from datetime import datetime 
import time 

import pydantic
from enum import Enum

import openprotein.config as config
from openprotein.base import APISession


class DesignMetadata(pydantic.BaseModel):
    y_mu: Optional[float]
    y_var: Optional[float]
class DesignSubscore(pydantic.BaseModel):
    score: int
    metadata: DesignMetadata

class DesignStep(pydantic.BaseModel):
    step: int
    sample_index: int
    sequence: str
    #scores: List[int]
    #subscores_metadata: List[List[DesignSubscore]]
    initial_scores: List[int] = pydantic.Field(..., alias='scores')  # renaming 'scores' to 'initial_scores'
    scores: List[List[DesignSubscore]] = pydantic.Field(..., alias='subscores_metadata')  # renaming 'subscores_metadata' to 'scores'
    umap1: float
    umap2: float


class DesignResults(pydantic.BaseModel):
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

class Criterion(pydantic.BaseModel):
    target: float
    weight: float
    direction: str

class ModelCriterion(pydantic.BaseModel):
    criterion_type: Literal["model"]
    model_id: str
    measurement_name: str
    criterion: Criterion

class NMutationCriterion(pydantic.BaseModel):
    criterion_type: Literal["n_mutations"]
    #sequences: Optional[List[str]]

CriterionItem = Union[ModelCriterion, NMutationCriterion]

class DesignJobCreate(pydantic.BaseModel):
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



class JobStatus(str, Enum):
    PENDING: str = 'PENDING'
    RUNNING: str = 'RUNNING'
    SUCCESS: str = 'SUCCESS'
    FAILURE: str = 'FAILURE'
    RETRYING: str = 'RETRYING'
    CANCELED: str = 'CANCELED'

    def done(self):
        return (self is self.SUCCESS) or (self is self.FAILURE) or (self is self.CANCELED)

    def cancelled(self):
        return self is self.CANCELED


class Job(pydantic.BaseModel):
    status: JobStatus
    job_id: str
    job_type: str
    created_date: Optional[datetime]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    prerequisite_job_id: Optional[str]
    progress_message: Optional[str]
    progress_counter: Optional[int]

    def refresh(self, session: APISession):
        """ refresh job status"""
        return job_get(session, self.job_id)

    def done(self) -> bool:
        """ Check if job is complete"""
        return self.status.done()

    def cancelled(self) -> bool:
        """ check if job is cancelled"""
        return self.status.cancelled()

    def wait(self, session: APISession,
             interval:int=config.POLLING_INTERVAL,
             timeout:Optional[int]=None,
             verbose:bool=False):
        """
        Wait for a job to finish, and then get the results. 

        Args:
            session (APISession): Auth'd APIsession
            interval (int, optional): Wait between polls (secs). Defaults to config.POLLING_INTERVAL.
            timeout (int, optional): Max. time to wait before raising error. Defaults to unlimited.
            verbose (bool, optional): print status updates. Defaults to False.

        Raises:
            TimeoutException: _description_

        Returns:
            _type_: _description_
        """
        start_time = time.time()
        
        def is_done(job: Job):
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutException(f'Wait time exceeded timeout {timeout}, waited {elapsed_time}')
            return job.done()
        
        pbar = None
        if verbose:
            pbar = tqdm.tqdm()

        job = self.refresh(session)
        while not is_done(job):
            if verbose:
                pbar.update(1)
                pbar.set_postfix({'status': job.status})
                #print(f'Retry {retries}, status={self.job.status}, time elapsed {time.time() - start_time:.2f}')
            time.sleep(interval)
            job = job.refresh(session)
        
        if verbose:
            pbar.update(1)
            pbar.set_postfix({'status': job.status})

        return job

class Jobplus(Job):
    sequence_length: Optional[int]
 
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
    num_sequences: Optional[int] = pydantic.Field(None, ge=0, lt=100)
    num_residues: Optional[int] = pydantic.Field(None, ge=0, lt=24577)
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    homology_level: float = pydantic.Field(0.8, ge=0, le=1)
    max_similarity: float = pydantic.Field(1.0, ge=0, le=1)
    min_similarity: float = pydantic.Field(0.0, ge=0, le=1)
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

class DesignJob(pydantic.BaseModel):
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