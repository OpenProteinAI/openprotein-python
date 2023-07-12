import pydantic
from pydantic import Field
from enum import Enum
from typing import Optional, List, Union
from io import BytesIO
from datetime import datetime 

from openprotein.api.jobs import Job
import openprotein.config as config

class AssayMetadata(pydantic.BaseModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: List[str]

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
