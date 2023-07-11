from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict, Union, BinaryIO
from io import BytesIO
import random
import csv
import codecs
import requests

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

#class PromptPostParams(BaseModel):
#    session: APISession
#    msa_id: str
#    num_sequences: Optional[int] = Field(None, ge=0, lt=100)
#    num_residues: Optional[int] = Field(None, ge=0, lt=24577)
#    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
#    homology_level: float = Field(0.8, ge=0, le=1)
#    max_similarity: float = Field(1.0, ge=0, le=1)
#    min_similarity: float = Field(0.0, ge=0, le=1)
#    always_include_seed_sequence: bool = False
#    num_ensemble_prompts: int = 1
#    random_seed: Optional[int] = None


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
