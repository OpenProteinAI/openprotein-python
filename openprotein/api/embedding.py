from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
from enum import Enum
from typing import Optional, List, Dict, Union, BinaryIO
from io import BytesIO
import random
import csv
import codecs
import requests


def embedding_models_get(session: APISession):
    endpoint = 'v1/embeddings/models'
    response = session.get(endpoint)
    result = response.json['models']
    return result


class EmbeddingResult(pydantic.BaseModel):
    data: bytes
    sequence: bytes
    shape: List[int]


class GetEmbeddingJob(Job):
    results: List[EmbeddingResult]


def embedding_get(session: APISession, job_id: str, offset: int = 0, limit: int = 10):
    endpoint = f'v1/embeddings/{job_id}'
    response = session.get(endpoint, params={'offset': offset, 'limit': limit})
    return GetEmbeddingJob(**response.json())


class ReductionSchema(pydantic.BaseModel):
    mean: Optional[bool]
    svd: Optional[str]


# TODO - the REST endpoint doesn't follow the job interface
class PostEmbeddingJob(Job):
    model: str
    output: str
    reduction: ReductionSchema
    sequences: List[bytes]
    user_id: Optional[int] # ?!?!?! TODO do we need this return value?


def embedding_post(session: APISession, model_id: str, sequences: List[bytes], reduction=None):
    endpoint = f'v1/embeddings/{model_id}/embed'

    # TODO - fix reduction parameters, improve clarity between mean and SVD options
    reduction_params = {}
    if reduction == 'mean':
        reduction_params['mean'] = True
    elif reduction is not None:
        reduction_params['svd'] = reduction

    body = {
        'sequences': sequences,
        'reduction': reduction_params,
    }
    response = session.post(endpoint, data=body)
    return PostEmbeddingJob(**response.json())
