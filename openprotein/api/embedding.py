from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, BinaryIO


from enum import Enum
from io import BytesIO
import random
import csv
import codecs
import requests


def embedding_models_get(session: APISession) -> List[str]:
    endpoint = 'v1/embeddings/models'
    response = session.get(endpoint)
    result = response.json()['models']
    return result


class EmbeddingResult(pydantic.BaseModel):
    data: bytes
    sequence: bytes
    shape: List[int]

    def to_numpy(self):
        dtype = np.float32
        shape = self.shape
        array = np.frombuffer(self.data, dtype=dtype).reshape(*shape)
        return array


class EmbeddingJob(Job):
    results: Optional[List[EmbeddingResult]]


def embedding_get(session: APISession, job_id: str, page_offset: int = 0, page_size: int = 10):
    endpoint = f'v1/embeddings/{job_id}'
    response = session.get(endpoint, params={'offset': page_offset, 'limit': page_size})
    return EmbeddingJob(**response.json())


class EmbeddingResultFuture(AsyncJobFuture):
    def __init__(self, session: APISession, job: Job, page_size=config.EMBEDDING_PAGE_SIZE):
        super().__init__(session, job)
        self.page_size = page_size

    def get_slice(self, start, end) -> List[Tuple[bytes, np.ndarray]]:
        assert end >= start
        response = embedding_get(
            self.session,
            self.job.job_id,
            page_offset=start,
            page_size=(end - start),
        )
        return [(r.sequence, r.to_numpy()) for r in response.results]

    def get(self) -> List[Tuple[bytes, np.ndarray]]:
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            result_page = self.get_slice(offset, offset + step)
            results += result_page
            num_returned = len(result_page)
            offset += num_returned
        
        return results


def embedding_post(session: APISession, model_id: str, sequences: List[bytes], reduction=None):
    endpoint = f'v1/embeddings/{model_id}/embed'

    # TODO - fix reduction parameters, improve clarity between mean and SVD options
    reduction_params = {}
    if reduction == 'mean':
        reduction_params['mean'] = True

    sequences = [s.decode() for s in sequences]
    body = {
        'sequences': sequences,
        'reduction': reduction_params,
    }
    response = session.post(endpoint, json=body)
    return EmbeddingJob(**response.json())


class SVDJob(Job):
    pass


def svd_fit_post(session: APISession, model_id: str, sequences: List[bytes]):
    endpoint = 'v1/embeddings/svd'

    sequences = [s.decode() for s in sequences]
    body = {
        'model': model_id,
        'sequences': sequences,
    }
    response = session.post(endpoint, json=body)
    return SVDJob(**response.json())


def svd_embed_post(session: APISession, svd_id: str, sequences: List[bytes]):
    endpoint = f'v1/embeddings/svd/{svd_id}/embed'

    sequences = [s.decode() for s in sequences]
    body = {'sequences': sequences}
    response = session.post(endpoint, json=body)
    return EmbeddingJob(**response.json())


class ProtembedModel:
    """
    Class providing inference endpoints for protein embedding models served by OpenProtein.
    """
    def __init__(self, session, model_id, metadata=None):
        self.session = session
        self.id = model_id
        self.metadata = metadata

    def __str__(self) -> str:
        return self.id
    
    def __repr__(self) -> str:
        return self.id

    def embed(self, sequences: List[bytes], reduction=None):
        job = embedding_post(self.session, self.id, sequences, reduction=reduction)
        return EmbeddingResultFuture(self.session, job)


class SVDModel:
    """
    Class providing embedding endpoint for SVD models.
    """
    def __init__(self, session, job):
        self.session = session
        self.job = job

    @property
    def id(self):
        return self.job.job_id
    
    def embed(self, sequences: List[bytes]):
        job = svd_embed_post(self.session, self.id, sequences)
        return EmbeddingResultFuture(self.session, job)


class EmbeddingAPI:
    """
    This class defines a high level interface for accessing the embeddings API.
    """
    def __init__(self, session: APISession):
        self.session = session

    def list_models(self):
        models = []
        for model_id in embedding_models_get(self.session):
            models.append(ProtembedModel(self.session, model_id))
        return models

    def embed(self, model: Union[ProtembedModel, SVDModel, str], sequences: List[bytes], reduction=None):
        """
        Embed sequences using the specified model.
        """
        if type(model) is ProtembedModel:
            model_id = model.id
            job = embedding_post(self.session, model_id, sequences, reduction=reduction)
        elif type(model) is SVDModel:
            svd_id = model.id
            job = svd_embed_post(self.session, svd_id, sequences)
        else:
            # we assume model is the model_id
            model_id = model
            job = embedding_post(self.session, model_id, sequences, reduction=reduction)
        return EmbeddingResultFuture(self.session, job)
    
    def svd_fit(self, model_id: str, sequences: List[bytes]):
        job = svd_fit_post(self.session, model_id, sequences)
        return SVDModel(self.session, job)

