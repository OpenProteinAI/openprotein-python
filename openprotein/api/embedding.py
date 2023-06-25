from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, PagedAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
import numpy as np
from base64 import b64decode
from typing import Optional, List, Tuple, Dict, Union



def embedding_models_get(session: APISession) -> List[str]:
    endpoint = 'v1/embeddings/models'
    response = session.get(endpoint)
    result = response.json()['models']
    return result


def decode_embedding(data, shape, dtype=np.float32):
    data = b64decode(data)
    array = np.frombuffer(data, dtype=dtype)
    # TODO - remove this work-around for error in returned shape of fixed-sized embeddings
    num_entries = 1
    for i in range(len(shape)):
        num_entries *= shape[i]
    if len(array) < num_entries:
        shape = shape[-1:] # only use last dimension for size estimate
    # end work-around
    array = array.reshape(*shape)
    return array


class EmbeddingResult(pydantic.BaseModel):
    data: bytes
    sequence: bytes
    shape: List[int]

    def to_numpy(self):
        dtype = np.float32
        array = decode_embedding(self.data, self.shape, dtype=dtype)
        return array


class EmbeddingJob(Job):
    results: Optional[List[EmbeddingResult]]


def embedding_get(session: APISession, job_id: str, page_offset: int = 0, page_size: int = 10):
    endpoint = f'v1/embeddings/{job_id}'
    response = session.get(endpoint, params={'offset': page_offset, 'limit': page_size})
    return EmbeddingJob(**response.json())


class EmbeddingResultFuture(PagedAsyncJobFuture):
    DEFAULT_PAGE_SIZE = config.EMBEDDING_PAGE_SIZE

    def get_slice(self, start, end) -> List[Tuple[bytes, np.ndarray]]:
        assert end >= start
        response = embedding_get(
            self.session,
            self.job.job_id,
            page_offset=start,
            page_size=(end - start),
        )
        #return response.results
        return [(r.sequence, r.to_numpy()) for r in response.results]


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


def svd_list_get(session: APISession):
    endpoint = 'v1/embeddings/svd'
    response = session.get(endpoint)
    return response.json()


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
    
    def fit_svd(self, sequences: List[bytes]):
        job = svd_fit_post(self.session, self.id, sequences)
        return SVDModel(self.session, job)


class SVDModel(EmbeddingResultFuture):
    """
    Class providing embedding endpoint for SVD models. Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    """
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

    def list_models(self) -> List[ProtembedModel]:
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
    
    def fit_svd(self, model_id: str, sequences: List[bytes]):
        job = svd_fit_post(self.session, model_id, sequences)
        return SVDModel(self.session, job)
    
    def get_svd(self, job):
        return SVDModel(self.session, job)
    
    def get_svd_results(self, job):
        return EmbeddingResultFuture(self.session, job)
    
    def list_svd(self):
        return svd_list_get(self.session)

