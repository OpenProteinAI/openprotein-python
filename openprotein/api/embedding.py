from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, PagedAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
import numpy as np
from base64 import b64decode
from typing import Optional, List, Tuple, Dict, Union


PATH_PREFIX = 'v1/embeddings'


def embedding_models_get(session: APISession) -> List[str]:
    endpoint = PATH_PREFIX + '/models'
    response = session.get(endpoint)
    result = response.json()
    return result


def decode_embedding(data, shape, dtype=np.float32):
    data = b64decode(data)
    array = np.frombuffer(data, dtype=dtype)
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
    endpoint = PATH_PREFIX + f'/{job_id}'
    response = session.get(endpoint, params={'page_offset': page_offset, 'page_size': page_size})
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


def embedding_model_post(session: APISession, model_id: str, sequences: List[bytes], reduction=None):
    endpoint = PATH_PREFIX + f'/models/{model_id}/embed'

    sequences = [s.decode() for s in sequences]
    body = {
        'sequences': sequences,
    }
    if reduction is not None:
        body['reduction'] = reduction
    response = session.post(endpoint, json=body)
    return EmbeddingJob(**response.json())


class SVDJob(Job):
    pass


class SVDModelMetadata(pydantic.BaseModel):
    id: str
    model_id: str
    n_components: int
    reduction: Optional[str]
    sequence_length: Optional[int]


def svd_list_get(session: APISession) -> List[SVDModelMetadata]:
    endpoint = PATH_PREFIX + '/svd'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[SVDModelMetadata], response.json())


def svd_get(session: APISession, svd_id: str) -> SVDModelMetadata:
    endpoint = PATH_PREFIX + f'/svd/{svd_id}'
    response = session.get(endpoint)
    return SVDModelMetadata(**response.json())


def svd_fit_post(session: APISession, model_id: str, sequences: List[bytes], n_components: int = 1024, reduction: Optional[str] = None):
    endpoint = PATH_PREFIX + '/svd'

    sequences = [s.decode() for s in sequences]
    body = {
        'model_id': model_id,
        'sequences': sequences,
        'n_components': n_components,
    }
    if reduction is not None:
        body['reduction'] = reduction
    response = session.post(endpoint, json=body)
    return SVDJob(**response.json())


def svd_embed_post(session: APISession, svd_id: str, sequences: List[bytes]):
    endpoint = PATH_PREFIX + f'/svd/{svd_id}/embed'

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
        job = embedding_model_post(self.session, self.id, sequences, reduction=reduction)
        return EmbeddingResultFuture(self.session, job)
    
    def fit_svd(self, sequences: List[bytes], n_components: int = 1024, reduction: Optional[str] = None):
        model_id = self.id
        job = svd_fit_post(self.session, model_id, sequences, n_components=n_components, reduction=reduction)
        metadata = svd_get(self.session, job.job_id)
        return SVDModel(self.session, job, metadata)


class SVDModel(EmbeddingResultFuture):
    """
    Class providing embedding endpoint for SVD models. Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    """
    def __init__(self, session: APISession, job: Job, metadata: SVDModelMetadata, page_size=None, max_workers=config.MAX_CONCURRENT_WORKERS):
        assert job.job_id == metadata.id
        super().__init__(session, job, page_size, max_workers)
        self.metadata = metadata

    def __str__(self) -> str:
        return str(self.metadata)
    
    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self):
        return self.metadata.id
    
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
            job = embedding_model_post(self.session, model_id, sequences, reduction=reduction)
        elif type(model) is SVDModel:
            svd_id = model.id
            job = svd_embed_post(self.session, svd_id, sequences)
        else:
            # we assume model is the model_id
            model_id = model
            job = embedding_model_post(self.session, model_id, sequences, reduction=reduction)
        return EmbeddingResultFuture(self.session, job)
    
    def fit_svd(self, model_id: str, sequences: List[bytes], n_components: int = 1024, reduction: Optional[str] = None):
        job = svd_fit_post(self.session, model_id, sequences, n_components=n_components, reduction=reduction)
        metadata = svd_get(self.session, job.job_id)
        return SVDModel(self.session, job, metadata)
    
    def get_svd(self, job):
        if job is str:
            job_id = job
        else:
            job_id = job.job_id
        metadata = svd_get(self.session, job_id)
        return SVDModel(self.session, job, metadata)
    
    def list_svd(self):
        svds = []
        for metadata in svd_list_get(self.session):
            job = job_get(self.session, metadata.id)
            svds.append(SVDModel(self.session, job, metadata))
        return svds
    
    def get_svd_results(self, job):
        return EmbeddingResultFuture(self.session, job)
