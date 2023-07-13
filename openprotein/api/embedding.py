from openprotein.base import APISession
from openprotein.api.jobs import Job, MappedAsyncJobFuture, PagedAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
import numpy as np
from base64 import b64decode
from typing import Optional, List, Tuple, Dict, Union
from collections import namedtuple
import io


PATH_PREFIX = 'v1/embeddings'


def embedding_models_list_get(session: APISession) -> List[str]:
    endpoint = PATH_PREFIX + '/models'
    response = session.get(endpoint)
    result = response.json()
    return result


class ModelDescription(pydantic.BaseModel):
    citation_title: Optional[str]
    doi: Optional[str]
    summary: str


class TokenInfo(pydantic.BaseModel):
    id: int
    token: str
    primary: bool
    description: str


class ModelMetadata(pydantic.BaseModel):
    model_id: str
    description: ModelDescription
    max_sequence_length: Optional[int]
    dimension: int
    output_types: List[str]
    input_tokens: List[str]
    output_tokens: List[str]
    token_descriptions: List[List[TokenInfo]]


def embedding_model_get(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f'/models/{model_id}'
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def _decode_embedding(data, shape, dtype=np.float32):
    data = b64decode(data)
    array = np.frombuffer(data, dtype=dtype)
    array = array.reshape(*shape)
    return array


def decode_embedding(data) -> np.ndarray:
    s = io.BytesIO(data)
    return np.load(s, allow_pickle=False)


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


def embedding_get_sequences(session: APISession, job_id: str) -> List[bytes]:
    endpoint = PATH_PREFIX + f'/{job_id}/sequences'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[bytes], response.json())


def embedding_get_sequence_result(session: APISession, job_id: str, sequence: bytes) -> bytes:
    sequence = sequence.decode()
    endpoint = PATH_PREFIX + f'/{job_id}/{sequence}'
    response = session.get(endpoint)
    return response.content


class EmbeddedSequence(pydantic.BaseModel):
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


class _EmbeddingResultFuture(PagedAsyncJobFuture):
    DEFAULT_PAGE_SIZE = config.EMBEDDING_PAGE_SIZE

    def get_slice(self, start, end) -> List[EmbeddedSequence]:
        assert end >= start
        response = embedding_get(
            self.session,
            self.job.job_id,
            page_offset=start,
            page_size=(end - start),
        )
        #return response.results
        return [EmbeddedSequence(sequence=r.sequence, embedding=r.to_numpy()) for r in response.results]
    

class EmbeddingResultFuture(MappedAsyncJobFuture):
    def __init__(self, session: APISession, job: Job, sequences=None, max_workers=config.MAX_CONCURRENT_WORKERS):
        super().__init__(session, job, max_workers)
        self._sequences = sequences

    @property
    def sequences(self):
        if self._sequences is None:
            self._sequences = embedding_get_sequences(self.session, self.job.job_id)
        return self._sequences
    
    def keys(self):
        return self.sequences
    
    def get_item(self, k):
        data = embedding_get_sequence_result(self.session, self.job.job_id, k)
        return decode_embedding(data)


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


class SVDMetadata(pydantic.BaseModel):
    id: str
    model_id: str
    n_components: int
    reduction: Optional[str]
    sequence_length: Optional[int]


def svd_list_get(session: APISession) -> List[SVDMetadata]:
    endpoint = PATH_PREFIX + '/svd'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[SVDMetadata], response.json())


def svd_get(session: APISession, svd_id: str) -> SVDMetadata:
    endpoint = PATH_PREFIX + f'/svd/{svd_id}'
    response = session.get(endpoint)
    return SVDMetadata(**response.json())


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
        self._metadata = metadata

    def __str__(self) -> str:
        return self.id
    
    def __repr__(self) -> str:
        return self.id
    
    @property
    def metadata(self):
        return self.get_metadata()
    
    def get_metadata(self):
        if self._metadata is not None:
            return self._metadata
        self._metadata = embedding_model_get(self.session, self.id)
        return self._metadata

    def embed(self, sequences: List[bytes], reduction=None):
        job = embedding_model_post(self.session, self.id, sequences, reduction=reduction)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)
    
    def fit_svd(self, sequences: List[bytes], n_components: int = 1024, reduction: Optional[str] = None):
        model_id = self.id
        job = svd_fit_post(self.session, model_id, sequences, n_components=n_components, reduction=reduction)
        metadata = svd_get(self.session, job.job_id)
        return SVDModel(self.session, metadata)


class SVDModel:
    """
    Class providing embedding endpoint for SVD models. Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    """
    def __init__(self, session: APISession, metadata: SVDMetadata):
        self.session = session
        self.metadata = metadata

    def __str__(self) -> str:
        return str(self.metadata)
    
    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self):
        return self.metadata.id
    
    @property
    def n_components(self):
        return self.metadata.n_components
    
    @property
    def sequence_length(self):
        return self.metadata.sequence_length
    
    @property
    def reduction(self):
        return self.metadata.reduction
    
    def get_model(self) -> ProtembedModel:
        model = ProtembedModel(self.session, self.metadata.model_id)
        return model
    
    @property
    def model(self) -> ProtembedModel:
        return self.get_model()
    
    def get_job(self) -> Job:
        return job_get(self.session, self.id)
    
    def get_inputs(self) -> List[bytes]:
        return embedding_get_sequences(self.id)
    
    def get_embeddings(self) -> EmbeddingResultFuture:
        return EmbeddingResultFuture(self.session, self.get_job())
    
    def embed(self, sequences: List[bytes]) -> EmbeddingResultFuture:
        job = svd_embed_post(self.session, self.id, sequences)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)


class EmbeddingAPI:
    """
    This class defines a high level interface for accessing the embeddings API.
    """
    def __init__(self, session: APISession):
        self.session = session

    def list_models(self) -> List[ProtembedModel]:
        models = []
        for model_id in embedding_models_list_get(self.session):
            models.append(ProtembedModel(self.session, model_id))
        return models
    
    def get_model(self, model_id) -> ProtembedModel:
        return ProtembedModel(self.session, model_id)

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
        return EmbeddingResultFuture(self.session, job, sequences=sequences)
    
    def get_results(self, job):
        return EmbeddingResultFuture(self.session, job)
    
    def fit_svd(self, model_id: str, *args, **kwargs):
        model = self.get_model(model_id)
        return model.fit_svd(*args, **kwargs)
    
    def get_svd(self, svd_id: str):
        metadata = svd_get(self.session, svd_id)
        return SVDModel(self.session, metadata)
    
    def list_svd(self):
        return [SVDModel(self.session, metadata) for metadata in svd_list_get(self.session)]
    
    def get_svd_results(self, job):
        return EmbeddingResultFuture(self.session, job)
