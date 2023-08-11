from openprotein.base import APISession
from openprotein.api.jobs import (
    Job,
    MappedAsyncJobFuture,
    PagedAsyncJobFuture,
    job_get,
    JobStatus,
)
from openprotein.errors import InvalidJob
import openprotein.config as config
from openprotein.api.jobs import load_job
from openprotein.models import (
    ModelDescription,
    TokenInfo,
    ModelMetadata,
    EmbeddedSequence,
    SVDMetadata,
    SVDJob,
    JobType,
)
import pydantic
import numpy as np
from datetime import datetime
from base64 import b64decode
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import namedtuple
import io


PATH_PREFIX = "v1/embeddings"


def embedding_models_get(session: APISession) -> List[str]:
    """
    List available embeddings models.

    Args:
        session (APISession): API session

    Returns:
        List[str]: list of model names.
    """

    endpoint = PATH_PREFIX + "/models"
    response = session.get(endpoint)
    result = response.json()
    return result


# alias embedding_models_list_get to embedding_models_get
embedding_models_list_get = embedding_models_get


def embedding_model_get(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def embedding_get(session: APISession, job_id: str) -> List[bytes]:
    """
    Get sequences associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to fetch

    Returns
    -------
    sequences : List[bytes]
    """

    endpoint = PATH_PREFIX + f"/{job_id}"
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[bytes], response.json())


def embedding_get_sequences(session: APISession, job_id: str) -> List[bytes]:
    """
    Get results associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to fetch

    Returns
    -------
    sequences : List[bytes]
    """
    endpoint = PATH_PREFIX + f"/{job_id}/sequences"
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[bytes], response.json())


def embedding_get_sequence_result(
    session: APISession, job_id: str, sequence: bytes
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from
    sequence : bytes
        sequence to retrieve results for

    Returns
    -------
    result : bytes
    """
    sequence = sequence.decode()
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def decode_embedding(data: bytes) -> np.ndarray:
    """
    Decode embedding.

    Args:
        data (bytes): raw bytes encoding the array received over the API

    Returns:
        np.ndarray: decoded array
    """
    s = io.BytesIO(data)
    return np.load(s, allow_pickle=False)


class EmbeddingResultFuture(MappedAsyncJobFuture):
    """Future Job for manipulating results"""
    def __init__(
        self,
        session: APISession,
        job: Job,
        sequences=None,
        max_workers=config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session, job, max_workers)
        self._sequences = sequences

    @property
    def sequences(self):
        if self._sequences is None:
            self._sequences = embedding_get_sequences(self.session, self.job.job_id)
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def keys(self):
        return self.sequences

    def get_item(self, sequence: bytes) -> np.ndarray:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: embeddings
        """
        data = embedding_get_sequence_result(self.session, self.job.job_id, sequence)
        return decode_embedding(data)


def embedding_model_post(session: APISession,
                         model_id: str,
                         sequences: List[bytes],
                         reduction: Optional[str]="MEAN"):
    """
    POST a request for embeddings from the given model ID. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for
    reduction : Optional[str]
        reduction to apply to the embeddings. options are None, "MEAN", or "SUM". defaul: "MEAN"

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/embed"

    sequences = [s.decode() for s in sequences]
    body = {
        "sequences": sequences,
    }
    body['reduction'] = reduction
    response = session.post(endpoint, json=body)
    return Job(**response.json())


# alias embedding_post to embedding_model_post
embedding_post = embedding_model_post


def embedding_model_logits_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes]
) -> Job:
    """
    POST a request for logits from the given model ID. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/logits"

    sequences = [s.decode() for s in sequences]
    body = {
        "sequences": sequences,
    }
    response = session.post(endpoint, json=body)
    return Job(**response.json())


def embedding_model_attn_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes] ) -> Job:
    """
    POST a request for attention embeddings from the given model ID. \
        Returns a Job object referring to this request \
            that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/attn"

    sequences = [s.decode() for s in sequences]
    body = {
        "sequences": sequences,
    }
    response = session.post(endpoint, json=body)
    return Job(**response.json())


def svd_list_get(session: APISession) -> List[SVDMetadata]:
    """Get SVD job metadata for all SVDs. Including SVD dimension and sequence lengths."""
    endpoint = PATH_PREFIX + "/svd"
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[SVDMetadata], response.json())


def svd_get(session: APISession, svd_id: str) -> SVDMetadata:
    """Get SVD job metadata. Including SVD dimension and sequence lengths."""
    endpoint = PATH_PREFIX + f"/svd/{svd_id}"
    response = session.get(endpoint)
    return SVDMetadata(**response.json())


def svd_delete(session: APISession, svd_id: str):
    """
    Delete and SVD model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    svd_id : str
        SVD model to delete

    Returns
    -------
    bool
    """

    endpoint = PATH_PREFIX + f"/svd/{svd_id}"
    response = session.delete(endpoint)
    return True


def svd_fit_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes],
    n_components: int = 1024,
    reduction: Optional[str] = None,
):
    """
    Create SVD fit job.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model to use
    sequences : List[bytes]
        sequences to SVD
    n_components : int
        number of SVD components to fit. default = 1024
    reduction : Optional[str]
        embedding reduction to use for fitting the SVD. default = None

    Returns
    -------
    SVDJob
    """

    endpoint = PATH_PREFIX + "/svd"

    sequences = [s.decode() for s in sequences]
    body = {
        "model_id": model_id,
        "sequences": sequences,
        "n_components": n_components,
    }
    if reduction is not None:
        body["reduction"] = reduction
    response = session.post(endpoint, json=body)
    return SVDJob(**response.json())


def svd_embed_post(session: APISession, svd_id: str, sequences: List[bytes]) -> Job:
    """
    POST a request for embeddings from the given SVD model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    svd_id : str
        SVD model to use
    sequences : List[bytes]
        sequences to SVD

    Returns
    -------
    Job
    """
    endpoint = PATH_PREFIX + f"/svd/{svd_id}/embed"

    sequences = [s.decode() for s in sequences]
    body = {"sequences": sequences}
    response = session.post(endpoint, json=body)
    return Job(**response.json())


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

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for this model.

        Returns
        -------
            ModelMetadata
        """
        if self._metadata is not None:
            return self._metadata
        self._metadata = embedding_model_get(self.session, self.id)
        return self._metadata

    def embed(self, sequences: List[bytes], reduction: Optional[str]="MEAN"):
        """
        Embed sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD
        reduction: str
            embeddings reduction to use (e.g. mean)

        Returns
        -------
            EmbeddingResultFuture
        """
        job = embedding_model_post(self.session,
                                   model_id=self.id,
                                   sequences=sequences,
                                   reduction=reduction)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)

    def logits(self, sequences: List[bytes]) -> EmbeddingResultFuture:
        """
        logit embeddings for sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD

        Returns
        -------
            EmbeddingResultFuture
        """
        job = embedding_model_logits_post(self.session, self.id, sequences)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)

    def attn(self, sequences: List[bytes]) -> EmbeddingResultFuture:
        """
        Attention embeddings for sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD

        Returns
        -------
            EmbeddingResultFuture
        """
        job = embedding_model_attn_post(self.session, self.id, sequences)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)

    def fit_svd(
        self,
        sequences: List[bytes],
        n_components: int = 1024,
        reduction: Optional[str] = None,
    ) -> Any:
        """
        Fit an SVD on the embedding results of this model. 

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        sequences : List[bytes] 
            sequences to SVD
        n_components: int 
            number of components in SVD. Will determine output shapes
        reduction: str
            embeddings reduction to use (e.g. mean)

        Returns
        -------
            SVDModel
        """
        model_id = self.id
        job = svd_fit_post(
            self.session,
            model_id,
            sequences,
            n_components=n_components,
            reduction=reduction,
        )
        metadata = svd_get(self.session, job.job_id)
        return SVDModel(self.session, metadata)


class SVDModel:
    """
    Class providing embedding endpoint for SVD models. \
        Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    """

    def __init__(self, session: APISession, metadata: SVDMetadata):
        self.session = session
        self._metadata = metadata

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def metadata(self):
        self._refresh_metadata()
        return self._metadata

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = svd_get(self.session, self.id)

    @property
    def id(self):
        return self._metadata.id

    @property
    def n_components(self):
        return self._metadata.n_components

    @property
    def sequence_length(self):
        return self._metadata.sequence_length

    @property
    def reduction(self):
        return self._metadata.reduction

    def get_model(self) -> ProtembedModel:
        """Fetch embeddings model"""
        model = ProtembedModel(self.session, self._metadata.model_id)
        return model

    @property
    def model(self) -> ProtembedModel:
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this SVD model.
        """
        return svd_delete(self.session, self.id)

    def get_job(self) -> Job:
        """Get job associated with this SVD model"""
        return job_get(self.session, self.id)

    def get_inputs(self) -> List[bytes]:
        """
        Get sequences used for embeddings job.

        Returns
        -------
            List[bytes]: list of sequences
        """
        return embedding_get_sequences(self.session, job_id=self.id)

    def get_embeddings(self) -> EmbeddingResultFuture:
        """
        Get SVD embedding results for this model.

        Returns
        -------
            EmbeddingResultFuture: class for futher job manipulation
        """
        return EmbeddingResultFuture(self.session, self.get_job())

    def embed(self, sequences: List[bytes]) -> EmbeddingResultFuture:
        """
        Use this SVD model to reduce embeddings results. 

        Parameters
        ----------
        sequences : List[bytes]
            List of protein sequences.

        Returns
        -------
        EmbeddingResultFuture
            Class for further job manipulation.
        """
        job = svd_embed_post(self.session, self.id, sequences)
        return EmbeddingResultFuture(self.session, job, sequences=sequences)


class EmbeddingAPI:
    """
    This class defines a high level interface for accessing the embeddings API.
    """

    def __init__(self, session: APISession):
        self.session = session

    def list_models(self) -> List[ProtembedModel]:
        """list models available for creating embeddings of your sequences"""
        models = []
        for model_id in embedding_models_list_get(self.session):
            models.append(ProtembedModel(self.session, model_id))
        return models

    def get_model(self, model_id: str) -> ProtembedModel:
        """
        Get model by model_id. 

        ProtembedModel allows all the usual job manipulation: \
            e.g. making POST and GET requests for this model specifically. 


        Parameters
        ----------
        model_id : str
            the model identifier

        Returns
        -------
        ProtembedModel
            The model

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return ProtembedModel(self.session, model_id)

    def embed(self,
              model: Union[ProtembedModel,SVDModel, str],
                           sequences: List[bytes],
                           reduction="MEAN"):
        """
        Embed sequences using the specified model.

        Parameters
        ----------
        model : Union[ProtembedModel, SVDModel, str]
            The model to use for embedding. This can be an instance of ProtembedModel, SVDModel,
            or a string representing the model_id.
        sequences : List[bytes]
            List of byte sequences to be embedded.
        reduction : str, optional
            The reduction operation to be applied on the embeddings.
            Options are None, "MEAN", or "SUM". Default is None.

        Returns
        -------
        EmbeddingResultFuture
            An instance of EmbeddingResultFuture

        Raises
        ------
        TypeError
            If the input model is neither ProtembedModel, SVDModel, nor str.
        """
        if isinstance(model, ProtembedModel):
            model_id = model.id
            job = embedding_model_post(
                self.session, model_id, sequences, reduction=reduction
            )
        elif isinstance(model, SVDModel):
            svd_id = model.id
            job = svd_embed_post(self.session, svd_id, sequences)
        else:
            # we assume model is the model_id
            model_id = model
            job = embedding_model_post(
                self.session, model_id, sequences, reduction=reduction
            )
        return EmbeddingResultFuture(self.session, job, sequences=sequences)

    def get_results(self, job) -> EmbeddingResultFuture:
        """
        Retrieves the results of an embedding job.

        Parameters
        ----------
        job : Job
            The embedding job whose results are to be retrieved.

        Returns
        -------
        EmbeddingResultFuture
            An instance of EmbeddingResultFuture
        """
        return EmbeddingResultFuture(self.session, job)

    def fit_svd(
        self,
        model_id: str,
        sequences: List[bytes],
        n_components: int = 1024,
        reduction: Optional[str] = None,
    ):
        """
        Fit an SVD on the sequences with the specified model_id and hyperparameters (n_components).

        Parameters
        ----------
        model_id : str
            The ID of the model to fit the SVD on.
        sequences : List[bytes]
            The list of sequences to use for the SVD fitting.
        n_components : int, optional
            The number of components for the SVD, by default 1024.
        reduction : Optional[str], optional
            The reduction method to apply to the embeddings, by default None.

        Returns
        -------
        SVDModel
            The model with the SVD fit.
        """
        model = self.get_model(model_id)
        return model.fit_svd(
            sequences=sequences, n_components=n_components, reduction=reduction
        )

    def get_svd(self, svd_id: str) -> SVDModel:
        """
        Get SVD job results. Including SVD dimension and sequence lengths.

        Requires a successful SVD job from fit_svd

        Parameters
        ----------
        svd_id : str
            The ID of the SVD  job.
        Returns
        -------
        SVDModel
            The model with the SVD fit.
        """
        metadata = svd_get(self.session, svd_id)
        return SVDModel(self.session, metadata)

    def delete_svd(self, svd_id: str) -> bool:
        """
        Delete SVD model.

        Parameters
        ----------
        svd_id : str
            The ID of the SVD  job.
        Returns
        -------
        bool
            True: successful deletion

        """
        return svd_delete(self.session, svd_id)

    def list_svd(self) -> List[SVDModel]:
        """
        List SVD models made by user.

        Takes no args.

        Returns
        -------
        list[SVDModel]
            SVDModels

        """
        return [
            SVDModel(self.session, metadata) for metadata in svd_list_get(self.session)
        ]

    def get_svd_results(self, job: Job):
        """
        Get SVD job results. Including SVD dimension and sequence lengths.

        Requires a successful SVD job from fit_svd

        Parameters
        ----------
        job : Job
            SVD JobFuture
        Returns
        -------
        SVDModel
            The model with the SVD fit.
        """
        return EmbeddingResultFuture(self.session, job)

    def load_job(self, job_id: str) -> Job:
        """
        Reload a Submitted job to resume from where you left off!


        Parameters
        ----------
        job_id : str
            The identifier of the job whose details are to be loaded.

        Returns
        -------
        Job
            Job

        Raises
        ------
        HTTPError
            If the request to the server fails.
        InvalidJob
            If the Job is of the wrong type

        """
        job_details = load_job(self.session, job_id)
        sequences = embedding_get_sequences(self.session, job_id=job_id)
        if "embed" not in job_details.job_type:
            raise InvalidJob(
                f"Job {job_id} is of type {job_details.job_type} not embeddings"
            )
        if len(sequences) == 0:
            raise InvalidJob(f"Unable to load sequences for job {job_id}")
        return EmbeddingResultFuture(self.session, job_details, sequences=sequences)
