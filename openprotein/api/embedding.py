from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture, MappedAsyncJobFuture, job_get
import openprotein.config as config
from openprotein.jobs import Job, ResultsParser, JobStatus
from openprotein.api.align import PromptFuture, validate_prompt
from openprotein.api.poet import (
    PoetGenerateFuture,
    poet_score_post,
    poet_single_site_post,
    poet_generate_post,
)
from openprotein.futures import FutureBase, FutureFactory

from openprotein.pydantic import BaseModel, parse_obj_as
import numpy as np
from typing import Optional, List, Union, Any
import io
from datetime import datetime


PATH_PREFIX = "v1/embeddings"


class ModelDescription(BaseModel):
    citation_title: Optional[str] = None
    doi: Optional[str] = None
    summary: str = "Protein language model for embeddings"


class TokenInfo(BaseModel):
    id: int
    token: str
    primary: bool
    description: str


class ModelMetadata(BaseModel):
    model_id: str
    description: ModelDescription
    max_sequence_length: Optional[int] = None
    dimension: int
    output_types: List[str]
    input_tokens: List[str]
    output_tokens: Optional[List[str]] = None
    token_descriptions: List[List[TokenInfo]]


class EmbeddedSequence(BaseModel):
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


class SVDMetadata(BaseModel):
    id: str
    status: JobStatus
    created_date: Optional[datetime] = None
    model_id: str
    n_components: int
    reduction: Optional[str] = None
    sequence_length: Optional[int] = None

    def is_done(self):
        return self.status.done()


# split these out by module for another layer of control
class EmbBase:
    # overridden by subclasses
    # get correct emb model
    model_id = None

    @classmethod
    def get_model(cls):
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id


class EmbFactory:
    """Factory class for creating Future instances based on job_type."""

    @staticmethod
    def create_model(session, model_id, default=None):
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Returns:
        - An instance of the appropriate Future class.
        """
        # Dynamically discover all subclasses of FutureBase
        future_classes = EmbBase.__subclasses__()

        # Find the Future class that matches the job type
        for future_class in future_classes:
            if model_id in future_class.get_model():
                return future_class(session=session, model_id=model_id)
        # default to ProtembedModel
        try:
            return default(session=session, model_id=model_id)
        except Exception:  # type: ignore
            raise ValueError(f"Unsupported model_id type: {model_id}")


def embedding_models_list_get(session: APISession) -> List[str]:
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


def embedding_model_get(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def embedding_get(session: APISession, job_id: str) -> FutureBase:
    """
    Get Job associated with the given request ID.

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

    # endpoint = PATH_PREFIX + f"/{job_id}"
    # response = session.get(endpoint)
    return FutureFactory.create_future(session=session, job_id=job_id)


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
    return parse_obj_as(List[bytes], response.json())


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
    if isinstance(sequence, bytes):
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


class EmbeddingResultFuture(MappedAsyncJobFuture, FutureBase):
    """Future Job for manipulating results"""

    job_type = [
        "/embeddings/embed",
        "/embeddings/svd",
        "/embeddings/attn",
        "/embeddings/logits",
        "/embeddings/embed_reduced",
        "/svd/fit",
        "/svd/embed",
    ]

    def __init__(
        self,
        session: APISession,
        job: Job,
        sequences=None,
        max_workers=config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session, job, max_workers)
        self._sequences = sequences

    def get(self, verbose=False) -> List:
        return super().get(verbose=verbose)

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


def embedding_model_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes],
    reduction: Optional[str] = "MEAN",
    prompt_id: Optional[str] = None,
):
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
    kwargs:
        accepts prompt_id for Poet Jobs

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/embed"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    if "prompt_id":
        body["prompt_id"] = prompt_id
    body["reduction"] = reduction
    response = session.post(endpoint, json=body)
    return FutureFactory.create_future(
        session=session, response=response, sequences=sequences
    )


def embedding_model_logits_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes],
    prompt_id: Optional[str] = None,
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

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    if "prompt_id":
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return FutureFactory.create_future(
        session=session, response=response, sequences=sequences
    )


def embedding_model_attn_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes],
    prompt_id: Optional[str] = None,
) -> Job:
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

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    if "prompt_id":
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return FutureFactory.create_future(
        session=session, response=response, sequences=sequences
    )


def svd_list_get(session: APISession) -> List[SVDMetadata]:
    """Get SVD job metadata for all SVDs. Including SVD dimension and sequence lengths."""
    endpoint = PATH_PREFIX + "/svd"
    response = session.get(endpoint)
    return parse_obj_as(List[SVDMetadata], response.json())


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
    session.delete(endpoint)
    return True


def svd_fit_post(
    session: APISession,
    model_id: str,
    sequences: List[bytes],
    n_components: int = 1024,
    reduction: Optional[str] = None,
    prompt_id: Optional[str] = None,
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

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "model_id": model_id,
        "sequences": sequences_unicode,
        "n_components": n_components,
    }
    if reduction is not None:
        body["reduction"] = reduction
    if prompt_id is not None:
        body["prompt_id"] = prompt_id

    response = session.post(endpoint, json=body)
    # return job for metadata
    return ResultsParser.parse_obj(response)


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

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return FutureFactory.create_future(
        session=session, response=response, sequences=sequences
    )


class ProtembedModel(EmbBase):
    """
    Class providing inference endpoints for protein embedding models served by OpenProtein.
    """

    model_id = "protembed"

    def __init__(self, session, model_id, metadata=None):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        self.__doc__ = self.__fmt_doc()

    def __fmt_doc(self):
        summary = str(self.metadata.description.summary)
        return f"""\t{summary}
        \t max_sequence_length = {self.metadata.max_sequence_length}
        \t supported outputs = {self.metadata.output_types}
        \t supported tokens = {self.metadata.input_tokens} 
        """

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

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

    def embed(
        self, sequences: List[bytes], reduction: Optional[str] = "MEAN"
    ) -> EmbeddingResultFuture:
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
        return embedding_model_post(
            self.session, model_id=self.id, sequences=sequences, reduction=reduction
        )

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
        return embedding_model_logits_post(self.session, self.id, sequences)

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
        return embedding_model_attn_post(self.session, self.id, sequences)

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
        if isinstance(job, Job):
            job_id = job.job_id
        else:
            job_id = job.job.job_id
        metadata = svd_get(self.session, job_id)
        return SVDModel(self.session, metadata)


class SVDModel(AsyncJobFuture, FutureBase):
    """
    Class providing embedding endpoint for SVD models. \
        Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    """

    # actually a future, not a model!
    job_type = "/svd"

    def __init__(self, session: APISession, metadata: SVDMetadata):
        self.session = session
        self._metadata = metadata
        self._job = None

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

    def get(self, verbose: bool = False):
        # overload for AsyncJobFuture
        return self

    @property
    def job(self) -> Job:
        if self._job is None:
            self._job = self.get_job()
        return self._job

    @job.setter
    def job(self, j):
        self._job = j

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
        return svd_embed_post(self.session, self.id, sequences)
        # return EmbeddingResultFuture(self.session, job, sequences=sequences)


class OpenProteinModel(ProtembedModel):
    """
    Class providing inference endpoints for proprietary protein embedding models served by OpenProtein.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.prot_seq?
    """


class PoETModel(OpenProteinModel, EmbBase):
    """
    Class for OpenProtein's foundation model PoET - NB. PoET functions are dependent on a prompt supplied via the align endpoints.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.poet?


    """

    model_id = "poet"

    # Add model to explicitly require prompt_id
    def __init__(self, session, model_id, metadata=None):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        # could add prompt here?

    def embed(
        self,
        prompt: Union[str, PromptFuture],
        sequences: List[bytes],
        reduction: Optional[str] = "MEAN",
    ) -> EmbeddingResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to embed.
        reduction: str
            embeddings reduction to use (e.g. mean)
        Returns
        -------
            EmbeddingResultFuture
        """
        prompt_id = validate_prompt(prompt)
        # return super().embed(sequences=sequences, reduction=reduction, prompt_id=prompt_id)
        return embedding_model_post(
            self.session,
            model_id=self.id,
            sequences=sequences,
            prompt_id=prompt_id,
            reduction=reduction,
        )

    def logits(
        self,
        prompt: Union[str, PromptFuture],
        sequences: List[bytes],
    ) -> EmbeddingResultFuture:
        """
        logit embeddings for sequences using this model.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.

        Returns
        -------
            EmbeddingResultFuture
        """
        prompt_id = validate_prompt(prompt)
        # return super().logits(sequences=sequences, prompt_id=prompt_id)
        return embedding_model_logits_post(
            self.session, self.id, sequences=sequences, prompt_id=prompt_id
        )

    def attn():
        """Not Available for Poet."""
        raise ValueError("Attn not yet supported for poet")

    def score(self, prompt: Union[str, PromptFuture], sequences: List[bytes]):
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.
        Returns
        -------
        results
            The scores of the query sequences.
        """
        prompt_id = validate_prompt(prompt)
        return poet_score_post(self.session, prompt_id, sequences)

    def single_site(self, prompt: Union[str, PromptFuture], sequence: bytes):
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.
        Returns
        -------
        results
            The scores of the mutated sequence.
        """
        prompt_id = validate_prompt(prompt)
        return poet_single_site_post(self.session, sequence, prompt_id=prompt_id)

    def generate(
        self,
        prompt: Union[str, PromptFuture],
        num_samples=100,
        temperature=1.0,
        topk=None,
        topp=None,
        max_length=1000,
        seed=None,
    ) -> PoetGenerateFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        num_samples : int, optional
            The number of samples to generate, by default 100.
        temperature : float, optional
            The temperature for sampling. Higher values produce more random outputs, by default 1.0.
        topk : int, optional
            The number of top-k residues to consider during sampling, by default None.
        topp : float, optional
            The cumulative probability threshold for top-p sampling, by default None.
        max_length : int, optional
            The maximum length of generated proteins, by default 1000.
        seed : int, optional
            Seed for random number generation, by default a random number.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        Job
            An object representing the status and information about the generation job.
        """
        prompt_id = validate_prompt(prompt)
        return poet_generate_post(
            self.session,
            prompt_id,
            num_samples=num_samples,
            temperature=temperature,
            topk=topk,
            topp=topp,
            max_length=max_length,
            random_seed=seed,
        )

    def fit_svd(
        self,
        prompt: Union[str, PromptFuture],
        sequences: List[bytes],
        n_components: int = 1024,
        reduction: Optional[str] = None,
    ) -> SVDModel:  # type: ignore
        """
        Fit an SVD on the embedding results of this model. 

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
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
        prompt = validate_prompt(prompt)

        job = svd_fit_post(
            self.session,
            model_id=self.id,
            sequences=sequences,
            n_components=n_components,
            reduction=reduction,
            prompt_id=prompt,
        )
        metadata = svd_get(self.session, job.job_id)
        return SVDModel(self.session, metadata)


class ESMModel(ProtembedModel):
    """
    Class providing inference endpoints for Facebook's ESM protein language Models.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.esm2_t12_35M_UR50D?"""


class EmbeddingAPI:
    """
    This class defines a high level interface for accessing the embeddings API.

    You can access all our models either via :meth:`get_model` or directly through the session's embedding attribute using the model's ID and the desired method. For example, to use the attention method on the protein sequence model, you would use ``session.embedding.prot_seq.attn()``.

    Examples
    --------
    Accessing a model's method:

    .. code-block:: python

        # To call the attention method on the protein sequence model:
        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.prot_seq.attn()

    Using the `get_model` method:

    .. code-block:: python

        # Get a model instance by name:
        import openprotein
        session = openprotein.connect(username="user", password="password")
        # list available models:
        print(session.embedding.list_models() )
        # init model by name
        model = session.embedding.get_model('prot-seq')
    """

    # added for static typing, eg pylance, for autocomplete
    # at init these are all overwritten.
    prot_seq: OpenProteinModel
    rotaprot_large_uniref50w: OpenProteinModel
    rotaprot_large_uniref90_ft: OpenProteinModel
    poet: PoETModel

    esm1b_t33_650M_UR50S: ESMModel
    esm1v_t33_650M_UR90S_1: ESMModel
    esm1v_t33_650M_UR90S_2: ESMModel
    esm1v_t33_650M_UR90S_3: ESMModel
    esm1v_t33_650M_UR90S_4: ESMModel
    esm1v_t33_650M_UR90S_5: ESMModel
    esm2_t12_35M_UR50D: ESMModel
    esm2_t30_150M_UR50D: ESMModel
    esm2_t33_650M_UR50D: ESMModel
    esm2_t36_3B_UR50D: ESMModel
    esm2_t6_8M_UR50D: ESMModel

    def __init__(self, session: APISession):
        self.session = session
        # dynamically add models from  api list
        self._load_models()

    def _load_models(self):
        # Dynamically add model instances as attributes - precludes any drift
        models = self.list_models()
        for model in models:
            model_name = model.id.replace("-", "_")  # hyphens out
            setattr(self, model_name, model)

    def list_models(self) -> List[ProtembedModel]:
        """list models available for creating embeddings of your sequences"""
        models = []
        for model_id in embedding_models_list_get(self.session):
            models.append(
                EmbFactory.create_model(
                    session=self.session, model_id=model_id, default=ProtembedModel
                )
            )
        return models

    def get_model(self, name: str):
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
        model_name = name.replace("-", "_")
        return getattr(self, model_name)

    def __get_results(self, job) -> EmbeddingResultFuture:
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
        return FutureFactory.create_future(job=job, session=self.session)

    def __fit_svd(
        self,
        model_id: str,
        sequences: List[bytes],
        n_components: int = 1024,
        reduction: Optional[str] = None,
        **kwargs,
    ) -> SVDModel:
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
            sequences=sequences,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
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

    def __delete_svd(self, svd_id: str) -> bool:
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

    def __get_svd_results(self, job: Job):
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
