from openprotein.base import APISession
from openprotein.api.jobs import Job, MappedAsyncJobFuture
import openprotein.config as config
from openprotein.api.embedding import ModelMetadata
from openprotein.api.align import validate_msa, MSAFuture
import openprotein.pydantic as pydantic
from typing import Optional, List, Union, Tuple
from openprotein.futures import FutureBase, FutureFactory
from abc import ABC, abstractmethod


PATH_PREFIX = "v1/fold"


class FoldModelBase:
    # overridden by subclasses
    # get correct fold model

    model_id = None

    @classmethod
    def get_model(cls):
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id


class FoldModelFactory:
    """Factory class for creating Future instances based on job_type."""

    @staticmethod
    def create_model(session, model_id, metadata=None, default=None):
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Returns:
        - An instance of the appropriate Future class.
        """
        # Dynamically discover all subclasses of FutureBase
        future_classes = FoldModelBase.__subclasses__()

        # Find the Future class that matches the job type
        for future_class in future_classes:
            if model_id in future_class.get_model():
                return future_class(
                    session=session, model_id=model_id, metadata=metadata
                )
        # default to FoldModel
        try:
            return default(session=session, model_id=model_id, metadata=metadata)
        except Exception:
            raise ValueError(f"Unsupported model_id type: {model_id}")


def fold_models_list_get(session: APISession) -> List[str]:
    """
    List available fold models.

    Args:
        session (APISession): API session

    Returns:
        List[str]: list of model names.
    """

    endpoint = PATH_PREFIX + "/models"
    response = session.get(endpoint)
    result = response.json()
    return result


def fold_model_get(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def fold_get_sequences(session: APISession, job_id: str) -> List[bytes]:
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


def fold_get_sequence_result(
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


class FoldResultFuture(MappedAsyncJobFuture, FutureBase):
    job_type = ["/embeddings/fold"]
    """Future Job for manipulating results"""

    def __init__(
        self,
        session: APISession,
        job: Job,
        sequences=None,
        max_workers=config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session, job, max_workers)
        if sequences is None:
            sequences = fold_get_sequences(self.session, job_id=job.job_id)
        self._sequences = sequences

    @property
    def sequences(self):
        if self._sequences is None:
            self._sequences = fold_get_sequences(self.session, self.job.job_id)
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def keys(self):
        return self.sequences

    def get(self, verbose=False) -> List[Tuple[str, str]]:
        return super().get(verbose=verbose)

    def get_item(self, sequence: bytes) -> bytes:
        """
        Get fold results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: fold
        """
        data = fold_get_sequence_result(self.session, self.job.job_id, sequence)
        return data  #


def fold_models_esmfold_post(
    session: APISession,
    sequences: List[bytes],
    num_recycles: Optional[int] = None,
):
    """
    POST a request for structure prediction using ESMFold. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    sequences : List[bytes]
        sequences to request results for
    num_recycles : Optional[int]
        number of recycles for structure prediction

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + "/models/esmfold"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    if num_recycles is not None:
        body["num_recycles"] = num_recycles

    response = session.post(endpoint, json=body)
    return FutureFactory.create_future(
        session=session, response=response, sequences=sequences
    )


def fold_models_alphafold2_post(
    session: APISession,
    msa: Union[str, MSAFuture],
    num_recycles: Optional[int] = None,
    num_models: Optional[int] = 1,
    num_relax: Optional[int] = 0,
):
    """
    POST a request for structure prediction using AlphaFold2. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa : Union[str, MSAfuture]
        MSA to use for structure prediction. The first sequence in the MSA is the query sequence.
    num_recycles : Optional[int] = None
        number of recycles for structure prediction
    num_models : Optional[int] = 1
        number of models to predict
    num_relax : Optional[int] = 0
        number of relaxation iterations to run

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + "/models/alphafold2"

    msa_id = msa
    if isinstance(msa, MSAFuture):
        msa_id = msa.msa_id

    body = {
        "msa_id": msa_id,
        "num_models": num_models,
        "num_relax": num_relax,
    }
    if num_recycles is not None:
        body["num_recycles"] = num_recycles

    response = session.post(endpoint, json=body)
    # GET endpoint for AF2 expects the query sequence (first sequence) within the MSA
    # since we don't know what the is, leave the sequence out of the future to be retrieved when calling get()
    return FutureFactory.create_future(session=session, response=response)


class FoldModel(ABC):
    """
    ABC Class providing inference endpoints for protein fold models served by OpenProtein.

    Must implement fold() method.
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
        self._metadata = fold_model_get(self.session, self.id)
        return self._metadata

    @abstractmethod
    def fold(self, sequence: str, **kwargs):
        pass


class ESMFoldModel(FoldModel, FoldModelBase):
    model_id = "esmfold"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

    def fold(self, sequences: List[bytes], num_recycles: int = 1) -> FoldResultFuture:
        """
        Fold sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to fold
        num_recycles : int
            number of times to recycle models
        Returns
        -------
            FoldResultFuture
        """
        return fold_models_esmfold_post(
            self.session, sequences, num_recycles=num_recycles
        )


class AlphaFold2Model(FoldModel, FoldModelBase):
    model_id = "alphafold2"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

    def fold(
        self,
        msa: Union[str, MSAFuture],
        num_recycles: Optional[int] = None,
        num_models: int = 1,
        num_relax: Optional[int] = 0,
    ):
        """
        Post sequences to alphafold model.

        Parameters
        ----------
        msa : Union[str, MSAFuture]
            msa
        num_recycles : int
            number of times to recycle models
        num_models : int
            number of models to train - best model will be used
        max_msa : Union[str, int]
            maximum number of sequences in the msa to use.
        relax_max_iterations : int
            maximum number of iterations

        Returns
        -------
        job : Job
        """
        if msa and not isinstance(msa, str):
            msa = validate_msa(msa)

        return fold_models_alphafold2_post(
            self.session,
            msa,
            num_recycles=num_recycles,
            num_models=num_models,
            num_relax=num_relax,
        )


def validate_fold_id(fold):
    if isinstance(fold, str):
        return fold
    return fold.id


class FoldAPI:
    """
    This class defines a high level interface for accessing the fold API.
    """

    esmfold: ESMFoldModel
    alphafold2: AlphaFold2Model

    def __init__(self, session: APISession):
        self.session = session
        self._load_models()

    @property
    def af2(self):
        """Alias for AlphaFold2"""
        return self.alphafold2

    def _load_models(self):
        # Dynamically add model instances as attributes - precludes any drift
        models = self.list_models()
        for model in models:
            model_name = model.id.replace("-", "_")  # hyphens out
            setattr(self, model_name, model)

    def list_models(self) -> List[FoldModel]:
        """list models available for creating folds of your sequences"""
        models = []
        for model_id in fold_models_list_get(self.session):
            models.append(
                FoldModelFactory.create_model(self.session, model_id, default=FoldModel)
            )
        return models

    def get_model(self, model_id: str) -> FoldModel:
        """
        Get model by model_id. 

        FoldModel allows all the usual job manipulation: \
            e.g. making POST and GET requests for this model specifically. 


        Parameters
        ----------
        model_id : str
            the model identifier

        Returns
        -------
        FoldModel
            The model

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return FoldModelFactory.create_model(
            session=self.session, model_id=model_id, default=FoldModel
        )

    def get_results(self, job) -> FoldResultFuture:
        """
        Retrieves the results of a fold job.

        Parameters
        ----------
        job : Job
            The fold job whose results are to be retrieved.

        Returns
        -------
        FoldResultFuture
            An instance of FoldResultFuture
        """
        return FutureFactory.create_future(job=job, session=self.session)
