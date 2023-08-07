from typing import Optional, List, Union
import pydantic

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture, Job
from openprotein.models import (
    SequenceDataset,
    SequenceData,
    PredictJob,
    PredictSingleSiteJob,
    JobType,
)
from openprotein.errors import InvalidParameterError, APIError, InvalidJob
from openprotein.api.train import TrainFuture
from openprotein.api.jobs import load_job


def _create_predict_job(
    session: APISession,
    endpoint: str,
    payload: dict,
    model_ids: Optional[List[str]] = None,
    train_job_id: Optional[str] = None,
) -> Job:
    """
    Creates a Predict request and returns the job object.

    This function makes a post request to the specified endpoint with the payload.
    Either 'model_ids' or 'train_job_id' should be provided but not both.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    endpoint : str
        The endpoint to which the post request is to be made.
        either predict or predict/single_site
    payload : dict
        The payload to be sent in the post request.
    model_ids : List[str], optional
        The list of model ids to be used for Predict. Default is None.
    train_job_id : str, optional
        The id of the train job to be used for Predict. Default is None.

    Returns
    -------
    Job
        The job object representing the Predict job.

    Raises
    ------
    InvalidParameterError
        If neither 'model_ids' nor 'train_job_id' is provided.
        If both 'model_ids' and 'train_job_id' are provided.
    HTTPError
        If the post request does not succeed.
    ValidationError
        If the response cannot be parsed into a 'Job' object.
    """

    if model_ids is None and train_job_id is None:
        raise InvalidParameterError(
            "Either a list of model IDs or a train job ID must be provided"
        )

    if model_ids is not None and train_job_id is not None:
        raise InvalidParameterError(
            "Only a list of model IDs OR a train job ID must be provided, not both"
        )

    if model_ids is not None:
        payload["model_id"] = model_ids
    else:
        payload["train_job_id"] = train_job_id

    response = session.post(endpoint, json=payload)
    return pydantic.parse_obj_as(Job, response.json())


def create_predict_job(
    session: APISession,
    sequences: SequenceDataset,
    train_job: TrainFuture,
    model_ids: Optional[List[str]] = None,
) -> Job:
    """
    Creates a predict job with a given set of sequences and a train job.

    This function will use the sequences and train job ID to create a new Predict job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    sequences : SequenceDataset
        The dataset containing the sequences to predict
    train_job : TrainFuture
        The Train job: this model will be used for making Predicts.
    model_ids: List[str]
        specific IDs for models

    Returns
    -------
    Job
        The job object representing the created Predict job.

    Raises
    ------
    InvalidParameterError
        If neither 'model_ids' nor 'train_job' is provided.
    InvalidParameterError
        If BOTH `model_ids` and `train_job` is provided
    HTTPError
        If the post request does not succeed.
    ValidationError
        If the response cannot be parsed into a 'Job' object.
    """
    if isinstance(model_ids, str):
        model_ids = [model_ids]
    endpoint = "v1/workflow/predict"
    payload = {"sequences": sequences.sequences}
    return _create_predict_job(
        session, endpoint, payload, model_ids=model_ids, train_job_id=train_job.id
    )


def create_predict_single_site(
    session: APISession,
    sequence: SequenceData,
    train_job: TrainFuture,
    model_ids: Optional[List[str]] = None,
) -> Job:
    """
    Creates a predict job for single site mutants with a given sequence and a train job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    sequence : SequenceData
        The sequence for which single site mutants predictions will be made.
    train_job : TrainFuture
        The train job whose model will be used for making Predicts.
    model_ids: List[str]
        specific IDs for models

    Returns
    -------
    Job
        The job object representing the created Predict job.

    Raises
    ------
    InvalidParameterError
        If neither 'model_ids' nor 'train_job' is provided.
    InvalidParameterError
        If BOTH `model_ids` and `train_job` is provided
    HTTPError
        If the post request does not succeed.
    ValidationError
        If the response cannot be parsed into a 'Job' object.
    """
    endpoint = "v1/workflow/predict/single_site"
    payload = {"sequence": sequence.sequence}
    return _create_predict_job(
        session, endpoint, payload, model_ids=model_ids, train_job_id=train_job.id
    )


def get_prediction_results(
    session: APISession,
    job_id: str,
    page_size: Optional[int] = None,
    page_offset: Optional[int] = None,
) -> PredictJob:
    """
    Retrieves the results of a Predict job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    job_id : str
        The ID of the job whose results are to be retrieved.
    page_size : Optional[int], default is None
        The number of results to be returned per page. If None, all results are returned.
    page_offset : Optional[int], default is None
        The number of results to skip. If None, defaults to 0.

    Returns
    -------
    PredictJob
        The job object representing the Predict job.

    Raises
    ------
    HTTPError
        If the GET request does not succeed.
    """
    endpoint = f"v1/workflow/predict/{job_id}"
    params = {}
    if page_size is not None:
        params["page_size"] = page_size
    if page_offset is not None:
        params["page_offset"] = page_offset

    response = session.get(endpoint, params=params)

    return PredictJob(**response.json())


def get_single_site_prediction_results(
    session: APISession,
    job_id: str,
    page_size: Optional[int] = None,
    page_offset: Optional[int] = None,
) -> PredictSingleSiteJob:
    """
    Retrieves the results of a single site Predict job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    job_id : str
        The ID of the job whose results are to be retrieved.
    page_size : Optional[int], default is None
        The number of results to be returned per page. If None, all results are returned.
    page_offset : Optional[int], default is None
        The number of results to skip. If None, defaults to 0.

    Returns
    -------
    PredictSingleSiteJob
        The job object representing the single site Predict job.

    Raises
    ------
    HTTPError
        If the GET request does not succeed.
    """
    endpoint = f"v1/workflow/predict/single_site/{job_id}"
    params = {}
    if page_size is not None:
        params["page_size"] = page_size
    if page_offset is not None:
        params["page_offset"] = page_offset

    response = session.get(endpoint, params=params)

    return PredictSingleSiteJob(**response.json())


class PredictFutureMixin:
    """
    Class to to retrieve results from a Predict job.

    Attributes
    ----------
    session : APISession
        APIsession with auth
    job : Job
        The job object that represents the current Predict job.

    Methods
    -------
    get_results(page_size: Optional[int] = None, page_offset: Optional[int] = None) -> Union[PredictSingleSiteJob, PredictJob]
        Retrieves results from a Predict job.
    """

    session: APISession
    job: Job

    def get_results(
        self, page_size: Optional[int] = None, page_offset: Optional[int] = None
    ) -> Union[PredictSingleSiteJob, PredictJob]:
        """
        Retrieves results from a Predict job.

        it uses the appropriate method to retrieve the results based on job_type.

        Parameters
        ----------
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        Union[PredictSingleSiteJob, PredictJob]
            The job object representing the Predict job. The exact type of job depends on the job type.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        if "single_site" in self.job.job_type:
            return get_single_site_prediction_results(
                self.session, self.id, page_size, page_offset
            )
        else:
            return get_prediction_results(self.session, self.id, page_size, page_offset)


class PredictFuture(PredictFutureMixin, AsyncJobFuture):
    """Future Job for manipulating results"""
    def __init__(self, session: APISession, job: Job, page_size=1000):
        super().__init__(session, job)
        self.page_size = page_size

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def get(self, verbose: bool = False) -> Union[PredictSingleSiteJob, PredictJob]:
        """
        Get all the results of the predict job.

        Args:
            verbose (bool, optional): If True, print verbose output. Defaults False.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            PredictJob: A list of predict objects representing the results.
        """
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                response = self.get_results(page_offset=offset, page_size=step)
                results += response.result
                num_returned = len(response.result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return results
        return results


class PredictAPI:
    """API interface for calling Predict endpoints"""

    def __init__(self, session: APISession):
        """
        Initialize a new instance of the PredictAPI class.

        Parameters
        ----------
        session : APISession
            APIsession with auth
        """
        self.session = session

    def create_predict_job(
        self,
        sequences: List,
        train_job: TrainFuture,
        model_ids: Optional[List[str]] = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for a given list of sequences and a trained model.

        Parameters
        ----------
        sequences : List
            The list of sequences to be used for the Predict job.
        train_job : TrainFuture
            The train job object representing the trained model.
        model_ids : List[str], optional
            The list of model ids to be used for Predict. Default is None.

        Returns
        -------
        PredictFuture
            The job object representing the Predict job.

        Raises
        ------
        InvalidParameterError
            If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
        InvalidParameterError
            If BOTH train_job and model_ids are specified
        InvalidParameterError
            If NEITHER train_job or model_ids is specified
        APIError
            If the backend refuses the job (due to sequence length or invalid inputs)
        """
        if train_job.assaymetadata is not None:
            if train_job.assaymetadata.sequence_length is not None:
                if any(
                    [
                        train_job.assaymetadata.sequence_length != len(s)
                        for s in sequences
                    ]
                ):
                    raise InvalidParameterError(
                        f"Predict sequences length {len(sequences[0])}  != training assaydata ({train_job.assaymetadata.sequence_length})"
                    )
        if not train_job.done():
            raise InvalidParameterError(
                f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            )

        sequence_dataset = SequenceDataset(sequences=sequences)
        job = create_predict_job(self.session, sequence_dataset, train_job, model_ids=model_ids)
        return PredictFuture(self.session, job)

    def create_predict_single_site(
        self,
        sequence: str,
        train_job: TrainFuture,
        model_ids: Optional[List[str]] = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for single site mutation analysis with a trained model.

        Parameters
        ----------
        sequence : str
            The sequence for single site analysis.
        train_job : TrainFuture
            The train job object representing the trained model.
        model_ids : List[str], optional
            The list of model ids to be used for Predict. Default is None.

        Returns
        -------
        PredictFuture
            The job object representing the Predict job.

        Raises
        ------
        InvalidParameterError
            If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
        InvalidParameterError
            If BOTH train_job and model_ids are specified
        InvalidParameterError
            If NEITHER train_job or model_ids is specified
        APIError
            If the backend refuses the job (due to sequence length or invalid inputs)
        """
        if train_job.assaymetadata is not None:
            if train_job.assaymetadata.sequence_length is not None:
                if any([train_job.assaymetadata.sequence_length != len(sequence)]):
                    raise InvalidParameterError(
                        f"Predict sequences length {len(sequence)}  != training assaydata ({train_job.assaymetadata.sequence_length})"
                    )
        train_job.refresh()
        if not train_job.done():
            raise InvalidParameterError(
                f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            )

        sequence_dataset = SequenceData(sequence=sequence)
        job = create_predict_single_site(self.session, sequence_dataset, train_job, model_ids=model_ids)
        return PredictFuture(self.session, job)

    def get_prediction_results(
        self,
        job_id: str,
        page_size: Optional[int] = None,
        page_offset: Optional[int] = None,
    ) -> PredictJob:
        """
        Retrieves the results of a Predict job.

        Parameters
        ----------
        job_id : str
            The ID of the Predict job.
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        PredictJob
            The job object representing the Predict job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        job_details = get_prediction_results(
            self.session, job_id, page_size, page_offset
        )
        return PredictFuture(self.session, job_details)

    def get_single_site_prediction_results(
        self,
        job_id: str,
        page_size: Optional[int] = None,
        page_offset: Optional[int] = None,
    ) -> PredictSingleSiteJob:
        """
        Retrieves the results of a single site Predict job.

        Parameters
        ----------
        job_id : str
            The ID of the Predict job.
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The page number to start retrieving results from. If None, defaults to 0.

        Returns
        -------
        PredictSingleSiteJob
            The job object representing the single site Predict job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        job_details = get_single_site_prediction_results(
            self.session, job_id, page_size, page_offset
        )
        return PredictFuture(self.session, job_details)

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
        if job_details.job_type not in [JobType.predict, JobType.predict_single_site]:
            raise InvalidJob(
                f"Job {job_id} is not of type {JobType.predict} or {JobType.predict_single_site}"
            )
        return PredictFuture(self.session, job_details)
