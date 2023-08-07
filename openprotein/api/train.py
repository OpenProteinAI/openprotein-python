from typing import Optional, List, Union
import pydantic
from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture, Job

from openprotein.models import TrainGraph, JobType, Jobplus, CVResults
from openprotein.errors import InvalidParameterError, APIError, InvalidJob
from openprotein.api.data import AssayDataset, AssayMetadata


def list_models(session: APISession, job_id: str) -> List:
    """
    List models assoicated with job

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID

    Returns
    -------
    List
        List of models
    """
    endpoint = "v1/models"
    response = session.get(endpoint, params={"job_id": job_id})
    return response.json()


def crossvalidate(session: APISession, train_job_id: str, n_splits: int = 5) -> Job:
    """
    Submit a cross-validation job.

    Args:
        session (APISession): auth session
        job_id (str): job id
        n_splits (int, optional): N of CV splits. Defaults to 5.

    Returns:
        Job:
    """
    endpoint = "v1/workflow/crossvalidate"
    response = session.post(
        endpoint, json={"train_job_id": train_job_id, "n_splits": n_splits}
    )
    return pydantic.parse_obj_as(Job, response.json())


def get_crossvalidation(
    session: APISession,
    job_id: str,
    page_size: Optional[int] = None,
    page_offset: Optional[int] = 0,
) -> CVResults:
    """
    Get CV results

    Args:
        session (APISession): auth'd session
        job_id (str): Job id

    Returns:
        _type_: _description_
    """
    endpoint = f"v1/workflow/crossvalidate/{job_id}"
    params = {"page_size": page_size, "page_offset": page_offset}
    response = session.get(endpoint, params=params)
    if response.status_code == 404:
        raise InvalidJob("No CV job has been submitted for this job!")
    return pydantic.parse_obj_as(CVResults, response.json())


def _train_job(
    session: APISession,
    endpoint: str,
    assaydataset: AssayDataset,
    measurement_name: Union[str, List[str]],
    model_name: str = "",
    force_preprocess: Optional[bool] = False,
) -> Jobplus:
    """
    Create a training job.

    Validate inputs, format  data, sends the job training request to the endpoint,

    Parses the response into a `Job` object.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    endpoint : str
        The endpoint to which the job training request is to be sent.
    assaydataset : AssayDataset
        An AssayDataset object from which the assay_id is extracted.
    measurement_name : str or List[str]
        The name(s) of the measurement(s) to be used in the training job.
    model_name : str, optional
        The name to give the model.
    force_preprocess : bool, optional
        If set to True, preprocessing is forced even if preprocessed data already exists.

    Returns
    -------
    Job
        A Job

    Raises
    ------
    InvalidParameterError
        If the `assaydataset` is not an AssayDataset object,
        If any measurement name provided does not exist in the AssayDataset,
        or if the AssayDataset has fewer than 3 data points.
    HTTPError
        If the request to the server fails.
    """
    if not isinstance(assaydataset, AssayDataset):
        raise InvalidParameterError("assaydataset should be an assaydata Job result")
    if isinstance(measurement_name, str):
        measurement_name = [measurement_name]

    for measurement in measurement_name:
        if measurement not in assaydataset.measurement_names:
            raise InvalidParameterError(f"No {measurement} in measurement names")
    if assaydataset.shape[0] < 3:
        raise InvalidParameterError("Assaydata must have >=3 data points for training!")
    if model_name is None:
        model_name = ""

    data = {
        "assay_id": assaydataset.id,
        "measurement_name": measurement_name,
        "model_name": model_name,
    }
    params = {"force_preprocess": str(force_preprocess).lower()}

    response = session.post(endpoint, params=params, json=data)
    response.raise_for_status()
    return pydantic.parse_obj_as(Jobplus, response.json())


def create_train_job(
    session: APISession,
    assaydataset: AssayDataset,
    measurement_name: Union[str, List[str]],
    model_name: str = "",
    force_preprocess: Optional[bool] = False,
):
    """
    Create a training job.

    Validate inputs, format  data, sends the job training request to the endpoint,

    Parses the response into a `Job` object.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    endpoint : str
        The endpoint to which the job training request is to be sent.
    assaydataset : AssayDataset
        An AssayDataset object from which the assay_id is extracted.
    measurement_name : str or List[str]
        The name(s) of the measurement(s) to be used in the training job.
    model_name : str, optional
        The name to give the model.
    force_preprocess : bool, optional
        If set to True, preprocessing is forced even if preprocessed data already exists.

    Returns
    -------
    Job
        A Job

    Raises
    ------
    InvalidParameterError
        If the `assaydataset` is not an AssayDataset object,
        If any measurement name provided does not exist in the AssayDataset,
        or if the AssayDataset has fewer than 3 data points.
    HTTPError
        If the request to the server fails.
    """
    endpoint = "v1/workflow/train"
    return _train_job(
        session, endpoint, assaydataset, measurement_name, model_name, force_preprocess
    )


def _create_train_job_br(
    session: APISession,
    assaydataset: AssayDataset,
    measurement_name: Union[str, List[str]],
    model_name: str = "",
    force_preprocess: Optional[bool] = False,
):
    """Alias for create_train_job"""
    endpoint = "v1/workflow/train/br"
    return _train_job(
        session, endpoint, assaydataset, measurement_name, model_name, force_preprocess
    )


def _create_train_job_gp(
    session: APISession,
    assaydataset: AssayDataset,
    measurement_name: Union[str, List[str]],
    model_name: str = "",
    force_preprocess: Optional[bool] = False,
):
    """Alias for create_train_job"""
    endpoint = "v1/workflow/train/gp"
    return _train_job(
        session, endpoint, assaydataset, measurement_name, model_name, force_preprocess
    )


def get_training_results(session: APISession, job_id: str) -> TrainGraph:
    """Get Training results (e.g. loss etc) of job."""
    endpoint = f"v1/workflow/train/{job_id}"
    response = session.get(endpoint)
    return TrainGraph(**response.json())


def load_job(session: APISession, job_id: str) -> Jobplus:
    """
    Reload a Submitted job to resume from where you left off!


    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
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

    """
    endpoint = f"v1/workflow/train/job/{job_id}"
    response = session.get(endpoint)
    return pydantic.parse_obj_as(Jobplus, response.json())


class CVFutureMixin:
    """
    A mixin class to provide cross-validation job submission and retrieval.

    Attributes
    ----------
    session : APISession
        The session object to use for API communication.
    train_job_id : str
        The id of the training job associated with this cross-validation job.
    job : Job
        The Job object for this cross-validation job.

    Methods
    -------
    crossvalidate():
        Submits a cross-validation job to the server.
    get_crossvalidation(page_size: Optional[int] = None, page_offset: Optional[int] = 0):
        Retrieves the results of the cross-validation job.
    """

    session: APISession
    train_job_id: str
    job: Job

    def crossvalidate(self):
        """
        Submit a cross-validation job to the server.

        If the job has already been submitted, an InvalidJob exception is raised.

        Returns
        -------
        Job
            The Job object for this cross-validation job.

        Raises
        ------
        InvalidJob
            If the job has already been submitted.
        """
        if self.job is None:
            self.job = crossvalidate(self.session, self.train_job_id)
            return self.job
        else:
            raise InvalidJob(
                "CV Job has already been submitted. Use get_crossvalidation() instead."
            )

    def get_crossvalidation(
        self, page_size: Optional[int] = None, page_offset: Optional[int] = 0
    ):
        """
        Retrieves the results of the cross-validation job.

        If no job has been submitted, an InvalidJob exception is raised.

        Parameters
        ----------
        page_size : int, optional
            The number of items to retrieve in a single request..
        page_offset : int, optional
            The offset to start retrieving items from. Default is 0.

        Returns
        -------
        dict
            The results of the cross-validation job.

        Raises
        ------
        InvalidJob
            If no job has been submitted.
        """
        if self.job is not None:
            return get_crossvalidation(
                self.session, self.job.job_id, page_size, page_offset
            )
        else:
            raise InvalidJob(
                "No CV Job has been submitted! Call crossvalidate() and try again."
            )


class CVFuture(CVFutureMixin, AsyncJobFuture):
    """
    This class helps initiating, submitting, and retrieving the
    results of a cross-validation job.

    Attributes
    ----------
    session : APISession
        The session object to use for API communication.
    train_job_id : str
        The id of the training job associated with this cross-validation job.
    job : Job
        The Job object for this cross-validation job.
    page_size : int
        The number of items to retrieve in a single request.

    Methods
    -------
    __str__():
        Returns a string representation of the cross-validation job.
    __repr__():
        Returns a detailed representation of the cross-validation job.
    """

    def __init__(self, session: APISession, train_job_id: str, job: Job = None):
        """
        Constructs a new CVFuture instance.

        Parameters
        ----------
        session : APISession
            The session object to use for API communication.
        train_job_id : str
            The id of the training job associated with this cross-validation job.
        job : Job, optional
            The Job object for this cross-validation job.
        """
        super().__init__(session, job)
        self.train_job_id = train_job_id
        self.page_size = 1000

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def get(self, verbose: bool = False) -> CVResults:
        """
        Get all the results of the CV job.

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
                response = self.get_crossvalidation(page_offset=offset, page_size=step)
                results += response.result
                num_returned = len(response.result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return results
        return results


class TrainFutureMixin:
    """
    This class provides functionality for retrieving the
    results of a training job and initiating cross-validation jobs.

    Attributes
    ----------
    session : APISession
        The session object to use for API communication.
    job : Job
        The Job object for this training job.
    crossvalidation : CVFuture
        The CVFuture object for the associated cross-validation job.

    Methods
    -------
    get_results() -> TrainGraph:
        Returns the results of the training job.
    crossvalidate():
        Submits a cross-validation job and returns it.
    """

    session: APISession
    job: Job
    crossvalidation: CVFuture = None

    def get_results(self) -> TrainGraph:
        """
        Gets the results of the training job.

        Returns
        -------
        TrainGraph
            The results of the training job.
        """
        return get_training_results(self.session, self.job.job_id)

    def crossvalidate(self):
        """
        Submits a cross-validation job and binds it.

        If a cross-validation job has already been created, it returns that job.
        Otherwise, it creates a new cross-validation job and returns it.

        Returns
        -------
        CVFuture
            The cross-validation job associated with this training job.
        """
        if self.crossvalidation is None:
            self.crossvalidation = CVFuture(self.session, self.job.job_id)
        return self.crossvalidation.crossvalidate()

    def get_assay_data(self):
        """
        NOT IMPLEMENTED.

        Get the assay data used for the training job.

        Returns:
            The assay data.
        """
        raise NotImplementedError("get_assay_data is not available.")

    def list_models(self):
        """
        List models assoicated with job

        Parameters
        ----------
        session : APISession
            Session object for API communication.
        job_id : str
            job ID

        Returns
        -------
        List
            List of models
        """
        return list_models(self.session, self.job.job_id)


class TrainFuture(TrainFutureMixin, AsyncJobFuture):
    """Future Job for manipulating results"""
    def __init__(
        self,
        session: APISession,
        job: Job,
        assaymetadata: Optional[AssayMetadata] = None,
    ):
        super().__init__(session, job)
        self.assaymetadata = assaymetadata
        self.crossvalidation: Optional[CVFuture] = CVFuture(session, job.job_id, None)

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def get_assay_data(self):
        """
        NOT IMPLEMENTED.

        Get the assay data used for the training job.

        Returns:
            The assay data.
        """
        return self.get_assay_data()

    def get(self, verbose: bool = False) -> TrainGraph:
        try:
            results = self.get_results()
        except APIError as exc:
            if verbose:
                print(f"Failed to get results: {exc}")
            raise exc
        return results


class TrainingAPI:
    """API interface for calling Train endpoints"""
    def __init__(
        self,
        session: APISession,
    ):
        self.session = session
        self.assay = None

    def create_training_job(
        self,
        assaydataset: AssayDataset,
        measurement_name: Union[str, List[str]],
        model_name: str = "",
        force_preprocess: Optional[bool] = False,
    ) -> TrainFuture:
        """
        Create a training job on your data.

        This function validates the inputs, formats the data, and sends the job.

        Parameters
        ----------
        assaydataset : AssayDataset
            An AssayDataset object from which the assay_id is extracted.
        measurement_name : str or List[str]
            The name(s) of the measurement(s) to be used in the training job.
        model_name : str, optional
            The name to give the model.
        force_preprocess : bool, optional
            If set to True, preprocessing is forced even if data already exists.

        Returns
        -------
        TrainFuture
            A TrainFuture Job

        Raises
        ------
        InvalidParameterError
            If the `assaydataset` is not an AssayDataset object,
            If any measurement name provided does not exist in the AssayDataset,
            or if the AssayDataset has fewer than 3 data points.
        HTTPError
            If the request to the server fails.
        """
        if isinstance(measurement_name, str):
            measurement_name = [measurement_name]
        job_details = create_train_job(
            self.session, assaydataset, measurement_name, model_name, force_preprocess
        )
        return TrainFuture(self.session, job_details, assaydataset)

    def _create_training_job_br(
        self,
        assaydataset: AssayDataset,
        measurement_name: Union[str, List[str]],
        model_name: str = "",
        force_preprocess: Optional[bool] = False,
    ) -> TrainFuture:
        """Same as create_training_job."""
        job_details = _create_train_job_br(
            self.session, assaydataset, measurement_name, model_name, force_preprocess
        )
        return TrainFuture(self.session, job_details, assaydataset)

    def _create_training_job_gp(
        self,
        assaydataset: AssayDataset,
        measurement_name: Union[str, List[str]],
        model_name: str = "",
        force_preprocess: Optional[bool] = False,
    ) -> TrainFuture:
        """Same as create_training_job."""
        job_details = _create_train_job_gp(
            self.session, assaydataset, measurement_name, model_name, force_preprocess
        )
        return TrainFuture(self.session, job_details, assaydataset)

    def get_training_results(self, job_id: str) -> TrainFuture:
        """
        Get training results (e.g. loss etc).

        Parameters
        ----------
        job_id : str
            job_id to get


        Returns
        -------
        TrainFuture
            A TrainFuture Job
        """
        job_details = get_training_results(self.session, job_id)
        return TrainFuture(self.session, job_details)

    def load_job(self, job_id: str) -> Jobplus:
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
        assay_metadata = None

        if job_details.job_type != JobType.train:
            raise InvalidJob(f"Job {job_id} is not of type {JobType.train}")
        return TrainFuture(self.session, job_details, assay_metadata)
