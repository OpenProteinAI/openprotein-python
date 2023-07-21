from typing import Optional, List, Union
import pydantic

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture

from openprotein.models import (TrainGraph, JobType, Job,Jobplus)
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
    response = session.get(endpoint, params={"job_id":job_id})
    return response.json()

def _train_job(session: APISession,
                     endpoint:str,
                     assaydataset: AssayDataset,
                     measurement_name: Union[str, List[str]],
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False) -> Jobplus:
    """
    Create a training job.

    This function validates the inputs, formats the data, sends the job training request to the endpoint,
    and then parses the response into a `Job` object.

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

    for mm in measurement_name:
        if mm not in assaydataset.measurement_names:
            raise InvalidParameterError(f"No {mm} in measurement names")
    if assaydataset.shape[0] <3:
        raise InvalidParameterError("Assaydata must have at least 3 data points for training")
    if model_name is None:
        model_name = ""

    data = {
        "assay_id": assaydataset.id,
        "measurement_name": measurement_name, 
        "model_name": model_name
    }
    params = {"force_preprocess": str(force_preprocess).lower()}

    response = session.post(endpoint, params=params, json=data)
    response.raise_for_status()
    return pydantic.parse_obj_as(Jobplus, response.json())

def create_train_job(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: Union[str, List[str]],
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    """
    Create a training job.

    This function validates the inputs, formats the data, sends the job training request to the endpoint,
    and then parses the response into a `Job` object.

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
    endpoint = 'v1/workflow/train'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def _create_train_job_br(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: Union[str, List[str]],
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    endpoint = 'v1/workflow/train/br'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def _create_train_job_gp(session: APISession,
                     assaydataset: AssayDataset,
                     measurement_name: Union[str, List[str]],
                     model_name: str = "",
                     force_preprocess: Optional[bool] = False):
    endpoint = 'v1/workflow/train/gp'
    return _train_job(session, endpoint, assaydataset, measurement_name, model_name, force_preprocess)


def get_training_results(session: APISession, job_id: str) -> TrainGraph:
    """Get Training results (e.g. loss etc) of job."""
    endpoint = f'v1/workflow/train/{job_id}'
    response = session.get(endpoint)
    return TrainGraph( ** response.json() )

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
    endpoint = f'v1/workflow/train/job/{job_id}'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(Jobplus, response.json())

class TrainFutureMixin:
    session: APISession
    job: Job

    def get_results(self) -> TrainGraph:
        return get_training_results(self.session, self.job.job_id)

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
    def __init__(self, session: APISession, job: Job, assaymetadata: Optional[AssayMetadata] = None):
        super().__init__(session, job)
        self.assaymetadata = assaymetadata

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
        return super().get_assay_data()

    def get(self, verbose:bool=False) -> TrainGraph:

        try:
            results = self.get_results()
        except APIError as exc:
            if verbose:
                print(f"Failed to get results: {exc}")
            raise exc
        return results



class TrainingAPI:
    """ API interface for calling Train endpoints"""

    def __init__(self, session: APISession, ):
        self.session = session
        self.assay= None
        

    def create_training_job(self,
                    assaydataset: AssayDataset,
                    measurement_name: Union[str, List[str]],
                    model_name:str ="",
                    force_preprocess: Optional[bool]=False) -> TrainFuture:
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
            If set to True, preprocessing is forced even if preprocessed data already exists.

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
        job_details = create_train_job(self.session,
                                       assaydataset,
                                       measurement_name,
                                       model_name,
                                       force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def _create_training_job_br(self,
                    assaydataset: AssayDataset,
                    measurement_name: Union[str, List[str]],
                    model_name:str="",
                    force_preprocess: Optional[bool]=False) -> TrainFuture:
        """Same as create_training_job."""
        job_details = _create_train_job_br(self.session,
                                           assaydataset,
                                           measurement_name,
                                           model_name,
                                           force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def _create_training_job_gp(self,
                    assaydataset: AssayDataset,
                    measurement_name: Union[str, List[str]],
                    model_name:str="",
                    force_preprocess: Optional[bool]=False) -> TrainFuture:
        """Same as create_training_job."""
        job_details = _create_train_job_gp(self.session,
                                           assaydataset,
                                           measurement_name,
                                           model_name,
                                           force_preprocess)
        return TrainFuture(self.session, job_details, assaydataset)

    def get_training_results(self, job_id: str) -> TrainFuture:
        """
        Get training results (e.g. loss etc).

        Parameters
        ----------
        assaydataset : str
            job_id to get


        Returns
        -------
        TrainFuture
            A TrainFuture Job 
        """
        job_details = get_training_results(self.session, job_id)
        return TrainFuture(self.session, job_details)

    def load_job(self, job_id:str) -> Jobplus:
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
        #assay_metadata = get_assay_metadata(self.session, assay_id)

        if job_details.job_type != JobType.train:
            raise InvalidJob(f"Job {job_id} is not of type {JobType.train}")
        return TrainFuture(self.session, job_details, assay_metadata)
        
    