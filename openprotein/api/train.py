from openprotein.base import APISession
from openprotein.errors import InvalidJob
from openprotein.schemas import Job, WorkflowCVJob, WorkflowTrainJob


def list_models(session: APISession, job_id: str) -> list:
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
    return Job.model_validate(response.json())


def get_crossvalidation(
    session: APISession,
    job_id: str,
    page_size: int | None = None,
    page_offset: int | None = 0,
) -> WorkflowCVJob:
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
    return WorkflowCVJob.model_validate(response.json())


def _train_job(
    session: APISession,
    endpoint: str,
    assay_id: str,
    measurement_name: str | list[str],
    model_name: str = "",
    force_preprocess: bool = False,
) -> Job:
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

    data = {
        "assay_id": assay_id,
        "measurement_name": measurement_name,
        "model_name": model_name,
    }
    params = {"force_preprocess": str(force_preprocess).lower()}

    response = session.post(endpoint, params=params, json=data)
    return Job.model_validate(response.json())


def create_train_job(
    session: APISession,
    assay_id: str,
    measurement_name: str | list[str],
    model_name: str = "",
    force_preprocess: bool = False,
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
        session=session,
        endpoint=endpoint,
        assay_id=assay_id,
        measurement_name=measurement_name,
        model_name=model_name,
        force_preprocess=force_preprocess,
    )


def _create_train_job_br(
    session: APISession,
    assay_id: str,
    measurement_name: str | list[str],
    model_name: str = "",
    force_preprocess: bool = False,
):
    """Alias for create_train_job"""
    endpoint = "v1/workflow/train/br"
    return _train_job(
        session=session,
        endpoint=endpoint,
        assay_id=assay_id,
        measurement_name=measurement_name,
        model_name=model_name,
        force_preprocess=force_preprocess,
    )


def _create_train_job_gp(
    session: APISession,
    assay_id: str,
    measurement_name: str | list[str],
    model_name: str = "",
    force_preprocess: bool = False,
):
    """Alias for create_train_job"""
    endpoint = "v1/workflow/train/gp"
    return _train_job(
        session=session,
        endpoint=endpoint,
        assay_id=assay_id,
        measurement_name=measurement_name,
        model_name=model_name,
        force_preprocess=force_preprocess,
    )


def get_training_results(session: APISession, job_id: str) -> WorkflowTrainJob:
    """Get Training results (e.g. loss etc) of job."""
    endpoint = f"v1/workflow/train/{job_id}"
    response = session.get(endpoint)
    return WorkflowTrainJob.model_validate(response.json())
