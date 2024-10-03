from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import WorkflowPredictJob, WorkflowPredictSingleSiteJob


def _create_predict_job(
    session: APISession,
    endpoint: str,
    payload: dict,
    model_ids: list[str] | None = None,
    train_job_id: str | None = None,
) -> WorkflowPredictJob:
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
    PredictJob
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
    return WorkflowPredictJob.model_validate(response.json())


def create_predict_job(
    session: APISession,
    sequences: list[str],
    train_job_id: str | None = None,
    model_ids: list[str] | None = None,
) -> WorkflowPredictJob:
    """
    Creates a predict job with a given set of sequences and a train job.

    This function will use the sequences and train job ID to create a new Predict job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    sequences : SequenceDataset
        The dataset containing the sequences to predict
    train_job : Any
        The Train job: this model will be used for making Predicts.
    model_ids: List[str]
        specific IDs for models

    Returns
    -------
    PredictJob
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
    payload = {"sequences": sequences}
    return _create_predict_job(
        session, endpoint, payload, model_ids=model_ids, train_job_id=train_job_id
    )


def create_predict_single_site(
    session: APISession,
    sequence: str,
    train_job_id: str | None = None,
    model_ids: list[str] | None = None,
) -> WorkflowPredictJob:
    """
    Creates a predict job for single site mutants with a given sequence and a train job.

    Parameters
    ----------
    session : APISession
        APIsession with auth
    sequence : SequenceData
        The sequence for which single site mutants predictions will be made.
    train_job : Any
        The train job whose model will be used for making Predicts.
    model_ids: List[str]
        specific IDs for models

    Returns
    -------
    PredictJob
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
    payload = {"sequence": sequence}
    return _create_predict_job(
        session, endpoint, payload, model_ids=model_ids, train_job_id=train_job_id
    )


def get_prediction_results(
    session: APISession,
    job_id: str,
    page_size: int | None = None,
    page_offset: int | None = None,
) -> WorkflowPredictJob:
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
    # get results to assemble into list
    return WorkflowPredictJob.model_validate(response.json())


def get_single_site_prediction_results(
    session: APISession,
    job_id: str,
    page_size: int | None = None,
    page_offset: int | None = None,
) -> WorkflowPredictSingleSiteJob:
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
    # get results to assemble into list
    return WorkflowPredictSingleSiteJob.model_validate(response)
