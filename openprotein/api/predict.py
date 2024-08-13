from typing import Optional, List, Union, Any, Dict, Literal
from openprotein.pydantic import BaseModel, root_validator

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture
from openprotein.jobs import ResultsParser, Job, register_job_type, JobType, JobStatus
from openprotein.errors import InvalidParameterError, APIError
from openprotein.futures import FutureFactory, FutureBase


class SequenceData(BaseModel):
    sequence: str


class SequenceDataset(BaseModel):
    sequences: List[str]


class _Prediction(BaseModel):
    """Prediction details."""

    @root_validator(pre=True)
    def extract_pred(cls, values):
        p = values.pop("properties")
        name = list(p.keys())[0]
        ymu = p[name]["y_mu"]
        yvar = p[name]["y_var"]
        p["name"] = name
        p["y_mu"] = ymu
        p["y_var"] = yvar

        values.update(p)
        return values

    model_id: str
    model_name: str
    y_mu: Optional[float] = None
    y_var: Optional[float] = None
    name: Optional[str]


class Prediction(BaseModel):
    """Prediction details."""

    model_id: str
    model_name: str
    properties: Dict[str, Dict[str, float]]


class PredictJobBase(Job):
    # might be none if just fetching
    job_id: Optional[str] = None
    job_type: str
    status: JobStatus


@register_job_type(JobType.workflow_predict)
class PredictJob(PredictJobBase):
    """Properties about predict job returned via API."""

    @root_validator(pre=True)
    def extract_pred(cls, values):
        # Extracting 'predictions' and 'sequences' from the input values
        v = values.pop("result")
        preds = [i["predictions"] for i in v]
        seqs = [i["sequence"] for i in v]
        values["result"] = [
            {"sequence": i, "predictions": p} for i, p in zip(seqs, preds)
        ]
        return values

    class SequencePrediction(BaseModel):
        """Sequence prediction."""

        sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None
    job_type: str


@register_job_type(JobType.worflow_predict_single_site)
class PredictSingleSiteJob(PredictJobBase):
    """Properties about single-site prediction job returned via API."""

    class SequencePrediction(BaseModel):
        """Sequence prediction."""

        position: int
        amino_acid: str
        # sequence: str
        predictions: List[Prediction] = []

    result: Optional[List[SequencePrediction]] = None
    job_type: Literal[JobType.worflow_predict_single_site] = (
        JobType.worflow_predict_single_site
    )


def _create_predict_job(
    session: APISession,
    endpoint: str,
    payload: dict,
    model_ids: Optional[List[str]] = None,
    train_job_id: Optional[str] = None,
) -> FutureBase:
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
    return FutureFactory.create_future(session=session, response=response)


def create_predict_job(
    session: APISession,
    sequences: SequenceDataset,
    train_job: Optional[Any] = None,
    model_ids: Optional[List[str]] = None,
) -> FutureBase:
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
    payload = {"sequences": sequences.sequences}
    train_job_id = train_job.id if train_job is not None else None
    return _create_predict_job(
        session, endpoint, payload, model_ids=model_ids, train_job_id=train_job_id
    )


def create_predict_single_site(
    session: APISession,
    sequence: SequenceData,
    train_job: Any,
    model_ids: Optional[List[str]] = None,
) -> FutureBase:
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
    # get results to assemble into list
    return ResultsParser.parse_obj(response.json())


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
    # get results to assemble into list
    return ResultsParser.parse_obj(response)


class PredictFutureMixin:
    """
    Class to to retrieve results from a Predict job.

    Attributes
    ----------
    session : APISession
        APIsession with auth
    job : PredictJob
        The job object that represents the current Predict job.

    Methods
    -------
    get_results(page_size: Optional[int] = None, page_offset: Optional[int] = None) -> Union[PredictSingleSiteJob, PredictJob]
        Retrieves results from a Predict job.
    """

    session: APISession
    job: PredictJob
    id: Optional[str] = None

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
        assert self.id is not None
        if "single_site" in self.job.job_type:
            return get_single_site_prediction_results(
                self.session, self.id, page_size, page_offset
            )
        else:
            return get_prediction_results(self.session, self.id, page_size, page_offset)


class PredictFuture(PredictFutureMixin, AsyncJobFuture, FutureBase):  # type: ignore
    """Future Job for manipulating results"""

    job_type = [JobType.workflow_predict, JobType.worflow_predict_single_site]

    def __init__(self, session: APISession, job: PredictJob, page_size=1000):
        super().__init__(session, job)
        self.page_size = page_size

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def _fmt_results(self, results):
        properties = set(
            list(i["properties"].keys())[0] for i in results[0].dict()["predictions"]
        )
        dict_results = {}
        for p in properties:
            dict_results[p] = {}
            for i, r in enumerate(results):
                s = r.sequence
                props = [i.properties[p] for i in r.predictions if p in i.properties][0]
                dict_results[p][s] = {"mean": props["y_mu"], "variance": props["y_var"]}
        dict_results
        return dict_results

    def _fmt_ssp_results(self, results):
        properties = set(
            list(i["properties"].keys())[0] for i in results[0].dict()["predictions"]
        )
        dict_results = {}
        for p in properties:
            dict_results[p] = {}
            for i, r in enumerate(results):
                s = s = f"{r.position+1}{r.amino_acid}"
                props = [i.properties[p] for i in r.predictions if p in i.properties][0]
                dict_results[p][s] = {"mean": props["y_mu"], "variance": props["y_var"]}
        return dict_results

    def get(self, verbose: bool = False) -> Dict:
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

        results: List = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                response = self.get_results(page_offset=offset, page_size=step)
                assert isinstance(response.result, list)
                results += response.result
                num_returned = len(response.result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")

        if self.job.job_type == JobType.workflow_predict:
            return self._fmt_results(results)
        else:
            return self._fmt_ssp_results(results)


class PredictService:
    """interface for calling Predict endpoints"""

    def __init__(self, session: APISession):
        """
        Initialize a new instance of the PredictService class.

        Parameters
        ----------
        session : APISession
            APIsession with auth
        """
        self.session = session

    def create_predict_job(
        self,
        sequences: List,
        train_job: Optional[Any] = None,
        model_ids: Optional[List[str]] = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for a given list of sequences and a trained model.

        Parameters
        ----------
        sequences : List
            The list of sequences to be used for the Predict job.
        train_job : Any
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
        if train_job is not None:
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
            print(f"WARNING: training job has status {train_job.status}")
            # raise InvalidParameterError(
            #    f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            # )

        sequence_dataset = SequenceDataset(sequences=sequences)
        return create_predict_job(
            self.session, sequence_dataset, train_job, model_ids=model_ids  # type: ignore
        )

    def create_predict_single_site(
        self,
        sequence: str,
        train_job: Any,
        model_ids: Optional[List[str]] = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for single site mutation analysis with a trained model.

        Parameters
        ----------
        sequence : str
            The sequence for single site analysis.
        train_job : Any
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
            print(f"WARNING: training job has status {train_job.status}")
            # raise InvalidParameterError(
            #    f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            # )

        sequence_dataset = SequenceData(sequence=sequence)
        return create_predict_single_site(
            self.session, sequence_dataset, train_job, model_ids=model_ids  # type: ignore
        )
