from openprotein.api import assaydata
from openprotein.api import job as job_api
from openprotein.api import train
from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.schemas import (
    WorkflowCVItem,
    WorkflowCVJob,
    WorkflowTrainJob,
    WorkflowTrainStep,
)

from .assaydata import AssayDataset
from .futures import Future, PagedFuture
from .predict import PredictFuture


class TrainFuture(Future):
    """
    This class provides functionality for retrieving the
    results of a training job and initiating cross-validation jobs.

    Attributes
    ----------
    session : APISession
        The session object to use for API communication.
    job : Job
        The Job object for this training job.

    Methods
    -------
    get_results() -> TrainGraph:
        Returns the results of the training job.
    crossvalidate() -> CVFuture:
        Submits a cross-validation job and returns the future.
    """

    job: WorkflowTrainJob

    def __init__(
        self,
        session: APISession,
        job: WorkflowTrainJob,
        traingraph: list[WorkflowTrainStep] | None = None,
        assay: AssayDataset | None = None,
    ):
        # local import for cyclic dependency on app services
        from ..services.predict import PredictService

        super().__init__(session, job)
        self._predict = PredictService(session)
        self._traingraph = traingraph
        self._assay = assay
        self._args = None

    @property
    def id(self):
        return self.job.job_id

    @property
    def args(self) -> dict:
        if self._args is None:
            self._args = job_api.job_args_get(session=self.session, job_id=self.id)
        return self._args

    @property
    def assay(self):
        if self._assay is None:
            assay_id: str | None = self.args.get("assay_id")
            if assay_id is None:
                raise InvalidTrainArgs(
                    "'assay_id' not in train args. Something went wrong."
                )
            self._assay = AssayDataset(
                session=self.session,
                metadata=assaydata.get_assay_metadata(
                    session=self.session, assay_id=assay_id
                ),
            )
        return self._assay

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    def _fmt_results(self, results: list[WorkflowTrainStep] | None) -> dict:
        train_dict = {}
        if results is not None:
            tags = set([i.tag for i in results])
            for tag in tags:
                train_dict[tag] = [
                    i.loss for i in results if i.model_dump()["tag"] == tag
                ]
        return train_dict

    def get(self, verbose: bool = False) -> dict:
        """
        Gets the results of the training job.

        Returns
        -------
        TrainGraph
            The results of the training job.
        """
        try:
            if self._traingraph is None:
                self._traingraph = train.get_training_results(
                    session=self.session, job_id=self.id
                ).traingraph
            results = self._traingraph
        except APIError as exc:
            if verbose:
                print(f"Failed to get results: {exc}")
            raise exc
        return self._fmt_results(results)

    def crossvalidate(self):
        """
        Submits a cross-validation job.

        If a cross-validation job has already been created, it returns that job.
        Otherwise, it creates a new cross-validation job and returns it.

        Returns
        -------
        CVFuture
            The cross-validation job associated with this training job.
        """
        cv_future = CVFuture.create(
            session=self.session,
            job=train.crossvalidate(
                session=self.session,
                train_job_id=self.id,
            ),
            train_job_id=self.id,
        )
        self.crossvalidation = cv_future
        return cv_future

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
        return train.list_models(self.session, self.job.job_id)

    def predict(
        self, sequences: list[str], model_ids: list[str] | None = None
    ) -> PredictFuture:
        """
        Creates a predict job based on the training job.

        Parameters
        ----------
        sequences : List[str]
            The list of sequences to be used for the Predict job.
        model_ids : List[str], optional
            The list of model ids to be used for Predict. Default is None.

        Returns
        -------
        PredictFuture
            The job object representing the Predict job.
        """
        return self._predict.create_predict_job(
            sequences=sequences, train_job=self, model_ids=model_ids
        )

    def predict_single_site(
        self,
        sequence: str,
        model_ids: list[str] | None = None,
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

        Creates a predict job based on the training job
        """
        return self._predict.create_predict_single_site(
            sequence=sequence, train_job=self, model_ids=model_ids
        )


class InvalidTrainArgs(Exception): ...


class CVFuture(PagedFuture, Future):
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

    """

    job: WorkflowCVJob

    def __init__(
        self,
        session: APISession,
        job: WorkflowCVJob,
        train_job_id: str | None = None,
        page_size: int = 1000,
    ):
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
        super().__init__(session=session, job=job, page_size=page_size)
        if train_job_id is None:
            assert (
                job.prerequisite_job_id is not None
            ), "expected prerequisite train job id"
            train_job_id = job.prerequisite_job_id
        self.train_job_id = train_job_id

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def _fmt_results(self, results: WorkflowCVJob) -> list[WorkflowCVItem]:
        return results.result if results.result is not None else []

    def get_slice(self, start: int, end: int):
        results = self.get_crossvalidation(page_size=end - start, page_offset=start)
        return self._fmt_results(results)

    def get_crossvalidation(
        self, page_size: int | None = None, page_offset: int | None = None
    ) -> WorkflowCVJob:
        """
        Retrieves the results of the cross-validation job.


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

        """
        return train.get_crossvalidation(
            session=self.session,
            job_id=self.job.job_id,
            page_size=page_size,
            page_offset=page_offset,
        )
