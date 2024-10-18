import logging

from openprotein.api import predict
from openprotein.base import APISession
from openprotein.schemas import (
    JobType,
    WorkflowPredictJob,
    WorkflowPredictSingleSiteJob,
)

from .futures import Future, PagedFuture

logger = logging.getLogger(__name__)


class PredictFuture(PagedFuture, Future):
    """Future Job for manipulating results"""

    job: WorkflowPredictJob | WorkflowPredictSingleSiteJob

    def __init__(
        self,
        session: APISession,
        job: WorkflowPredictJob | WorkflowPredictSingleSiteJob,
        page_size: int = 1000,
    ):
        super().__init__(session=session, job=job, page_size=page_size)

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    def _fmt_results(self, results: list[WorkflowPredictJob.SequencePrediction]):
        dict_results = {}
        if len(results) > 0:
            properties = set(
                list(i["properties"].keys())[0]
                for i in results[0].model_dump()["predictions"]
            )
            for p in properties:
                dict_results[p] = {}
                for r in results:
                    s = r.sequence
                    props = [
                        i.properties[p] for i in r.predictions if p in i.properties
                    ][0]
                    dict_results[p][s] = {
                        "mean": props["y_mu"],
                        "variance": props["y_var"],
                    }
        return dict_results

    def _fmt_ssp_results(
        self, results: list[WorkflowPredictSingleSiteJob.MutantPrediction]
    ):
        dict_results = {}
        if len(results) > 0:
            properties = set(
                list(i["properties"].keys())[0]
                for i in results[0].model_dump()["predictions"]
            )
            for p in properties:
                dict_results[p] = {}
                for r in results:
                    s = s = f"{r.position+1}{r.amino_acid}"
                    props = [
                        i.properties[p] for i in r.predictions if p in i.properties
                    ][0]
                    dict_results[p][s] = {
                        "mean": props["y_mu"],
                        "variance": props["y_var"],
                    }
        return dict_results

    # def get(self, verbose: bool = False) -> dict:
    #     """
    #     Get all the results of the predict job.

    #     Args:
    #         verbose (bool, optional): If True, print verbose output. Defaults False.

    #     Raises:
    #         APIError: If there is an issue with the API request.

    #     Returns:
    #         PredictJob: A list of predict objects representing the results.
    #     """
    #     step = self.page_size

    #     results = []
    #     num_returned = step
    #     offset = 0

    #     while num_returned >= step:
    #         try:
    #             response = self.get_results(page_offset=offset, page_size=step)
    #             assert isinstance(response.result, list)
    #             results += response.result
    #             num_returned = len(response.result)
    #             offset += num_returned
    #         except APIError as exc:
    #             if verbose:
    #                 print(f"Failed to get results: {exc}")

    #     if self.job.job_type == JobType.workflow_predict:
    #         return self._fmt_results(results)
    #     else:
    #         return self._fmt_ssp_results(results)

    def get_dict(self, verbose: bool = False) -> dict:

        results: list = []
        num_returned = self.page_size
        offset = 0

        while num_returned >= self.page_size:
            try:
                predict_job_results = self.get_results(
                    page_offset=offset, page_size=self.page_size
                )
                if predict_job_results.result is not None:
                    results += predict_job_results.result
                    num_returned = len(predict_job_results.result)
                offset += num_returned
            except Exception as exc:
                if verbose:
                    logging.error(f"Failed to get results: {exc}")

        if self.job.job_type == JobType.workflow_predict_single_site:
            return self._fmt_ssp_results(results)
        else:
            return self._fmt_results(results)

    def get_slice(self, start: int, end: int):
        results = self.get_results(page_size=end - start, page_offset=start)
        return results.result or []  # could be none

    def get_results(
        self, page_size: int | None = None, page_offset: int | None = None
    ) -> WorkflowPredictSingleSiteJob | WorkflowPredictJob:
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
        if self.job.job_type is JobType.workflow_predict_single_site:
            return predict.get_single_site_prediction_results(
                session=self.session,
                job_id=self.id,
                page_size=page_size,
                page_offset=page_offset,
            )
        else:
            return predict.get_prediction_results(
                session=self.session,
                job_id=self.id,
                page_size=page_size,
                page_offset=page_offset,
            )
