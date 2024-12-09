"""
(Deprecated) OpenProtein objects for working with workflow design APIs.
"""

from openprotein.api.deprecated import design
from openprotein.base import APISession
from openprotein.schemas.deprecated import WorkflowDesign, WorkflowDesignJob

from ..futures import Future, PagedFuture


class DesignFuture(PagedFuture, Future):
    """Future Job for manipulating results"""

    job: WorkflowDesignJob

    def __init__(
        self, session: APISession, job: WorkflowDesignJob, page_size: int = 1000
    ):
        super().__init__(session, job)
        self.page_size = page_size

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    def _fmt_results(self, results: WorkflowDesign):
        # return [i.model_dump() for i in results.result]
        return results.result

    @property
    def id(self):
        return self.job.job_id

    # def get(self, step: int | None = None, verbose: bool = False) -> list[dict]:
    #     """
    #     Get all the results of the design job.

    #     Args:
    #         verbose (bool, optional): If True, print verbose output. Defaults False.

    #     Raises:
    #         APIError: If there is an issue with the API request.

    #     Returns:
    #         List: A list of predict objects representing the results.
    #     """
    #     page = self.page_size

    #     results = []
    #     num_returned = page
    #     offset = 0

    #     while num_returned >= page:
    #         try:
    #             response = self.get_results(
    #                 page_offset=offset, step=step, page_size=page
    #             )
    #             results += response.result
    #             num_returned = len(response.result)
    #             offset += num_returned
    #         except APIError as exc:
    #             if verbose:
    #                 print(f"Failed to get results: {exc}")
    #     return self._fmt_results(results)

    def get_slice(self, start: int, end: int, step: int | None = None, **kwargs):
        results = self.get_results(
            page_offset=start, page_size=end - start, step=step, **kwargs
        )
        return self._fmt_results(results)

    def get_results(
        self,
        step: int | None = None,
        page_size: int | None = None,
        page_offset: int | None = None,
    ) -> WorkflowDesign:
        """
        Retrieves the results of a Design job.

        This function retrieves the results of a Design job by making a GET request to design..

        Parameters
        ----------
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        WorkflowDesignJob
            The job object representing the Design job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return design.get_design_results(
            self.session,
            job_id=self.job.job_id,
            step=step,
            page_size=page_size,
            page_offset=page_offset,
        )
