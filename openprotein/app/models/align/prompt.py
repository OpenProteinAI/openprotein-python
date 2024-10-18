from typing import Iterator

from openprotein import config
from openprotein.api import align
from openprotein.api import job as job_api
from openprotein.base import APISession
from openprotein.schemas import PromptJob

from ..futures import Future
from .base import AlignFuture


class PromptFuture(AlignFuture, Future):
    """
    Represents a result of a prompt job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.
    page_size : int
        The number of results to fetch in a single page.

    Methods
    -------
    get(verbose=False)
        Get the final results of the PoET scoring job.

    Returns
    -------
    List[PoetScoreResult]
        The list of results from the PoET scoring job.
    """

    job: PromptJob

    def __init__(
        self,
        session: APISession,
        job: PromptJob,
        page_size: int = config.POET_PAGE_SIZE,
        msa_id: str | None = None,
    ):
        """
        init a PoetScoreFuture instance.

        Parameters
        ----------
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET scoring job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

        if msa_id is None:
            msa_id = job_api.job_args_get(self.session, job.job_id).get("root_msa")
        self._msa_id = msa_id
        self.prompt_id = self.job.job_id

    # def wait(self, verbose: bool = False, **kwargs) -> Iterator[list[str]]:
    #     _ = self.job.wait(
    #         session=self.session,
    #         interval=config.POLLING_INTERVAL,
    #         timeout=config.POLLING_TIMEOUT,
    #         verbose=verbose,
    #     )  # no progress to track
    #     return self.get(verbose=verbose, **kwargs)

    def get(
        self, prompt_index: int | None = None, verbose: bool = False
    ) -> Iterator[list[str]]:
        return align.get_prompt(
            session=self.session, job=self.job, prompt_index=prompt_index
        )
