from typing import Optional
import pydantic

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture, Job
import openprotein.config as config
from openprotein.api.jobs import load_job
from openprotein.models import DesignJobCreate, JobType, DesignResults
from openprotein.errors import (
    APIError,
    InvalidJob,
)

def create_design_job(session: APISession, design_job: DesignJobCreate) -> Job:
    """
    Send a POST request for protein design job.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    design_job : DesignJobCreate
        The details of the design job to be created, with the following parameters:
        - assay_id: The ID for the assay.
        - criteria: A list of CriterionItem lists for evaluating the design.
        - num_steps: The number of steps in the genetic algo. Default is 8.
        - pop_size: The population size for the genetic algo. Default is None.
        - n_offsprings: The number of offspring for the genetic algo. Default is None.
        - crossover_prob: The crossover probability for the genetic algo. Default is None.
        - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
        - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
        - mutation_positions: A list of positions where mutations may occur. Default is None.

    Returns
    -------
    Job
        The created job as a Job instance.
    """
    params = design_job.dict(exclude_none=True)
    # print(f"sending design: {params}")
    response = session.post("v1/workflow/design/genetic-algorithm", json=params)
    return Job(**response.json())


def get_design_results(
    session: APISession,
    job_id: str,
    page_size: Optional[int] = None,
    page_offset: Optional[int] = None,
) -> DesignResults:
    """
    Retrieves the results of a Design job.

    This function retrieves the results of a Design job by making a GET request to design..

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
    DesignJob
        The job object representing the Design job.

    Raises
    ------
    HTTPError
        If the GET request does not succeed.
    """
    endpoint = f"v1/workflow/design/{job_id}"
    params = {}
    if page_size is not None:
        params["page_size"] = page_size
    if page_offset is not None:
        params["page_offset"] = page_offset

    response = session.get(endpoint, params=params)

    return DesignResults(**response.json())


class DesignFutureMixin:
    session: APISession
    job: Job

    def get_results(
        self, page_size: Optional[int] = None, page_offset: Optional[int] = None
    ) -> DesignResults:
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
        DesignJob
            The job object representing the Design job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return get_design_results(self.session, self.job.job_id, page_size, page_offset)


class DesignFuture(DesignFutureMixin, AsyncJobFuture):
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

    def get(self, verbose: bool = False):
        """
        Get all the results of the design job.

        Args:
            verbose (bool, optional): If True, print verbose output. Defaults False.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            DesignJob: A list of predict objects representing the results.
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


class DesignAPI:
    """API interface for calling Design endpoints"""

    session: APISession

    def __init__(self, session: APISession):
        self.session = session

    def create_design_job(self, design_job: DesignJobCreate) -> DesignFuture:
        """
        Start a protein design job based on your assaydata, a trained ML model and Criteria (specified here).

        Parameters
        ----------
        design_job : DesignJobCreate
            The details of the design job to be created, with the following parameters:
            - assay_id: The ID for the assay.
            - criteria: A list of CriterionItem lists for evaluating the design.
            - num_steps: The number of steps in the genetic algo. Default is 8.
            - pop_size: The population size for the genetic algo. Default is None.
            - n_offsprings: The number of offspring for the genetic algo. Default is None.
            - crossover_prob: The crossover probability for the genetic algo. Default is None.
            - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
            - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
            - mutation_positions: A list of positions where mutations may occur. Default is None.

        Returns
        -------
        DesignFuture
            The created job as a DesignFuture instance.
        """
        job = create_design_job(self.session, design_job)
        return DesignFuture(self.session, job)

    def get_design_results(
        self,
        job_id: str,
        page_size: Optional[int] = None,
        page_offset: Optional[int] = None,
    ) -> DesignResults:
        """
        Retrieves the results of a Design job.

        Parameters
        ----------
        job_id : str
            The ID for the design job
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        DesignJob
            The job object representing the Design job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return get_design_results(self.session, job_id, page_size, page_offset)

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
        if job_details.job_type not in [JobType.design, JobType.predict_single_site]:
            raise InvalidJob(f"Job {job_id} is not of type {JobType.design}")
        return DesignFuture(self.session, job_details)
