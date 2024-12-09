from openprotein.api.deprecated import design
from openprotein.app.models.deprecated import WorkflowDesignFuture
from openprotein.base import APISession
from openprotein.schemas.deprecated import WorkflowDesign, WorkflowDesignJobCreate


class DesignAPI:
    """Interface for calling Workflow Design endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def create_design_job(
        self, design_job: WorkflowDesignJobCreate
    ) -> WorkflowDesignFuture:
        """
        Start a protein design job based on your assaydata, a trained ML model and Criteria (specified here).

        Parameters
        ----------
        design_job : DesignJobCreate
            The details of the design job to be created, with the following parameters:
            - assay_id: The ID for the assay.
            - criteria: Criteria for evaluating the design.
            - num_steps: The number of steps in the genetic algo. Default is 8.
            - pop_size: The population size for the genetic algo. Default is None.
            - n_offsprings: The number of offspring for the genetic algo. Default is None.
            - crossover_prob: The crossover probability for the genetic algo. Default is None.
            - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
            - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
            - allowed_tokens: A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Default is None.

        Returns
        -------
        DesignFuture
            The created job as a DesignFuture instance.
        """
        return WorkflowDesignFuture.create(
            session=self.session, job=design.create_design_job(self.session, design_job)
        )

    def get_design_results(
        self,
        job_id: str,
        step: int | None = None,
        page_size: int | None = None,
        page_offset: int | None = None,
    ) -> WorkflowDesign:
        """
        Retrieves the results of a Design job.

        Parameters
        ----------
        job_id : str
            The ID for the design job
        step: int
            The design step to retrieve, if None: retrieve all.
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
            step=step,
            job_id=job_id,
            page_size=page_size,
            page_offset=page_offset,
        )
