from openprotein.base import APISession
from openprotein.schemas import DesignJobCreate, DesignResults, Job


def create_design_job(session: APISession, design_job: DesignJobCreate):
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
        - allowed_tokens: A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Default is None.

    Returns
    -------
    Job
        The created job as a Job instance.
    """
    params = design_job.model_dump(exclude_none=True)
    # print(f"sending design: {params}")
    response = session.post("v1/workflow/design/genetic-algorithm", json=params)
    return Job.model_validate(response.json())


def get_design_results(
    session: APISession,
    job_id: str,
    step: int | None = None,
    page_size: int | None = None,
    page_offset: int | None = None,
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
    step: int
        The step to retrieve. -1 indicates the last step.
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
    if step is not None:
        params["step"] = step

    response = session.get(endpoint, params=params)

    return DesignResults.model_validate(response.json())
