# Design workflows

### *class* openprotein.api.design.DesignAPI(session: APISession)

API interface for calling Design endpoints

#### create_design_job(design_job: DesignJobCreate)

Start a protein design job based on your assaydata, a trained ML model and Criteria (specified here).

* **Parameters:**
  **design_job** (*DesignJobCreate*) – The details of the design job to be created, with the following parameters:
  - assay_id: The ID for the assay.
  - criteria: A list of CriterionItem lists for evaluating the design.
  - num_steps: The number of steps in the genetic algo. Default is 8.
  - pop_size: The population size for the genetic algo. Default is None.
  - n_offsprings: The number of offspring for the genetic algo. Default is None.
  - crossover_prob: The crossover probability for the genetic algo. Default is None.
  - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
  - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
  - mutation_positions: A list of positions where mutations may occur. Default is None.
* **Returns:**
  The created job as a DesignFuture instance.
* **Return type:**
  [DesignFuture](#openprotein.api.design.DesignFuture)

#### get_design_results(job_id: str, page_size: int | None = None, page_offset: int | None = None)

Retrieves the results of a Design job.

* **Parameters:**
  * **job_id** (*str*) – The ID for the design job
  * **page_size** (*Optional**[**int**]**,* *default is None*) – The number of results to be returned per page. If None, all results are returned.
  * **page_offset** (*Optional**[**int**]**,* *default is None*) – The number of results to skip. If None, defaults to 0.
* **Returns:**
  The job object representing the Design job.
* **Return type:**
  DesignJob
* **Raises:**
  **HTTPError** – If the GET request does not succeed.

#### load_job(job_id: str)

Reload a Submitted job to resume from where you left off!

* **Parameters:**
  **job_id** (*str*) – The identifier of the job whose details are to be loaded.
* **Returns:**
  Job
* **Return type:**
  Job
* **Raises:**
  * **HTTPError** – If the request to the server fails.
  * **InvalidJob** – If the Job is of the wrong type

### *class* openprotein.api.design.DesignFuture(session: APISession, job: Job, page_size=1000)

Future Job for manipulating results

#### get(verbose: bool = False)

Get all the results of the design job.

* **Parameters:**
  **verbose** (*bool**,* *optional*) – If True, print verbose output. Defaults False.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  A list of predict objects representing the results.
* **Return type:**
  DesignJob
