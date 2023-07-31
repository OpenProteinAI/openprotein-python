# Predict workflows

### *class* openprotein.api.predict.PredictAPI(session: APISession)

API interface for calling Predict endpoints

#### create_predict_job(sequences: List, train_job: [TrainFuture](train.md#openprotein.api.train.TrainFuture), model_ids: List[str] | None = None)

Creates a new Predict job for a given list of sequences and a trained model.

* **Parameters:**
  * **sequences** (*List*) – The list of sequences to be used for the Predict job.
  * **train_job** ([*TrainFuture*](train.md#openprotein.api.train.TrainFuture)) – The train job object representing the trained model.
  * **model_ids** (*List**[**str**]**,* *optional*) – The list of model ids to be used for Predict. Default is None.
* **Returns:**
  The job object representing the Predict job.
* **Return type:**
  [PredictFuture](#openprotein.api.predict.PredictFuture)
* **Raises:**
  * **InvalidParameterError** – If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
  * **InvalidParameterError** – If BOTH train_job and model_ids are specified
  * **InvalidParameterError** – If NEITHER train_job or model_ids is specified
  * **APIError** – If the backend refuses the job (due to sequence length or invalid inputs)

#### create_predict_single_site(sequence: str, train_job: [TrainFuture](train.md#openprotein.api.train.TrainFuture), model_ids: List[str] | None = None)

Creates a new Predict job for single site mutation analysis with a trained model.

* **Parameters:**
  * **sequence** (*str*) – The sequence for single site analysis.
  * **train_job** ([*TrainFuture*](train.md#openprotein.api.train.TrainFuture)) – The train job object representing the trained model.
  * **model_ids** (*List**[**str**]**,* *optional*) – The list of model ids to be used for Predict. Default is None.
* **Returns:**
  The job object representing the Predict job.
* **Return type:**
  [PredictFuture](#openprotein.api.predict.PredictFuture)
* **Raises:**
  * **InvalidParameterError** – If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
  * **InvalidParameterError** – If BOTH train_job and model_ids are specified
  * **InvalidParameterError** – If NEITHER train_job or model_ids is specified
  * **APIError** – If the backend refuses the job (due to sequence length or invalid inputs)

#### get_prediction_results(job_id: str, page_size: int | None = None, page_offset: int | None = None)

Retrieves the results of a Predict job.

* **Parameters:**
  * **job_id** (*str*) – The ID of the Predict job.
  * **page_size** (*Optional**[**int**]**,* *default is None*) – The number of results to be returned per page. If None, all results are returned.
  * **page_offset** (*Optional**[**int**]**,* *default is None*) – The number of results to skip. If None, defaults to 0.
* **Returns:**
  The job object representing the Predict job.
* **Return type:**
  PredictJob
* **Raises:**
  **HTTPError** – If the GET request does not succeed.

#### get_single_site_prediction_results(job_id: str, page_size: int | None = None, page_offset: int | None = None)

Retrieves the results of a single site Predict job.

* **Parameters:**
  * **job_id** (*str*) – The ID of the Predict job.
  * **page_size** (*Optional**[**int**]**,* *default is None*) – The number of results to be returned per page. If None, all results are returned.
  * **page_offset** (*Optional**[**int**]**,* *default is None*) – The page number to start retrieving results from. If None, defaults to 0.
* **Returns:**
  The job object representing the single site Predict job.
* **Return type:**
  PredictSingleSiteJob
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

### *class* openprotein.api.predict.PredictFuture(session: APISession, job: Job, page_size=1000)

Future Job for manipulating results

#### get(verbose: bool = False)

Get all the results of the predict job.

* **Parameters:**
  **verbose** (*bool**,* *optional*) – If True, print verbose output. Defaults False.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  A list of predict objects representing the results.
* **Return type:**
  PredictJob
