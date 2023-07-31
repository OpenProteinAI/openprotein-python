# Train workflows

### *class* openprotein.api.train.TrainingAPI(session: APISession)

API interface for calling Train endpoints

#### create_training_job(assaydataset: [AssayDataset](data.md#openprotein.api.data.AssayDataset), measurement_name: str | List[str], model_name: str = '', force_preprocess: bool | None = False)

Create a training job on your data.

This function validates the inputs, formats the data, and sends the job.

* **Parameters:**
  * **assaydataset** ([*AssayDataset*](data.md#openprotein.api.data.AssayDataset)) – An AssayDataset object from which the assay_id is extracted.
  * **measurement_name** (*str* *or* *List**[**str**]*) – The name(s) of the measurement(s) to be used in the training job.
  * **model_name** (*str**,* *optional*) – The name to give the model.
  * **force_preprocess** (*bool**,* *optional*) – If set to True, preprocessing is forced even if data already exists.
* **Returns:**
  A TrainFuture Job
* **Return type:**
  [TrainFuture](#openprotein.api.train.TrainFuture)
* **Raises:**
  * **InvalidParameterError** – If the assaydataset is not an AssayDataset object,
        If any measurement name provided does not exist in the AssayDataset,
        or if the AssayDataset has fewer than 3 data points.
  * **HTTPError** – If the request to the server fails.

#### get_training_results(job_id: str)

Get training results (e.g. loss etc).

* **Parameters:**
  **job_id** (*str*) – job_id to get
* **Returns:**
  A TrainFuture Job
* **Return type:**
  [TrainFuture](#openprotein.api.train.TrainFuture)

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

### *class* openprotein.api.train.TrainFuture(session: APISession, job: Job, assaymetadata: AssayMetadata | None = None)

Future Job for manipulating results

#### get_assay_data()

NOT IMPLEMENTED.

Get the assay data used for the training job.

* **Returns:**
  The assay data.
