# AssayData workflows

### *class* openprotein.api.data.DataAPI(session: APISession)

API interface for calling AssayData endpoints

#### create(table: DataFrame, name: str, description: str | None = None)

Create a new assay dataset.

* **Parameters:**
  * **table** (*pd.DataFrame*) – DataFrame containing the assay data.
  * **name** (*str*) – Name of the assay dataset.
  * **description** (*str**,* *optional*) – Description of the assay dataset, by default None.
* **Returns:**
  Created assay dataset.
* **Return type:**
  [AssayDataset](#openprotein.api.data.AssayDataset)

#### get(assay_id: str)

Get an assay dataset by its ID.

* **Parameters:**
  **assay_id** (*str*) – ID of the assay dataset.
* **Returns:**
  Assay dataset with the specified ID.
* **Return type:**
  [AssayDataset](#openprotein.api.data.AssayDataset)
* **Raises:**
  **KeyError** – If no assay dataset with the given ID is found.

#### list()

List all assay datasets.

* **Returns:**
  List of all assay datasets.
* **Return type:**
  List[[AssayDataset](#openprotein.api.data.AssayDataset)]

#### load_job(assay_id: str)

Reload a Submitted job to resume from where you left off!

* **Parameters:**
  **assay_id** (*str*) – The identifier of the job whose details are to be loaded.
* **Returns:**
  Job
* **Return type:**
  Job
* **Raises:**
  * **HTTPError** – If the request to the server fails.
  * **InvalidJob** – If the Job is of the wrong type

### *class* openprotein.api.data.AssayDataset(session: APISession, metadata: AssayMetadata)

Future Job for manipulating results

#### get_first()

Get head slice of assay data.

* **Returns:**
  Dataframe containing the slice of assay data.
* **Return type:**
  pd.DataFrame

#### get_slice(start: int, end: int)

Get a slice of assay data.

* **Parameters:**
  * **start** (*int*) – Start index of the slice.
  * **end** (*int*) – End index of the slice.
* **Returns:**
  Dataframe containing the slice of assay data.
* **Return type:**
  pd.DataFrame

#### list_models()

List models assoicated with assay.

* **Returns:**
  List of models
* **Return type:**
  List

#### update(assay_name: str | None = None, assay_description: str | None = None)

Update the assay metadata.

* **Parameters:**
  * **assay_name** (*str**,* *optional*) – New name of the assay, by default None.
  * **assay_description** (*str**,* *optional*) – New description of the assay, by default None.
* **Return type:**
  None
