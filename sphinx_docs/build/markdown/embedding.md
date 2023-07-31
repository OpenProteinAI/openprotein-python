# Embedding workflows

### *class* openprotein.api.embedding.EmbeddingAPI(session: APISession)

This class defines a high level interface for accessing the embeddings API.

#### delete_svd(svd_id: str)

Delete SVD model.

* **Parameters:**
  **svd_id** (*str*) – The ID of the SVD  job.
* **Returns:**
  True: successful deletion
* **Return type:**
  bool

#### embed(model: [ProtembedModel](#openprotein.api.embedding.ProtembedModel) | [SVDModel](#openprotein.api.embedding.SVDModel) | str, sequences: List[bytes], reduction='MEAN')

Embed sequences using the specified model.

* **Parameters:**
  * **model** (*Union**[*[*ProtembedModel*](#openprotein.api.embedding.ProtembedModel)*,* [*SVDModel*](#openprotein.api.embedding.SVDModel)*,* *str**]*) – The model to use for embedding. This can be an instance of ProtembedModel, SVDModel,
    or a string representing the model_id.
  * **sequences** (*List**[**bytes**]*) – List of byte sequences to be embedded.
  * **reduction** (*str**,* *optional*) – The reduction operation to be applied on the embeddings.
    Options are None, “MEAN”, or “SUM”. Default is None.
* **Returns:**
  An instance of EmbeddingResultFuture
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)
* **Raises:**
  **TypeError** – If the input model is neither ProtembedModel, SVDModel, nor str.

#### fit_svd(model_id: str, sequences: List[bytes], n_components: int = 1024, reduction: str | None = None)

Fit an SVD on the sequences with the specified model_id and hyperparameters (n_components).

* **Parameters:**
  * **model_id** (*str*) – The ID of the model to fit the SVD on.
  * **sequences** (*List**[**bytes**]*) – The list of sequences to use for the SVD fitting.
  * **n_components** (*int**,* *optional*) – The number of components for the SVD, by default 1024.
  * **reduction** (*Optional**[**str**]**,* *optional*) – The reduction method to apply to the embeddings, by default None.
* **Returns:**
  The model with the SVD fit.
* **Return type:**
  [SVDModel](#openprotein.api.embedding.SVDModel)

#### get_model(model_id: str)

Get model by model_id.

ProtembedModel allows all the usual job manipulation:             e.g. making POST and GET requests for this model specifically.

* **Parameters:**
  **model_id** (*str*) – the model identifier
* **Returns:**
  The model
* **Return type:**
  [ProtembedModel](#openprotein.api.embedding.ProtembedModel)
* **Raises:**
  **HTTPError** – If the GET request does not succeed.

#### get_results(job)

Retrieves the results of an embedding job.

* **Parameters:**
  **job** (*Job*) – The embedding job whose results are to be retrieved.
* **Returns:**
  An instance of EmbeddingResultFuture
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)

#### get_svd(svd_id: str)

Get SVD job results. Including SVD dimension and sequence lengths.

Requires a successful SVD job from fit_svd

* **Parameters:**
  **svd_id** (*str*) – The ID of the SVD  job.
* **Returns:**
  The model with the SVD fit.
* **Return type:**
  [SVDModel](#openprotein.api.embedding.SVDModel)

#### get_svd_results(job: Job)

Get SVD job results. Including SVD dimension and sequence lengths.

Requires a successful SVD job from fit_svd

* **Parameters:**
  **job** (*Job*) – SVD JobFuture
* **Returns:**
  The model with the SVD fit.
* **Return type:**
  [SVDModel](#openprotein.api.embedding.SVDModel)

#### list_models()

list models available for creating embeddings of your sequences

#### list_svd()

List SVD models made by user.

Takes no args.

* **Returns:**
  SVDModels
* **Return type:**
  list[[SVDModel](#openprotein.api.embedding.SVDModel)]

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

### *class* openprotein.api.embedding.EmbeddingResultFuture(session: APISession, job: Job, sequences=None, max_workers=10)

Future Job for manipulating results

#### get_item(sequence: bytes)

Get embedding results for specified sequence.

* **Parameters:**
  **sequence** (*bytes*) – sequence to fetch results for
* **Returns:**
  embeddings
* **Return type:**
  np.ndarray

### *class* openprotein.api.embedding.ProtembedModel(session, model_id, metadata=None)

Class providing inference endpoints for protein embedding models served by OpenProtein.

#### attn(sequences: List[bytes])

Attention embeddings for sequences using this model.

* **Parameters:**
  **sequences** (*List**[**bytes**]*) – sequences to SVD
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)

#### embed(sequences: List[bytes], reduction: str | None = 'MEAN')

Embed sequences using this model.

* **Parameters:**
  * **sequences** (*List**[**bytes**]*) – sequences to SVD
  * **reduction** (*str*) – embeddings reduction to use (e.g. mean)
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)

#### fit_svd(sequences: List[bytes], n_components: int = 1024, reduction: str | None = None)

Fit an SVD on the embedding results of this model.

This function will create an SVDModel based on the embeddings from this model             as well as the hyperparameters specified in the args.

* **Parameters:**
  * **sequences** (*List**[**bytes**]*) – sequences to SVD
  * **n_components** (*int*) – number of components in SVD. Will determine output shapes
  * **reduction** (*str*) – embeddings reduction to use (e.g. mean)
* **Return type:**
  [SVDModel](#openprotein.api.embedding.SVDModel)

#### get_metadata()

Get model metadata for this model.

* **Return type:**
  ModelMetadata

#### logits(sequences: List[bytes])

logit embeddings for sequences using this model.

* **Parameters:**
  **sequences** (*List**[**bytes**]*) – sequences to SVD
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)

### *class* openprotein.api.embedding.SVDModel(session: APISession, metadata: SVDMetadata)

Class providing embedding endpoint for SVD models.         Also allows retrieving embeddings of sequences used to fit the SVD with get.

#### delete()

Delete this SVD model.

#### embed(sequences: List[bytes])

Use this SVD model to reduce embeddings results.

* **Parameters:**
  **sequences** (*List**[**bytes**]*) – List of protein sequences.
* **Returns:**
  Class for further job manipulation.
* **Return type:**
  [EmbeddingResultFuture](#openprotein.api.embedding.EmbeddingResultFuture)

#### get_embeddings()

Get SVD embedding results for this model.

* **Returns:**
  **EmbeddingResultFuture**
* **Return type:**
  class for futher job manipulation

#### get_inputs()

Get sequences used for embeddings job.

* **Returns:**
  **List[bytes]**
* **Return type:**
  list of sequences

#### get_job()

Get job associated with this SVD model

#### get_model()

Fetch embeddings model
