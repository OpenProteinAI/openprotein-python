# PoET workflows

### *class* openprotein.api.poet.PoetAPI(session: APISession)

API interface for calling Poet and Align endpoints

#### create_msa(seed: bytes)

Construct an MSA via homology search with the seed sequence.

* **Parameters:**
  **seed** (*bytes*) – Seed sequence for the MSA construction.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  Job object containing the details of the MSA construction.
* **Return type:**
  MSAJob

#### generate(prompt: [PromptFuture](#openprotein.api.poet.PromptFuture) | str, num_samples=100, temperature=1.0, topk=None, topp=None, max_length=1000, seed=None)

Generate protein sequences conditioned on a prompt.

* **Parameters:**
  * **prompt** (*PromptJob*) – The prompt to use for generating sequences.
  * **num_samples** (*int**,* *optional*) – The number of samples to generate, by default 100.
  * **temperature** (*float**,* *optional*) – The temperature for sampling. Higher values produce more random outputs, by default 1.0.
  * **topk** (*int**,* *optional*) – The number of top-k residues to consider during sampling, by default None.
  * **topp** (*float**,* *optional*) – The cumulative probability threshold for top-p sampling, by default None.
  * **max_length** (*int**,* *optional*) – The maximum length of generated proteins, by default 1000.
  * **seed** (*int**,* *optional*) – Seed for random number generation, by default a random number.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  An object representing the status and information about the generation job.
* **Return type:**
  Job

#### get_msa(job: Job)

Get generated MSA for a given job.

* **Parameters:**
  **job** (*Job*) – The job for which to retrieve data.
* **Returns:**
  A CSV reader for the response data.
* **Return type:**
  csv.reader

#### get_msa_job(job_id: str)

Get MSA job based on job_id.

* **Parameters:**
  **job_id** (*str*) – job ID for a prompt job
* **Returns:**
  A prompt job instance
* **Return type:**
  MSAJob

#### get_prompt(job: Job, prompt_index: int | None = None)

Get prompts for a given job.

* **Parameters:**
  * **job** (*Job*) – The job for which to retrieve data.
  * **prompt_index** (*Optional**[**int**]*) – The replicate number for the prompt (input_type=-PROMPT only)
* **Returns:**
  A CSV reader for the response data.
* **Return type:**
  csv.reader

#### get_prompt_job(job_id: str)

Get prompt job based on job_id.

* **Parameters:**
  **job_id** (*str*) – job ID for a prompt job
* **Returns:**
  A prompt job instance
* **Return type:**
  PromptJob

#### get_seed(job: Job)

Get input data for a given msa job.

* **Parameters:**
  **job** (*Job*) – The job for which to retrieve data.
* **Returns:**
  A CSV reader for the response data.
* **Return type:**
  csv.reader

#### load_msa_job(msa_id: str)

Reload a previously ran MSA job to resume where you left off.

* **Parameters:**
  **msa_id** (*str*) – ID for job.
* **Raises:**
  **InvalidJob** – If job is of incorrect type.
* **Returns:**
  Job to resume workflows.
* **Return type:**
  [PromptFuture](#openprotein.api.poet.PromptFuture)

#### load_poet_job(job_id: str)

Reload a previously ran Poet job to resume where you left off.

* **Parameters:**
  **job_id** (*str*) – ID for job.
* **Raises:**
  **InvalidJob** – If job is of incorrect type.
* **Returns:**
  Job to resume workflows.
* **Return type:**
  PoetFuture

#### load_prompt_job(prompt_id: str)

Reload a previously ran prompt job to resume where you left off.

* **Parameters:**
  **prompt_id** (*str*) – ID for job.
* **Raises:**
  **InvalidJob** – If job is of incorrect type.
* **Returns:**
  Job to resume workflows.
* **Return type:**
  [PromptFuture](#openprotein.api.poet.PromptFuture)

#### score(prompt: [PromptFuture](#openprotein.api.poet.PromptFuture) | str, queries: List[bytes])

Score query sequences using the specified prompt.

* **Parameters:**
  * **prompt** (*Prompt*) – Prompt job to use for scoring the sequences.
  * **queries** (*List**[**bytes**]*) – Sequences to score.
* **Returns:**
  The scores of the query sequences.
* **Return type:**
  results

#### single_site(prompt: [PromptFuture](#openprotein.api.poet.PromptFuture) | str, sequence: bytes)

Score all single substitutions of the query sequence using the specified prompt.

* **Parameters:**
  * **prompt** (*Prompt*) – Prompt job to use for scoring the sequences.
  * **sequence** (*bytes*) – Sequence to analyse.
* **Returns:**
  The scores of the mutated sequence.
* **Return type:**
  results

#### upload_msa(msa_file)

Upload an MSA from file.

* **Parameters:**
  **msa_file** (*str**,* *optional*) – Ready-made MSA. If not provided, default value is None.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  Job object containing the details of the MSA upload.
* **Return type:**
  MSAJob

#### upload_prompt(prompt_file: BinaryIO)

Directly upload a prompt.

Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
You can specify multiple prompts (one per replicate) with an <END_PROMPT> and newline between CSVs.

* **Parameters:**
  **prompt_file** (*BinaryIO*) – Binary I/O object representing the prompt file.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  An object representing the status and results of the prompt job.
* **Return type:**
  PromptJob

### *class* openprotein.api.poet.PoetScoreFuture(session: APISession, job: Job, page_size=50000)

Represents a result of a PoET scoring job.

#### session

An instance of APISession for API interactions.

* **Type:**
  APISession

#### job

The PoET scoring job.

* **Type:**
  Job

#### page_size

The number of results to fetch in a single page.

* **Type:**
  int

#### get(verbose=False)

Get the final results of the PoET  job.

#### get(verbose=False)

Get the final results of the PoET scoring job.

* **Parameters:**
  **verbose** (*bool**,* *optional*) – If True, print verbose output. Defaults to False.
* **Raises:**
  **APIError** – If there is an issue with the API request.
* **Returns:**
  A list of PoetScoreResult objects representing the scoring results.
* **Return type:**
  List[PoetScoreResult]

### *class* openprotein.api.poet.PoetSingleSiteFuture(session: APISession, job: Job, page_size=50000)

Represents a result of a PoET single-site analysis job.

#### session

An instance of APISession for API interactions.

* **Type:**
  APISession

#### job

The PoET scoring job.

* **Type:**
  Job

#### page_size

The number of results to fetch in a single page.

* **Type:**
  int

#### get(verbose=False)

Get the final results of the PoET  job.

#### get(verbose=False)

Get the results of a PoET single-site analysis job.

* **Parameters:**
  **verbose** (*bool**,* *optional*) – If True, print verbose output. Defaults to False.
* **Returns:**
  A dictionary mapping mutation codes to scores.
* **Return type:**
  Dict[bytes, float]
* **Raises:**
  **APIError** – If there is an issue with the API request.

### *class* openprotein.api.poet.PoetGenerateFuture(session: APISession, job: Job | str)

Represents a result of a PoET generation job.

#### session

An instance of APISession for API interactions.

* **Type:**
  APISession

#### job

The PoET scoring job.

* **Type:**
  Job

#### Methods

stream() -> Iterator[PoetScoreResult]:
: Stream the results of the PoET generation job.

#### stream()

Stream the results from the response.

* **Returns:**
  **PoetScoreResult** – A result object containing the sequence, score, and name.
* **Return type:**
  Yield
* **Raises:**
  **APIError** – If the request fails.

### *class* openprotein.api.poet.PromptFuture(session: APISession, job: Job, page_size=50000, msa_id: str | None = None)

Represents a result of a prompt job.

#### session

An instance of APISession for API interactions.

* **Type:**
  APISession

#### job

The PoET scoring job.

* **Type:**
  Job

#### page_size

The number of results to fetch in a single page.

* **Type:**
  int

#### get(verbose=False)

Get the final results of the PoET scoring job.

* **Returns:**
  The list of results from the PoET scoring job.
* **Return type:**
  List[PoetScoreResult]

### *class* openprotein.api.poet.MSAFuture(session: APISession, job: Job, page_size=50000)

Represents a result of a MSA job.

#### session

An instance of APISession for API interactions.

* **Type:**
  APISession

#### job

The PoET scoring job.

* **Type:**
  Job

#### page_size

The number of results to fetch in a single page.

* **Type:**
  int

#### get(verbose=False)

Get the final results of the PoET scoring job.

* **Returns:**
  The list of results from the PoET scoring job.
* **Return type:**
  List[PoetScoreResult]

#### sample_prompt(num_sequences: int | None = None, num_residues: int | None = None, method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT, homology_level: float = 0.8, max_similarity: float = 1.0, min_similarity: float = 0.0, always_include_seed_sequence: bool = False, num_ensemble_prompts: int = 1, random_seed: int | None = None)

Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.

* **Parameters:**
  * **num_sequences** (*int**,* *optional*) – Maximum number of sequences in the prompt. Must be  <100.
  * **num_residues** (*int**,* *optional*) – Maximum number of residues (tokens) in the prompt. Must be less than 24577.
  * **method** (*MSASamplingMethod**,* *optional*) – Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
  * **homology_level** (*float**,* *optional*) – Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
  * **max_similarity** (*float**,* *optional*) – Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
  * **min_similarity** (*float**,* *optional*) – Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
  * **always_include_seed_sequence** (*bool**,* *optional*) – Whether to always include the seed sequence in the MSA. Defaults to False.
  * **num_ensemble_prompts** (*int**,* *optional*) – Number of ensemble jobs to run. Defaults to 1.
  * **random_seed** (*int**,* *optional*) – Seed for random number generation. Defaults to a random number between 0 and 2\*\*32-1.
* **Raises:**
  * **InvalidParameterError** – If provided parameter values are not in the allowed range.
  * **MissingParameterError** – If both or none of ‘num_sequences’, ‘num_residues’ is specified.
* **Return type:**
  PromptJob

#### wait(verbose: bool = False)

Wait for job to complete, then fetch results.

* **Parameters:**
  * **interval** (*int**,* *optional*) – time between polling. Defaults to config.POLLING_INTERVAL.
  * **timeout** (*int**,* *optional*) – max time to wait. Defaults to None.
  * **verbose** (*bool**,* *optional*) – verbosity flag. Defaults to False.
* **Returns:**
  results of job
* **Return type:**
  results
