[![PyPI version](https://badge.fury.io/py/openprotein-python.svg)](https://pypi.org/project/openprotein-python/)
[![Coverage](https://dev.docs.openprotein.ai/api-python/_images/coverage.svg)](https://pypi.org/project/openprotein-python/)
[![Conda version](https://anaconda.org/openprotein/openprotein-python/badges/version.svg)](https://anaconda.org/openprotein/openprotein-python)


# openprotein-python
The OpenProtein.AI Python Interface provides a user-friendly library to interact with the OpenProtein.AI REST API, enabling various tasks related to protein analysis and modeling.



# Table of Contents

|   | Workflow                                           | Description                                          |
|---|----------------------------------------------------|------------------------------------------------------|
| 0 | [`Quick start`](#Quick-start)                    | Quick start guide                     |
| 1 | [`Installation`](https://docs.openprotein.ai/api-python/installation.html)                    | Install guide for pip and conda.                     |
| 2 | [`Session management`](https://docs.openprotein.ai/api-python/overview.html)        | An overview of the OpenProtein Python Client & the asynchronous jobs system. |
| 3 | [`Asssay-based Sequence Learning`](https://docs.openprotein.ai/api-python/core_workflow.html) | Covers core tasks such as data upload, model training & prediction, and sequence design. |
| 4 | [`De Novo prediction & generative models (PoET)`](https://docs.openprotein.ai/api-python/poet_workflow.html) | Covers PoET, a protein LLM for *de novo* scoring, as well as sequence generation. |
| 5 | [`Protein Language Models & Embeddings`](https://docs.openprotein.ai/api-python/embedding_workflow.html) | Covers methods for creating sequence embeddings with proprietary & open-source models. |


# Quick-start

Get started with our quickstart README! You can peruse the [official documentation](https://docs.openprotein.ai/api-python/) for more details!
## Installation 

To install the python interface using pip, run the following command: 
```
pip install openprotein-python
```

or with conda:
```
conda install -c openprotein openprotein-python
```

### Requirements

- Python 3.8 or higher.
- pydantic version 1.0 or newer.
- requests version 2.0 or newer.
- tqdm version 4.0 or newer.
- pandas version 1.0 or newer.

# Getting started


Read on below for the quick-start guide, or see the [docs](https://docs.openprotein.ai/api-python/) for more information!

To begin, create a session using your login credentials.
```
import openprotein

# replace USERNAME and PASSWORD with your actual login credentials
session = openprotein.connect(USERNAME, PASSWORD)
```
## Job Status

The interface offers `AsyncJobFuture` objects for asynchronous calls, allowing tracking of job status and result retrieval when ready. Given a future, you can check its status and retrieve results.

### Checking Job Status
Check the status of an `AsyncJobFuture` using the following methods:
```
future.refresh()  # call the backend to update the job status
future.done()     # returns True if the job is done, meaning the status could be SUCCESS, FAILED, or CANCELLED
```

### Retrieving Job Results
Once the job has finished, retrieve the results using the following methods:
```
result = future.wait()     # wait until done and then fetch results

#verbosity is controlled with verbose arg
result = future.get(verbose=True)  # get the result from a finished job
```

## Jobs Interface

### Listing Jobs
To view all jobs associated with each session, the following method is available, providing an option to filter results by date, job type, or status.
```
session.jobs.list() 
```

### Retrieving Specific Job
For detailed information about a particular job, use the following command with the corresponding job ID:
``` 
session.jobs.get(JOB_ID)  # Replace JOB_ID with the ID of the specific job to be retrieved
```

### Resuming Jobs
Jobs from prior workflows can be resumed using the load_job method provided by each API. 
```
session.load_job(JOB_ID)  # Replace JOB_ID with the ID of the training job to resume
```

## PoET interface
The PoET Interface allows scoring, generating, and retrieving sequences using the PoET model.

### Scoring Sequences
To score sequences, use the score function. Provide a prompt and a list of queries. The results will be a list of (sequence, score) pydantic objects.

```
prompt_seqs = b'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN'

prompt = session.poet.upload_prompt(prompt_seqs)
```

```
queries = [
    b'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
    b'MALWMRLLPLLVLLALWGPDPASAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
    b'MALWTRLRPLLALLALWPPPPARAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREVEGPQVGALELAGGPGAGGLEGPPQKRGIVEQCCASVCSLYQLENYCN',
    b'MALWIRSLPLLALLVFSGPGTSYAAANQHLCGSHLVEALYLVCGERGFFYSPKARRDVEQPLVSSPLRGEAGVLPFQQEEYEKVKRGIVEQCCHNTCSLYQLENYCN',
    b'MALWMRLLPLLALLALWAPAPTRAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREVEDLQVRDVELAGAPGEGGLQPLALEGALQKRGIVEQCCTSICSLYQLENYCN',
]
```

```
future = session.poet.score(prompt, queries)
result = future.wait()
# result is a list of (sequence, score) pydantic objects
```

### Scoring Single Site Variants
For scoring single site variants, use the `single_site function`, providing the original sequence and setting `prompt_is_seed` to True if the prompt is a seed sequence.
```
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
future = session.poet.single_site(prompt, sequence, prompt_is_seed=True) 
result = future.wait()
# result is a dictionary of {variant: score}
```

### Generating Sequences
To generate sequences from the PoET model, use the `generate` function with relevant parameters. The result will be a list of generated samples.
```
future = session.poet.generate(
    prompt,
    max_seqs_from_msa=1024,
    num_samples=100,
    temperature=1.0,
    topk=15
)
samples = future.wait()
```

### Retrieving Input Sequences
You can retrieve the prompt, MSA, or seed sequences for a PoET job using the `get_input` function or the individual functions for each type.
```
future.get_input(INPUT_TYPE)
# or, functions for each type
future.get_prompt()
future.get_msa()
future.get_seed()
```

See more at our [Homepage](https://docs.openprotein.ai/)