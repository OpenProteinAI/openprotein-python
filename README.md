[![PyPI version](https://badge.fury.io/py/openprotein-python.svg)](https://pypi.org/project/openprotein-python/)
[![Coverage](https://docs.openprotein.ai/_static/coverage.svg)](https://pypi.org/project/openprotein-python/)
[![Conda version](https://anaconda.org/openprotein/openprotein-python/badges/version.svg)](https://anaconda.org/openprotein/openprotein-python)


# openprotein-python
The OpenProtein.AI Python Interface provides a user-friendly library to interact with the OpenProtein.AI REST API, enabling various tasks related to protein analysis and modeling.



# Table of Contents

|   | Workflow                                                                       | Description                                                                   |
|---|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 0 | [`Quick start`](#quick-start)                                                  | Quick start guide                                                             |
| 1 | [`Installation`](#installation)                                                | Install with pip or conda.                                                    |
| 2 | [`Python API reference`](https://docs.openprotein.ai/references/python-api/)   | Sessions, jobs and every interface in the Python client.                      |
| 3 | [`Models`](https://docs.openprotein.ai/references/python-api/models/)          | Embedding, scoring, generative and structure models, plus GP, SVD and UMAP.   |
| 4 | [`PoET`](https://docs.openprotein.ai/references/python-api/models/#poet)       | Score and generate sequences conditioned on a protein family.                 |
| 5 | [`Tutorials`](https://docs.openprotein.ai/tutorials/)                          | End-to-end workflows.                                                         |


# Quick-start

Get started with our quickstart README! You can peruse the [official documentation](https://docs.openprotein.ai/quickstart/) for more details!
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

- Python 3.10 or higher.
- Other dependencies as in `pyproject.toml`

# Getting started


Read on below for the quick-start guide, or see the [docs](https://docs.openprotein.ai/references/python-api/) for more information!

To begin, create a session using your login credentials.
```
import openprotein

# replace USERNAME and PASSWORD with your actual login credentials
session = openprotein.connect(USERNAME, PASSWORD)
```
## Job Status

The interface offers `Future` objects for asynchronous calls, allowing tracking of job status and result retrieval when ready. Given a future, you can check its status and retrieve results.

### Checking Job Status
Check the status of an `Future` using the following methods:
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
To view all jobs associated with each session, the following method is available, providing an option to filter results by date, job type, status, or name.
```
session.jobs.list() 
session.jobs.list(name="my-run")  # filter to jobs with an exact name match
```

### Retrieving Specific Job
For detailed information about a particular job, use the following command with the corresponding job ID:
``` 
session.jobs.get(JOB_ID)  # Replace JOB_ID with the ID of the specific job to be retrieved
```

### Naming Jobs
A job can be given a name and description once it's created, to make it easier to find:
```
session.jobs.update(JOB_ID, name="my-run", description="baseline model")

# equivalent, from a Future
future.rename(name="my-run", description="baseline model")
```

### Resuming Jobs
Jobs from prior workflows can be resumed by loading them by ID.
```
future = session.jobs.get(JOB_ID)  # Replace JOB_ID with the ID of the job to resume
```

## PoET interface
PoET scores and generates sequences conditioned on a prompt, a set of sequences from the protein family. Model handles are on `session.models`.

### Building a Prompt
Create a prompt from sequences you already have, or sample one from an MSA built by homology search.
```
model = session.models.poet

seed = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

prompt = session.prompt.create_prompt([seed])

# or
msa = session.align.create_msa(seed)
prompt = msa.sample_prompt(num_sequences=50)
```

### Scoring Sequences
To score sequences, use `score` with a list of queries and the prompt.
```
queries = [
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "MALWMRLLPLLVLLALWGPDPASAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "MALWTRLRPLLALLALWPPPPARAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREVEGPQVGALELAGGPGAGGLEGPPQKRGIVEQCCASVCSLYQLENYCN",
]

future = model.score(queries, prompt=prompt)
scores = future.wait()
# one log-likelihood per query, in the order submitted
# a prompt with several replicates gives one score per replicate
```

### Scoring Single Site Variants
To score every single-site substitution of a sequence, use `single_site`.
```
future = model.single_site(seed, prompt=prompt)
scores = future.wait()
# one score per substitution, 20 per position: positions in sequence order,
# substitutions in the order of openprotein.utils.sequence.AMINO_ACIDS
```

### Generating Sequences
To sample new sequences from the prompt, use `generate`.
```
future = model.generate(
    prompt=prompt,
    num_samples=100,
    temperature=1.0,
    top_k=15,
)
sequences = future.wait()                   # generated sequences as strings
scores = future.get(output="loglikelihood")  # one log-likelihood per sequence
```

See more at our [Homepage](https://docs.openprotein.ai/)
