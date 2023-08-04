[![PyPI version](https://badge.fury.io/py/openprotein-python.svg)](https://pypi.org/project/openprotein-python/)
[![Coverage](https://github.com/OpenProteinAI/openprotein-python-private/raw/develop/apidocs/source/coverage.svg)](https://pypi.org/project/openprotein-python/)

# openprotein-python
Python interface for the OpenProtein.AI REST API.

## Installation 

You can install with pip: 

```
pip install openprotein-python
```
## Getting started

First, create a session using your login credentials.
```
import openprotein
session = openprotein.connect(USERNAME, PASSWORD)
```

Async calls return `AsyncJobFuture` objects that allow tracking the status of the job and retrieving the result when it's ready.

Given a future, check its status and retrieve results
```
future.refresh() # call the backend to update the job status
future.done() # returns True if the job is done, meaning the status could be SUCCESS, FAILED, or CANCELLED
future.wait() # wait until done and then fetch results, verbosity is controlled with verbose arg.
result = future.get() # get the result from a finished job
```


### Jobs interface

List your jobs, optionally filtered by date, job type, and status.
```
session.jobs.list() # list all jobs
session.jobs.get(JOB_ID) # get a specific job
```

Resume an `AsyncJobFuture` from where you left off with each API's load_job:

For example for training jobs:

```
session.train.load_job(JOB_ID)
```
### PoET interface

Score sequences using the PoET interface.
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

Score single site variants using the PoET interface.
```
sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
future = session.poet.single_site(prompt, sequence, prompt_is_seed=True) 
result = future.wait()
# result is a dictionary of {variant: score}
```

Generate sequences from the PoET model.
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

Retrieve the prompt, MSA, or input (seed) sequences for a PoET job.
```
future.get_input(INPUT_TYPE)
# or, functions for each type
future.get_prompt()
future.get_msa()
future.get_seed()
```

See more at our [Homepage](https://docs.openprotein.ai/)