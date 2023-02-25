# openprotein-python
Python interface for the OpenProtein.AI REST API.

Work-in-progress. Currently supports basic exampination of jobs and running predictions for query sequences and single site variants with prots2prot. 

Each module has a raw/low level set of functions that directly call the REST API endpoints. On top of that, a higher level interface exists for authenticating a session, then accessing the functionality via high-level APIs. Long running POST/GET paradigm calls return a Future object that can be polled to see if the result is ready, and then used to retrieve the result. It also implements an interface for synchronously waiting for the result.

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
result = future.get() # get the result from a finished job
```

To wait for a job to finish and return the result, use `future.wait()`
```
# this will poll the backend for the job status every 2.5 seconds
# for up to 600 seconds. If it takes longer than 600 seconds for the result
# to be ready, this will raise a TimeoutException
# verbose=True will print the time elapsed and job status using `tqdm`
result = future.wait(interval=2.5, timeout=600, verbose=True)
```


### Jobs interface

List your jobs, optionally filtered by date, job type, and status.
```
session.jobs.list() # list all jobs
session.jobs.get(JOB_ID) # get a specific job
```

### Prots2prot interface

Score sequences using the prots2prot interface.
```
prompt = b'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN'
queries = [
    b'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
    b'MALWMRLLPLLVLLALWGPDPASAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
    b'MALWTRLRPLLALLALWPPPPARAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREVEGPQVGALELAGGPGAGGLEGPPQKRGIVEQCCASVCSLYQLENYCN',
    b'MALWIRSLPLLALLVFSGPGTSYAAANQHLCGSHLVEALYLVCGERGFFYSPKARRDVEQPLVSSPLRGEAGVLPFQQEEYEKVKRGIVEQCCHNTCSLYQLENYCN',
    b'MALWMRLLPLLALLALWAPAPTRAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREVEDLQVRDVELAGAPGEGGLQPLALEGALQKRGIVEQCCTSICSLYQLENYCN',
]
future = session.prots2prot.score(prompt, queries, prompt_is_seed=True)
result = future.wait()
# result is a list of (sequence, score) pydantic objects
```

Score single site variants using the prots2prot interface.
```
prompt = b'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN'
sequence = prompt
future = session.prots2prot.single_site(prompt, sequence, prompt_is_seed=True) 
result = future.wait()
# result is a dictionary of {variant: score}
```


## TODOs

- [] Error parsing for requests
- [] Better interface for creating async POST/GET requests
- [] Interfaces for other REST API modules
 
