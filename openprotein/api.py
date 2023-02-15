import requests
# from requests.auth import HTTPBasicAuth
from io import BytesIO
import pydantic
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np
import warnings


class OpenProtein:
    def __init__(self, username, password):
        session = requests.Session()
        session.auth = (username, password)
        session.verify = True
        self.session = session
        self.url_prefix = 'https://backend-dev.openprotein.ai'

    def prots2prot(self):
        return Prots2ProtAPI(self)


class Status(str, Enum):
    PENDING: str = 'PENDING'
    RUNNING: str = 'RUNNING'
    SUCCESS: str = 'SUCCESS'
    FAILURE: str = 'FAILURE'
    RETRYING: str = 'RETRYING'
    CANCELED: str = 'CANCELED'


class Job(pydantic.BaseModel):
    status: Status
    job_id: str
    job_type: str
    created_date: datetime
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    prerequisite_job_id: Optional[str]
    progress_message: Optional[str]
    progress_count: Optional[int]


def jobs_list(session: OpenProtein, status=None, job_type=None, assay_id=None, more_recent_than=None):
    url = session.url_prefix + '/api/v1/jobs'

    params = {}
    if status is not None:
        params['status'] = status
    if job_type is not None:
        params['job_type'] = job_type
    if assay_id is not None:
        params['assay_id'] = assay_id
    if more_recent_than is not None:
        params['more_recent_than'] = more_recent_than
    
    response = session.session.get(url, params=params)
    return pydantic.parse_obj_as(List[Job], response.json())


def job_get(session: OpenProtein, job_id):
    url = session.url_prefix + '/api/v1/jobs/' + job_id
    response = session.session.get(url)
    return Job(**response.json())


class Prots2ProtScoreResult(pydantic.BaseModel):
    sequence: bytes
    score: float
    name: Optional[str]


class Prots2ProtScoreJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[Prots2ProtScoreResult]]
    n_completed: Optional[int]


def prots2prot_score_post(session: OpenProtein, prompt, queries, prompt_is_seed=False):
    url = session.url_prefix + '/api/v1/workflow/prots2prot/score'

    msa_file = BytesIO(b'\n'.join(prompt))
    variant_file = BytesIO(b'\n'.join(queries))

    params = {'msa_is_seed': prompt_is_seed}

    response = session.session.post(
        url,
        files={'variant_file': variant_file, 'msa_file': msa_file},
        params=params,
    )
    return Prots2ProtScoreJob(**response.json())


def prots2prot_score_get(session: OpenProtein, job_id, page_size=1000, page_offset=0):
    url = session.url_prefix + '/api/v1/workflow/prots2prot/score'
    assert page_size <= 1000 # 1000 is the maximum page size...
    response = session.session.get(
        url,
        params={'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    )
    return Prots2ProtScoreJob(**response.json())


class Prots2ProtScoreFuture:
    PAGE_SIZE = 256

    def __init__(self, session: OpenProtein, job: Job, queries):
        self.session = session
        self.job = job
        self.queries = queries

    def update(self):
        self.job = job_get(self.session, self.job.job_id)

    def get(self):
        job_id = self.job.job_id
        step = self.PAGE_SIZE
        results = {}
        n_completed = 1
        offset = 0
        while len(results) < n_completed:
            response = prots2prot_score_get(
                self.session,
                job_id,
                page_offset=offset,
                page_size=step,
            )
            for r in response.result:
                results[r.sequence] = r.score
            #results += response.result
            offset += step

        # prots2prot de-duplicates the queries
        # so we need to match the results back against the queries
        # to return results exactly matched
        x = np.zeros(len(self.queries))
        for i in range(len(x)):
            x[i] = results[self.queries[i]]
        return x

    @property
    def status(self):
        return self.job.status

    def done(self):
        status = self.status
        return status == Status.CANCELED or status == Status.FAILURE or status == Status.SUCCESS
    



def prots2prot_single_site_post(session: OpenProtein, variant, parent_id=None, prompt=None):
    url = session.url_prefix + '/api/v1/workflow/prots2prot/single_site'

    files = None
    if prompt is not None:
        msa_file = BytesIO(b'\n'.join(prompt))
        files = {'msa_file': msa_file}
    
    params = {'variant': variant}
    if parent_id is not None:
        params['parent_id'] = parent_id

    response = session.session.post(
        url,
        params=params,
        files=files,
    )
    return response


def prots2prot_single_site_get(session: OpenProtein, job_id, page_size=100, page_offset=0):
    url = session.url_prefix + '/api/v1/workflow/prots2prot/single_site'

    params = {'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    response = session.session.get(url, params=params)
    return response


class Prots2ProtAPI:
    def __init__(self, session: OpenProtein):
        self.session = session

    def score(self, prompt, queries, prompt_is_seed=False):
        if prompt_is_seed and len(prompt) > 1:
            warnings.warn('When prompt_is_seed=True, only the first prompt sequence is used to build the expanded MSA.')
        elif not prompt_is_seed and len(prompt) == 1:
            warnings.warn('Prots2prot works best with more contextual sequences in the prompt. You passed one prompt sequence, but set prompt_is_seed=False. Set prompt_is_seed=True to expand the prompt via homology search.')
        response = prots2prot_score_post(self.session, prompt, queries, prompt_is_seed=prompt_is_seed)
        return Prots2ProtScoreFuture(self.session, response, queries)
