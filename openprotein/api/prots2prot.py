from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture
import openprotein.config as config

import pydantic
from typing import Optional, List, Dict, Union
from io import BytesIO
import warnings


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


def prots2prot_score_post(session: APISession, prompt, queries, prompt_is_seed=False):
    endpoint = 'v1/workflow/prots2prot/score'

    msa_file = BytesIO(b'\n'.join(prompt))
    variant_file = BytesIO(b'\n'.join(queries))

    params = {'msa_is_seed': prompt_is_seed}

    response = session.post(
        endpoint,
        files={'variant_file': variant_file, 'msa_file': msa_file},
        params=params,
    )
    return Prots2ProtScoreJob(**response.json())


def prots2prot_score_get(session: APISession, job_id, page_size=1000, page_offset=0):
    endpoint = 'v1/workflow/prots2prot/score'
    assert page_size <= 1000 # 1000 is the maximum page size...
    response = session.get(
        endpoint,
        params={'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    )
    return Prots2ProtScoreJob(**response.json()) 


class Prots2ProtScoreFuture(AsyncJobFuture):
    def __init__(self, session: APISession, job: Job, page_size=config.PROTS2PROT_PAGE_SIZE):
        super().__init__(session, job)
        self.page_size = page_size

    def get(self) -> List[Prots2ProtScoreResult]:
        job_id = self.job.job_id
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            response = prots2prot_score_get(
                self.session,
                job_id,
                page_offset=offset,
                page_size=step,
            )
            results += response.result
            num_returned = len(response.result)
            offset += num_returned
        
        return results


class Prots2ProtSiteResult(pydantic.BaseModel):
    sequence: bytes
    score: float
    name: Optional[str]


class Prots2ProtSingleSiteJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[Prots2ProtSiteResult]]
    #n_completed: Optional[int]


def prots2prot_single_site_post(session: APISession, variant, parent_id=None, prompt=None, prompt_is_seed=False):
    endpoint = 'v1/workflow/prots2prot/single_site'

    assert (parent_id is not None) or (prompt is not None), 'One of parent_id or prompt must be set.'
    assert not ((parent_id is not None) and (prompt is not None)), 'Both parent_id and prompt cannot be set.'

    files = None
    if prompt is not None:
        msa_file = BytesIO(b'\n'.join(prompt))
        files = {'msa_file': msa_file}
    
    params = {'variant': variant, 'msa_is_seed': prompt_is_seed}
    if parent_id is not None:
        params['parent_id'] = parent_id
    else:
        params['parent_id'] = ''

    response = session.post(
        endpoint,
        params=params,
        files=files,
    )
    return Prots2ProtSingleSiteJob(**response.json())


def prots2prot_single_site_get(session: APISession, job_id, page_size=100, page_offset=0):
    endpoint = 'v1/workflow/prots2prot/single_site'

    params = {'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    response = session.get(endpoint, params=params)

    return Prots2ProtSingleSiteJob(**response.json())


class Prots2ProtSingleSiteFuture(AsyncJobFuture):
    def __init__(self, session: APISession, job: Job, page_size=config.PROTS2PROT_PAGE_SIZE):
        super().__init__(session, job)
        self.page_size = page_size

    def get(self) -> Dict[bytes, float]:
        job_id = self.job.job_id
        step = self.page_size
        results = {}
        offset = 0
        num_returned = step
        while num_returned >= step:
            response = prots2prot_single_site_get(
                self.session,
                job_id,
                page_offset=offset,
                page_size=step,
            )
            for r in response.result:
                results[r.sequence] = r.score
            offset += step
            num_returned = len(response.result)
        
        return results


Prompt = Union[bytes, List[bytes]]


def validate_prompt(prompt: Prompt, prompt_is_seed):
    if type(prompt) is bytes:
        prompt = [prompt]
    if prompt_is_seed and len(prompt) > 1:
        warnings.warn('When prompt_is_seed=True, only the first prompt sequence is used to build the expanded MSA.')
    elif not prompt_is_seed and len(prompt) == 1:
        warnings.warn('Prots2prot works best with more contextual sequences in the prompt. You passed one prompt sequence, but set prompt_is_seed=False. Set prompt_is_seed=True to expand the prompt via homology search.')
    return prompt, prompt_is_seed


class Prots2ProtAPI:
    def __init__(self, session: APISession):
        self.session = session

    def score(self, prompt: Prompt, queries: List[bytes], prompt_is_seed=False):
        prompt, prompt_is_seed = validate_prompt(prompt, prompt_is_seed)
        response = prots2prot_score_post(self.session, prompt, queries, prompt_is_seed=prompt_is_seed)
        return Prots2ProtScoreFuture(self.session, response)

    def single_site(self, prompt: Union[Prompt, Job], sequence: bytes, prompt_is_seed=False):
        parent_id = None
        if isinstance(prompt, Job):
            parent_id = prompt.job_id
        else:
            prompt, prompt_is_seed = validate_prompt(prompt, prompt_is_seed=prompt_is_seed)

        response = prots2prot_single_site_post(self.session, sequence, prompt=prompt, parent_id=parent_id, prompt_is_seed=prompt_is_seed)
        return Prots2ProtSingleSiteFuture(self.session, response)
