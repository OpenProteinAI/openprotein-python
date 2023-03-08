from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture
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


def prots2prot_generate_post(
        session: APISession,
        prompt=None,
        prompt_is_seed=False,
        msa_sampling_method='NEIGHBORS_NONGAP_NORM_NO_LIMIT',
        max_seqs_from_msa=33,
        parent_id=None,
        num_samples=100,
        temperature=1.0,
        topk=None,
        topp=None,
        max_length=1000,
    ) -> Job:
    endpoint = 'v1/workflow/prots2prot/generate'

    assert (parent_id is not None) or (prompt is not None), 'One of parent_id or prompt must be set.'
    assert not ((parent_id is not None) and (prompt is not None)), 'Both parent_id and prompt cannot be set.'

    files = None
    if prompt is not None:
        msa_file = BytesIO(b'\n'.join(prompt))
        files = {'msa_file': msa_file}
    
    params = {
        'msa_is_seed': prompt_is_seed,
        'max_msa': max_seqs_from_msa,
        'msa_method': msa_sampling_method,
        'generate_n': num_samples,
        'temperature': temperature,
        'maxlen': max_length,
    }
    if parent_id is not None:
        params['parent_id'] = parent_id
    if topk is not None:
        params['topk'] = topk
    if topp is not None:
        params['topp'] = topp

    response = session.post(
        endpoint,
        params=params,
        files=files,
    )
    return Job(**response.json())


def prots2prot_generate_get(session: APISession, job_id):
    endpoint = 'v1/workflow/prots2prot/generate'

    params = {'job_id': job_id}
    response = session.get(endpoint, params=params, stream=True)

    return response


class Prots2ProtGenerateFuture(StreamingAsyncJobFuture):
    def stream(self):
        """
        Yield results from the response stream.
        """
        response = prots2prot_generate_get(self.session, self.job.job_id)
        for line in response.iter_lines():
            name, sequence, score = line.split(b',')
            score = float(score)
            name = name.decode()
            sample = Prots2ProtScoreResult(sequence=sequence, score=score, name=name)
            yield sample


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

    def generate(
            self,
            prompt: Union[Prompt, Job],
            prompt_is_seed=False,
            msa_sampling_method='NEIGHBORS_NONGAP_NORM_NO_LIMIT',
            max_seqs_from_msa=33,
            num_samples=100,
            temperature=1.0,
            topk=None,
            topp=None,
            max_length=1000,
        ):
        parent_id = None
        if isinstance(prompt, Job):
            parent_id = prompt.job_id
        else:
            prompt, prompt_is_seed = validate_prompt(prompt, prompt_is_seed=prompt_is_seed)
        
        job = prots2prot_generate_post(
            self.session,
            prompt=prompt,
            prompt_is_seed=prompt_is_seed,
            parent_id=parent_id,
            msa_sampling_method=msa_sampling_method,
            max_seqs_from_msa=max_seqs_from_msa,
            num_samples=num_samples,
            temperature=temperature,
            topk=topk,
            topp=topp,
            max_length=max_length,
        )
        return Prots2ProtGenerateFuture(self.session, job)
