from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

import pydantic
from enum import Enum
from typing import Optional, List, Dict, Union
from io import BytesIO
import random
import csv
import codecs
import requests


def csv_stream(response: requests.Response):
    raw_content = response.raw # the raw bytes stream

    # force the response to be encoded as utf-8
    # NOTE - this isn't ideal, as the response could be encoded differently in the future
    # but, the csv parser requires str not bytes
    content = codecs.getreader('utf-8')(raw_content)
    return csv.reader(content)


class Prots2ProtInputType(str, Enum):
    INPUT = 'RAW'
    MSA = 'GENERATED'
    PROMPT = 'PROMPT'


def get_prots2prot_job_inputs(session: APISession, job_id, input_type: Prots2ProtInputType, prompt_index: Optional[int] = None):
    endpoint = 'v1/workflow/align/inputs'

    params = {'job_id': job_id, 'msa_type': input_type}
    if prompt_index is not None:
        params['replicate'] = prompt_index
    response = session.get(endpoint, params=params, stream=True)

    return response


def get_input(self: APISession, job: Job, input_type: Prots2ProtInputType, prompt_index: Optional[int] = None):
    job_id = job.job_id
    response = get_prots2prot_job_inputs(self, job_id, input_type, prompt_index=prompt_index)
    return csv_stream(response)


def get_prompt(self: APISession, job: Job, prompt_index: Optional[int] = None):
    return get_input(self, job, Prots2ProtInputType.PROMPT, prompt_index=prompt_index)


def get_seed(self: APISession, job: Job):
    return get_input(self, job, Prots2ProtInputType.INPUT)


def get_msa(self: APISession, job: Job):
    return get_input(self, job, Prots2ProtInputType.MSA)


class Prots2ProtFutureMixin:
    session: APISession
    job: Job

    def get_input(self, input_type: Prots2ProtInputType):
        return get_input(self.session, self.job, input_type)

    def get_prompt(self, prompt_index: Optional[int] = None):
        return get_prompt(self.session, self.job, prompt_index=prompt_index)

    def get_seed(self):
        return get_seed(self.session, self.job)
    
    def get_msa(self):
        return get_msa(self.session, self.job)


class MSAJob(Job):
    msa_id: str


def msa_post(session: APISession, msa_file=None, seed=None):
    endpoint = 'v1/workflow/align/msa'
    assert msa_file is not None or seed is not None, 'One of msa_file or seed must be set'
    assert msa_file is None or seed is None, 'Both msa_file and seed cannot be set'

    is_seed = False
    if seed is not None:
        msa_file = BytesIO(b'\n'.join([b'>seed', seed]))
        is_seed = True

    params = {'is_seed': is_seed}
    files = {'msa_file': msa_file}

    response = session.post(endpoint, files=files, params=params)
    return MSAJob(**response.json())


class MSASamplingMethod(str, Enum):
    RANDOM = 'RANDOM'
    NEIGHBORS = 'NEIGHBORS'
    NEIGHBORS_NO_LIMIT = 'NEIGHBORS_NO_LIMIT'
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = 'NEIGHBORS_NONGAP_NORM_NO_LIMIT'
    TOP = 'TOP'


class PromptJob(Job):
    prompt_id: str


def prompt_post(
        session: APISession,
        msa_id: str,
        num_sequences: Optional[int] = None,
        num_residues: Optional[int] = None,
        method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
        homology_level: float = 0.8,
        max_similarity: float = 1.0,
        min_similarity: float = 0.0,
        always_include_seed_sequence: bool = False,
        num_ensemble_prompts: int = 1,
        random_seed: Optional[int] = None,
    ):
    endpoint = 'v1/workflow/align/prompt'

    assert 0 <= homology_level and homology_level <= 1
    assert 0 <= max_similarity and max_similarity <= 1
    assert 0 <= min_similarity and min_similarity <= 1

    if num_residues is None and num_sequences is None:
        num_residues = 12288

    assert (num_sequences is not None) or (num_residues is not None), 'One of num_sequences or num_tokens must be set'
    assert (num_sequences is None) or (num_residues is None), 'Both num_sequences and num_tokens cannot be set'
    
    if num_sequences is not None:
        assert 0 <= num_sequences < 100

    if num_residues is not None:
        assert 0 <= num_residues < 24577

    if random_seed is None:
        random_seed = random.randrange(2**32)

    params = {
        'msa_id': msa_id,
        'msa_method': method,
        'homology_level': homology_level,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity,
        'force_include_first': always_include_seed_sequence,
        'replicates': num_ensemble_prompts,
        'seed': random_seed,
    }
    if num_sequences is not None:
        params['max_msa_sequences'] = num_sequences
    if num_residues is not None:
        params['max_msa_tokens'] = num_residues

    response = session.post(endpoint, params=params)
    return PromptJob(**response.json())


class Prots2ProtScoreResult(pydantic.BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str]


class Prots2ProtScoreJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[Prots2ProtScoreResult]]
    n_completed: Optional[int]


def prots2prot_score_post(session: APISession, prompt_id: str, queries):
    endpoint = 'v1/workflow/prots2prot/score'

    variant_file = BytesIO(b'\n'.join(queries))

    params = {'prompt_id': prompt_id}

    response = session.post(
        endpoint,
        files={'variant_file': variant_file},
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


class Prots2ProtScoreFuture(Prots2ProtFutureMixin, AsyncJobFuture):
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
    score: List[float]
    name: Optional[str]


class Prots2ProtSingleSiteJob(Job):
    parent_id: Optional[str]
    s3prefix: Optional[str]
    page_size: Optional[int]
    page_offset: Optional[int]
    num_rows: Optional[int]
    result: Optional[List[Prots2ProtSiteResult]]
    #n_completed: Optional[int]


def prots2prot_single_site_post(session: APISession, variant, parent_id=None, prompt_id=None):
    endpoint = 'v1/workflow/prots2prot/single_site'

    assert (parent_id is not None) or (prompt_id is not None), 'One of parent_id or prompt_id must be set.'
    assert not ((parent_id is not None) and (prompt_id is not None)), 'Both parent_id and prompt_id cannot be set.'
    
    params = {'variant': variant}
    if prompt_id is not None:
        params['prompt_id'] = prompt_id
    if parent_id is not None:
        params['parent_id'] = parent_id
    #else:
    #    params['parent_id'] = ''

    response = session.post(
        endpoint,
        params=params,
    )
    return Prots2ProtSingleSiteJob(**response.json())


def prots2prot_single_site_get(session: APISession, job_id, page_size=100, page_offset=0):
    endpoint = 'v1/workflow/prots2prot/single_site'

    params = {'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    response = session.get(endpoint, params=params)

    return Prots2ProtSingleSiteJob(**response.json())


class Prots2ProtSingleSiteFuture(Prots2ProtFutureMixin, AsyncJobFuture):
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
        prompt_id: str,
        num_samples=100,
        temperature=1.0,
        topk=None,
        topp=None,
        max_length=1000,
        random_seed=None,
    ) -> Job:
    endpoint = 'v1/workflow/prots2prot/generate'

    if random_seed is None:
        random_seed = random.randrange(2**32)

    params = {
        'prompt_id': prompt_id,
        'generate_n': num_samples,
        'temperature': temperature,
        'maxlen': max_length,
        'seed': random_seed,
    }
    if topk is not None:
        params['topk'] = topk
    if topp is not None:
        params['topp'] = topp

    response = session.post(
        endpoint,
        params=params,
    )
    return Job(**response.json())


def prots2prot_generate_get(session: APISession, job_id):
    endpoint = 'v1/workflow/prots2prot/generate'

    params = {'job_id': job_id}
    response = session.get(endpoint, params=params, stream=True)

    return response


class Prots2ProtGenerateFuture(Prots2ProtFutureMixin, StreamingAsyncJobFuture):
    def stream(self):
        """
        Yield results from the response stream.
        """
        response = prots2prot_generate_get(self.session, self.job.job_id)
        for tokens in csv_stream(response):
            name, sequence = tokens[:2]
            score = [float(s) for s in tokens[2:]]
            sequence = sequence.encode() # tokens are string type, but we encode sequences as bytes type
            sample = Prots2ProtScoreResult(sequence=sequence, score=score, name=name)
            yield sample


Prompt = Union[PromptJob, str]


def validate_prompt(prompt: Prompt):
    prompt_id = prompt
    if isinstance(prompt, PromptJob):
        prompt_id = prompt.prompt_id
    return prompt_id


class Prots2ProtAPI:
    def __init__(self, session: APISession):
        self.session = session

    def upload_msa(self, msa_file) -> MSAJob:
        return msa_post(self.session, msa_file=msa_file)

    def create_msa(self, seed: bytes) -> MSAJob:
        return msa_post(self.session, seed=seed)

    def sample_prompt(
            self,
            msa: Union[MSAJob, str],
            num_sequences: Optional[int] = None,
            num_residues: Optional[int] = None,
            method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
            homology_level: float = 0.8,
            max_similarity: float = 1.0,
            min_similarity: float = 0.0,
            always_include_seed_sequence: bool = False,
            num_ensemble_prompts: int = 1,
            random_seed: Optional[int] = None,
        ) -> PromptJob:
        msa_id = msa
        if isinstance(msa, MSAJob):
            msa_id = msa.msa_id
        return prompt_post(
            self.session,
            msa_id,
            num_sequences=num_sequences,
            num_residues=num_residues,
            method=method,
            homology_level=homology_level,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            always_include_seed_sequence=always_include_seed_sequence,
            num_ensemble_prompts=num_ensemble_prompts,
            random_seed=random_seed,
        )

    def get_prompt(self, job: Job, prompt_index: Optional[int] = None):
        return get_input(self.session, job, Prots2ProtInputType.PROMPT, prompt_index=prompt_index)

    def get_seed(self, job: Job):
        return get_input(self.session, job, Prots2ProtInputType.INPUT)

    def get_msa(self, job: Job):
        return get_input(self.session, job, Prots2ProtInputType.MSA)

    def get_prompt_job(self, job_id: str) -> PromptJob:
        job = job_get(self.session, job_id)
        assert job.job_type == 'workflow/align/prompt'
        return PromptJob(**job.dict(), prompt_id=job.job_id)

    def get_msa_job(self, job_id: str) -> MSAJob:
        job = job_get(self.session, job_id)
        assert job.job_type == 'workflow/align/align'
        return MSAJob(**job.dict(), msa_id=job.job_id)

    def score(self, prompt: Prompt, queries: List[bytes]):
        prompt_id = validate_prompt(prompt)
        response = prots2prot_score_post(self.session, prompt_id, queries)
        return Prots2ProtScoreFuture(self.session, response)

    def single_site(self, prompt: Prompt, sequence: bytes):
        prompt_id = validate_prompt(prompt)
        response = prots2prot_single_site_post(self.session, sequence, prompt_id=prompt_id)
        return Prots2ProtSingleSiteFuture(self.session, response)

    def generate(
            self,
            prompt: Prompt,
            num_samples=100,
            temperature=1.0,
            topk=None,
            topp=None,
            max_length=1000,
        ):
        prompt_id = validate_prompt(prompt)
        job = prots2prot_generate_post(
            self.session,
            prompt_id,
            num_samples=num_samples,
            temperature=temperature,
            topk=topk,
            topp=topp,
            max_length=max_length,
        )
        return Prots2ProtGenerateFuture(self.session, job)
