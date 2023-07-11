from typing import Optional, List, Dict, Union, BinaryIO, Iterator
from io import BytesIO
import random
import csv
import codecs
import requests

from openprotein.base import APISession
from openprotein.api.jobs import Job, AsyncJobFuture, StreamingAsyncJobFuture, job_get
import openprotein.config as config

from .models import (MSASamplingMethod, PoetInputType, PoetScoreJob, PoetScoreResult, PromptJob, PoetSingleSiteJob, MSAJob)
from .errors import InvalidParameterError, MissingParameterError, APIError

def csv_stream(response: requests.Response) -> csv.reader:
    """
    Returns a CSV reader from a requests.Response object.

    Parameters
    ----------
    response : requests.Response
        The response object to parse.

    Returns
    -------
    csv.reader
        A csv reader object for the response.
    """
    raw_content = response.raw # the raw bytes stream
    content = codecs.getreader('utf-8')(raw_content)  # force the response to be encoded as utf-8
    return csv.reader(content)


def get_align_job_inputs(session: APISession,
                        job_id,
                        input_type: PoetInputType,
                        prompt_index: Optional[int] = None) -> requests.Response:
    """
    Get MSA and related data for an align job. 

    Returns either the original user seed (RAW), the generated MSA or the prompt.

    Specify prompt_index to retreive the specific prompt for each replicate when input_type is PROMPT. 

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : int or str
        The job identifier.
    input_type : PoetInputType
        The type of MSA data.
    prompt_index : Optional[int]
        The replicate number for the prompt (input_type=-PROMPT only)

    Returns
    -------
    requests.Response
        The response from the server.
    """
    endpoint = 'v1/poet/align/inputs'

    params = {'job_id': job_id, 'msa_type': input_type}
    if prompt_index is not None:
        params['replicate'] = prompt_index

    response = session.get(endpoint, params=params, stream=True)
    return response


def get_input(self: APISession, 
              job: Job,
              input_type: PoetInputType,
              prompt_index: Optional[int] = None) -> csv.reader:
    """
    Get input data for a given job.

    Parameters
    ----------
    self : APISession
        The API session.
    job : Job
        The job for which to retrieve data.
    input_type : PoetInputType
        The type of MSA data.
    prompt_index : Optional[int]
        The replicate number for the prompt (input_type=-PROMPT only)

    Returns
    -------
    csv.reader
        A CSV reader for the response data.
    """
    job_id = job.job_id
    response = get_align_job_inputs(self, job_id, input_type, prompt_index=prompt_index)
    return csv_stream(response)

def get_prompt(self: APISession, job: Job, prompt_index: Optional[int] = None) -> csv.reader:
    """
    Get the prompt for a given job.

    Parameters
    ----------
    self : APISession
        The API session.
    job : Job
        The job for which to retrieve the prompt.
    prompt_index : Optional[int], default=None
        The index of the prompt. If None, it returns all. 

    Returns
    -------
    csv.reader
        A CSV reader for the prompt data.
    """
    return get_input(self, job, PoetInputType.PROMPT, prompt_index=prompt_index)


def get_seed(self: APISession, job: Job) -> csv.reader:
    """
    Get the seed for a given MSA job.

    Parameters
    ----------
    self : APISession
        The API session.
    job : Job
        The job for which to retrieve the seed.

    Returns
    -------
    csv.reader
        A CSV reader for the seed sequence.
    """
    return get_input(self, job, PoetInputType.INPUT)


def get_msa(self: APISession, job: Job) -> csv.reader:
    """
    Get the generated MSA (Multiple Sequence Alignment) for a given job.

    Parameters
    ----------
    self : APISession
        The API session.
    job : Job
        The job for which to retrieve the MSA.

    Returns
    -------
    csv.reader
        A CSV reader for the MSA data.
    """
    return get_input(self, job, PoetInputType.MSA)


class PoetFutureMixin:
    session: APISession
    job: Job

    def get_input(self, input_type: PoetInputType):
        """See child function docs."""
        return get_input(self.session, self.job, input_type)

    def get_prompt(self, prompt_index: Optional[int] = None):
        """See child function docs."""
        return get_prompt(self.session, self.job, prompt_index=prompt_index)

    def get_seed(self):
        """See child function docs."""
        return get_seed(self.session, self.job)
    
    def get_msa(self):
        """See child function docs."""
        return get_msa(self.session, self.job)

def msa_post(session: APISession, msa_file=None, seed=None):
    """
    Create an MSA. 
    
    Either via a seed sequence (which will trigger MSA creation) or a ready-to-use MSA (via msa_file). 

    Note that seed and msa_file are mutually exclusive, and one or the other must be set

    Args:
        session (APISession): authorized session
        msa_file (str, optional): ready-made MSA. Defaults to None.
        seed (str, optional): Seed to trigger MSA job. Defaults to None.
    
    Raises:
        Exception: if msa_file and seed are both None. 

    Returns:
        MSAJob: Job details
    """
	
    if (msa_file is None and seed is None) or (msa_file is not None and seed is not None):
        raise MissingParameterError("seed OR msa_file must be provided.")
    endpoint = 'v1/poet/align/msa'

    is_seed = False
    if seed is not None:
        msa_file = BytesIO(b'\n'.join([b'>seed', seed]))
        is_seed = True

    params = {'is_seed': is_seed}
    files = {'msa_file': msa_file}

    response = session.post(endpoint, files=files, params=params)
    return MSAJob(**response.json())

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
        random_seed: Optional[int] = None
    ):
    """
    Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.

    The MSA is specified by msa_id and created in msa_post.
    
    Args:
        session (APISession): An instance of APISession to manage interactions with the API.
        msa_id (str): The ID of the Multiple Sequence Alignment to use for the prompt.
        num_sequences (int, optional): Maximum number of sequences in the prompt. Must be  <100.
        num_residues (int, optional): Maximum number of residues (tokens) in the prompt. Must be less than 24577.
        method (MSASamplingMethod, optional): Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
        homology_level (float, optional): Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
        max_similarity (float, optional): Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
        min_similarity (float, optional): Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
        always_include_seed_sequence (bool, optional): Whether to always include the seed sequence in the MSA. Defaults to False.
        num_ensemble_prompts (int, optional): Number of ensemble jobs to run. Defaults to 1.
        random_seed (int, optional): Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

    Raises:
        InvalidParameterError: If provided parameter values are not in the allowed range.
        MissingParameterError: If both or none of 'num_sequences', 'num_residues' is specified.

    Returns:
        PromptJob 
    """
    endpoint = 'v1/poet/align/prompt'

    if not (0 <= homology_level <= 1):
        raise InvalidParameterError("The 'homology_level' must be between 0 and 1.")
    if not (0 <= max_similarity <= 1):
        raise InvalidParameterError("The 'max_similarity' must be between 0 and 1.")
    if not (0 <= min_similarity <= 1):
        raise InvalidParameterError("The 'min_similarity' must be between 0 and 1.")

    if num_residues is None and num_sequences is None:
        num_residues = 12288

    if (num_sequences is None and num_residues is None) or (num_sequences is not None and num_residues is not None):
        raise MissingParameterError("Either 'num_sequences' or 'num_residues' must be set, but not both.")
    
    if num_sequences is not None and not (0 <= num_sequences < 100):
        raise InvalidParameterError("The 'num_sequences' must be between 0 and 100.")

    if num_residues is not None and not (0 <= num_residues < 24577):
        raise InvalidParameterError("The 'num_residues' must be between 0 and 24577.")

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

def upload_prompt_post(
        session: APISession,
        prompt_file: BinaryIO,
    ) -> PromptJob:
    """
    Directly upload a prompt.
    
    Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
    You can specify multiple prompts (one per replicate) with an `<END_PROMPT>\n` between CSVs. 
    
    Args:
        session (APISession): An instance of APISession to manage interactions with the API.
        prompt_file (BinaryIO): Binary I/O object representing the prompt file.
        
    Raises:
        APIError: If there is an issue with the API request.

    Returns:
        PromptJob: An object representing the status and results of the prompt job.
    """
    endpoint = 'v1/align/upload_prompt'

    try:
        body = {'prompt_file': prompt_file}
        response = session.post(endpoint, body=body)
        return PromptJob(**response.json())
    except Exception as exc:
        raise APIError(f"Failed to upload prompt post: {exc}") from exc

def poet_score_post(session: APISession, prompt_id: str, queries: List[str]) -> PoetScoreJob:
    """
    Submits a job to score sequences based on the given prompt.
    
    Args:
        session (APISession): An instance of APISession to manage interactions with the API.
        prompt_id (str): The ID of the prompt.
        queries (List[str]): A list of query sequences to be scored.

    Raises:
        APIError: If there is an issue with the API request.
        
    Returns:
        PoetScoreJob: An object representing the status and results of the scoring job.
    """
    endpoint = 'v1/poet/score'

    try:
        variant_file = BytesIO(b'\n'.join(queries))
        params = {'prompt_id': prompt_id}
        response = session.post(endpoint, files={'variant_file': variant_file}, params=params)
        return PoetScoreJob(**response.json())
    except Exception as exc:
        raise APIError(f"Failed to post poet score: {exc}") from exc



def poet_score_get(session: APISession, job_id, page_size=config.POET_PAGE_SIZE, page_offset=0):
    """
    Fetch a page of results from a PoET score job.

    Args:
        session (APISession): An instance of APISession to manage interactions with the API.
        job_id (str): The ID of the PoET scoring job to fetch results from.
        page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.
        page_offset (int, optional): The offset (number of results) to start fetching results from. Defaults to 0.

    Raises:
        APIError: If the provided page size is larger than the maximum allowed page size.

    Returns:
        PoetScoreJob: An object representing the PoET scoring job, including its current status and results (if any).
    """
    endpoint = 'v1/poet/score'

    if page_size > config.POET_MAX_PAGE_SIZE:
        raise APIError(f'Page size must be less than the max for PoET: {config.POET_MAX_PAGE_SIZE}')

    response = session.get(
        endpoint,
        params={'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}
    )

    return PoetScoreJob(**response.json())


class PoetScoreFuture(PoetFutureMixin, AsyncJobFuture):
    """
    Represents a result of a PoET scoring job.

    Attributes:
        session (APISession): An instance of APISession for API interactions.
        job (Job): The PoET scoring job.
        page_size (int): The number of results to fetch in a single page.

    Methods:
        get(verbose=False) -> List[PoetScoreResult]:
            Get the final results of the PoET scoring job.

    """

    def __init__(self, session: APISession, job: Job, page_size=config.POET_PAGE_SIZE):
        """
        Initialize a PoetScoreFuture instance.

        Args:
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET scoring job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

    def get(self, verbose=False) -> List[PoetScoreResult]:
        """
        Get the final results of the PoET scoring job.

        Args:
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            List[PoetScoreResult]: A list of PoetScoreResult objects representing the scoring results.
        """
        job_id = self.job.job_id
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                response = poet_score_get(
                    self.session,
                    job_id,
                    page_offset=offset,
                    page_size=step,
                )
                results += response.result
                num_returned = len(response.result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return results
        
        return results

def poet_single_site_post(session: APISession,
                          variant,
                          parent_id=None,
                          prompt_id=None) -> PoetSingleSiteJob:
    """
    Request PoET single-site analysis for a variant.

    Will mutate every position in the variant to every amino acid and return the scores.

    Note that if parent_id is set then will inherit all prompt properties of that parent.

    Args:
        session (APISession): An instance of APISession for API interactions.
        variant (str): The variant to analyze.
        parent_id (str, optional): The ID of the parent job. Either parent_id or prompt_id must be set. Defaults to None.
        prompt_id (str, optional): The ID of the prompt. Either parent_id or prompt_id must be set. Defaults to None.

    Raises:
        APIError: If the input parameters are invalid or there is an issue with the API request.

    Returns:
        PoetSingleSiteJob: An object representing the status and results of the PoET single-site analysis job.
        Note that the input variant score is given as `X0X`
    """
    endpoint = 'v1/poet/single_site'

    if (parent_id is None and prompt_id is None) or (parent_id is not None and prompt_id is not None):
        raise APIError('Either parent_id or prompt_id must be set.')

    params = {'variant': variant}
    if prompt_id is not None:
        params['prompt_id'] = prompt_id
    if parent_id is not None:
        params['parent_id'] = parent_id

    try:
        response = session.post(endpoint, params=params)
        return PoetSingleSiteJob(**response.json())
    except Exception as exc:
        raise APIError(f"Failed to post poet single-site analysis: {exc}") from exc

def poet_single_site_get(session: APISession,
                         job_id:str,
                         page_size:int=100,
                         page_offset:int=0) -> PoetSingleSiteJob:
    """
    Fetch paged results of a PoET single-site analysis job.

    Args:
        session (APISession): An instance of APISession for API interactions.
        job_id (str): The ID of the PoET single-site analysis job to fetch results from.
        page_size (int, optional): The number of results to fetch in a single page. Defaults to 100.
        page_offset (int, optional): The offset (number of results) to start fetching results from. Defaults to 0.

    Raises:
        APIError: If there is an issue with the API request.

    Returns:
        PoetSingleSiteJob: An object representing the status and results of the PoET single-site analysis job.
    """
    endpoint = 'v1/poet/single_site'

    params = {'job_id': job_id, 'page_size': page_size, 'page_offset': page_offset}

    try:
        response = session.get(endpoint, params=params)
        return PoetSingleSiteJob(**response.json())
    except Exception as exc:
        raise APIError(f"Failed to get poet single-site analysis results: {exc}") from exc

class PoetSingleSiteFuture(PoetFutureMixin, AsyncJobFuture):
    """
    Represents a result of a PoET single-site analysis job.

    Attributes:
        session (APISession): An instance of APISession for API interactions.
        job (Job): The PoET single-site analysis job.
        page_size (int): The number of results to fetch in a single page.

    Methods:
        get(verbose=False) -> Dict[bytes, float]:
            Get the final results of the PoET single-site analysis job.

    """

    def __init__(self, session: APISession, job: Job, page_size=config.POET_PAGE_SIZE):
        """
        Initialize a PoetSingleSiteFuture instance.

        Args:
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET single-site analysis job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

    def get(self, verbose=False) -> Dict[bytes, float]:
        """
        Get the results of a PoET single-site analysis job.

        Args:
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Returns:
            Dict[bytes, float]: A dictionary mapping mutation codes to scores.

        Raises:
            APIError: If there is an issue with the API request.

        """
        job_id = self.job.job_id
        step = self.page_size
        results = {}
        offset = 0
        num_returned = step

        while num_returned >= step:
            try:
                response = poet_single_site_get(
                    self.session,
                    job_id,
                    page_offset=offset,
                    page_size=step,
                )
                for r in response.result:
                    results[r.sequence] = r.score
                offset += step
                num_returned = len(response.result)
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return results
        
        return results

def poet_generate_post(
        session: APISession,
        prompt_id: str,
        num_samples=100,
        temperature=1.0,
        topk=None,
        topp=None,
        max_length=1000,
        random_seed=None,
    ) -> Job:
    """
    Generate protein sequences with a prompt. 

    Args:
        session (APISession): An instance of APISession for API interactions.
        prompt_id (str): The ID of the prompt to generate samples from.
        num_samples (int, optional): The number of samples to generate. Defaults to 100.
        temperature (float, optional): The temperature for sampling. Higher values produce more random outputs. Defaults to 1.0.
        topk (int, optional): The number of top-k residues to consider during sampling. Defaults to None.
        topp (float, optional): The cumulative probability threshold for top-p sampling. Defaults to None.
        max_length (int, optional): The maximum length of generated proteins. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to a random number.

    Raises:
        APIError: If there is an issue with the API request.

    Returns:
        Job: An object representing the status and information about the generation job.
    """
    endpoint = 'v1/poet/generate'

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

    try:
        response = session.post(endpoint, params=params)
        return Job(**response.json())
    except Exception as exc:
        raise APIError(f"Failed to post PoET generation request: {exc}") from exc



def poet_generate_get(session: APISession, job_id) -> requests.Response:
    """
    Get the results of a PoET generation job.

    Args:
        session (APISession): An instance of APISession for API interactions.
        job_id (str): job ID from a poet/generate job. 

    Raises:
        APIError: If there is an issue with the API request.

    Returns:
        requests.Response: The response object containing the results of the PoET generation job.
    """
    endpoint = 'v1/poet/generate'

    params = {'job_id': job_id}

    try:
        response = session.get(endpoint, params=params, stream=True)
        return response
    except Exception as exc:
        raise APIError(f"Failed to get poet generation results: {exc}") from exc


class PoetGenerateFuture(PoetFutureMixin, StreamingAsyncJobFuture):
    """
    Represents a result of a PoET generation job.

    Attributes:
        session (APISession): An instance of APISession for API interactions.
        job (Job): The PoET generation job.

    Methods:
        stream() -> Iterator[PoetScoreResult]:
            Stream the results of the PoET generation job.

    """

    def stream(self) -> Iterator[PoetScoreResult]:
        """
        Stream the results from the response.

        Yields:
            PoetScoreResult: A result object containing the sequence, score, and name.

        Raises:
            APIError: if request fails

        """
        try:
            response = poet_generate_get(self.session, self.job.job_id)
            for tokens in csv_stream(response):
                try:
                    name, sequence = tokens[:2]
                    score = [float(s) for s in tokens[2:]]
                    sequence = sequence.encode()
                    sample = PoetScoreResult(sequence=sequence, score=score, name=name)
                    yield sample
                except (IndexError, ValueError) as exc:
                    # Skip malformed or incomplete tokens
                    print(f"Skipping malformed or incomplete tokens: {tokens} with {exc}")
        except APIError as exc:
            print(f"Failed to stream PoET generation results: {exc}")


Prompt = Union[PromptJob, str]


def validate_prompt(prompt: Prompt):
    """ helper function to validate prompt_id is prompt type"""
    prompt_id = prompt
    if isinstance(prompt, PromptJob):
        prompt_id = prompt.prompt_id
    return prompt_id


class PoetAPI:
    """
    This class defines a high level interface for accessing PoET.
    See the subfunctons that are called for more detailed documentation.
    """
    def __init__(self, session: APISession):
        self.session = session

    def upload_msa(self, msa_file) -> MSAJob:
        """
        Upload an MSA from file. 

        Args:
            msa_file (str, optional): ready-made MSA. Defaults to None.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            MSAJob:
        """        
        return msa_post(self.session, msa_file=msa_file)

    def create_msa(self, seed: bytes) -> MSAJob:
        """
        Construct an MSA via homology search with the seed sequence.

        Args:
            seed (bytes): seed sequence

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            MSAJob
        """        

        return msa_post(self.session, seed=seed)

    def upload_prompt(self, prompt_file) -> PromptJob:
        """ 
        Directly upload a prompt.
        
        Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
        You can specify multiple prompts (one per replicate) with an `<END_PROMPT>\n` between CSVs. 
        
        Args:
            prompt_file (BinaryIO): Binary I/O object representing the prompt file.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            PromptJob: An object representing the status and results of the prompt job.
        """
        return upload_prompt_post(self.session, prompt_file)

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
        """
        Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.
        
        Args:
            msa (str): The msa Job to use in prompt creation.
            num_sequences (int, optional): Maximum number of sequences in the prompt. Must be  <100.
            num_residues (int, optional): Maximum number of residues (tokens) in the prompt. Must be less than 24577.
            method (MSASamplingMethod, optional): Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
            homology_level (float, optional): Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
            max_similarity (float, optional): Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
            min_similarity (float, optional): Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
            always_include_seed_sequence (bool, optional): Whether to always include the seed sequence in the MSA. Defaults to False.
            num_ensemble_prompts (int, optional): Number of ensemble jobs to run. Defaults to 1.
            random_seed (int, optional): Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

        Raises:
            InvalidParameterError: If provided parameter values are not in the allowed range.
            MissingParameterError: If both or none of 'num_sequences', 'num_residues' is specified.

        Returns:
            PromptJob 
        """
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

    def get_prompt(self, job: Job, prompt_index: Optional[int] = None) -> csv.reader:
        """
        Get prompts for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.
        prompt_index : Optional[int]
            The replicate number for the prompt (input_type=-PROMPT only)

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return get_input(self.session, job, PoetInputType.PROMPT, prompt_index=prompt_index)

    def get_seed(self, job: Job) -> csv.reader:
        """
        Get input data for a given msa job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return get_input(self.session, job, PoetInputType.INPUT)

    def get_msa(self, job: Job) -> csv.reader:
        """
        Get generated MSA for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return get_input(self.session, job, PoetInputType.MSA)

    def get_prompt_job(self, job_id: str) -> PromptJob:
        """
        Get prompt job based on job_id. 

        Args:
            job_id (str): job ID for a prompt job 

        Returns:
            PromptJob:
        """
        job = job_get(self.session, job_id)
        assert job.job_type == '/align/prompt'
        return PromptJob(**job.dict(), prompt_id=job.job_id)

    def get_msa_job(self, job_id: str) -> MSAJob:
        """
        Get MSA job based on job_id. 

        Args:
            job_id (str): job ID for a MSA job 

        Returns:
            MSAJob:
        """
        job = job_get(self.session, job_id)
        assert job.job_type == '/align/align'
        return MSAJob(**job.dict(), msa_id=job.job_id)

    def score(self, prompt: Prompt, queries: List[bytes]) -> PoetScoreFuture:
        """
        Score query sequences using the specified prompt.


        Args:
            prompt (Prompt): prompt job.
            queries (List[bytes]): sequences to score

        Returns:
            results
        """
        prompt_id = validate_prompt(prompt)
        response = poet_score_post(self.session, prompt_id, queries)
        return PoetScoreFuture(self.session, response)

    def single_site(self, prompt: Prompt, sequence: bytes) -> PoetSingleSiteFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.


        Args:
            prompt (Prompt): prompt job.
            sequence (bytes): sequence to mutate and score

        Returns:
            results
        """
        prompt_id = validate_prompt(prompt)
        response = poet_single_site_post(self.session, sequence, prompt_id=prompt_id)
        return PoetSingleSiteFuture(self.session, response)

    def generate(
            self,
            prompt: Prompt,
            num_samples=100,
            temperature=1.0,
            topk=None,
            topp=None,
            max_length=1000,
            seed=None,
        ) -> PoetGenerateFuture:
        """
            Generate protein sequences conditioned on a prompt.. 

            Args:
                prompt (PromptJob): The prompt to use.
                num_samples (int, optional): The number of samples to generate. Defaults to 100.
                temperature (float, optional): The temperature for sampling. Higher values produce more random outputs. Defaults to 1.0.
                topk (int, optional): The number of top-k residues to consider during sampling. Defaults to None.
                topp (float, optional): The cumulative probability threshold for top-p sampling. Defaults to None.
                max_length (int, optional): The maximum length of generated proteins. Defaults to 1000.
                seed (int, optional): Seed for random number generation. Defaults to a random number.

            Raises:
                APIError: If there is an issue with the API request.

            Returns:
                Job: An object representing the status and information about the generation job.
            """
        prompt_id = validate_prompt(prompt)
        job = poet_generate_post(
            self.session,
            prompt_id,
            num_samples=num_samples,
            temperature=temperature,
            topk=topk,
            topp=topp,
            max_length=max_length,
            random_seed=seed,
        )
        return PoetGenerateFuture(self.session, job)
