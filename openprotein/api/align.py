from typing import Iterator, Optional, List, BinaryIO, Literal, Union
from openprotein.pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from io import BytesIO
import random
import csv
import codecs
import requests

from openprotein.base import APISession
from openprotein.api.jobs import (
    AsyncJobFuture,
)

from openprotein.jobs import (
    ResultsParser,
    Job,
    register_job_type,
    JobType,
    job_args_get,
)

import openprotein.config as config

from openprotein.errors import (
    InvalidParameterError,
    MissingParameterError,
    APIError,
)
from openprotein.futures import FutureBase, FutureFactory


class PoetInputType(str, Enum):
    INPUT = "RAW"
    MSA = "GENERATED"
    PROMPT = "PROMPT"


class MSASamplingMethod(str, Enum):
    RANDOM = "RANDOM"
    NEIGHBORS = "NEIGHBORS"
    NEIGHBORS_NO_LIMIT = "NEIGHBORS_NO_LIMIT"
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = "NEIGHBORS_NONGAP_NORM_NO_LIMIT"
    TOP = "TOP"


class PromptPostParams(BaseModel):
    msa_id: str
    num_sequences: Optional[int] = Field(None, ge=0, lt=100)
    num_residues: Optional[int] = Field(None, ge=0, lt=24577)
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    homology_level: float = Field(0.8, ge=0, le=1)
    max_similarity: float = Field(1.0, ge=0, le=1)
    min_similarity: float = Field(0.0, ge=0, le=1)
    always_include_seed_sequence: bool = False
    num_ensemble_prompts: int = 1
    random_seed: Optional[int] = None


@register_job_type(JobType.align_align)
class MSAJob(Job):
    msa_id: Optional[str] = None
    job_type: Literal[JobType.align_align] = JobType.align_align

    @root_validator
    def set_msa_id(cls, values):
        if not values.get("msa_id"):
            values["msa_id"] = values.get("job_id")
        return values


@register_job_type(JobType.align_prompt)
class PromptJob(MSAJob):
    prompt_id: Optional[str] = None
    job_type: Literal[JobType.align_prompt] = JobType.align_prompt

    @root_validator
    def set_prompt_id(cls, values):
        if not values.get("prompt_id"):
            values["prompt_id"] = values.get("job_id")
        return values


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
    raw_content = response.raw  # the raw bytes stream
    content = codecs.getreader("utf-8")(
        raw_content
    )  # force the response to be encoded as utf-8
    return csv.reader(content)


def get_align_job_inputs(
    session: APISession,
    job_id,
    input_type: PoetInputType,
    prompt_index: Optional[int] = None,
) -> requests.Response:
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
    endpoint = "v1/align/inputs"

    params = {"job_id": job_id, "msa_type": input_type}
    if prompt_index is not None:
        params["replicate"] = prompt_index

    response = session.get(endpoint, params=params, stream=True)
    return response


def get_input(
    self: APISession,
    job: Job,
    input_type: PoetInputType,
    prompt_index: Optional[int] = None,
) -> csv.reader:
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


def get_prompt(
    self: APISession, job: Job, prompt_index: Optional[int] = None
) -> csv.reader:
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


def msa_post(session: APISession, msa_file=None, seed=None):
    """
    Create an MSA.

    Either via a seed sequence (which will trigger MSA creation) or a ready-to-use MSA (via msa_file).

    Note that seed and msa_file are mutually exclusive, and one or the other must be set.

    Parameters
    ----------
    session : APISession
        Authorized session.
    msa_file : str, optional
        Ready-made MSA. Defaults to None.
    seed : str, optional
        Seed to trigger MSA job. Defaults to None.

    Raises
    ------
    Exception
        If msa_file and seed are both None.

    Returns
    -------
    MSAJob
        Job details.
    """
    if (msa_file is None and seed is None) or (
        msa_file is not None and seed is not None
    ):
        raise MissingParameterError("seed OR msa_file must be provided.")
    endpoint = "v1/align/msa"

    is_seed = False
    if seed is not None:
        msa_file = BytesIO(b"\n".join([b">seed", seed]))
        is_seed = True

    params = {"is_seed": is_seed}
    files = {"msa_file": msa_file}

    response = session.post(endpoint, files=files, params=params)
    return FutureFactory.create_future(session=session, response=response)


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
) -> PromptJob:
    """
    Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.

    The MSA is specified by msa_id and created in msa_post.

    Parameters
    ----------
    session : APISession
        An instance of APISession to manage interactions with the API.
    msa_id : str
        The ID of the Multiple Sequence Alignment to use for the prompt.
    num_sequences : int, optional
        Maximum number of sequences in the prompt. Must be <100.
    num_residues : int, optional
        Maximum number of residues (tokens) in the prompt. Must be less than 24577.
    method : MSASamplingMethod, optional
        Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
    homology_level : float, optional
        Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
    max_similarity : float, optional
        Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
    min_similarity : float, optional
        Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
    always_include_seed_sequence : bool, optional
        Whether to always include the seed sequence in the MSA. Defaults to False.
    num_ensemble_prompts : int, optional
        Number of ensemble jobs to run. Defaults to 1.
    random_seed : int, optional
        Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

    Raises
    ------
    InvalidParameterError
        If provided parameter values are not in the allowed range.
    MissingParameterError
        If both or none of 'num_sequences', 'num_residues' is specified.

    Returns
    -------
    PromptJob
    """
    endpoint = "v1/align/prompt"

    if not (0 <= homology_level <= 1):
        raise InvalidParameterError("The 'homology_level' must be between 0 and 1.")
    if not (0 <= max_similarity <= 1):
        raise InvalidParameterError("The 'max_similarity' must be between 0 and 1.")
    if not (0 <= min_similarity <= 1):
        raise InvalidParameterError("The 'min_similarity' must be between 0 and 1.")

    if num_residues is None and num_sequences is None:
        num_residues = 12288

    if (num_sequences is None and num_residues is None) or (
        num_sequences is not None and num_residues is not None
    ):
        raise MissingParameterError(
            "Either 'num_sequences' or 'num_residues' must be set, but not both."
        )

    if num_sequences is not None and not (0 <= num_sequences < 100):
        raise InvalidParameterError("The 'num_sequences' must be between 0 and 100.")

    if num_residues is not None and not (0 <= num_residues < 24577):
        raise InvalidParameterError("The 'num_residues' must be between 0 and 24577.")

    if random_seed is None:
        random_seed = random.randrange(2**32)

    params = {
        "msa_id": msa_id,
        "msa_method": method,
        "homology_level": homology_level,
        "max_similarity": max_similarity,
        "min_similarity": min_similarity,
        "force_include_first": always_include_seed_sequence,
        "replicates": num_ensemble_prompts,
        "seed": random_seed,
    }
    if num_sequences is not None:
        params["max_msa_sequences"] = num_sequences
    if num_residues is not None:
        params["max_msa_tokens"] = num_residues

    response = session.post(endpoint, params=params)
    return FutureFactory.create_future(session=session, response=response)


def upload_prompt_post(
    session: APISession,
    prompt_file: BinaryIO,
):
    """
    Directly upload a prompt.

    Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
    You can specify multiple prompts (one per replicate) with an `<END_PROMPT>\n` between CSVs.

    Parameters
    ----------
    session : APISession
        An instance of APISession to manage interactions with the API.
    prompt_file : BinaryIO
        Binary I/O object representing the prompt file.

    Raises
    ------
    APIError
        If there is an issue with the API request.

    Returns
    -------
    PromptJob
        An object representing the status and results of the prompt job.
    """

    endpoint = "v1/align/upload_prompt"
    files = {"prompt_file": prompt_file}
    try:
        response = session.post(endpoint, files=files)
        return FutureFactory.create_future(session=session, response=response)
    except Exception as exc:
        raise APIError(f"Failed to upload prompt post: {exc}") from exc


def poet_score_post(session: APISession, prompt_id: str, queries: List[bytes]):
    """
    Submits a job to score sequences based on the given prompt.

    Parameters
    ----------
    session : APISession
        An instance of APISession to manage interactions with the API.
    prompt_id : str
        The ID of the prompt.
    queries : List[str]
        A list of query sequences to be scored.

    Raises
    ------
    APIError
        If there is an issue with the API request.

    Returns
    -------
    PoetScoreJob
        An object representing the status and results of the scoring job.
    """
    endpoint = "v1/poet/score"

    if len(queries) == 0:
        raise MissingParameterError("Must include queries for scoring!")
    if not prompt_id:
        raise MissingParameterError("Must include prompt_id in request!")

    if isinstance(queries[0], str):
        queries = [i.encode() for i in queries]
    try:
        variant_file = BytesIO(b"\n".join(queries))
        params = {"prompt_id": prompt_id}
        response = session.post(
            endpoint, files={"variant_file": variant_file}, params=params
        )
        return FutureFactory.create_future(session=session, response=response)
    except Exception as exc:
        raise APIError(f"Failed to post poet score: {exc}") from exc


def poet_score_get(
    session: APISession, job_id, page_size=config.POET_PAGE_SIZE, page_offset=0
):
    """
    Fetch a page of results from a PoET score job.

    Parameters
    ----------
    session : APISession
        An instance of APISession to manage interactions with the API.
    job_id : str
        The ID of the PoET scoring job to fetch results from.
    page_size : int, optional
        The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.
    page_offset : int, optional
        The offset (number of results) to start fetching results from. Defaults to 0.

    Raises
    ------
    APIError
        If the provided page size is larger than the maximum allowed page size.

    Returns
    -------
    PoetScoreJob
        An object representing the PoET scoring job, including its current status and results (if any).
    """
    endpoint = "v1/poet/score"

    if page_size > config.POET_MAX_PAGE_SIZE:
        raise APIError(
            f"Page size must be less than the max for PoET: {config.POET_MAX_PAGE_SIZE}"
        )

    response = session.get(
        endpoint,
        params={"job_id": job_id, "page_size": page_size, "page_offset": page_offset},
    )

    return FutureFactory.create_future(session=session, response=response)


class AlignFutureMixin:
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

    @property
    def id(self):
        return self.job.job_id


class MSAFuture(AlignFutureMixin, AsyncJobFuture, FutureBase):
    """
    Represents a result of a MSA job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.
    page_size : int
        The number of results to fetch in a single page.

    Methods
    -------
    get(verbose=False)
        Get the final results of the PoET scoring job.

    Returns
    -------
    List[PoetScoreResult]
        The list of results from the PoET scoring job.
    """

    job_type = "/align/align"

    def __init__(self, session: APISession, job: Job, page_size=config.POET_PAGE_SIZE):
        """
        init a PoetScoreFuture instance.

        Parameters
        ----------
        session : APISession
            An instance of APISession for API interactions.
        job : Job
            The PoET scoring job.
        page_size : int
            The number of results to fetch in a single page.

        """
        super().__init__(session, job)
        self.page_size = page_size
        self._msa_id = None
        self._prompt_id = None

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    @property
    def id(self):
        return self.job.job_id

    @property
    def prompt_id(self):
        if self.job.job_type == "/align/prompt" and self._prompt_id is None:
            self._prompt_id = self.job.job_id
        return self._prompt_id

    @property
    def msa_id(self):
        if self.job.job_type == "/align/align" and self._msa_id is None:
            self._msa_id = self.job.job_id
        return self._msa_id

    def wait(self, verbose: bool = False):
        _ = self.job.wait(
            self.session,
            interval=config.POLLING_INTERVAL,
            timeout=config.POLLING_TIMEOUT,
            verbose=False,
        )  # no progress to track
        return self.get()

    def get(self, verbose: bool = False) -> csv.reader:
        return self.get_msa()

    def sample_prompt(
        self,
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

        Parameters
        ----------
        num_sequences : int, optional
            Maximum number of sequences in the prompt. Must be  <100.
        num_residues : int, optional
            Maximum number of residues (tokens) in the prompt. Must be less than 24577.
        method : MSASamplingMethod, optional
            Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
        homology_level : float, optional
            Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
        max_similarity : float, optional
            Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
        min_similarity : float, optional
            Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
        always_include_seed_sequence : bool, optional
            Whether to always include the seed sequence in the MSA. Defaults to False.
        num_ensemble_prompts : int, optional
            Number of ensemble jobs to run. Defaults to 1.
        random_seed : int, optional
            Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

        Raises
        ------
        InvalidParameterError
            If provided parameter values are not in the allowed range.
        MissingParameterError
            If both or none of 'num_sequences', 'num_residues' is specified.

        Returns
        -------
        PromptJob
        """
        msa_id = self.msa_id
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


class PromptFuture(MSAFuture, FutureBase):
    """
    Represents a result of a prompt job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.
    page_size : int
        The number of results to fetch in a single page.

    Methods
    -------
    get(verbose=False)
        Get the final results of the PoET scoring job.

    Returns
    -------
    List[PoetScoreResult]
        The list of results from the PoET scoring job.
    """

    job_type = "/align/prompt"

    def __init__(
        self,
        session: APISession,
        job: Job,
        page_size=config.POET_PAGE_SIZE,
        msa_id: Optional[str] = None,
    ):
        """
        init a PoetScoreFuture instance.

        Parameters
        ----------
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET scoring job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

        if msa_id is None:
            msa_id = job_args_get(self.session, job.job_id).get("root_msa")
        self._msa_id = msa_id

    def get(self, verbose: bool = False) -> csv.reader:
        return self.get_prompt()


Prompt = Union[PromptFuture, str]


def validate_prompt(prompt: Prompt):
    """helper function to validate prompt_id is prompt type"""
    if not (isinstance(prompt, PromptFuture) or isinstance(prompt, str)):
        raise ValueError(
            f"Expect prompt to be either a PromptFuture or str, got {type(prompt)}"
        )
    if isinstance(prompt, str):
        return prompt
    return prompt.prompt_id


def validate_msa(msa: Union[MSAFuture, str]):
    """helper function to validate prompt_id is prompt type"""
    if not (isinstance(msa, MSAFuture) or isinstance(msa, str)):
        raise ValueError(
            f"Expect prompt to be either a MSAFuture or str, got {type(msa)}"
        )
    if isinstance(msa, str):
        return msa
    return msa.msa_id


class AlignAPI:
    """API interface for calling Poet and Align endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def upload_msa(self, msa_file) -> MSAFuture:
        """
        Upload an MSA from file.

        Parameters
        ----------
        msa_file : str, optional
            Ready-made MSA. If not provided, default value is None.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAJob
            Job object containing the details of the MSA upload.
        """
        return msa_post(self.session, msa_file=msa_file)

    def create_msa(self, seed: bytes) -> MSAFuture:
        """
        Construct an MSA via homology search with the seed sequence.

        Parameters
        ----------
        seed : bytes
            Seed sequence for the MSA construction.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAJob
            Job object containing the details of the MSA construction.
        """
        return msa_post(self.session, seed=seed)

    def upload_prompt(self, prompt_file: BinaryIO) -> Job:
        """
        Directly upload a prompt.

        Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
        You can specify multiple prompts (one per replicate) with an <END_PROMPT> and newline between CSVs.

        Parameters
        ----------
        prompt_file : BinaryIO
            Binary I/O object representing the prompt file.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        PromptJob
            An object representing the status and results of the prompt job.
        """
        return upload_prompt_post(self.session, prompt_file)

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
        return get_input(
            self.session, job, PoetInputType.PROMPT, prompt_index=prompt_index
        )

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
