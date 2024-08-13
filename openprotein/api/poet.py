from typing import Iterator, Optional, List, Literal, Dict
from openprotein.pydantic import BaseModel, validator
from io import BytesIO
import random
import requests

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture, StreamingAsyncJobFuture
import numpy as np
from openprotein.jobs import ResultsParser, Job, register_job_type, JobType
import openprotein.config as config

from openprotein.errors import (
    InvalidParameterError,
    MissingParameterError,
    APIError,
)
from openprotein.api.align import csv_stream, AlignFutureMixin
from openprotein.futures import FutureBase, FutureFactory


class PoetSSPResult(BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str] = None
    _n: int = 0

    @validator("sequence", pre=True, always=True)
    def replacename(cls, value):
        """rename X0X"""
        if "X0X" in str(value):
            return b"WT"
        return value

    @validator("name", pre=True, always=True)
    def incrementname(cls, value):
        if value is None:
            cls._n += 1
            return f"Mutant{cls._n}"
        return value


class PoetScoreResult(BaseModel):
    sequence: bytes
    score: List[float]
    name: Optional[str] = None


@register_job_type(JobType.poet_score)
class PoetScoreJob(Job):
    parent_id: Optional[str] = None
    s3prefix: Optional[str] = None
    page_size: Optional[int] = None
    page_offset: Optional[int] = None
    num_rows: Optional[int] = None
    result: Optional[List[PoetScoreResult]] = None
    n_completed: Optional[int] = None

    job_type: Literal[JobType.poet_score] = JobType.poet_score


@register_job_type(JobType.poet_single_site)
class PoetSSPJob(PoetScoreJob):
    parent_id: Optional[str] = None
    s3prefix: Optional[str] = None
    page_size: Optional[int] = None
    page_offset: Optional[int] = None
    num_rows: Optional[int] = None
    result: Optional[List[PoetSSPResult]] = None
    n_completed: Optional[int] = None

    job_type: Literal[JobType.poet_single_site] = JobType.poet_single_site


@register_job_type(JobType.poet_generate)
class PoetGenerateJob(Job):
    parent_id: Optional[str] = None
    s3prefix: Optional[str] = None
    page_size: Optional[int] = None
    page_offset: Optional[int] = None
    num_rows: Optional[int] = None
    result: Optional[List[PoetScoreResult]] = None
    n_completed: Optional[int] = None

    job_type: Literal[JobType.poet_generate] = JobType.poet_generate


def poet_score_post(
    session: APISession, prompt_id: str, queries: List[bytes]
) -> FutureFactory:
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

    # return results to be assembled together
    return ResultsParser.parse_obj(response)


def poet_single_site_post(
    session: APISession, variant, parent_id=None, prompt_id=None
) -> FutureFactory:
    """
    Request PoET single-site analysis for a variant.

    This function will mutate every position in the variant to every amino acid and return the scores.
    Note that if parent_id is set then it will inherit all prompt properties of that parent.

    Parameters
    ----------
    session : APISession
        An instance of APISession for API interactions.
    variant : str
        The variant to analyze.
    parent_id : str, optional
        The ID of the parent job. Either parent_id or prompt_id must be set. Defaults to None.
    prompt_id : str, optional
        The ID of the prompt. Either parent_id or prompt_id must be set. Defaults to None.

    Raises
    ------
    APIError
        If the input parameters are invalid or there is an issue with the API request.

    Returns
    -------
    PoetSSPJob
        An object representing the status and results of the PoET single-site analysis job.
        Note that the input variant score is given as `X0X`.
    """
    endpoint = "v1/poet/single_site"

    if (parent_id is None and prompt_id is None) or (
        parent_id is not None and prompt_id is not None
    ):
        raise InvalidParameterError("Either parent_id or prompt_id must be set.")

    if isinstance(variant, str):
        variant = variant.encode()

    params = {"variant": variant}
    if prompt_id is not None:
        params["prompt_id"] = prompt_id
    if parent_id is not None:
        params["parent_id"] = parent_id

    try:
        response = session.post(endpoint, params=params)
        return FutureFactory.create_future(session=session, response=response)
    except Exception as exc:
        raise APIError(f"Failed to post poet single-site analysis: {exc}") from exc


def poet_single_site_get(
    session: APISession, job_id: str, page_size: int = 100, page_offset: int = 0
) -> FutureFactory:
    """
    Fetch paged results of a PoET single-site analysis job.

    Parameters
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job_id : str
        The ID of the PoET single-site analysis job to fetch results from.
    page_size : int, optional
        The number of results to fetch in a single page. Defaults to 100.
    page_offset : int, optional
        The offset (number of results) to start fetching results from. Defaults to 0.

    Raises
    ------
    APIError
        If there is an issue with the API request.

    Returns
    -------
    PoetSSPJob
        An object representing the status and results of the PoET single-site analysis job.
    """
    endpoint = "v1/poet/single_site"

    params = {"job_id": job_id, "page_size": page_size, "page_offset": page_offset}

    try:
        response = session.get(endpoint, params=params)

    except Exception as exc:
        raise APIError(
            f"Failed to get poet single-site analysis results: {exc}"
        ) from exc
    # return results to be assembled together
    return ResultsParser.parse_obj(response)


def poet_generate_post(
    session: APISession,
    prompt_id: str,
    num_samples=100,
    temperature=1.0,
    topk=None,
    topp=None,
    max_length=1000,
    random_seed=None,
) -> FutureFactory:
    """
    Generate protein sequences with a prompt.

    Parameters
    ----------
    session : APISession
        An instance of APISession for API interactions.
    prompt_id : str
        The ID of the prompt to generate samples from.
    num_samples : int, optional
        The number of samples to generate. Defaults to 100.
    temperature : float, optional
        The temperature for sampling. Higher values produce more random outputs. Defaults to 1.0.
    topk : int, optional
        The number of top-k residues to consider during sampling. Defaults to None.
    topp : float, optional
        The cumulative probability threshold for top-p sampling. Defaults to None.
    max_length : int, optional
        The maximum length of generated proteins. Defaults to 1000.
    random_seed : int, optional
        Seed for random number generation. Defaults to a random number.

    Raises
    ------
    APIError
        If there is an issue with the API request.

    Returns
    -------
    Job
        An object representing the status and information about the generation job.
    """
    endpoint = "v1/poet/generate"

    if not (0.1 <= temperature <= 2):
        raise InvalidParameterError("The 'temperature' must be between 0.1 and 2.")
    if topk:
        if not (2 <= topk <= 20):
            raise InvalidParameterError("The 'topk' must be between 2 and 22.")
    if topp:
        if not (0 <= topp <= 1):
            raise InvalidParameterError("The 'topp' must be between 0 and 1.")
    if random_seed:
        if not (0 <= random_seed <= 2**32):
            raise InvalidParameterError("The 'random_seed' must be between 0 and 1.")

    if random_seed is None:
        random_seed = random.randrange(2**32)

    params = {
        "prompt_id": prompt_id,
        "generate_n": num_samples,
        "temperature": temperature,
        "maxlen": max_length,
        "seed": random_seed,
    }
    if topk is not None:
        params["topk"] = topk
    if topp is not None:
        params["topp"] = topp

    try:
        response = session.post(endpoint, params=params)
        return FutureFactory.create_future(session=session, response=response)
    except Exception as exc:
        raise APIError(f"Failed to post PoET generation request: {exc}") from exc


def poet_generate_get(session: APISession, job_id) -> requests.Response:
    """
    Get the results of a PoET generation job.

    Parameters
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job_id : str
        Job ID from a poet/generate job.

    Raises
    ------
    APIError
        If there is an issue with the API request.

    Returns
    -------
    requests.Response
        The response object containing the results of the PoET generation job.
    """
    endpoint = "v1/poet/generate"

    params = {"job_id": job_id}

    try:
        response = session.get(endpoint, params=params, stream=True)
        return response
    except Exception as exc:
        raise APIError(f"Failed to get poet generation results: {exc}") from exc


class PoetFuture(AlignFutureMixin, AsyncJobFuture):
    def _fmt_results(self, results):
        # Format results after getting is complete
        return [(p.name, p.sequence, np.asarray(p.score)) for p in results]

    def get(self, verbose=False) -> List:
        return super().get(verbose=verbose)


class PoetScoreFuture(PoetFuture, FutureBase):
    """
    Represents a result of a PoET scoring job.

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
        Get the final results of the PoET  job.

    """

    job_type = ["/poet", "/poet/score"]

    def __init__(
        self, session: APISession, job: Job, page_size=config.POET_PAGE_SIZE, **kwargs
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

    def get(self, verbose=False) -> List[tuple]:
        """
        Get the final results of the PoET scoring job.

        Parameters
        ----------
        verbose : bool, optional
            If True, print verbose output. Defaults to False.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        List[PoetScoreResult]
            A list of PoetScoreResult objects representing the scoring results.
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
                return self._fmt_results(results)
        return self._fmt_results(results)


class PoetSingleSiteFuture(PoetFuture, FutureBase):
    """
    Represents a result of a PoET single-site analysis job.

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
        Get the final results of the PoET  job.

    """

    job_type = "/poet/single_site"

    def _fmt_results(self, results):
        # Format results after getting is complete
        return {p.sequence: np.asarray(p.score) for p in results}

    def __init__(
        self, session: APISession, job: Job, page_size=config.POET_PAGE_SIZE, **kwargs
    ):
        """
        init a PoetSingleSiteFuture instance.

        Parameters
        ----------
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET single-site analysis job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

    def get(self, verbose=False) -> Dict:
        """
        Get the results of a PoET single-site analysis job.

        Parameters
        ----------
        verbose : bool, optional
            If True, print verbose output. Defaults to False.

        Returns
        -------
        Dict[bytes, float]
            A dictionary mapping mutation codes to scores.

        Raises
        ------
        APIError
            If there is an issue with the API request.
        """

        job_id = self.job.job_id
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                response = poet_single_site_get(
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
                return self._fmt_results(results)
        return self._fmt_results(results)


class PoetGenerateFuture(PoetFuture, StreamingAsyncJobFuture, FutureBase):
    """
    Represents a result of a PoET generation job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.

    Methods:
        stream() -> Iterator[PoetScoreResult]:
            Stream the results of the PoET generation job.

    """

    job_type = "/poet/generate"

    def stream(self) -> Iterator[PoetScoreResult]:
        """
        Stream the results from the response.

        Returns
        ------
        PoetScoreResult: Yield
            A result object containing the sequence, score, and name.

        Raises
        ------
        APIError
            If the request fails.
        """
        try:
            response = poet_generate_get(self.session, self.job.job_id)
            for tokens in csv_stream(response):
                try:
                    name, sequence = tokens[:2]
                    score = [float(s) for s in tokens[2:]]
                    sequence = sequence.encode()
                    sample = PoetScoreResult(sequence=sequence, score=score, name=name)
                    yield self._fmt_results([sample])[0]
                except (IndexError, ValueError) as exc:
                    # Skip malformed or incomplete tokens
                    print(
                        f"Skipping malformed or incomplete tokens: {tokens} with {exc}"
                    )
        except APIError as exc:
            print(f"Failed to stream PoET generation results: {exc}")

    def get(self, verbose=False) -> List:
        return super().get(verbose=verbose)
