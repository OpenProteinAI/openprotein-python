import io
import random
from typing import Iterator

import numpy as np
from openprotein.api.align import csv_stream
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import (
    AttnJob,
    EmbeddingsJob,
    GenerateJob,
    LogitsJob,
    ModelMetadata,
    ScoreJob,
    ScoreSingleSiteJob,
)
from pydantic import TypeAdapter

PATH_PREFIX = "v1/embeddings"


def list_models(session: APISession) -> list[str]:
    """
    List available embeddings models.

    Args:
        session (APISession): API session

    Returns:
        list[str]: list of model names.
    """

    endpoint = PATH_PREFIX + "/models"
    response = session.get(endpoint)
    result = response.json()
    return result


def get_model(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def get_request_sequences(session: APISession, job_id: str) -> list[bytes]:
    """
    Get results associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to fetch

    Returns
    -------
    sequences : List[bytes]
    """
    endpoint = PATH_PREFIX + f"/{job_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def request_get_sequence_result(
    session: APISession, job_id: str, sequence: str | bytes
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from
    sequence : bytes
        sequence to retrieve results for

    Returns
    -------
    result : bytes
    """
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def result_decode(data: bytes) -> np.ndarray:
    """
    Decode embedding.

    Args:
        data (bytes): raw bytes encoding the array received over the API

    Returns:
        np.ndarray: decoded array
    """
    s = io.BytesIO(data)
    return np.load(s, allow_pickle=False)


def request_get_score_result(session: APISession, job_id: str) -> Iterator[list[str]]:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from

    Returns
    -------
    csv.reader
    """
    endpoint = PATH_PREFIX + f"/{job_id}/scores"
    response = session.get(endpoint, stream=True)
    return csv_stream(response)


def request_get_generate_result(
    session: APISession, job_id: str
) -> Iterator[list[str]]:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from

    Returns
    -------
    csv.reader
    """
    endpoint = PATH_PREFIX + f"/{job_id}/generate"
    response = session.get(endpoint, stream=True)
    return csv_stream(response)


def request_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    reduction: str | None = "MEAN",
    prompt_id: str | None = None,
) -> EmbeddingsJob:
    """
    POST a request for embeddings from the given model ID. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for
    reduction : str | None
        reduction to apply to the embeddings. options are None, "MEAN", or "SUM". defaul: "MEAN"
    kwargs:
        accepts prompt_id for Poet Jobs

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/embed"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    if reduction is not None:
        body["reduction"] = reduction
    response = session.post(endpoint, json=body)
    return EmbeddingsJob.model_validate(response.json())


def request_logits_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    prompt_id: str | None = None,
) -> LogitsJob:
    """
    POST a request for logits from the given model ID. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/logits"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return LogitsJob.model_validate(response.json())


def request_attn_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    prompt_id: str | None = None,
) -> AttnJob:
    """
    POST a request for attention embeddings from the given model ID. \
        Returns a Job object referring to this request \
            that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/attn"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return AttnJob.model_validate(response.json())


def request_score_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    prompt_id: str | None = None,
) -> ScoreJob:
    """
    POST a request for sequence scoring for the given model ID. \
        Returns a Job object referring to this request \
            that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/score"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return ScoreJob.model_validate(response.json())


def request_score_single_site_post(
    session: APISession,
    model_id: str,
    base_sequence: bytes | str,
    prompt_id: str | None = None,
) -> ScoreSingleSiteJob:
    """
    POST a request for single site mutation scoring for the given model ID. \
        Returns a Job object referring to this request \
            that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from
    sequences : List[bytes]
        sequences to request results for

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/score_single_site"

    body: dict = {
        "base_sequence": (
            base_sequence.decode()
            if isinstance(base_sequence, bytes)
            else base_sequence
        ),
    }
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return ScoreSingleSiteJob.model_validate(response.json())


def request_generate_post(
    session: APISession,
    model_id: str,
    num_samples: int = 100,
    temperature: float = 1.0,
    topk: float | None = None,
    topp: float | None = None,
    max_length: int = 1000,
    random_seed: int | None = None,
    prompt_id: str | None = None,
) -> GenerateJob:
    """
    POST a request for sequence generation for the given model ID. \
        Returns a Job object referring to this request \
            that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model ID to request results from

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/generate"

    if not (0.1 <= temperature <= 2):
        raise InvalidParameterError("The 'temperature' must be between 0.1 and 2.")
    if topk is not None and not (2 <= topk <= 20):
        raise InvalidParameterError("The 'topk' must be between 2 and 20.")
    if topp is not None and not (0 <= topp <= 1):
        raise InvalidParameterError("The 'topp' must be between 0 and 1.")
    if random_seed is not None and not (0 <= random_seed <= 2**32):
        raise InvalidParameterError("The 'random_seed' must be between 0 and 2^32.")

    if random_seed is None:
        random_seed = random.randrange(2**32)

    body: dict = {
        "n_sequences": num_samples,
        "temperature": temperature,
        "maxlen": max_length,
    }
    if topk is not None:
        body["topk"] = topk
    if topp is not None:
        body["topp"] = topp
    if random_seed is not None:
        body["seed"] = random_seed
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
    response = session.post(endpoint, json=body)
    return GenerateJob.model_validate(response.json())
