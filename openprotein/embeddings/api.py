"""Embeddings REST API for making HTTP calls to our embeddings backend."""

import io
import random
import struct
from io import BytesIO
from typing import BinaryIO, Iterator

import numpy as np
from pydantic import TypeAdapter

from openprotein import csv
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.errors import InvalidParameterError

from .schemas import (
    AttnJob,
    EmbeddingsJob,
    GenerateJob,
    JobType,
    LogitsJob,
    ScoreIndelJob,
    ScoreJob,
    ScoreSingleSiteJob,
)

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


def get_request_sequences(
    session: APISession, job_id: str, job_type: JobType = JobType.embeddings_embed
) -> list[bytes]:
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
    # NOTE - allow to handle svd/embed and umap/embed directly too instead of redirect
    path = "v1" + job_type.value
    endpoint = path + f"/{job_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def request_get_sequence_result(
    session: APISession,
    job_id: str,
    sequence: str | bytes,
    job_type: JobType = JobType.embeddings_embed,
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
    # NOTE - allow to handle svd/embed and umap/embed directly too instead of redirect
    path = "v1" + job_type.value
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    endpoint = path + f"/{job_id}/{sequence}"
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
    return csv.parse_stream(response.iter_lines())


def request_get_embeddings_stream(
    session: APISession, job_id: str
) -> Iterator[np.ndarray]:
    """
    Stream back the raw embeddings for a given embeddings job.

    This will open an HTTP GET to `v1/embeddings/{job_id}/embeddings`
    with `stream=True`, then read a sequence of framed `.npy` payloads
    where each chunk is prefixed by an 8-byte big-endian length header.
    Each chunk is decoded into a NumPy array and yielded as soon as it’s
    received.

    Parameters
    ----------
    session : APISession
        The API session to use for making requests.
    job_id : str
        The embeddings job identifier returned by `request_post`.

    Yields
    ------
    numpy.ndarray
        An embedding array for each input sequence.

    Raises
    ------
    requests.HTTPError
        If the HTTP request returns a non‐2xx status code.
    ValueError
        If the framed stream is malformed (e.g. incomplete header or payload).
    """
    endpoint = PATH_PREFIX + f"/{job_id}/stream"
    response = session.get(endpoint, stream=True)
    response.raise_for_status()
    response.raw.decode_content = True
    buffered = io.BufferedReader(response.raw)  # type: ignore
    for array in parse_framed_npy_stream(buffered):
        yield array


def parse_framed_npy_stream(stream: BinaryIO) -> Iterator[np.ndarray]:
    """
    Read a binary stream of length‐prefixed NumPy .npy arrays.

    This function parses a stream composed of consecutive frames. Each frame
    starts with an 8‐byte big‐endian unsigned integer indicating the size of
    the subsequent .npy payload. It then reads exactly that many bytes and
    deserializes them into a NumPy array via np.load(…, allow_pickle=False).
    Frames are yielded one by one until the stream is exhausted.

    Parameters
    ----------
    stream : BinaryIO
        A binary stream supporting read(n) that contains zero or more
        concatenated frames in the format:
          [8‐byte big‐endian length][.npy payload].

    Yields
    ------
    np.ndarray
        Each deserialized NumPy array from the stream.

    Raises
    ------
    ValueError
        If an 8‐byte header cannot be read in full (unless at end of stream),
        or if a payload shorter than the declared length is encountered.
    """
    while True:
        # Read the 8-byte length header
        try:
            length_bytes = stream.read(8)
        except ValueError:
            # underlying file got closed → treat as EOF
            break
        if len(length_bytes) < 8:
            if length_bytes:
                raise ValueError("Incomplete length header")
            break  # End of stream

        (npy_len,) = struct.unpack(">Q", length_bytes)
        npy_bytes = stream.read(npy_len)
        if len(npy_bytes) < npy_len:
            raise ValueError("Incomplete npy payload")

        arr = np.load(BytesIO(npy_bytes), allow_pickle=False)
        yield arr


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
    return csv.parse_stream(response.iter_lines())


def request_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    reduction: str | None = "MEAN",
    **kwargs,
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
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/embed"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if reduction is not None:
        body["reduction"] = reduction
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if kwargs.get("decoder_type"):
        body["decoder_type"] = kwargs["decoder_type"]
    response = session.post(endpoint, json=body)
    return EmbeddingsJob.model_validate(response.json())


def request_logits_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    **kwargs,
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
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/logits"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if kwargs.get("decoder_type"):
        body["decoder_type"] = kwargs["decoder_type"]
    response = session.post(endpoint, json=body)
    return LogitsJob.model_validate(response.json())


def request_attn_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    **kwargs,
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
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/attn"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    response = session.post(endpoint, json=body)
    return AttnJob.model_validate(response.json())


def request_score_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str],
    **kwargs,
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
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if kwargs.get("decoder_type"):
        body["decoder_type"] = kwargs["decoder_type"]
    response = session.post(endpoint, json=body)
    return ScoreJob.model_validate(response.json())


def request_score_indel_post(
    session: APISession,
    model_id: str,
    base_sequence: bytes | str,
    insert: str | None = None,
    delete: list[int] | None = None,
    **kwargs,
) -> ScoreIndelJob:
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
    insert: str | None
        Insertion fragment at each site.
    delete: int | None
        Range of size of fragment to delete at each site.
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}/score/indel"

    body: dict = {
        "base_sequence": (
            base_sequence.decode()
            if isinstance(base_sequence, bytes)
            else base_sequence
        ),
    }
    if insert is not None:
        body["insert"] = insert
    if delete is not None:
        body["delete"] = delete
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if kwargs.get("decoder_type"):
        body["decoder_type"] = kwargs["decoder_type"]
    response = session.post(endpoint, json=body)
    return ScoreIndelJob.model_validate(response.json())


def request_score_single_site_post(
    session: APISession,
    model_id: str,
    base_sequence: bytes | str,
    **kwargs,
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
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

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
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if kwargs.get("decoder_type"):
        body["decoder_type"] = kwargs["decoder_type"]
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
    **kwargs,
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
    **kwargs:
        Optional parameters for models, e.g. prompt_id for PoET

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
    if kwargs.get("prompt_id"):
        body["prompt_id"] = kwargs["prompt_id"]
    if kwargs.get("query_id"):
        assert model_id != "poet", f"Model with id {model_id} does not support query"
        body["query_id"] = kwargs["query_id"]
        if "use_query_structure_in_decoder" in kwargs:
            body["use_query_structure_in_decoder"] = kwargs[
                "use_query_structure_in_decoder"
            ]
    if (ensemble_weights := kwargs.get("ensemble_weights")) is not None:
        assert (
            model_id != "poet"
        ), f"Model with id {model_id} does not support ensemble_weights parameter"
        body["ensemble_weights"] = list(ensemble_weights)
    if (ensemble_method := kwargs.get("ensemble_method")) is not None:
        assert (
            model_id != "poet"
        ), f"Model with id {model_id} does not support ensemble_method parameter"
        body["ensemble_method"] = ensemble_method
    response = session.post(endpoint, json=body)
    return GenerateJob.model_validate(response.json())
