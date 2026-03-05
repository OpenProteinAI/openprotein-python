"""Fold REST API interface for making HTTP calls to our fold backend."""

import io
import typing
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.errors import HTTPError

from .schemas import FoldJob, FoldMetadata

if TYPE_CHECKING:
    import pandas as pd

PATH_PREFIX = "v1/fold"


def fold_models_list_get(session: APISession) -> list[str]:
    """
    List available fold models.

    Parameters
    ----------
    session : APISession
        API session.

    Returns
    -------
    list of str
        List of model names.
    """
    endpoint = PATH_PREFIX + "/models"
    response = session.get(endpoint)
    result = response.json()
    return result


def fold_model_get(session: APISession, model_id: str) -> ModelMetadata:
    """
    Get metadata for a specific fold model.

    Parameters
    ----------
    session : APISession
        API session.
    model_id : str
        Model ID to fetch.

    Returns
    -------
    ModelMetadata
        Metadata for the specified model.
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def fold_get(session: APISession, job_id: str) -> FoldMetadata:
    """
    Get metadata associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Fold ID to fetch.

    Returns
    -------
    FoldMetadata
        Metadata about the fold job.
    """
    endpoint = PATH_PREFIX + f"/{job_id}"
    response = session.get(endpoint)
    fold = FoldMetadata.model_validate(response.json())
    return fold


def fold_get_sequences(session: APISession, job_id: str) -> list[bytes]:
    """
    Get sequences associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to fetch.

    Returns
    -------
    list of bytes
        List of sequences as bytes.
    """
    endpoint = PATH_PREFIX + f"/{job_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def fold_get_sequence_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    format: str = "mmcif",
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from.
    sequence_or_index : bytes or str or int
        Sequence to retrieve results for or its index in the job.

    Returns
    -------
    bytes
        Encoded result for the sequence.
    """
    if isinstance(sequence_or_index, bytes):
        sequence_or_index = sequence_or_index.decode()
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence_or_index}"
    response = session.get(endpoint, params={"format": format})
    return response.content


@typing.overload
def fold_get_extra_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    key: Literal["pae", "pde", "plddt", "ptm"],
) -> np.ndarray: ...


@typing.overload
def fold_get_extra_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    key: Literal["confidence"],
) -> list[dict]: ...


@typing.overload
def fold_get_extra_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    key: Literal["affinity"],
) -> dict: ...


@typing.overload
def fold_get_extra_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    key: Literal["score", "metrics"],
) -> pd.DataFrame: ...


def fold_get_extra_result(
    session: APISession,
    job_id: str,
    sequence_or_index: bytes | str | int,
    key: Literal[
        "pae", "pde", "plddt", "ptm", "confidence", "affinity", "score", "metrics"
    ],
) -> "np.ndarray | list[dict] | dict | pd.DataFrame":
    """
    Get extra result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from.
    sequence_or_index : bytes or str or int
        Sequence to retrieve results for or its index in the job.
    key : {'pae', 'pde', 'plddt', 'ptm', 'confidence', 'affinity', 'score', 'metrics'}
        The type of result to retrieve.

    Returns
    -------
    numpy.ndarray or list of dict
        The result as a numpy array (for "pae", "pde", "plddt") or a list of dictionaries (for "confidence", "affinity").
    """
    if key in {"pae", "pde", "plddt", "ptm"}:

        def formatter(response):
            return np.load(io.BytesIO(response.content))
    elif key in {"confidence", "affinity"}:

        def formatter(response):
            return response.json()
    elif key in {"score", "metrics"}:
        import pandas as pd

        def formatter(response):
            return pd.read_csv(io.StringIO(response.content.decode()))
    else:
        raise ValueError(f"Unexpected key: {key}")
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence_or_index}/{key}"
    try:
        response = session.get(
            endpoint,
        )
    except HTTPError as e:
        if e.status_code == 400 and key == "affinity":
            raise ValueError("affinity not found for request") from None
        raise e
    output = formatter(response)
    return output


def fold_models_post(
    session: APISession,
    model_id: str,
    sequences: Sequence[Sequence[Mapping[str, Any]]],
    **kwargs,
) -> FoldJob:
    """
    POST a request for structure prediction.

    Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        Model ID to use for prediction.
    sequences : sequence of sequence of dict
        Sequences/complexes to request results for.
        The outer list represents the batch of requests, and the inner
        list represents the complex, with each item in the list being
        an entity in that complex. A monomer would thus be a single item.
    **kwargs
        Additional keyword arguments to be sent with POST body.

    Returns
    -------
    FoldJob
        Job object referring to this request.
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}"

    body: dict = {
        "sequences": sequences,
    }
    # add non-None args
    for k, v in kwargs.items():
        if v is not None:
            body[k] = v

    response = session.post(endpoint, json=body)
    return FoldJob.model_validate(response.json())
