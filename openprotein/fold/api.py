"""Fold REST API interface for making HTTP calls to our fold backend."""

import io
from typing import TYPE_CHECKING, Literal

import numpy as np
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
    Get results associated with the given request ID.

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
    session: APISession, job_id: str, sequence: bytes | str
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from.
    sequence : bytes or str
        Sequence to retrieve results for.

    Returns
    -------
    bytes
        Encoded result for the sequence.
    """
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def fold_get_complex_result(
    session: APISession, job_id: str, format: Literal["pdb", "mmcif"]
) -> bytes:
    """
    Get encoded result for a complex from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from.
    format : {'pdb', 'mmcif'}
        Format of the result.

    Returns
    -------
    bytes
        Encoded result for the complex.
    """
    endpoint = PATH_PREFIX + f"/{job_id}/complex"
    response = session.get(
        endpoint,
        params={
            "format": format,
        },
    )
    return response.content


def fold_get_complex_extra_result(
    session: APISession,
    job_id: str,
    key: Literal["pae", "pde", "plddt", "confidence", "affinity", "score", "metrics"],
) -> "np.ndarray | list[dict] | pd.DataFrame":
    """
    Get extra result for a complex from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from.
    key : {'pae', 'pde', 'plddt', 'confidence', 'affinity'}
        The type of result to retrieve.

    Returns
    -------
    numpy.ndarray or list of dict
        The result as a numpy array (for "pae", "pde", "plddt") or a list of dictionaries (for "confidence", "affinity").
    """
    if key in {"pae", "pde", "plddt"}:
        formatter = lambda response: np.load(io.BytesIO(response.content))
    elif key in {"confidence", "affinity"}:
        formatter = lambda response: response.json()
    elif key in {"score", "metrics"}:
        import pandas as pd

        formatter = lambda response: pd.read_csv(io.StringIO(response.content.decode()))
    else:
        raise ValueError(f"Unexpected key: {key}")
    endpoint = PATH_PREFIX + f"/{job_id}/complex/{key}"
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
    sequences : sequence of bytes or str, optional
        Sequences to request results for.
    msa_id : str, optional
        MSA ID to use.
    num_recycles : int, optional
        Number of recycles for structure prediction.
    num_models : int, optional
        Number of models to generate.
    num_relax : int, optional
        Number of relaxation steps.
    use_potentials : bool, optional
        Whether to use potentials.
    diffusion_samples : int, optional
        Number of diffusion samples (boltz).
    recycling_steps : int, optional
        Number of recycling steps (boltz).
    sampling_steps : int, optional
        Number of sampling steps (boltz).
    step_scale : float, optional
        Step scale (boltz).
    constraints : dict, optional
        Constraints to apply.
    templates : list, optional
        Templates to use.
    properties : dict, optional
        Additional properties.

    Returns
    -------
    FoldJob
        Job object referring to this request.
    """
    endpoint = PATH_PREFIX + f"/models/{model_id}"

    body: dict = {}
    if kwargs.get("sequences"):
        sequences = kwargs["sequences"]
        # NOTE we are handling the boltz form here too
        sequences = [s.decode() if isinstance(s, bytes) else s for s in sequences]
        kwargs["sequences"] = sequences
    # add non-None args - note this doesnt affect msa_id which is nested
    for k, v in kwargs.items():
        if v is not None:
            body[k] = v

    response = session.post(endpoint, json=body)
    return FoldJob.model_validate(response.json())
