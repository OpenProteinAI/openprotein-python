"""UMAP REST API for making HTTP calls to our UMAP backend."""

import io

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.errors import APIError, InvalidParameterError

from .schemas import FeatureType, UMAPEmbeddingsJob, UMAPFitJob, UMAPMetadata

PATH_PREFIX = "v1/umap"


def umap_list_get(session: APISession) -> list[UMAPMetadata]:
    """Get UMAP job metadata for all UMAPs. Including UMAP dimension and sequence lengths."""
    endpoint = PATH_PREFIX
    response = session.get(endpoint)
    return TypeAdapter(list[UMAPMetadata]).validate_python(response.json())


def umap_get(session: APISession, umap_id: str) -> UMAPMetadata:
    """Get UMAP job metadata. Including UMAP dimension and sequence lengths."""
    endpoint = PATH_PREFIX + f"/{umap_id}"
    response = session.get(endpoint)
    return UMAPMetadata.model_validate(response.json())


def umap_get_sequences(session: APISession, umap_id: str) -> list[bytes]:
    """
    Get sequences used to fit an UMAP.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    umap_id : str
        UMAP ID whose sequences to fetch

    Returns
    -------
    sequences : List[bytes]
    """
    endpoint = PATH_PREFIX + f"/{umap_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def embed_get_sequence_result(
    session: APISession, job_id: str, sequence: str | bytes
) -> bytes:
    """
    Get encoded umap embeddings result for a sequence from the request ID.

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
    endpoint = PATH_PREFIX + f"/embed/{job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def embed_get_batch_result(session: APISession, job_id: str) -> bytes:
    """
    Get encoded umap embeddings batched result from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        Job ID to retrieve results from

    Returns
    -------
    result : bytes
    """
    endpoint = PATH_PREFIX + f"/embed/{job_id}/csv"
    response = session.get(endpoint)
    return response.content


def embed_decode(data: bytes) -> np.ndarray:
    """
    Decode embedding as numpy array.

    Parameters
    ----------
        data (bytes): raw bytes encoding the array received over the API

    Returns
    -------
        np.ndarray: decoded array
    """
    s = io.BytesIO(data)
    return np.load(s, allow_pickle=False)


def embed_batch_decode(data: bytes) -> np.ndarray:
    """
    Decode prediction scores.

    Args:
        data (bytes): raw bytes encoding the array received over the API
        batched (bool): whether or not the result was batched. affects the retrieved csv format whether they contain additional columns and header rows.

    Returns:
        mus (np.ndarray): decoded array of means
        vars (np.ndarray): decoded array of variances
    """
    s = io.BytesIO(data)
    # should contain header and sequence column
    df = pd.read_csv(s)
    umaps = df.iloc[:, 1:].values
    return umaps


def umap_delete(session: APISession, umap_id: str) -> bool:
    """
    Delete and UMAP model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    umap_id : str
        UMAP model to delete

    Returns
    -------
    bool
    """

    endpoint = PATH_PREFIX + f"/{umap_id}"
    response = session.delete(endpoint)
    if 200 <= response.status_code < 300:
        return True
    else:
        raise APIError(response.text)


def umap_fit_post(
    session: APISession,
    model_id: str,
    feature_type: str,
    sequences: list[bytes] | list[str] | None = None,
    assay_id: str | None = None,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    reduction: str | None = None,
    **kwargs,
) -> UMAPFitJob:
    """
    Create UMAP fit job.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        Model to use. Can be either svd_id or id of a foundational model.
    feature_type: str
        Type of feature to use for fitting UMAP. Either PLM or SVD.
    sequences : list[bytes] | None, optional
        Optional sequences to fit UMAP with. Either use sequences or
        assay_id. sequences is preferred.
    assay_id: str | None, optional
        Optional ID of assay containing sequences to fit UMAP with.
        Either use sequences or assay_id. Ignored if sequences are
        provided.
    n_components: int
        Number of UMAP components to fit. Defaults to 2.
    n_neighbors: int
        Number of neighbors to use for fitting. Defaults to 15.
    min_dist: float
        Minimum distance in UMAP fitting. Defaults to 0.1.
    reduction : str | None
        Embedding reduction to use for fitting the UMAP. Defaults to None.
    kwargs:
        Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

    Returns
    -------
    UMAPFitJob
    """

    endpoint = PATH_PREFIX

    body = {
        "model_id": model_id,
        "feature_type": feature_type,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }
    if reduction is not None:
        body["reduction"] = reduction
    if sequences is not None:
        # both provided
        if assay_id is not None:
            raise InvalidParameterError("Expected only either sequences or assay_id")
        sequences = [(s if isinstance(s, str) else s.decode()) for s in sequences]
        body["sequences"] = sequences
    else:
        # both are none
        if assay_id is None:
            raise InvalidParameterError("Expected either sequences or assay_id")
        body["assay_id"] = assay_id
    # add kwargs for embeddings kwargs
    body.update(**kwargs)

    response = session.post(endpoint, json=body)
    # return job for metadata
    return UMAPFitJob.model_validate(response.json())


def umap_embed_post(
    session: APISession, umap_id: str, sequences: list[bytes] | list[str]
) -> UMAPEmbeddingsJob:
    """
    POST a request for embeddings from the given UMAP model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    umap_id : str
        UMAP model to use
    sequences : List[bytes]
        sequences to UMAP

    Returns
    -------
    UMAPEmbeddingsJob
    """
    endpoint = PATH_PREFIX + f"/{umap_id}/embed"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return UMAPEmbeddingsJob.model_validate(response.json())
