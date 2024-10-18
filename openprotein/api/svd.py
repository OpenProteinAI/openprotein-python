import io

import numpy as np
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import FitJob, SVDEmbeddingsJob, SVDMetadata
from pydantic import TypeAdapter

PATH_PREFIX = "v1/embeddings/svd"


def svd_list_get(session: APISession) -> list[SVDMetadata]:
    """Get SVD job metadata for all SVDs. Including SVD dimension and sequence lengths."""
    endpoint = PATH_PREFIX
    response = session.get(endpoint)
    return TypeAdapter(list[SVDMetadata]).validate_python(response.json())


def svd_get(session: APISession, svd_id: str) -> SVDMetadata:
    """Get SVD job metadata. Including SVD dimension and sequence lengths."""
    endpoint = PATH_PREFIX + f"/{svd_id}"
    response = session.get(endpoint)
    return SVDMetadata(**response.json())


def svd_get_sequences(session: APISession, svd_id: str) -> list[bytes]:
    """
    Get sequences used to fit an SVD.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    svd_id : str
        SVD ID whose sequences to fetch

    Returns
    -------
    sequences : List[bytes]
    """
    endpoint = PATH_PREFIX + f"/{svd_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def embed_get_sequence_result(
    session: APISession, job_id: str, sequence: str | bytes
) -> bytes:
    """
    Get encoded svd embeddings result for a sequence from the request ID.

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


def embed_decode(data: bytes) -> np.ndarray:
    """
    Decode embedding.

    Args:
        data (bytes): raw bytes encoding the array received over the API

    Returns:
        np.ndarray: decoded array
    """
    s = io.BytesIO(data)
    return np.load(s, allow_pickle=False)


def svd_delete(session: APISession, svd_id: str):
    """
    Delete and SVD model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    svd_id : str
        SVD model to delete

    Returns
    -------
    bool
    """

    endpoint = PATH_PREFIX + f"/{svd_id}"
    session.delete(endpoint)
    return True


def svd_fit_post(
    session: APISession,
    model_id: str,
    sequences: list[bytes] | list[str] | None = None,
    assay_id: str | None = None,
    n_components: int = 1024,
    reduction: str | None = None,
    prompt_id: str | None = None,
) -> FitJob:
    """
    Create SVD fit job.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    model_id : str
        model to use
    sequences : list[bytes] | None, optional
        Optional sequences to fit SVD with. Either use sequences or assay_id. sequences is preferred.
    assay_id: str | None, optional
        Optional ID of assay containing sequences to fit SVD with. Either use sequences or assay_id. Ignored if sequences are provided.
    n_components : int
        number of SVD components to fit. default = 1024
    reduction : str | None
        embedding reduction to use for fitting the SVD. default = None

    Returns
    -------
    Job
    """

    endpoint = PATH_PREFIX

    body = {
        "model_id": model_id,
        "n_components": n_components,
    }
    if reduction is not None:
        body["reduction"] = reduction
    if prompt_id is not None:
        body["prompt_id"] = prompt_id
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

    response = session.post(endpoint, json=body)
    # return job for metadata
    return FitJob.model_validate(response.json())


def svd_embed_post(
    session: APISession, svd_id: str, sequences: list[bytes] | list[str]
) -> SVDEmbeddingsJob:
    """
    POST a request for embeddings from the given SVD model.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    svd_id : str
        SVD model to use
    sequences : List[bytes]
        sequences to SVD

    Returns
    -------
    Job
    """
    endpoint = PATH_PREFIX + f"/{svd_id}/embed"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return SVDEmbeddingsJob.model_validate(response.json())
