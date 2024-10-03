import io

import numpy as np
import pandas as pd
from openprotein.base import APISession
from openprotein.schemas import Job, PredictorMetadata
from pydantic import TypeAdapter

PATH_PREFIX = "v1/predictor"


def predictor_list(session: APISession) -> list[PredictorMetadata]:
    """
    List trained predictors.

    Parameters
    ----------
    session : APISession
        Session object for API communication.

    Returns
    -------
    list[PredictorMetadata]
        List of predictors
    """
    endpoint = PATH_PREFIX
    response = session.get(endpoint)
    return TypeAdapter(list[PredictorMetadata]).validate_python(response.json())


def predictor_get(session: APISession, predictor_id: str) -> PredictorMetadata:
    endpoint = PATH_PREFIX + f"/{predictor_id}"
    response = session.get(endpoint)
    return TypeAdapter(PredictorMetadata).validate_python(response.json())


def predictor_fit_gp_post(
    session: APISession,
    assay_id: str,
    properties: list[str],
    feature_type: str,
    model_id: str,
    reduction: str | None = None,
    prompt_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Job:
    """
    Create SVD fit job.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_id : str
        Assay ID to fit GP on.
    properties: list[str]
        Properties in the assay to fit the gp on.
    feature_type: str
        Type of features to use for encoding sequences. PLM or SVD.
    model_id : str
        Protembed/SVD model to use depending on feature type.
    reduction : str | None
        Type of embedding reduction to use for computing features. default = None
    prompt_id: str | None
        Prompt ID if using PoET-based models.
    name: str | None
        Optional name of predictor model. Randomly generated if not provided.
    description: str | None
        Optional description to attach to the model.

    Returns
    -------
    PredictorTrainJob
    """
    endpoint = PATH_PREFIX + "/gp"

    body = {
        "dataset": {
            "assay_id": assay_id,
            "properties": properties,
        },
        "features": {
            "type": feature_type,
            "model_id": model_id,
        },
        "kernel": {
            "type": "rbf",
            # "multitask": True
        },
    }
    if reduction is not None:
        body["features"]["reduction"] = reduction
    if prompt_id is not None:
        body["features"]["prompt_id"] = prompt_id
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description

    response = session.post(endpoint, json=body)
    return Job.model_validate(response.json())


def predictor_delete(session: APISession, predictor_id: str):
    raise NotImplementedError()


def predictor_predict_post(
    session: APISession, predictor_id: str, sequences: list[bytes] | list[str]
):
    endpoint = PATH_PREFIX + f"/{predictor_id}/predict"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return Job.model_validate(response.json())


def predictor_predict_get_sequences(
    session: APISession, prediction_job_id: str
) -> list[bytes]:
    endpoint = PATH_PREFIX + f"/predict/{prediction_job_id}/sequences"

    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def predictor_predict_get_sequence_result(
    session: APISession, prediction_job_id: str, sequence: bytes | str
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from
    sequence from: bytes
        sequence to retrieve results for

    Returns
    -------
    result : bytes
    """
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    endpoint = PATH_PREFIX + f"/predict/{prediction_job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def predictor_predict_get_batched_result(
    session: APISession, prediction_job_id: str
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    prediction_job_id : str
        job ID to retrieve results from
    sequence : bytes
        sequence to retrieve results for

    Returns
    -------
    result : bytes
    """
    endpoint = PATH_PREFIX + f"/predict/{prediction_job_id}"
    response = session.get(endpoint)
    return response.content


def decode_score(data: bytes, batched: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode embedding.

    Args:
        data (bytes): raw bytes encoding the array received over the API
        batched (bool): whether or not the result was batched. affects the retrieved csv format whether they contain additional columns and header rows.

    Returns:
        mus (np.ndarray): decoded array of means
        vars (np.ndarray): decoded array of variances
    """
    s = io.BytesIO(data)
    if batched:
        # should contain header and sequence column
        df = pd.read_csv(s)
        scores = df.iloc[:, 1:].values
    else:
        # should be a single row with 2n columns
        df = pd.read_csv(s, header=None)
        scores = df.values
    mus = scores[:, ::2]
    vars = scores[:, 1::2]
    return mus, vars
