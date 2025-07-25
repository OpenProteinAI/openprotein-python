"""Predictor REST API for making HTTP calls to our predictor backend."""

import io

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.jobs import Job

from .schemas import (
    Job,
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictorCVJob,
    PredictorMetadata,
    PredictorTrainJob,
    PredictSingleSiteJob,
)

PATH_PREFIX = "v1/predictor"


def predictor_list(
    session: APISession,
    limit: int = 100,
    offset: int = 0,
    include_stats: bool = False,
    include_calibration_curves: bool = False,
) -> list[PredictorMetadata]:
    """
    List trained predictors.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    limit: int
        Limit of the number of predictors to return in list.
    offset: int
        Offset to the predictors to query for paged queries.
    include_stats: bool
        Whether to include stats of the predictor from latest evaluation, i.e. pearson, spearman, ece.
    include_calibration_curves: bool
        Whether to include calibration curves of the predictor from latest evaluation.

    Returns
    -------
    list[PredictorMetadata]
        List of predictors
    """
    endpoint = PATH_PREFIX
    response = session.get(
        endpoint,
        params={
            "limit": limit,
            "offset": offset,
            "stats": include_stats,
            "curve": include_calibration_curves,
        },
    )
    return TypeAdapter(list[PredictorMetadata]).validate_python(response.json())


def predictor_get(
    session: APISession,
    predictor_id: str,
    include_stats: bool = False,
    include_calibration_curves: bool = False,
) -> PredictorMetadata:
    """
    Get a trained predictor by its identifier.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    predictor_id : str
        Unique identifier of the predictor.
    include_stats : bool
        Whether to include stats of the predictor from the latest evaluation (pearson, spearman, ece).
    include_calibration_curves : bool
        Whether to include calibration curves of the predictor from the latest evaluation.
    Returns
    -------
    PredictorMetadata
        Metadata of the requested predictor.
    """
    endpoint = PATH_PREFIX + f"/{predictor_id}"
    response = session.get(
        endpoint,
        params={
            "stats": include_stats,
            "curve": include_calibration_curves,
        },
    )
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
    **kwargs,
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
    # add kwargs
    body.update(kwargs)

    response = session.post(endpoint, json=body)
    return PredictorTrainJob.model_validate(response.json())


def predictor_ensemble(session: APISession, predictor_ids: list[str]):
    endpoint = PATH_PREFIX + f"/ensemble"

    body = {
        "model_ids": predictor_ids,
    }

    response = session.post(endpoint, json=body)
    return TypeAdapter(PredictorMetadata).validate_python(response.json())


def predictor_delete(session: APISession, predictor_id: str):
    endpoint = PATH_PREFIX + f"/{predictor_id}"
    response = session.delete(endpoint)
    if 200 <= response.status_code < 300:
        return True
    else:
        raise APIError(response.text)


def predictor_crossvalidate_post(
    session: APISession, predictor_id: str, n_splits: int | None = None
):
    endpoint = PATH_PREFIX + f"/{predictor_id}/crossvalidate"

    params = {}
    if n_splits is not None:
        params["n_splits"] = n_splits
    response = session.post(endpoint, params=params)

    return PredictorCVJob.model_validate(response.json())


def predictor_crossvalidate_get(session: APISession, crossvalidate_job_id: str):
    endpoint = PATH_PREFIX + f"/crossvalidate/{crossvalidate_job_id}"

    response = session.get(endpoint)
    return response.content


def predictor_predict_post(
    session: APISession, predictor_id: str, sequences: list[bytes] | list[str]
):
    endpoint = PATH_PREFIX + f"/{predictor_id}/predict"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return PredictJob.model_validate(response.json())


def predictor_predict_multi_post(
    session: APISession, predictor_ids: list[str], sequences: list[bytes] | list[str]
):
    endpoint = PATH_PREFIX + f"/predict"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "model_ids": predictor_ids,
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return PredictMultiJob.model_validate(response.json())


def predictor_predict_single_site_post(
    session: APISession,
    predictor_id: str,
    base_sequence: bytes | str,
):
    endpoint = PATH_PREFIX + f"/{predictor_id}/predict_single_site"

    base_sequence = (
        base_sequence.decode() if isinstance(base_sequence, bytes) else base_sequence
    )
    body = {
        "base_sequence": base_sequence,
    }
    response = session.post(endpoint, json=body)

    return PredictSingleSiteJob.model_validate(response.json())


def predictor_predict_multi_single_site_post(
    session: APISession,
    predictor_ids: list[str],
    base_sequence: bytes | str,
):
    endpoint = PATH_PREFIX + f"/predict_single_site"

    base_sequence = (
        base_sequence.decode() if isinstance(base_sequence, bytes) else base_sequence
    )
    body = {
        "model_ids": predictor_ids,
        "base_sequence": base_sequence,
    }
    response = session.post(endpoint, json=body)

    return PredictMultiSingleSiteJob.model_validate(response.json())


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
        sequence to retrieve predictions for

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


def decode_predict(data: bytes, batched: bool = False) -> tuple[np.ndarray, np.ndarray]:
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


def decode_crossvalidate(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode crossvalidate scores.

    Args:
        data (bytes): raw bytes encoding the array received over the API

    Returns:
        mus (np.ndarray): decoded array of means
        vars (np.ndarray): decoded array of variances
    """
    s = io.BytesIO(data)
    # should contain header and sequence column
    df = pd.read_csv(s)
    scores = df.values
    # row_num, seq, measurement_name, y, y_mu, y_var
    y = scores[:, 3]
    mus = scores[:, 4]
    vars = scores[:, 5]
    return y, mus, vars
