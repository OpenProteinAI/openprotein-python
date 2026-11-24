"""Predictor REST API for making HTTP calls to our predictor backend."""

import io

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.models.kernels import _GPHyperparameters

from .schemas import (
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictorCVJob,
    PredictorMetadata,
    PredictorTrainJob,
    PredictSingleSiteJob,
)

__all__ = [
    "decode_crossvalidate",
    "decode_predict",
    "predictor_crossvalidate_get",
    "predictor_crossvalidate_post",
    "predictor_delete",
    "predictor_ensemble",
    "predictor_fit_gp_post",
    "predictor_get",
    "predictor_list",
    "predictor_predict_get_batched_result",
    "predictor_predict_get_sequence_result",
    "predictor_predict_get_sequences",
    "predictor_predict_multi_post",
    "predictor_predict_multi_single_site_post",
    "predictor_predict_post",
    "predictor_predict_single_site_post",
]


_PATH_PREFIX = "v1/predictor"


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
    endpoint = _PATH_PREFIX
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
    endpoint = _PATH_PREFIX + f"/{predictor_id}"
    response = session.get(
        endpoint,
        params={
            "stats": include_stats,
            "curve": include_calibration_curves,
        },
    )
    return TypeAdapter(PredictorMetadata).validate_python(response.json())


def _predictor_get_hyperparameters(
    session: APISession, predictor_id: str
) -> dict[str, _GPHyperparameters]:
    """
    Get the trained hyperparameters of a GP predictor.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    predictor_id : str
        Unique identifier of the predictor.

    Returns
    -------
    dict of str to _GPHyperparameters
        Trained hyperparameters, keyed by trained property.
    """
    endpoint = _PATH_PREFIX + f"/{predictor_id}/hyperparameters"
    response = session.get(endpoint, timeout=session.timeout)
    return TypeAdapter(dict[str, _GPHyperparameters]).validate_python(response.json())


def predictor_fit_gp_post(
    session: APISession,
    assay_id: str,
    properties: list[str],
    feature_type: str,
    model_id: str,
    reduction: str | None = None,
    name: str | None = None,
    description: str | None = None,
    kernel: str = "rbf",
    period: float | None = None,
    alpha: float | None = None,
    **kwargs,
) -> PredictorTrainJob:
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
    name: str | None
        Optional name of predictor model. Randomly generated if not provided.
    description: str | None
        Optional description to attach to the model.
    kernel : str
        Kernel to train the GP with (e.g. "rbf", "matern52", "periodic",
        "rational_quadratic"). default = "rbf".
    period : float | None
        Period length for the periodic kernel. default = None.
    alpha : float | None
        Scale-mixture parameter for the rational_quadratic kernel. default = None.
    kwargs:
        Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

    Returns
    -------
    PredictorTrainJob
        Train job that can be tracked for progress.
    """
    endpoint = _PATH_PREFIX + "/gp"

    kernel_spec: dict = {"type": kernel}
    if period is not None:
        kernel_spec["period"] = period
    if alpha is not None:
        kernel_spec["alpha"] = alpha

    body = {
        "dataset": {
            "assay_id": assay_id,
            "properties": properties,
        },
        "features": {
            "type": feature_type,
            "model_id": model_id,
        },
        "kernel": kernel_spec,
    }
    if reduction is not None:
        body["features"]["reduction"] = reduction
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    body["features"].update(kwargs)

    response = session.post(endpoint, json=body)
    return PredictorTrainJob.model_validate(response.json())


def predictor_ensemble(
    session: APISession, predictor_ids: list[str]
) -> PredictorMetadata:
    endpoint = _PATH_PREFIX + f"/ensemble"

    body = {
        "model_ids": predictor_ids,
    }

    response = session.post(endpoint, json=body)
    return PredictorMetadata.model_validate(response.json())


def predictor_delete(session: APISession, predictor_id: str):
    endpoint = _PATH_PREFIX + f"/{predictor_id}"
    response = session.delete(endpoint)
    if 200 <= response.status_code < 300:
        return True
    else:
        raise APIError(response.text)


def predictor_crossvalidate_post(
    session: APISession, predictor_id: str, n_splits: int | None = None
):
    endpoint = _PATH_PREFIX + f"/{predictor_id}/crossvalidate"

    params = {}
    if n_splits is not None:
        params["n_splits"] = n_splits
    response = session.post(endpoint, params=params)

    return PredictorCVJob.model_validate(response.json())


def predictor_crossvalidate_get(session: APISession, crossvalidate_job_id: str):
    endpoint = _PATH_PREFIX + f"/crossvalidate/{crossvalidate_job_id}"

    response = session.get(endpoint)
    return response.content


def predictor_predict_post(
    session: APISession, predictor_id: str, sequences: list[bytes] | list[str]
):
    endpoint = _PATH_PREFIX + f"/{predictor_id}/predict"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body = {
        "sequences": sequences_unicode,
    }
    response = session.post(endpoint, json=body)

    return PredictJob.model_validate(response.json())


def predictor_predict_multi_post(
    session: APISession, predictor_ids: list[str], sequences: list[bytes] | list[str]
):
    endpoint = _PATH_PREFIX + f"/predict"

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
    endpoint = _PATH_PREFIX + f"/{predictor_id}/predict_single_site"

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
    endpoint = _PATH_PREFIX + f"/predict_single_site"

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
    endpoint = _PATH_PREFIX + f"/predict/{prediction_job_id}/sequences"

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
    endpoint = _PATH_PREFIX + f"/predict/{prediction_job_id}/{sequence}"
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
    endpoint = _PATH_PREFIX + f"/predict/{prediction_job_id}"
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

    Parameters
    ----------
    data : bytes
        Raw bytes of the crossvalidation csv received over the API.

    Returns
    -------
    tuple of np.ndarray
        The observed ``y`` and the cross-validated ``y_mu`` and ``y_var`` of each
        evaluated row.
    """
    df = pd.read_csv(io.BytesIO(data))
    scores = df.values
    # row_num, seq, measurement_name, y, y_mu, y_var
    return scores[:, 3], scores[:, 4], scores[:, 5]
