"""Clustering REST API — HTTP calls to the backend."""

from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.errors import APIError, InvalidParameterError

from .schemas import (
    ClusteringMetadata,
    HierarchicalClusteringResult,
    HierarchicalFitJob,
)

PATH_PREFIX = "v1/clustering"


def clustering_list_get(
    session: APISession,
    method: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[ClusteringMetadata]:
    """List clustering jobs, optionally filtered by method."""
    params: dict = {}
    if method is not None:
        params["method"] = method
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    response = session.get(PATH_PREFIX, params=params or None)
    return TypeAdapter(list[ClusteringMetadata]).validate_python(response.json())


def clustering_get(session: APISession, clustering_id: str) -> ClusteringMetadata:
    """Fetch clustering job metadata."""
    response = session.get(f"{PATH_PREFIX}/{clustering_id}")
    return ClusteringMetadata.model_validate(response.json())


def clustering_get_result(
    session: APISession, clustering_id: str
) -> HierarchicalClusteringResult:
    """Fetch the clustering result (linkage + leaf_order). Sequences are NOT
    filled by this function — callers that need them should call
    `clustering_get_sequences` and assign to `.sequences`."""
    response = session.get(f"{PATH_PREFIX}/{clustering_id}/result")
    return HierarchicalClusteringResult.model_validate(response.json())


def clustering_get_sequences(session: APISession, clustering_id: str) -> list[bytes]:
    """Fetch the input sequences used for the clustering job."""
    response = session.get(f"{PATH_PREFIX}/{clustering_id}/sequences")
    return TypeAdapter(list[bytes]).validate_python(response.json())


def clustering_delete(session: APISession, clustering_id: str) -> bool:
    """Delete a clustering job."""
    response = session.delete(f"{PATH_PREFIX}/{clustering_id}")
    if 200 <= response.status_code < 300:
        return True
    raise APIError(response.text)


def clustering_redispatch_post(
    session: APISession, clustering_id: str
) -> HierarchicalFitJob:
    """Redispatch a clustering job."""
    response = session.post(f"{PATH_PREFIX}/{clustering_id}/redispatch")
    return HierarchicalFitJob.model_validate(response.json())


def clustering_hierarchical_post(
    session: APISession,
    model_id: str,
    feature_type: str,
    linkage_method: str,
    metric: str,
    sequences: list[bytes] | list[str] | None = None,
    assay_id: str | None = None,
    reduction: str | None = None,
    svd_id: str | None = None,
    **kwargs,
) -> HierarchicalFitJob:
    """POST to create a hierarchical clustering fit job."""
    body: dict = {
        "model_id": model_id,
        "feature_type": feature_type,
        "linkage_method": linkage_method,
        "metric": metric,
    }
    if reduction is not None:
        body["reduction"] = reduction
    if svd_id is not None:
        body["svd_id"] = svd_id
    if sequences is not None:
        if assay_id is not None:
            raise InvalidParameterError("Expected only either sequences or assay_id")
        body["sequences"] = [
            (s if isinstance(s, str) else s.decode()) for s in sequences
        ]
    else:
        if assay_id is None:
            raise InvalidParameterError("Expected either sequences or assay_id")
        body["assay_id"] = assay_id
    body.update(**kwargs)

    response = session.post(f"{PATH_PREFIX}/hierarchical", json=body)
    return HierarchicalFitJob.model_validate(response.json())
