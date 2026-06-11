"""Tests for openprotein.clustering.api — HTTP layer."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.clustering import api
from openprotein.clustering.schemas import (
    ClusteringMetadata,
    HierarchicalClusteringResult,
    HierarchicalFitJob,
)
from openprotein.errors import APIError, InvalidParameterError


def _session_with_response(payload, status_code=200):
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = payload
    resp.status_code = status_code
    session.get.return_value = resp
    session.post.return_value = resp
    session.delete.return_value = resp
    return session


def _metadata_payload(**overrides):
    base = {
        "id": "abc",
        "status": "SUCCESS",
        "method": "hierarchical",
        "linkage_method": "ward",
        "metric": "euclidean",
        "model_id": "prot-seq",
        "feature_type": "PLM",
        "reduction": "MEAN",
    }
    base.update(overrides)
    return base


def test_clustering_list_get_no_params():
    session = _session_with_response([_metadata_payload()])
    result = api.clustering_list_get(session)
    session.get.assert_called_once_with("v1/clustering", params=None)
    assert len(result) == 1
    assert isinstance(result[0], ClusteringMetadata)


def test_clustering_list_get_with_params():
    session = _session_with_response([])
    api.clustering_list_get(session, method="hierarchical", page_size=10, page_offset=5)
    session.get.assert_called_once_with(
        "v1/clustering",
        params={"method": "hierarchical", "page_size": 10, "page_offset": 5},
    )


def test_clustering_get():
    session = _session_with_response(_metadata_payload(id="xyz"))
    md = api.clustering_get(session, "xyz")
    session.get.assert_called_once_with("v1/clustering/xyz")
    assert md.id == "xyz"


def test_clustering_get_sequences():
    session = _session_with_response(["MKTA", "MRTV"])
    seqs = api.clustering_get_sequences(session, "abc")
    session.get.assert_called_once_with("v1/clustering/abc/sequences")
    assert seqs == [b"MKTA", b"MRTV"]


def test_clustering_delete_success():
    session = _session_with_response(None, status_code=204)
    assert api.clustering_delete(session, "abc") is True
    session.delete.assert_called_once_with("v1/clustering/abc")


def test_clustering_delete_error():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    resp.text = "not found"
    session.delete.return_value = resp
    with pytest.raises(APIError):
        api.clustering_delete(session, "abc")


def test_clustering_redispatch_post():
    session = _session_with_response(
        {
            "job_id": "abc",
            "job_type": "/clustering/hierarchical",
            "status": "PENDING",
            "created_date": "2026-04-23T00:00:00",
        }
    )
    job = api.clustering_redispatch_post(session, "abc")
    session.post.assert_called_once_with("v1/clustering/abc/redispatch")
    assert isinstance(job, HierarchicalFitJob)


def test_clustering_get_result():
    payload = {
        "n_leaves": 3,
        "linkage": [[0, 1, 0.12, 2], [3, 2, 0.43, 3]],
        "leaf_order": [0, 1, 2],
    }
    session = _session_with_response(payload)
    result = api.clustering_get_result(session, "abc")
    session.get.assert_called_once_with("v1/clustering/abc/result")
    assert isinstance(result, HierarchicalClusteringResult)
    assert result.n_leaves == 3
    assert isinstance(result.linkage, np.ndarray)
    assert result.linkage.shape == (2, 4)


def _fit_response():
    return {
        "job_id": "new",
        "job_type": "/clustering/hierarchical",
        "status": "PENDING",
        "created_date": "2026-04-23T00:00:00",
    }


def test_hierarchical_post_plm_sequences():
    session = _session_with_response(_fit_response())
    job = api.clustering_hierarchical_post(
        session,
        model_id="prot-seq",
        feature_type="PLM",
        linkage_method="ward",
        metric="euclidean",
        sequences=[b"MKTA", b"MRTV"],
        reduction="MEAN",
    )
    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == "v1/clustering/hierarchical"
    body = kwargs["json"]
    assert body["model_id"] == "prot-seq"
    assert body["feature_type"] == "PLM"
    assert body["linkage_method"] == "ward"
    assert body["metric"] == "euclidean"
    assert body["reduction"] == "MEAN"
    assert body["sequences"] == ["MKTA", "MRTV"]
    assert "svd_id" not in body
    assert isinstance(job, HierarchicalFitJob)


def test_hierarchical_post_svd_assay():
    session = _session_with_response(_fit_response())
    api.clustering_hierarchical_post(
        session,
        model_id="some-svd-id",
        feature_type="SVD",
        linkage_method="average",
        metric="cosine",
        assay_id="assay-1",
        svd_id="some-svd-id",
    )
    body = session.post.call_args.kwargs["json"]
    assert body["feature_type"] == "SVD"
    assert body["svd_id"] == "some-svd-id"
    assert body["assay_id"] == "assay-1"
    assert "sequences" not in body
    assert "reduction" not in body


def test_hierarchical_post_rejects_both_sequences_and_assay():
    session = _session_with_response(_fit_response())
    with pytest.raises(InvalidParameterError):
        api.clustering_hierarchical_post(
            session,
            model_id="prot-seq",
            feature_type="PLM",
            linkage_method="ward",
            metric="euclidean",
            sequences=[b"MKTA", b"MRTV"],
            assay_id="assay-1",
        )


def test_hierarchical_post_rejects_neither():
    session = _session_with_response(_fit_response())
    with pytest.raises(InvalidParameterError):
        api.clustering_hierarchical_post(
            session,
            model_id="prot-seq",
            feature_type="PLM",
            linkage_method="ward",
            metric="euclidean",
        )


def test_hierarchical_post_force_recompute_true():
    session = _session_with_response(_fit_response())
    api.clustering_hierarchical_post(
        session,
        model_id="prot-seq",
        feature_type="PLM",
        linkage_method="ward",
        metric="euclidean",
        sequences=[b"MKTA", b"MRTV"],
        reduction="MEAN",
        force_recompute=True,
    )
    _, kwargs = session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_hierarchical_post_force_recompute_default():
    session = _session_with_response(_fit_response())
    api.clustering_hierarchical_post(
        session,
        model_id="prot-seq",
        feature_type="PLM",
        linkage_method="ward",
        metric="euclidean",
        sequences=[b"MKTA", b"MRTV"],
        reduction="MEAN",
    )
    _, kwargs = session.post.call_args
    assert kwargs.get("params") is None
