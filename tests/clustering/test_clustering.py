"""Tests for openprotein.clustering.clustering.ClusteringAPI."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.clustering.clustering import ClusteringAPI
from openprotein.clustering.models import HierarchicalClusteringFuture
from openprotein.clustering.schemas import ClusteringMetadata, HierarchicalFitJob
from openprotein.common import FeatureType
from openprotein.embeddings import EmbeddingModel
from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobsAPI, JobStatus
from openprotein.svd import SVDModel


def _metadata(**overrides):
    base = dict(
        id="abc",
        status=JobStatus.SUCCESS,
        method="hierarchical",
        linkage_method="ward",
        metric="euclidean",
        model_id="prot-seq",
        feature_type="PLM",
        reduction="MEAN",
    )
    base.update(overrides)
    return ClusteringMetadata.model_validate(base)


def test_get_returns_future():
    session = MagicMock()
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = HierarchicalFitJob.model_validate(
        {
            "job_id": "abc",
            "job_type": "/clustering/hierarchical",
            "status": "SUCCESS",
            "created_date": datetime(2026, 4, 23).isoformat(),
        }
    )
    with patch("openprotein.clustering.clustering.api") as capi_api:
        capi_api.clustering_get.return_value = _metadata()
        capi = ClusteringAPI(session=session)
        future = capi.get("abc")
    assert isinstance(future, HierarchicalClusteringFuture)
    assert future.id == "abc"


def test_list_returns_futures():
    session = MagicMock()
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = HierarchicalFitJob.model_validate(
        {
            "job_id": "a",
            "job_type": "/clustering/hierarchical",
            "status": "SUCCESS",
            "created_date": datetime(2026, 4, 23).isoformat(),
        }
    )
    with patch("openprotein.clustering.clustering.api") as capi_api:
        capi_api.clustering_list_get.return_value = [
            _metadata(id="a"),
            _metadata(id="b"),
        ]
        capi = ClusteringAPI(session=session)
        futures = capi.list()
    assert [f.id for f in futures] == ["a", "b"]


def test_list_passes_method_filter():
    session = MagicMock()
    with patch("openprotein.clustering.clustering.api") as capi_api:
        capi_api.clustering_list_get.return_value = []
        capi = ClusteringAPI(session=session)
        capi.list(method="hierarchical", limit=5, offset=10)
        capi_api.clustering_list_get.assert_called_once_with(
            session, method="hierarchical", limit=5, offset=10
        )


def _hierarchical_job_payload():
    return {
        "job_id": "new",
        "job_type": "/clustering/hierarchical",
        "status": "PENDING",
        "created_date": datetime(2026, 4, 23).isoformat(),
    }


def test_hierarchical_with_embedding_model():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    with (
        patch("openprotein.clustering.clustering.api") as capi_api,
        patch("openprotein.clustering.models.api") as m,
    ):
        capi_api.clustering_hierarchical_post.return_value = (
            HierarchicalFitJob.model_validate(_hierarchical_job_payload())
        )
        m.clustering_get.return_value = _metadata(id="new", status=JobStatus.PENDING)
        capi = ClusteringAPI(session=session)
        f = capi.hierarchical(
            model=emb_model,
            reduction="MEAN",
            sequences=[b"MKTA", b"MRTV"],
        )
    capi_api.clustering_hierarchical_post.assert_called_once()
    call_kwargs = capi_api.clustering_hierarchical_post.call_args.kwargs
    assert call_kwargs["model_id"] == "prot-seq"
    assert call_kwargs["feature_type"] == "PLM"
    assert call_kwargs["reduction"] == "MEAN"
    assert isinstance(f, HierarchicalClusteringFuture)


def test_hierarchical_with_svd_model():
    session = MagicMock()
    svd_model = MagicMock(spec=SVDModel)
    svd_model.id = "svd-42"
    with (
        patch("openprotein.clustering.clustering.api") as capi_api,
        patch("openprotein.clustering.models.api") as m,
    ):
        capi_api.clustering_hierarchical_post.return_value = (
            HierarchicalFitJob.model_validate(_hierarchical_job_payload())
        )
        m.clustering_get.return_value = _metadata(
            id="new", feature_type="SVD", svd_id="svd-42"
        )
        capi = ClusteringAPI(session=session)
        capi.hierarchical(
            model=svd_model,
            sequences=[b"MKTA", b"MRTV"],
        )
    call_kwargs = capi_api.clustering_hierarchical_post.call_args.kwargs
    assert call_kwargs["model_id"] == "svd-42"
    assert call_kwargs["feature_type"] == "SVD"
    assert call_kwargs["svd_id"] == "svd-42"


def test_hierarchical_with_str_model_requires_feature_type():
    session = MagicMock()
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="feature_type"):
        capi.hierarchical(model="some-id", sequences=[b"MKTA", b"MRTV"])


def test_hierarchical_rejects_below_min_sequences():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="at least 2"):
        capi.hierarchical(model=emb_model, reduction="MEAN", sequences=[b"MKTA"])


def test_hierarchical_rejects_exceeds_max_sequences():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="10000"):
        capi.hierarchical(model=emb_model, reduction="MEAN", sequences=[b"M"] * 10001)


def test_hierarchical_rejects_illegal_ward_cosine():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="ward"):
        capi.hierarchical(
            model=emb_model,
            reduction="MEAN",
            linkage_method="ward",
            metric="cosine",
            sequences=[b"MKTA", b"MRTV"],
        )


def test_hierarchical_rejects_centroid_hamming():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="centroid"):
        capi.hierarchical(
            model=emb_model,
            reduction="MEAN",
            linkage_method="centroid",
            metric="hamming",
            sequences=[b"MKTA", b"MRTV"],
        )


def test_hierarchical_rejects_plm_without_reduction():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="reduction"):
        capi.hierarchical(model=emb_model, sequences=[b"MKTA", b"MRTV"])


def test_hierarchical_rejects_svd_with_reduction():
    session = MagicMock()
    svd_model = MagicMock(spec=SVDModel)
    svd_model.id = "svd-42"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="reduction"):
        capi.hierarchical(
            model=svd_model, reduction="MEAN", sequences=[b"MKTA", b"MRTV"]
        )


def test_hierarchical_rejects_both_sequences_and_assay():
    session = MagicMock()
    emb_model = MagicMock(spec=EmbeddingModel)
    emb_model.id = "prot-seq"
    capi = ClusteringAPI(session=session)
    with pytest.raises(InvalidParameterError, match="either sequences or assay"):
        capi.hierarchical(
            model=emb_model,
            reduction="MEAN",
            sequences=[b"MKTA", b"MRTV"],
            assay="assay-1",
        )
