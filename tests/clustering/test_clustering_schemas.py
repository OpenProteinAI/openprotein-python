"""Tests for openprotein.clustering.schemas."""

from datetime import datetime

from openprotein.clustering.schemas import (
    ClusteringMetadata,
    HierarchicalFitJob,
    LinkageMethod,
    Metric,
)
from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType


def test_linkage_method_values():
    assert LinkageMethod.WARD.value == "ward"
    assert LinkageMethod.SINGLE.value == "single"
    assert LinkageMethod.COMPLETE.value == "complete"
    assert LinkageMethod.AVERAGE.value == "average"
    assert LinkageMethod.WEIGHTED.value == "weighted"
    assert LinkageMethod.CENTROID.value == "centroid"
    assert LinkageMethod.MEDIAN.value == "median"


def test_linkage_method_is_str():
    assert LinkageMethod.WARD == "ward"


def test_metric_values():
    expected = {
        "euclidean", "cosine", "correlation", "hamming", "chebyshev",
        "cityblock", "sqeuclidean", "canberra", "braycurtis",
    }
    assert {m.value for m in Metric} == expected


def test_metric_is_str():
    assert Metric.EUCLIDEAN == "euclidean"


def test_clustering_metadata_parses_plm():
    md = ClusteringMetadata.model_validate(
        {
            "id": "abc",
            "status": "SUCCESS",
            "method": "hierarchical",
            "linkage_method": "ward",
            "metric": "euclidean",
            "model_id": "prot-seq",
            "feature_type": "PLM",
            "reduction": "MEAN",
        }
    )
    assert md.id == "abc"
    assert md.status is JobStatus.SUCCESS
    assert md.linkage_method is LinkageMethod.WARD
    assert md.metric is Metric.EUCLIDEAN
    assert md.feature_type is FeatureType.PLM


def test_clustering_metadata_parses_svd():
    md = ClusteringMetadata.model_validate(
        {
            "id": "abc",
            "status": "PENDING",
            "method": "hierarchical",
            "linkage_method": "average",
            "metric": "cosine",
            "model_id": "some-svd-id",
            "feature_type": "SVD",
            "svd_id": "some-svd-id",
        }
    )
    assert md.svd_id == "some-svd-id"
    assert md.reduction is None


def test_hierarchical_fit_job_parses():
    job = HierarchicalFitJob.model_validate(
        {
            "job_id": "xyz",
            "job_type": "/clustering/hierarchical",
            "status": "PENDING",
            "created_date": datetime(2026, 4, 23).isoformat(),
        }
    )
    assert job.job_type == JobType.clustering_hierarchical


import numpy as np

from openprotein.clustering.schemas import HierarchicalClusteringResult


def test_clustering_result_linkage_becomes_ndarray():
    r = HierarchicalClusteringResult.model_validate(
        {
            "n_leaves": 3,
            "linkage": [[0, 1, 0.12, 2], [3, 2, 0.43, 3]],
            "leaf_order": [0, 1, 2],
        }
    )
    assert isinstance(r.linkage, np.ndarray)
    assert r.linkage.shape == (2, 4)
    assert r.linkage.dtype == np.float64
    assert r.linkage[0, 2] == 0.12
    assert r.sequences == []


def test_clustering_result_accepts_ndarray():
    arr = np.asarray([[0.0, 1.0, 0.12, 2.0]])
    r = HierarchicalClusteringResult(
        n_leaves=2, linkage=arr, leaf_order=[0, 1]
    )
    assert r.linkage is arr
