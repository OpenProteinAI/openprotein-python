"""Tests for openprotein.clustering.models.HierarchicalClusteringFuture."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.clustering.models import HierarchicalClusteringFuture
from openprotein.clustering.schemas import (
    ClusteringMetadata,
    HierarchicalClusteringResult,
    HierarchicalFitJob,
)
from openprotein.jobs import JobStatus, JobsAPI


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


def _job(status="PENDING"):
    return HierarchicalFitJob.model_validate(
        {
            "job_id": "abc",
            "job_type": "/clustering/hierarchical",
            "status": status,
            "created_date": datetime(2026, 4, 23).isoformat(),
        }
    )


def test_future_init_with_job_fetches_metadata():
    session = MagicMock()
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata()
        f = HierarchicalClusteringFuture(session=session, job=_job())
        assert f.id == "abc"
        m.clustering_get.assert_called_once_with(session, "abc")


def test_future_init_with_metadata_only():
    session = MagicMock()
    jobs_api = MagicMock(spec=JobsAPI)
    jobs_api.get_job.return_value = _job()
    session.jobs = jobs_api
    md = _metadata()
    f = HierarchicalClusteringFuture(session=session, metadata=md)
    assert f.id == "abc"
    assert f._metadata is md


def test_future_init_requires_job_or_metadata():
    session = MagicMock()
    with pytest.raises(ValueError):
        HierarchicalClusteringFuture(session=session)


def test_future_get_fetches_result_and_sequences():
    session = MagicMock()
    linkage_payload = {
        "n_leaves": 2,
        "linkage": [[0, 1, 0.1, 2]],
        "leaf_order": [0, 1],
    }
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata()
        m.clustering_get_result.return_value = (
            HierarchicalClusteringResult.model_validate(linkage_payload)
        )
        m.clustering_get_sequences.return_value = [b"MKTA", b"MRTV"]
        f = HierarchicalClusteringFuture(session=session, job=_job("SUCCESS"))
        result = f._get()
    assert result.n_leaves == 2
    assert result.sequences == [b"MKTA", b"MRTV"]


def test_future_metadata_refreshes_when_not_terminal():
    session = MagicMock()
    pending = _metadata(status=JobStatus.PENDING)
    done = _metadata(status=JobStatus.SUCCESS)
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.side_effect = [pending, done]
        f = HierarchicalClusteringFuture(session=session, job=_job("PENDING"))
        # First init call used the side_effect (pending)
        # Accessing .metadata again should re-fetch because is_done() is False
        _ = f.metadata
        assert m.clustering_get.call_count == 2


def test_future_metadata_no_refresh_when_terminal():
    session = MagicMock()
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata(status=JobStatus.SUCCESS)
        f = HierarchicalClusteringFuture(session=session, job=_job("SUCCESS"))
        _ = f.metadata
        _ = f.metadata
        assert m.clustering_get.call_count == 1


def test_future_sequences_cached():
    session = MagicMock()
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata()
        m.clustering_get_sequences.return_value = [b"MKTA"]
        f = HierarchicalClusteringFuture(session=session, job=_job())
        assert f.sequences == [b"MKTA"]
        assert f.sequences == [b"MKTA"]
        assert m.clustering_get_sequences.call_count == 1


def test_future_delete():
    session = MagicMock()
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata()
        m.clustering_delete.return_value = True
        f = HierarchicalClusteringFuture(session=session, job=_job())
        assert f.delete() is True
        m.clustering_delete.assert_called_once_with(session, "abc")


def test_future_redispatch():
    session = MagicMock()
    session.jobs = MagicMock(spec=JobsAPI)
    new_job = _job(status="PENDING")
    with patch("openprotein.clustering.models.api") as m:
        m.clustering_get.return_value = _metadata()
        m.clustering_redispatch_post.return_value = new_job
        f = HierarchicalClusteringFuture(session=session, job=_job(status="SUCCESS"))
        new_future = f.redispatch()
    m.clustering_redispatch_post.assert_called_once_with(session, "abc")
    assert isinstance(new_future, HierarchicalClusteringFuture)
    assert new_future.id == "abc"
