"""E2E tests for the clustering domain."""

import time

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.clustering import HierarchicalClusteringFuture, HierarchicalClusteringResult, LinkageMethod, Metric
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.errors import HTTPError, InvalidParameterError
from openprotein.svd.models import SVDModel
from tests.e2e.config import scaled_timeout

E2E_TIMEOUT = scaled_timeout(1.0)


@pytest.mark.e2e
def test_clustering_plm_happy_path(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Happy path: PLM embedding model → hierarchical clustering → HierarchicalClusteringResult."""
    emb_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        sequences=sequences,
    )
    assert isinstance(future, HierarchicalClusteringFuture)

    result = future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(result, HierarchicalClusteringResult)

    n = len(sequences)
    assert result.linkage.shape == (n - 1, 4), (
        f"Expected linkage shape ({n - 1}, 4), got {result.linkage.shape}"
    )
    assert len(result.leaf_order) == n
    assert result.sequences == sequences


@pytest.mark.e2e
def test_clustering_svd_features(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """SVD features: fit an SVD model, then cluster on its reduced embeddings."""
    embedding_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length

    # fit SVD first
    svd_future = embedding_model.fit_svd(sequences=sequences, n_components=32)
    svd_model = svd_future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(svd_model, SVDModel)

    # cluster using SVD features — reduction must be None
    future = session.clustering.hierarchical(
        model=svd_model,
        sequences=sequences,
    )
    assert isinstance(future, HierarchicalClusteringFuture)

    result = future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(result, HierarchicalClusteringResult)

    n = len(sequences)
    assert result.linkage.shape == (n - 1, 4)
    assert len(result.leaf_order) == n


@pytest.mark.e2e
@pytest.mark.parametrize(
    "linkage_method,metric",
    [
        ("ward", "euclidean"),
        ("average", "cosine"),
        ("complete", "correlation"),
    ],
)
def test_clustering_linkage_parameters(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
    linkage_method: str,
    metric: str,
):
    """Parametrize over (linkage_method, metric) combos; verify result shape and metadata."""
    emb_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        linkage_method=linkage_method,
        metric=metric,
        sequences=sequences,
    )
    result = future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(result, HierarchicalClusteringResult)

    n = len(sequences)
    assert result.linkage.shape == (n - 1, 4)
    assert len(result.leaf_order) == n

    # metadata round-trip
    md = future.metadata
    assert md.linkage_method is not None and md.linkage_method.value == linkage_method
    assert md.metric is not None and md.metric.value == metric


@pytest.mark.e2e
def test_clustering_from_assay(
    session: OpenProtein,
    fixture_lookup,
):
    """Cluster from an assay dataset (uses assay_small fixture via fixture_lookup)."""
    assay: AssayDataset = fixture_lookup("assay_small")
    emb_model = session.embedding.get_model("prot-seq")

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        assay=assay,
    )
    assert isinstance(future, HierarchicalClusteringFuture)

    result = future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(result, HierarchicalClusteringResult)
    assert result.linkage.ndim == 2
    assert result.linkage.shape[1] == 4


@pytest.mark.e2e
def test_clustering_retrieval_by_id(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Retrieve a clustering job by ID and verify it returns the same result."""
    emb_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        sequences=sequences,
    )
    result = future.wait(timeout=E2E_TIMEOUT)
    clustering_id = future.id

    # fetch by ID
    retrieved = session.clustering.get(clustering_id)
    assert isinstance(retrieved, HierarchicalClusteringFuture)
    assert retrieved.id == clustering_id

    retrieved_result = retrieved.wait(timeout=E2E_TIMEOUT)
    assert isinstance(retrieved_result, HierarchicalClusteringResult)
    assert retrieved_result.linkage.shape == result.linkage.shape


@pytest.mark.e2e
def test_clustering_list_filter(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """A just-created hierarchical job appears in the list filtered by method."""
    emb_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length[:10]

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        sequences=sequences,
    )
    created_id = future.id

    jobs = session.clustering.list(method="hierarchical")
    ids = [j.id for j in jobs]
    assert created_id in ids, (
        f"Newly created job {created_id!r} not found in list: {ids}"
    )


@pytest.mark.e2e
def test_clustering_validation_error_too_few_sequences(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Client-side validation: fewer than 2 sequences raises InvalidParameterError."""
    emb_model = session.embedding.get_model("prot-seq")

    with pytest.raises(InvalidParameterError, match="at least 2"):
        session.clustering.hierarchical(
            model=emb_model,
            reduction=ReductionType.MEAN,
            sequences=[b"MKTA"],
        )


@pytest.mark.e2e
def test_clustering_validation_error_ward_cosine(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Client-side validation: ward+cosine is rejected (ward requires euclidean)."""
    emb_model = session.embedding.get_model("prot-seq")

    with pytest.raises(InvalidParameterError, match="ward"):
        session.clustering.hierarchical(
            model=emb_model,
            reduction=ReductionType.MEAN,
            linkage_method="ward",
            metric="cosine",
            sequences=test_sequences_same_length[:10],
        )


@pytest.mark.e2e
def test_clustering_validation_error_svd_with_reduction(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Client-side validation: passing reduction with an SVD model raises InvalidParameterError."""
    embedding_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length

    svd_future = embedding_model.fit_svd(sequences=sequences, n_components=32)
    svd_model = svd_future.wait(timeout=E2E_TIMEOUT)
    assert isinstance(svd_model, SVDModel)

    with pytest.raises(InvalidParameterError, match="reduction"):
        session.clustering.hierarchical(
            model=svd_model,
            reduction=ReductionType.MEAN,
            sequences=sequences,
        )


@pytest.mark.e2e
def test_clustering_delete(
    session: OpenProtein,
    test_sequences_same_length: list[bytes],
):
    """Delete a clustering job; subsequent get raises HTTPError."""
    emb_model = session.embedding.get_model("prot-seq")
    sequences = test_sequences_same_length[:10]

    future = session.clustering.hierarchical(
        model=emb_model,
        reduction=ReductionType.MEAN,
        sequences=sequences,
    )
    # wait for the job to be ready before deleting
    future.wait(timeout=E2E_TIMEOUT)
    clustering_id = future.id

    result = future.delete()
    assert result is True

    # poll until unavailable (may be async)
    for _ in range(10):
        try:
            session.clustering.get(clustering_id)
        except HTTPError:
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"Expected deleted clustering job {clustering_id!r} to become unavailable")
