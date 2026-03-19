from unittest.mock import MagicMock, patch

import numpy as np

from openprotein.embeddings.future import (
    EmbeddingsGenerateFuture,
    EmbeddingsScoreFuture,
    EmbeddingsScoreSingleSiteFuture,
)


# --- Generate ---


def test_embeddings_generate_future_stream_without_query_id_column(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["name", "sequence", "score_a", "score_b"],
            ["sample-1", "ACDE", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_generate_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsGenerateFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].name == "sample-1"
    assert results[0].sequence == "ACDE"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id is None


def test_embeddings_generate_future_stream_with_query_id_column(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["query_id", "name", "sequence", "score_a", "score_b"],
            ["query-abc", "sample-1", "ACDE", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_generate_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsGenerateFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].name == "sample-1"
    assert results[0].sequence == "ACDE"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id == "query-abc"


def test_embeddings_generate_future_stream_with_empty_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["query_id", "name", "sequence", "score_a", "score_b"],
            ["", "sample-1", "ACDE", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_generate_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsGenerateFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].query_id is None


# --- Score ---


def test_embeddings_score_future_stream_without_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["name", "sequence", "score_a", "score_b"],
            ["sample-1", "ACDE", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_score_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsScoreFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].name == "sample-1"
    assert results[0].sequence == "ACDE"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id is None


def test_embeddings_score_future_stream_with_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["query_id", "name", "sequence", "score_a", "score_b"],
            ["query-abc", "sample-1", "ACDE", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_score_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsScoreFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].name == "sample-1"
    assert results[0].sequence == "ACDE"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id == "query-abc"


def test_embeddings_score_future_stream_with_empty_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["query_id", "name", "sequence", "score_a"],
            ["", "sample-1", "ACDE", "3.0"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_score_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsScoreFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].query_id is None


# --- SingleSite ---


def test_embeddings_single_site_future_stream_without_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["mut_code", "score_a", "score_b"],
            ["A1G", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_score_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsScoreSingleSiteFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].mut_code == "A1G"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id is None


def test_embeddings_single_site_future_stream_with_query_id(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["query_id", "mut_code", "score_a", "score_b"],
            ["query-abc", "A1G", "1.5", "2.5"],
        ]
    )

    with patch(
        "openprotein.embeddings.future.api.request_get_score_result",
        return_value=stream_rows,
    ):
        future = EmbeddingsScoreSingleSiteFuture(session=mock_session, job=job)
        results = future.get()

    assert len(results) == 1
    assert results[0].mut_code == "A1G"
    assert np.array_equal(results[0].score, np.array([1.5, 2.5]))
    assert results[0].query_id == "query-abc"
