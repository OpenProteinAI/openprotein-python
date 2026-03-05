from unittest.mock import MagicMock, patch

import numpy as np

from openprotein.embeddings.future import EmbeddingsGenerateFuture


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


def test_embeddings_generate_future_stream_with_trailing_query_id_column(mock_session):
    job = MagicMock()
    job.job_id = "job-123"

    stream_rows = iter(
        [
            ["name", "sequence", "score_a", "score_b", "query_id"],
            ["sample-1", "ACDE", "1.5", "2.5", "query-abc"],
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
