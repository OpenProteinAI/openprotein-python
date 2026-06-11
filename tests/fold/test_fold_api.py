"""Test the api for the fold domain."""

import io
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from requests import Response

from openprotein.base import APISession
from openprotein.common.model_metadata import ModelMetadata
from openprotein.errors import HTTPError
from openprotein.fold import api
from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs.schemas import JobType


def test_fold_models_list_get(mock_session: MagicMock):
    """Test fold_models_list_get."""
    mock_session.get.return_value.json.return_value = ["model1", "model2"]
    result = api.fold_models_list_get(mock_session)
    mock_session.get.assert_called_once_with("v1/fold/models")
    assert result == ["model1", "model2"]


def test_fold_model_get(mock_session: MagicMock):
    """Test fold_model_get."""
    mock_session.get.return_value.json.return_value = {
        "model_id": "model1",
        "description": {"summary": "A test model"},
        "dimension": 128,
        "output_types": ["pdb"],
        "input_tokens": ["protein"],
        "token_descriptions": [],
    }
    result = api.fold_model_get(mock_session, "model1")
    mock_session.get.assert_called_once_with("v1/fold/models/model1")
    assert isinstance(result, ModelMetadata)
    assert result.id == "model1"


def test_fold_get(mock_session: MagicMock):
    """Test fold_get."""
    mock_session.get.return_value.json.return_value = {
        "job_id": "job1",
        "model_id": "model1",
    }
    result = api.fold_get(mock_session, "job1")
    mock_session.get.assert_called_once_with("v1/fold/job1")
    assert isinstance(result, FoldMetadata)
    assert result.job_id == "job1"


def test_fold_get_sequences(mock_session: MagicMock):
    """Test fold_get_sequences."""
    mock_session.get.return_value.json.return_value = ["seq1".encode(), "seq2".encode()]
    result = api.fold_get_sequences(mock_session, "job1")
    mock_session.get.assert_called_once_with("v1/fold/job1/sequences")
    assert result == ["seq1".encode(), "seq2".encode()]


def test_fold_get_sequence_result(mock_session: MagicMock):
    """Test fold_get_sequence_result."""
    mock_session.get.return_value.content = b"pdb_content"
    result = api.fold_get_sequence_result(mock_session, "job1", "seq1")
    mock_session.get.assert_called_once_with(
        "v1/fold/job1/seq1", params={"format": "mmcif"}
    )
    assert result == b"pdb_content"


def test_fold_get_extra_result_ipae(mock_session: MagicMock):
    """ipae is a valid single-unit extra-result key backed by a shape-(1,) npy."""
    buf = io.BytesIO()
    np.save(buf, np.array([0.42], dtype=np.float32))
    mock_session.get.return_value.content = buf.getvalue()
    result = api.fold_get_extra_result(mock_session, "job1", 0, "ipae")
    mock_session.get.assert_called_once_with("v1/fold/job1/0/ipae")
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)


def test_fold_get_batch_extra_result_npy(mock_session: MagicMock):
    """Batch npy key returns a single stacked np.ndarray via /results/{key}."""
    buf = io.BytesIO()
    stacked = np.zeros((3, 10, 10), dtype=np.float32)
    np.save(buf, stacked)
    mock_session.get.return_value.content = buf.getvalue()
    result = api.fold_get_batch_extra_result(mock_session, "job1", "pae")
    mock_session.get.assert_called_once_with("v1/fold/job1/results/pae")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 10, 10)


def test_fold_get_batch_extra_result_ipae(mock_session: MagicMock):
    """Batch iPAE returns a shape-(N,) np.ndarray."""
    buf = io.BytesIO()
    np.save(buf, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    mock_session.get.return_value.content = buf.getvalue()
    result = api.fold_get_batch_extra_result(mock_session, "job1", "ipae")
    mock_session.get.assert_called_once_with("v1/fold/job1/results/ipae")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_fold_get_batch_extra_result_json(mock_session: MagicMock):
    """Batch json key returns a length-N list with null→None for missing units."""
    mock_session.get.return_value.json.return_value = [
        [{"score": 0.9}],
        None,
        [{"score": 0.7}],
    ]
    result = api.fold_get_batch_extra_result(mock_session, "job1", "confidence")
    mock_session.get.assert_called_once_with("v1/fold/job1/results/confidence")
    assert result == [[{"score": 0.9}], None, [{"score": 0.7}]]


def test_fold_get_batch_extra_result_rejects_unknown_key(mock_session: MagicMock):
    """Unknown keys are rejected before the HTTP call."""
    with pytest.raises(ValueError, match="Unexpected key"):
        api.fold_get_batch_extra_result(mock_session, "job1", "unknown")  # ty: ignore[no-matching-overload]
    mock_session.get.assert_not_called()


def test_fold_models_post(mock_session: MagicMock):
    """Test fold_models_post."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job1",
        "status": "PENDING",
        "job_type": JobType.embeddings_fold.value,
        "created_date": "2024-01-01T00:00:00",
    }
    result = api.fold_models_post(
        mock_session, "model1", sequences=[[{"protein": {"sequence": "AAA"}}]]
    )
    mock_session.post.assert_called_once_with(
        "v1/fold/models/model1",
        json={"sequences": [[{"protein": {"sequence": "AAA"}}]]},
        params=None,
    )
    assert isinstance(result, FoldJob)
    assert result.job_id == "job1"


def test_fold_models_post_force_recompute_sends_param(mock_session: MagicMock):
    """force_recompute=True sends ?force=true."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job1",
        "status": "PENDING",
        "job_type": JobType.embeddings_fold.value,
        "created_date": "2024-01-01T00:00:00",
    }
    api.fold_models_post(
        mock_session,
        "model1",
        sequences=[[{"protein": {"sequence": "AAA"}}]],
        force_recompute=True,
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"force": "true"}


def test_fold_models_post_no_force_recompute_by_default(mock_session: MagicMock):
    """Without force_recompute, no ?force param is sent."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job1",
        "status": "PENDING",
        "job_type": JobType.embeddings_fold.value,
        "created_date": "2024-01-01T00:00:00",
    }
    api.fold_models_post(
        mock_session, "model1", sequences=[[{"protein": {"sequence": "AAA"}}]]
    )
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("params") is None
