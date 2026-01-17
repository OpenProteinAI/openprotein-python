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


def test_fold_get_complex_result(mock_session: MagicMock):
    """Test fold_get_complex_result."""
    mock_session.get.return_value.content = b"pdb_content"
    result = api.fold_get_complex_result(mock_session, "job1", "pdb")
    mock_session.get.assert_called_once_with(
        "v1/fold/job1/complex", params={"format": "pdb"}
    )
    assert result == b"pdb_content"


def test_fold_get_complex_extra_result_pae(mock_session: MagicMock):
    """Test fold_get_complex_extra_result for pae."""
    arr = np.array([1, 2, 3])
    with io.BytesIO() as f:
        np.save(f, arr)
        f.seek(0)
        mock_session.get.return_value.content = f.read()

    result = api.fold_get_complex_extra_result(mock_session, "job1", "pae")
    mock_session.get.assert_called_once_with("v1/fold/job1/complex/pae")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_fold_get_complex_extra_result_affinity(mock_session: MagicMock):
    """Test fold_get_complex_extra_result for affinity."""
    mock_session.get.return_value.json.return_value = [{"key": "value"}]
    result = api.fold_get_complex_extra_result(mock_session, "job1", "affinity")
    mock_session.get.assert_called_once_with("v1/fold/job1/complex/affinity")
    assert isinstance(result, list)
    assert result == [{"key": "value"}]


def test_fold_get_complex_extra_result_invalid_key(mock_session: MagicMock):
    """Test fold_get_complex_extra_result for invalid key."""
    with pytest.raises(ValueError):
        api.fold_get_complex_extra_result(mock_session, "job1", "invalid_key")  # type: ignore - testing runtime error


def test_fold_get_complex_extra_result_affinity_not_found(mock_session: MagicMock):
    """Test fold_get_complex_extra_result for affinity not found."""
    mock_response = Mock(spec=Response)
    mock_response.status_code = 400
    mock_response.url = "http://test.url"
    mock_session.get.side_effect = HTTPError(mock_response)
    with pytest.raises(ValueError, match="affinity not found for request"):
        api.fold_get_complex_extra_result(mock_session, "job1", "affinity")


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
    )
    assert isinstance(result, FoldJob)
    assert result.job_id == "job1"
