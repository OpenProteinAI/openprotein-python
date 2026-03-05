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
