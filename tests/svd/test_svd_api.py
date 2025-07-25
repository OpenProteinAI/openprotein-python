"""Test the api for the svd domain."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobType
from openprotein.svd import api
from openprotein.svd.schemas import SVDMetadata


def test_svd_list_get(mock_session: MagicMock):
    """Test svd_list_get."""
    mock_session.get.return_value.json.return_value = [
        {
            "id": "svd-1",
            "status": "SUCCESS",
            "model_id": "model-1",
            "n_components": 10,
        }
    ]
    result = api.svd_list_get(mock_session)
    mock_session.get.assert_called_once_with("v1/embeddings/svd")
    assert len(result) == 1
    assert isinstance(result[0], SVDMetadata)


def test_svd_get(mock_session: MagicMock):
    """Test svd_get."""
    mock_session.get.return_value.json.return_value = {
        "id": "svd-1",
        "status": "SUCCESS",
        "model_id": "model-1",
        "n_components": 10,
    }
    result = api.svd_get(mock_session, "svd-1")
    mock_session.get.assert_called_once_with("v1/embeddings/svd/svd-1")
    assert isinstance(result, SVDMetadata)


def test_svd_delete(mock_session: MagicMock):
    """Test svd_delete."""
    mock_session.delete.return_value.status_code = 200
    result = api.svd_delete(mock_session, "svd-1")
    mock_session.delete.assert_called_once_with("v1/embeddings/svd/svd-1")
    assert result is True


def test_svd_fit_post_with_sequences(mock_session: MagicMock):
    """Test svd_fit_post with sequences."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "job_type": JobType.svd_fit.value,
        "status": "SUCCESS",
        "created_date": "2023-01-01T00:00:00",
    }
    api.svd_fit_post(mock_session, "model-1", sequences=["ACGT"])
    mock_session.post.assert_called_once()
    call_args, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["sequences"] == ["ACGT"]
    assert "assay_id" not in call_kwargs["json"]


def test_svd_fit_post_with_assay_id(mock_session: MagicMock):
    """Test svd_fit_post with an assay_id."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "job_type": JobType.svd_fit.value,
        "status": "SUCCESS",
        "created_date": "2023-01-01T00:00:00",
    }
    api.svd_fit_post(mock_session, "model-1", assay_id="assay-1")
    mock_session.post.assert_called_once()
    call_args, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["assay_id"] == "assay-1"
    assert "sequences" not in call_kwargs["json"]


def test_svd_fit_post_invalid_params(mock_session: MagicMock):
    """Test svd_fit_post with invalid parameters."""
    with pytest.raises(InvalidParameterError):
        api.svd_fit_post(mock_session, "model-1")  # Neither sequences nor assay_id
    with pytest.raises(InvalidParameterError):
        api.svd_fit_post(
            mock_session, "model-1", sequences=["ACGT"], assay_id="assay-1"
        )  # Both


def test_embed_decode():
    """Test embed_decode."""
    arr = np.array([1.0, 2.0, 3.0])
    # The api uses np.save, which includes a header.
    # We need to create the bytes in the same way.
    with np.testing.assert_raises(ValueError):
        api.embed_decode(arr.tobytes())
