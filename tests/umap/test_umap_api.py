"""Test the api for the umap domain."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.common import FeatureType
from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobType
from openprotein.umap import api
from openprotein.umap.schemas import UMAPMetadata


def test_umap_list_get(mock_session: MagicMock):
    """Test umap_list_get."""
    mock_session.get.return_value.json.return_value = [
        {
            "id": "umap-1",
            "status": "SUCCESS",
            "model_id": "model-1",
            "feature_type": "PLM",
        }
    ]
    result = api.umap_list_get(mock_session)
    mock_session.get.assert_called_once_with("v1/umap")
    assert len(result) == 1
    assert isinstance(result[0], UMAPMetadata)


def test_umap_fit_post_invalid_params(mock_session: MagicMock):
    """Test umap_fit_post with invalid parameters."""
    with pytest.raises(InvalidParameterError):
        api.umap_fit_post(
            mock_session, "model-1", FeatureType.PLM
        )  # Neither sequences nor assay_id
    with pytest.raises(InvalidParameterError):
        api.umap_fit_post(
            mock_session,
            "model-1",
            FeatureType.PLM,
            sequences=["ACGT"],
            assay_id="assay-1",
        )  # Both


def test_embed_decode(mock_session: MagicMock):
    """Test embed_decode."""
    arr = np.array([1.0, 2.0, 3.0])
    with np.testing.assert_raises(ValueError):
        api.embed_decode(arr.tobytes())


def test_embed_batch_decode(mock_session: MagicMock):
    """Test embed_batch_decode."""
    csv_data = "sequence,0,1\nACGT,1.0,2.0\n"
    arr = api.embed_batch_decode(csv_data.encode("utf-8"))
    assert arr.shape == (1, 2)
    assert np.allclose(arr, np.array([[1.0, 2.0]]))
