"""
L2 integration tests for the data domain.
These tests verify that the high-level DataAPI and AssayDataset objects
correctly call the low-level API functions, mocking only the session calls.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data.assaydataset import AssayDataset
from openprotein.data.data import DataAPI
from openprotein.data.schemas import AssayDataPage, AssayDataRow, AssayMetadata


@pytest.fixture
def sample_assay_metadata_dict() -> dict:
    """Fixture for a sample AssayMetadata dictionary."""
    return {
        "assay_name": "Test Assay",
        "assay_description": "A test assay",
        "assay_id": "assay123",
        "original_filename": "data.csv",
        "created_date": "2023-01-01T00:00:00",
        "num_rows": 100,
        "num_entries": 200,
        "measurement_names": ["activity"],
        "sequence_length": 150,
    }


# ===================================
# DataAPI Tests
# ===================================


def test_data_api_list(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    """Test DataAPI.list() method."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [sample_assay_metadata_dict]
    data_api = DataAPI(mock_session)

    result = data_api.list()

    mock_session.get.assert_called_once_with("v1/assaydata", params={})
    assert len(result) == 1
    assert isinstance(result[0], AssayDataset)
    assert result[0].id == "assay123"


def test_data_api_create(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    """Test DataAPI.create() method."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = sample_assay_metadata_dict
    data_api = DataAPI(mock_session)
    df = pd.DataFrame({"sequence": ["SEQ"], "activity": [1.0]})

    result = data_api.create(df, "New Assay", "Description")

    mock_session.post.assert_called_once()
    assert isinstance(result, AssayDataset)
    assert result.name == "Test Assay"


def test_data_api_get(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    """Test DataAPI.get() method."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = sample_assay_metadata_dict
    data_api = DataAPI(mock_session)

    dataset = data_api.get("assay123")

    mock_session.get.assert_called_once_with(
        "v1/assaydata/metadata", params={"assay_id": "assay123"}
    )
    assert isinstance(dataset, AssayDataset)
    assert dataset.id == "assay123"


# ===================================
# AssayDataset Tests
# ===================================


def test_assay_dataset_update(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    """Test AssayDataset.update() method."""
    updated_meta_dict = sample_assay_metadata_dict.copy()
    updated_meta_dict["assay_name"] = "Updated Name"
    mock_session.put.return_value.status_code = 200
    mock_session.put.return_value.json.return_value = updated_meta_dict

    dataset = AssayDataset(mock_session, AssayMetadata(**sample_assay_metadata_dict))
    dataset.update(assay_name="Updated Name")

    mock_session.put.assert_called_once_with(
        "v1/assaydata/assay123", data={"assay_name": "Updated Name"}
    )
    assert dataset.name == "Updated Name"


def test_assay_dataset_get_slice(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    """Test AssayDataset.get_slice() method."""
    row = {"mut_sequence": "SEQ", "measurement_values": [1.0]}
    page_data = {
        "assaymetadata": sample_assay_metadata_dict,
        "page_size": 1,
        "page_offset": 0,
        "assaydata": [row],
    }
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = page_data

    dataset = AssayDataset(mock_session, AssayMetadata(**sample_assay_metadata_dict))
    df = dataset.get_slice(0, 1)

    mock_session.get.assert_called_once_with(
        "v1/assaydata/assay123",
        params={"page_offset": 0, "page_size": 1, "format": "wide"},
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert "sequence" in df.columns
