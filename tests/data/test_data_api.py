from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from openprotein.base import APISession
from openprotein.data.api import (
    assaydata_list,
    assaydata_page_get,
    assaydata_post,
    assaydata_put,
    get_assay_metadata,
    list_models,
)
from openprotein.data.schemas import AssayDataPage, AssayMetadata
from openprotein.errors import APIError


@pytest.fixture
def sample_assay_metadata() -> dict:
    """Fixture for sample assay metadata dictionary."""
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


def test_list_models(mock_session: MagicMock) -> None:
    """Test the list_models API function."""
    mock_session.get.return_value.json.return_value = [{"model_id": "1"}]
    result = list_models(mock_session, "assay123")
    mock_session.get.assert_called_once_with(
        "v1/models", params={"assay_id": "assay123"}
    )
    assert result == [{"model_id": "1"}]


def test_assaydata_post(mock_session: MagicMock, sample_assay_metadata: dict) -> None:
    """Test the assaydata_post API function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_assay_metadata
    mock_session.post.return_value = mock_response

    assay_file = "file_content"
    result = assaydata_post(mock_session, assay_file, "Test Assay", "A description")

    mock_session.post.assert_called_once_with(
        "v1/assaydata",
        files={"assay_data": assay_file},
        data={"assay_name": "Test Assay", "assay_description": "A description"},
    )
    assert isinstance(result, AssayMetadata)
    assert result.assay_id == "assay123"


def test_assaydata_post_error(mock_session: MagicMock) -> None:
    """Test assaydata_post for non-200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_session.post.return_value = mock_response
    with pytest.raises(APIError):
        assaydata_post(mock_session, "file", "name")


def test_assaydata_list(mock_session: MagicMock, sample_assay_metadata: dict) -> None:
    """Test the assaydata_list API function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [sample_assay_metadata]
    mock_session.get.return_value = mock_response

    result = assaydata_list(mock_session)
    mock_session.get.assert_called_once_with("v1/assaydata")
    assert isinstance(result[0], AssayMetadata)
    assert len(result) == 1


def test_assaydata_list_error(mock_session: MagicMock) -> None:
    """Test assaydata_list for non-200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_session.get.return_value = mock_response
    with pytest.raises(APIError):
        assaydata_list(mock_session)


def test_get_assay_metadata(
    mock_session: MagicMock, sample_assay_metadata: dict
) -> None:
    """Test the get_assay_metadata API function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_assay_metadata
    mock_session.get.return_value = mock_response

    result = get_assay_metadata(mock_session, "assay123")
    mock_session.get.assert_called_once_with(
        "v1/assaydata/metadata", params={"assay_id": "assay123"}
    )
    assert isinstance(result, AssayMetadata)
    assert result.assay_id == "assay123"


def test_get_assay_metadata_not_found(mock_session: MagicMock) -> None:
    """Test get_assay_metadata for an empty response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Simulate not found
    mock_session.get.return_value = mock_response
    with pytest.raises(APIError, match="No assay with id=notfound found"):
        get_assay_metadata(mock_session, "notfound")


def test_assaydata_put(mock_session: MagicMock, sample_assay_metadata: dict) -> None:
    """Test the assaydata_put API function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_assay_metadata
    mock_session.put.return_value = mock_response

    result = assaydata_put(mock_session, "assay123", "New Name")
    mock_session.put.assert_called_once_with(
        "v1/assaydata/assay123", data={"assay_name": "New Name"}
    )
    assert isinstance(result, AssayMetadata)


def test_assaydata_page_get(
    mock_session: MagicMock, sample_assay_metadata: dict
) -> None:
    """Test the assaydata_page_get API function."""
    page_data = {
        "assaymetadata": sample_assay_metadata,
        "page_size": 1,
        "page_offset": 0,
        "assaydata": [{"mut_sequence": "SEQ", "measurement_values": [1.0]}],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = page_data
    mock_session.get.return_value = mock_response

    result = assaydata_page_get(mock_session, "assay123", page_size=1, page_offset=0)

    expected_params = {"page_offset": 0, "page_size": 1, "format": "wide"}
    mock_session.get.assert_called_once_with(
        "v1/assaydata/assay123", params=expected_params
    )
    assert isinstance(result, AssayDataPage)
    assert result.page_size == 1
