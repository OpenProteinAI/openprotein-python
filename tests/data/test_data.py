"""
L2 integration tests for the data domain.
These tests verify that the high-level DataAPI and AssayDataset objects
correctly call the low-level API functions, mocking only the session calls.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data.assaydataset import AssayDataset
from openprotein.data.data import DataAPI
from openprotein.data.schemas import AssayMetadata
from openprotein.errors import HTTPError
from openprotein.jobs.schemas import JobStatus


@pytest.fixture
def sample_assay_metadata_dict() -> dict:
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


# DataAPI Tests
def test_data_api_list(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [sample_assay_metadata_dict]
    data_api = DataAPI(mock_session)

    result = data_api.list()

    mock_session.get.assert_called_once_with(
        "v1/assaydata", params={"limit": 1000, "offset": 0}
    )
    assert len(result) == 1
    assert isinstance(result[0], AssayDataset)
    assert result[0].id == "assay123"


def test_data_api_create(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = sample_assay_metadata_dict
    data_api = DataAPI(mock_session)
    df = pd.DataFrame({"sequence": ["SEQ"], "activity": [1.0]})

    result = data_api.create(df, "New Assay", "Description")

    mock_session.post.assert_called_once()
    assert isinstance(result, AssayDataset)
    assert result.name == "Test Assay"


def test_data_api_get(mock_session: MagicMock, sample_assay_metadata_dict: dict):
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = sample_assay_metadata_dict
    data_api = DataAPI(mock_session)

    dataset = data_api.get("assay123")

    mock_session.get.assert_called_once_with(
        "v1/assaydata/metadata", params={"assay_id": "assay123"}
    )
    assert isinstance(dataset, AssayDataset)
    assert dataset.id == "assay123"


# AssayDataset Tests
def test_assay_dataset_update(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    updated_meta_dict = sample_assay_metadata_dict.copy()
    updated_meta_dict["assay_name"] = "Updated Name"
    mock_session.put.return_value.status_code = 200
    mock_session.put.return_value.json.return_value = updated_meta_dict

    dataset = AssayDataset(mock_session, AssayMetadata(**sample_assay_metadata_dict))
    dataset.update(assay_name="Updated Name")

    mock_session.put.assert_called_once_with(
        "v1/assaydata/assay123", json={"assay_name": "Updated Name"}
    )
    assert dataset.name == "Updated Name"


def test_assay_dataset_get_slice(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
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


def test_data_api_create_metadata_columns(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        **sample_assay_metadata_dict,
        "metadata_names": ["notes"],
    }
    data_api = DataAPI(mock_session)
    df = pd.DataFrame({"sequence": ["SEQ"], "activity": [1.0], "notes": ["run 1"]})

    result = data_api.create(df, "New Assay", metadata_columns=["notes"])

    sent = mock_session.post.call_args.kwargs["data"]
    assert sent["metadata_columns"] == '["notes"]'
    assert result.metadata_names == ["notes"]


def test_assay_dataset_get_slice_include_metadata(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    metadata = {**sample_assay_metadata_dict, "metadata_names": ["notes", "run"]}
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "assaymetadata": metadata,
        "page_size": 2,
        "page_offset": 0,
        "assaydata": [
            {
                "mut_sequence": "SEQ",
                "measurement_values": [1.0],
                "metadata_values": ["007", "r1"],
            },
            {
                "mut_sequence": "SEQ",
                "measurement_values": [None],
                "metadata_values": [None, "r2"],
            },
        ],
    }

    dataset = AssayDataset(mock_session, AssayMetadata(**metadata))
    df = dataset.get_slice(0, 2, include_metadata=True)

    assert mock_session.get.call_args.kwargs["params"]["include_metadata"] is True
    assert list(df.columns) == ["sequence", "activity", "notes", "run"]
    assert df["notes"].tolist() == ["007", None]
    assert df["run"].tolist() == ["r1", "r2"]


@pytest.mark.parametrize("metadata_values", [None, [], ["007"]])
def test_a_row_with_missing_metadata_reads_none_for_each_missing_value(
    mock_session: MagicMock, sample_assay_metadata_dict: dict, metadata_values
):
    metadata = {**sample_assay_metadata_dict, "metadata_names": ["notes", "run"]}
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "assaymetadata": metadata,
        "page_size": 1,
        "page_offset": 0,
        "assaydata": [
            {
                "mut_sequence": "SEQ",
                "measurement_values": [1.0],
                "metadata_values": metadata_values,
            }
        ],
    }

    dataset = AssayDataset(mock_session, AssayMetadata.model_validate(metadata))
    df = dataset.get_slice(0, 1, include_metadata=True)

    expected = (metadata_values or []) + [None] * (2 - len(metadata_values or []))
    assert [df["notes"][0], df["run"][0]] == expected


def test_assay_dataset_metadata_names_default(sample_assay_metadata_dict: dict):
    dataset = AssayDataset(MagicMock(), AssayMetadata(**sample_assay_metadata_dict))
    assert dataset.metadata_names == []


def test_data_api_create_empty_metadata_columns(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = sample_assay_metadata_dict
    data_api = DataAPI(mock_session)
    df = pd.DataFrame({"sequence": ["SEQ"], "activity": [1.0]})

    data_api.create(df, "New Assay", metadata_columns=[])

    sent = mock_session.post.call_args.kwargs["data"]
    assert sent["metadata_columns"] == "[]"


def test_data_api_create_metadata_columns_rejects_str(mock_session: MagicMock):
    data_api = DataAPI(mock_session)
    df = pd.DataFrame({"sequence": ["SEQ"], "notes": ["run 1"]})

    with pytest.raises(TypeError, match="list of column names"):
        data_api.create(df, "New Assay", metadata_columns="notes")

    mock_session.post.assert_not_called()


def test_assay_dataset_get_slice_include_metadata_without_metadata(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "assaymetadata": sample_assay_metadata_dict,
        "page_size": 1,
        "page_offset": 0,
        "assaydata": [{"mut_sequence": "SEQ", "measurement_values": [1.0]}],
    }

    dataset = AssayDataset(mock_session, AssayMetadata(**sample_assay_metadata_dict))
    df = dataset.get_slice(0, 1, include_metadata=True)

    assert list(df.columns) == ["sequence", "activity"]
    assert df["activity"].tolist() == [1.0]


def test_assay_dataset_exposes_status_and_job_id(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    dataset = AssayDataset(
        mock_session,
        AssayMetadata(
            **sample_assay_metadata_dict, status="RUNNING", align_job_id="job1"
        ),
    )
    assert dataset.status == JobStatus.RUNNING
    assert dataset.job_id == "job1"


def test_assay_dataset_without_a_job_reads_success_and_no_job_id(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    dataset = AssayDataset(
        mock_session, AssayMetadata(**sample_assay_metadata_dict, status=None)
    )
    assert dataset.status == JobStatus.SUCCESS
    assert dataset.job_id is None


def test_assay_dataset_get_slice_raises_a_409_for_a_dataset_that_is_not_ready(
    mock_session: MagicMock, sample_assay_metadata_dict: dict
):
    response = MagicMock()
    response.status_code = 409
    response.json.return_value = {}
    mock_session.get.side_effect = HTTPError(response)
    dataset = AssayDataset(mock_session, AssayMetadata(**sample_assay_metadata_dict))
    with pytest.raises(HTTPError) as raised:
        dataset.get_slice(0, 1)
    assert raised.value.status_code == 409
