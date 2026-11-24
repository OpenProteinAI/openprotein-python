from datetime import datetime

import pytest
from pydantic import ValidationError

from openprotein.data.schemas import AssayDataPage, AssayDataRow, AssayMetadata
from openprotein.jobs.schemas import JobStatus


def test_assay_metadata_schema_valid() -> None:
    data = {
        "assay_name": "Test Assay",
        "assay_description": "A test assay",
        "assay_id": "assay123",
        "original_filename": "data.csv",
        "created_date": datetime.now(),
        "num_rows": 100,
        "num_entries": 200,
        "measurement_names": ["activity", "stability"],
        "sequence_length": 150,
    }
    assay_metadata = AssayMetadata(**data)
    for key, value in data.items():
        assert getattr(assay_metadata, key) == value


def test_assay_metadata_schema_optional_fields() -> None:
    data = {
        "assay_name": "Test Assay",
        "assay_description": "A test assay",
        "assay_id": "assay123",
        "original_filename": "data.csv",
        "created_date": datetime.now(),
        "num_rows": 100,
        "num_entries": 200,
        "measurement_names": ["activity", "stability"],
    }
    assay_metadata = AssayMetadata(**data)
    assert assay_metadata.sequence_length is None
    # has_name_column defaults to False so older servers without the field still parse
    assert assay_metadata.has_name_column is False


@pytest.mark.parametrize("has_name_column", [True, False])
def test_assay_metadata_schema_has_name_column(has_name_column: bool) -> None:
    data = {
        "assay_name": "Test Assay",
        "assay_description": "A test assay",
        "assay_id": "assay123",
        "original_filename": "data.csv",
        "created_date": datetime.now(),
        "num_rows": 100,
        "num_entries": 200,
        "measurement_names": ["activity"],
        "has_name_column": has_name_column,
    }
    assay_metadata = AssayMetadata(**data)
    assert assay_metadata.has_name_column is has_name_column


def test_assay_metadata_schema_invalid_data() -> None:
    with pytest.raises(ValidationError):
        AssayMetadata(
            assay_name="Test Assay",
            assay_description="A test assay",
            assay_id="assay123",
            original_filename="data.csv",
            created_date=datetime.now(),
            num_rows="not-an-int",  # type: ignore - testing invalid type
            num_entries=200,
            measurement_names=["activity", "stability"],
        )


def test_assay_data_row_schema_valid() -> None:
    data = {"mut_sequence": "SEQ", "measurement_values": [1.0, 2.5, None]}
    assay_data_row = AssayDataRow(**data)
    assert assay_data_row.mut_sequence == "SEQ"
    assert assay_data_row.measurement_values == [1.0, 2.5, None]


def test_assay_data_row_schema_invalid() -> None:
    with pytest.raises(ValidationError):
        AssayDataRow(mut_sequence="SEQ", measurement_values=["not-a-float"])  # type: ignore - testing invalid type


def test_assay_data_page_schema_valid() -> None:
    metadata_data = {
        "assay_name": "Test Assay",
        "assay_description": "A test assay",
        "assay_id": "assay123",
        "original_filename": "data.csv",
        "created_date": datetime.now(),
        "num_rows": 100,
        "num_entries": 200,
        "measurement_names": ["activity"],
    }
    row_data = {"mut_sequence": "SEQ", "measurement_values": [1.0]}

    page_data = {
        "assaymetadata": metadata_data,
        "page_size": 1,
        "page_offset": 0,
        "assaydata": [row_data],
    }

    page = AssayDataPage(**page_data)
    assert isinstance(page.assaymetadata, AssayMetadata)
    assert len(page.assaydata) == 1
    assert isinstance(page.assaydata[0], AssayDataRow)
    assert page.assaydata[0].mut_sequence == "SEQ"


_BASE_METADATA = {
    "assay_name": "Test Assay",
    "assay_description": "A test assay",
    "assay_id": "assay123",
    "original_filename": "data.csv",
    "created_date": "2023-01-01T00:00:00",
    "num_rows": 100,
    "num_entries": 200,
    "measurement_names": ["activity"],
}


def test_assay_metadata_null_status_reads_success_with_no_job() -> None:
    metadata = AssayMetadata(**_BASE_METADATA, status=None)
    assert metadata.status == JobStatus.SUCCESS
    assert metadata.align_job_id is None


def test_assay_metadata_without_status_reads_success_with_no_job() -> None:
    metadata = AssayMetadata(**_BASE_METADATA)
    assert metadata.status == JobStatus.SUCCESS
    assert metadata.align_job_id is None


def test_assay_metadata_parses_a_pending_dataset() -> None:
    metadata = AssayMetadata(
        **{
            **_BASE_METADATA,
            "num_rows": 0,
            "num_entries": 0,
            "measurement_names": [],
            "status": "RUNNING",
            "align_job_id": "job1",
        }
    )
    assert metadata.status == JobStatus.RUNNING
    assert metadata.align_job_id == "job1"
    assert metadata.measurement_names == []


@pytest.mark.parametrize("status", ["FAILURE", "CANCELED"])
def test_assay_metadata_parses_terminal_job_statuses(status: str) -> None:
    assert AssayMetadata(**_BASE_METADATA, status=status).status == JobStatus(status)
