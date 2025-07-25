from datetime import datetime

import pytest
from pydantic import ValidationError

from openprotein.data.schemas import AssayDataPage, AssayDataRow, AssayMetadata


def test_assay_metadata_schema_valid() -> None:
    """Test AssayMetadata schema with valid data."""
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
    """Test AssayMetadata with optional sequence_length."""
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


def test_assay_metadata_schema_invalid_data() -> None:
    """Test AssayMetadata with invalid data types."""
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
    """Test AssayDataRow schema with valid data."""
    data = {"mut_sequence": "SEQ", "measurement_values": [1.0, 2.5, None]}
    assay_data_row = AssayDataRow(**data)
    assert assay_data_row.mut_sequence == "SEQ"
    assert assay_data_row.measurement_values == [1.0, 2.5, None]


def test_assay_data_row_schema_invalid() -> None:
    """Test AssayDataRow with invalid data."""
    with pytest.raises(ValidationError):
        AssayDataRow(mut_sequence="SEQ", measurement_values=["not-a-float"])  # type: ignore - testing invalid type


def test_assay_data_page_schema_valid() -> None:
    """Test AssayDataPage schema with nested valid data."""
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
