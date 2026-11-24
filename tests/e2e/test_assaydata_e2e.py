"""E2E tests for the assay data domain."""

import os

import numpy as np
import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.data import AssayDataset
from openprotein.errors import APIError
from tests.utils.strings import random_string

TEST_ASSAY_DATA_DIR = "tests/data"
TEST_ASSAY_DATA_FILE = "AMIE_PSEAE_Whitehead.wide.15.csv"
TEST_ASSAY_DATA_FILEPATH = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_DATA_FILE)
TEST_ASSAY_NUM_ENTRIES = 39
TEST_ASSAY_NUM_ROWS = 15
TEST_ASSAY_SEQUENCE_LENGTH = 346
TEST_ASSAY_MEASUREMENT_NAMES = [
    "acetamide_normalized_fitness",
    "isobutyramide_normalized_fitness",
    "propionamide_normalized_fitness",
]


@pytest.mark.e2e
def test_assaydata_workflow_e2e(session: OpenProtein):
    """
    Upload a CSV as an assay, then read its metadata and data back.
    """
    # 1. Session is already connected via the session fixture.

    # 2. Reads a CSV
    df = pd.read_csv(TEST_ASSAY_DATA_FILEPATH)

    # 3. Upload to platform
    assay_name = f"Test {random_string(10)}"
    assay_description = f"This is a test description: {random_string(50)}"
    data = session.data.Assay.create(
        table=df, name=assay_name, description=assay_description
    )
    assert isinstance(data, AssayDataset)

    # 4. Validate metadata
    assert data.name == assay_name
    assert data.description == assay_description
    assert data.sequence_length == TEST_ASSAY_SEQUENCE_LENGTH
    assert data.metadata.num_entries == TEST_ASSAY_NUM_ENTRIES
    assert data.metadata.num_rows == TEST_ASSAY_NUM_ROWS
    assert data.measurement_names == TEST_ASSAY_MEASUREMENT_NAMES

    # Validate data
    out_df = data.get_slice(0, data.metadata.num_rows)
    assert isinstance(out_df, pd.DataFrame)
    assert (out_df.sequence == df.sequence).all()
    for measurement in data.measurement_names:
        assert (
            np.isclose(out_df[measurement], df[measurement])
            | (np.isnan(out_df[measurement]) & np.isnan(df[measurement]))
        ).all()


@pytest.mark.e2e
def test_assaydata_with_nan_values(session: OpenProtein):
    """
    Missing measurements survive the upload as NaN.
    """
    df = pd.read_csv(TEST_ASSAY_DATA_FILEPATH)

    assay_name = f"Test_NaN_{random_string(10)}"
    data = session.data.Assay.create(
        table=df, name=assay_name, description="Test with NaN"
    )

    # Verify NaN values are preserved
    out_df = data.get_slice(0, data.metadata.num_rows)
    for measurement in data.measurement_names:
        # Check that NaN positions match
        assert (np.isnan(out_df[measurement]) == np.isnan(df[measurement])).all()


@pytest.mark.e2e
def test_assaydata_retrieval_by_id(session: OpenProtein, assay_small: AssayDataset):
    # Retrieve by ID
    retrieved = session.data.Assay(id=assay_small.id)

    assert retrieved.id == assay_small.id
    assert retrieved.name == assay_small.name
    assert retrieved.sequence_length == assay_small.sequence_length
    assert retrieved.measurement_names == assay_small.measurement_names


@pytest.mark.e2e
def test_assaydata_slice_boundaries(session: OpenProtein, assay_small: AssayDataset):
    # Get first row
    df_first = assay_small.get_slice(0, 1)
    assert len(df_first) == 1

    # Get last row
    last_idx = assay_small.metadata.num_rows - 1
    df_last = assay_small.get_slice(last_idx, assay_small.metadata.num_rows)
    assert len(df_last) == 1

    # Get all rows
    df_all = assay_small.get_slice(0, assay_small.metadata.num_rows)
    assert len(df_all) == assay_small.metadata.num_rows


@pytest.mark.e2e
@pytest.mark.skip("todo: reject invalid slices")
def test_assaydata_invalid_slice(session: OpenProtein, assay_small: AssayDataset):
    # Out of bounds slice
    with pytest.raises(Exception):
        assay_small.get_slice(0, assay_small.metadata.num_rows + 100)

    # Negative indices
    with pytest.raises(Exception):
        assay_small.get_slice(-1, 5)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "assay_fixture",
    ["assay_small", "assay_medium", "assay_large"],
)
def test_assaydata_different_sizes(session: OpenProtein, assay_fixture: str, request):
    """
    Small, medium and large assay datasets all upload and read back.
    """
    # Get the fixture dynamically
    try:
        assay = request.getfixturevalue(assay_fixture)
    except pytest.FixtureLookupError:
        pytest.skip(f"Fixture {assay_fixture} not available")

    # Validate basic properties
    assert assay.id is not None
    assert assay.name is not None
    assert assay.sequence_length > 0
    assert len(assay.measurement_names) > 0

    # Validate data retrieval
    df = assay.get_slice(0, min(10, assay.metadata.num_rows))
    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    for measurement in assay.measurement_names:
        assert measurement in df.columns


@pytest.mark.e2e
def test_assaydata_list_datasets(session: OpenProtein):
    datasets = session.data.Assay.list()
    assert isinstance(datasets, list)
    # Should have at least the fixtures we created
    assert len(datasets) > 0


@pytest.mark.e2e
def test_assaydata_metadata_consistency(
    session: OpenProtein, assay_small: AssayDataset
):
    # Get the same dataset again
    retrieved = session.data.Assay(id=assay_small.id)

    # Metadata should match
    assert retrieved.metadata.num_entries == assay_small.metadata.num_entries
    assert retrieved.metadata.num_rows == assay_small.metadata.num_rows
    assert retrieved.sequence_length == assay_small.sequence_length
    assert retrieved.measurement_names == assay_small.measurement_names


@pytest.mark.e2e
@pytest.mark.parametrize("with_name_column", [True, False])
def test_assaydata_has_name_column(session: OpenProtein, with_name_column: bool):
    """Server should report has_name_column based on the uploaded CSV."""
    table = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG"],
            "activity": [0.1, 0.2, 0.3],
        }
    )
    if with_name_column:
        table.insert(0, "name", ["s1", "s2", "s3"])

    assay_name = f"Test_name_col_{with_name_column}_{random_string(10)}"
    data = session.data.Assay.create(
        table=table, name=assay_name, description="name col test"
    )

    assert data.metadata.has_name_column is with_name_column


@pytest.mark.e2e
def test_assaydata_sequence_validation(session: OpenProtein, assay_small: AssayDataset):
    df = assay_small.get_slice(0, assay_small.metadata.num_rows)

    for seq in df.sequence:
        assert len(seq) == assay_small.sequence_length, (
            f"Sequence length mismatch: expected {assay_small.sequence_length}, "
            f"got {len(seq)}"
        )


NOTES_TABLE = pd.DataFrame(
    {
        "name": ["s1", "s2", "s3"],
        "sequence": ["AAAA", "CCCC", "GGGG"],
        "activity": [0.1, None, 0.3],
        "notes": ["007", None, 'run 2, "rerun"\nsee lab book'],
        "run": ["1.0", "r2", "NA"],
    }
)


@pytest.mark.e2e
def test_assaydata_metadata_columns_e2e(session: OpenProtein):
    """Metadata columns are stored verbatim per row and kept out of the measurements."""
    data = session.data.Assay.create(
        table=NOTES_TABLE.copy(),
        name=f"Test_metadata_{random_string(10)}",
        description="metadata columns",
        metadata_columns=["notes", "run"],
    )

    assert data.measurement_names == ["activity"]
    assert data.metadata_names == ["notes", "run"]
    assert data.metadata.num_entries == 2
    assert data.metadata.num_rows == 3

    out = data.get_slice(0, 3, include_metadata=True)
    assert list(out.columns) == ["sequence", "activity", "notes", "run"]
    assert out["sequence"].tolist() == ["AAAA", "CCCC", "GGGG"]
    assert out["notes"].tolist() == ["007", None, 'run 2, "rerun"\nsee lab book']
    assert out["run"].tolist() == ["1.0", "r2", "NA"]

    plain = data.get_slice(0, 3)
    assert list(plain.columns) == ["sequence", "activity"]

    retrieved = session.data.Assay(id=data.id)
    assert retrieved.metadata_names == ["notes", "run"]
    assert retrieved.measurement_names == ["activity"]


@pytest.mark.e2e
def test_assaydata_metadata_columns_empty_list_e2e(session: OpenProtein):
    """An empty metadata_columns list uploads like a plain assay."""
    data = session.data.Assay.create(
        table=NOTES_TABLE[["sequence", "activity"]].copy(),
        name=f"Test_metadata_empty_{random_string(10)}",
        metadata_columns=[],
    )

    assert data.metadata_names == []
    assert data.measurement_names == ["activity"]


@pytest.mark.e2e
def test_assaydata_text_column_requires_metadata_columns_e2e(session: OpenProtein):
    """A text column is rejected as a measurement unless declared as metadata."""
    with pytest.raises(APIError, match="cannot be converted to float"):
        session.data.Assay.create(
            table=NOTES_TABLE.copy(),
            name=f"Test_metadata_undeclared_{random_string(10)}",
        )


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("metadata_columns", "error"),
    [
        (["missing"], "not found"),
        (["sequence"], "reserved"),
        (["name"], "reserved"),
        (["Sequence"], "reserved"),
        (["NAME"], "reserved"),
    ],
)
def test_assaydata_invalid_metadata_columns_e2e(
    session: OpenProtein, metadata_columns: list[str], error: str
):
    """Missing or reserved metadata columns are rejected."""
    table = NOTES_TABLE.rename(
        columns={c.lower(): c for c in metadata_columns if c.lower() in NOTES_TABLE}
    )
    with pytest.raises(APIError, match=error):
        session.data.Assay.create(
            table=table,
            name=f"Test_metadata_invalid_{random_string(10)}",
            metadata_columns=metadata_columns,
        )
