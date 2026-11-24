"""The assay verbs on ``session.data`` still work, and say what replaces them."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data import AssayDataset, DataAPI

METADATA = {
    "assay_name": "stability",
    "assay_description": "d",
    "assay_id": "a1",
    "original_filename": "data.csv",
    "created_date": "2023-01-01T00:00:00",
    "num_rows": 100,
    "num_entries": 200,
    "measurement_names": ["activity"],
    "sequence_length": 4,
}


@pytest.fixture
def data() -> DataAPI:
    session = MagicMock()
    for verb in ("get", "post"):
        getattr(session, verb).return_value.status_code = 200
    session.get.return_value.json.return_value = METADATA
    session.post.return_value.json.return_value = METADATA
    return DataAPI(session)


def _message(name: str, use: str) -> str:
    return (
        f"`{name}` is deprecated since 0.18.0 and will be removed in 0.19.0. "
        f"Use `{use}`."
    )


def _one_warning(record, name: str, use: str) -> None:
    assert [str(warning.message) for warning in record] == [_message(name, use)]
    assert record[0].filename == __file__


def test_create_returns_the_assay_and_names_assay_create(data):
    table = pd.DataFrame({"sequence": ["ACDE"], "activity": [1.0]})
    with pytest.warns(FutureWarning) as record:
        assay = data.create(table, "stability", "d")
    assert isinstance(assay, AssayDataset) and assay.id == "a1"
    _one_warning(record, "session.data.create", "session.data.Assay.create")


@pytest.mark.parametrize("verb", ["get", "load_assay"])
def test_get_and_load_assay_return_the_assay_and_name_the_id_load(data, verb):
    with pytest.warns(FutureWarning) as record:
        assay = getattr(data, verb)("a1")
    assert isinstance(assay, AssayDataset) and assay.id == "a1"
    _one_warning(record, f"session.data.{verb}", "session.data.Assay(id=...)")


def test_list_returns_the_assays_and_names_assay_list(data):
    data.session.get.return_value.json.return_value = [METADATA]
    with pytest.warns(FutureWarning) as record:
        assays = data.list(limit=5, offset=2)
    assert [assay.id for assay in assays] == ["a1"]
    data.session.get.assert_called_once_with(
        "v1/assaydata", params={"limit": 5, "offset": 2}
    )
    _one_warning(record, "session.data.list", "session.data.Assay.list")


def test_len_counts_the_assays_and_names_the_list(data):
    data.session.get.return_value.json.return_value = [METADATA, METADATA]
    with pytest.warns(FutureWarning) as record:
        count = len(data)
    assert count == 2
    _one_warning(record, "len(session.data)", "len(session.data.Assay.list())")


@pytest.mark.parametrize("verb", ["create", "get", "list", "load_assay", "__len__"])
def test_each_docstring_marks_the_deprecation(verb):
    assert ".. deprecated:: 0.18.0" in getattr(DataAPI, verb).__doc__
