"""Assays through ``session.data.Assay``."""

import pickle
import warnings
from unittest.mock import MagicMock

import pandas as pd
import pytest

from openprotein.data import Assay, AssayDataset, DataAPI
from openprotein.data.schemas import AssayMetadata
from openprotein.errors import HTTPError
from openprotein.jobs.schemas import JobStatus

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


def _response(payload=None, status=200):
    response = MagicMock(status_code=status)
    response.json.return_value = payload
    return response


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def assay(session) -> Assay:
    return Assay(session, AssayMetadata.model_validate(METADATA))


def test_assay_dataset_is_the_same_class():
    assert AssayDataset is Assay


def test_session_data_hands_out_one_bound_assay_class(session):
    data = DataAPI(session)
    assert data.Assay is data.Assay
    assert issubclass(data.Assay, AssayDataset)


def test_a_session_and_metadata_construct_an_assay_dataset(session):
    dataset = AssayDataset(session, AssayMetadata.model_validate(METADATA))
    assert (dataset.id, dataset.session, dataset.page_size) == ("a1", session, 1000)


def test_create_uploads_the_table_and_returns_the_handle(session):
    session.post.return_value = _response(METADATA)
    table = pd.DataFrame({"sequence": ["ACDE"], "activity": [1.0], "notes": ["x"]})
    assay = DataAPI(session).Assay.create(
        table, "stability", metadata_columns=["notes"]
    )
    assert isinstance(assay, Assay) and assay.id == "a1"
    args, kwargs = session.post.call_args
    assert args == ("v1/assaydata",)
    assert kwargs["data"] == {
        "assay_name": "stability",
        "assay_description": None,
        "metadata_columns": '["notes"]',
    }


def test_create_keeps_the_stored_sequence_length(session):
    session.post.return_value = _response(METADATA | {"sequence_length": None})
    table = pd.DataFrame({"Sequence": ["ACDE", "AC"], "activity": [1.0, 2.0]})

    assay = DataAPI(session).Assay.create(table, "stability")

    assert assay.sequence_length is None
    assert list(table.columns) == ["Sequence", "activity"]


def test_list_follows_every_page(session, monkeypatch):
    monkeypatch.setattr("openprotein.data.assaydataset._LIST_PAGE", 2)
    rows = [METADATA | {"assay_id": f"a{i}"} for i in range(3)]
    session.get.side_effect = [_response(rows[:2]), _response(rows[2:])]
    assert [a.id for a in DataAPI(session).Assay.list()] == ["a0", "a1", "a2"]
    assert [c.kwargs["params"] for c in session.get.call_args_list] == [
        {"limit": 2, "offset": 0},
        {"limit": 2, "offset": 2},
    ]


def test_list_stops_at_the_limit_and_starts_at_the_offset(session):
    rows = [METADATA | {"assay_id": f"a{i}"} for i in range(3)]
    session.get.return_value = _response(rows)
    assert len(DataAPI(session).Assay.list(limit=3, offset=5)) == 3
    session.get.assert_called_once_with(
        "v1/assaydata", params={"limit": 3, "offset": 5}
    )


def test_id_loads_one_assay(session):
    session.get.return_value = _response(METADATA)
    assay = DataAPI(session).Assay(id="a1")
    assert (assay.id, assay.name, assay.description) == ("a1", "stability", "d")
    assert (assay.status, assay.job_id) == (JobStatus.SUCCESS, None)
    assert assay.created_at.year == 2023
    session.get.assert_called_once_with(
        "v1/assaydata/metadata", params={"assay_id": "a1"}
    )


def test_an_unknown_id_raises_the_404(session):
    session.get.side_effect = HTTPError(_response({"detail": "x"}, status=404))
    with pytest.raises(HTTPError) as error:
        DataAPI(session).Assay(id="nope")
    assert error.value.status_code == 404


def test_len_is_the_row_count_without_a_request(assay, session):
    assert len(assay) == 100
    session.get.assert_not_called()


def test_an_assay_is_true_when_it_has_rows(session):
    empty = Assay(session, AssayMetadata.model_validate(METADATA | {"num_rows": 0}))
    assert not empty
    assert Assay(session, AssayMetadata.model_validate(METADATA))
    session.get.assert_not_called()


def test_update_sends_only_the_given_fields_and_returns_the_handle(assay, session):
    session.put.return_value = _response(METADATA | {"assay_name": "new"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assay.update(name="new") is assay
    session.put.assert_called_once_with("v1/assaydata/a1", json={"assay_name": "new"})
    assert assay.name == "new"


def test_update_takes_the_assay_prefixed_keywords_with_a_warning(assay, session):
    session.put.return_value = _response(METADATA)
    with pytest.warns(FutureWarning, match=r"Use `Assay\.update\(name=") as record:
        assay.update(assay_name="new", assay_description="e")
    assert record[0].filename == __file__
    assert session.put.call_args.kwargs["json"] == {
        "assay_name": "new",
        "assay_description": "e",
    }


def test_updating_a_pending_assay_keeps_its_job(session):
    pending = METADATA | {"status": "RUNNING", "align_job_id": "job-1"}
    assay = Assay(session, AssayMetadata.model_validate(pending))
    session.put.return_value = _response(METADATA | {"assay_name": "new"})
    session.get.return_value = _response(pending | {"assay_name": "new"})

    assay.update(name="new")

    assert (assay.name, assay.status, assay.job_id) == (
        "new",
        JobStatus.RUNNING,
        "job-1",
    )
    assert not assay.done()
    session.get.assert_called_once_with(
        "v1/assaydata/metadata", params={"assay_id": "a1"}
    )


def test_update_refuses_both_keywords_for_one_field(assay, session):
    with pytest.raises(TypeError, match="without assay_name"):
        assay.update(name="a", assay_name="b")
    session.put.assert_not_called()


def test_delete_deletes_the_assay(assay, session):
    assay.delete()
    session.delete.assert_called_once_with("v1/assaydata/a1")


def test_refresh_reads_the_metadata_again(assay, session):
    session.get.return_value = _response(METADATA | {"num_rows": 7})
    assert assay.refresh() is assay
    assert len(assay) == 7


def test_wait_on_a_ready_assay_returns_it(assay, session):
    assert assay.wait() is assay
    session.get.assert_not_called()


def test_an_assay_has_no_upload():
    assert not hasattr(Assay, "upload")


def test_an_assay_from_session_data_pickles():
    from openprotein.base import APISession

    session = APISession.__new__(APISession)
    assay = DataAPI(session).Assay(session, AssayMetadata.model_validate(METADATA))
    assay.page_size = 10
    restored = pickle.loads(pickle.dumps(assay))
    assert isinstance(restored, Assay)
    assert (restored.id, len(restored), restored.page_size) == ("a1", 100, 10)
