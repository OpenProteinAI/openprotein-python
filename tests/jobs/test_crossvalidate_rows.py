"""`CrossvalidateFuture` reads evaluation rows."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.jobs import Job
from openprotein.jobs.futures import CrossvalidateFuture

JOB = {
    "job_id": "cv-1",
    "job_type": "/models",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
}


def _row(index: int, **extra) -> dict:
    return {
        "row_index": index,
        "sequence": "MKT",
        "property": "activity",
        "y": 1.0,
        "mean": 0.9,
        "variance": 0.1,
    } | extra


def _read(session, rows: list[dict]):
    gp = MagicMock()
    gp.evaluations.return_value = {"rows": rows}
    return CrossvalidateFuture(session, Job.model_validate(JOB), gp).get()


def test_rows_without_a_fold_key_give_no_fold(mock_session):
    """Rows that omit ``fold`` read as results with ``fold`` None."""
    results = _read(mock_session, [_row(0), _row(1)])
    assert results.fold is None
    assert results.row_index.tolist() == [0, 1]


def test_rows_with_a_null_fold_give_no_fold(mock_session):
    """Any null ``fold`` makes the result's ``fold`` None."""
    assert _read(mock_session, [_row(0, fold=1), _row(1, fold=None)]).fold is None


def test_rows_with_folds_give_an_int_array(mock_session):
    """Rows that all carry a ``fold`` read as an int array."""
    results = _read(mock_session, [_row(0, fold=0), _row(1, fold=-1)])
    assert results.fold is not None
    assert results.fold.dtype == int
    assert np.array_equal(results.fold, [0, -1])


def test_a_row_missing_the_fold_key_among_folds_gives_no_fold(mock_session):
    """A single row without ``fold`` makes the result's ``fold`` None."""
    assert _read(mock_session, [_row(0, fold=0), _row(1)]).fold is None


def test_get_and_wait_reject_an_output_name(mock_session, monkeypatch):
    """A cross-validation has one result, so naming an output raises."""
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    mock_session.get.return_value.json.return_value = JOB
    gp = MagicMock()
    gp.evaluations.return_value = {"rows": [_row(0)]}
    future = CrossvalidateFuture(mock_session, Job.model_validate(JOB), gp)

    with pytest.raises(ValueError, match="one result.*without output"):
        future.get(output="nonexistent")
    with pytest.raises(ValueError, match="one result.*without output"):
        future.wait(output="nonexistent")
    gp.evaluations.assert_not_called()


def test_wait_rejects_an_output_name_before_polling(mock_session):
    """Naming an output on a cross-validation wait raises without polling."""
    pending = JOB | {"status": "PENDING"}
    gp = MagicMock()
    future = CrossvalidateFuture(mock_session, Job.model_validate(pending), gp)
    mock_session.get.reset_mock()

    with pytest.raises(ValueError, match="one result.*without output"):
        future.wait(output="nonexistent")
    mock_session.get.assert_not_called()
