"""Checks on the custom folds of a cross-validation."""

import pytest

from openprotein.errors import InvalidParameterError
from openprotein.models._folds import EXCLUDED_FOLD, MAX_NUM_FOLDS, validate_folds


def test_valid_folds_are_normalized_to_ints():
    assert validate_folds(folds=[0, 1, 0, 1], num_assay_rows=4) == [0, 1, 0, 1]


def test_numpy_folds_are_accepted():
    np = pytest.importorskip("numpy")
    folds = np.array([0, 1, EXCLUDED_FOLD, 1, 0])
    out = validate_folds(folds=folds, num_assay_rows=5)
    assert out == [0, 1, -1, 1, 0]
    assert all(isinstance(f, int) for f in out)


def test_excluded_rows_are_allowed():
    assert validate_folds(folds=[0, EXCLUDED_FOLD, 1, 0, 1], num_assay_rows=5) == [
        0,
        -1,
        1,
        0,
        1,
    ]


@pytest.mark.parametrize(
    "folds",
    [
        [0, 1, 0, 1],  # each repeated sequence split across both folds
        [0, 0, 1, 1],  # each repeated sequence kept together in one fold
        [0, 1, 1, 0],
    ],
)
def test_folds_ignore_which_rows_share_a_sequence(folds):
    """Folds are per row, so nothing couples rows that share a sequence."""
    assert validate_folds(folds=folds, num_assay_rows=4) == folds


@pytest.mark.parametrize(
    "folds,num_assay_rows,message",
    [
        ([0, 1, 0], 4, "expected one fold per row of the training assay (4), got 3"),
        (
            [0, 1, 0, 1, 0],
            4,
            "expected one fold per row of the training assay (4), got 5",
        ),
        ([0, 1, -2, 1], 4, "got -2 at row 2"),
        ([0, 2, 0, 2], 4, "no row is in fold 1"),
        ([1, 2, 1, 2], 4, "no row is in fold 0"),
        ([0, 0, 0, 0], 4, "expected between 2 and 10 folds, got 1"),
        ([EXCLUDED_FOLD] * 4, 4, "expected between 2 and 10 folds, got 0"),
        (list(range(MAX_NUM_FOLDS + 1)), MAX_NUM_FOLDS + 1, "got 11"),
        (["a", "b"], 2, "folds must all be integers"),
    ],
)
def test_invalid_folds_are_rejected(folds, num_assay_rows: int, message: str):
    with pytest.raises(InvalidParameterError) as e:
        validate_folds(folds=folds, num_assay_rows=num_assay_rows)
    assert message in str(e.value)


@pytest.mark.parametrize(
    "bad",
    [-0.9, 1.0, 1.9, "1", True, False],
)
def test_folds_refuse_non_integer_values(bad):
    """Only integers are folds; floats, numeric strings and booleans are refused."""
    with pytest.raises(InvalidParameterError, match="folds must all be integers"):
        validate_folds(folds=[0, 1, 0, 1, bad], num_assay_rows=5)


def test_folds_refuse_numpy_floats_and_booleans():
    """NumPy floats and booleans are refused, NumPy integers are accepted."""
    np = pytest.importorskip("numpy")
    for bad in (np.float64(1.0), np.bool_(True)):
        with pytest.raises(InvalidParameterError, match="folds must all be integers"):
            validate_folds(folds=[0, 1, 0, 1, bad], num_assay_rows=5)
    out = validate_folds(folds=[np.int8(0), np.int64(1)], num_assay_rows=2)
    assert out == [0, 1]
    assert all(type(f) is int for f in out)
