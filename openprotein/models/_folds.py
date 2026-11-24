"""Checks on the custom folds of a cross-validation."""

import numbers
from collections.abc import Sequence

from openprotein.errors import InvalidParameterError

MIN_NUM_FOLDS = 2
MAX_NUM_FOLDS = 10
EXCLUDED_FOLD = -1


def validate_folds(folds: Sequence[int], num_assay_rows: int) -> list[int]:
    """``folds`` as ints, after the checks the API applies to them."""
    out: list[int] = []
    for i, f in enumerate(folds):
        if isinstance(f, bool) or not isinstance(f, numbers.Integral):
            raise InvalidParameterError(
                f"folds must all be integers, got {f!r} at row {i}"
            )
        out.append(int(f))
    if len(out) != num_assay_rows:
        raise InvalidParameterError(
            f"expected one fold per row of the training assay ({num_assay_rows}), "
            f"got {len(out)}"
        )
    counts: dict[int, int] = {}
    for i, f in enumerate(out):
        if f < EXCLUDED_FOLD:
            raise InvalidParameterError(
                f"fold must be {EXCLUDED_FOLD} to exclude a row or >= 0, "
                f"got {f} at row {i}"
            )
        if f != EXCLUDED_FOLD:
            counts[f] = counts.get(f, 0) + 1
    k = len(counts)
    if not MIN_NUM_FOLDS <= k <= MAX_NUM_FOLDS:
        raise InvalidParameterError(
            f"expected between {MIN_NUM_FOLDS} and {MAX_NUM_FOLDS} folds, got {k}"
        )
    missing = [f for f in range(k) if f not in counts]
    if missing:
        raise InvalidParameterError(
            f"folds must be labelled 0..{k - 1} with none empty, but no row is in "
            f"fold {missing[0]}"
        )
    return out


def check_exclusive(n_splits: int | None, folds: Sequence[int] | None) -> None:
    if n_splits is not None and folds is not None:
        raise InvalidParameterError(
            "n_splits and folds are mutually exclusive; folds already defines the "
            "number of folds"
        )
