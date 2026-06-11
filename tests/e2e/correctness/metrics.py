"""Numeric correctness helpers for E2E tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors; 0.0 if either has zero norm."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def argmax_match_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of positions where argmax over the last axis agrees between a and b."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ia = np.argmax(a, axis=-1)
    ib = np.argmax(b, axis=-1)
    return float(np.mean(ia == ib))


def argmax_match_fraction_tokens(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of equal entries between two 1-D integer token arrays of equal length."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 1.0
    return float(np.mean(a == b))


def _drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, dropping NaN/inf pairs first."""
    xv, yv = _drop_nan_pairs(x, y)
    return float(spearmanr(xv, yv).statistic)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, dropping NaN/inf pairs first."""
    xv, yv = _drop_nan_pairs(x, y)
    return float(pearsonr(xv, yv).statistic)


def sequence_recovery(designed: str, native: str) -> float:
    """Fraction of positions where designed matches native. Sequences must be equal length."""
    if isinstance(designed, bytes):
        designed = designed.decode()
    if isinstance(native, bytes):
        native = native.decode()
    if len(designed) != len(native):
        raise ValueError(f"length mismatch: {len(designed)} vs {len(native)}")
    if len(native) == 0:
        raise ValueError("empty sequences")
    matches = sum(1 for d, n in zip(designed, native) if d == n)
    return matches / len(native)


def assert_close(
    actual: np.ndarray,
    baseline: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    """NaN-aware closeness assertion with a debuggable error message."""
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if actual.shape != baseline.shape:
        raise AssertionError(f"shape mismatch: actual {actual.shape} vs baseline {baseline.shape}")
    if np.allclose(actual, baseline, rtol=rtol, atol=atol, equal_nan=True):
        return
    both_finite = np.isfinite(actual) & np.isfinite(baseline)
    diff = np.abs(actual[both_finite] - baseline[both_finite])
    max_abs = float(diff.max()) if diff.size else float("nan")
    denom = np.abs(baseline[both_finite])
    rel = diff / np.where(denom == 0, np.inf, denom)
    max_rel = float(rel.max()) if rel.size else float("nan")
    raise AssertionError(
        f"arrays not close: max abs dev={max_abs:.3e} (atol={atol:.1e}), "
        f"max rel dev={max_rel:.3e} (rtol={rtol:.1e})"
    )


def derive_tolerance(
    samples: list[np.ndarray],
    *,
    safety: float = 4.0,
    atol_floor: float = 1e-5,
    rtol_floor: float = 1e-4,
) -> tuple[float, float]:
    """Derive (atol, rtol) from run-to-run jitter across capture samples.

    Tolerance is `safety` times the largest observed deviation from the per-element
    mean, floored. With a single sample (no repeats) jitter is zero and the floors
    apply.
    """
    stacked = np.stack([np.asarray(s, dtype=float) for s in samples], axis=0)
    mean = np.nanmean(stacked, axis=0)
    dev = np.abs(stacked - mean[None, ...])
    finite = np.isfinite(dev)
    max_abs_dev = float(dev[finite].max()) if finite.any() else 0.0
    denom = np.abs(mean)
    rel = dev / np.where(denom == 0, np.inf, denom)[None, ...]
    rel_finite = np.isfinite(rel)
    max_rel_dev = float(rel[rel_finite].max()) if rel_finite.any() else 0.0
    atol = max(safety * max_abs_dev, atol_floor)
    rtol = max(safety * max_rel_dev, rtol_floor)
    return atol, rtol
