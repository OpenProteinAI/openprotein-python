"""Offline unit tests for correctness metric helpers."""

import numpy as np
import pytest

from tests.e2e.correctness import metrics


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert metrics.cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert metrics.cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert metrics.cosine(a, b) == 0.0


def test_argmax_match_fraction_all_match():
    a = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert metrics.argmax_match_fraction(a, a) == pytest.approx(1.0)


def test_argmax_match_fraction_half_match():
    a = np.array([[0.1, 0.9], [0.8, 0.2]])  # argmax -> [1, 0]
    b = np.array([[0.1, 0.9], [0.2, 0.8]])  # argmax -> [1, 1]
    assert metrics.argmax_match_fraction(a, b) == pytest.approx(0.5)


def test_argmax_match_fraction_tokens_all_match():
    a = np.array([3, 1, 4, 1, 5])
    assert metrics.argmax_match_fraction_tokens(a, a.copy()) == pytest.approx(1.0)


def test_argmax_match_fraction_tokens_half_match():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 9, 3, 9])
    assert metrics.argmax_match_fraction_tokens(a, b) == pytest.approx(0.5)


def test_argmax_match_fraction_tokens_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.argmax_match_fraction_tokens(np.array([1, 2, 3]), np.array([1, 2]))


def test_spearman_monotonic_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 25.0, 40.0])  # strictly increasing
    assert metrics.spearman(x, y) == pytest.approx(1.0)


def test_pearson_perfect_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    assert metrics.pearson(x, y) == pytest.approx(1.0)


def test_spearman_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # NaN pair dropped; remaining (1,2,4) vs (1,2,4) is perfectly monotonic
    assert metrics.spearman(x, y) == pytest.approx(1.0)


def test_sequence_recovery_identical():
    assert metrics.sequence_recovery("ACDEF", "ACDEF") == pytest.approx(1.0)


def test_sequence_recovery_partial():
    assert metrics.sequence_recovery("ACDEF", "AGDEG") == pytest.approx(3 / 5)


def test_sequence_recovery_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.sequence_recovery("ACDEF", "ACDE")


def test_assert_close_passes_within_tol():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0001, 2.0, 2.9999])
    metrics.assert_close(a, b, rtol=1e-3, atol=1e-3)  # no raise


def test_assert_close_fails_outside_tol():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    with pytest.raises(AssertionError) as exc:
        metrics.assert_close(a, b, rtol=1e-6, atol=1e-6)
    assert "max abs dev" in str(exc.value)


def test_assert_close_nan_aware():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    metrics.assert_close(a, b, rtol=1e-9, atol=1e-9)  # NaNs in same place are equal


def test_derive_tolerance_zero_jitter_uses_floors():
    samples = [np.array([1.0, 2.0, 3.0]) for _ in range(3)]
    atol, rtol = metrics.derive_tolerance(samples, safety=4.0, atol_floor=1e-5, rtol_floor=1e-4)
    assert atol == pytest.approx(1e-5)
    assert rtol == pytest.approx(1e-4)


def test_derive_tolerance_scales_with_jitter():
    base = np.array([10.0, 10.0])
    samples = [base, base + 0.1, base - 0.1]  # max abs dev from mean = 0.1
    atol, rtol = metrics.derive_tolerance(samples, safety=4.0, atol_floor=1e-5, rtol_floor=1e-4)
    assert atol == pytest.approx(0.4, rel=1e-6)
    assert rtol == pytest.approx(0.04, rel=1e-6)
