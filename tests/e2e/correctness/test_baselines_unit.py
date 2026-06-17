"""Offline unit tests for the baseline store and comparator."""

import numpy as np
import pytest

from tests.e2e.correctness import baselines as bl


def test_store_array_roundtrip(tmp_path):
    store = bl.BaselineStore(tmp_path)
    key = ("esm2", None, "ubiquitin", "embed_mean")
    arr = np.arange(6, dtype=float).reshape(2, 3)
    store.save_array(key, arr, atol=1e-3, rtol=1e-4, provenance={"backend": "prod"})

    fresh = bl.BaselineStore(tmp_path)  # reload from disk
    rec = fresh.load(key)
    assert rec is not None
    assert rec.kind == "array"
    np.testing.assert_array_equal(rec.array, arr)
    assert rec.atol == 1e-3
    assert rec.rtol == 1e-4


def test_store_scalar_roundtrip(tmp_path):
    store = bl.BaselineStore(tmp_path)
    key = ("poet", None, "amie", "variant_effect_spearman")
    store.save_scalar(key, 0.42, atol=0.05, rtol=0.0, provenance={"backend": "prod"})
    rec = bl.BaselineStore(tmp_path).load(key)
    assert rec.kind == "scalar"
    assert rec.value == pytest.approx(0.42)
    assert rec.atol == 0.05


def test_store_tokens_roundtrip(tmp_path):
    store = bl.BaselineStore(tmp_path)
    key = ("esm2", None, "ubiquitin", "logits_argmax")
    store.save_tokens(key, [3, 1, 4, 1, 5], mismatch_fraction=0.0, provenance={})
    rec = bl.BaselineStore(tmp_path).load(key)
    assert rec.kind == "tokens"
    assert rec.tokens == [3, 1, 4, 1, 5]
    assert rec.mismatch_fraction == 0.0


def test_load_missing_returns_none(tmp_path):
    store = bl.BaselineStore(tmp_path)
    assert store.load(("nope", None, "x", "y")) is None


def test_two_keys_same_model_coexist(tmp_path):
    store = bl.BaselineStore(tmp_path)
    store.save_scalar(("esm2", None, "a", "t1"), 1.0, atol=0.1, rtol=0.0, provenance={})
    store.save_scalar(("esm2", None, "b", "t2"), 2.0, atol=0.1, rtol=0.0, provenance={})
    fresh = bl.BaselineStore(tmp_path)
    assert fresh.load(("esm2", None, "a", "t1")).value == pytest.approx(1.0)
    assert fresh.load(("esm2", None, "b", "t2")).value == pytest.approx(2.0)


def _make_comparator(tmp_path, mode, repeats=1):
    store = bl.BaselineStore(tmp_path)
    return bl.Comparator(store=store, mode=mode, repeats=repeats, provenance={"backend": "test"})


def test_comparator_capture_then_assert_array(tmp_path):
    key = ("esm2", None, "ubiquitin", "embed_mean")
    truth = np.array([1.0, 2.0, 3.0])

    # Capture with tiny jitter across repeats.
    seq = iter([truth, truth + 0.001, truth - 0.001])
    cap = _make_comparator(tmp_path, "capture", repeats=3)
    cap.check_array(key, lambda: next(seq))

    # Assert: a value within the derived tolerance passes.
    asr = _make_comparator(tmp_path, "assert")
    asr.check_array(key, lambda: truth + 0.0005)  # no raise


def test_comparator_assert_fails_when_far(tmp_path):
    key = ("esm2", None, "ubiquitin", "embed_mean")
    truth = np.array([1.0, 2.0, 3.0])
    cap = _make_comparator(tmp_path, "capture", repeats=1)  # zero jitter -> floor tol
    cap.check_array(key, lambda: truth)

    asr = _make_comparator(tmp_path, "assert")
    with pytest.raises(AssertionError):
        asr.check_array(key, lambda: truth + 1.0)


def test_comparator_assert_skips_without_baseline(tmp_path):
    asr = _make_comparator(tmp_path, "assert")
    with pytest.raises(pytest.skip.Exception):
        asr.check_array(("missing", None, "x", "y"), lambda: np.array([1.0]))


def test_comparator_capture_then_assert_scalar(tmp_path):
    key = ("poet", None, "amie", "spearman")
    cap = _make_comparator(tmp_path, "capture", repeats=2)
    seq = iter([0.40, 0.42])
    cap.check_scalar(key, lambda: next(seq))
    asr = _make_comparator(tmp_path, "assert")
    asr.check_scalar(key, lambda: 0.41)  # within derived tol -> no raise


def test_comparator_tokens_exact_match(tmp_path):
    key = ("esm2", None, "ubiquitin", "logits_argmax")
    cap = _make_comparator(tmp_path, "capture", repeats=1)
    cap.check_tokens(key, lambda: np.array([3, 1, 4, 1, 5]))
    asr = _make_comparator(tmp_path, "assert")
    asr.check_tokens(key, lambda: np.array([3, 1, 4, 1, 5]))  # exact -> no raise
    with pytest.raises(AssertionError):
        asr.check_tokens(key, lambda: np.array([3, 1, 4, 1, 9]))  # one differs


# --- dtype parity (records + asserts the wire dtype, not just values) --------


def test_store_array_records_and_loads_dtype(tmp_path):
    store = bl.BaselineStore(tmp_path)
    key = ("poet-2", None, "ubiquitin", "embed")
    arr = np.arange(6, dtype=float).reshape(2, 3)
    store.save_array(key, arr, atol=1e-3, rtol=1e-4, provenance={}, dtype="float16")
    rec = bl.BaselineStore(tmp_path).load(key)
    assert rec.dtype == "float16"
    # values are still stored/loaded as float64 for the closeness comparison
    assert rec.array.dtype == np.float64


def test_store_array_without_dtype_loads_none(tmp_path):
    # Legacy baselines (captured before dtype tracking) carry no dtype.
    store = bl.BaselineStore(tmp_path)
    key = ("esm2", None, "ubiquitin", "embed_mean")
    store.save_array(key, np.zeros(3), atol=1e-3, rtol=1e-4, provenance={})
    rec = bl.BaselineStore(tmp_path).load(key)
    assert rec.dtype is None


def test_comparator_capture_records_wire_dtype(tmp_path):
    key = ("poet-2", None, "ubiquitin", "embed")
    cap = _make_comparator(tmp_path, "capture", repeats=1)
    cap.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float16))
    rec = bl.BaselineStore(tmp_path).load(key)
    assert rec.dtype == "float16"


def test_comparator_assert_fails_on_dtype_mismatch(tmp_path):
    # The disparity this feature exists to catch: same values, wrong dtype.
    key = ("poet-2", None, "ubiquitin", "mean")
    cap = _make_comparator(tmp_path, "capture", repeats=1)
    cap.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float32))

    asr = _make_comparator(tmp_path, "assert")
    with pytest.raises(AssertionError, match="dtype mismatch"):
        asr.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float16))


def test_comparator_assert_passes_on_dtype_match(tmp_path):
    key = ("poet-2", None, "ubiquitin", "embed")
    cap = _make_comparator(tmp_path, "capture", repeats=1)
    cap.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float16))
    asr = _make_comparator(tmp_path, "assert")
    asr.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float16))  # no raise


def test_comparator_assert_skips_dtype_when_baseline_legacy(tmp_path):
    # A legacy baseline (no recorded dtype) must not fail on dtype — only values.
    store = bl.BaselineStore(tmp_path)
    key = ("esm2", None, "ubiquitin", "embed_mean")
    store.save_array(
        key, np.array([1.0, 2.0, 3.0]), atol=1e-3, rtol=1e-4, provenance={}
    )
    asr = _make_comparator(tmp_path, "assert")
    asr.check_array(key, lambda: np.array([1.0, 2.0, 3.0], dtype=np.float16))  # no raise
