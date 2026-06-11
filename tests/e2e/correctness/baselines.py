"""Baseline storage and the capture/assert Comparator for correctness tests.

Layout (per model_id, so diffs are reviewable):
    <root>/<model_id>.json        # scalars/tokens + tolerances + provenance for all keys
    <root>/<model_id>.arrays.npz  # arrays, keyed by the same flattened key string

A key is a 4-tuple (model_id, model_version, input_id, output_type); model_version
may be None and is omitted from the flattened string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from . import metrics

Key = tuple

# Tolerance-calibration constants: jitter-derived tolerance = SAFETY * observed
# run-to-run deviation, floored at ATOL_FLOOR / RTOL_FLOOR.
SAFETY = 4.0
ATOL_FLOOR = 1e-5
RTOL_FLOOR = 1e-4


def _model_id(key: Key) -> str:
    return str(key[0])


def _key_str(key: Key) -> str:
    return "/".join(str(p) for p in key if p is not None)


@dataclass
class BaselineRecord:
    kind: str  # "array" | "scalar" | "tokens"
    atol: float = 0.0
    rtol: float = 0.0
    value: float | None = None
    tokens: list[int] | None = None
    mismatch_fraction: float = 0.0
    array: np.ndarray | None = None
    provenance: dict | None = None


class BaselineStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, key: Key) -> Path:
        return self.root / f"{_model_id(key)}.json"

    def _npz_path(self, key: Key) -> Path:
        return self.root / f"{_model_id(key)}.arrays.npz"

    def _read_json(self, key: Key) -> dict:
        path = self._json_path(key)
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _write_json(self, key: Key, data: dict) -> None:
        self._json_path(key).write_text(json.dumps(data, indent=2, sort_keys=True))

    def _read_npz(self, key: Key) -> dict[str, np.ndarray]:
        path = self._npz_path(key)
        if path.exists():
            with np.load(path) as npz:
                return {k: npz[k] for k in npz.files}
        return {}

    def _write_npz(self, key: Key, arrays: dict[str, np.ndarray]) -> None:
        np.savez(self._npz_path(key), **arrays)

    def save_array(self, key: Key, array, *, atol: float, rtol: float, provenance: dict) -> None:
        array = np.asarray(array, dtype=float)
        data = self._read_json(key)
        data[_key_str(key)] = {
            "kind": "array",
            "atol": atol,
            "rtol": rtol,
            "shape": list(array.shape),
            "provenance": provenance,
        }
        self._write_json(key, data)
        arrays = self._read_npz(key)
        arrays[_key_str(key)] = array
        self._write_npz(key, arrays)

    def save_scalar(self, key: Key, value: float, *, atol: float, rtol: float, provenance: dict) -> None:
        data = self._read_json(key)
        data[_key_str(key)] = {
            "kind": "scalar",
            "value": float(value),
            "atol": atol,
            "rtol": rtol,
            "provenance": provenance,
        }
        self._write_json(key, data)

    def save_tokens(self, key: Key, tokens, *, mismatch_fraction: float, provenance: dict) -> None:
        data = self._read_json(key)
        data[_key_str(key)] = {
            "kind": "tokens",
            "tokens": [int(t) for t in tokens],
            "mismatch_fraction": float(mismatch_fraction),
            "provenance": provenance,
        }
        self._write_json(key, data)

    def load(self, key: Key) -> BaselineRecord | None:
        data = self._read_json(key)
        entry = data.get(_key_str(key))
        if entry is None:
            return None
        kind = entry["kind"]
        if kind == "array":
            arrays = self._read_npz(key)
            return BaselineRecord(
                kind="array",
                atol=entry["atol"],
                rtol=entry["rtol"],
                array=arrays[_key_str(key)],
                provenance=entry.get("provenance"),
            )
        if kind == "scalar":
            return BaselineRecord(
                kind="scalar",
                value=entry["value"],
                atol=entry["atol"],
                rtol=entry["rtol"],
                provenance=entry.get("provenance"),
            )
        if kind == "tokens":
            return BaselineRecord(
                kind="tokens",
                tokens=list(entry["tokens"]),
                mismatch_fraction=entry["mismatch_fraction"],
                provenance=entry.get("provenance"),
            )
        raise ValueError(f"unknown baseline kind: {kind}")


class Comparator:
    """Runs a value-producing thunk in capture or assert mode against a BaselineStore."""

    def __init__(self, *, store: BaselineStore, mode: str, repeats: int, provenance: dict):
        if mode not in ("capture", "assert"):
            raise ValueError(f"mode must be 'capture' or 'assert', got {mode!r}")
        self.store = store
        self.mode = mode
        self.repeats = repeats
        self.provenance = provenance

    def check_array(self, key: Key, produce, *, atol: float | None = None, rtol: float | None = None) -> None:
        """Capture or assert an array. In assert mode, `atol`/`rtol` override the
        baseline's stored (jitter-derived) tolerance -- use this to absorb accepted
        cross-environment jitter on a genuinely-recomputed output (e.g. PoET)."""
        if self.mode == "capture":
            samples = [np.asarray(produce(), dtype=float) for _ in range(self.repeats)]
            value = np.nanmean(np.stack(samples, axis=0), axis=0)
            d_atol, d_rtol = metrics.derive_tolerance(
                samples, safety=SAFETY, atol_floor=ATOL_FLOOR, rtol_floor=RTOL_FLOOR
            )
            self.store.save_array(key, value, atol=d_atol, rtol=d_rtol, provenance=self.provenance)
            return
        rec = self.store.load(key)
        if rec is None:
            pytest.skip(f"no baseline for {key}; run --update-baselines against prod")
        actual = np.asarray(produce(), dtype=float)
        metrics.assert_close(
            actual,
            rec.array,
            rtol=rtol if rtol is not None else rec.rtol,
            atol=atol if atol is not None else rec.atol,
        )

    def check_scalar(self, key: Key, produce, *, atol: float | None = None, rtol: float | None = None) -> None:
        """Capture or assert a scalar. `atol`/`rtol` override the stored tolerance in
        assert mode (see `check_array`)."""
        if self.mode == "capture":
            samples = [np.asarray(float(produce())) for _ in range(self.repeats)]
            value = float(np.nanmean(samples))
            d_atol, d_rtol = metrics.derive_tolerance(
                samples, safety=SAFETY, atol_floor=ATOL_FLOOR, rtol_floor=RTOL_FLOOR
            )
            self.store.save_scalar(key, value, atol=d_atol, rtol=d_rtol, provenance=self.provenance)
            return
        rec = self.store.load(key)
        if rec is None:
            pytest.skip(f"no baseline for {key}; run --update-baselines against prod")
        actual = float(produce())
        metrics.assert_close(
            np.asarray([actual]),
            np.asarray([rec.value]),
            rtol=rtol if rtol is not None else rec.rtol,
            atol=atol if atol is not None else rec.atol,
        )

    def check_tokens(self, key: Key, produce) -> None:
        if self.mode == "capture":
            samples = [np.asarray(produce()).astype(int).ravel() for _ in range(self.repeats)]
            value = samples[0]
            mismatch = 0.0
            for s in samples[1:]:
                mismatch = max(mismatch, 1.0 - metrics.argmax_match_fraction_tokens(value, s))
            self.store.save_tokens(key, value.tolist(), mismatch_fraction=mismatch, provenance=self.provenance)
            return
        rec = self.store.load(key)
        if rec is None:
            pytest.skip(f"no baseline for {key}; run --update-baselines against prod")
        actual = np.asarray(produce()).astype(int).ravel()
        baseline = np.asarray(rec.tokens, dtype=int)
        if actual.shape != baseline.shape:
            raise AssertionError(f"token length mismatch: {actual.shape} vs {baseline.shape}")
        mismatch = 1.0 - metrics.argmax_match_fraction_tokens(actual, baseline)
        allowed = max(rec.mismatch_fraction, 0.0)
        if mismatch > allowed + 1e-9:
            raise AssertionError(
                f"token mismatch fraction {mismatch:.3f} exceeds allowed {allowed:.3f}"
            )
