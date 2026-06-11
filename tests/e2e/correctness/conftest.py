"""Pytest config for correctness tests: capture/assert options and the `baseline` fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.correctness.baselines import BaselineStore, Comparator

BASELINE_DIR = Path(__file__).parent / "baselines"


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Capture mode: probe the configured backend and (re)write baseline fixtures.",
    )
    parser.addoption(
        "--baseline-repeats",
        action="store",
        type=int,
        default=1,
        help="Number of capture repeats per probe, used to calibrate tolerance from jitter.",
    )


@pytest.fixture
def baseline(request, session) -> Comparator:
    """A Comparator wired to the committed baseline store, in capture or assert mode."""
    mode = "capture" if request.config.getoption("--update-baselines") else "assert"
    repeats = request.config.getoption("--baseline-repeats")
    store = BaselineStore(BASELINE_DIR)
    provenance = {"backend": session.backend}
    return Comparator(store=store, mode=mode, repeats=repeats, provenance=provenance)
