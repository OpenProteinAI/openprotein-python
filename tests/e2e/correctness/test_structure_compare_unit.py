"""Offline unit tests for structure comparison helpers."""

from pathlib import Path

import numpy as np
import pytest

from tests.e2e.correctness import structure_compare as sc


def _random_coords(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * 10.0


def _rotation_matrix() -> np.ndarray:
    # A fixed proper rotation about the z axis by 37 degrees.
    t = np.deg2rad(37.0)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_ca_rmsd_identical_is_zero():
    p = _random_coords(50)
    assert sc.ca_rmsd(p, p.copy()) == pytest.approx(0.0, abs=1e-8)


def test_ca_rmsd_invariant_to_rotation_translation():
    p = _random_coords(50)
    q = p @ _rotation_matrix().T + np.array([5.0, -3.0, 2.0])
    # Superposition should remove the rigid-body transform -> rmsd ~ 0
    assert sc.ca_rmsd(p, q) == pytest.approx(0.0, abs=1e-6)


def test_tm_score_identical_is_one():
    p = _random_coords(80)
    assert sc.tm_score(p, p.copy()) == pytest.approx(1.0, abs=1e-6)


def test_tm_score_rigid_transform_is_one():
    p = _random_coords(80)
    q = p @ _rotation_matrix().T + np.array([1.0, 2.0, 3.0])
    assert sc.tm_score(p, q) == pytest.approx(1.0, abs=1e-6)


def test_tm_score_noise_below_one():
    p = _random_coords(80)
    rng = np.random.default_rng(1)
    q = p + rng.normal(scale=5.0, size=p.shape)  # large noise
    assert sc.tm_score(p, q) < 0.7


def test_ca_coords_from_cif_reads_committed_structure():
    path = Path("tests/data/8bo9.cif")
    if not path.exists():
        pytest.skip("8bo9.cif not present")
    coords = sc.ca_coords_from_cif(path, chain_id="A")
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert coords.shape[0] > 50  # 8bo9 chain A has many residues
    assert np.isfinite(coords).all()
