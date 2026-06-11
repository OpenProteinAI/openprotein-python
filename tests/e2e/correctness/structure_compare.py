"""Backbone structure comparison: Kabsch superposition, CA-RMSD, TM-score."""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np


def _superpose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return p optimally rotated+translated onto q (Kabsch). p, q are (N, 3)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected matching (N,3) arrays, got {p.shape} and {q.shape}")
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return pc @ rot.T + q.mean(axis=0)


def ca_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """CA-RMSD between two equal-length coordinate sets after optimal superposition."""
    p_aligned = _superpose(p, q)
    diff = p_aligned - np.asarray(q, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def tm_score(p: np.ndarray, q: np.ndarray) -> float:
    """TM-score of p vs q (normalized by len(q)) after optimal superposition.

    TM = (1/L) * sum_i 1 / (1 + (d_i / d0)^2),  d0 = 1.24*(L-15)^(1/3) - 1.8.
    """
    q = np.asarray(q, dtype=float)
    L = q.shape[0]
    p_aligned = _superpose(p, q)
    d = np.linalg.norm(p_aligned - q, axis=1)
    if L > 21:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def ca_coords_from_cif(path: str | Path, chain_id: str) -> np.ndarray:
    """Extract CA coordinates (N, 3) for one chain from a CIF/PDB file, first model."""
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"no models in {path}")
    model = structure[0]
    coords: list[list[float]] = []
    for chain in model:
        if chain.name != chain_id:
            continue
        for residue in chain:
            atom = residue.find_atom("CA", "*")
            if atom is not None:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        raise ValueError(f"no CA atoms found for chain {chain_id} in {path}")
    return np.asarray(coords, dtype=float)
