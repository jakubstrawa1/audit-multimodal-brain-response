"""Spin test (Alexander-Bloch et al. 2018): a null for comparing two cortical maps
that keeps their spatial smoothness. The map is rotated on the sphere (mirror
rotation for the right hemisphere) and every vertex takes the value of the
nearest rotated vertex."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import N_HEMI


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def spin_indices(sphere: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """[n, 20484] index arrays: spun_map = map[idx[k]]."""
    rng = np.random.default_rng(seed)
    L, R = sphere[:N_HEMI], sphere[N_HEMI:]
    tL, tR = cKDTree(L), cKDTree(R)
    flip = np.diag([-1.0, 1.0, 1.0])
    out = np.empty((n, 2 * N_HEMI), np.int64)
    for k in range(n):
        rot = random_rotation(rng)
        # vertex v takes the value of the vertex whose ROTATED position is nearest to v
        _, iL = cKDTree(L @ rot.T).query(L)
        _, iR = cKDTree(R @ (flip @ rot @ flip).T).query(R)
        out[k, :N_HEMI] = iL
        out[k, N_HEMI:] = iR + N_HEMI
    return out


def spin_p(stat_fn, pred_maps: dict, idx: np.ndarray, observed: float, larger_is_better=True):
    """Re-compute stat_fn on spun copies of the predicted maps (dict of arrays).
    Returns (p, null distribution)."""
    null = np.array([stat_fn({k: v[ix] for k, v in pred_maps.items()}) for ix in idx])
    null = null[np.isfinite(null)]
    if larger_is_better:
        p = (np.sum(null >= observed) + 1) / (len(null) + 1)
    else:
        p = (np.sum(null <= observed) + 1) / (len(null) + 1)
    return float(p), null
