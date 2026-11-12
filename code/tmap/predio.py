"""Saving / loading predictions ([T, 20484] per experiment + the time of each row)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_predictions(path: Path, preds: np.ndarray, times: np.ndarray, source: str, **extra) -> None:
    np.savez_compressed(path, preds=np.asarray(preds, np.float32), times=np.asarray(times, float),
                        source=source, **extra)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z["preds"].astype(np.float32), z["times"].astype(float)


def infer_segment_times(starts, offsets, n: int, tr: float) -> tuple[np.ndarray, str]:
    """TRIBE returns one prediction per TR-long segment. Work out each row's time
    (s from stimulus onset, centre of the segment) from whatever the segment
    objects expose, falling back to row index x TR. Returns (times, method)."""
    def ok(t):
        if t is None or len(t) != n or not np.all(np.isfinite(t)):
            return False
        d = np.diff(np.sort(t))
        return len(d) == 0 or (np.median(d) > 0.5 * tr and np.median(d) < 1.5 * tr)

    s = None if starts is None else np.asarray(starts, float)
    o = None if offsets is None else np.asarray(offsets, float)
    if ok(s):
        return s + tr / 2, "segment.start"
    if s is not None and o is not None and ok(s + o):
        return s + o + tr / 2, "segment.start + segment.offset"
    return np.arange(n) * tr + tr / 2, "row index x TR (fallback)"
