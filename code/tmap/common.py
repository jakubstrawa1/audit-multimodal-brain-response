"""Config loading and small shared helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
N_HEMI = 10242          # fsaverage5 vertices per hemisphere (TRIBE: left first, then right)
N_VERT = 2 * N_HEMI


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else (ROOT / p)
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1))


def load_json(path):
    return json.loads(Path(path).read_text())


def hemi_of(n: int = N_VERT) -> np.ndarray:
    """0 = left hemisphere, 1 = right, for each of the 20,484 vertices."""
    return (np.arange(n) >= N_HEMI).astype(int)


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    s[s == 0] = 1
    return (x - m) / s
