#!/usr/bin/env python
"""Step 2b - predictions from the STAND-IN model (known ground truth), in TRIBE's
format. Use it to check the analysis and to see what each outcome looks like.

usage: python scripts/02b_standin.py config.yaml --visual retinotopic|pooled
                                                  --auditory tonotopic|flat [--out DIR]
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import numpy as np

from tmap.common import load_config, load_json
from tmap.predio import save_predictions
from tmap.references import load_references, mesh_fsaverage5
from tmap.standin import make_predictions

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--visual", default="retinotopic", choices=["retinotopic", "pooled"])
p.add_argument("--auditory", default="tonotopic", choices=["tonotopic", "flat"])
p.add_argument("--noise", type=float, default=0.3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None, help="default: <predictions>/standin_<visual>_<auditory>")
a = p.parse_args()

cfg = load_config(a.config)
refs = load_references(cfg["paths"]["references"])
mesh = mesh_fsaverage5()
out = Path(a.out) if a.out else cfg["paths"]["predictions"].parent / f"predictions_standin_{a.visual}_{a.auditory}"
out.mkdir(parents=True, exist_ok=True)
for i, m in enumerate(load_json(cfg["paths"]["stimuli"] / "manifest.json")):
    design = dict(np.load(cfg["paths"]["stimuli"] / f"{m['name']}_design.npz", allow_pickle=True))
    P, t = make_predictions(design, m["kind"], refs, mesh, a.visual, a.auditory, seed=1000 * a.seed + i, noise=a.noise)
    save_predictions(out / f"{m['name']}.npz", P, t, f"standin:{a.visual}/{a.auditory}")
    print(f"  {m['name']:13s} {P.shape}")
print(f"stand-in predictions -> {out}")
