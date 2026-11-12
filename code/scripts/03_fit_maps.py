#!/usr/bin/env python
"""Step 3 - fit classic mapping models to the predicted time courses.

  retinotopy  2D Gaussian pRF (x, y, size) per vertex, wedges + rings jointly;
              plus Fourier amplitude at the wedge frequency f and at 2f
  hemifield   right-minus-left effect size -> 'contralateral' effect per vertex
  motion      moving-minus-static effect size
  tonotopy    Gaussian tuning on log frequency (best frequency, width), up + down sweeps

usage: python scripts/03_fit_maps.py config.yaml [--preds DIR]
"""
import argparse
from pathlib import Path

import _setup  # noqa: F401
import numpy as np

from tmap.common import hemi_of, load_config
from tmap.fit import block_contrast, block_labels_at, fit_prf, fit_tuning, periodic_power
from tmap.predio import load_predictions

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--preds", default=None, help="predictions folder (default: paths.predictions)")
a = p.parse_args()
cfg = load_config(a.config)
pdir = Path(a.preds) if a.preds else cfg["paths"]["predictions"]
tag = pdir.name
sdir = cfg["paths"]["stimuli"]
out = cfg["paths"]["results"] / tag
out.mkdir(parents=True, exist_ok=True)
F = cfg["fit"]
E = cfg["screen"]["max_ecc_deg"]
res = {}


def have(name):
    return (pdir / f"{name}.npz").exists()


def design(name):
    return dict(np.load(sdir / f"{name}_design.npz", allow_pickle=True))


if have("retino_wedge") and have("retino_ring"):
    dw, dr = design("retino_wedge"), design("retino_ring")
    Pw, tw = load_predictions(pdir / "retino_wedge.npz")
    Pr, trr = load_predictions(pdir / "retino_ring.npz")
    prf = fit_prf([dw, dr], [Pw, Pr], [tw, trr], F, E)
    res.update({f"prf_{k}": v for k, v in prf.items()})
    R = cfg["retinotopy"]
    t0 = R["blank_pad"]
    t1 = t0 + 2 * R["wedge_cycles_each_direction"] * R["wedge_period"]
    # analyse each direction separately (phases differ) and average the amplitudes
    half = t0 + R["wedge_cycles_each_direction"] * R["wedge_period"]
    f1a, f2a = periodic_power(Pw, tw, t0, half, R["wedge_period"])
    f1b, f2b = periodic_power(Pw, tw, half, t1, R["wedge_period"])
    res["wedge_f"], res["wedge_2f"] = (f1a + f1b) / 2, (f2a + f2b) / 2
    print(f"retinotopy: pRF fit r median {np.median(prf['r']):.2f}")

if have("hemifield"):
    d = design("hemifield")
    P, t = load_predictions(pdir / "hemifield.npz")
    lab = block_labels_at(d["side"], float(d["dt"]), t)
    right_minus_left = block_contrast(P, lab, 1, 0)
    res["hemifield_contra_d"] = np.where(hemi_of() == 0, right_minus_left, -right_minus_left)
    res["hemifield_stim_d"] = block_contrast(P, np.where(np.isin(lab, [0, 1]), 1, lab), 1, 2)
    print("hemifield: done")

if have("motion"):
    d = design("motion")
    P, t = load_predictions(pdir / "motion.npz")
    lab = block_labels_at(d["moving"], float(d["dt"]), t)
    res["motion_d"] = block_contrast(P, lab, 1, 0)
    print("motion: done")

if have("tono_up") and have("tono_down"):
    du, dd = design("tono_up"), design("tono_down")
    Pu, tu = load_predictions(pdir / "tono_up.npz")
    Pd, td = load_predictions(pdir / "tono_down.npz")
    tun = fit_tuning([du, dd], [Pu, Pd], [tu, td], F)
    res.update({f"tono_{k}": v for k, v in tun.items()})
    print(f"tonotopy: tuning fit r median {np.median(tun['r']):.2f}")

np.savez_compressed(out / "maps.npz", **res)
print(f"maps -> {out / 'maps.npz'}")
