"""A stand-in 'model brain' with KNOWN behaviour, producing predictions in exactly
TRIBE's output format ([T, 20484] per second, left hemisphere first, already
shifted to undo the haemodynamic lag).

visual_mode
  retinotopic  V1-V3 vertices have Gaussian pRFs taken from the Benson template
               (what a position-aware model should reproduce)
  pooled       every visual vertex sees only global stimulus statistics
               (stimulated area + wedge orientation), i.e. position-blind - what
               TRIBE v2's spatial averaging of video features would predict
auditory_mode
  tonotopic    core/belt vertices have best frequencies falling from
               posteromedial to anterolateral along Heschl's gyrus
  flat         auditory vertices respond to sound on/off only
MT+ responds to moving > static dots in both modes.

The analysis must recover the planted pattern - and must NOT find retinotopy or
tonotopy in the pooled / flat modes. It uses a slightly different HRF and adds
noise so the test is not perfectly circular.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma

from .common import N_VERT
from .fit import TRIBE_SHIFT_S
from .references import heschl_axis_position


def _hrf(dt):
    t = np.arange(0, 30, dt)
    h = gamma.pdf(t, 5.2) - gamma.pdf(t, 15) / 7.0     # deliberately not the SPM HRF
    return h / h.sum()


def _to_pred(drive: np.ndarray, dt: float, duration: float, rng, noise: float) -> tuple:
    conv = fftconvolve(drive, _hrf(dt)[:, None], axes=0)[: len(drive)]
    tgrid = (np.arange(len(drive)) + 0.5) * dt
    t_pred = np.arange(int(duration)) + 0.5
    idx = np.clip(np.searchsorted(tgrid, t_pred + TRIBE_SHIFT_S), 0, len(drive) - 1)
    P = conv[idx]
    sd = P.std(0)
    sd[sd == 0] = 1
    P = P / sd + noise * rng.standard_normal(P.shape)
    return P.astype(np.float32), t_pred


def hemifield_aperture(side: np.ndarray, G: int = 64) -> np.ndarray:
    c = (np.arange(G) + 0.5) / G * 2 - 1
    x = c[None, :] * np.ones((G, 1))
    y = -c[:, None] * np.ones((1, G))
    disc = np.hypot(x, y) <= 1
    out = np.zeros((len(side), G, G), np.uint8)
    out[side == 0] = (disc & (x < 0))
    out[side == 1] = (disc & (x > 0))
    return out


def motion_aperture(moving: np.ndarray, G: int = 64) -> np.ndarray:
    c = (np.arange(G) + 0.5) / G * 2 - 1
    disc = np.hypot(c[None, :], c[:, None]) <= 1
    out = np.zeros((len(moving), G, G), np.uint8)
    out[moving >= 0] = disc
    return out


def make_predictions(design: dict, kind: str, refs: dict, mesh: dict, visual_mode: str,
                     auditory_mode: str, seed: int = 0, noise: float = 0.3) -> tuple:
    rng = np.random.default_rng(seed)
    dt = float(design["dt"])
    dur = float(design["duration"])
    rois = refs["rois"]
    visual = refs["varea"] > 0
    D = np.zeros((int(round(dur / dt)), N_VERT), np.float32)
    D += 0.2 * rng.standard_normal((len(D), 1)).astype(np.float32)          # shared slow-ish signal

    # ---------------------------------------------------------------- vision
    if kind in ("retinotopy", "hemifield", "motion"):
        if kind == "retinotopy":
            ap = design["aperture"]
        elif kind == "hemifield":
            ap = hemifield_aperture(design["side"])
        else:
            ap = motion_aperture(design["moving"])
        E = float(design.get("max_ecc", 10.0))
        G = ap.shape[1]
        flat = ap.reshape(len(ap), -1).astype(np.float32)
        area = flat.mean(1)
        if visual_mode == "retinotopic":
            sel = rois["V1-V3"] & (refs["ecc"] <= 1.5 * E)
            c = (np.arange(G) + 0.5) / G * 2 - 1
            gx = (c[None, :] * E * np.ones((G, 1))).ravel()
            gy = (-c[:, None] * E * np.ones((1, G))).ravel()
            px, py = refs["vf_x"][sel], refs["vf_y"][sel]
            ps = np.clip(refs["sigma"][sel], 0.3, None)
            W = np.exp(-((gx[:, None] - px) ** 2 + (gy[:, None] - py) ** 2) / (2 * ps ** 2)).astype(np.float32)
            W /= W.sum(0, keepdims=True) + 1e-12
            D[:, sel] += flat @ W
            D[:, visual & ~sel] += area[:, None] * rng.uniform(0.2, 1.0, (visual & ~sel).sum())
        else:
            ori = np.zeros(len(ap), np.float32)
            if kind == "retinotopy" and np.isfinite(design["param"]).any() and str(design["kind"]) == "wedge":
                ang = np.radians(np.nan_to_num(design["param"]))
                ori = np.where(np.isfinite(design["param"]), np.cos(ang) ** 2, 0)  # vertical vs horizontal wedge
            a = rng.uniform(0.5, 1.0, visual.sum())
            b = rng.uniform(-0.5, 0.5, visual.sum())
            D[:, visual] += area[:, None] * a + ori[:, None] * b * area.max()
        if kind == "motion":
            mv = (design["moving"] == 1).astype(np.float32)
            D[:, rois["MT+"]] += 1.5 * mv[:, None] + 0.3 * area[:, None]

    # ---------------------------------------------------------------- audition
    if kind == "tonotopy":
        aud = rois["auditory_core_belt"]
        band = design["band"]
        on = (band >= 0).astype(np.float32)
        if auditory_mode == "tonotopic":
            pos = heschl_axis_position(mesh["pial"], aud, refs["hemi"])
            lo, hi = np.log2(design["band_centres_hz"][[0, -1]])
            p = pos[aud]
            pr = np.zeros_like(p)
            for h in (0, 1):
                m = refs["hemi"][aud] == h
                if m.any():
                    pr[m] = (p[m] - p[m].min()) / (np.ptp(p[m]) + 1e-9)
            bf = hi - pr * (hi - lo)                         # posteromedial = high
            f = np.log2(design["band_centres_hz"][np.clip(band, 0, None)])
            T = np.exp(-((f[:, None] - bf[None, :]) ** 2) / (2 * 1.0 ** 2)) * on[:, None]
            D[:, aud] += T.astype(np.float32)
        else:
            D[:, aud] += on[:, None]
    return _to_pred(D, dt, dur, rng, noise)
