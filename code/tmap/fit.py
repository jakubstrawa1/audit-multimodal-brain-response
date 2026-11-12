"""Fitting classic mapping models to PREDICTED time courses.

TRIBE v2 outputs one map per second, already shifted 5 s back to undo the
haemodynamic lag. A candidate response is therefore built as
    stimulus (10 Hz) -> canonical HRF -> read out 5 s later (+ a small residual lag)
and compared with each vertex's predicted time course by correlation.
Runs (e.g. wedges and rings) are z-scored separately and fitted jointly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma

TRIBE_SHIFT_S = 5.0


def spm_hrf(dt: float, length: float = 32.0) -> np.ndarray:
    t = np.arange(0, length, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def candidate_timecourses(drive: np.ndarray, dt: float, t_pred: np.ndarray, lags) -> np.ndarray:
    """drive [T10, C] at spacing dt -> [n_lags, len(t_pred), C] sampled where each
    prediction time t corresponds to BOLD at t + 5 s (+ lag)."""
    h = spm_hrf(dt).astype(np.float32)
    conv = fftconvolve(drive.astype(np.float32), h[:, None], axes=0)[: len(drive)]
    tgrid = (np.arange(len(drive)) + 0.5) * dt
    out = np.empty((len(lags), len(t_pred), drive.shape[1]), np.float32)
    for i, lag in enumerate(lags):
        tq = np.clip(t_pred + TRIBE_SHIFT_S + lag, tgrid[0], tgrid[-1])
        idx = np.clip(np.searchsorted(tgrid, tq) - 1, 0, len(tgrid) - 2)
        w = ((tq - tgrid[idx]) / dt)[:, None]
        out[i] = (1 - w) * conv[idx] + w * conv[idx + 1]
    return out


def _z(x, axis):
    m = x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True)
    s[s == 0] = 1
    return (x - m) / s


def grid_search(cands_runs: list[np.ndarray], data_runs: list[np.ndarray], chunk: int = 2048):
    """cands_runs: per run [n_lags, T_r, C]; data_runs: per run [T_r, V].
    Returns best candidate index, best lag index and correlation per vertex."""
    Zc = np.concatenate([_z(c, 1) for c in cands_runs], axis=1)       # [L, T, C]
    Zd = np.concatenate([_z(d, 0) for d in data_runs], axis=0)        # [T, V]
    T = Zd.shape[0]
    V = Zd.shape[1]
    best_r = np.full(V, -np.inf, np.float32)
    best_c = np.zeros(V, int)
    best_l = np.zeros(V, int)
    for s in range(0, V, chunk):
        D = Zd[:, s:s + chunk]
        for li in range(Zc.shape[0]):
            R = (Zc[li].T @ D) / T                                     # [C, chunk]
            c = R.argmax(0)
            r = R[c, np.arange(R.shape[1])]
            better = r > best_r[s:s + chunk]
            best_r[s:s + chunk][better] = r[better]
            best_c[s:s + chunk][better] = c[better]
            best_l[s:s + chunk][better] = li
    return best_c, best_l, best_r


# ------------------------------------------------------------------ 2D pRF
def prf_candidates(max_ecc: float, n_ecc: int, n_angle: int, sigmas) -> np.ndarray:
    """Candidate (x, y, sigma) in degrees: log-spaced eccentricities x angles x sizes."""
    eccs = np.geomspace(0.25, 1.2 * max_ecc, n_ecc)
    angles = np.radians(np.arange(n_angle) * 360.0 / n_angle)
    out = [(0.0, 0.0, s) for s in sigmas]
    for e in eccs:
        for a in angles:
            for s in sigmas:
                out.append((e * np.sin(a), e * np.cos(a), s))
    return np.array(out)


def prf_drive(aperture: np.ndarray, max_ecc: float, cands: np.ndarray) -> np.ndarray:
    """aperture [T, G, G] (rows = +y top) -> neural drive [T, C] (overlap with each Gaussian)."""
    T, G, _ = aperture.shape
    c = (np.arange(G) + 0.5) / G * 2 - 1
    gx = (c[None, :] * max_ecc) * np.ones((G, 1))
    gy = (-c[:, None] * max_ecc) * np.ones((1, G))
    W = np.exp(-((gx.ravel()[:, None] - cands[:, 0]) ** 2 + (gy.ravel()[:, None] - cands[:, 1]) ** 2)
               / (2 * cands[:, 2] ** 2)).astype(np.float32)
    W /= W.sum(0, keepdims=True) + 1e-12
    return aperture.reshape(T, -1).astype(np.float32) @ W


def fit_prf(designs: list[dict], preds: list[np.ndarray], times: list[np.ndarray], cfg_fit: dict,
            max_ecc: float) -> dict:
    cands = prf_candidates(max_ecc, cfg_fit["prf_ecc_grid"], cfg_fit["prf_angle_grid"],
                           cfg_fit["prf_sigmas_deg"])
    lags = cfg_fit["lags"]
    cr = [candidate_timecourses(prf_drive(d["aperture"], max_ecc, cands), float(d["dt"]), t, lags)
          for d, t in zip(designs, times)]
    c, l, r = grid_search(cr, preds)
    x, y, s = cands[c, 0], cands[c, 1], cands[c, 2]
    return {"x": x, "y": y, "sigma": s, "ecc": np.hypot(x, y),
            "angle": np.degrees(np.arctan2(x, y)), "lag": np.asarray(lags)[l], "r": r}


# ------------------------------------------------------------------ 1D tuning
def tuning_drive(band: np.ndarray, band_log2f: np.ndarray, centres: np.ndarray, widths) -> tuple:
    """band [T] (-1 = silence) -> drive [T, C] for Gaussian tuning on log2 frequency."""
    cand = np.array([(c, w) for c in centres for w in widths])
    f = np.where(band >= 0, band_log2f[np.clip(band, 0, None)], np.nan)
    D = np.exp(-((f[:, None] - cand[:, 0]) ** 2) / (2 * cand[:, 1] ** 2))
    D[np.isnan(f)] = 0.0
    return D.astype(np.float32), cand


def fit_tuning(designs: list[dict], preds: list[np.ndarray], times: list[np.ndarray], cfg_fit: dict,
               band_order: np.ndarray | None = None) -> dict:
    """band_order: optional permutation of band -> frequency (for the label-permutation null)."""
    centres_hz = designs[0]["band_centres_hz"]
    log2f = np.log2(centres_hz)
    if band_order is not None:
        log2f = log2f[band_order]
    centres = np.linspace(np.log2(centres_hz[0]) - 0.5, np.log2(centres_hz[-1]) + 0.5,
                          cfg_fit["tuning_centres"])
    lags = cfg_fit["lags"]
    cr, cand = [], None
    for d, t in zip(designs, times):
        D, cand = tuning_drive(d["band"], log2f, centres, cfg_fit["tuning_widths_oct"])
        cr.append(candidate_timecourses(D, float(d["dt"]), t, lags))
    c, l, r = grid_search(cr, preds)
    return {"bf_hz": 2 ** cand[c, 0], "width_oct": cand[c, 1], "lag": np.asarray(lags)[l], "r": r}


# ------------------------------------------------------------------ block designs
def block_labels_at(design_labels: np.ndarray, dt: float, t_pred: np.ndarray,
                    settle_s: float = 3.0) -> np.ndarray:
    """Condition label at each prediction time; -9 for samples within settle_s of a
    condition change (responses are still rising/falling there)."""
    T = len(design_labels)
    idx = np.clip((t_pred / dt).astype(int), 0, T - 1)
    lab = design_labels[idx].copy()
    back = np.clip(((t_pred - settle_s) / dt).astype(int), 0, T - 1)
    lab[design_labels[back] != lab] = -9
    return lab


def block_contrast(pred: np.ndarray, labels: np.ndarray, a: int, b: int) -> np.ndarray:
    """Effect size (mean_a - mean_b) / pooled SD, per vertex."""
    A, B = pred[labels == a], pred[labels == b]
    sd = np.sqrt((A.var(0) * (len(A) - 1) + B.var(0) * (len(B) - 1)) / max(len(A) + len(B) - 2, 1))
    return (A.mean(0) - B.mean(0)) / np.maximum(sd, 1e-6)


def periodic_power(pred: np.ndarray, t_pred: np.ndarray, t0: float, t1: float, period: float):
    """Fourier amplitude at the stimulus frequency f = 1/period and at 2f within [t0, t1).
    A position-coding vertex responds at f (the wedge passes its pRF once per cycle);
    a position-blind but orientation-sensitive one responds at 2f."""
    sel = (t_pred >= t0) & (t_pred < t1)
    x = pred[sel] - pred[sel].mean(0)
    tt = t_pred[sel]
    out = []
    for k in (1, 2):
        w = 2 * np.pi * k / period
        out.append(np.abs(np.exp(-1j * w * tt) @ x) / len(tt))
    total = np.sqrt((x ** 2).mean(0)) + 1e-12
    return out[0] / total, out[1] / total
