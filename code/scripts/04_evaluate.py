#!/usr/bin/env python
"""Step 4 - compare the model's maps with the reference maps, with nulls.

Retinotopy (V1-V3, template eccentricity <= max_ecc_deg, pRF fit r >= min_fit_r)
  R1 contralaterality   mean (contra - ipsi) effect size in V1 for hemifield blocks
  R2 polar angle        circular correlation with the Benson template
  R3 eccentricity       Spearman correlation of log eccentricity with the template
  R4 position vs shape  share of wedge-frequency power at f (position) vs 2f (orientation)
Motion
  M1 MT+ selectivity    mean moving-static effect in MT+ vs the rest of cortex;
                        share of the top 1% vertices that fall inside MT+
Tonotopy (A1 with tuning fit r >= min_fit_r)
  T1 Heschl's gradient  Spearman(log best frequency, position posteromedial->anterolateral);
                        expected NEGATIVE (high frequencies posteromedial)
  T2 spread             interquartile range of best frequency across core + belt (octaves)
p-values: spin test (spatially smooth null) for map comparisons; stimulus-label
permutation for tonotopy (is the frequency ORDER what produces the gradient?).

usage: python scripts/04_evaluate.py config.yaml [--preds DIR]
"""
import argparse
import warnings
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tmap.common import load_config
from tmap.fit import fit_tuning
from tmap.predio import load_predictions
from tmap.references import heschl_axis_position, load_references, mesh_fsaverage5
from tmap.spin import spin_indices, spin_p

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--preds", default=None)
p.add_argument("--no-figures", action="store_true")
a = p.parse_args()
warnings.filterwarnings("ignore")
cfg = load_config(a.config)
pdir = Path(a.preds) if a.preds else cfg["paths"]["predictions"]
tag = pdir.name
out = cfg["paths"]["results"] / tag
M = dict(np.load(out / "maps.npz"))
refs = load_references(cfg["paths"]["references"])
mesh = mesh_fsaverage5()
R, Ev = refs["rois"], cfg["evaluate"]
E = cfg["screen"]["max_ecc_deg"]
nspin = int(Ev["n_spins"])
spin_file = cfg["paths"]["results"] / f"spin_idx_{nspin}.npy"
if spin_file.exists():
    SPIN = np.load(spin_file)
else:
    print(f"computing {nspin} sphere rotations (cached afterwards) ...", flush=True)
    SPIN = spin_indices(mesh["sphere"], nspin)
    np.save(spin_file, SPIN)
rows = []


def add(test, value, p=np.nan, null_mean=np.nan, n=np.nan, expect=""):
    rows.append({"test": test, "value": value, "null_mean": null_mean, "p": p, "n_vertices": n,
                 "expected_if_model_knows_it": expect})


def circ_corr(a_deg, b_deg):
    a, b = np.radians(a_deg), np.radians(b_deg)
    am = np.angle(np.exp(1j * a).mean())
    bm = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum() + 1e-12))


thr = float(Ev["min_fit_r"])
base = R["V1-V3"] & (refs["ecc"] <= E) & (refs["ecc"] >= 0.5)

# ------------------------------------------------------------------ retinotopy
if "hemifield_contra_d" in M:
    stat = lambda mm: float(np.nanmean(mm["c"][R["V1"]]))  # noqa: E731
    obs = stat({"c": M["hemifield_contra_d"]})
    pv, null = spin_p(stat, {"c": M["hemifield_contra_d"]}, SPIN, obs)
    add("R1 V1 contralateral effect (d)", obs, pv, null.mean(), int(R["V1"].sum()), "> 0")

if "prf_x" in M:
    def sel_of(r):
        return base & (r >= thr)

    # vertex selection is fixed from the observed fit; the spin null rotates only the
    # parameter maps, asking whether their LAYOUT matches the template
    s = sel_of(M["prf_r"])

    def angle_stat(mm):
        return circ_corr(mm["angle"][s], refs["signed_angle"][s]) if s.sum() > 10 else np.nan

    def ecc_stat(mm):
        return float(spearmanr(np.log(mm["ecc"][s]), np.log(refs["ecc"][s])).statistic) if s.sum() > 10 else np.nan

    pm = {"angle": M["prf_angle"], "ecc": M["prf_ecc"]}
    add("R0 V1-V3 vertices with pRF fit r >= thr (fraction)", float(s.sum() / max(base.sum(), 1)), n=int(base.sum()),
        expect="high")
    obs = angle_stat(pm)
    pv, null = spin_p(angle_stat, pm, SPIN, obs)
    add("R2 polar angle vs template (circular r)", obs, pv, np.nanmean(null), int(s.sum()), "> 0")
    lat = s & (np.abs(M["prf_x"]) >= 0.5) & (np.abs(refs["vf_x"]) >= 0.5)   # ignore pRFs on the vertical meridian
    side_ok = np.mean(np.sign(M["prf_x"][lat]) == np.sign(refs["vf_x"][lat])) if lat.sum() else np.nan
    add("R2b pRFs in the contralateral hemifield (fraction)", float(side_ok), n=int(lat.sum()),
        expect="~1 (0.5 = chance)")
    obs = ecc_stat(pm)
    pv, null = spin_p(ecc_stat, pm, SPIN, obs)
    add("R3 eccentricity vs template (Spearman)", obs, pv, np.nanmean(null), int(s.sum()), "> 0")
    share = M["wedge_f"] / (M["wedge_f"] + M["wedge_2f"] + 1e-12)
    add("R4 wedge power at f / (f + 2f), V1-V3 median", float(np.median(share[base])), n=int(base.sum()),
        expect="> 0.5 (position); < 0.5 means orientation-driven")

# ------------------------------------------------------------------ motion
if "motion_d" in M:
    rest = ~R["MT+"]
    stat = lambda mm: float(np.mean(mm["d"][R["MT+"]]))  # noqa: E731
    obs = stat({"d": M["motion_d"]})
    pv, null = spin_p(stat, {"d": M["motion_d"]}, SPIN, obs)
    add("M1 moving-static effect in MT+ (d)", obs, pv, null.mean(), int(R["MT+"].sum()), "> rest of cortex")
    add("M1b moving-static effect, rest of cortex (d)", float(np.mean(M["motion_d"][rest])))
    top = M["motion_d"] >= np.quantile(M["motion_d"], 0.99)
    add("M2 top-1% motion vertices inside MT+ (fraction)", float(R["MT+"][top].mean()),
        null_mean=float(R["MT+"].mean()), n=int(top.sum()), expect=f"> {R['MT+'].mean():.3f} (MT+ share of cortex)")

# ------------------------------------------------------------------ tonotopy
if "tono_bf_hz" in M:
    pos = heschl_axis_position(mesh["pial"], R["G_A1"], refs["hemi"])
    s = R["G_A1"] & (M["tono_r"] >= thr)          # fixed selection; nulls change only the BF map

    def tono_stat(mm):
        if s.sum() < 8:
            return np.nan
        return float(spearmanr(np.log2(mm["bf"][s]), pos[s]).statistic)

    obs = tono_stat({"bf": M["tono_bf_hz"]})
    pv_spin, null_spin = spin_p(tono_stat, {"bf": M["tono_bf_hz"]}, SPIN, obs, larger_is_better=False)
    # stimulus-label permutation: refit A1 with a shuffled band -> frequency mapping.
    # A real tonotopic gradient depends on the ORDER of frequencies; this null keeps
    # everything else (timing, which vertices respond) intact.
    sdir = cfg["paths"]["stimuli"]
    du = dict(np.load(sdir / "tono_up_design.npz"))
    dd = dict(np.load(sdir / "tono_down_design.npz"))
    Pu, tu = load_predictions(pdir / "tono_up.npz")
    Pd, td = load_predictions(pdir / "tono_down.npz")
    rng = np.random.default_rng(0)
    null_lab = []
    for k in range(int(Ev["n_label_perms"])):
        order = rng.permutation(len(du["band_centres_hz"]))
        f = fit_tuning([du, dd], [Pu[:, s], Pd[:, s]], [tu, td], cfg["fit"], band_order=order)
        bf = np.full(len(s), np.nan)
        bf[s] = f["bf_hz"]
        null_lab.append(tono_stat({"bf": bf}))
    null_lab = np.array(null_lab, float)
    null_lab = null_lab[np.isfinite(null_lab)]
    pv_lab = (np.sum(null_lab <= obs) + 1) / (len(null_lab) + 1) if np.isfinite(obs) else np.nan
    add("T0 auditory core+belt vertices with tuning fit r >= thr (fraction)",
        float(np.mean(M["tono_r"][R["auditory_core_belt"]] >= thr)), n=int(R["auditory_core_belt"].sum()),
        expect="high")
    add("T1 A1 best frequency vs Heschl's axis (Spearman), spin p", obs, pv_spin, np.nanmean(null_spin),
        int(s.sum()), "< 0 (high posteromedial, low anterolateral)")
    add("T1b same, stimulus-label permutation p", obs, pv_lab, np.nanmean(null_lab), int(s.sum()), "< 0")
    cb = R["auditory_core_belt"] & (M["tono_r"] >= thr)
    lb = np.log2(M["tono_bf_hz"][cb])
    add("T2 spread of best frequency in core+belt (IQR, octaves)",
        float(np.subtract(*np.percentile(lb, [75, 25]))) if cb.sum() > 4 else np.nan, n=int(cb.sum()),
        expect="> ~1 octave if there is a map; ~0 if all vertices share one tuning")

T = pd.DataFrame(rows)
T.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(f"\n=== {tag} ===")
print(T.round(3).to_string(index=False))

# ------------------------------------------------------------------ figures
if a.no_figures:
    raise SystemExit
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from nilearn import datasets, plotting  # noqa: E402

fs = datasets.fetch_surf_fsaverage("fsaverage5")
N = len(refs["hemi"]) // 2


def surf(ax, data, hemi, view, cmap, vmin, vmax, title, threshold=1e-6):
    h = "left" if hemi == 0 else "right"
    d = data[:N] if hemi == 0 else data[N:]
    plotting.plot_surf_stat_map(fs[f"infl_{h}"], np.nan_to_num(d, nan=0.0), hemi=h, view=view,
                                bg_map=fs[f"sulc_{h}"], axes=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                                colorbar=False, threshold=threshold, title=title)


if "prf_x" in M:
    s = base & (M["prf_r"] >= thr)
    show = lambda v: np.where(s, v, np.nan)  # noqa: E731
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "3d"})
    for row, hemi in enumerate((0, 1)):
        surf(axes[row, 0], np.where(base, np.abs(refs["signed_angle"]) + 1, np.nan), hemi, "medial",
             "hsv", 1, 181, "template |angle|")
        surf(axes[row, 1], show(np.abs(M["prf_angle"]) + 1), hemi, "medial", "hsv", 1, 181, "model |angle|")
        surf(axes[row, 2], np.where(base, np.log(refs["ecc"]) + 3, np.nan), hemi, "medial", "viridis",
             0.1, np.log(E) + 3, "template log ecc")
        surf(axes[row, 3], show(np.log(M["prf_ecc"]) + 3), hemi, "medial", "viridis", 0.1, np.log(E) + 3,
             "model log ecc")
    fig.suptitle(f"{tag}: retinotopy (V1-V3, fitted vertices only)")
    fig.savefig(out / "fig_retinotopy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

panels = [k for k in ("hemifield_contra_d", "motion_d", "tono_bf_hz") if k in M]
if panels:
    fig, axes = plt.subplots(2, len(panels), figsize=(4.5 * len(panels), 7),
                             subplot_kw={"projection": "3d"}, squeeze=False)
    for j, k in enumerate(panels):
        for hemi in (0, 1):
            if k == "hemifield_contra_d":
                surf(axes[hemi, j], M[k], hemi, "medial", "RdBu_r", -3, 3, "contra - ipsi (d)", 0.5)
            elif k == "motion_d":
                surf(axes[hemi, j], M[k], hemi, "lateral", "RdBu_r", -3, 3, "moving - static (d)", 0.5)
            else:
                v = np.where(R["auditory_core_belt"] & (M["tono_r"] >= thr), np.log2(M[k]) - 6, np.nan)
                surf(axes[hemi, j], v, hemi, "lateral", "jet", 1, 7, "best frequency (log2 Hz - 6)")
    fig.suptitle(tag)
    fig.savefig(out / "fig_hemifield_motion_tonotopy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

if "tono_bf_hz" in M:
    fig, ax = plt.subplots(figsize=(4, 3.2))
    s = R["G_A1"] & (M["tono_r"] >= thr)
    for h, c in ((0, "C0"), (1, "C3")):
        m = s & (refs["hemi"] == h)
        ax.scatter(pos[m], M["tono_bf_hz"][m], s=12, c=c, label=["left", "right"][h])
    ax.set_yscale("log")
    ax.set_xlabel("A1 position: posteromedial -> anterolateral (mm)")
    ax.set_ylabel("best frequency (Hz)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_tonotopy_gradient.png", dpi=130)
    plt.close(fig)
print(f"\nsummary + figures -> {out}")
