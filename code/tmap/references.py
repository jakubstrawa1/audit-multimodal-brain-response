"""Reference maps on TRIBE's output space (fsaverage5, 20,484 vertices, left then right).

Sources (all small; downloaded once into paths.references, < 1 MB):
  * Benson et al. (2014/2018) retinotopy template: polar angle, eccentricity,
    pRF size and visual-area labels (V1..V3, hV4, VO1/2, LO1/2, TO1/2, V3a/b)
  * Glasser et al. (2016) HCP-MMP1 atlas: V1, MT, MST, A1, auditory belt
  * Wang et al. (2015) probabilistic visual atlas (maximum-probability labels)
all as distributed on fsaverage in the neuropythy package
(github.com/noahbenson/neuropythy, lib/data/fsaverage/surf).

fsaverage5 is the first 10,242 vertices of each fsaverage hemisphere (the
icosahedral meshes are nested), so no resampling is needed.

Label numbers used here were checked against anatomy: Glasser 1 = V1 (calcarine),
23 = MT and 2 = MST (lateral occipito-temporal, ~(+-47, -72, 6)), 24 = A1
(Heschl's gyrus, ~(+-44, -25, 11)), 173/174/124 = medial/lateral/parabelt.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .common import N_HEMI

BASE_URL = "https://raw.githubusercontent.com/noahbenson/neuropythy/main/neuropythy/lib/data/fsaverage/surf"
FILES = ["benson14_angle.v4_0.mgz", "benson14_eccen.v4_0.mgz", "benson14_sigma.v4_0.mgz",
         "benson14_varea.v4_0.mgz", "glasser16_atlas.v1_0.mgz", "wang15_mplbl.v1_0.mgz"]

BENSON_AREAS = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2", 7: "LO1", 8: "LO2",
                9: "TO1", 10: "TO2", 11: "V3b", 12: "V3a"}
GLASSER = {"V1": [1], "MT": [23], "MST": [2], "A1": [24], "MBelt": [173], "LBelt": [174], "PBelt": [124]}


def download(ref_dir: Path, local_source: str | None = None) -> None:
    """Fetch the template files (or copy them from a local neuropythy checkout)."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        for f in FILES:
            dst = ref_dir / f"{hemi}.{f}"
            if dst.exists():
                continue
            if local_source:
                dst.write_bytes((Path(local_source) / f"{hemi}.{f}").read_bytes())
            else:
                urllib.request.urlretrieve(f"{BASE_URL}/{hemi}.{f}", dst)


def _load(ref_dir: Path, name: str) -> np.ndarray:
    import nibabel as nib
    out = []
    for hemi in ("lh", "rh"):
        d = np.nan_to_num(np.asarray(nib.load(ref_dir / f"{hemi}.{name}").get_fdata()).ravel())
        out.append(d[:N_HEMI])
    return np.concatenate(out)


def mesh_fsaverage5() -> dict:
    """Vertex coordinates (pial, sphere) for both hemispheres, stacked left then right."""
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out = {}
    for kind in ("pial", "sphere", "infl"):
        out[kind] = np.vstack([surface.load_surf_mesh(fs[f"{kind}_left"]).coordinates,
                               surface.load_surf_mesh(fs[f"{kind}_right"]).coordinates])
    out["faces_left"] = surface.load_surf_mesh(fs["pial_left"]).faces
    out["faces_right"] = surface.load_surf_mesh(fs["pial_right"]).faces
    return out


def load_references(ref_dir: Path) -> dict:
    """Everything as arrays of length 20,484 (left hemisphere first)."""
    hemi = (np.arange(2 * N_HEMI) >= N_HEMI).astype(int)
    angle = _load(ref_dir, FILES[0])          # unsigned 0..180, 0 = upper vertical meridian
    ecc = _load(ref_dir, FILES[1])
    sigma = _load(ref_dir, FILES[2])
    varea = _load(ref_dir, FILES[3]).astype(int)
    glasser = _load(ref_dir, FILES[4]).astype(int)
    wang = _load(ref_dir, FILES[5]).astype(int)
    # left hemisphere represents the RIGHT visual field and vice versa
    side = np.where(hemi == 0, 1.0, -1.0)
    rad = np.radians(angle)
    vf_x = side * ecc * np.sin(rad)
    vf_y = ecc * np.cos(rad)
    signed_angle = side * angle               # -180..180, same convention as stimuli.py
    rois = {name: np.isin(varea, [k]) for k, name in BENSON_AREAS.items()}
    rois["V1-V3"] = np.isin(varea, [1, 2, 3])
    for name, labs in GLASSER.items():
        rois[f"G_{name}"] = np.isin(glasser, labs)
    rois["MT+"] = rois["G_MT"] | rois["G_MST"]
    rois["auditory_core_belt"] = rois["G_A1"] | rois["G_MBelt"] | rois["G_LBelt"] | rois["G_PBelt"]
    return {"hemi": hemi, "angle": angle, "signed_angle": signed_angle, "ecc": ecc,
            "sigma": sigma, "vf_x": vf_x, "vf_y": vf_y, "varea": varea, "glasser": glasser,
            "wang": wang, "rois": rois}


def heschl_axis_position(pial: np.ndarray, mask: np.ndarray, hemi: np.ndarray) -> np.ndarray:
    """Position of each A1 vertex along its long (Heschl's gyrus) axis, running
    from POSTEROMEDIAL (low values) to ANTEROLATERAL (high values), estimated per
    hemisphere from the A1 label's own shape (first principal axis in the
    lateral-|x| / anterior-y plane). NaN outside the mask."""
    pos = np.full(len(mask), np.nan)
    for h in (0, 1):
        sel = mask & (hemi == h)
        if sel.sum() < 3:
            continue
        P = np.c_[np.abs(pial[sel, 0]), pial[sel, 1]]
        P = P - P.mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        ax = vt[0]
        if ax[0] < 0:                 # orient so that lateral (larger |x|) is positive
            ax = -ax
        pos[sel] = P @ ax
    return pos
