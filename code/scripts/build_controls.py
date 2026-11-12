#!/usr/bin/env python
"""Step 4 prep - build every control from the README's robustness checklist.

Runs entirely LOCALLY (no GPU, no tribev2/torch needed): writes one config.yaml
variant per control, regenerates only the stimulus/stimuli that control actually
changes, and copies everything unaffected (stimuli + predictions) straight from
the baseline run so the GPU step only has to compute what's new.

Controls built (see README Step 4 / paper Section 4):
  no_audio    - drop the (silent) audio track from videos; same stimuli, new inference
  ring_area   - ring aperture holds stimulated area constant across eccentricity
  ecc5/ecc20  - screen.max_ecc_deg 5 / 20 instead of 10 (all 4 video stimuli change)
  wedge30/90  - retinotopy.wedge_width_deg 30 / 90 instead of 45 (wedge only)
  loud        - tonotopy tones equal-loudness corrected (ISO 226) instead of equal-amplitude

usage: python scripts/build_controls.py config.yaml
"""
import copy
import shutil
import sys
from pathlib import Path

import _setup  # noqa: F401
import yaml

from tmap import stimuli
from tmap.common import ROOT, load_config, load_json, save_json

BASE_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
base_cfg = load_config(BASE_PATH)
with open(BASE_PATH) as f:
    base_raw = yaml.safe_load(f)                  # unresolved, relative paths (to dump back out)

CONTROLS_DIR = ROOT / "controls"
CONTROLS_DIR.mkdir(exist_ok=True)

base_stim_dir = base_cfg["paths"]["stimuli"]
base_pred_dir = base_cfg["paths"]["predictions"]
base_manifest = {m["name"]: m for m in load_json(base_stim_dir / "manifest.json")}
ALL_NAMES = ["retino_wedge", "retino_ring", "hemifield", "motion", "tono_up", "tono_down"]


def write_control_config(name: str, overrides: dict, same_stimuli: bool = False) -> Path:
    """overrides: dotted-path -> value, e.g. {'screen.max_ecc_deg': 5}."""
    raw = copy.deepcopy(base_raw)
    for dotted, val in overrides.items():
        d, *rest = dotted.split(".")
        node = raw[d]
        for k in rest[:-1]:
            node = node[k]
        node[rest[-1]] = val
    raw["paths"]["stimuli"] = base_raw["paths"]["stimuli"] if same_stimuli else f"work/stimuli_{name}"
    raw["paths"]["predictions"] = f"work/predictions_{name}"
    raw["paths"]["results"] = base_raw["paths"]["results"]      # shared: spin-index cache + tags
    raw["paths"]["references"] = base_raw["paths"]["references"]  # shared: same atlases
    path = CONTROLS_DIR / f"config_{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return path


def populate(name: str, cfg: dict, regenerate: dict, reuse_stimuli: bool = False) -> None:
    """regenerate: {experiment_name: callable() -> manifest_dict}. Everything in
    ALL_NAMES not in `regenerate` is copied unchanged from the baseline run."""
    stim_dir, pred_dir = cfg["paths"]["stimuli"], cfg["paths"]["predictions"]
    manifest = []
    for exp_name in ALL_NAMES:
        if exp_name in regenerate:
            m = regenerate[exp_name]()
            manifest.append(m)
            print(f"  [{name}] generated {exp_name} ({m['duration']:.0f}s) -- needs fresh GPU inference")
        else:
            manifest.append(base_manifest[exp_name])
            if not reuse_stimuli:
                src = base_stim_dir / base_manifest[exp_name]["file"]
                shutil.copy2(src, stim_dir / src.name)
                design = base_stim_dir / f"{exp_name}_design.npz"
                if design.exists():
                    shutil.copy2(design, stim_dir / design.name)
            src_pred = base_pred_dir / f"{exp_name}.npz"
            if src_pred.exists() and pred_dir != base_pred_dir:
                shutil.copy2(src_pred, pred_dir / src_pred.name)
                print(f"  [{name}] reused baseline {exp_name}.npz (no GPU needed)")
    if not reuse_stimuli:
        save_json(manifest, stim_dir / "manifest.json")


# ---------------------------------------------------------------- no_audio
# Same stimuli as baseline; only the video experiments need fresh (--no-audio) inference.
cfg = load_config(write_control_config("no_audio", {}, same_stimuli=True))
populate("no_audio", cfg, regenerate={}, reuse_stimuli=True)
for v in ("retino_wedge", "retino_ring", "hemifield", "motion"):
    (cfg["paths"]["predictions"] / f"{v}.npz").unlink(missing_ok=True)  # force fresh --no-audio run
print("  [no_audio] video predictions removed -- rerun with --no-audio to get the comparison")

# ---------------------------------------------------------------- ring_area
cfg = load_config(write_control_config("ring_area", {"retinotopy.ring_area_match": True}))
populate("ring_area", cfg, regenerate={
    "retino_ring": lambda: stimuli.make_retinotopy(cfg["paths"]["stimuli"], cfg, "ring"),
})

# ---------------------------------------------------------------- ecc5 / ecc20
for ecc in (5, 20):
    name = f"ecc{ecc}"
    cfg = load_config(write_control_config(name, {"screen.max_ecc_deg": float(ecc)}))
    populate(name, cfg, regenerate={
        "retino_wedge": lambda cfg=cfg: stimuli.make_retinotopy(cfg["paths"]["stimuli"], cfg, "wedge"),
        "retino_ring": lambda cfg=cfg: stimuli.make_retinotopy(cfg["paths"]["stimuli"], cfg, "ring"),
        "hemifield": lambda cfg=cfg: stimuli.make_hemifield(cfg["paths"]["stimuli"], cfg),
        "motion": lambda cfg=cfg: stimuli.make_motion(cfg["paths"]["stimuli"], cfg),
    })

# ---------------------------------------------------------------- wedge30 / wedge90
for width in (30, 90):
    name = f"wedge{width}"
    cfg = load_config(write_control_config(name, {"retinotopy.wedge_width_deg": float(width)}))
    populate(name, cfg, regenerate={
        "retino_wedge": lambda cfg=cfg: stimuli.make_retinotopy(cfg["paths"]["stimuli"], cfg, "wedge"),
    })

# ---------------------------------------------------------------- loud (equal-loudness tones)
cfg = load_config(write_control_config("loud", {"tonotopy.equalize_loudness": True}))
populate("loud", cfg, regenerate={
    "tono_up": lambda: stimuli.make_tonotopy(cfg["paths"]["stimuli"], cfg, "up"),
    "tono_down": lambda: stimuli.make_tonotopy(cfg["paths"]["stimuli"], cfg, "down"),
})

print("\nAll control configs + stimuli ready under controls/ and work/stimuli_*; "
      "baseline predictions reused wherever a control didn't change that experiment. "
      "See controls/README.md for the exact GPU commands.")
