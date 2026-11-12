#!/usr/bin/env python
"""Step 0 - download the reference maps (< 1 MB, once).

usage: python scripts/00_get_references.py config.yaml [--local-source DIR]
"""
import argparse

import _setup  # noqa: F401
import numpy as np

from tmap.common import load_config
from tmap.references import download, load_references, mesh_fsaverage5

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--local-source", default=None, help="neuropythy .../lib/data/fsaverage/surf folder")
a = p.parse_args()
cfg = load_config(a.config)
download(cfg["paths"]["references"], a.local_source)
refs = load_references(cfg["paths"]["references"])
mesh = mesh_fsaverage5()
print("Reference ROIs on fsaverage5 (vertices; centre of mass in MNI-like coords, left hemisphere):")
for name in ["V1", "V2", "V3", "TO1", "MT+", "G_A1", "G_MBelt", "G_LBelt"]:
    m = refs["rois"][name]
    left = m & (refs["hemi"] == 0)
    print(f"  {name:8s} {m.sum():5d}   {np.round(mesh['pial'][left].mean(0)).astype(int)}")
