#!/usr/bin/env python
"""Step 1 - generate the localizer stimuli and their design files.

  retino_wedge.mp4   rotating checkerboard wedge (clockwise, then counter-clockwise)
  retino_ring.mp4    expanding, then contracting checkerboard ring
  hemifield.mp4      left / right / blank hemifield blocks
  motion.mp4         moving (expanding/contracting) vs static dots
  tono_up.wav        tone-pip bands sweeping 175 Hz -> 5.6 kHz
  tono_down.wav      ... and back down
All videos are 512 x 512 with a silent audio track.

usage: python scripts/01_make_stimuli.py config.yaml
"""
import sys
import time

import _setup  # noqa: F401

from tmap.common import load_config, save_json
from tmap import stimuli

cfg = load_config(sys.argv[1])
out = cfg["paths"]["stimuli"]
manifest = []
for label, fn in [("wedge", lambda: stimuli.make_retinotopy(out, cfg, "wedge")),
                  ("ring", lambda: stimuli.make_retinotopy(out, cfg, "ring")),
                  ("hemifield", lambda: stimuli.make_hemifield(out, cfg)),
                  ("motion", lambda: stimuli.make_motion(out, cfg)),
                  ("tono_up", lambda: stimuli.make_tonotopy(out, cfg, "up")),
                  ("tono_down", lambda: stimuli.make_tonotopy(out, cfg, "down"))]:
    t0 = time.time()
    m = fn()
    manifest.append(m)
    print(f"  {m['file']:18s} {m['duration']:6.0f} s  ({time.time() - t0:.0f} s to render)", flush=True)
save_json(manifest, out / "manifest.json")
vid = sum(m["duration"] for m in manifest if m["modality"] == "video")
aud = sum(m["duration"] for m in manifest if m["modality"] == "audio")
print(f"\nTotal: {vid / 60:.1f} min of video, {aud / 60:.1f} min of audio -> {out}")
