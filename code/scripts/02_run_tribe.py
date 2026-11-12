#!/usr/bin/env python
"""Step 2 - run TRIBE v2 on every stimulus (needs `pip install tribev2`, see README).

No text model is needed: all stimuli are non-verbal, so we bypass TRIBE's
speech-transcription step and Llama is never used (no Hugging Face token).

Where to run it
  * audio stimuli (tonotopy): fine on a laptop CPU (~1 min of compute per minute
    of audio measured on a 2-core cloud VM; a modern laptop is faster)
  * video stimuli: the V-JEPA 2 video encoder runs on 64-frame clips twice per
    second; use a GPU (free Google Colab works, see notebooks/colab_run_tribe.ipynb)
    or leave a Mac running for many hours.
  * Without an NVIDIA GPU the released config would crash (its extractors are
    hard-coded to CUDA); this script switches them to the CPU. Meta's `neuralset`
    only accepts "cpu" or "cuda" for the encoders, so Apple's GPU is not used.

usage: python scripts/02_run_tribe.py config.yaml [--only tono_up tono_down] [--no-audio]
                                      [--preds DIR]
"""
import argparse
import time
from pathlib import Path

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from tqdm import tqdm

from tmap.common import load_config, load_json
from tmap.predio import infer_segment_times, save_predictions

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--only", nargs="*", default=None, help="experiment names from manifest.json")
p.add_argument("--device", default="auto", help="auto | cuda | cpu (| mps for the TRIBE head only)")
p.add_argument("--cache", default="work/tribe_cache", help="TRIBE feature cache folder")
p.add_argument("--no-audio", action="store_true",
               help="drop the (silent) audio track of videos instead of feeding silence")
p.add_argument("--preds", default=None,
               help="predictions output folder (default: paths.predictions); use a "
                    "separate folder for control runs (e.g. --no-audio) that share the "
                    "same stimuli but must not overwrite the main predictions")
a = p.parse_args()

import torch  # noqa: E402
from tribev2 import TribeModel  # noqa: E402
from tribev2.demo_utils import get_audio_and_text_events  # noqa: E402

cfg = load_config(a.config)
stim_dir = cfg["paths"]["stimuli"]
out_dir = Path(a.preds) if a.preds else cfg["paths"]["predictions"]
out_dir.mkdir(parents=True, exist_ok=True)
device = a.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TRIBE v2 on {device} ...", flush=True)
# The released config hard-codes the feature extractors to CUDA, which crashes on
# machines without an NVIDIA GPU (e.g. Macs). Point them at the CPU instead.
# (neuralset accepts only "cpu"/"cuda" for extractors, so Apple's GPU can't be used here.)
update = None
if not torch.cuda.is_available():
    update = {"data.audio_feature.device": "cpu", "data.video_feature.image.device": "cpu",
              "data.text_feature.device": "cpu", "data.image_feature.image.device": "cpu"}
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=a.cache, device=device,
                                   config_update=update)
tr = float(cfg["fit"]["tr"])

manifest = [m for m in load_json(stim_dir / "manifest.json")
            if not a.only or m["name"] in a.only]
for m in tqdm(manifest, desc="experiments", unit="exp"):
    out = out_dir / f"{m['name']}.npz"
    if out.exists():
        tqdm.write(f"  {m['name']}: already done, skipping")
        continue
    t0 = time.time()
    event = {"type": "Video" if m["modality"] == "video" else "Audio",
             "filepath": str((stim_dir / m["file"]).resolve()),
             "start": 0, "timeline": "default", "subject": "default"}
    # audio_only=True: extract audio from the video, chunk, but NO transcription / text
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    if m["modality"] == "video" and a.no_audio:
        events = events[events["type"] != "Audio"].reset_index(drop=True)
    tqdm.write(f"  {m['name']}: {len(events)} events {sorted(events['type'].unique())}")
    preds, segments = model.predict(events=events)
    starts = [getattr(s, "start", np.nan) for s in segments]
    offsets = [getattr(s, "offset", np.nan) for s in segments]
    times, how = infer_segment_times(starts, offsets, len(preds), tr)
    order = np.argsort(times)
    save_predictions(out, preds[order], times[order], "tribev2", time_method=how,
                     no_audio=a.no_audio, device=device)
    tqdm.write(f"  {m['name']}: {preds.shape} in {time.time() - t0:.0f} s (times from {how})")
    if abs(len(preds) - m["duration"] / tr) > 3:
        tqdm.write(f"  WARNING: {len(preds)} predictions for a {m['duration']} s stimulus - check timing")
