# Controls (Section 4 / paper robustness checklist)

Everything that doesn't need a GPU is already done locally: each control's stimuli are
generated, and every experiment a control does *not* change has its baseline `.npz`
prediction already copied in (so `02_run_tribe.py` skips it automatically — it only computes
a file if it doesn't already exist).

## What's pre-built

| Control | What changed | What needs fresh GPU inference |
|---|---|---|
| `no_audio` | drop the (silent) audio track from videos | hemifield, motion, retino_wedge, retino_ring |
| `ring_area` | ring aperture holds stimulated area constant across eccentricity | retino_ring only |
| `ecc5` | `screen.max_ecc_deg = 5` (vs. 10) | all 4 video stimuli |
| `ecc20` | `screen.max_ecc_deg = 20` (vs. 10) | all 4 video stimuli |
| `wedge30` | `wedge_width_deg = 30` (vs. 45) | retino_wedge only |
| `wedge90` | `wedge_width_deg = 90` (vs. 45) | retino_wedge only |
| `loud` | tones ISO 226 equal-loudness corrected instead of equal-amplitude | tono_up, tono_down only |

Audio (tono) reruns are fast; video reruns are the expensive part. Rough estimate from the
main run's per-video timings: **~2.5 hours total GPU time on an H100 SXM** for all seven
controls (most of that is `ecc5` + `ecc20`, since both need all 4 videos re-run). You can
comment out any `run ...` lines in `run_controls_remote.sh` to skip controls you don't need
yet, or run them one at a time.

## How to run

Same upload/setup as the main run (`README.md` Step 1), then on the pod:

```bash
cd /workspace/tribe-mapping
bash scripts/run_controls_remote.sh
```

That runs `02_run_tribe.py` (GPU) + `03_fit_maps.py` + `04_evaluate.py` (CPU) for each
control in turn, against its own `controls/config_<name>.yaml`, and zips all seven
`results/predictions_<name>/summary.csv` + figures + raw predictions into
`/workspace/tribe_controls_results.zip` at the end.

To run just one control (e.g. to sanity-check before committing to all seven):

```bash
python scripts/02_run_tribe.py controls/config_wedge30.yaml --device cuda --cache work/tribe_cache
python scripts/03_fit_maps.py controls/config_wedge30.yaml
python scripts/04_evaluate.py controls/config_wedge30.yaml
```

(`no_audio` needs the extra `--no-audio` flag on the `02_run_tribe.py` call — already
included in `run_controls_remote.sh`.)

## Reading the results

Each control writes `work/results/predictions_<name>/summary.csv` in exactly the same format
as the main run's `work/results/predictions_tribev2/summary.csv` (same R0-R4/M1/M2/T0-T2
rows) — diff the two to see what the control changed. Specifically, for the paper:

- **`no_audio`**: compare R1-R4/M1 against baseline `predictions_tribev2` — should be
  essentially unchanged if TRIBE isn't using the silent audio track.
- **`ring_area`**: compare R3 (eccentricity) against baseline — if R3 drops toward 0, the
  baseline's R3 was about stimulated area, not position.
- **`ecc5`/`ecc20`**: all of R1-T2 should be close to baseline's rank-based results (they're
  rank-based specifically so this choice shouldn't matter) — large swings would flag an
  unexpected sensitivity.
- **`wedge30`/`wedge90`**: compare R4 especially, since that's the metric most directly tied
  to wedge width.
- **`loud`**: compare T1/T2 (best-frequency gradient/spread) against baseline — if the
  tonotopic null survives loudness equalization, the paper's "absence of gradient" claim is
  fully settled rather than caveated.
