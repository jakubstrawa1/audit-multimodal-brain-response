#!/usr/bin/env bash
# Run on the GPU pod, after uploading tribe-mapping.zip (which now includes
# controls/*.yaml, work/stimuli_*, and the pre-populated work/predictions_* dirs)
# and setting up the .venv exactly as for the main run (see README Step 1).
#
# Each control's predictions dir is already pre-populated with the baseline
# .npz for every experiment that control did NOT change -- 02_run_tribe.py
# skips any output file that already exists, so each line below only computes
# what's actually new for that control.
set -euo pipefail
cd /workspace/tribe-mapping
source .venv/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=300

run() {
  name=$1; shift
  echo "=== $name ==="
  time python scripts/02_run_tribe.py "controls/config_${name}.yaml" \
      --device cuda --cache work/tribe_cache "$@"
  python scripts/03_fit_maps.py "controls/config_${name}.yaml"
  python scripts/04_evaluate.py "controls/config_${name}.yaml"
}

run no_audio --no-audio          # only hemifield/motion/retino_wedge/retino_ring recompute
run ring_area                    # only retino_ring recomputes
run ecc5                         # all 4 video stimuli recompute (smaller frame)
run ecc20                        # all 4 video stimuli recompute (larger frame)
run wedge30                      # only retino_wedge recomputes
run wedge90                      # only retino_wedge recomputes
run loud                         # only tono_up/tono_down recompute

cd work && zip -q -r /workspace/tribe_controls_results.zip \
    predictions_no_audio predictions_ring_area predictions_ecc5 predictions_ecc20 \
    predictions_wedge30 predictions_wedge90 predictions_loud \
    results/predictions_no_audio results/predictions_ring_area results/predictions_ecc5 \
    results/predictions_ecc20 results/predictions_wedge30 results/predictions_wedge90 \
    results/predictions_loud
echo "DONE: /workspace/tribe_controls_results.zip ready to download"
