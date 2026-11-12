# Does a brain foundation model know the brain's maps?
### An in-silico mapping audit of TRIBE v2: retinotopy, tonotopy and motion

This folder is a complete, runnable project. You show Meta's TRIBE v2 the classic
stimuli neuroscientists use to map sensory cortex (checkerboard wedges and rings, hemifield
flicker, moving dots, tone sweeps). You fit the same models to its *predicted* brain
responses that you would fit to real fMRI. Then you check the maps it produces against
published human atlases.

No fMRI data to download. The only heavy step is running TRIBE on about 12 minutes of video,
which a free Google Colab GPU handles.

---

## 0. Why this is a real gap (read first)

* **What TRIBE v2 is:** it predicts fMRI responses on the fsaverage5 cortical surface
  (20,484 vertices) from video, audio and text (d'Ascoli et al. 2026, arXiv 2605.04326).
* **What the paper tested:** its "in-silico neuroscience" experiments covered only
  - visual category localizers (faces → FFA, places → PPA, bodies → EBA, words → VWFA);
  - language contrasts.
* **What it did not test:** retinotopy, tonotopy and motion were **not** tested.
* **Why retinotopy should fail:** the paper's methods say the video features (V-JEPA 2
  patch tokens) are **averaged over space**. The authors note this discards position
  information and expect low-level, retinotopic visual areas to suffer.

So there is a clear, architecture-based prediction nobody has checked. Your paper checks it
quantitatively, and checks the auditory and motion counterparts the paper didn't cover.

**Working title:** *What a brain foundation model doesn't know: in-silico retinotopy,
tonotopy and motion mapping of TRIBE v2.*

### Hypotheses (pre-register these on OSF before running the real model on video)

| | Prediction | Why |
|---|---|---|
| **H1** polar angle / contralaterality | **Absent**: no contralateral V1 response to hemifield stimuli, no polar-angle map | spatial averaging of video features |
| **H2** eccentricity | **Partly present**: an eccentricity-like gradient from global image statistics (ring size changes the whole image's statistics) | averaged features still change with ring size |
| **H3** motion | **Present**: moving > static dots peaks in MT+ | V-JEPA 2 is a video model, and the paper found a "motion" component |
| **H4** tonotopy | **Open**: best-frequency gradient in A1 | audio features are not pooled over frequency, but training data were mostly speech and film sound |

Any outcome is publishable:

* If H1 holds, you have quantified a limitation that matters for everyone using TRIBE for
  "in-silico" experiments on early visual cortex.
* If H1 fails, position information leaks through the averaged tokens (e.g. through position
  encodings or border effects). That is a surprising finding about what averaged
  transformer features retain.

---

## Step 0 — Run everything with a stand-in model (5 minutes, laptop)

Before spending GPU time, run the full pipeline with a **stand-in model**: a fake brain with
known behaviour, producing predictions in exactly TRIBE's format. It has two switches:

* **Retinotopic + tonotopic:** visual vertices get pRFs from the Benson template; A1 gets a
  frequency gradient. This is what a model that "knows" the maps would produce.
* **Pooled + flat:** visual vertices only see global image statistics (stimulated area,
  wedge orientation); auditory vertices only see sound on/off. This is what TRIBE's
  architecture predicts.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/00_get_references.py config.yaml          # < 1 MB of atlases
python scripts/01_make_stimuli.py config.yaml            # ~2 min: 4 videos + 2 audio files
python scripts/02b_standin.py config.yaml --visual retinotopic --auditory tonotopic
python scripts/02b_standin.py config.yaml --visual pooled --auditory flat
for d in predictions_standin_retinotopic_tonotopic predictions_standin_pooled_flat; do
  python scripts/03_fit_maps.py config.yaml --preds work/$d
  python scripts/04_evaluate.py config.yaml --preds work/$d
done
```

What we got when testing this pipeline:

| Test | Retinotopic stand-in | Pooled stand-in |
|---|---|---|
| R1 V1 contralateral effect (d) | 2.73 (spin p = 0.007) | 0.01 (p = 0.12) |
| R2 polar angle vs template (circular r) | 0.997 (p = 0.001) | −0.01 (p = 0.53) |
| R2b pRFs on the contralateral side | 100% | 52–57% (chance) |
| R3 eccentricity vs template (Spearman) | 0.991 (p = 0.001) | 0.01–0.12 (see note) |
| R4 wedge power at f vs 2f | 0.60 (position-driven) | 0.48 (orientation-driven) |
| M1 MT+ moving − static (d) | 3.95 (p = 0.001) | 3.90 (planted in both) |
| T1 A1 frequency gradient (Spearman) | −0.99 (spin p = 0.006; label p = 0.007) | +0.12 (p ≈ 0.75) |
| T2 best-frequency spread (octaves) | 2.0 | 0.0 |

Note on R3: across four random seeds of the pooled model, the eccentricity correlation was
significant once (0.12, p = 0.04) and not in the other three. Treat a weak eccentricity
effect with care, and use the controls in Step 4.

Example figures are in `example_results/`.

---

## First real result: TRIBE v2 has no tonotopic map (preliminary)

While building this project, the real TRIBE v2 was run on the two tone-sweep files on a
2-CPU cloud machine. The predictions are included in `example_results/tribev2_real_tonotopy/`,
so you can re-run Steps 2–3 on them without running the model.

* **Auditory cortex responds to the tones:**
  - every core and belt vertex is well fitted (median r = 0.89);
  - A1 is strongly suppressed during silence (z ≈ −2.1).
* **There is no map:**
  - all auditory vertices prefer the same mid band, about 1.1–1.3 kHz (the speech range),
    with very broad tuning (most vertices sit at the widest width tested, 2 octaves);
  - best-frequency spread across core and belt is **0.14 octaves**, against 2.0 in the
    planted-map control;
  - **no gradient** along A1 (ρ = −0.07; spin p = 0.42; label-permutation p = 0.34);
  - the posteromedial and anterolateral halves of A1 have identical band-by-band profiles.
* **Timing:** responses peak 2–3 s later than TRIBE's built-in 5-second shift assumes. The
  lag grid in `config.yaml` was widened to −3…+8 s, so fits are not distorted.

The obvious next control is loudness-equalised tones (Step 4). Without it, "prefers ~1.2 kHz
everywhere" could partly reflect loudness rather than frequency. The *absence of a spatial
gradient* doesn't depend on loudness, though, because every vertex sees the same sounds.
Treat this as preliminary until the video experiments and controls are done.

---

## Step 1 — Run the real TRIBE v2

### What it needs

* **Model:** `facebook/tribev2` (0.7 GB).
* **Audio encoder:** Wav2Vec-BERT 2.0 (2.3 GB).
* **Video encoder:** V-JEPA 2 giant (large; downloaded on first video run).
* **No Hugging Face token:** the stimuli contain no speech, so TRIBE removes its text
  extractor itself and Llama is never loaded.
* **Licence:** TRIBE v2 is CC BY-NC 4.0. Research and publication are fine; commercial use
  isn't.

### Two problems in the official code you will hit (handled by `02_run_tribe.py`)

1. **CUDA hard-coded.** The released config sets the feature extractors to `cuda`, so the
   official code crashes on any machine without an NVIDIA GPU, including Macs. The script
   switches them to `cpu` when there is no CUDA.
2. **No Apple GPU.** Meta's `neuralset` only accepts `cpu` or `cuda` for the encoders, so a
   Mac runs them on the CPU.

### Where to run what

* **Audio (tonotopy) — your Mac is fine:**
  ```bash
  uv pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"
  python scripts/02_run_tribe.py config.yaml --only tono_up tono_down
  ```
  `tribev2` pins `torch<2.7`, so install it in its own virtual environment. Measured
  here: a one-time ~3 GB download, then about 4.5–5 min per 280-second tone file on a
  2-CPU cloud VM. A MacBook should be faster.
* **Video (retinotopy, hemifield, motion) — use Colab:** the video encoder runs a 64-frame
  clip twice per second of video, so this is the expensive part. Open
  `notebooks/colab_run_tribe.ipynb` in Google Colab, choose a T4 GPU (free), upload
  `tribe-mapping.zip` and *Run all*. It regenerates the stimuli in the cloud, runs TRIBE,
  fits and evaluates the maps, and downloads a small results zip. Each experiment is saved
  as soon as it finishes. Turn on the Google Drive option if your session might disconnect.
* **Mac-only alternative for video:** an unofficial Rust port (`tribev2` on crates.io) claims
  numerical parity and runs on Apple's GPU via Metal. It's untested here; if you use it,
  check a few predictions against the Python output first.

---

## Step 2 — Fit and evaluate

```bash
python scripts/03_fit_maps.py config.yaml     # pRFs, contrasts, frequency tuning (seconds)
python scripts/04_evaluate.py config.yaml     # comparisons with atlases + nulls (~1 min)
```

**`03_fit_maps.py`** fits the same models used on real fMRI to each vertex's *predicted*
time course:

* **Retinotopy:**
  - a 2D Gaussian pRF (position and size) via grid search, from wedges and rings jointly;
  - Fourier power at the wedge frequency (f) vs twice that frequency (2f). A position-coding
    vertex responds once per rotation; an orientation-sensitive but position-blind vertex
    responds twice, because a wedge looks the same when pointing up and down.
* **Hemifield:** contralateral − ipsilateral effect size.
* **Motion:** moving − static effect size.
* **Tonotopy:** Gaussian tuning on log frequency, giving best frequency and tuning width,
  from up and down sweeps jointly.

TRIBE already shifts its outputs 5 s back to undo the haemodynamic lag. The fitting allows
an extra −3…+8 s, because the real model's auditory responses turned out to lag a further
2–3 s.

**`04_evaluate.py`** compares the maps with references on the same fsaverage5 mesh, all
downloaded in Step 0 from the neuropythy package:

* **Retinotopy:** the Benson retinotopy template (polar angle, eccentricity, V1–V3).
* **Motion:** the Glasser HCP-MMP atlas (MT + MST).
* **Tonotopy:** A1. The test is whether best frequency falls from the **posteromedial** to
  the **anterolateral** end of A1 (along Heschl's gyrus), the best-replicated pattern in
  human tonotopy.

**Nulls.** TRIBE is deterministic, so there is no sampling noise. The p-values ask a
different question: *is the map's spatial layout aligned with the brain's?*

* **Spin test** (maps rotated on the sphere): keeps each map's smoothness and breaks its
  alignment with anatomy.
* **Stimulus-label permutation** (tonotopy): shuffles which band is which frequency. This
  checks that the gradient depends on frequency *order*, not on timing.

Outputs land in `work/results/<predictions folder>/`: `summary.csv` plus
`fig_retinotopy.png`, `fig_hemifield_motion_tonotopy.png` and
`fig_tonotopy_gradient.png`.

---

## Step 3 — Interpreting the outcome

| Pattern | Meaning |
|---|---|
| R1 ≈ 0, R2 ≈ 0, R2b ≈ 0.5, R4 < 0.5 | **H1 confirmed**: no positional code; responses follow orientation and area, not location. In-silico experiments that depend on *where* something is in the visual field are invalid with TRIBE v2. |
| R1 > 0 and R2b > 0.5, but R2 weak | Coarse left/right information survives averaging; fine polar angle does not. |
| R3 > 0 while R1/R2 ≈ 0 | Eccentricity from global statistics (ring size), not position. Confirm with the controls below. |
| M1 large, M2 well above MT+'s share of cortex | **H3**: motion selectivity in the right place. |
| T1 < 0 with both p-values small, T2 > 1 octave | **H4**: a genuine tonotopic gradient, learned without ever being trained on tones. |
| T2 ≈ 0 | Auditory vertices share one tuning: no tonotopy. |

---

## Step 4 — Controls and robustness (what reviewers will ask)

All five controls below are already built and ready to run: `scripts/build_controls.py` has
generated each control's stimuli/config locally and pre-populated its predictions folder with
the baseline `.npz` for every experiment that control doesn't change, so the GPU step only
computes what's actually new. See `controls/README.md` for the exact commands
(`scripts/run_controls_remote.sh` runs all seven end to end).

1. **Silent audio vs no audio.** Videos carry a silent track by default. Re-run video
   experiments with `--no-audio` (drops the audio input) into a separate predictions folder
   and check that the visual maps don't change.
2. **Is eccentricity really about position?** Make a ring control: change `ring_min_ecc_deg`
   and the ring width so the stimulated *area* stays constant while position changes. If
   R3 disappears, eccentricity was driven by area.
3. **Stimulus size.** Re-run with `screen.max_ecc_deg` 5 and 20. This changes only which
   template vertices are compared; rank-based results should be stable.
4. **Wedge width** (`wedge_width_deg` 30/90) and **check size** in `stimuli.checkerboard`.
5. **Tonotopy loudness.** Tones are equal in amplitude, not equal in loudness. A
   loudness-equalised version (ISO 226 contours) rules out "best frequency = loudest
   frequency".
6. **Positive control.** The stand-in model shows the analysis *can* find maps when they
   exist (Step 0). Report it as Figure 1 of the methods.

---

## Step 5 — The paper

**Figures:**

1. Stimuli and the analysis pipeline, with stand-in validation (positive and negative
   control).
2. Hemifield contrast on the medial surface.
3. pRF polar-angle/eccentricity maps vs the Benson template (`fig_retinotopy.png`), with
   spin-test distributions.
4. Wedge f vs 2f power: position vs orientation.
5. Motion contrast with the MT+ outline.
6. Tonotopy: best-frequency map and the gradient along A1 (`fig_tonotopy_gradient.png`).
7. Controls (silent vs no audio, ring-area control, stimulus size).

**Venues:**

* a NeurIPS/ICLR workshop on neuro-AI or model evaluation;
* CCN (two-page format);
* a short paper in *Imaging Neuroscience* or *PLOS Computational Biology*.

The result is also directly useful to the TRIBE authors, so consider opening a GitHub issue
with the findings.

**Timeline (part-time):**

| Week | What |
|---|---|
| 1 | Stand-in run, pre-registration, real audio run on your Mac |
| 2 | Colab video runs, fitting, evaluation |
| 3 | Controls (Step 4) |
| 4 | Write |

---

## Compute (rough)

| Step | Where | Time |
|---|---|---|
| Stimuli, fitting, evaluation, stand-in | laptop | minutes |
| TRIBE on 9.3 min of tone sweeps | laptop CPU | ~10 min measured on a 2-CPU cloud VM (+ one-time ~3 GB download) |
| TRIBE on 12.4 min of video | free Colab T4 | likely 1–3 h in total (rough estimate from model size, not measured) |
| TRIBE on video | Mac CPU | many hours; possible overnight |

---

## Code map

```
config.yaml                stimulus, fitting and evaluation settings
tmap/stimuli.py            wedge/ring/hemifield checkerboards, moving dots, tone sweeps (+ design files)
tmap/references.py         Benson / Glasser / Wang maps on fsaverage5; Heschl's-gyrus axis
tmap/fit.py                pRF and frequency-tuning grid search on predicted time courses; contrasts; f vs 2f
tmap/spin.py               spin-test nulls
tmap/standin.py            the stand-in model with known behaviour
tmap/predio.py             prediction files and segment timing
scripts/00..04             the numbered steps; 02b = stand-in predictions
notebooks/colab_run_tribe.ipynb   cloud run on a free GPU
```

## Limitations to state in the paper

* **Resolution:** fsaverage5 vertices are about 4 mm apart. A1 is only ~45 vertices per
  hemisphere at this resolution, so fine tonotopic structure is out of reach.
* **Average subject:** TRIBE predicts a group-average subject, so maps are compared with
  group templates, not individuals.
* **Visual angle:** this is undefined for TRIBE. The frame is *defined* to span
  `max_ecc_deg`; results use rank-based comparisons that don't depend on this choice.
* **Tonotopic layout:** the orientation of tonotopic gradients relative to Heschl's gyrus is
  still debated. T1 tests the posteromedial-high / anterolateral-low pattern within A1; T2
  is direction-free.

## References

* d'Ascoli S. et al. (2026) A foundation model of vision, audition, and language for
  in-silico neuroscience. arXiv 2605.04326.
* Benson N. C. et al. (2014) Correction of distortion in flattened representations of the
  cortical surface allows prediction of V1–V3 functional organization from anatomy.
  *PLOS Comput Biol*.
* Benson N. C. & Winawer J. (2018) Bayesian analysis of retinotopic maps. *eLife*.
* Glasser M. F. et al. (2016) A multi-modal parcellation of human cerebral cortex.
  *Nature*. HCP data-use terms apply to the atlas.
* Wang L. et al. (2015) Probabilistic maps of visual topography in human cortex.
  *Cereb Cortex*.
* Dumoulin S. O. & Wandell B. A. (2008) Population receptive field estimates in human
  visual cortex. *NeuroImage*.
* Alexander-Bloch A. F. et al. (2018) On testing for spatial correspondence between maps of
  human brain structure and function. *NeuroImage*.
* Saenz M. & Langers D. R. M. (2014) Tonotopic mapping of human auditory cortex.
  *Hear Res*.
* Moerel M., De Martino F. & Formisano E. (2014) An anatomical and functional topography
  of human auditory cortical areas. *Front Neurosci*.
* Huk A. C., Dougherty R. F. & Heeger D. J. (2002) Retinotopy and functional subdivision of
  human areas MT and MST. *J Neurosci*.
