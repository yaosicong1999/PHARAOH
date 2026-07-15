# PHARAOH — version 4.1 (unified pipeline, GUI + CLI)

PHARAOH is a scalable, generalizable framework for multimodal tissue-image
alignment and spatial-transcriptomics enhancement. It performs fast, robust,
GPU-free registration between a **moving** image and a fixed **H&E** image,
supporting both same-section and adjacent-section alignment.

**version_4.1 unifies the two earlier pipelines** (`version_4.0` DAPI→H&E and
`version_4.0_he` H&E→H&E) into one project that handles **both** moving
modalities and can run **two ways**:

- **DAPI → H&E** — register DAPI/DNA from multiplexed imaging (Xenium, CosMx,
  Orion, CODEX, CyCIF) to histology **H&E**.
- **H&E-FG → H&E** — register a second (foreground) **H&E** to the fixed H&E.
  In general the **H&E-FG is expected to be smaller** than (a sub-region of) the
  fixed H&E — i.e. the foreground H&E is warped/placed onto the larger background
  H&E.

Both modes share the same `engine_dapi/` / `engine_he/` compute and the single
integrated `parameters.json`.

---

## 📂 Required input files

1. A **moving** image, one of:
   - **DAPI/DNA** channel (on the first channel if multi-channel), `.ome.tif` /
     `.tif` / `.jpg`. Xenium DAPI is usually `morphology_focus.ome.tif` or
     `morphology_focus/morphology_focus_0000.ome.tif`.
   - or a second **H&E** image (**H&E-FG**), `.ome.tif` / `.tif` / `.jpg`.
2. A fixed **H&E** image (same or adjacent section), `.ome.tif` / `.tif` / `.jpg`.
3. *(Optional, Xenium)* `cells.csv.gz` with cell centroids, for Stage-6
   visualization.

> For best alignment, use **raw, unscaled** images.

---

## ⚙️ Setup

Install into the dedicated Conda environment (from the repo root), then enter
this folder:

```bash
bash install_conda_env.sh      # ~2–5 min first time
conda activate PHARAOH
cd version_4.1
```

Tested on Apple Silicon (M1/M2 Pro), Windows 11, and Ubuntu (ARM64); supports
native ARM64 and Rosetta 2 x86 on macOS.

---

## 🚀 Two ways to run

**1. GUI** — interactive; choose the moving modality in Stage 1:
```bash
./run_pharaoh_gui.sh           # opens gui_pipeline.py
```

**2. CLI** — fully automatic / headless; **ORB** replaces the manual Stage-2 overlay:
```bash
./run_pharaoh_cli.sh --dapi="dapi.ome.tif" --he="he.ome.tif"    # DAPI → H&E
./run_pharaoh_cli.sh --hefg="hefg.ome.tif" --he="he.ome.tif"    # H&E-FG → H&E
```
Extra CLI flags: `--run-dir DIR`, `--force` (recompute), `--stop-after N`. Both
`--flag=value` and `--flag value` work; `--he0` is accepted as an alias for
`--hefg`. Use `PYTHON=/path/to/python ./run_pharaoh_*.sh` to pick an interpreter.

Run folders are created as `version_4.1/runs_YYYYMMDDHHMMSS/`. The GUI writes a
`mode` field into `images_info.json`; the launcher then dispatches Stages 2–6 to
`engine_dapi/` or `engine_he/` accordingly.

---

## 🖥️ GUI usage

Launch `./run_pharaoh_gui.sh`. Click **New RUN_DIR** (top right) to start a run,
or **Choose RUN_DIR** to resume an existing `runs_<id>/`. The first stage launch
may take ~1–2 min to load dependencies.

### Stage 1 — Select images
One window, two columns:
- **Left**: the fixed **H&E** — `Select H&E Image`, plus a **threshold** slider
  (H-channel visualization only).
- **Right (moving)**: two stacked buttons — **`Select DAPI Image`** /
  **`Select H&E-FG Image`**. Pick one to set the mode:
  - **DAPI** → shown LUT-colored, with a **DAPI LUT threshold** slider (keep it
    clearly visible, not over-saturated/patchy).
  - **H&E-FG** → a second brightfield H&E, with an **H&E-FG threshold** slider.

  Use the **Rotate / Flip** buttons to match the moving image's orientation to
  the H&E. Then click **Confirm & Save**.
  Outputs: `images_info.json` (+ a `mode` field) and `1_*` PNGs.

### Stage 2 — Initial alignment
An overlay of the moving image on the H&E:
- **Auto-align (ORB)** computes a rough alignment automatically in the
  background; click the button to apply it (it shows the inlier count).
  > ⚠️ Auto ORB matching can **sometimes fail** (few/low-quality feature matches,
  > or very different DAPI↔H&E appearance). If the applied result looks wrong (or
  > the inlier count is low), just ignore it and align manually below, or re-run
  > it. In the CLI this is unattended, so a bad ORB init can propagate — check the
  > Stage-2 inlier count / the `2_manual_overlay_*` preview.
- Manual refine: **Affine** mode — drag blue corners to scale (hold **Shift** for
  proportional), drag to move, hold **Ctrl** to hide/reveal the overlay. Click
  **Mode: Affine** to switch to **Perspective** (drag corners to warp).
  `Ctrl` + `+`/`-` adjusts the view size.
- **Load H (.json)** loads a previously saved matrix (must match the pyramid
  level in `images_info.json`).
- **Save Alignment** writes `manual_initial_alignment.json` (+ `2_*` PNGs).

### Stage 3 — Extract tiles
- **Sample Tile Centroids** — samples tiles per `parameters.json`; auto-reduces
  the count if it would oversample.
- **Tile Pilot Examination** *(optional; recommended for adjacent sections)* —
  10 pilot tiles for tuning the DAPI offset / H&E intensity threshold.
- **Extract Current Tiles** — extracts all sampled tiles.
  Outputs: `sampled_points.json`, `tiles/`, (`pilot_tiles/`), `3_*` PNGs.

### Stage 4 — Extract nuclei patches
- **Run Nuclei Masking** — nuclei masks for each moving + H&E tile.
- **Run Standout Nuclei Detection** — aligns each mask pair to find anchor
  nuclei (falls back to aligned tile centers when few standouts exist; ~3–5 min).
- **Run Nuclei Patch Cropping** — paired patches from the anchors.
  Output: `nuclei_patches/`.

### Stage 5 — Nuclei gallery + final alignment
Inspect the paired patches. **Calculate alignment matrix** computes and saves the
final homography (`<moving>_to_he_homography_level0.json`, i.e.
`dapi_to_he_…` or `he0_to_he_…`). You can toggle auto centroids, switch DAPI
LUT/Raw, drop a pair, or (test feature) click an enlarged image to add/delete
refined keypoints.

### Stage 6 — View final alignment
**Load Keypoints + Alignment Matrix** builds the overlay; **Toggle H&E /
Overlay** compares; **Load Cell Data (`cells.csv.gz`)** overlays Xenium
centroids. Outputs: `6_*` PNGs / GIF.

---

## ⌨️ CLI details

`run_pharaoh_cli.sh` runs the same six stages headlessly:
- **Stage 1** reads the images and builds masks (GUI defaults, identity orientation).
- **Stage 2** uses **automatic ORB** registration instead of the manual overlay.
- **Stages 3 & 6** reuse the engine GUI compute via reflection (no window).
- **Stage 4** runs the engines' `4a/4b/4c` scripts.
- **Stage 5** is a headless transcription of the alignment computation.

Final artifact: `<moving>_to_he_homography_level0.json` (+ `.csv`); Stage 6 also
writes `6_overlay_*` previews. ORB on cross-modal DAPI/H&E pairs can misalign —
accepted for the CLI; the recorded inlier count is the quality signal.

---

## 🕹️ Parameter controls (`parameters.json`)

One integrated, **mode-nested** file at the top level:

```json
{ "dapi": { "stage3": {...}, "stage4a": {...}, ... },
  "he0fg": { "stage3": {...}, "stage4a": {...}, ... } }
```

Edit the **`dapi`** block for DAPI runs and the **`he0fg`** block for H&E-FG
runs. Every engine stage reads it directly via `my_utils.cli_params_block()`
(which selects the block from its engine folder), so there is **no per-engine
copy**. ***Italic*** parameters are the important ones. The moving-modality keys
are `dapi_*` in the `dapi` block and `he0_*` in the `he0fg` block; the rest match.

#### 🧩 Stage 3 — Tile sampling
- ***n_tiles***: number of tiles sampled from the slide.
- ***tile_size***: pixel size of each sampled tile.
- **min_dist_factor**: larger → more spatial separation between tiles.
- **dapi_level_override / he0_level_override / he_level_override**: pyramid-level
  override for the moving / H&E image; `"None"` = auto-select.
- **he_tile_margin_ratio**: extra margin (relative to tile size) for H&E tiles.

#### 🧬 Stage 4A — Nuclei mask extraction
- **DAPI** block: `dapi_thr_offset`, `dapi_mask_min_area_factor`,
  `dapi_mask_upscale_factor`.
- **H&E-FG** block: `he0_mask_n_smooth`, `he0_mask_intensity_threshold`,
  `he0_mask_upscale_factor`.
- **H&E (both)**: `he_mask_n_smooth`, `he_mask_intensity_threshold`,
  `he_mask_upscale_factor`.

#### 🔍 Stage 4B — Tile matching & global initialization
- **Filtering**: ***good_nuclei_min***, ***min_good_tiles***,
  ***fallback_score_thr***, ***min_fallback_tiles***.
- **Pair selection**: ***pair_top_k***, ***pairs_to_take_per_tile***.
- **Alignment search** (`phase1` coarse / `phase2` refine / `phase3` fine):
  `n_tiles`, `ds`, `scale_*`, `shift_*` ranges/steps.

#### 🔬 Stage 4C — Patch extraction
- ***dapi_patch_len*** / ***he0_patch_len***: patch size (px) for the moving
  modality. Other keys mirror Stage 4A.

#### 🧠 Stage 5 — Final transformation
- ***transform_mode***: `"affine"`, `"homography"` (default), `"tps"`,
  `"local_tps"`. *(CLI Stage 5 supports `affine`/`homography`.)*
- ***balance_points_bool***: `false` = all nuclei pairs; `true` = grid-balanced
  sampling to reduce spatial bias.

---

## 🗂️ Layout

```
version_4.1/
  run_pharaoh_gui.sh    # GUI launcher -> gui_pipeline.py
  gui_pipeline.py       # unified GUI launcher (dispatches Stages 2-6 by mode)
  gui_read_images.py    # unified GUI Stage 1 (H&E + DAPI/H&E-FG chooser)

  run_pharaoh_cli.sh    # CLI launcher -> run_pharaoh.py
  run_pharaoh.py        # headless orchestrator (Stages 1-6), ORB Stage 2
  cli_common.py         # shared CLI helpers
  cli_stage{1,2_orb,3,5,6}.py   # headless stage runners (reuse the engines)

  parameters.json       # integrated params: {"dapi": {...}, "he0fg": {...}}
  my_utils.py  glasbey*.lut     # shared assets
  engine_dapi/          # DAPI -> H&E stage scripts (2-6, 4a/4b/4c, …)
  engine_he/            # H&E-FG -> H&E stage scripts (2-6, 4a/4b/4c, …)
```

## Notes
- Run dirs default to `version_4.1/runs_<timestamp>` and are independent of the
  engine folders (params are resolved from the top-level file, not the run dir).
- Stage 1 saves the chosen orientation as `*_gui_affine` / `*_orientation_case`.
- The engines still contain their original single-modality launchers
  (`engine_*/0_pipeline.py`, `1_read_*.py`); the unified launcher supersedes
  them — only Stages 2–6 of each engine are used here.
- Demo GIFs for each stage are in the repository-root `README.md`.
