# PHARAOH 5.0

PHARAOH registers a **moving** whole-slide image to a **fixed** whole-slide image, fast and
without a GPU. It was built to place spatial-omics data on histology: the DAPI/DNA image of
Xenium, CosMx, Orion, CODEX or CyCIF onto the H&E image of the same or an adjacent section.
The result is one transformation from moving level-0 pixels to fixed level-0 pixels, which
can be applied to the image itself, to transcripts, cell centroids or cell boundaries.

PHARAOH registers four kinds of image pair, named `<moving>-<fixed>`:

| `--pair` | moving image | fixed image | stages run |
| --- | --- | --- | --- |
| `dapi-he` (default) | DAPI/DNA | H&E | 1–8 |
| `he-dapi` | H&E | DAPI/DNA | 1–8 |
| `he-he` | H&E | H&E | 1–6 |
| `dapi-dapi` | DAPI/DNA | DAPI/DNA | 1–6 |

Mirrored images are detected automatically, no initial alignment is needed, and a whole
slide typically registers in about a minute on a laptop.

---

## Repository layout

```text
PHARAOH/
├── install_conda_env.sh        # Conda environment, macOS and Linux
├── install_conda_env.ps1       # Conda environment, Windows
├── version_5.0_macos_linux/    # PHARAOH 5.0 with .sh launchers
├── version_5.0_windows/        # the same pipeline with Windows support and .ps1 launchers
├── examples/                   # example data downloads and example alignments
└── assets/                     # figures and demo GIFs
```

Both version directories hold the same pipeline:

```text
version_5.0_<platform>/
├── run_pharaoh.py               # one image pair, headless
├── run_pharaoh_cli.sh           # launcher for run_pharaoh.py
├── run_pharaoh_parallel.py/.sh  # many image pairs from a JSON manifest
├── run_pharaoh_failsafe.sh      # manual alignment for pairs automatic Stage 2 could not align
├── run_pharaoh_gui.sh           # interactive pipeline, one window per stage
├── scripts/
│   ├── parameters.json          # all pipeline parameters, one block per pair
│   ├── pre_process/             # convert images to pyramidal OME-TIFF
│   └── post_process/            # apply a finished registration to images and coordinates
└── outputs/                     # run directories (created on first run)
```

On Windows every `.sh` launcher has a `.ps1` counterpart with the same arguments.

---

## Installation

PHARAOH runs in its own Conda environment named `PHARAOH`.

**macOS and Linux**

```bash
bash install_conda_env.sh
conda activate PHARAOH
cd version_5.0_macos_linux
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy Bypass -File install_conda_env.ps1
cd version_5.0_windows
```

Add `-InstallMiniforge` on a machine without Conda: Miniforge is then installed for the
current user first. The `.ps1` launchers find the environment's `python.exe` themselves, so
`conda activate` is not needed; set `$env:PYTHON` to use another interpreter. On macOS and
Linux, `PYTHON=/path/to/python ./run_pharaoh_cli.sh ...` does the same.

PHARAOH 5.0 has been run on Apple M1 Pro and M2 Pro Macs (native arm64 and x86_64 under
Rosetta 2) and on an Intel Core i7 PC with Windows 10. On Windows, the command-line and batch
launchers have been tested; the GUI and the manual fail-safe have not yet been run there.

---

## Input images

- **Format.** Pyramidal OME-TIFF is read directly, one pyramid level and region at a time.
  Any other `.btf`, `.tif`, `.tiff`, `.ndpi`, `.svs`, `.jpg`, `.jpeg` or `.czi` input is
  converted to a pyramidal OME-TIFF before the run starts, written next to the source, and
  reused by later runs (see [Convert an image to OME-TIFF](#convert-an-image-to-ome-tiff)).
- **Pixel size.** PHARAOH reads the physical pixel size (`PhysicalSizeX/Y`) when the image
  has one and uses it to choose pyramid levels and match scales. Images without it still
  register; PHARAOH then tries the usual scanner pixel sizes and the image extent.
- **DAPI channel.** In a multi-channel image, the nuclear channel is found from the OME
  channel names (DAPI, Hoechst, DNA, …), falling back to channel 0 with a warning. Choose it
  yourself with `--moving-channel C` or `--fixed-channel C` (zero-based).
- **Xenium.** The DAPI image is `morphology_focus.ome.tif` or
  `morphology_focus/morphology_focus_0000.ome.tif`, depending on the Xenium release. Use the
  raw, unscaled H&E image.
- **Orientation.** The moving image is tried as given and flipped top–bottom and left–right,
  so a mirrored H&E needs no preparation.

---

## Register one image pair

From the version directory:

```bash
# DAPI onto H&E (the default pair)
./run_pharaoh_cli.sh --moving /path/to/dapi.ome.tif --fixed /path/to/he.ome.tif

# the other three pairs
./run_pharaoh_cli.sh --pair he-dapi   --moving /path/to/he.ome.tif   --fixed /path/to/dapi.ome.tif
./run_pharaoh_cli.sh --pair he-he     --moving /path/to/he_a.ome.tif --fixed /path/to/he_b.ome.tif
./run_pharaoh_cli.sh --pair dapi-dapi --moving /path/to/dapi_a.ome.tif --fixed /path/to/dapi_b.ome.tif
```

On Windows: `.\run_pharaoh_cli.ps1 --moving C:\data\dapi.ome.tif --fixed C:\data\he.ome.tif`.

Each run writes to a new `outputs/runs_<timestamp>/`, or to `--run-dir DIR`. Both
`--flag value` and `--flag=value` work. Useful options:

| option | effect |
| --- | --- |
| `--run-dir DIR` | write to (or resume) this run directory |
| `--stop-after N` | stop after stage N (1–8) |
| `--force` | recompute every stage instead of resuming |
| `--monitor` | record memory and time per stage (see [Monitoring](#monitoring-memory-and-time)) |
| `--qc-images [GROUPS]` | write quality-control images (see [Quality-control images](#quality-control-images)) |
| `--stage1-moving-level N`, `--stage1-fixed-level N` | use this pyramid level for the global alignment |
| `--moving-channel C`, `--fixed-channel C` | use this DAPI channel |
| `--moving-threshold T`, `--fixed-threshold T` | Stage-1 mask threshold (default 240 for H&E, data-type aware for DAPI) |

`python run_pharaoh.py --help` lists every option.

**Resuming.** Run the same command with the same `--run-dir` and finished stages are
skipped. A run directory is never silently reused for different inputs: if the images,
`--pair` or the pair's parameter block changed, the run is refused until you pass `--force`,
which clears each stage's earlier products before the stage reruns.

### What the stages do

| stage | what happens |
| --- | --- |
| 1 | Read one pyramid level of each image (short side 1,024–2,048 px, preferably 3–9 µm/px) and build tissue masks |
| 2 | Global alignment: ORB, then SIFT features, over reflections and matching scales; the first fit that passes every quality gate is kept |
| 3 | Sample 100–200 tiles across the moving tissue and read the matching fixed regions at about 0.43 µm/px |
| 4 | Segment nuclei in each tile pair and find matched nuclei and density anchors |
| 5 | Fit the transformation to the anchors (affine or homography, chosen by cross-validation over tiles), with an optional local correction |
| 6 | Write the transformation and the final overlay |
| 7 | Mixed DAPI/H&E pairs: pair all detected nuclei one to one under the current transformation |
| 8 | Mixed DAPI/H&E pairs: refit on the dense nucleus pairs; kept only if held-out errors improve |

### Outputs

The transformation is named by the two sides of the pair:

| pair | transformation file |
| --- | --- |
| `dapi-he` | `dapi_to_he_homography_level0.json` / `.csv` |
| `he-dapi` | `he_to_dapi_homography_level0.json` / `.csv` |
| `he-he` | `hefg_to_hebg_homography_level0.json` / `.csv` |
| `dapi-dapi` | `dapifg_to_dapibg_homography_level0.json` / `.csv` |

It maps **moving level-0 pixel coordinates to fixed level-0 pixel coordinates**. The `.csv`
is the 3 × 3 matrix; the `.json` is the complete result, including a local correction field
when one was accepted, and is what the post-processing tools read. Other files in the run
directory:

| file | content |
| --- | --- |
| `6_overlay_final_L*.png` | the moving image warped onto the fixed image (always written) |
| `initial_alignment.json`, `2_initial_overlay_original.png` | the Stage-2 global alignment |
| `images_info.json` | the input paths, pair, pyramid levels and orientation |
| `run_parameters.json` | every parameter the run used |
| `pipeline_times.json` | stage timings and pipeline events |
| `dense_nucleus_pairs/` | Stage-7 nucleus pairs (mixed DAPI/H&E pairs) |
| `tiles/`, `nuclei_patches/` | Stage-3 tiles and Stage-4 nucleus correspondences |

---

## Register many image pairs

`run_pharaoh_parallel.sh` (Windows: `run_pharaoh_parallel.ps1`) runs the image pairs listed in
a JSON manifest, several at once, each in its own run directory with its own log.

```json
{
  "pipeline_args": [],
  "datasets": [
    {"name": "sample_a", "pair": "dapi-he",
     "moving": "/path/to/a/morphology_focus.ome.tif", "fixed": "/path/to/a/he.ome.tif"},
    {"name": "sample_b",
     "moving": "/path/to/b/morphology_focus.ome.tif", "fixed": "/path/to/b/he.ome.tif",
     "pipeline_args": ["--moving-channel", "0"]}
  ]
}
```

`name` must be unique, `pair` defaults to `dapi-he`, and paths may be absolute or relative to
the manifest. `pipeline_args` (for every dataset, or for one) are passed to `run_pharaoh.py`.

```bash
./run_pharaoh_parallel.sh my_datasets.json --dry-run            # check inputs, run nothing
./run_pharaoh_parallel.sh my_datasets.json --jobs 4             # four pipelines at once
./run_pharaoh_parallel.sh my_datasets.json --jobs 8 --threads-per-dataset 2
```

The batch writes `outputs/parallel_runs_<timestamp>/` (or `--output-root DIR`) with one
`runs_<name>/` and one `<name>.log` per dataset, plus:

- `parallel_summary.json`: each dataset's command, exit code, timing and memory;
- `batch_timeline.html`: one row per dataset over the batch's wall clock, colored by stage,
  with time spent waiting for memory hatched;
- `manual_alignment_queue.json`: the pairs Stage 2 could not align (see below).

A batch waits for every dataset even if one fails, then exits nonzero. Rerun the same command
with the same `--output-root` to resume unfinished datasets.

### Memory-aware scheduling

Stage 3, which reads a whole pyramid level of both images, and Stage 4, nucleus segmentation,
need the most memory. In a batch, a dataset enters either stage only when the machine has room
for it: the estimated peak of that stage must fit in the available memory minus a reserve
(4 GiB), and total CPU use must be below a ceiling (80%, raised slightly as the queue grows).
Waiting datasets are admitted first-in, first-out. The Stage-3 estimate comes from the pyramid
levels the dataset will read; the Stage-4 estimate is 5 GiB.

| option | effect |
| --- | --- |
| `--jobs N` | pipelines in flight (default 4) |
| `--stage3-jobs N`, `--stage4-jobs N` | hard maximum datasets in Stage 3 or Stage 4 at once (default 4); the memory check usually admits fewer |
| `--stage3-memory-per-job-gib G`, `--stage4-memory-per-job-gib G` | override the peak-memory estimate |
| `--stage4-memory-reserve-gib G`, `--stage4-cpu-threshold P` | reserve and CPU ceiling (shared by both stages) |
| `--stage3-fixed`, `--stage4-fixed` | ignore memory and CPU; enforce only the count |
| `--stage3-jobs 0`, `--stage4-jobs 0` | no admission control (only for machines with ample memory) |

Admission changes only when a stage runs, never its inputs or results.

---

## When automatic alignment fails

If no reflection, scale or detector passes the Stage-2 quality gates, PHARAOH first retries at
other pyramid levels: one round tries one level coarser and then one level finer
(`--stage2-relevel-rounds N`, default 1). If that also fails, the pair is handed to a
**manual alignment window**:

- a **single run** opens the window at once and continues with Stage 3 as soon as you save;
- a **batch** lists the pair in `manual_alignment_queue.json` and, by default, opens the
  window for each queued pair after all other datasets have finished
  (`--failsafe-mode end`); `instant` opens each window as soon as its Stage 2 fails, and `off`
  only writes the queue.

In the window, first set the moving image's orientation (90° rotations and flips), then place
it on the fixed image (see [Stage 2 in the GUI](#use-the-gui)) and click **Save Alignment**.
Stages 3–8 then run as usual. An unattended batch should use `--failsafe-mode off`, and the
queue can be worked through later:

```bash
./run_pharaoh_failsafe.sh outputs/parallel_runs_<timestamp>                 # the whole queue
./run_pharaoh_failsafe.sh outputs/parallel_runs_<timestamp> --name sample_a # one dataset
./run_pharaoh_failsafe.sh outputs/parallel_runs_<timestamp> --dry-run       # show the plan only
./run_pharaoh_failsafe.sh --run-dir RUN_DIR --pair dapi-he --moving D --fixed H   # one run, no queue
```

Every queued pair is aligned first; their pipelines then finish one at a time
(`--resume-jobs N` for more). Results are summarized in
`fail_safe_manual_alignment_summary.json` next to the queue.

| option | relevel at other levels | manual alignment |
| --- | --- | --- |
| *(default)* | 1 round | yes |
| `--stage2-relevel-rounds N` | N rounds | yes |
| `--stage2-nomanual` | as set | no |
| `--no-failsafe` (single run) | as set | no window; the run fails and can be aligned later with `run_pharaoh_failsafe.sh` |
| `--stage2-nofallback` | no | no |

---

## Use the GUI

```bash
./run_pharaoh_gui.sh
```

Click **New RUN_DIR** to start a run or **Choose RUN_DIR…** to continue one, then run the stages
in order:

1. **Stage 1 – Select images.** Choose the moving and the fixed image, each as DAPI or H&E,
   rotate or flip the moving image to match the fixed one, adjust the display thresholds and
   save.
2. **Stage 2 – Manual alignment.** The moving image is shown as an overlay on the fixed one. A
   feature-based fit is computed in the background and applied as a starting point when
   available. Drag to move, drag a corner to scale (hold Shift to keep the aspect ratio), press
   Q or E to rotate by 5°, and click **Mode: Affine** to switch to perspective mode, where each
   corner moves on its own. **Load H (.json)** loads a saved matrix. Nothing is accepted until
   you click **Save Alignment**.
3. **Stage 3 – Tile extraction.** Sample tile centroids, optionally inspect pilot tiles, then
   extract the tiles.
4. **Stage 4 – Detect nuclei.** Segment nuclei and find matched nuclei in every tile pair.
5. **Stage 5 – Keypoint QC (optional).** Review nucleus pairs, correct centroids or drop pairs.
6. **Stage 6 – Calculate matrix and display.** Fit the transformation and show the result. For
   mixed DAPI/H&E pairs, Stages 7–8 follow automatically.

The GUI uses the same computations and `parameters.json` as the command line.

---

## Convert an image to OME-TIFF

The launchers convert non-OME-TIFF inputs for you. To choose tiling, compression or pixel size,
or to prepare inputs in advance, run the converter yourself; it writes the result next to the
source:

```bash
python scripts/pre_process/convert_image_to_ome_tif.py /path/to/slide.svs   # writes /path/to/slide.ome.tif
python scripts/pre_process/convert_image_to_ome_tif.py /path/to/image.jpg --mpp 0.25
python scripts/pre_process/convert_image_to_ome_tif.py /path/to/slide.czi --channel 0
```

The physical pixel size is copied from the source when it records one (NDPI/TIFF resolution
tags, the SVS `MPP`, OME `PhysicalSizeX/Y`, the CZI scaling); `--mpp` supplies it otherwise.
Large slides are converted in strips, so memory stays well below the image size. A multi-scene
CZI is split into one `NAME_sceneN.ome.tif` per scene (`--scene N` or `--scene all` to choose),
and a multi-channel input needs `--channel`. Other options: `--tile` (default 512),
`--compression` (`zlib` by default, lossless) and `--series`.

---

## Quality-control images

Diagnostic images are off by default; the final overlay `6_overlay_final_L*.png` is always
written. Turn them on per run:

```bash
./run_pharaoh_cli.sh --moving D --fixed H --qc-images                   # every group
./run_pharaoh_cli.sh --moving D --fixed H --qc-images stage6_overlays   # selected groups
./run_pharaoh_cli.sh --moving D --fixed H --no-qc-images                # none
```

Groups: `stage1_extras`, `stage3_debug`, `stage4b_overlays`, `stage4b_density`,
`stage4c_patches`, `stage4c_overlays`, `stage6_overlays` (alignment overlays and an
alternating GIF), `stage7_anchors` and `stage7_tree`. The same flags work for a batch, and
`PHARAOH_QC_IMAGES=<groups>` in the environment does the same for every entry point. Asking for
a group on an existing run draws it without recomputing anything.

## Monitoring memory and time

`--monitor` samples the memory of the whole process tree once a second (`--monitor-interval`
to change) and writes `monitor_memory_time.svg` (memory over time, colored by stage),
`monitor_memory_time.csv` and `monitor_summary.json` into the run directory, also when a run
fails or is interrupted. Batches monitor every dataset by default (`--no-monitor` to switch off)
and add `batch_timeline.html`; `python scripts/batch_timeline.py <output-root>` redraws it for a
finished batch.

---

## Apply the registration

The tools in `scripts/post_process/` read a finished run and apply its transformation; they
never fit a new one:

- warp the whole moving image into the fixed image's pixel grid (`warp_dapi_to_he.py`);
- map Xenium transcripts, cell centroids, cell boundaries and nucleus boundaries to H&E pixels;
- map CosMx cell-boundary polygons to H&E pixels;
- bin aligned transcripts and plot genes on the H&E image;
- check a registration visually with nucleus outlines drawn on DAPI and H&E side by side.

See [`scripts/post_process/README.md`](version_5.0_macos_linux/scripts/post_process/README.md)
for the commands.

---

## Parameters

All parameters are in `scripts/parameters.json`, with one block per pair (`dapi-he`, `he-dapi`,
`he-he`, `dapi-dapi`) and, inside each, one block per stage (`stage2`, `stage3`, `stage4a`,
`stage4b`, `stage4c`, `stage5`, and for the mixed pairs `stage7_dense_pairs` and
`stage8_refinement`). The top-level `qc_images` block sets the default quality-control groups.
The defaults were used for every result in the manuscript except where its Methods state
otherwise. A run records the parameters it used in `run_parameters.json`.

---

## Example data

```bash
cd examples
bash download_example_dapi_he_data.sh    # Xenium Human Colon Cancer P1 -> xenium_human_CRC_P1/
bash download_example_orb_fail_data.sh   # Xenium Human Colon Cancer P5 -> xenium_human_CRC_P5/
bash download_example_he_he_data.sh      # Visium and Xenium H&E of Colon Cancer P2 -> he_he_human_CRC_P2/
cd ../version_5.0_macos_linux
./run_pharaoh_cli.sh --moving ../examples/xenium_human_CRC_P1/morphology_focus.ome.tif \
                     --fixed  ../examples/xenium_human_CRC_P1/he.ome.tif
```

For the H&E pair, use `--pair he-he` with the Visium image as moving and the Xenium H&E as
fixed; the Visium `.btf` is converted to OME-TIFF automatically. `examples/alignment_examples/`
holds, for 16 Xenium datasets, an initial alignment (`manual_initial_alignment.json`; load it in
Stage 2 of the GUI with **Load H (.json)**) and a final transformation
(`dapi_to_he_homography_level0.json`). Other datasets can
be downloaded from the sources listed in the manuscript's Data Availability section.

---

## License

See [LICENSE](LICENSE).
