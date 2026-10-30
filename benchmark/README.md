# PHARAOH benchmark test

This folder holds the code of the Xenium DAPI–H&E benchmark in the PHARAOH manuscript, set up to run
end to end on one public test dataset. The method wrappers, the configurations, the run protocol and
the scoring are the ones that produced the published results; the manuscript applied them to 25
datasets, and this folder applies them to one.

**Test dataset.** 10x Genomics Xenium In Situ, human colorectal cancer Sample P1 (FFPE, Xenium v1
Human Colon panel with add-on): the DAPI morphology image (moving), the post-Xenium H&E image (fixed)
and the 8 DAPI–H&E landmark pairs annotated by 10x Genomics. Accuracy is the target registration
error (TRE) at these landmarks, in H&E micrometres.

**Methods.** PHARAOH 5.0 (this repository) in its single default configuration, and 29
configurations of three other methods:

| Method | Package | Configurations | What varies |
|---|---|---|---|
| PHARAOH | `../version_5.0_macos_linux` | 1 (`default`) | — |
| VoltRon | VoltRon 0.2.6 (R) | 5 | input cap 1k, 2k, 4k, 8k, 16k pixels (longest side) |
| RegWSI | DeeperHistReg 1.0.1 | 15 | preset initial, nonrigid, hires × cap 1k–16k |
| VALIS | valis-wsi 1.2.0 | 9 | rigid, non-rigid, micro-registration × cap 1k, 2k, 4k |

VALIS is run twice: with the DAPI image decoded by tifffile (`valis_tifffile`, the main benchmark)
and read by VALIS's own Bio-Formats reader (`valis_bioformats`, an ablation).

## Contents

```text
benchmark/
├── download_test_data.sh   1. downloads the test dataset to data/human_crc_p1/
├── run_pharaoh.sh          2. per-method runners: every configuration of one method
├── run_voltron.sh
├── run_regwsi.sh
├── run_valis.sh               (VALIS with both DAPI readers)
├── run_all.sh              3. overall runner: all methods in order
├── eval_all.sh             4. overall evaluator: scores every method, one summary table, then plot.sh
├── plot.sh                 5. plotter: per-dataset figures
├── settings.sh                settings shared by the runners (series, configs, resource limits)
├── configs/
│   ├── environment.json       interpreters of each method and output folders  <- edit first
│   ├── methods/<method>.json  every benchmark configuration of a method
│   ├── quick/<method>.json    two configurations per method, for a fast check
│   ├── resources.json         default resource limits
│   └── eval.json
├── manifests/keypoints.json   the test dataset (paths relative to this folder)
├── methods/<method>/wrapper.py   one wrapper per method: reads the images, registers, writes the transform
├── methods/valis_mirror/      VALIS with an option for mirrored image pairs (not part of the runs above; see
│                              "VALIS and mirrored image pairs")
├── run.py                     runner: plans and launches runs, guards memory and swap, records timing
├── prepare_valis_inputs.py    re-encodes the DAPI image for VALIS once (time charged to each VALIS run)
├── eval/
│   ├── eval_<method>.sh       per-method evaluators
│   ├── eval_summary.sh        one table across methods
│   ├── keypoints/<method>.py  maps the landmarks with each method's own transform and computes TRE
│   ├── common/                shared scoring code (point mapping, VALIS replay, RegWSI field inversion)
│   ├── summarize.py
│   └── visualization_dataset_keypoints.py
└── bench/                     shared library of the runner and evaluators
```

## Requirements

- macOS (Apple silicon); Linux has not been tested. The benchmark ran on a 32 GiB workstation: runs are limited to
  28 GiB of resident memory, and a run above the limit is recorded as a failure, as in the benchmark.
- About 11.5 GB of disk for the data (about 22 GB is downloaded; the Xenium output bundle is deleted
  after the DAPI image is extracted) and a few GB for the runs.
- `curl` and `unzip`.
- Four environments. Their interpreters are named in `configs/environment.json`; edit the paths there.

| Environment | Used for | Contents |
|---|---|---|
| `PHARAOH` | runner, PHARAOH, VoltRon wrapper, evaluation of PHARAOH and VoltRon, plots | Python 3.10; `../install_conda_env.sh` |
| `voltron` | VoltRon (`rscript`) | R 4.6 with VoltRon 0.2.6 |
| `regwsi` | RegWSI | Python 3.10, `pip install deeperhistreg==1.0.1`, PyTorch 2.13 |
| `valis` | VALIS, evaluation of VALIS and RegWSI, `methods/valis_mirror/` | Python 3.10, `pip install valis-wsi==1.2.0` (with libvips; Java for Bio-Formats), numba, zarr |

The evaluation of VALIS replays VALIS's own transformation code, so it uses the same valis-wsi version
as the runs.

## Run

```bash
./download_test_data.sh
```

Then edit `configs/environment.json` and run every method:

```bash
./run_all.sh
```

or one method at a time with `./run_pharaoh.sh`, `./run_voltron.sh`, `./run_regwsi.sh` and
`./run_valis.sh`. For a fast check of the installation, use two configurations per method:

```bash
CONFIGS=configs/quick ./run_all.sh
```

Each configuration is run once after one discarded warm-up run. Re-running a runner skips completed
runs and retries failed ones, so an interrupted run resumes. Runs are written to
`runs/test/keypoints/<method>/<condition>/human_crc_p1/`, each with its transform, `run_record.json`
(outcome, method time, wall time, peak memory) and `run_metadata.json` (package versions, inputs and
per-step timing). Settings such as the output series and the resource limits are in `settings.sh`.
Before each run, the runner waits until the machine's swap use is back below 5 GiB, so that no run
starts on a machine still paging out the previous one. On a machine that already uses more swap, it
waits up to 30 min and then pauses; set `SWAP_SETTLE=off` to skip the wait, e.g.
`SWAP_SETTLE=off CONFIGS=configs/quick ./run_all.sh`.

Score the runs and draw the figures:

```bash
./eval_all.sh
```

or per method, e.g. `eval/eval_valis_tifffile.sh --mode keypoints --series test`, and then
`eval/eval_summary.sh --mode keypoints --series test` and `./plot.sh`.

PHARAOH runs on a transcript dataset can also be scored, with
`eval/eval_pharaoh.sh --mode matrix` (every transcript's distance between PHARAOH's and 10x's
mapping). `../minitest/` uses this for a Xenium human heart section. It points the runner and
the scoring at its own manifests, environment and limits through `BENCH_MANIFESTS`,
`BENCH_ENVIRONMENT` and `BENCH_RESOURCES`.

## Outputs

| Folder | Contents |
|---|---|
| `results/test/keypoints/<method>/` | `runs.csv` (one row per run) and `points/human_crc_p1/<condition>.csv` (TRE of every landmark) |
| `results/summary/test/keypoints/` | `runs.csv`, `conditions.csv`, `best.csv` (each method's best configuration), `summary.json` |
| `results/visualization/test/keypoints/human_crc_p1/` | TRE box plot of each method's best configuration, TRE across input resolution, TRE against method time and against peak memory (PNG and editable PDF) |

In the benchmark, the median TRE on this dataset was 0.51 µm for PHARAOH, and for each other method's
best configuration 1.64 µm for VoltRon (16k), 2.08 µm for RegWSI (hires, 16k), 1.74 µm for VALIS
(micro-registration, 4k; tifffile) and 1.63 µm for VALIS (micro-registration, 4k; Bio-Formats)
(Supplementary Table 2). PHARAOH's transformations are deterministic, so its result should match
exactly on the same platform; on another platform it may differ slightly through the OpenCV build.
VALIS's result can occasionally vary between identical runs: valis-wsi 1.2.0 builds the H&E tissue mask
with multi-threaded k-means, and for an image close to the threshold between two colour counts the
thread scheduling can flip that choice, which changes the rigid transform (by up to about 6 µm in the
benchmark). `OMP_NUM_THREADS=1` makes it repeatable but slows VALIS down; the benchmark ran VALIS
unchanged.

## How runs are measured

- **Inputs.** Every method reads the same pyramidal OME-TIFF images. Input preparation that a method
  needs (VALIS's DAPI re-encode, RegWSI's 8-bit rendering) is charged to that method's time.
- **Time and memory.** Method time is the method's own time, input preparation included; peak memory
  is the resident memory of the run's process tree, plus GPU memory for RegWSI.
- **Scoring.** Each landmark's DAPI position is mapped into H&E with the method's own transform:
  PHARAOH's and VoltRon's matrices, RegWSI's displacement field (inverted numerically) and VALIS's
  transformation replayed with valis-wsi. TRE is the distance to the annotated H&E landmark, in H&E
  micrometres. A configuration counts as a failure if any landmark is more than 12 µm off or the run
  produced no transform.

## VALIS and mirrored image pairs

An H&E section can be mounted face down, so that it is the mirror image of the DAPI image.
PHARAOH tests the reflection itself. VALIS does not by default (`check_for_reflections=False`), and the
benchmark ran it that way. `methods/valis_mirror/` adds a way to handle such a pair. It is not part of
`run_all.sh` or `eval_all.sh`, and the benchmark's results above do not use it.

`methods/valis_mirror/wrapper.py` takes every argument of `methods/valis/wrapper.py`, plus `--mirror`:

- `--mirror off` (default): runs `methods/valis/wrapper.py` with the same arguments; same output.
- `--mirror double --he-flipped H_FLIPPED`: the reflection is chosen outside VALIS. VALIS runs twice,
  unchanged: onto the H&E as given (`RUN/as_given/`) and onto its left-right mirror image
  (`RUN/flipped/`). One flip is enough, because a top-bottom flip is a left-right flip plus a
  180° rotation, which VALIS finds itself. The run is kept whose warped DAPI correlates better with the
  H&E's hematoxylin over the tissue (Pearson r at a coarse pyramid level; no landmarks are used). A
  sub-run that fails counts as no match. `RUN/double.json` holds the choice and both correlations.
- `--mirror valis`: VALIS's own reflection search, `check_for_reflections=True`. As shipped in
  valis-wsi 1.2.0 it stops with an error (a float64/float32 mismatch in LightGlue, then a failed fit of a
  reflected candidate). The wrapper adds a float32 cast and counts a failed fit as no match, so that it
  runs. Each candidate's error and VALIS's choice are recorded in `RUN/run_metadata.json`
  (`reflection_check`). In our tests it was not reliable. For some pairs that were not mirrored it
  chose a flip, because a reflected candidate with only a few matches can be fitted almost exactly. When
  it chose the right flip for a mirrored pair, the transform it returned did not include the flip (see
  the result below).

`methods/valis_mirror/flip_he.py` writes the mirror image of an OME-TIFF pyramid, losslessly and with
the same levels and pixel size:

```bash
VALIS=~/miniconda3/envs/valis/bin/python          # the valis interpreter of configs/environment.json
D=data/human_crc_p1
HE=$D/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_he_image.ome.tif
$VALIS methods/valis_mirror/flip_he.py $HE $D/he_image_flipx.ome.tif
```

Register the DAPI image to the H&E image, choosing the reflection automatically (the other arguments
are those of `methods/valis/wrapper.py`; `--stage nonrigid` works the same way):

```bash
DAPI=$D/morphology_focus/morphology_focus_0000.ome.tif
$VALIS methods/valis_mirror/wrapper.py --dapi $DAPI --he $HE --dapi-mpp 0.2125 --he-mpp 0.2737768141407522 \
    --stage rigid --caps-scale 1 --dapi-reader tifffile --device auto \
    --input-cache cache/valis_inputs --cache-series mirror \
    --mirror double --he-flipped $D/he_image_flipx.ome.tif --outdir runs/mirror/as_given
```

To test it on a mirrored pair, swap the two H&E images. `--he` is now the flipped copy, and the wrapper
should keep `flipped`:

```bash
$VALIS methods/valis_mirror/wrapper.py --dapi $DAPI --he $D/he_image_flipx.ome.tif --dapi-mpp 0.2125 \
    --he-mpp 0.2737768141407522 --stage rigid --caps-scale 1 --dapi-reader tifffile --device auto \
    --input-cache cache/valis_inputs --cache-series mirror \
    --mirror double --he-flipped $HE --outdir runs/mirror/mirrored
```

The result is a contract of kind `valis_double`. To map DAPI points into the given H&E, use the kept
sub-run's point map and then, if the kept sub-run is `flipped`, the flip `he_flip_matrix` in
`double.json`. For example, the TRE at the 10x landmarks, run from `benchmark/` with the valis
interpreter:

```python
import json, sys
import numpy as np
sys.path.insert(0, ".")
from eval.common.pointmap import PointMap

def landmark_tre(run, landmarks_csv, he_mpp, he_mirrored):
    """(kept sub-run, median TRE in um) of a --mirror double run; he_mirrored: --he was the flipped copy."""
    choice = json.load(open(f"{run}/double.json"))
    flip = np.asarray(choice["he_flip_matrix"])                     # x -> W - 1 - x at level 0
    lm = np.loadtxt(landmarks_csv, delimiter=",", skiprows=1)       # DAPI x, y | H&E x, y (level 0)
    he, measured, _ = PointMap(f"{run}/{choice['chosen']}").apply(lm[:, :2])
    if choice["chosen"] == "flipped":                               # back into the given H&E
        he = (np.column_stack([he, np.ones(len(he))]) @ flip.T)[:, :2]
    truth = lm[:, 2:].copy()
    if he_mirrored:                                                 # the landmarks in the flipped H&E
        truth[:, 0] = flip[0, 2] - truth[:, 0]
    return choice["chosen"], float(np.median(np.hypot(*(he - truth).T)) * he_mpp)

csv = "data/human_crc_p1/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_he_imagealignment.csv"
print(landmark_tre("runs/mirror/as_given", csv, 0.2737768141407522, he_mirrored=False))
print(landmark_tre("runs/mirror/mirrored", csv, 0.2737768141407522, he_mirrored=True))
```

`PointMap` reads `run_record.json` (outcome `success`) in the run folder. The wrapper writes one for each
sub-run; for a run of `methods/valis/wrapper.py` or `--mirror valis` made outside `run.py`, write
`{"outcome": "success"}` there first.

Result on the test dataset (rigid, 1k caps, the commands above):

| `--he` | `--mirror` | Kept or chosen | Correlation (kept, other) | Median TRE |
|---|---|---|---|---|
| H&E as given | `double` | `as_given` | 0.276, 0.130 | 16.95 µm |
| flipped H&E | `double` | `flipped` | 0.276, 0.130 | 16.95 µm |
| flipped H&E, `--stage nonrigid` | `double` | `flipped` | 0.289, 0.150 | 6.95 µm |
| H&E as given | `valis` | no flip | | 16.95 µm |
| flipped H&E | `valis` | left-right flip, transform not mirrored | | 7241 µm |

16.95 µm and 6.95 µm are the TRE of `methods/valis/wrapper.py` on the H&E as given at the same caps
(`valis_tifffile` rigid_1k and nonrigid_1k): with `double`, the mirrored pair is registered as well as the
unmirrored one.
