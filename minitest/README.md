# Minitest: install, download, run and check PHARAOH on two public datasets

Four scripts install PHARAOH, download two public 10x Genomics Xenium datasets, register them with
PHARAOH's default configuration, and score the result exactly as the manuscript's benchmark does.
The last step reports each dataset's error next to the benchmark's; the same metrics from four
platforms are listed below.

| Dataset | Reference | Benchmark median error |
|---|---|---|
| `human_crc_p1`: human colorectal cancer P1 | 8 DAPI–H&E landmark pairs annotated by 10x Genomics | 0.51 µm (landmark TRE) |
| `human_heart`: non-diseased human heart | 10x's DAPI-to-H&E matrix, applied to all 2,452,409 transcripts | 0.35 µm (transcript distance) |

Errors are in H&E micrometers. The minitest uses `../benchmark` (its runner, PHARAOH wrapper and
scoring) and the PHARAOH release in `../version_5.0_macos_linux`, so its numbers are the
benchmark's. Everything it writes stays in this folder.

## Requirements

- macOS, with conda (Miniconda or Miniforge) and `curl`, `unzip`. Linux has not been tested.
- About 16 GB of memory: PHARAOH peaks at about 5.5 GB on CRC P1 and 3.6 GB on heart.
- About 12.5 GB of disk for the data. About 26 GB is downloaded; the Xenium output bundles are
  deleted once the DAPI image and transcripts are extracted (`KEEP_ZIP=yes` keeps them).

## Steps

Run from this folder:

```bash
./1_install.sh
./2_download_data.sh
./3_run.sh
./4_evaluate.sh
```

1. **`1_install.sh`**
   - Builds the conda environment `PHARAOH` with `../install_conda_env.sh` if it doesn't exist
     yet, then imports the packages PHARAOH and the scoring need.
   - Writes `environment.json`: the interpreter and the folders the other steps use.
   - To use another interpreter: `PHARAOH_PYTHON=/path/to/python ./1_install.sh`.
2. **`2_download_data.sh`**
   - Downloads both datasets into `data/` from 10x Genomics and checks every file against its
     published size.
   - An interrupted download resumes when you run the script again.
   - If `../benchmark/data/human_crc_p1` is already complete, CRC P1 is linked to it instead of
     downloaded again.
3. **`3_run.sh`**
   - Registers both datasets with PHARAOH's single default configuration (Stage-4 tile fraction
     0.30), one at a time, through `../benchmark/run.py`. About 1–2 minutes per dataset.
   - Runs go to `runs/minitest/<keypoints|matrix>/pharaoh/default/<dataset>/`, including
     `dapi_to_he_homography_level0.json`, PHARAOH's DAPI-to-H&E transformation at full
     resolution.
   - A dataset that already ran successfully is skipped.
4. **`4_evaluate.sh`**
   - Scores both runs with the benchmark's scoring and writes the tables to
     `results/minitest/<keypoints|matrix>/pharaoh/`.
   - Then `check.py` prints one line per dataset: the median, P95 and largest error, and whether
     the median is within 0.4 µm of the benchmark's. It exits non-zero if a dataset is further off
     or failed.

## Reading the check

```text
dataset       error (H&E um)      median    P95    max  benchmark  verdict
human_crc_p1  landmark TRE          0.51   0.96   1.00       0.51  CLOSE: median within 0.4 um of the benchmark's
human_heart   transcript vs 10x     0.35   0.39   0.46       0.35  CLOSE: median within 0.4 um of the benchmark's
```

- **CLOSE.** The median is within 0.4 µm of the benchmark's. Compare all three numbers with the
  table below; they should be roughly the same as on the platform nearest yours.
- **FAR.** The median is further off.
- **FAIL.** The run failed or was not scored.

**The same metrics on four platforms**, PHARAOH 5.0 with the same Python and package versions
(H&E µm; median / P95 / largest error):

| Platform | CRC P1, 8 landmarks | Heart, 2,452,409 transcripts |
|---|---|---|
| Apple M2 Pro, macOS 15.8.2, x86_64 under Rosetta 2 (the benchmark) | 0.51 / 0.96 / 1.00 | 0.35 / 0.39 / 0.46 |
| Apple M2 Pro, macOS 15.8.2, arm64 | 0.51 / 0.96 / 1.00 | 0.40 / 0.52 / 0.70 |
| Apple M1 Pro, macOS 26.4.1, arm64 | 0.51 / 0.96 / 1.00 | 0.40 / 0.52 / 0.70 |
| Intel Core i7-7700, Windows 10, x64 | 0.56 / 1.01 / 1.03 | 0.33 / 0.42 / 0.48 |

**Why platforms differ slightly.** On the same platform, PHARAOH's results are identical from
run to run. Across platforms, the processor, chip, operating-system version and system math
libraries change the last bits of floating-point results, and thresholds and random sampling in
the pipeline occasionally amplify such differences. Across four machines, a dataset's median
moved by at most 0.38 µm (Supplementary Fig. 10), which is where the 0.4 µm tolerance comes from.
Intel Macs have not been checked, and Linux has not been tested.

## Files

| File | Contents |
|---|---|
| `settings.sh` | Shared paths; points `../benchmark` at this folder's manifests, environment and limits |
| `manifests/keypoints.json`, `manifests/matrix.json` | The two datasets, in the benchmark's manifest format |
| `resources.json` | Run limits: 28 GiB memory and 90 minutes per run, no waiting for swap to settle |
| `check.py` | The summary of step 4: this run's errors next to the benchmark's |

Logs of the scoring step go to `../benchmark/logs/`. To start over, delete `runs/`, `results/`
and `cache/`. `data/` can stay.
