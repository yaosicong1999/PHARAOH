# PHARAOH 5.0 for macOS and Linux

This directory holds the complete PHARAOH 5.0 pipeline with its `.sh` launchers. The
installation, all options and the input and output formats are described in the
[repository README](../README.md).

## Quick start

Install the environment once, from this directory:

```bash
bash ../install_conda_env.sh
```

Then register a DAPI image onto an H&E image:

```bash
conda activate PHARAOH
./run_pharaoh_cli.sh --moving /path/to/dapi.ome.tif --fixed /path/to/he.ome.tif
```

For the other pairs, add `--pair he-dapi`, `--pair he-he` or `--pair dapi-dapi`, for example:

```bash
./run_pharaoh_cli.sh --pair he-he --moving /path/to/he_a.ome.tif --fixed /path/to/he_b.ome.tif
```

The result is in `outputs/runs_<timestamp>/`: the transformation
`<moving>_to_<fixed>_homography_level0.json` (and `.csv`), which maps moving level-0 pixels to
fixed level-0 pixels, and the overlay `6_overlay_final_L*.png`.

## Launchers

| launcher | use |
| --- | --- |
| `run_pharaoh_cli.sh` | one image pair, headless (`python run_pharaoh.py --help` for every option) |
| `run_pharaoh_parallel.sh` | many image pairs from a JSON manifest, several at once |
| `run_pharaoh_failsafe.sh` | manual alignment for pairs that automatic Stage 2 could not align, then finish them |
| `run_pharaoh_gui.sh` | the interactive pipeline, one window per stage |

Helper tools for preparing images and applying a finished registration are described in
[`scripts/README.md`](scripts/README.md).
