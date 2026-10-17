# PHARAOH 5.0 for Windows

This directory holds the complete PHARAOH 5.0 pipeline with Windows support. It behaves exactly
like `version_5.0_macos_linux/` and takes the same arguments; on Windows use the `.ps1`
launchers. The installation, all options and the input and output formats are described in the
[repository README](../README.md).

## Quick start (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File ..\install_conda_env.ps1   # once

# register a DAPI image onto an H&E image
.\run_pharaoh_cli.ps1 --moving C:\data\dapi.ome.tif --fixed C:\data\he.ome.tif

# other pairs: he-dapi, he-he, dapi-dapi
.\run_pharaoh_cli.ps1 --pair he-he --moving C:\data\he_a.ome.tif --fixed C:\data\he_b.ome.tif
```

The launchers find the `PHARAOH` environment's `python.exe` themselves, so `conda activate` is
not needed. Set `$env:PYTHON` to use another interpreter, or `$env:PHARAOH_CONDA_ENV` for an
environment with a different name.

The result is in `outputs\runs_<timestamp>\`: the transformation
`<moving>_to_<fixed>_homography_level0.json` (and `.csv`), which maps moving level-0 pixels to
fixed level-0 pixels, and the overlay `6_overlay_final_L*.png`.

## Launchers

| launcher | use |
| --- | --- |
| `run_pharaoh_cli.ps1` | one image pair, headless (`python run_pharaoh.py --help` for every option) |
| `run_pharaoh_parallel.ps1` | many image pairs from a JSON manifest, several at once |
| `run_pharaoh_failsafe.ps1` | manual alignment for pairs that automatic Stage 2 could not align, then finish them |
| `run_pharaoh_gui.ps1` | the interactive pipeline, one window per stage |

The command-line and batch launchers have been tested on Windows 10; the GUI and the manual
fail-safe have not yet been run on Windows. The `.sh` launchers are kept for macOS and Linux.

Helper tools for preparing images and applying a finished registration are described in
[`scripts/README.md`](scripts/README.md).
