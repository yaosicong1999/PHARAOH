# ============================================================
# run_pharaoh_parallel.ps1 - Windows counterpart of run_pharaoh_parallel.sh: run the
# datasets of a JSON manifest concurrently in the PHARAOH conda environment.
#
#   .\run_pharaoh_parallel.ps1 my_datasets.json --dry-run
#   .\run_pharaoh_parallel.ps1 my_datasets.json --jobs 4
#
# Everything after the script name goes to run_pharaoh_parallel.py unchanged; see
# run_pharaoh_parallel.sh or `--help` for the manifest format and options.
# Finds the PHARAOH environment's python.exe itself (see _pharaoh_python.ps1): no
# `conda activate` and no `conda run`. Every dataset's pipeline runs with that same
# interpreter (the driver passes sys.executable on). Set $env:PYTHON to use another one,
# or $env:PHARAOH_CONDA_ENV for an environment with a different name.
# ============================================================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_pharaoh_python.ps1")
$python = Resolve-PharaohPython
$env:PYTHONUTF8 = "1"
Write-Host "[parallel] python: $python"
& $python (Join-Path $PSScriptRoot "run_pharaoh_parallel.py") @args
exit $LASTEXITCODE
