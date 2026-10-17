# ============================================================
# run_pharaoh_cli.ps1 - Windows counterpart of run_pharaoh_cli.sh; same arguments.
#
#   .\run_pharaoh_cli.ps1 --moving dapi.ome.tif --fixed he.ome.tif
#   .\run_pharaoh_cli.ps1 --pair he-he --moving a.ome.tif --fixed b.ome.tif
#
# Everything after the script name goes to run_pharaoh.py unchanged.
# Finds the PHARAOH environment's python.exe itself (see _pharaoh_python.ps1); no
# `conda activate` needed. Set $env:PYTHON to use another interpreter.
# ============================================================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_pharaoh_python.ps1")
$python = Resolve-PharaohPython
$env:PYTHONUTF8 = "1"
& $python (Join-Path $PSScriptRoot "run_pharaoh.py") @args
exit $LASTEXITCODE
