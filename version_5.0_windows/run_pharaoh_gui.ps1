# ============================================================
# run_pharaoh_gui.ps1 - Windows counterpart of run_pharaoh_gui.sh: the interactive
# pipeline GUI. Finds the PHARAOH environment's python.exe itself (see
# _pharaoh_python.ps1); set $env:PYTHON to use another interpreter.
# ============================================================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_pharaoh_python.ps1")
$python = Resolve-PharaohPython
$env:PYTHONUTF8 = "1"
& $python (Join-Path $PSScriptRoot "scripts\gui_pipeline.py") @args
exit $LASTEXITCODE
