# ============================================================
# run_pharaoh_failsafe.ps1 - Windows counterpart of run_pharaoh_failsafe.sh: manual
# alignment for the datasets automatic Stage 2 could not align, then finish them.
#
#   .\run_pharaoh_failsafe.ps1 outputs\parallel_runs_...\manual_alignment_queue.json
#   .\run_pharaoh_failsafe.ps1 --run-dir <dir> --moving <img> --fixed <img>
#
# Finds the PHARAOH environment's python.exe itself (see _pharaoh_python.ps1); set
# $env:PYTHON to use another interpreter.
# ============================================================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_pharaoh_python.ps1")
$python = Resolve-PharaohPython
$env:PYTHONUTF8 = "1"
& $python (Join-Path $PSScriptRoot "scripts\gui_fail_safe\fail_safe_gui_manual_align.py") @args
exit $LASTEXITCODE
