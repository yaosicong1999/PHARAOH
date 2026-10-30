#!/usr/bin/env bash
# Step 1: the PHARAOH environment, checked, and recorded for the other steps.
#
#   ./1_install.sh                         # build the conda env PHARAOH if missing, then check it
#   PHARAOH_PYTHON=/path/to/python ./1_install.sh   # use an existing PHARAOH interpreter instead
#
# The environment is the one the repository's README installs (../install_conda_env.sh: conda env
# "PHARAOH", Python 3.10 and the pinned packages). This script runs that installer only when no env
# of that name exists, then imports what PHARAOH and the scoring need and writes environment.json:
# the interpreter, PHARAOH's release folder (../version_5.0_macos_linux) and the minitest's own
# runs/, cache/ and results/ folders.
set -euo pipefail
source "$(dirname "$0")/settings.sh"

PY="${PHARAOH_PYTHON:-}"
if [ -z "$PY" ]; then
    command -v conda >/dev/null 2>&1 || { echo "[install] conda not found on PATH; install Miniconda or Miniforge, or set PHARAOH_PYTHON" >&2; exit 1; }
    if ! conda env list | awk '{print $1}' | grep -qx PHARAOH; then
        _log "[install] no conda env PHARAOH yet: running ../install_conda_env.sh"
        (cd "$REPO_DIR" && bash install_conda_env.sh)
    fi
    PY="$(conda run -n PHARAOH python -c 'import sys; print(sys.executable)')"
fi
[ -x "$PY" ] || { echo "[install] $PY is not an executable Python" >&2; exit 1; }
_log "[install] PHARAOH interpreter: $PY"

"$PY" - <<'EOF'
import ctypes, platform, sys
import cv2, imagecodecs, numpy, pandas, psutil, pyarrow, scipy, skimage, tifffile, zarr  # noqa: F401
translated = False
if sys.platform == "darwin":
    value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    libc = ctypes.CDLL(None)
    if libc.sysctlbyname(b"sysctl.proc_translated", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
        translated = bool(value.value)
print(f"[install] Python {sys.version.split()[0]} {platform.machine()} on {platform.system()}"
      + (" (Rosetta 2)" if translated else ""))
print(f"[install] numpy {numpy.__version__}, opencv {cv2.__version__}, tifffile {tifffile.__version__}, "
      f"imagecodecs {imagecodecs.__version__}, pyarrow {pyarrow.__version__}")
EOF

"$PY" - "$BENCH_ENVIRONMENT" "$PY" <<'EOF'
import json, sys
path, python = sys.argv[1:]
# Relative locations are relative to ../benchmark (bench/config.py resolves them there).
env = {
    "runs_root": "../minitest/runs",
    "cache_root": "../minitest/cache",
    "results_root": "../minitest/results",
    "pharaoh_root": "../version_5.0_macos_linux",
    "python": {"runner": python, "pharaoh": python, "voltron": python, "regwsi": python, "valis": python},
    "rscript": "Rscript",
    "eval_python": {"pharaoh": python, "voltron": python, "regwsi": python, "valis": python},
}
open(path, "w").write(json.dumps(env, indent=2) + "\n")
print(f"[install] wrote {path}")
EOF
_log "[install] done; next: ./2_download_data.sh"
