#!/usr/bin/env bash
# ============================================================
# run_pharaoh_cli.sh - headless (no-GUI) pipeline, ORB auto-alignment
#
#   ./run_pharaoh_cli.sh --dapi="dapi_img_path" --he="he_img_path"
#   ./run_pharaoh_cli.sh --hefg="hefg_img_path"   --he="he_img_path"
#
# Runs Stages 1-6 automatically (ORB replaces the manual Stage-2 overlay).
# Both --flag=value and --flag value work. Extra flags (e.g. --run-dir,
# --force, --stop-after) are forwarded to run_pharaoh.py.
# Set PYTHON=... to choose a specific interpreter.
#
# For the interactive GUI version instead, use ./run_pharaoh_gui.sh
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$HERE/run_pharaoh.py" "$@"
