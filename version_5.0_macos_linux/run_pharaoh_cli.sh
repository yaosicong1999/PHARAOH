#!/usr/bin/env bash
# ============================================================
# run_pharaoh_cli.sh - headless (no-GUI) pipeline, ORB -> SIFT auto-alignment
#
#   ./run_pharaoh_cli.sh --moving=dapi.ome.tif --fixed=he.ome.tif          # dapi-he
#   ./run_pharaoh_cli.sh --pair he-he --moving=a.ome.tif --fixed=b.ome.tif
#   ./run_pharaoh_cli.sh --pair dapi-dapi --moving=a.ome.tif --fixed=b.ome.tif
#   ./run_pharaoh_cli.sh --pair he-dapi --moving=he.ome.tif --fixed=dapi.ome.tif
#
# A run is named by what it registers onto what: --pair is one of dapi-he
# (the default), he-he, he-dapi, dapi-dapi. All four are runnable in the CLI.
# Runs Stages 1-8 for the mixed pairs (dapi-he, he-dapi) and Stages 1-6 for he-he and dapi-dapi.
# Both --flag=value and --flag value work. Extra flags (e.g. --run-dir,
# --force, --stop-after) are forwarded to run_pharaoh.py -- this script is a
# pass-through, so anything run_pharaoh.py accepts works here unchanged.
#
# Quality-control images are off by default and --qc-images turns them on for
# one run, without editing parameters.json:
#
#   ./run_pharaoh_cli.sh --moving=... --fixed=... --qc-images=stage4b_overlays
#   ./run_pharaoh_cli.sh --moving=... --fixed=... --qc-images   # every group
#
# Groups: stage1_extras, stage3_debug, stage4b_overlays, stage4b_density,
# stage4c_patches, stage4c_overlays, stage6_overlays, stage7_anchors,
# stage7_tree. PHARAOH_QC_IMAGES=<groups> in the
# environment does the same thing and reaches every entry point. A group turned on
# for a run that already exists is drawn on resume by the stage's QC-only mode;
# nothing the later stages read is rewritten. Resuming a run directory with other
# inputs (moving, fixed, --pair) or a changed parameters.json block is refused unless
# --force starts it over; --force clears each stage's earlier products before the
# stage reruns, so nothing from the previous run is read as this one's.
# Set PYTHON=... to choose a specific interpreter.
# Optional process-tree memory/stage-time monitoring (default: off):
#   bash run_pharaoh_cli.sh --moving=... --fixed=... --monitor
#   bash run_pharaoh_cli.sh --moving=... --fixed=... --monitor --monitor-interval 0.5
# Writes monitor_memory_time.svg, monitor_memory_time.csv and monitor_summary.json
# in the run directory, including partial reports on failure or Ctrl+C.
#
# For the interactive GUI version instead, use ./run_pharaoh_gui.sh
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$HERE/run_pharaoh.py" "$@"
