#!/usr/bin/env bash
# ============================================================
# run_pharaoh_failsafe.sh - manual alignment for the datasets automatic
# Stage 2 could not align, then finish their pipelines
#
#   ./run_pharaoh_failsafe.sh outputs/parallel_runs_.../manual_alignment_queue.json
#   ./run_pharaoh_failsafe.sh outputs/parallel_runs_...        # finds the queue
#   ./run_pharaoh_failsafe.sh <queue> --name human_crc_p5 --no-resume
#   ./run_pharaoh_failsafe.sh <queue> --pharaoh-arg=--qc-images=stage4b_overlays
#   ./run_pharaoh_failsafe.sh --run-dir <dir> --moving <img> --fixed <img>   # one dataset
#
# --pharaoh-arg is repeatable and appends its value to every resumed
# run_pharaoh.py command, which is how a flag like --qc-images reaches the
# Stage 3-8 pipelines this script finishes.
#
# Opens scripts/gui_fail_safe/. Per queued dataset: a Stage-1
# window with both images already loaded (orientation and thresholds only),
# then the ordinary Stage-2 manual alignment window. Every dataset is aligned
# first; the resumed Stage 3-8 pipelines run afterwards.
# Set PYTHON=... to choose a specific interpreter.
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$HERE/scripts/gui_fail_safe/fail_safe_gui_manual_align.py" "$@"
