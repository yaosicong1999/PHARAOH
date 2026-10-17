#!/usr/bin/env bash
# ============================================================
# run_pharaoh_gui.sh - launch the unified GUI pipeline (Stages 1-8)
#
#   ./run_pharaoh_gui.sh
#
# Opens scripts/gui_pipeline.py. Create/choose a RUN_DIR, then run Stage 1
# (select moving/foreground and fixed/background, then orient the moving image).
# Stage 2 opens interactive alignment with an optional in-window ORB suggestion.
# Stages 3-6 use the selected pair; DAPI-H&E continues through Stages 7-8.
# Set PYTHON=... to choose a specific interpreter.
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$HERE/scripts/gui_pipeline.py" "$@"
