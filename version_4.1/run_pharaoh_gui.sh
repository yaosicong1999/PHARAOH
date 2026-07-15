#!/usr/bin/env bash
# ============================================================
# run_pharaoh_gui.sh - launch the unified GUI pipeline (Stages 1-6)
#
#   ./run_pharaoh_gui.sh
#
# Opens gui_pipeline.py. Create/choose a RUN_DIR, then run Stage 1
# (pick DAPI or HE0 as the moving image next to the fixed H&E);
# Stages 2-6 are dispatched to engine_dapi/ or engine_he/ based on
# the mode chosen in Stage 1.
# Set PYTHON=... to choose a specific interpreter.
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$HERE/gui_pipeline.py" "$@"
