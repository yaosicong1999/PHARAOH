#!/usr/bin/env bash
# Step 3: PHARAOH on the two datasets, exactly as in the benchmark.
#
#   ./3_run.sh
#
# Runs ../benchmark/run.py with PHARAOH's single default configuration
# (../benchmark/configs/methods/pharaoh.json: no arguments, so Stage-4 tile fraction 0.30) through
# the benchmark's PHARAOH wrapper and release (../version_5.0_macos_linux). One registration at a
# time; each is limited to 28 GiB of memory and 90 minutes (resources.json). Runs are written to
# runs/minitest/<keypoints|matrix>/pharaoh/default/<dataset>/; a dataset already run successfully
# is skipped (delete its folder to run it again).
set -euo pipefail
source "$(dirname "$0")/settings.sh"
PY="$(pharaoh_python)"

for group in keypoints matrix; do
    _log "[run] PHARAOH, $group dataset"
    (cd "$BENCH_DIR" && "$PY" run.py --config configs/methods/pharaoh.json \
        --manifest "$BENCH_MANIFESTS/$group.json" --series "$SERIES" --keep-artifacts yes)
done
_log "[run] done; next: ./4_evaluate.sh"
