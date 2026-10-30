#!/usr/bin/env bash
# Step 4: score the two runs as the benchmark does, and compare them with the benchmark's.
#
#   ./4_evaluate.sh
#
# Scoring is the benchmark's own (../benchmark/eval/eval_pharaoh.sh): landmark TRE for
# human_crc_p1 and every transcript's distance to 10x's mapping for human_heart, in H&E
# micrometers, written to results/minitest/<keypoints|matrix>/pharaoh/. check.py then prints each
# dataset's median, P95 and largest error and whether the median is CLOSE to the benchmark's (within
# 0.4 um); README.md lists the same metrics from four platforms. Exits non-zero on FAR or FAIL.
set -euo pipefail
source "$(dirname "$0")/settings.sh"
PY="$(pharaoh_python)"

for mode in keypoints matrix; do
    _log "[evaluate] $mode"
    bash "$BENCH_DIR/eval/eval_pharaoh.sh" --mode "$mode" --series "$SERIES"
done
"$PY" "$MINITEST_DIR/check.py"
