#!/bin/bash
# Overall evaluator: scores every method's runs on the test dataset against the 10x landmark pairs,
# then puts all methods in one table, and draws the figures (plot.sh).
#   ./eval_all.sh
# Per method: eval/eval_<method>.sh --mode keypoints --series test
#   -> results/<SERIES>/keypoints/<method>/ (runs.csv, points/<dataset>/<condition>.csv: TRE per landmark)
# Summary: results/summary/<SERIES>/keypoints/ (runs.csv, conditions.csv, best.csv, summary.json):
#   median and per-landmark TRE (µm), method time and peak memory of every configuration.
# A method with no runs is skipped; runs already scored with the same code are reused.
set -euo pipefail
source "$(dirname "$0")/bench/eval.sh"

SERIES="${SERIES:-test}"      # series written by the runners (settings.sh)
RUN_DIR="${RUN_DIR:-}"        # folder holding <SERIES>/; empty = runs_root of configs/environment.json
KEEP_AWAKE="${KEEP_AWAKE:-yes}"
STOP_ON_ERROR="${STOP_ON_ERROR:-yes}"

_eval_start
eval_pharaoh          --mode keypoints
eval_voltron          --mode keypoints
eval_regwsi           --mode keypoints
eval_valis_tifffile   --mode keypoints
eval_valis_bioformats --mode keypoints
eval_summary          --mode keypoints
"$BENCH_DIR/plot.sh"
_log "evaluation done: results/summary/$SERIES/keypoints/ and results/visualization/$SERIES/keypoints/"
