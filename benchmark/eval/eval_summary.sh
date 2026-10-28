#!/bin/bash
# One table across methods (and series) from the per-method evaluations.
#   eval/eval_summary.sh --mode keypoints|matrix --series iter1 [--methods ...]
# Output: results/summary/<series>/<mode>/ (runs.csv, conditions.csv, best.csv, summary.json);
# several series go to results/summary/<s1>-<s2>-.../<mode>/.
#   eval/eval_summary.sh --mode keypoints --series iter1 --time-series iter1 iter2 iter3
# errors and memory from --series, time = mean over --time-series (their run records; no evaluation
# of those series needed) -> results/summary/iter1__time_iter1-iter2-iter3/<mode>/.
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_summary "$@"
