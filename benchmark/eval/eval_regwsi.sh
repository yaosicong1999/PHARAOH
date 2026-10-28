#!/bin/bash
# Evaluate RegWSI runs.
#   eval/eval_regwsi.sh --mode keypoints|matrix --series iter1 [--run_dir <runs root>] [--datasets ...] [--conditions ...] [--rescore]
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_eval regwsi "$@"
