#!/bin/bash
# Evaluate VoltRon runs.
#   eval/eval_voltron.sh --mode keypoints|matrix --series iter1 [--run_dir <runs root>] [--datasets ...] [--conditions ...] [--rescore]
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_eval voltron "$@"
