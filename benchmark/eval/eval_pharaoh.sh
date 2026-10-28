#!/bin/bash
# Evaluate PHARAOH runs.
#   eval/eval_pharaoh.sh --mode keypoints|matrix --series iter1 [--run_dir <runs root>] [--datasets ...] [--conditions ...] [--rescore]
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_eval pharaoh "$@"
