#!/bin/bash
# Evaluate VALIS, DAPI decoded by tifffile runs.
#   eval/eval_valis_tifffile.sh --mode keypoints|matrix --series iter1 [--run_dir <runs root>] [--datasets ...] [--conditions ...] [--rescore]
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_eval valis_tifffile "$@"
