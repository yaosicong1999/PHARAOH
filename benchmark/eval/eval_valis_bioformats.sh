#!/bin/bash
# Evaluate VALIS, DAPI read by VALIS's Bio-Formats runs.
#   eval/eval_valis_bioformats.sh --mode keypoints|matrix --series iter1 [--run_dir <runs root>] [--datasets ...] [--conditions ...] [--rescore]
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
_eval valis_bioformats "$@"
