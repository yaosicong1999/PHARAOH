#!/bin/bash
# Per-dataset keypoint figures: best-condition box plot, resolution sweep, TRE vs time / memory.
#   eval/visualization_dataset_keypoints.sh --series iter1 [--datasets ...] [--time method|wall] [--methods ...]
# Output: results/visualization/<series>/keypoints/<dataset>/ and .../ablation_bioformats/<dataset>/.
set -euo pipefail
source "$(dirname "$0")/../bench/eval.sh"
cd "$BENCH_DIR"
"$(_eval_python pharaoh)" eval/visualization_dataset_keypoints.py "$@"
