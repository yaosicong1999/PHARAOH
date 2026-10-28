#!/bin/bash
# Plotter: per-dataset figures of the scored runs (run eval_all.sh, or the per-method evaluators, first).
#   ./plot.sh [--time method|wall]
# Writes results/visualization/<SERIES>/keypoints/human_crc_p1/ (PNG and editable PDF):
#   *_tre_boxplot                    every landmark TRE of each method's best configuration
#   *_resolution_tre                 median TRE across the input-resolution caps, one line per preset
#   *_runtime_tre, *_memory_tre      median TRE against method time and against peak memory
#   *_runtime_tre_with_size_memory   median TRE against time, point area = peak memory
# plus *_conditions.csv (every plotted run). .../ablation_bioformats/ adds VALIS with Bio-Formats.
set -euo pipefail
source "$(dirname "$0")/bench/eval.sh"
SERIES="${SERIES:-test}"
visualization_dataset_keypoints "$@"
