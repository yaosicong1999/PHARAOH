#!/bin/bash
# VALIS runner: registers the test dataset with every VALIS configuration (rigid, non-rigid and
# micro-registration at three size settings), once with the DAPI image decoded by tifffile
# ($CONFIGS/valis_tifffile.json) and once read by VALIS's Bio-Formats reader
# ($CONFIGS/valis_bioformats.json), each after one discarded warm-up run.
# The DAPI image is first re-encoded once for VALIS (prepare_valis_inputs.py; its time is charged
# to every VALIS run) and the re-encode is removed at the end. Wrapper: methods/valis/wrapper.py.
set -euo pipefail
source "$(dirname "$0")/bench/launch.sh"
source "$BENCH_DIR/settings.sh"
_start
prepare_valis_inputs --dataset_json "$MANIFEST"
run_valis_tifffile   --config_json "$CONFIGS/valis_tifffile.json"   --dataset_json "$MANIFEST" --warmstart
run_valis_bioformats --config_json "$CONFIGS/valis_bioformats.json" --dataset_json "$MANIFEST" --warmstart
clean_valis_inputs
_log "VALIS done"
