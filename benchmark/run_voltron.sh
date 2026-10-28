#!/bin/bash
# VoltRon runner: registers the test dataset with every VoltRon configuration in
# $CONFIGS/voltron.json (input caps 1k-16k), each after one discarded warm-up run.
# Wrapper: methods/voltron/wrapper.py (calls VoltRon through rscript of configs/environment.json).
set -euo pipefail
source "$(dirname "$0")/bench/launch.sh"
source "$BENCH_DIR/settings.sh"
_start
run_voltron --config_json "$CONFIGS/voltron.json" --dataset_json "$MANIFEST" --warmstart
_log "VoltRon done"
