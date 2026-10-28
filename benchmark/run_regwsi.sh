#!/bin/bash
# RegWSI (DeeperHistReg) runner: registers the test dataset with every RegWSI configuration in
# $CONFIGS/regwsi.json (presets initial, nonrigid and hires at caps 1k-16k), each after one
# discarded warm-up run. Wrapper: methods/regwsi/wrapper.py.
set -euo pipefail
source "$(dirname "$0")/bench/launch.sh"
source "$BENCH_DIR/settings.sh"
_start
run_regwsi --config_json "$CONFIGS/regwsi.json" --dataset_json "$MANIFEST" --warmstart
_log "RegWSI done"
