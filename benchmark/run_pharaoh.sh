#!/bin/bash
# PHARAOH runner: registers the test dataset with PHARAOH's single default configuration
# ($CONFIGS/pharaoh.json), after one discarded warm-up run. Wrapper: methods/pharaoh/wrapper.py,
# which runs the release in pharaoh_root of configs/environment.json. Settings: settings.sh.
set -euo pipefail
source "$(dirname "$0")/bench/launch.sh"
source "$BENCH_DIR/settings.sh"
_start
run_pharaoh --config_json "$CONFIGS/pharaoh.json" --dataset_json "$MANIFEST" --warmstart
_log "PHARAOH done"
