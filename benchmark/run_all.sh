#!/bin/bash
# Overall runner: every method on the test dataset, in the benchmark's order. Completed runs are
# skipped when the script is run again, so an interrupted run resumes. Settings: settings.sh.
#   ./run_all.sh                          every benchmark configuration
#   CONFIGS=configs/quick ./run_all.sh    two configurations per method
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/run_pharaoh.sh"
"$HERE/run_voltron.sh"
"$HERE/run_regwsi.sh"
"$HERE/run_valis.sh"
echo "all methods done; next: ./eval_all.sh"
