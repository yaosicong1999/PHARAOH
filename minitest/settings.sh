# Shared settings of the minitest scripts. Sourced by them, not run on its own.
#
# The minitest drives ../benchmark (its runner, PHARAOH wrapper and scoring) with its own
# manifests, environment and output folders, so nothing it writes lands in ../benchmark/.

MINITEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$MINITEST_DIR/.." && pwd)"
BENCH_DIR="$REPO_DIR/benchmark"
SERIES="minitest"                              # runs/minitest/<group>/pharaoh/default/<dataset>/

export BENCH_MANIFESTS="$MINITEST_DIR/manifests"          # the two minitest datasets
export BENCH_ENVIRONMENT="$MINITEST_DIR/environment.json" # written by 1_install.sh
export BENCH_RESOURCES="$MINITEST_DIR/resources.json"     # run limits; no waiting on swap

_log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

pharaoh_python() {  # the PHARAOH interpreter 1_install.sh recorded
    if [ ! -f "$BENCH_ENVIRONMENT" ]; then
        echo "[minitest] $BENCH_ENVIRONMENT is missing; run ./1_install.sh first" >&2
        return 1
    fi
    python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['python']['pharaoh'])" "$BENCH_ENVIRONMENT"
}
