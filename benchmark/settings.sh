# Settings shared by run_pharaoh.sh, run_voltron.sh, run_regwsi.sh, run_valis.sh and run_all.sh.
# Sourced, not run. Each value can also be set from the command line, e.g.
#   CONFIGS=configs/quick ./run_all.sh

SERIES="${SERIES:-test}"                      # output series: runs/<SERIES>/keypoints/<method>/<condition>/<dataset>/
CONFIGS="${CONFIGS:-configs/methods}"         # configs/methods: every benchmark configuration (29 + PHARAOH)
                                              # configs/quick: two configurations per method, for a fast check
MANIFEST="${MANIFEST:-manifests/keypoints.json}"   # the test dataset (download_test_data.sh)

# Resource limits of the benchmark (a 32 GiB workstation). A run that exceeds them is
# recorded as a failure and the batch continues (ON_LIMIT=skip).
MAX_RSS_GIB="${MAX_RSS_GIB:-28}"              # per-run process-tree RSS budget (GiB); 0 = off
MAX_SWAP_GIB="${MAX_SWAP_GIB:-30}"            # whole-host swap budget (GiB); 0 = off
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-5400}"   # per-run timeout
ON_LIMIT="${ON_LIMIT:-skip}"                  # after a budget stop: skip (record, continue) | stop (end batch)

SWAP_SETTLE="${SWAP_SETTLE:-on}"              # before each launch, wait until host swap is back down
SETTLE_CAP_GIB="${SETTLE_CAP_GIB:-5}"         #   ... to at most this (GiB)
SETTLE_TOLERANCE_GIB="${SETTLE_TOLERANCE_GIB:-3}"   # ... or the batch's starting swap + this, whichever is lower
SETTLE_TIMEOUT_SECONDS="${SETTLE_TIMEOUT_SECONDS:-1800}"   # longest wait; then the batch pauses (exit 75)
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-0}"     # pause after every run (seconds)
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-yes}"       # keep each method's transform files (needed for evaluation)

KEEP_AWAKE="${KEEP_AWAKE:-yes}"               # keep a Mac from sleeping while a runner runs (caffeinate)
STOP_ON_BATCH_STOP="${STOP_ON_BATCH_STOP:-yes}"   # a batch that ends with an error or a pause ends the script

# The runner interpreter is python.runner in configs/environment.json.
RUNNER_PYTHON="$(cd "$BENCH_DIR" && python3 -c "from bench.config import environment; print(environment()['python']['runner'])")"
LOG_DIR="$BENCH_DIR/logs"
