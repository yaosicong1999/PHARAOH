# Shell functions for eval_all.sh, plot.sh and eval/eval_<method>.sh. Not run on its own.
#
#   eval_<method>.sh --mode keypoints|matrix --series iter1 [iter2 ...] [--run_dir <runs root>]
#                    [--datasets ...] [--conditions ...] [--rescore]
#
# --run_dir is the folder holding <series>/<group>/<method>/ (default: runs_root of
# configs/environment.json). The interpreter is the method's eval_python there. Results go
# to results/<series>/<mode>/<method>/; the output is also written to logs/.
# Under eval_all.sh, SERIES and RUN_DIR from its settings block fill in --series and
# --run_dir when a line does not give them.

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

_eval_python() {  # the eval interpreter configs/environment.json names for a method
    (cd "$BENCH_DIR" && python3 -c "import sys
from bench.config import PACKAGES, environment
print(environment()['eval_python'][PACKAGES[sys.argv[1]]])" "$1")
}

_eval_usage() {
    sed -n '3,4p' "$BENCH_DIR/bench/eval.sh" | sed "s/eval_<method>.sh/$(basename "$0")/; s/^#//"
}

_has() {  # _has <flag> <args...>: is the flag among the args?
    local flag="$1"; shift
    local a
    for a in "$@"; do [ "$a" = "$flag" ] && return 0; done
    return 1
}

_eval() {
    local method="$1"; shift
    local mode="" runs_root="${RUN_DIR:-}" args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --mode)    mode="$2"; shift 2 ;;
            --run_dir) runs_root="$2"; shift 2 ;;
            -h|--help) _eval_usage; return 0 ;;
            *)         args+=("$1"); shift ;;
        esac
    done
    if [ "$mode" != "keypoints" ] && [ "$mode" != "matrix" ]; then
        echo "--mode must be keypoints or matrix"; _eval_usage; return 2
    fi
    if ! _has --series ${args[@]+"${args[@]}"}; then
        if [ -z "${SERIES:-}" ]; then echo "--series is required"; _eval_usage; return 2; fi
        args+=(--series $SERIES)
    fi
    if [ -n "$runs_root" ]; then
        args+=(--runs-root "$runs_root")
    fi
    local python log status
    python="$(_eval_python "$method")" || return 2
    mkdir -p "$BENCH_DIR/logs"
    log="$BENCH_DIR/logs/$(date '+%Y%m%dT%H%M%S')_eval_${mode}_${method}.log"
    _log "[eval] $method $mode -> log $log"
    set +e
    (cd "$BENCH_DIR" && "$python" "eval/$mode/$method.py" ${args[@]+"${args[@]}"}) 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

_summary() {
    local args=("$@")
    if ! _has --series "$@"; then
        if [ -z "${SERIES:-}" ]; then echo "--series is required"; return 2; fi
        args+=(--series $SERIES)
    fi
    if _has --time-series "$@" && [ -n "${RUN_DIR:-}" ] && ! _has --runs-root "$@"; then
        args+=(--runs-root "$RUN_DIR")
    fi
    (cd "$BENCH_DIR" && "$(_eval_python pharaoh)" eval/summarize.py "${args[@]}")
}

# ---- eval_all.sh / plot.sh -----------------------------------------------------------------------
_eval_step() {  # one line of eval_all.sh or plot.sh; STOP_ON_ERROR=yes ends the script on a failure
    local status=0
    "$@" || status=$?
    if [ "$status" -ne 0 ]; then
        _log "$* ended with exit $status"
        if [ "${STOP_ON_ERROR:-yes}" = "yes" ]; then
            _log "STOP_ON_ERROR=yes: not starting the remaining evaluations"; exit "$status"
        fi
    fi
    return 0
}

eval_pharaoh()          { _eval_step _eval pharaoh "$@"; }
eval_voltron()          { _eval_step _eval voltron "$@"; }
eval_regwsi()           { _eval_step _eval regwsi "$@"; }
eval_valis_tifffile()   { _eval_step _eval valis_tifffile "$@"; }
eval_valis_bioformats() { _eval_step _eval valis_bioformats "$@"; }
eval_summary()          { _eval_step _summary "$@"; }
visualization_dataset_keypoints() {  # per-dataset keypoint figures -> results/visualization/<series>/keypoints/<dataset>/
    local args=("$@")
    if ! _has --series "$@"; then args+=(--series $SERIES); fi
    _eval_step bash -c 'cd "$1" && shift && "$@"' _ "$BENCH_DIR" "$(_eval_python pharaoh)" eval/visualization_dataset_keypoints.py "${args[@]}"
}
_eval_start() {
    if [ "${KEEP_AWAKE:-no}" = "yes" ] && command -v caffeinate >/dev/null 2>&1; then
        caffeinate -ims -w $$ &
        _log "keeping the Mac awake while this script runs (caffeinate)"
    fi
}
