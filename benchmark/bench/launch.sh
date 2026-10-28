# Shell functions for run_benchmark.sh. Not run on its own.
#
#   run_<method> --config_json <config> --dataset_json <manifest> [--warmstart] [run.py options]
#
# Each call is one batch: python run.py with the settings block of run_benchmark.sh passed
# as resource overrides. Anything else (--datasets, --conditions, --list-plan, --dry-run,
# --rerun-completed, --skip-failed) goes through to run.py unchanged.

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

_run_method() {
    local method="$1"; shift
    local config="" dataset="" extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --config_json)  config="$2"; shift 2 ;;
            --dataset_json) dataset="$2"; shift 2 ;;
            *)              extra+=("$1"); shift ;;
        esac
    done
    if [ -z "$config" ] || [ -z "$dataset" ]; then
        _log "run_${method}: --config_json and --dataset_json are required"; return 2
    fi

    local log_file="${LOG_DIR}/$(date '+%Y%m%dT%H%M%S')_${SERIES}_${method}_$(basename "${dataset%.json}").log"
    _log "run_${method}: series ${SERIES}, config ${config}, datasets ${dataset} -> log ${log_file}"
    local status
    set +e
    (cd "$BENCH_DIR" && "$RUNNER_PYTHON" run.py --method "$method" \
        --config "$config" --manifest "$dataset" --series "$SERIES" \
        --max-rss-gib "$MAX_RSS_GIB" --max-swap-gib "$MAX_SWAP_GIB" \
        --timeout "$RUN_TIMEOUT_SECONDS" --on-limit "$ON_LIMIT" \
        --swap-settle "$SWAP_SETTLE" --settle-cap-gib "$SETTLE_CAP_GIB" \
        --settle-tolerance-gib "$SETTLE_TOLERANCE_GIB" --settle-timeout "$SETTLE_TIMEOUT_SECONDS" \
        --cooldown "$COOLDOWN_SECONDS" --keep-artifacts "${KEEP_ARTIFACTS:-yes}" \
        ${extra[@]+"${extra[@]}"}) 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
        _log "run_${method}: ended with exit ${status} (75 = host-resource gate or PAUSE; other = error)"
        if [ "$STOP_ON_BATCH_STOP" = "yes" ]; then
            _log "STOP_ON_BATCH_STOP=yes: not starting the remaining batches"; exit "$status"
        fi
    fi
    return 0
}

# Re-encode the VALIS inputs of a manifest for this SERIES before its VALIS batches, so the
# series' conversion is measured once, up front, and charged to each of its VALIS runs.
prepare_valis_inputs() {
    local dataset=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --dataset_json) dataset="$2"; shift 2 ;;
            *) _log "prepare_valis_inputs: unknown option $1"; return 2 ;;
        esac
    done
    [ -n "$dataset" ] || { _log "prepare_valis_inputs: --dataset_json is required"; return 2; }
    local python log_file status
    python="$(cd "$BENCH_DIR" && python3 -c "from bench.config import environment; print(environment()['python']['valis'])")"
    log_file="${LOG_DIR}/$(date '+%Y%m%dT%H%M%S')_${SERIES}_prepare_valis_inputs_$(basename "${dataset%.json}").log"
    _log "prepare_valis_inputs: series ${SERIES}, datasets ${dataset} -> log ${log_file}"
    set +e
    (cd "$BENCH_DIR" && "$python" prepare_valis_inputs.py --series "$SERIES" --manifest "$dataset") 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
        _log "prepare_valis_inputs: ended with exit ${status}"
        if [ "$STOP_ON_BATCH_STOP" = "yes" ]; then exit "$status"; fi
    fi
    return 0
}

# Delete the VALIS input re-encodes once a manifest's VALIS batches are done. The next series
# re-encodes its own anyway (prepare_valis_inputs), so this only frees the disk. Not reached
# if a batch stops the script, so a resumed script can reuse them.
clean_valis_inputs() {
    local cache size
    cache="$(cd "$BENCH_DIR" && python3 -c "from bench.config import environment; print(environment()['cache_root'])")/valis_inputs"
    case "$cache" in
        */valis_inputs) ;;
        *) _log "clean_valis_inputs: unexpected cache path $cache; nothing removed"; return 2 ;;
    esac
    if [ -d "$cache" ]; then
        size="$(du -sh "$cache" | cut -f1)"
        rm -rf "$cache"
        _log "clean_valis_inputs: removed $cache ($size)"
    else
        _log "clean_valis_inputs: $cache is already gone"
    fi
}

run_pharaoh()          { _run_method pharaoh "$@"; }
run_voltron()          { _run_method voltron "$@"; }
run_regwsi()           { _run_method regwsi "$@"; }
run_valis_tifffile()   { _run_method valis_tifffile "$@"; }    # DAPI decoded by tifffile
run_valis_bioformats() { _run_method valis_bioformats "$@"; }  # DAPI read by VALIS's Bio-Formats

_start() {
    mkdir -p "$LOG_DIR"
    if [ "$KEEP_AWAKE" = "yes" ] && command -v caffeinate >/dev/null 2>&1; then
        # -i idle sleep, -m disk sleep, -s system sleep (on AC power); ends with this script.
        caffeinate -ims -w $$ &
        _log "keeping the Mac awake while this script runs (caffeinate)"
    fi
}
