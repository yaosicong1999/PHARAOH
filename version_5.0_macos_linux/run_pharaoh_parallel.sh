#!/usr/bin/env bash
# Run manifest datasets concurrently in the PHARAOH conda environment.
#
# USAGE
#   ./run_pharaoh_parallel.sh <manifest.json> [options]
#
# Every argument after the script name is forwarded verbatim to
# run_pharaoh_parallel.py. The first positional argument is required and must be
# a JSON manifest; there are no other positionals.
#
# MANIFEST FORMAT (paths may be absolute or relative to the manifest file)
#   {
#     "pipeline_args": ["--optional", "--flags-applied-to-every-dataset"],
#     "datasets": [
#       {
#         "name":   "sample_a",                  # unique; becomes runs_<slug>/ and <slug>.log
#         "pair":   "dapi-he",                   # optional (default dapi-he):
#                                                # dapi-he | he-he | he-dapi | dapi-dapi
#         "moving": "/path/moving_dapi.ome.tif",
#         "fixed":  "/path/fixed_he.ome.tif",
#         "pipeline_args": ["--per-dataset-flags"]
#       }
#     ]
#   }
#   pipeline_args are passed straight to run_pharaoh.py (after everything the wrapper
#   adds, so a dataset's own --qc-images wins), but may not override what the wrapper
#   manages:
#     inputs/run dir     --pair --moving --fixed --run-dir
#     Stage-2 fallback   --stage2-relevel-rounds --stage2-nomanual --stage2-nofallback
#                        --no-failsafe (and the old --alignment-recovery* spellings)
#     Stage-1 levels     --stage1-fixed-level / --stage1-he-level --stage1-moving-level
#     Stage-3/4 gating   --stage3-* and --stage4-* scheduler options
#   (RESERVED_PIPELINE_OPTIONS in run_pharaoh_parallel.py is the exact list.)
#
# INPUTS THAT ARE NOT OME-TIFF
#   Any .btf/.tif/.tiff/.ndpi/.svs/.jpg/.jpeg/.czi named in the manifest is converted to
#   .ome.tif before the batch starts, written beside the source, and used for that
#   run. The manifest file itself is never rewritten -- it keeps naming the original
#   image. An .ome.tif already sitting beside the source is reused, never
#   overwritten (delete it to force a reconversion). Conversion happens before the
#   batch clock starts, so it is not counted in any dataset's elapsed time or in the
#   batch total; it is reported separately and recorded under "input_conversions" in
#   parallel_summary.json. --dry-run reports what would be converted and writes
#   nothing.
#
# OPTIONS (see `./run_pharaoh_parallel.sh --help` for the authoritative list)
#   --output-root PATH               batch output dir
#                                    (default: <this version dir>/outputs/parallel_runs_<timestamp>)
#   --jobs N                         max concurrent datasets (default: 4)
#   --python PATH                    interpreter for each pipeline (default: current)
#   --force                          recompute all stages in every dataset
#   --stop-after {1..8}              stop every pipeline after this stage
#   --dry-run                        validate the manifest and print commands only
#
#   Quality-control images (overrides parameters.json['qc_images'] for this batch;
#   a dataset can override the batch with its own --qc-images in pipeline_args)
#   --qc-images [GROUPS]             write QC images; bare = every group, or a
#                                    comma-separated subset of stage1_extras,
#                                    stage3_debug, stage4b_overlays, stage4b_density,
#                                    stage4c_patches, stage4c_overlays,
#                                    stage6_overlays, stage7_anchors, stage7_tree.
#                                    stage6_overlays is the one that writes
#                                    6_overlay_manual_*.png, 6_alternating_*.gif and
#                                    the cached keypoint panels
#   --no-qc-images                   write none (6_overlay_final_*.png is always written)
#
#   Stage-2 fallback (two independent axes: relevel, and manual)
#   Every child runs with relevelling and its own fail-safe window off; the batch
#   relevels the Stage-2 failures itself once every initial pipeline has finished.
#   --stage2-relevel-rounds N        retry rounds at other Stage-1 pyramid levels; each
#                                    round tries one level coarser then one finer.
#                                    0 skips relevelling (default: 1)
#   --stage2-nomanual                do not enter failures into
#                                    manual_alignment_queue.json (relevel as usual,
#                                    then leave them failed)
#   --stage2-nofallback              both of the above off: a Stage-2 failure is final
#   --failsafe-mode {end,instant,off}
#                                    when a person aligns what Stage 2 could not:
#                                    end     (default) work through
#                                            manual_alignment_queue.json after every
#                                            pipeline has finished, then finish each
#                                            aligned dataset
#                                    instant each dataset opens its own window the
#                                            moment its Stage 2 fails, the rest keep
#                                            running (its relevel rounds are skipped)
#                                    off     only write the queue, for
#                                            ./run_pharaoh_failsafe.sh later
#
#   Stage 4b
#   --stage4b-tile-topup-max-attempts N
#                                    tile top-up rounds per dataset when Stage 4b runs
#                                    short of usable tiles (default: run_pharaoh.py's;
#                                    0 switches it off)
#
#   Resources
#   --threads-per-dataset N          BLAS/OpenMP thread cap per dataset process
#
#   Stage 3 gating (pyramid-level reads)
#   --stage3-jobs N                  hard ceiling in Stage 3 (default: 4; 0 disables)
#   --stage3-adaptive                admit by RAM/CPU/queue depth (default)
#   --stage3-fixed                   use only the hard ceiling
#   --stage3-memory-per-job-gib F    override the per-dataset peak-RSS estimate
#                                    (default: derived per dataset from its pyramid levels)
#
#   Stage 4 gating (memory-heavy stage)
#   --stage4-jobs N                  hard ceiling in Stage 4 (default: 4; 0 disables)
#   --stage4-adaptive                admit by RAM/CPU/queue depth (default)
#   --stage4-fixed                   use only --stage4-jobs
#   --stage4-memory-per-job-gib F    peak RSS per dataset (default: 5.0)
#   --stage4-memory-reserve-gib F    RAM kept free (default: 4.0)
#   --stage4-cpu-threshold F         base total-CPU admission ceiling (default: 80)
#   --stage4-queue-cpu-boost F       CPU ceiling boost per queued dataset (default: 2)
#   --stage4-max-queue-cpu-boost F   cap on that boost (default: 10)
#   --stage4-cpu-bypass-after F      ignore CPU after this idle wait, seconds (default: 120)
#   --stage4-admission-cooldown F    spacing between admissions, seconds (default: 5)
#   --stage4-poll-seconds F          scheduler poll interval (default: 0.5)
#
#   Monitoring (on by default here, unlike a single run_pharaoh.py)
#   --monitor / --no-monitor         sample process-tree RSS; writes a stage-colored
#                                    plot per dataset and batch_timeline.html
#                                    (default: on)
#   --monitor-interval F             sampling interval in seconds (default: 1.0)
#
# EXAMPLES
#   ./run_pharaoh_parallel.sh my_datasets.json --dry-run
#   ./run_pharaoh_parallel.sh my_datasets.json --jobs 2 --threads-per-dataset 4
#   ./run_pharaoh_parallel.sh my_datasets.json --jobs 2 --stop-after 2
#   ./run_pharaoh_parallel.sh large_batch.json --jobs 10 --stage3-jobs 2 --stage4-jobs 3
#   ./run_pharaoh_parallel.sh one_sample.json --jobs 1 --monitor-interval 0.5
#   ./run_pharaoh_parallel.sh my_datasets.json --qc-images stage6_overlays
#   ./run_pharaoh_parallel.sh my_datasets.json --failsafe-mode off
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="PHARAOH"

# Avoid nesting `conda run` when the requested environment is already active.
# run_pharaoh_parallel.py uses sys.executable for every child pipeline, so all
# datasets inherit the same PHARAOH Python automatically.
if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    exec python "$HERE/run_pharaoh_parallel.py" "$@"
fi

CONDA_COMMAND="${CONDA_EXE:-}"
if [[ -z "$CONDA_COMMAND" ]]; then
    CONDA_COMMAND="$(command -v conda || true)"
fi
if [[ -z "$CONDA_COMMAND" ]]; then
    echo "[parallel] conda was not found; activate the PHARAOH environment first." >&2
    exit 127
fi

exec "$CONDA_COMMAND" run --no-capture-output \
    --name "$CONDA_ENV_NAME" \
    python "$HERE/run_pharaoh_parallel.py" "$@"
