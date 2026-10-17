#!/usr/bin/env python3
"""Run multiple PHARAOH datasets concurrently with isolated outputs.

The input manifest is a JSON object containing a ``datasets`` list. Each
dataset needs a unique name, a fixed ``he`` image, and exactly one moving image
(``dapi``, ``hefg``, or the ``he0`` alias). Paths in the manifest are resolved
relative to the manifest file.
"""

import argparse
from collections import deque
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
PRE_PROCESS = SCRIPTS / "pre_process"
if str(PRE_PROCESS) not in sys.path:
    sys.path.insert(0, str(PRE_PROCESS))
from windows_utf8 import ensure_utf8_mode
ensure_utf8_mode()  # Windows: re-run under -X utf8; no-op elsewhere


from monitoring import (
    MemorySample,
    StageTracker,
    sample_running_processes,
    write_monitor_artifacts,
)
from stage2_relevel import (
    pyramid_top_levels,
    relevel_pairs,
    resolve_stage2_fallback,
)
from ensure_ome_tiff import ensure_ome_tiff
from my_utils import QC_IMAGE_GROUPS, normalize_qc_images_setting
from scheduling import Stage4SchedulerConfig, replace_with_retry
import run_roles
from stage3_memory import estimate_stage3_peak_gib


PIPELINE = HERE / "run_pharaoh.py"
FAILSAFE_LAUNCHER = SCRIPTS / "gui_fail_safe" / "fail_safe_gui_manual_align.py"
FAILSAFE_SUMMARY_NAME = "fail_safe_manual_alignment_summary.json"
MANUAL_ALIGNMENT_QUEUE_NAME = "manual_alignment_queue.json"
RESERVED_PIPELINE_OPTIONS = {
    "--pair",
    "--moving",
    "--fixed",
    "--run-dir",
    "--stage2-relevel-rounds",
    "--stage2-nomanual",
    "--stage2-nofallback",
    "--no-failsafe",
    "--alignment-recovery",
    "--no-alignment-recovery",
    "--alignment-recovery-rounds",
    # Both spellings of the fixed level: run_pharaoh.py accepts either, and the
    # relevel phase injects the level itself.
    "--stage1-fixed-level",
    "--stage1-he-level",
    "--stage1-moving-level",
    "--stage3-slots-dir",
    "--stage3-max-concurrent",
    "--stage3-adaptive",
    "--stage3-memory-per-job-gib",
    "--stage3-memory-reserve-gib",
    "--stage3-cpu-threshold",
    "--stage3-queue-cpu-boost",
    "--stage3-max-queue-cpu-boost",
    "--stage3-cpu-bypass-after",
    "--stage3-admission-cooldown",
    "--stage3-poll-seconds",
    "--stage4-slots-dir",
    "--stage4-max-concurrent",
    "--stage4-adaptive",
    "--stage4-memory-per-job-gib",
    "--stage4-memory-reserve-gib",
    "--stage4-cpu-threshold",
    "--stage4-queue-cpu-boost",
    "--stage4-max-queue-cpu-boost",
    "--stage4-cpu-bypass-after",
    "--stage4-admission-cooldown",
    "--stage4-poll-seconds",
}
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class Dataset:
    name: str
    slug: str
    pair: str
    moving: Path
    fixed: Path
    pipeline_args: Sequence[str]


@dataclass
class RunningDataset:
    dataset: Dataset
    command: List[str]
    run_dir: Path
    log_path: Path
    process: subprocess.Popen
    log_file: TextIO
    reader: threading.Thread
    started_at: str
    started_perf: float
    stage_tracker: StageTracker
    monitor_samples: List[MemorySample]
    relevel_attempt: Optional[dict] = None


@dataclass
class Stage2Relevel:
    dataset: Dataset
    command: List[str]
    run_dir: Path
    log_path: Path
    initial_result: dict
    attempts: List[dict]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE).strip("._-")
    if not slug:
        raise ValueError(f"dataset name has no usable filename characters: {name!r}")
    return slug


def _resolve_input(value, *, field: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{field} image not found: {path}")
    return path


def _validate_pipeline_args(value, *, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON list of strings")
    for arg in value:
        option = arg.split("=", 1)[0]
        if option in RESERVED_PIPELINE_OPTIONS:
            raise ValueError(
                f"{field} cannot override {option}; inputs and run directories "
                "are managed by the parallel wrapper"
            )
    return list(value)


def load_manifest(path: Path):
    """Load and validate a parallel-run manifest."""
    manifest_path = path.expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("manifest must contain a non-empty 'datasets' list")

    common_args = _validate_pipeline_args(
        raw.get("pipeline_args"), field="manifest pipeline_args"
    )
    datasets = []
    names = set()
    slugs = set()
    for index, item in enumerate(raw_datasets, start=1):
        label = f"datasets[{index - 1}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be a JSON object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        name = name.strip()
        slug = _slugify(name)
        if name in names:
            raise ValueError(f"duplicate dataset name: {name!r}")
        slug_key = slug.casefold()
        if slug_key in slugs:
            raise ValueError(
                f"dataset names produce the same output directory name: {slug!r}"
            )
        names.add(name)
        slugs.add(slug_key)

        # A dataset names its pair and its two images. `dapi` + `he` is 4.10's shape
        # and every shipped manifest used it, so it still reads as `dapi-he`; `hefg`
        # and `he0` are gone with the flags that wrote them.
        if item.get("moving") or item.get("fixed"):
            pair_name = str(item.get("pair") or run_roles.DEFAULT_PAIR)
            moving_field, fixed_field = "moving", "fixed"
        elif item.get("dapi"):
            pair_name = "dapi-he"
            moving_field, fixed_field = "dapi", "he"
        else:
            raise ValueError(
                f"{label} must provide 'moving' and 'fixed' (with an optional 'pair'), "
                f"or 4.10's 'dapi' and 'he'"
            )
        if not item.get(fixed_field):
            raise ValueError(f"{label} must provide {fixed_field!r}")
        try:
            pair = run_roles.pair_from_name(pair_name)
        except ValueError as exc:
            raise ValueError(f"{label}.pair: {exc}") from None
        if not pair.implemented:
            raise ValueError(f"{label}.pair: {pair.name} is not implemented yet")
        dataset_args = _validate_pipeline_args(
            item.get("pipeline_args"), field=f"{label}.pipeline_args"
        )
        datasets.append(
            Dataset(
                name=name,
                slug=slug,
                pair=pair.name,
                moving=_resolve_input(
                    item[moving_field],
                    field=f"{label}.{moving_field}",
                    base_dir=manifest_path.parent,
                ),
                fixed=_resolve_input(
                    item.get(fixed_field),
                    field=f"{label}.{fixed_field}",
                    base_dir=manifest_path.parent,
                ),
                pipeline_args=common_args + dataset_args,
            )
        )
    return manifest_path, datasets, raw


def prepare_ome_tiff_inputs(datasets, *, dry_run=False, log=print):
    """Convert any non-OME-TIFF input, returning the rewritten datasets and a report.

    Runs before the batch clock starts, so conversion never lands in a dataset's
    elapsed time or in the batch total. Conversions are sequential on purpose: they
    are IO- and memory-heavy, and running one per dataset at once would compete with
    nothing useful, since no pipeline has started yet.

    The manifest file is never rewritten -- the substitution lives only in the
    Dataset objects this run uses.
    """
    cache = {}
    conversions = []
    prepared = []
    for dataset in datasets:
        moving, moving_record = ensure_ome_tiff(
            dataset.moving, log=log, cache=cache, dry_run=dry_run, label=dataset.name
        )
        he, he_record = ensure_ome_tiff(
            dataset.fixed, log=log, cache=cache, dry_run=dry_run, label=dataset.name
        )
        conversions.extend(record for record in (moving_record, he_record) if record)
        prepared.append(replace(dataset, moving=moving, fixed=he))
    return prepared, conversions


def build_command(
    dataset: Dataset,
    *,
    python: str,
    run_dir: Path,
    force: bool,
    stop_after: Optional[str],
    stage4_slots_dir: Optional[Path] = None,
    stage4_max_concurrent: Optional[int] = None,
    stage4_scheduler_config: Optional[Stage4SchedulerConfig] = None,
    stage3_slots_dir: Optional[Path] = None,
    stage3_max_concurrent: Optional[int] = None,
    stage3_scheduler_config: Optional[Stage4SchedulerConfig] = None,
    qc_images: Optional[str] = None,
    stage2_nomanual: bool = False,
    failsafe: bool = False,
    stage4b_tile_topup_max_attempts: Optional[int] = None,
) -> List[str]:
    command = [
        python,
        str(PIPELINE),
        # Relevelling is coordinated globally after all initial work ends; an
        # in-process retry here would bypass its fair serialized queue.
        "--stage2-relevel-rounds",
        "0",
        # A batch decides when a person is asked for a manual alignment: right when
        # a dataset fails (--failsafe-mode instant, the child opens its own window)
        # or once the batch is done (the default, handled after every pipeline ends).
        *(() if failsafe else ("--no-failsafe",)),
        *(("--stage2-nomanual",) if stage2_nomanual else ()),
        # Left to run_pharaoh.py's own default unless the batch overrides it, so the two
        # entry points cannot disagree about how many top-up rounds a dataset gets.
        *(() if stage4b_tile_topup_max_attempts is None else
          ("--stage4b-tile-topup-max-attempts", str(stage4b_tile_topup_max_attempts))),
        "--pair",
        dataset.pair,
        "--moving",
        str(dataset.moving),
        "--fixed",
        str(dataset.fixed),
        "--run-dir",
        str(run_dir),
    ]
    if force:
        command.append("--force")
    if stop_after is not None:
        command.extend(("--stop-after", stop_after))
    if qc_images is not None:
        # Placed before the manifest's pipeline_args, which are appended last, so a
        # dataset that names its own --qc-images still wins over the batch default.
        command.extend(("--qc-images", qc_images))
    if stage4_slots_dir is not None and stage4_max_concurrent is not None:
        command.extend((
            "--stage4-slots-dir",
            str(stage4_slots_dir),
            "--stage4-max-concurrent",
            str(stage4_max_concurrent),
        ))
        if stage4_scheduler_config is not None:
            if stage4_scheduler_config.max_concurrent != stage4_max_concurrent:
                raise ValueError(
                    "Stage-4 scheduler config and hard concurrency limit disagree"
                )
            if stage4_scheduler_config.adaptive:
                command.append("--stage4-adaptive")
            command.extend((
                "--stage4-memory-per-job-gib",
                str(stage4_scheduler_config.memory_per_job_gib),
                "--stage4-memory-reserve-gib",
                str(stage4_scheduler_config.memory_reserve_gib),
                "--stage4-cpu-threshold",
                str(stage4_scheduler_config.cpu_threshold_percent),
                "--stage4-queue-cpu-boost",
                str(stage4_scheduler_config.queue_cpu_boost_per_waiter),
                "--stage4-max-queue-cpu-boost",
                str(stage4_scheduler_config.queue_cpu_boost_max),
                "--stage4-cpu-bypass-after",
                str(stage4_scheduler_config.cpu_bypass_after_seconds),
                "--stage4-admission-cooldown",
                str(stage4_scheduler_config.admission_cooldown_seconds),
                "--stage4-poll-seconds",
                str(stage4_scheduler_config.poll_seconds),
            ))
    # Stage 3 gets the same admission gate. It holds a whole pyramid level of both
    # slides at once and measured 4.7-6.1 GiB per dataset on the CRC slides, above
    # Stage 4's 4.0 GiB, so two datasets entering it together is what actually risks
    # exhausting memory.
    if stage3_slots_dir is not None and stage3_max_concurrent is not None:
        command.extend((
            "--stage3-slots-dir", str(stage3_slots_dir),
            "--stage3-max-concurrent", str(stage3_max_concurrent),
        ))
        if stage3_scheduler_config is not None:
            if stage3_scheduler_config.max_concurrent != stage3_max_concurrent:
                raise ValueError(
                    "Stage-3 scheduler config and hard concurrency limit disagree"
                )
            if stage3_scheduler_config.adaptive:
                command.append("--stage3-adaptive")
            command.extend((
                "--stage3-memory-per-job-gib", str(stage3_scheduler_config.memory_per_job_gib),
                "--stage3-memory-reserve-gib", str(stage3_scheduler_config.memory_reserve_gib),
                "--stage3-cpu-threshold", str(stage3_scheduler_config.cpu_threshold_percent),
                "--stage3-queue-cpu-boost", str(stage3_scheduler_config.queue_cpu_boost_per_waiter),
                "--stage3-max-queue-cpu-boost", str(stage3_scheduler_config.queue_cpu_boost_max),
                "--stage3-cpu-bypass-after", str(stage3_scheduler_config.cpu_bypass_after_seconds),
                "--stage3-admission-cooldown", str(stage3_scheduler_config.admission_cooldown_seconds),
                "--stage3-poll-seconds", str(stage3_scheduler_config.poll_seconds),
            ))
    command.extend(dataset.pipeline_args)
    return command


def _pump_output(
    name: str,
    stream,
    log_file: TextIO,
    print_lock: threading.Lock,
    stage_tracker: StageTracker,
):
    try:
        for line in iter(stream.readline, ""):
            stage_tracker.observe_line(line)
            log_file.write(line)
            log_file.flush()
            with print_lock:
                print(f"[{name}] {line}", end="", flush=True)
    finally:
        stream.close()


def _launch(
    dataset: Dataset,
    command: List[str],
    run_dir: Path,
    log_path: Path,
    env: Dict[str, str],
    print_lock: threading.Lock,
    *,
    relevel_attempt: Optional[dict] = None,
) -> RunningDataset:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open(
        "a" if relevel_attempt is not None else "w", encoding="utf-8"
    )
    started_at = datetime.now().isoformat(timespec="microseconds")
    started_perf = time.perf_counter()
    stage_tracker = StageTracker(started_perf)
    if relevel_attempt is not None:
        log_file.write(
            f"\n########## Stage-2 relevel attempt "
            f"{relevel_attempt['attempt']} ##########\n"
        )
    log_file.write(f"started_at: {started_at}\ncommand: {_display_command(command)}\n\n")
    log_file.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(HERE),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_NEW_PROCESS_GROUP,
        )
    except Exception:
        log_file.close()
        raise
    assert process.stdout is not None
    reader = threading.Thread(
        target=_pump_output,
        args=(dataset.name, process.stdout, log_file, print_lock, stage_tracker),
        name=f"pharaoh-log-{dataset.slug}",
        daemon=True,
    )
    reader.start()
    return RunningDataset(
        dataset=dataset,
        command=command,
        run_dir=run_dir,
        log_path=log_path,
        process=process,
        log_file=log_file,
        reader=reader,
        started_at=started_at,
        started_perf=started_perf,
        stage_tracker=stage_tracker,
        monitor_samples=[],
        relevel_attempt=relevel_attempt,
    )


def _display_command(command) -> str:
    """The command as this platform's shell would take it, for logs and --dry-run."""
    if sys.platform == "win32":
        return subprocess.list2cmdline([str(part) for part in command])
    return shlex.join(command)


# Each pipeline runs in its own process group so the batch can stop it together with
# its stage scripts and their worker pools. POSIX: a new session, signalled with
# killpg. Windows has neither; the child gets a new process group (so the console's
# Ctrl+C reaches only the batch, as with a POSIX session) and is stopped by walking
# its process tree.
_NEW_PROCESS_GROUP = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32" else {"start_new_session": True}
)
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _stop_process_tree_windows(process: subprocess.Popen, sig):
    import psutil

    try:
        root = psutil.Process(process.pid)
        members = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for member in members:
        try:
            # Windows has no graceful signal for a process in another group;
            # terminate() and kill() are both TerminateProcess.
            member.kill() if sig == _KILL_SIGNAL else member.terminate()
        except psutil.NoSuchProcess:
            pass


def _stop_process_group(process: subprocess.Popen, sig):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        _stop_process_tree_windows(process, sig)
        return
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _current_auto_alignment_failure(run_dir: Path, started_at: str):
    """Return this invocation's final Stage-2 failure event, if present."""
    event_path = run_dir / "pipeline_times.json"
    try:
        data = json.loads(event_path.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(started_at)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None

    current_events = []
    for event in data.get("stage2_events", []):
        if not isinstance(event, dict):
            continue
        try:
            event_time = datetime.fromisoformat(str(event.get("time")))
        except (TypeError, ValueError):
            continue
        if event_time >= started:
            current_events.append(event)

    for event in reversed(current_events):
        event_name = event.get("event")
        if event_name == "cli_stage2_orb_done":
            return None
        if event_name == "cli_stage2_orb_failed":
            record = dict(event)
            record["event_file"] = str(event_path.resolve())
            return record
    return None


def _manual_alignment_case(
    dataset: Dataset,
    *,
    run_dir: Path,
    log_path: Path,
    started_at: str,
    returncode: int,
):
    """Build one manifest-compatible manual-alignment queue entry."""
    if returncode == 0:
        return None
    failure = _current_auto_alignment_failure(run_dir, started_at)
    if failure is None:
        return None

    attempts = failure.get("attempts", [])
    methods = [
        str(attempt.get("method", "")).lower()
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("method")
    ]
    return {
        "name": dataset.name,
        "pair": dataset.pair,
        "moving": str(dataset.moving),
        "fixed": str(dataset.fixed),
        "run_dir": str(run_dir.resolve()),
        "log": str(log_path.resolve()),
        # Honor Stage 2's explicit manual-fallback flag for every supported pair.
        # Do not reinterpret old queue entries or override a disabled fallback.
        "manual_fallback": failure.get("manual_fallback", True) is not False,
        "failure": {
            "type": "automatic_alignment_failed",
            "stage": 2,
            "methods": methods,
            "event_time": failure.get("time"),
            "attempts": attempts,
            "event_file": failure["event_file"],
        },
    }


def _next_relevel_levels(dataset: Dataset, run_dir: Path, rounds: int,
                           attempts=()):
    """The level pair for the next relevel attempt, or None when the rounds are spent.

    Shares `scripts/stage2_relevel.py` with the single-dataset path so the two
    cannot drift; the order is coarser first, then finer, one of each per round.

    The plan is anchored on the levels Stage 1 *originally* chose, taken from the
    first attempt when one has been made. Re-reading `images_info.json` would anchor
    it on the levels the previous attempt just wrote, so round 1 would step coarser to
    4 and then "finer" back to 3 rather than down to 2 -- retrying the level that had
    already failed.
    """
    pair = run_roles.pair_from_name(dataset.pair)
    moving_key = f"{pair.key_prefix(run_roles.MOVING)}_level"
    fixed_key = f"{pair.key_prefix(run_roles.FIXED)}_level"
    try:
        info = json.loads(
            (run_dir / "images_info.json").read_text(encoding="utf-8")
        )
        he_level = info[fixed_key]
        moving_level = info[moving_key]
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    for value in (he_level, moving_level):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None

    attempts = list(attempts or ())
    if attempts:
        first = attempts[0]
        he_level = first.get("previous_he_level", he_level)
        moving_level = first.get("previous_moving_level", moving_level)
    he_top, moving_top = pyramid_top_levels(info, moving_key, fixed_key)
    pairs = relevel_pairs(he_level, moving_level, rounds, he_top, moving_top)
    if len(attempts) >= len(pairs):
        return None
    next_he, next_moving = pairs[len(attempts)]
    return {
        "previous_he_level": he_level,
        "previous_moving_level": moving_level,
        "he_level": next_he,
        "moving_level": next_moving,
        "direction": "coarser" if next_he > he_level or next_moving > moving_level
                     else "finer",
    }


def _alignment_stage_in_progress(item: RunningDataset) -> bool:
    """Keep high-resolution Stage 1/2 work serialized across failed samples."""
    if item.relevel_attempt is None or item.process.poll() is not None:
        return False
    stage = item.stage_tracker.current_stage()
    return stage == "Startup" or re.match(r"^Stage\s+[12](?:\s|$)", stage) is not None


def _stage2_relevel_details(state: Stage2Relevel, *, status: str, reason=None):
    details = {
        "status": status,
        "recovered": status == "recovered",
        "initial_returncode": state.initial_result["returncode"],
        "initial_failure": (state.initial_result.get("alignment_failure_case")
                            or state.initial_result.get("manual_alignment_case", {})).get("failure"),
        "attempts": [dict(attempt) for attempt in state.attempts],
    }
    if reason is not None:
        details["reason"] = reason
    return details


def _finish(
    item: RunningDataset,
    returncode: int,
    *,
    monitor: bool,
    monitor_interval: float,
):
    item.reader.join()
    elapsed = time.perf_counter() - item.started_perf
    finished_at = datetime.now().isoformat(timespec="microseconds")
    item.log_file.write(
        f"\nfinished_at: {finished_at}\nreturncode: {returncode}\n"
        f"elapsed_seconds: {elapsed:.6f}\n"
    )
    item.log_file.close()
    result = {
        "name": item.dataset.name,
        "returncode": int(returncode),
        "started_at": item.started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 6),
        "run_dir": str(item.run_dir),
        "log": str(item.log_path),
        "command": item.command,
    }
    manual_case = _manual_alignment_case(
        item.dataset,
        run_dir=item.run_dir,
        log_path=item.log_path,
        started_at=item.started_at,
        returncode=returncode,
    )
    if manual_case is not None:
        # Two facts, recorded apart: Stage 2 failed (what the relevel rounds act on),
        # and whether a person may take it over (what the manual queue acts on).
        result["automatic_alignment_failed"] = True
        result["alignment_failure_case"] = manual_case
        result["requires_manual_alignment"] = bool(manual_case["manual_fallback"])
        if manual_case["manual_fallback"]:
            result["manual_alignment_case"] = manual_case
    scheduler_record = item.run_dir / "stage4_scheduler.json"
    if scheduler_record.is_file():
        try:
            result["stage4_scheduler"] = json.loads(
                scheduler_record.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            result["stage4_scheduler"] = {"error": str(exc)}
    if monitor:
        try:
            result["monitor"] = write_monitor_artifacts(
                output_dir=item.run_dir,
                sample_name=item.dataset.name,
                samples=item.monitor_samples,
                stage_intervals=item.stage_tracker.intervals(elapsed),
                duration_seconds=elapsed,
                sample_interval_seconds=monitor_interval,
            )
        except Exception as exc:
            result["monitor"] = {"error": str(exc)}
    return result


def _write_summary(path: Path, summary: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def failsafe_mode(args) -> str:
    """When this batch asks a person to align what Stage 2 could not.

    "end"     - after every pipeline has finished, work through the whole queue in
                one pass, then let each aligned dataset finish its stages.
    "instant" - each child opens its own window the moment its Stage 2 fails, so a
                person can align it while the rest of the batch is still running.
                That skips the relevel rounds for it: those run after the initial
                batch, and the point of "instant" is not to wait.
    "off"     - write the queue and stop, for ./run_pharaoh_failsafe.sh later.
    """
    if args.stage2_nomanual:
        # Manual alignment is switched off for this batch; there is nobody to ask.
        return "off"
    return args.failsafe_mode


def run_failsafe_after_batch(queue_path: Path, queue_document: dict, *,
                             mode: str, dry_run: bool, python: str = sys.executable):
    """Work through the finished batch's queue, once, with the person watching.

    Returns None when nothing was run, so the summary says "not applicable" rather
    than "zero datasets aligned".
    """
    if mode != "end" or dry_run or not queue_document.get("count"):
        return None
    # The batch's own interpreter: it is the one with the GUI dependencies.
    command = [python, str(FAILSAFE_LAUNCHER), str(queue_path)]
    print(
        f"\n[parallel] {queue_document['count']} dataset(s) need manual alignment; "
        f"opening the fail-safe GUI.\n[parallel] $ {' '.join(command)}",
        flush=True,
    )
    # stdio is inherited: this is an interactive step, and its progress belongs on
    # the terminal the batch was started from.
    returncode = subprocess.call(command)
    record = {
        "mode": mode,
        "command": command,
        "returncode": returncode,
        "queued": queue_document["count"],
        "summary": str(queue_path.parent / FAILSAFE_SUMMARY_NAME),
        "completed": [],
    }
    summary_path = queue_path.parent / FAILSAFE_SUMMARY_NAME
    if summary_path.is_file():
        try:
            document = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = {}
        record["completed"] = sorted(
            entry.get("name") for entry in document.get("datasets", [])
            if entry.get("status") == "completed"
        )
    return record


def _batch_completed_by_hand(results, planned, queue_document, failsafe, *,
                             interrupted: bool) -> bool:
    """Whether the manual pass turned this batch into a complete one.

    Only when nothing else went wrong: every planned dataset ran, the only failures
    were the ones Stage 2 could not align, and the fail-safe finished all of them.
    A dataset that failed in Stage 4 is still a failed batch, GUI or no GUI.
    """
    if interrupted or len(results) != len(planned):
        return False
    failed = {result["name"] for result in results if result["returncode"] != 0}
    if not failed:
        return False
    queued = {case.get("name") for case in queue_document.get("datasets", [])}
    completed = set(failsafe.get("completed", []))
    return failed <= queued and failed <= completed


def _write_manual_alignment_queue(
    path: Path,
    *,
    manifest_path: Path,
    output_root: Path,
    results,
    enabled: bool = True,
):
    """Write the queue file, empty but explicit when --stage2-nomanual is set.

    `automatic_alignment_failed` marks every dataset Stage 2 could not align -- the
    relevel phase is driven off it -- and `requires_manual_alignment` the subset a
    person may take over when manual fallback is enabled, for any supported pair.
    Switching manual alignment off is applied here, at the point where a failure would
    be handed on to a person.
    """
    cases = [
        result["manual_alignment_case"]
        for result in results
        if result.get("requires_manual_alignment")
        and isinstance(result.get("manual_alignment_case"), dict)
    ]
    no_manual = sum(1 for result in results
                    if _alignment_failed(result) and not result.get("requires_manual_alignment"))
    document = {
        "schema_version": 1,
        "kind": "pharaoh_manual_alignment_queue",
        "generated_at": datetime.now().isoformat(timespec="microseconds"),
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "enabled": bool(enabled),
        "count": len(cases) if enabled else 0,
        "datasets": cases if enabled else [],
    }
    if not enabled:
        # Say how many were dropped; an empty queue must not read as a clean batch.
        document["excluded_by_stage2_nomanual"] = len(cases)
    if no_manual:
        document["failed_without_manual_fallback"] = no_manual
    _write_summary(path, document)
    return document


def _carry_alignment_failure(result, initial_result):
    """A relevel attempt that could not even launch keeps the initial failure's
    verdicts: Stage 2 failed, and whether a person may take it over."""
    result["automatic_alignment_failed"] = True
    for key in ("alignment_failure_case", "manual_alignment_case"):
        if isinstance(initial_result.get(key), dict):
            result[key] = initial_result[key]
    result["requires_manual_alignment"] = bool(initial_result.get("requires_manual_alignment"))


def _alignment_failed(result) -> bool:
    """Whether Stage 2 failed for this dataset -- what the relevel rounds act on.
    Older result records carry only `requires_manual_alignment`, which then meant
    the same thing."""
    return bool(result.get("automatic_alignment_failed", result.get("requires_manual_alignment")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, 
        description="Run PHARAOH datasets concurrently with separate outputs and logs."
    )
    parser.add_argument("manifest", type=Path, help="JSON dataset manifest")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="batch output directory "
        "(default: <this version dir>/outputs/parallel_runs_<timestamp>)",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="maximum concurrent datasets (default: 4)",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter for each pipeline",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute all stages in every dataset",
    )
    parser.add_argument(
        "--stop-after",
        choices=[str(number) for number in range(1, 9)],
        help="stop every pipeline after this stage",
    )
    parser.add_argument(
        "--qc-images", dest="qc_images", nargs="?", const="all", default=None,
        metavar="GROUPS",
        help="write quality-control images in every dataset, overriding "
             "parameters.json['qc_images']. Bare --qc-images turns on every group; "
             "pass a comma-separated subset of "
             f"{{{','.join(QC_IMAGE_GROUPS)}}} to pick. A dataset can override the "
             "batch setting with its own --qc-images in the manifest's pipeline_args",
    )
    parser.add_argument(
        "--no-qc-images", dest="qc_images", action="store_const", const="none",
        help="write no quality-control images in any dataset",
    )
    # What happens to a dataset whose Stage 2 fails, on two independent axes:
    # relevel (retry at other Stage-1 levels) and manual (enter the manual queue).
    parser.add_argument(
        "--failsafe-mode",
        choices=("end", "instant", "off"),
        default="end",
        help="when to ask a person to align what Stage 2 could not: 'end' (default) "
             "works through manual_alignment_queue.json once every pipeline has "
             "finished, then finishes each aligned dataset; 'instant' lets each "
             "dataset open its own window the moment its Stage 2 fails, while the "
             "rest of the batch keeps running (its relevel rounds are skipped); "
             "'off' only writes the queue, for ./run_pharaoh_failsafe.sh later",
    )
    parser.add_argument(
        "--stage2-nomanual", action="store_true",
        help="do not enter failed datasets into manual_alignment_queue.json: "
             "relevel as usual, then leave them failed",
    )
    parser.add_argument(
        "--stage2-nofallback", action="store_true",
        help="shorthand for --stage2-relevel-rounds 0 --stage2-nomanual: a Stage-2 "
             "failure is final",
    )
    parser.add_argument(
        "--stage4b-tile-topup-max-attempts", type=int, default=None,
        help="rounds of Stage-4b tile top-up per dataset when it runs short of usable "
             "tiles; omitted leaves run_pharaoh.py's default, 0 switches it off",
    )
    parser.add_argument(
        "--stage2-relevel-rounds",
        type=int,
        default=1,
        help="rounds of Stage-1 level retries after a Stage-2 failure. Each "
             "round tries one level coarser and then one level finer. 0 skips "
             "relevelling altogether (default: 1)",
    )
    # Superseded spellings, kept silently so existing invocations keep working.
    parser.add_argument("--alignment-recovery", dest="legacy_relevel_on",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-alignment-recovery", dest="legacy_relevel_off",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--alignment-recovery-rounds", dest="legacy_relevel_rounds",
                        type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--alignment-recovery-max-attempts",
                        dest="legacy_relevel_attempts",
                        type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--threads-per-dataset",
        type=int,
        help="set common BLAS/OpenMP thread limits for each dataset process",
    )
    parser.add_argument(
        "--stage3-jobs",
        type=int,
        default=4,
        help=(
            "hard maximum datasets allowed in Stage 3 at once (default: 4; 0 disables "
            "gating). This is a ceiling, not a target: Stage 3 holds a whole pyramid "
            "level of both slides at once, so the adaptive RAM check normally admits "
            "far fewer"
        ),
    )
    stage3_mode = parser.add_mutually_exclusive_group()
    stage3_mode.add_argument(
        "--stage3-adaptive",
        dest="stage3_adaptive",
        action="store_true",
        default=True,
        help="adapt Stage-3 admission to RAM, CPU, and FIFO queue depth (default)",
    )
    stage3_mode.add_argument(
        "--stage3-fixed",
        dest="stage3_adaptive",
        action="store_false",
        help="use only the hard Stage-3 concurrency limit",
    )
    parser.add_argument(
        "--stage3-memory-per-job-gib",
        type=float,
        default=None,
        help=(
            "estimated Stage-3 peak RSS, applied to every dataset. Omit it and each "
            "dataset derives its own from the pyramid levels it will read, which is "
            "what the peak is dominated by -- a single number cannot fit a manifest "
            "that mixes slide sizes"
        ),
    )
    parser.add_argument(
        "--stage4-jobs",
        type=int,
        default=4,
        help=(
            "hard maximum datasets allowed in memory-heavy Stage 4 "
            "(default: 4; adaptive limits normally admit fewer; 0 disables gating)"
        ),
    )
    stage4_mode = parser.add_mutually_exclusive_group()
    stage4_mode.add_argument(
        "--stage4-adaptive",
        dest="stage4_adaptive",
        action="store_true",
        default=True,
        help="adapt Stage-4 admission to RAM, CPU, and FIFO queue depth (default)",
    )
    stage4_mode.add_argument(
        "--stage4-fixed",
        dest="stage4_adaptive",
        action="store_false",
        help="use only the fixed --stage4-jobs concurrency limit",
    )
    parser.add_argument(
        "--stage4-memory-per-job-gib",
        type=float,
        default=5.0,
        help="estimated Stage-4 peak RSS per dataset (default: 5 GiB)",
    )
    parser.add_argument(
        "--stage4-memory-reserve-gib",
        type=float,
        default=4.0,
        help="RAM that must remain available (default: 4 GiB)",
    )
    parser.add_argument(
        "--stage4-cpu-threshold",
        type=float,
        default=80.0,
        help="base total-CPU admission ceiling (default: 80%%)",
    )
    parser.add_argument(
        "--stage4-queue-cpu-boost",
        type=float,
        default=2.0,
        help="CPU ceiling boost per additional queued dataset (default: 2 points)",
    )
    parser.add_argument(
        "--stage4-max-queue-cpu-boost",
        type=float,
        default=10.0,
        help="maximum queue-driven CPU ceiling boost (default: 10 points)",
    )
    parser.add_argument(
        "--stage4-cpu-bypass-after",
        type=float,
        default=120.0,
        help="ignore CPU after this wait only if Stage 4 is idle (default: 120s)",
    )
    parser.add_argument(
        "--stage4-admission-cooldown",
        type=float,
        default=5.0,
        help="spacing between overlapping Stage-4 admissions (default: 5s)",
    )
    parser.add_argument(
        "--stage4-poll-seconds",
        type=float,
        default=0.5,
        help="adaptive scheduler polling interval (default: 0.5s)",
    )
    # On by default: a run without it cannot answer afterwards how long a stage took
    # or how much memory it held, and the sampling is one RSS read per second.
    parser.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample process-tree RSS and write a stage-coloured memory/time plot per "
             "dataset plus batch_timeline.html for the run (default: on; "
             "--no-monitor switches it off)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="monitor sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and print commands only",
    )
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.qc_images is not None:
        # Validated here rather than in the children: a mistyped group would otherwise
        # reach every dataset and quietly write no QC images, which looks exactly like
        # a run that never asked for any. (`--qc-images` takes an optional value, so
        # put the manifest first: "--qc-images batch.json" eats the manifest as the
        # group list, and argparse then reports the missing positional.)
        try:
            args.qc_images = normalize_qc_images_setting(args.qc_images)
        except ValueError as exc:
            parser.error(f"--qc-images: {exc}")
    if args.threads_per_dataset is not None and args.threads_per_dataset < 1:
        parser.error("--threads-per-dataset must be at least 1")
    if args.stage4_jobs < 0:
        parser.error("--stage4-jobs cannot be negative")
    resolve_stage2_fallback(args, parser.error, "parallel")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval must be greater than 0")
    try:
        Stage4SchedulerConfig(
            max_concurrent=max(1, args.stage4_jobs),
            adaptive=args.stage4_adaptive,
            memory_per_job_gib=args.stage4_memory_per_job_gib,
            memory_reserve_gib=args.stage4_memory_reserve_gib,
            cpu_threshold_percent=args.stage4_cpu_threshold,
            queue_cpu_boost_per_waiter=args.stage4_queue_cpu_boost,
            queue_cpu_boost_max=args.stage4_max_queue_cpu_boost,
            cpu_bypass_after_seconds=args.stage4_cpu_bypass_after,
            admission_cooldown_seconds=args.stage4_admission_cooldown,
            poll_seconds=args.stage4_poll_seconds,
        ).validate()
    except ValueError as exc:
        parser.error(str(exc))
    try:
        manifest_path, datasets, _raw = load_manifest(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))

    # Before anything is timed or launched: Stage 1 reads pyramid levels and pixel
    # size through tifffile, so every input has to be an OME-TIFF by this point.
    try:
        datasets, input_conversions = prepare_ome_tiff_inputs(
            datasets, dry_run=args.dry_run
        )
    except ValueError as exc:
        parser.error(str(exc))
    converted = [record for record in input_conversions
                 if record["action"] == "converted"]
    if converted:
        print(f"[pre_process] converted {len(converted)} input(s) in "
              f"{sum(record['seconds'] for record in converted):.1f}s "
              f"(not counted in any pipeline time)", flush=True)

    if args.output_root is None:
        batch_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        output_root = HERE / "outputs" / f"parallel_runs_{batch_id}"
    else:
        output_root = args.output_root.expanduser().resolve()

    effective_stage4_jobs = (
        min(args.stage4_jobs, args.jobs, len(datasets))
        if args.stage4_jobs > 0
        else None
    )
    stage4_slots_dir = (
        output_root / ".stage4_slots"
        if effective_stage4_jobs is not None
        else None
    )
    stage4_scheduler_config = (
        Stage4SchedulerConfig(
            max_concurrent=effective_stage4_jobs,
            adaptive=args.stage4_adaptive,
            memory_per_job_gib=args.stage4_memory_per_job_gib,
            memory_reserve_gib=args.stage4_memory_reserve_gib,
            cpu_threshold_percent=args.stage4_cpu_threshold,
            queue_cpu_boost_per_waiter=args.stage4_queue_cpu_boost,
            queue_cpu_boost_max=args.stage4_max_queue_cpu_boost,
            cpu_bypass_after_seconds=args.stage4_cpu_bypass_after,
            admission_cooldown_seconds=args.stage4_admission_cooldown,
            poll_seconds=args.stage4_poll_seconds,
        )
        if effective_stage4_jobs is not None
        else None
    )

    # Stage 3 mirrors Stage 4's gate with its own slot directory, so the two stages
    # queue independently: one dataset can be sampling tiles while another segments
    # nuclei, but two cannot hold a full pyramid level each at the same time.
    effective_stage3_jobs = (
        min(args.stage3_jobs, args.jobs, len(datasets))
        if args.stage3_jobs > 0
        else None
    )
    stage3_slots_dir = (
        output_root / ".stage3_slots" if effective_stage3_jobs is not None else None
    )
    def stage3_memory_estimate(dataset):
        """Per-dataset Stage-3 estimate, unless the user pinned one for the batch."""
        if args.stage3_memory_per_job_gib is not None:
            return float(args.stage3_memory_per_job_gib)
        return estimate_stage3_peak_gib(dataset.moving, dataset.fixed)

    stage3_scheduler_config = (
        Stage4SchedulerConfig(
            max_concurrent=effective_stage3_jobs,
            adaptive=args.stage3_adaptive,
            memory_per_job_gib=args.stage3_memory_per_job_gib or 8.0,
            memory_reserve_gib=args.stage4_memory_reserve_gib,
            cpu_threshold_percent=args.stage4_cpu_threshold,
            queue_cpu_boost_per_waiter=args.stage4_queue_cpu_boost,
            queue_cpu_boost_max=args.stage4_max_queue_cpu_boost,
            cpu_bypass_after_seconds=args.stage4_cpu_bypass_after,
            admission_cooldown_seconds=args.stage4_admission_cooldown,
            poll_seconds=args.stage4_poll_seconds,
        )
        if effective_stage3_jobs is not None
        else None
    )

    planned = []
    for dataset in datasets:
        run_dir = output_root / f"runs_{dataset.slug}"
        log_path = output_root / f"{dataset.slug}.log"
        command = build_command(
            dataset,
            python=args.python,
            run_dir=run_dir,
            force=args.force,
            stop_after=args.stop_after,
            stage4_slots_dir=stage4_slots_dir,
            stage4_max_concurrent=effective_stage4_jobs,
            stage4_scheduler_config=stage4_scheduler_config,
            stage3_slots_dir=stage3_slots_dir,
            stage3_max_concurrent=effective_stage3_jobs,
            stage3_scheduler_config=(
                replace(stage3_scheduler_config,
                        memory_per_job_gib=stage3_memory_estimate(dataset))
                if stage3_scheduler_config is not None else None
            ),
            qc_images=args.qc_images,
            stage2_nomanual=args.stage2_nomanual,
            stage4b_tile_topup_max_attempts=args.stage4b_tile_topup_max_attempts,
            failsafe=failsafe_mode(args) == "instant",
        )
        planned.append((dataset, command, run_dir, log_path))

    print(
        f"[parallel] datasets={len(planned)} jobs={min(args.jobs, len(planned))} "
        f"stage3_jobs={effective_stage3_jobs or 'unlimited'} "
        f"stage4_jobs={effective_stage4_jobs or 'unlimited'} "
        f"stage4_scheduler="
        f"{('adaptive' if args.stage4_adaptive else 'fixed') if stage4_slots_dir else 'off'} "
        f"stage2_relevel_rounds={args.stage2_relevel_rounds} "
        f"stage2_manual={'off' if args.stage2_nomanual else 'on'} "
        f"monitor={'on' if args.monitor else 'off'} "
        f"qc_images={args.qc_images or 'parameters.json'} output_root={output_root}",
        flush=True,
    )
    for dataset, command, run_dir, log_path in planned:
        print(f"[parallel] {dataset.name}: {_display_command(command)}", flush=True)
        print(f"[parallel] {dataset.name}: log={log_path} run_dir={run_dir}", flush=True)
    if args.dry_run:
        print("[parallel] dry run complete; no processes started", flush=True)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manual_alignment_queue_path = output_root / MANUAL_ALIGNMENT_QUEUE_NAME
    _write_manual_alignment_queue(
        manual_alignment_queue_path,
        manifest_path=manifest_path,
        output_root=output_root,
        results=[],
        enabled=not args.stage2_nomanual,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.platform == "win32":
        # Child output goes through a pipe, where Windows Python falls back to the
        # ANSI code page and fails on the pipeline's non-ASCII text (µm, ±, →).
        env["PYTHONUTF8"] = "1"
    if args.threads_per_dataset is not None:
        for key in THREAD_ENV_VARS:
            env[key] = str(args.threads_per_dataset)

    batch_started_at = datetime.now().isoformat(timespec="microseconds")
    batch_started_perf = time.perf_counter()
    pending = deque(planned)
    relevel_pending = deque()
    relevel_states = {}
    initial_alignment_failures = 0
    phase = "initial"
    running = []
    results = []
    print_lock = threading.Lock()
    interrupted = False
    monitor_active = bool(args.monitor)
    monitor_error = None
    next_monitor_sample = 0.0

    def latest_results():
        return list({result["name"]: result for result in results}.values())

    def refresh_manual_alignment_queue():
        return _write_manual_alignment_queue(
            manual_alignment_queue_path,
            manifest_path=manifest_path,
            output_root=output_root,
            results=latest_results(),
            enabled=not args.stage2_nomanual,
        )

    try:
        while pending or running or relevel_pending or phase == "initial":
            if phase == "initial" and not pending and not running:
                phase = "relevel"
                initial_failures = [
                    result for result in latest_results()
                    if _alignment_failed(result)
                ]
                initial_alignment_failures = len(initial_failures)
                if args.stage2_relevel_rounds > 0:
                    planned_by_name = {
                        dataset.name: (dataset, command, run_dir, log_path)
                        for dataset, command, run_dir, log_path in planned
                    }
                    for result in initial_failures:
                        dataset, command, run_dir, log_path = planned_by_name[
                            result["name"]
                        ]
                        state = Stage2Relevel(
                            dataset=dataset,
                            command=command,
                            run_dir=run_dir,
                            log_path=log_path,
                            initial_result=result,
                            attempts=[],
                        )
                        relevel_states[dataset.name] = state
                        if _next_relevel_levels(
                            dataset, run_dir, args.stage2_relevel_rounds
                        ) is None:
                            result["stage2_relevel"] = _stage2_relevel_details(
                                state,
                                status="unavailable",
                                reason="no alternative Stage-1 pyramid levels remain",
                            )
                            print(
                                f"[relevel] {dataset.name}: no alternative usable "
                                f"pyramid level; " + (
                                    "no fallback left (--stage2-nomanual)"
                                    if args.stage2_nomanual
                                    else "keeping the manual-alignment case"),
                                flush=True,
                            )
                        else:
                            result["stage2_relevel"] = _stage2_relevel_details(
                                state, status="queued"
                            )
                            relevel_pending.append(state)
                    if relevel_pending:
                        print(
                            f"[relevel] normal batch finished; starting fair "
                            f"high-resolution relevelling for "
                            f"{len(relevel_pending)} dataset(s), max_attempts="
                            f"{2 * args.stage2_relevel_rounds}",
                            flush=True,
                        )
                elif initial_failures:
                    # Nothing was tried, so say so on each result rather than
                    # leaving the summary silent about why.
                    for result in initial_failures:
                        result["stage2_relevel"] = {
                            "status": "off",
                            "recovered": False,
                            "initial_returncode": result["returncode"],
                            "initial_failure": (result.get("alignment_failure_case")
                                                or result.get("manual_alignment_case", {})).get("failure"),
                            "attempts": [],
                            "reason": "relevelling off (--stage2-relevel-rounds 0)",
                        }
                    print(
                        f"[relevel] off (--stage2-relevel-rounds 0); "
                        f"{len(initial_failures)} failed dataset(s) go straight to "
                        + ("no fallback (--stage2-nomanual)"
                           if args.stage2_nomanual else "the manual queue"),
                        flush=True,
                    )
                if not relevel_pending:
                    break

            launch_queue = pending if phase == "initial" else relevel_pending
            while launch_queue and len(running) < args.jobs:
                relevel_state = None
                relevel_attempt = None
                if phase == "relevel":
                    if any(_alignment_stage_in_progress(item) for item in running):
                        break
                    relevel_state = relevel_pending.popleft()
                    levels = _next_relevel_levels(
                        relevel_state.dataset, relevel_state.run_dir,
                        args.stage2_relevel_rounds, relevel_state.attempts,
                    )
                    if levels is None:
                        previous = next(
                            result for result in reversed(results)
                            if result["name"] == relevel_state.dataset.name
                        )
                        previous["stage2_relevel"] = _stage2_relevel_details(
                            relevel_state,
                            status="exhausted",
                            reason=("Stage-2 relevel rounds set to 0"
                                        if args.stage2_relevel_rounds == 0
                                        else "no alternative Stage-1 pyramid levels remain"),
                        )
                        print(
                            f"[relevel] {relevel_state.dataset.name}: "
                            "all finer pyramid levels exhausted",
                            flush=True,
                        )
                        continue
                    relevel_attempt = {
                        "attempt": len(relevel_state.attempts) + 1,
                        **levels,
                        "status": "running",
                    }
                    relevel_state.attempts.append(relevel_attempt)
                    dataset = relevel_state.dataset
                    run_dir = relevel_state.run_dir
                    log_path = relevel_state.log_path
                    command = list(relevel_state.command)
                    command.extend((
                        "--stage1-he-level", str(levels["he_level"]),
                        "--stage1-moving-level", str(levels["moving_level"]),
                    ))
                else:
                    dataset, command, run_dir, log_path = pending.popleft()
                try:
                    item = _launch(
                        dataset, command, run_dir, log_path, env, print_lock,
                        relevel_attempt=relevel_attempt,
                    )
                except OSError as exc:
                    now = datetime.now().isoformat(timespec="microseconds")
                    print(
                        f"[parallel] {dataset.name}: launch failed: {exc}",
                        file=sys.stderr,
                    )
                    result = {
                        "name": dataset.name,
                        "returncode": 127,
                        "started_at": now,
                        "finished_at": now,
                        "elapsed_seconds": 0.0,
                        "run_dir": str(run_dir),
                        "log": str(log_path),
                        "command": command,
                        "launch_error": str(exc),
                    }
                    if relevel_state is not None:
                        relevel_attempt.update({
                            "status": "launch_failed",
                            "returncode": 127,
                            "finished_at": now,
                        })
                        _carry_alignment_failure(result, relevel_state.initial_result)
                        result["stage2_relevel"] = _stage2_relevel_details(
                            relevel_state,
                            status="launch_failed",
                            reason=str(exc),
                        )
                    results.append(result)
                    if relevel_state is not None:
                        refresh_manual_alignment_queue()
                    continue
                running.append(item)
                if relevel_attempt is None:
                    print(
                        f"[parallel] {dataset.name}: started pid={item.process.pid}",
                        flush=True,
                    )
                else:
                    relevel_attempt["started_at"] = item.started_at
                    print(
                        f"[relevel] {dataset.name}: attempt "
                        f"{relevel_attempt['attempt']} started pid="
                        f"{item.process.pid} HE level "
                        f"{relevel_attempt['previous_he_level']}->"
                        f"{relevel_attempt['he_level']}, moving level "
                        f"{relevel_attempt['previous_moving_level']}->"
                        f"{relevel_attempt['moving_level']}, "
                        f"waiting={len(relevel_pending)}",
                        flush=True,
                    )

            monitor_now = time.monotonic()
            if monitor_active and running and monitor_now >= next_monitor_sample:
                try:
                    sample_running_processes(running)
                except RuntimeError as exc:
                    monitor_error = str(exc)
                    monitor_active = False
                    print(f"[parallel] monitor disabled: {exc}", file=sys.stderr)
                next_monitor_sample = monitor_now + args.monitor_interval

            finished = []
            for item in running:
                returncode = item.process.poll()
                if returncode is not None:
                    result = _finish(
                        item,
                        returncode,
                        monitor=args.monitor and monitor_error is None,
                        monitor_interval=args.monitor_interval,
                    )
                    if item.relevel_attempt is not None:
                        state = relevel_states[item.dataset.name]
                        attempt = item.relevel_attempt
                        attempt.update({
                            "returncode": int(returncode),
                            "finished_at": result["finished_at"],
                            "elapsed_seconds": result["elapsed_seconds"],
                        })
                        if _alignment_failed(result):
                            attempt["status"] = "alignment_failed"
                            more_levels = _next_relevel_levels(
                                item.dataset, item.run_dir,
                                args.stage2_relevel_rounds, state.attempts,
                            )
                            if more_levels is not None:
                                relevel_pending.append(state)
                                relevel_status = "queued"
                                relevel_reason = None
                                print(
                                    f"[relevel] {item.dataset.name}: attempt "
                                    f"{attempt['attempt']} still failed; "
                                    f"requeued behind {len(relevel_pending) - 1} "
                                    "waiting dataset(s)",
                                    flush=True,
                                )
                            else:
                                relevel_status = "exhausted"
                                relevel_reason = (
                                    "every relevel round was tried"
                                    if more_levels is None
                                    else "no alternative Stage-1 pyramid levels remain"
                                )
                                print(
                                    f"[relevel] {item.dataset.name}: "
                                    f"{relevel_reason}; " + (
                                        "no fallback left (--stage2-nomanual)"
                                        if args.stage2_nomanual
                                        else "manual alignment required"),
                                    flush=True,
                                )
                            result["stage2_relevel"] = _stage2_relevel_details(
                                state,
                                status=relevel_status,
                                reason=relevel_reason,
                            )
                            for key in ("alignment_failure_case", "manual_alignment_case"):
                                case = result.get(key)
                                if isinstance(case, dict):
                                    case["failure"]["stage2_relevel"] = result["stage2_relevel"]
                        elif returncode == 0:
                            attempt["status"] = "succeeded"
                            result["stage2_relevel"] = _stage2_relevel_details(
                                state, status="recovered"
                            )
                            print(
                                f"[relevel] {item.dataset.name}: automatic "
                                f"alignment recovered on attempt {attempt['attempt']}",
                                flush=True,
                            )
                        else:
                            attempt["status"] = "pipeline_failed"
                            result["stage2_relevel"] = _stage2_relevel_details(
                                state, status="pipeline_failed"
                            )
                    results.append(result)
                    if result.get("requires_manual_alignment"):
                        queue_document = refresh_manual_alignment_queue()
                        print(
                            f"[parallel] {item.dataset.name}: queued for manual "
                            f"alignment ({queue_document['count']} total) -> "
                            f"{manual_alignment_queue_path}",
                            flush=True,
                        )
                    elif item.relevel_attempt is not None:
                        refresh_manual_alignment_queue()
                    state = "OK" if returncode == 0 else f"FAILED ({returncode})"
                    print(f"[parallel] {item.dataset.name}: {state}", flush=True)
                    monitor_result = result.get("monitor", {})
                    if monitor_result.get("plot"):
                        print(
                            f"[parallel] {item.dataset.name}: monitor plot="
                            f"{monitor_result['plot']}",
                            flush=True,
                        )
                    elif monitor_result.get("error"):
                        print(
                            f"[parallel] {item.dataset.name}: monitor output failed: "
                            f"{monitor_result['error']}",
                            file=sys.stderr,
                        )
                    finished.append(item)
            for item in finished:
                running.remove(item)
            if running and not finished:
                time.sleep(min(0.1, args.monitor_interval) if monitor_active else 0.1)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[parallel] interrupt received; terminating active pipelines ...",
            file=sys.stderr,
        )
        for item in running:
            _stop_process_group(item.process, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            item.process.poll() is None for item in running
        ):
            time.sleep(0.1)
        for item in running:
            if item.process.poll() is None:
                _stop_process_group(item.process, _KILL_SIGNAL)
        for item in running:
            results.append(_finish(
                item,
                item.process.wait(),
                monitor=args.monitor and monitor_error is None,
                monitor_interval=args.monitor_interval,
            ))

    batch_finished_at = datetime.now().isoformat(timespec="microseconds")
    results_by_name = {result["name"]: result for result in results}
    ordered_results = [
        results_by_name[dataset.name]
        for dataset, *_rest in planned
        if dataset.name in results_by_name
    ]
    all_succeeded = not interrupted and len(ordered_results) == len(planned) and all(
        result["returncode"] == 0 for result in ordered_results
    )
    manual_alignment_queue = _write_manual_alignment_queue(
        manual_alignment_queue_path,
        manifest_path=manifest_path,
        output_root=output_root,
        results=ordered_results,
        enabled=not args.stage2_nomanual,
    )
    summary = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "started_at": batch_started_at,
        "finished_at": batch_finished_at,
        "elapsed_seconds": round(time.perf_counter() - batch_started_perf, 6),
        # Input conversion finishes before batch_started_perf, so its cost is reported
        # here rather than folded into elapsed_seconds or any dataset's own time.
        "input_conversions": input_conversions,
        "jobs": min(args.jobs, len(planned)),
        "stage4_jobs_requested": args.stage4_jobs,
        "stage4_jobs_effective": effective_stage4_jobs,
        "stage3_slots_dir": str(stage3_slots_dir) if stage3_slots_dir else None,
        "stage4_slots_dir": str(stage4_slots_dir) if stage4_slots_dir else None,
        "stage4_scheduler": (
            asdict(stage4_scheduler_config)
            if stage4_scheduler_config is not None
            else None
        ),
        "threads_per_dataset": args.threads_per_dataset,
        "monitor": bool(args.monitor),
        "monitor_interval_seconds": args.monitor_interval if args.monitor else None,
        "monitor_error": monitor_error,
        "stage2_relevel": {
            "enabled": args.stage2_relevel_rounds > 0,
            "relevel_rounds_per_dataset": args.stage2_relevel_rounds,
            "manual_queue_enabled": not args.stage2_nomanual,
            "initial_failures": initial_alignment_failures,
            "datasets_recovered": sum(
                1 for result in ordered_results
                if result.get("stage2_relevel", {}).get("recovered")
            ),
            "datasets_still_failed": sum(
                1 for result in ordered_results
                if _alignment_failed(result)
            ),
            "attempts": sorted(
                (
                    {"name": state.dataset.name, **dict(attempt)}
                    for state in relevel_states.values()
                    for attempt in state.attempts
                ),
                key=lambda attempt: attempt.get(
                    "started_at", attempt.get("finished_at", "")
                ),
            ),
        },
        "manual_alignment_queue": {
            "path": str(manual_alignment_queue_path),
            "count": manual_alignment_queue["count"],
        },
        "interrupted": interrupted,
        "all_succeeded": all_succeeded,
        "datasets": ordered_results,
    }
    summary_path = output_root / "parallel_summary.json"
    _write_summary(summary_path, summary)
    print(f"[parallel] summary={summary_path}", flush=True)

    # After the summary is on disk, so the launcher can resume each dataset with the
    # exact command this batch ran it with.
    failsafe = run_failsafe_after_batch(
        manual_alignment_queue_path,
        manual_alignment_queue,
        mode=failsafe_mode(args),
        dry_run=args.dry_run,
        python=args.python,
    )
    if failsafe is not None:
        summary["fail_safe_manual_alignment"] = failsafe
        if _batch_completed_by_hand(ordered_results, planned, manual_alignment_queue,
                                    failsafe, interrupted=interrupted):
            # Every pipeline that failed did so only for want of an initial
            # alignment, and every one of those has now finished by hand.
            all_succeeded = True
            summary["all_succeeded"] = True
            summary["all_succeeded_after_manual_alignment"] = True
        _write_summary(summary_path, summary)
        print(
            f"[parallel] fail-safe manual alignment finished "
            f"({len(failsafe['completed'])}/{failsafe['queued']} dataset(s) completed); "
            f"summary={failsafe['summary']}",
            flush=True,
        )

    if args.monitor:
        # The per-dataset memory plots say how much each one used; this says what the
        # machine was doing at any moment -- which datasets held a slot, and which
        # were queued behind one. It reads the summary just written plus the stage
        # samples, so it never blocks the batch from reporting its result.
        try:
            from batch_timeline import render as render_batch_timeline

            print(f"[parallel] timeline={render_batch_timeline(output_root)}", flush=True)
        except Exception as exc:
            print(f"[parallel] timeline unavailable ({exc})", flush=True)
    return 0 if all_succeeded else (130 if interrupted else 1)


if __name__ == "__main__":
    raise SystemExit(main())
