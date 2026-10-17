#!/usr/bin/env python3
# ============================================================
# run_pharaoh.py - Headless end-to-end pipeline (Stages 1-8)
#
# Fully automatic counterpart of the two GUI pipelines in `scripts/`:
# no manual selection, no manual alignment. Stage 2 uses
# automatic feature registration instead of the interactive overlay:
# ORB first, then SIFT if the ORB fit fails its quality gate.
#
#   DAPI -> H&E:  python run_pharaoh.py --pair dapi-he --moving <dapi> --fixed <he>
#   H&E  -> H&E:  python run_pharaoh.py --pair he-he --moving <he-fg> --fixed <he-bg>
#   DAPI -> DAPI: python run_pharaoh.py --pair dapi-dapi --moving <a> --fixed <b>
#                  [--moving-channel C] [--fixed-channel C]
#
# Stages:
#   1  read images + build masks         (cli_stage1.py)
#   2  automatic ORB->SIFT alignment     (cli_stage2_orb.py)
#   3  tile sampling + extraction        (cli_stage3.py)
#   4  nuclei masks/standout/patches     engine 4a -> 4b -> 4c
#   5  final alignment from nuclei pairs (cli_stage5.py)
#   6  final-alignment display outputs      (cli_stage6.py)
#   7  dense DAPI--H&E nucleus-pair export (engine_dapi, DAPI mode only)
#   8  high-confidence-pair refinement     (engine_dapi, DAPI mode only)
#
# Each stage is skipped if its output already exists (use --force to
# recompute). Outputs land in <run-dir> (default: ./outputs/runs_<timestamp>).
# ============================================================
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
PRE_PROCESS = SCRIPTS / "pre_process"
if str(PRE_PROCESS) not in sys.path:
    sys.path.insert(0, str(PRE_PROCESS))
from windows_utf8 import ensure_utf8_mode
ensure_utf8_mode()  # Windows: re-run under -X utf8; no-op elsewhere


from cli_common import finalize_pipeline_timing, log_stage_event
from run_artifacts import (INITIAL_ALIGNMENT, STAGE_ORDER, clear_from_stage,
                           clear_stage_products, initial_alignment_path, stage_products)
from stage2_relevel import (
    pyramid_top_levels,
    relevel_pairs,
    resolve_stage2_fallback,
)
from ensure_ome_tiff import ensure_ome_tiff
from my_utils import (
    PAIR_ENV_VAR,
    QC_IMAGE_GROUPS,
    QC_IMAGES_ENV_VAR,
    cli_params_block,
    normalize_qc_images_setting,
    qc_images_enabled,
)
import run_roles
import stage4b_modes
from patch_qc import patch_images_requested, patch_images_ready
import qc_artifacts
from scheduling import Stage4SchedulerConfig, stage4_admission
from stage3_memory import estimate_stage3_peak_gib
from monitoring import monitor_current_process

PY = sys.executable
FAILSAFE_LAUNCHER = SCRIPTS / "gui_fail_safe" / "fail_safe_gui_manual_align.py"


@contextmanager
def resource_slot(
    slot_dir,
    max_concurrent,
    run_dir,
    poll_seconds=0.5,
    *,
    adaptive=False,
    memory_per_job_gib=5.0,
    memory_reserve_gib=4.0,
    cpu_threshold_percent=80.0,
    queue_cpu_boost_per_waiter=2.0,
    queue_cpu_boost_max=10.0,
    cpu_bypass_after_seconds=120.0,
    admission_cooldown_seconds=5.0,
    label="Stage 4",
):
    """Hold a fair fixed or resource-adaptive filesystem slot.

    Written for Stage 4 and now shared with Stage 3, which measures as the larger
    consumer of the two: it holds a whole pyramid level of both slides in memory at
    once (2.97 GiB for the H&E alone on CRC P1) and peaked at 4.7-6.1 GiB on every
    dataset, above Stage 4's 4.0 GiB.
    """
    if slot_dir is None or max_concurrent is None:
        yield None
        return
    config = Stage4SchedulerConfig(
        max_concurrent=int(max_concurrent),
        adaptive=bool(adaptive),
        memory_per_job_gib=float(memory_per_job_gib),
        memory_reserve_gib=float(memory_reserve_gib),
        cpu_threshold_percent=float(cpu_threshold_percent),
        queue_cpu_boost_per_waiter=float(queue_cpu_boost_per_waiter),
        queue_cpu_boost_max=float(queue_cpu_boost_max),
        cpu_bypass_after_seconds=float(cpu_bypass_after_seconds),
        admission_cooldown_seconds=float(admission_cooldown_seconds),
        poll_seconds=float(poll_seconds),
    )
    with stage4_admission(slot_dir, run_dir, config, label) as admission:
        yield admission


def run(cmd, cwd=None):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(f"[run_pharaoh] step failed (exit {proc.returncode}): {printable}")


def failsafe_manual_alignment(args, run_dir: Path, pair, moving: str, he: str) -> bool:
    """Hand a spent Stage 2 to the person at the terminal, now, and wait for them.

    Returns True when they saved an alignment, in which case this same process
    carries straight on to Stage 3: nothing has to be resumed, because the run
    never ended. The launcher is told --no-resume for that reason.

    A closed window is not an error here -- the caller re-raises the Stage-2
    failure it was already handling.
    """
    if not args.failsafe or args.stage2_nomanual:
        return False
    marker = run_dir / INITIAL_ALIGNMENT
    before = marker.stat().st_mtime if marker.exists() else 0.0
    log_stage_event(run_dir, "pipeline_events", "stage2_failsafe_start",
                    pair=pair.name)
    print("\n########## Stage 2 (fail-safe manual alignment) ##########", flush=True)
    try:
        run([
            PY, FAILSAFE_LAUNCHER,
            "--run-dir", run_dir,
            "--pair", pair.name,
            "--moving", moving,
            "--fixed", he,
            "--no-resume",
        ])
    except SystemExit as exc:
        print(f"[run_pharaoh] fail-safe manual alignment did not finish: {exc}",
              flush=True)
    after = marker.stat().st_mtime if marker.exists() else 0.0
    saved = after > 0.0 and after != before
    log_stage_event(
        run_dir, "pipeline_events",
        "stage2_failsafe_saved" if saved else "stage2_failsafe_cancelled",
    )
    print(
        "[run_pharaoh] manual alignment saved; continuing at Stage 3" if saved else
        "[run_pharaoh] no manual alignment was saved",
        flush=True,
    )
    return saved


def stage1_completion_marker(run_dir, pair, qc_extras=False):
    """Keep the metadata resume contract; require both role extras for QC."""
    required = [run_dir / "images_info.json"]
    if qc_extras:
        for role in (run_roles.MOVING, run_roles.FIXED):
            side = run_roles.image_role(pair, role, "", 0)
            required.append(qc_artifacts.stage1_extra(run_dir, side))
    return next((path for path in required if not path.exists()), required[0])


def late_qc_step(name, missing, command, cwd=None, incomplete=None):
    """Regenerate a QC group a resumed run asks for and does not hold.

    `missing()` names what is absent (paths or group names) and must be empty after
    `command` has run; the command is that stage's QC-only mode, which draws from the
    saved state and rewrites nothing a later stage reads. Nothing runs when nothing is
    missing, so an ordinary resume with QC on costs no process. A group the stage
    reports as drawn short of its inputs (`incomplete()` maps it to the reason) is a
    warning, not a failure: the run goes on, the group stays missing, and the next
    resume tries again -- when the inputs are back, it completes.
    """
    absent = missing()
    if not absent:
        return False
    print(f"\n########## {name} ##########", flush=True)
    print(f"[run_pharaoh] missing: {', '.join(str(item) for item in absent)}", flush=True)
    run(command, cwd=cwd)
    still = missing()
    if still:
        notes = incomplete() if incomplete is not None else {}
        unexplained = [item for item in still if str(item) not in notes]
        if unexplained:
            raise SystemExit(f"[run_pharaoh] {name} finished but still missing: "
                             f"{', '.join(str(item) for item in unexplained)}")
        for item in still:
            print(f"[run_pharaoh] WARN {name}: {item} incomplete -- {notes[str(item)]}; it "
                  f"will be drawn again on the next resume", flush=True)
        print(f"### {name}: INCOMPLETE", flush=True)
        return True
    print(f"### {name}: OK", flush=True)
    return True


def _last_stage4b_level(run_dir: Path):
    """(level, all_masked) Stage 4b shipped with: the last attempt on record, else the
    level the anchors say, else the first rung."""
    try:
        events = json.loads((run_dir / "pipeline_times.json").read_text()).get("pipeline_events", [])
        for record in reversed(events):
            if isinstance(record, dict) and record.get("event") == LADDER_ATTEMPT_EVENT:
                return str(record["level"]), bool(record.get("all_masked", False))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        records = json.loads((run_dir / "tiles" / "standout_nuclei.json").read_text())
        level = records[0]["meta"]["level"] if records else "nuclei_pairs"
        return str(level), False
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return "nuclei_pairs", False


RUN_PARAMETERS = "run_parameters.json"


def parameter_block_with_cli_overrides(block: dict, stage4b_anchor_mode=None):
    """The parameter block this run actually uses, without mutating the shared file."""
    effective = copy.deepcopy(block)
    if stage4b_anchor_mode is not None:
        stage4b = dict(effective.get("stage4b", {}))
        stage4b[stage4b_modes.PARAM_KEY] = stage4b_modes.normalize_mode(
            stage4b_anchor_mode
        )
        effective["stage4b"] = stage4b
    return effective


def record_run_parameters(run_dir: Path, block_key: str, block: dict, source: Path,
                          overrides=None):
    """Snapshot the parameter block a run starts under, so a resume can tell whether
    the stages on disk and the ones still to run would read the same numbers."""
    digest = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
    (Path(run_dir) / RUN_PARAMETERS).write_text(json.dumps({
        "params_block": block_key,
        "source": str(source),
        "overrides": dict(overrides or {}),
        "sha256": digest,
        "block": block,
    }, indent=2))
    return digest


def check_resume_parameters(run_dir: Path, block_key: str, block: dict, source: Path,
                            overrides=None):
    """Refuse to resume a run whose parameter block has changed since it started.

    A stage that ran under one set of numbers and a stage that would now run under
    another do not make one result; the resumed run would carry both without saying so.
    A run from before this snapshot existed gets one now and continues.
    """
    path = Path(run_dir) / RUN_PARAMETERS
    if not path.exists():
        record_run_parameters(run_dir, block_key, block, source, overrides=overrides)
        print(f"[run_pharaoh] no {RUN_PARAMETERS} in this run directory (made before it was "
              f"kept); recorded the current parameters and resuming", flush=True)
        return
    try:
        saved = json.loads(path.read_text())
        saved_block = saved["block"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[run_pharaoh] {path} cannot be read ({exc}); use --force")
    if saved_block == block:
        return
    changed = sorted(key for key in set(saved_block) | set(block)
                     if saved_block.get(key) != block.get(key))
    raise SystemExit(
        f"[run_pharaoh] the effective parameters for ['{block_key}'] have changed "
        f"since this run started "
        f"(keys: {', '.join(changed)}); the stages on disk were computed under the old "
        f"values. Check {source} and this run's CLI overrides. Use --force to rerun "
        f"the whole run under the new ones, or restore them."
    )


def check_missing_run_metadata(run_dir: Path):
    """A directory with stage products but no input binding is not a fresh run.

    Inspect all known pairs: without images_info.json, the requested pair cannot
    tell us which pair left the products. Logs/monitoring/parameter snapshots alone
    are allowed, since a launcher can create those before Stage 1 writes anything.
    This check is read-only and must precede recording the new parameter snapshot.
    """
    products = sorted({
        path
        for pair in run_roles.PAIRS.values()
        for stage in STAGE_ORDER
        for path in stage_products(run_dir, stage, pair.homography_stem)
    })
    if products:
        examples = ", ".join(str(path.relative_to(run_dir)) for path in products[:5])
        if len(products) > 5:
            examples += ", ..."
        raise SystemExit(
            f"[run_pharaoh] {run_dir} contains existing stage products ({examples}), "
            "but images_info.json is missing; their inputs and parameters cannot be verified."
            "\nUse a new --run-dir, or --force to start this one over."
        )


def check_resume_inputs(run_dir: Path, pair, moving: str, fixed: str):
    """A run directory is bound to the images and pair it was started from.

    Resuming it with another moving or fixed image used to go unnoticed: every stage
    found its artifact and skipped, and the run ended with the new image named on the
    command line and the old one in every result. Refuse unless --force, which starts
    the directory over.
    """
    info_path = run_dir / "images_info.json"
    try:
        saved = run_roles.read(json.loads(info_path.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[run_pharaoh] {info_path} cannot be read ({exc}); "
                         f"use --force to start the run directory over")
    problems = []
    if saved.pair.name != pair.name:
        problems.append(f"pair: run directory holds {saved.pair.name}, requested {pair.name}")
    for role, requested in (("moving", moving), ("fixed", fixed)):
        recorded = getattr(saved, role).path
        if Path(recorded).resolve() != Path(requested).resolve():
            problems.append(f"{role} image: run directory holds {recorded}, requested {requested}")
    if problems:
        raise SystemExit(
            "[run_pharaoh] this run directory was started from different inputs and "
            "cannot be resumed with these:\n  " + "\n  ".join(problems)
            + "\nUse a new --run-dir, or --force to start this one over."
        )


def _clear_before_rerun(name, clear):
    """Under --force a stage's earlier products go before it runs again, so that a
    rerun can only read what it wrote itself (see run_artifacts.STAGE_PRODUCTS)."""
    if clear is None:
        return
    removed = clear()
    if removed:
        print(f"[run_pharaoh] {name}: cleared {len(removed)} earlier product(s) before the "
              f"forced rerun ({', '.join(p.name for p in removed[:4])}"
              f"{', ...' if len(removed) > 4 else ''})", flush=True)


def step(name, artifact: Path, force: bool, fn, clear=None):
    """Run `fn` unless `artifact` already exists (and `force` is off). `clear`, when
    given, removes the stage's earlier products first; pass it only for a forced rerun,
    never for an incremental mode (extras-only, patches-only, a top-up) that reads them."""
    if artifact.exists() and not force:
        print(f"\n### {name}: SKIP (found {artifact.name})", flush=True)
        return
    print(f"\n########## {name} ##########", flush=True)
    _clear_before_rerun(name, clear)
    fn()
    if not artifact.exists():
        raise SystemExit(f"[run_pharaoh] {name} finished but {artifact} is missing")
    print(f"### {name}: OK ({artifact})", flush=True)


def step_glob(name, directory: Path, pattern: str, force: bool, fn, clear=None):
    """Like step() but the output filename is level-dependent, matched by glob."""
    existing = sorted(directory.glob(pattern))
    if existing and not force:
        print(f"\n### {name}: SKIP (found {existing[0].name})", flush=True)
        return
    print(f"\n########## {name} ##########", flush=True)
    _clear_before_rerun(name, clear)
    fn()
    produced = sorted(directory.glob(pattern))
    if not produced:
        raise SystemExit(f"[run_pharaoh] {name} finished but no {pattern} in {directory}")
    print(f"### {name}: OK ({produced[0]})", flush=True)


def _masked_tile_count(run_dir, tiles_dir) -> int:
    """How many tiles Stage 4a has masked, which is what Stage 4b can score.

    Used to notice when a top-up added nothing: Stage 3's pool is finite, and once it is
    exhausted another round masks the same tiles and scores them the same way. Counted
    by the fixed side's token: `*_he_mask.png` was `dapi-he`'s spelling only, and made
    every `he-he` pool look empty.
    """
    from glob import glob
    token = run_roles.read(run_dir).fixed.token
    return len(glob(str(Path(tiles_dir) / f"*_{token}_mask.png")))


def _current_stage4b_tile_shortfall(run_dir: Path, started_at: datetime):
    """Return Stage 4b's tile-shortfall record from this attempt, or None.

    Mirrors `_current_stage2_alignment_failure`: the event has to belong to the try
    that just failed, or a stale record from an earlier attempt would trigger a top-up
    for a failure that had nothing to do with tiles.
    """
    path = run_dir / "pipeline_times.json"
    if not path.exists():
        return None
    try:
        events = json.loads(path.read_text()).get("stage4b_events", [])
    except Exception:
        return None
    for record in reversed(events):
        if not isinstance(record, dict):
            continue
        if record.get("event") != "stage4b_insufficient_tiles":
            continue
        stamp = record.get("time")
        try:
            when = datetime.fromisoformat(str(stamp))
        except (TypeError, ValueError):
            continue
        if when >= started_at:
            return {key: record[key] for key in
                    ("level", "scored_tiles", "good_tiles", "required_tiles")
                    if key in record}
    return None


TOPUP_MIN_TILES = 10


def _stage3_pool_size(run_dir, tiles_dir):
    """How many tile pairs Stage 3 extracted, which is all a top-up can ever draw on.

    None when the tile metadata cannot be read, in which case no cap is applied.
    """
    try:
        roles = run_roles.read(run_dir)
        moving = json.loads((Path(tiles_dir) / f"{roles.moving.token}_tile_info.json").read_text())
        fixed = json.loads((Path(tiles_dir) / f"{roles.fixed.token}_tile_info.json").read_text())
        return len(set(moving).intersection(fixed))
    except Exception:
        return None


def _topup_size(shortfall, masked_now, pool, args):
    """How many tiles to mask for the next attempt, and why; 0 means no top-up.

    Sized from the rung's own yield: with g good tiles among s scored and r required,
    the tiles expected to close the gap are (r - g) * s / g, plus half again as slack,
    never fewer than TOPUP_MIN_TILES. A rung that yielded nothing gets no top-up at
    all -- more tiles of the same slide do not change a pairs level that found no
    pairs (TODO.md item 5: artery_P3D went 10 pairs -> 11 on its whole pool) -- so the
    climb, or the failure, comes now rather than after two rounds. The estimate is
    capped by what Stage 3 left unmasked, and when it exceeds that, everything left
    is taken in one round. `--stage4b-tile-topup-size N` (N > 0) replaces the
    estimate with a fixed N.
    """
    remaining = None if pool is None else int(pool) - int(masked_now)
    if remaining is not None and remaining <= 0:
        return 0, f"Stage 3 has no unmasked tiles left ({masked_now} of {pool} masked)"
    fixed = int(getattr(args, "stage4b_tile_topup_size", 0) or 0)
    if fixed > 0:
        size = fixed if remaining is None else min(fixed, remaining)
        return size, f"fixed --stage4b-tile-topup-size {fixed}"
    scored = int(shortfall.get("scored_tiles") or 0)
    good = int(shortfall.get("good_tiles") or 0)
    required = int(shortfall.get("required_tiles") or 0)
    if good <= 0 or scored <= 0:
        return 0, (f"the rung yielded no good tile on {scored} scored; more tiles of the "
                   f"same slide will not change that")
    need = max(int(math.ceil(1.5 * max(required - good, 1) * scored / good)), TOPUP_MIN_TILES)
    size = need if remaining is None else min(need, remaining)
    return size, (f"yield {good}/{scored} good against {required} required -> {need} more"
                  + ("" if remaining is None else f", {remaining} left unmasked"))


def invalidate_downstream(run_dir: Path, stage: str, why: str):
    """Clear the products of `stage` and everything after it, now, and say why.

    Called at the moment the launcher decides those stages must be recomputed -- a
    forced start, a Stage-2 relevel, a Stage-4b top-up or climb -- rather than when
    each stage happens to rerun, so a run that stops before reaching them leaves no
    earlier result behind for a resume to skip over.
    """
    try:
        stems = [run_roles.read(run_dir).pair.homography_stem]
    except Exception:
        # An explicitly forced run can recover a directory with missing/unreadable
        # metadata. Clear every known pair's matrices, not just token-free products,
        # before an early stop can leave them standing for the next resume.
        stems = [pair.homography_stem for pair in run_roles.PAIRS.values()]
    removed = []
    for stem in stems:
        removed.extend(clear_from_stage(run_dir, stage, stem))
    if removed:
        print(f"[run_pharaoh] {why}: cleared {len(removed)} product(s) of Stage {stage} and "
              f"later ({', '.join(p.name for p in removed[:4])}"
              f"{', ...' if len(removed) > 4 else ''})", flush=True)
    return removed


def _stage4b_attempt(run_dir, tiles_dir, engine, artifact, level, all_masked, force):
    """One Stage-4b process running one level of the ladder.

    Returns (failure, shortfall): the SystemExit if the step failed, and the
    structured tile shortfall this attempt recorded if that is why it failed.
    The failure is kept, not just noted: a bare `raise` after the `except` block has
    no active exception and itself fails with "No active exception to reraise".
    """
    started_at = datetime.now()
    command = [PY, "4b_find_standout_nuclei.py", tiles_dir, "--anchor-mode", level]
    if all_masked:
        command.append("--all-masked")
    # Stage 4b's overlays are gated by `qc_images.stage4b_overlays`, which the engine
    # reads for itself and `--qc-images` carries in the environment. The banner stays
    # "Stage 4b (standout nuclei)" whatever the level: the monitor keys its intervals
    # on it, and a climb is one stage taking longer, not a new one.
    print(f"[run_pharaoh] Stage 4b level: {level}"
          f"{' on every masked tile' if all_masked else ' on the selected tiles'}",
          flush=True)
    failure = None
    try:
        step("Stage 4b (standout nuclei)", artifact, force,
             lambda: run(command, cwd=engine),
             clear=(lambda: clear_stage_products(run_dir, "4b")) if force else None)
    except SystemExit as exc:
        failure = exc
    shortfall = _current_stage4b_tile_shortfall(run_dir, started_at) if failure else None
    return failure, shortfall


LADDER_ATTEMPT_EVENT = "stage4b_ladder_attempt"


def _stage4b_ladder_resume_point(run_dir: Path, anchor_mode: str, masked_now: int):
    """Where a resumed run re-enters the ladder: the rung of the last recorded attempt.

    Every Stage-4b attempt is logged (`stage4b_ladder_attempt`) with the ladder it
    belongs to, its rung, scope, whether it scored every masked tile, the top-ups spent
    so far, and how many tiles were masked when it ran. Resuming re-runs that attempt
    -- not the rungs below it, which the earlier run already showed to be short. The
    record is trusted only while the mask set is the one it saw: a different count
    means Stage 4a was re-run, and the ladder starts from the first rung. Returns
    None when there is nothing to resume from.
    """
    path = run_dir / "pipeline_times.json"
    if not path.exists():
        return None
    try:
        events = json.loads(path.read_text()).get("pipeline_events", [])
    except Exception:
        return None
    for record in reversed(events):
        if not isinstance(record, dict) or record.get("event") != LADDER_ATTEMPT_EVENT:
            continue
        try:
            if str(record.get("anchor_mode")) != str(anchor_mode):
                return None
            if int(record.get("masked_tiles", -1)) != int(masked_now):
                return None
            return {
                "rung": int(record["rung"]),
                "all_masked": bool(record.get("all_masked", False)),
                "topups": int(record.get("topups", 0)),
            }
        except (KeyError, TypeError, ValueError):
            return None
    return None


def climb_stage4b_ladder(args, run_dir, tiles_dir, engine, anchor_mode, artifact):
    """Run Stage 4b up the ladder `stage4b.anchor_mode` names (scripts/stage4b_modes.py).

    Each rung is one Stage-4b process running one level -- nuclei pairs, or density
    architecture anchors -- over one scope. A rung short of tiles says so in a
    structured event. On a `selected` rung the answer is to mask more of what Stage 3
    already extracted and score every masked tile, up to
    --stage4b-tile-topup-max-attempts times or until the pool is spent; then the next
    rung, which discards what the short rung found rather than mixing two alignment
    estimates into one fit. `auto`'s last rung is density over every Stage-3 tile.
    A rung that fails for any other reason fails the run; so does the last rung being
    short. There is no tile-centre floor any more.

    A resumed run re-enters at the rung of its last recorded attempt (see
    `_stage4b_ladder_resume_point`) instead of proving the lower rungs short again;
    --force climbs from the first rung.
    """
    # A finished Stage 4b is finished: say so before touching the ladder, or a run that
    # completed on the density rung would re-enter there on resume and launch an empty
    # `4a --add-tiles all` before `step` got the chance to skip the attempt.
    if artifact.exists() and not args.force:
        print(f"\n### Stage 4b (standout nuclei): SKIP (found {artifact.name})", flush=True)
        return
    rungs = stage4b_modes.ladder(anchor_mode)
    print(f"[run_pharaoh] Stage 4b anchor ladder ({anchor_mode}): "
          + " -> ".join(f"{level} on {scope.replace('_', ' ')}" for level, scope in rungs),
          flush=True)

    def attempt(rung, level, scope, all_masked, topups, force):
        log_stage_event(
            run_dir, "pipeline_events", LADDER_ATTEMPT_EVENT,
            anchor_mode=anchor_mode, rung=int(rung), level=level, scope=scope,
            all_masked=bool(all_masked), topups=int(topups),
            masked_tiles=_masked_tile_count(run_dir, tiles_dir),
        )
        return _stage4b_attempt(run_dir, tiles_dir, engine, artifact, level, all_masked, force)

    start, resumed = 0, None
    if not args.force:
        resumed = _stage4b_ladder_resume_point(
            run_dir, anchor_mode, _masked_tile_count(run_dir, tiles_dir))
        if resumed is not None and 0 < resumed["rung"] < len(rungs) or (
                resumed is not None and resumed["rung"] == 0 and (resumed["all_masked"] or resumed["topups"])):
            start = resumed["rung"]
            level, scope = rungs[start]
            print(f"[run_pharaoh] Stage 4b resumes at rung {start + 1} of {len(rungs)}: "
                  f"{level} on {scope.replace('_', ' ')}"
                  f"{' (every masked tile)' if resumed['all_masked'] else ''}, "
                  f"{resumed['topups']} top-up(s) already spent; the rungs below were "
                  f"short last time (--force climbs from the first)", flush=True)
            log_stage_event(run_dir, "pipeline_events", "stage4b_ladder_resume",
                            rung=start, level=level, scope=scope,
                            all_masked=resumed["all_masked"], topups=resumed["topups"])
        else:
            resumed = None
    force = args.force
    failure = shortfall = None
    for index in range(start, len(rungs)):
        level, scope = rungs[index]
        if index > start:
            print(f"\n[run_pharaoh] Stage 4b ({shortfall.get('level', '?')}) short of "
                  f"tiles after the top-ups: good={shortfall.get('good_tiles')}, "
                  f"needed={shortfall.get('required_tiles')} of "
                  f"{shortfall.get('scored_tiles')} scored; its anchors are discarded "
                  f"and the ladder climbs to {level} on {scope.replace('_', ' ')}",
                  flush=True)
            log_stage_event(run_dir, "pipeline_events", "stage4b_ladder_climb",
                            to_level=level, scope=scope, **shortfall)
            # Everything downstream, if any of it exists, was fitted to anchors that
            # are now gone -- and is gone with them, now, not when it happens to rerun.
            force = args.force = True
            invalidate_downstream(run_dir, "4c", f"Stage 4b climbs to {level}")
        if scope == "all_tiles":
            run([PY, "4a_generate_nuclei_masks.py", tiles_dir, "--add-tiles", "all"],
                cwd=engine)
            failure, shortfall = attempt(index, level, scope, True, 0, force)
        else:
            on_resumed_rung = resumed is not None and index == start
            all_masked = resumed["all_masked"] if on_resumed_rung else False
            tile_topups = resumed["topups"] if on_resumed_rung else 0
            failure, shortfall = attempt(index, level, scope, all_masked, tile_topups, force)
            while failure is not None and shortfall:
                if tile_topups >= args.stage4b_tile_topup_max_attempts:
                    print(f"\n[run_pharaoh] Stage 4b ({level}) still short of usable "
                          f"tiles after {tile_topups} top-up(s)", flush=True)
                    break
                masked_before = _masked_tile_count(run_dir, tiles_dir)
                size, why = _topup_size(
                    shortfall, masked_before, _stage3_pool_size(run_dir, tiles_dir), args)
                if size <= 0:
                    print(f"\n[run_pharaoh] Stage 4b ({level}) short of usable tiles "
                          f"(good={shortfall.get('good_tiles')}, "
                          f"needed={shortfall.get('required_tiles')} of "
                          f"{shortfall.get('scored_tiles')} scored); no top-up: {why}",
                          flush=True)
                    log_stage_event(run_dir, "pipeline_events", "stage4b_tile_topup_skipped",
                                    reason=why, masked_tiles=masked_before, **shortfall)
                    break
                tile_topups += 1
                print(
                    f"\n[run_pharaoh] Stage 4b ({level}) short of usable tiles "
                    f"(good={shortfall.get('good_tiles')}, "
                    f"needed={shortfall.get('required_tiles')} of "
                    f"{shortfall.get('scored_tiles')} scored); topping the tile supply "
                    f"up by {size} ({why}), attempt {tile_topups} of "
                    f"{args.stage4b_tile_topup_max_attempts}",
                    flush=True,
                )
                log_stage_event(
                    run_dir, "pipeline_events", "stage4b_tile_topup",
                    attempt=tile_topups, tiles=int(size), reason=why, **shortfall,
                )
                run([PY, "4a_generate_nuclei_masks.py", tiles_dir,
                     "--add-tiles", str(int(size))], cwd=engine)
                if _masked_tile_count(run_dir, tiles_dir) <= masked_before:
                    # Stage 3's pool is exhausted: another round would mask the same
                    # tiles and score them the same way. Say so rather than spend the
                    # attempts.
                    print(f"\n[run_pharaoh] the tile supply did not grow "
                          f"({masked_before} masked); Stage 3 has no more to give",
                          flush=True)
                    break
                force = args.force = True      # rescore: the candidate set has changed
                invalidate_downstream(run_dir, "4c", "Stage 4b tile top-up")
                failure, shortfall = attempt(index, level, scope, True, tile_topups, force)
        if failure is None:
            return
        if not shortfall:
            raise failure
    raise failure


def _current_stage2_alignment_failure(run_dir: Path, started_at: datetime) -> bool:
    """Recognize a structured ORB/SIFT failure from the current Stage-2 try."""
    try:
        data = json.loads(
            (run_dir / "pipeline_times.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False

    for event in reversed(data.get("stage2_events", [])):
        if not isinstance(event, dict):
            continue
        try:
            event_time = datetime.fromisoformat(str(event.get("time")))
        except (TypeError, ValueError):
            continue
        if event_time < started_at:
            continue
        if event.get("event") == "cli_stage2_orb_done":
            return False
        if event.get("event") == "cli_stage2_orb_failed":
            return True
    return False


def _stage1_relevel_plan(run_dir: Path, mode: str, rounds: int):
    """The Stage-1 level pairs to retry for this run, finer first (see
    `scripts/stage2_relevel.py` for why that order)."""
    relevel_pair = run_roles.coerce_pair(mode)
    moving_key = f"{relevel_pair.key_prefix(run_roles.MOVING)}_level"
    fixed_key = f"{relevel_pair.key_prefix(run_roles.FIXED)}_level"
    try:
        info = json.loads(
            (run_dir / "images_info.json").read_text(encoding="utf-8")
        )
        he_level = info[fixed_key]
        moving_level = info[moving_key]
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError):
        return []
    for value in (he_level, moving_level):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return []
    he_top, moving_top = pyramid_top_levels(info, moving_key, fixed_key)
    return [
        {
            "previous_he_level": he_level,
            "previous_moving_level": moving_level,
            "he_level": pair[0],
            "moving_level": pair[1],
            "direction": "coarser" if pair[0] > he_level or pair[1] > moving_level
                         else "finer",
        }
        for pair in relevel_pairs(
            he_level, moving_level, rounds, he_top, moving_top
        )
    ]


def _record_input_conversions(run_dir: Path, conversions):
    """Note pre-run input conversions in the run's timing file, outside its stages."""
    path = Path(run_dir) / "pipeline_times.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    existing = data.get("input_conversions")
    data["input_conversions"] = (existing if isinstance(existing, list) else []) + list(
        conversions
    )
    path.write_text(json.dumps(data, indent=2))


def _resolve_pair(ap, args):
    """Settle which pair this run is.

    A run is named by what it registers onto what: `--pair dapi-he --moving A
    --fixed B`. 4.10's `--dapi` / `--hefg` / `--he0` / `--he` are gone. They named the
    image by modality and could not make the fixed side a choice, which is why
    DAPI-onto-DAPI was inexpressible, and they left one image with four spellings.
    """
    missing = [name for name, value in
               (("--moving", args.moving), ("--fixed", args.fixed)) if not value]
    if missing:
        ap.error(f"a run needs {' and '.join(missing)}; a pair is "
                 f"--pair <{'|'.join(sorted(run_roles.PAIRS))}> "
                 f"--moving <path> --fixed <path>")
    try:
        pair = run_roles.pair_from_name(args.pair)
    except ValueError as exc:
        ap.error(str(exc))
    if not pair.implemented:
        ap.error(f"pair {pair.name} is not implemented yet ({pair.note}); "
                 f"runnable pairs are {', '.join(run_roles.implemented_pairs())}")
    return pair


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False, description="Headless PHARAOH pipeline (ORB->SIFT auto-alignment).")
    ap.add_argument("--pair", default=run_roles.DEFAULT_PAIR,
                    help="what is registered onto what (default: "
                         f"{run_roles.DEFAULT_PAIR}); one of "
                    + ", ".join(sorted(run_roles.PAIRS)))
    ap.add_argument("--moving", help="moving image path: the one that gets warped")
    ap.add_argument("--fixed", help="fixed image path: the one it is warped onto")
    ap.add_argument("--run-dir", default=None,
                    help="output dir (default ./outputs/runs_<timestamp>)")
    ap.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    ap.add_argument(
        "--monitor", action=argparse.BooleanOptionalAction, default=False,
        help="sample process-tree RSS and write stage-coloured memory/time SVG, CSV "
             "and monitor_summary.json in the run directory (default: off)",
    )
    ap.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="monitor sampling interval in seconds (default: 1.0; requires --monitor)",
    )
    ap.add_argument("--stop-after", type=str, default=None,
                    choices=["1", "2", "3", "4", "5", "6", "7", "8"], help="stop after this stage")
    # Not a mutually exclusive group on purpose: the parallel wrapper puts its own
    # choice on the command line and appends the manifest's `pipeline_args` after it,
    # so a per-dataset override has to be "last one wins" rather than a hard error.
    ap.add_argument(
        "--qc-images", dest="qc_images", nargs="?", const="all", default=None,
        metavar="GROUPS",
        help="write quality-control images for this run, overriding "
             "parameters.json['qc_images']. Bare --qc-images turns on every group; "
             "pass a comma-separated subset of "
             f"{{{','.join(QC_IMAGE_GROUPS)}}} to pick. Stage 6's group "
             "(stage6_overlays) is what writes 6_overlay_manual_*.png, "
             "6_alternating_*.gif and the cached keypoint panels",
    )
    ap.add_argument(
        "--no-qc-images", dest="qc_images", action="store_const", const="none",
        help="write no quality-control images for this run (final overlay excepted; "
             "it is always written)",
    )
    # What to do when Stage 2's automatic alignment fails, on two independent axes:
    # relevel (retry at other Stage-1 pyramid levels) and manual (hand the pair to
    # the interactive GUI / the batch's manual queue).
    ap.add_argument(
        "--stage2-nomanual", action="store_true",
        help="do not hand a Stage-2 failure to manual alignment: relevel as usual, "
             "then fail",
    )
    ap.add_argument(
        "--no-failsafe", dest="failsafe", action="store_false",
        help="do not open the fail-safe manual-alignment GUI when Stage 2 fails; "
             "just fail, leaving the run for a batch's manual queue or a later "
             "./run_pharaoh_failsafe.sh (default: open it)",
    )
    ap.add_argument(
        "--stage2-nofallback", action="store_true",
        help="shorthand for --stage2-relevel-rounds 0 --stage2-nomanual: fail on the "
             "first Stage-2 failure and do nothing else",
    )
    ap.add_argument(
        "--stage4b-anchor-mode",
        choices=list(stage4b_modes.MODES),
        default=None,
        help="override <pair>.stage4b.anchor_mode for this run: auto, nuclei_pairs, "
             "or density. The effective value is recorded in run_parameters.json",
    )
    ap.add_argument(
        "--stage4b-tile-topup-max-attempts", type=int, default=2,
        help="when Stage 4b runs short of usable tiles, mask more of the tiles Stage 3 "
             "already extracted and score again, up to this many times (default: 2; "
             "0 switches it off)",
    )
    ap.add_argument(
        "--stage4b-tile-topup-size", type=int, default=0,
        help="tiles added per top-up attempt. Candidates pass the nuclear-density/"
             "pigment screen, then spatial gap filling uses the masked tiles as "
             "anchors. Default 0 sizes each top-up from the rung's own yield -- the "
             "tiles expected to close the good-tile gap, plus half again, capped by "
             "what Stage 3 left unmasked -- and skips the top-up entirely when the "
             "rung yielded nothing; a positive N adds a fixed N instead",
    )
    ap.add_argument(
        "--stage2-relevel-rounds", type=int, default=1,
        help="rounds of Stage-1 level retries after a Stage-2 failure. Each "
             "round tries one level coarser and then one level finer. 0 skips "
             "relevelling altogether (default: 1)",
    )
    # Superseded spellings, kept silently so existing invocations keep working.
    ap.add_argument("--alignment-recovery", dest="legacy_relevel_on",
                    action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-alignment-recovery", dest="legacy_relevel_off",
                    action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--alignment-recovery-rounds", dest="legacy_relevel_rounds",
                    type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--alignment-recovery-max-attempts", dest="legacy_relevel_attempts",
                    type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stage1-fixed-level", "--stage1-he-level", dest="stage1_he_level", type=int, default=None,
                    help="explicit fixed-image pyramid level for Stage 1")
    ap.add_argument("--stage1-moving-level", type=int, default=None,
                    help="explicit moving-image pyramid level for Stage 1")
    ap.add_argument(
        "--stage1-fixed-channel", "--fixed-channel", "--fixed_channel",
        dest="stage1_fixed_channel", type=int, default=None,
        help="zero-based fixed-image DAPI channel; valid only when the fixed side "
             "of --pair is dapi",
    )
    ap.add_argument(
        "--stage1-moving-channel", "--moving-channel", "--moving_channel",
        dest="stage1_moving_channel", type=int, default=None,
        help="zero-based moving-image DAPI channel; valid only when the moving side "
             "of --pair is dapi",
    )
    ap.add_argument(
        "--stage1-fixed-threshold", "--fixed-threshold",
        dest="stage1_fixed_threshold", type=int, default=None,
        help="explicit fixed-image Stage-1 threshold (default: 240 for H&E; "
             "dtype-aware for DAPI)",
    )
    ap.add_argument(
        "--stage1-moving-threshold", "--moving-threshold",
        dest="stage1_moving_threshold", type=int, default=None,
        help="explicit moving-image Stage-1 threshold (default: 240 for H&E; "
             "dtype-aware for DAPI)",
    )
    ap.add_argument(
        "--stage3-slots-dir",
        default=None,
        help="shared lock directory used to stagger Stage 3 across datasets",
    )
    ap.add_argument(
        "--stage3-max-concurrent",
        type=int,
        default=None,
        help="hard maximum datasets allowed in Stage 3",
    )
    ap.add_argument(
        "--stage3-adaptive",
        action="store_true",
        help="gate Stage 3 using available RAM, total CPU, and FIFO queue depth",
    )
    ap.add_argument(
        "--stage3-memory-per-job-gib",
        type=float,
        default=None,
        help="estimated Stage-3 peak RSS for this dataset (default: derived from the "
             "pyramid levels Stage 3 will read, which is what the peak is dominated "
             "by; pass a number to override)",
    )
    ap.add_argument(
        "--stage3-memory-reserve-gib",
        type=float,
        default=4.0,
        help="RAM that the adaptive scheduler must leave available (default: 4 GiB)",
    )
    ap.add_argument(
        "--stage3-cpu-threshold",
        type=float,
        default=80.0,
        help="base total-CPU admission ceiling (default: 80%%)",
    )
    ap.add_argument(
        "--stage3-queue-cpu-boost",
        type=float,
        default=2.0,
        help="CPU ceiling boost per additional queued dataset (default: 2 points)",
    )
    ap.add_argument(
        "--stage3-max-queue-cpu-boost",
        type=float,
        default=10.0,
        help="maximum queue-driven CPU ceiling boost (default: 10 points)",
    )
    ap.add_argument(
        "--stage3-cpu-bypass-after",
        type=float,
        default=120.0,
        help="ignore CPU after this wait only when no Stage 3 is active (default: 120s)",
    )
    ap.add_argument(
        "--stage3-admission-cooldown",
        type=float,
        default=5.0,
        help="minimum spacing between overlapping Stage-3 admissions (default: 5s)",
    )
    ap.add_argument(
        "--stage3-poll-seconds",
        type=float,
        default=0.5,
        help="adaptive scheduler polling interval (default: 0.5s)",
    )

    ap.add_argument(
        "--stage4-slots-dir",
        default=None,
        help="shared lock directory used to stagger Stage 4 across datasets",
    )
    ap.add_argument(
        "--stage4-max-concurrent",
        type=int,
        default=None,
        help="hard maximum datasets allowed in Stage 4",
    )
    ap.add_argument(
        "--stage4-adaptive",
        action="store_true",
        help="gate Stage 4 using available RAM, total CPU, and FIFO queue depth",
    )
    ap.add_argument(
        "--stage4-memory-per-job-gib",
        type=float,
        default=5.0,
        help="estimated Stage-4 peak RSS per dataset (default: 5 GiB)",
    )
    ap.add_argument(
        "--stage4-memory-reserve-gib",
        type=float,
        default=4.0,
        help="RAM that the adaptive scheduler must leave available (default: 4 GiB)",
    )
    ap.add_argument(
        "--stage4-cpu-threshold",
        type=float,
        default=80.0,
        help="base total-CPU admission ceiling (default: 80%%)",
    )
    ap.add_argument(
        "--stage4-queue-cpu-boost",
        type=float,
        default=2.0,
        help="CPU ceiling boost per additional queued dataset (default: 2 points)",
    )
    ap.add_argument(
        "--stage4-max-queue-cpu-boost",
        type=float,
        default=10.0,
        help="maximum queue-driven CPU ceiling boost (default: 10 points)",
    )
    ap.add_argument(
        "--stage4-cpu-bypass-after",
        type=float,
        default=120.0,
        help="ignore CPU after this wait only when no Stage 4 is active (default: 120s)",
    )
    ap.add_argument(
        "--stage4-admission-cooldown",
        type=float,
        default=5.0,
        help="minimum spacing between overlapping Stage-4 admissions (default: 5s)",
    )
    ap.add_argument(
        "--stage4-poll-seconds",
        type=float,
        default=0.5,
        help="adaptive scheduler polling interval (default: 0.5s)",
    )
    args = ap.parse_args(argv)
    if not math.isfinite(args.monitor_interval) or args.monitor_interval <= 0:
        ap.error("--monitor-interval must be finite and greater than 0")

    pair = _resolve_pair(ap, args)
    if args.stage1_he_level is not None and args.stage1_he_level < 0:
        ap.error("--stage1-he-level must be non-negative")
    if args.stage1_moving_level is not None and args.stage1_moving_level < 0:
        ap.error("--stage1-moving-level must be non-negative")
    if args.stage1_fixed_channel is not None and args.stage1_fixed_channel < 0:
        ap.error("--fixed-channel must be non-negative")
    if args.stage1_moving_channel is not None and args.stage1_moving_channel < 0:
        ap.error("--moving-channel must be non-negative")
    if args.stage1_fixed_channel is not None and pair.fixed != "dapi":
        ap.error(f"--fixed-channel is only valid for a fixed DAPI image; "
                 f"--pair {pair.name} has fixed {pair.fixed.upper()}")
    if args.stage1_moving_channel is not None and pair.moving != "dapi":
        ap.error(f"--moving-channel is only valid for a moving DAPI image; "
                 f"--pair {pair.name} has moving {pair.moving.upper()}")
    if args.stage1_fixed_threshold is not None and args.stage1_fixed_threshold < 0:
        ap.error("--fixed-threshold must be non-negative")
    if args.stage1_moving_threshold is not None and args.stage1_moving_threshold < 0:
        ap.error("--moving-threshold must be non-negative")
    if args.stage4b_anchor_mode is None:
        os.environ.pop(stage4b_modes.ANCHOR_MODE_ENV_VAR, None)
    else:
        # Stage 4b is a child process. Export the same per-run choice the launcher
        # uses for its ladder so the engine metadata and selection agree with it.
        os.environ[stage4b_modes.ANCHOR_MODE_ENV_VAR] = args.stage4b_anchor_mode
    if args.qc_images is not None:
        # Export before any stage runs: every stage is its own process and reads the
        # setting from the environment it inherits, so this one assignment is what
        # makes the choice per-run instead of a shared parameters.json edit.
        try:
            args.qc_images = normalize_qc_images_setting(args.qc_images)
        except ValueError as exc:
            ap.error(f"--qc-images: {exc}")
        os.environ[QC_IMAGES_ENV_VAR] = args.qc_images

    # Stage 1 reads pyramid levels and PhysicalSizeX/Y through tifffile, so a BTF,
    # NDPI, SVS, plain TIFF, JPEG or CZI input is converted to OME-TIFF first, written beside the
    # source. It happens here, before the Stage-3 memory estimate reads those pyramids
    # and well before the pipeline clock starts, so conversion time never lands in a
    # stage duration or in the run total.
    input_conversions = []
    cache = {}
    for field in ("moving", "fixed"):
        original = getattr(args, field)
        if not original:
            continue
        try:
            converted, record = ensure_ome_tiff(original, cache=cache)
        except ValueError as exc:
            ap.error(str(exc))
        setattr(args, field, str(converted))
        if record is not None:
            input_conversions.append(record)
    resolve_stage2_fallback(args, ap.error, "run_pharaoh")
    stage1_level_override = (
        args.stage1_he_level is not None or args.stage1_moving_level is not None
    )
    stage1_channel_override = (
        args.stage1_fixed_channel is not None
        or args.stage1_moving_channel is not None
    )
    if stage1_level_override or stage1_channel_override:
        # A changed Stage-1 coordinate system invalidates every existing
        # downstream artifact, so never resume those artifacts unchanged.
        args.force = True
    if bool(args.stage3_slots_dir) != (args.stage3_max_concurrent is not None):
        ap.error(
            "--stage3-slots-dir and --stage3-max-concurrent must be provided together"
        )
    if args.stage3_max_concurrent is not None and args.stage3_max_concurrent < 1:
        ap.error("--stage3-max-concurrent must be at least 1")
    if args.stage3_adaptive and not args.stage3_slots_dir:
        ap.error("--stage3-adaptive requires --stage3-slots-dir")
    if args.stage3_memory_per_job_gib is None:
        # Derive it from the pyramid levels Stage 3 will read. A constant cannot fit
        # every slide: too low admits a dataset the RAM cannot take, too high can put
        # the admission threshold above the machine's free memory.
        args.stage3_memory_per_job_gib = estimate_stage3_peak_gib(
            args.moving, args.fixed
        )
        if args.stage3_slots_dir:
            print(f"[run_pharaoh] Stage 3 memory estimate: "
                  f"{args.stage3_memory_per_job_gib:.2f} GiB (derived from the tile "
                  f"levels this pair of slides will be read at)", flush=True)
    elif args.stage3_memory_per_job_gib <= 0:
        ap.error("--stage3-memory-per-job-gib must be positive")
    if args.stage3_slots_dir:
        try:
            Stage4SchedulerConfig(
                max_concurrent=args.stage3_max_concurrent,
                adaptive=args.stage3_adaptive,
                memory_per_job_gib=args.stage3_memory_per_job_gib,
                memory_reserve_gib=args.stage3_memory_reserve_gib,
                cpu_threshold_percent=args.stage3_cpu_threshold,
                queue_cpu_boost_per_waiter=args.stage3_queue_cpu_boost,
                queue_cpu_boost_max=args.stage3_max_queue_cpu_boost,
                cpu_bypass_after_seconds=args.stage3_cpu_bypass_after,
                admission_cooldown_seconds=args.stage3_admission_cooldown,
                poll_seconds=args.stage3_poll_seconds,
            ).validate()
        except ValueError as exc:
            ap.error(str(exc))

    if bool(args.stage4_slots_dir) != (args.stage4_max_concurrent is not None):
        ap.error(
            "--stage4-slots-dir and --stage4-max-concurrent must be provided together"
        )
    if args.stage4_max_concurrent is not None and args.stage4_max_concurrent < 1:
        ap.error("--stage4-max-concurrent must be at least 1")
    if args.stage4_adaptive and not args.stage4_slots_dir:
        ap.error("--stage4-adaptive requires --stage4-slots-dir")
    if args.stage4_slots_dir:
        try:
            Stage4SchedulerConfig(
                max_concurrent=args.stage4_max_concurrent,
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
            ap.error(str(exc))

    mode = pair.name
    moving = args.moving
    engine = (SCRIPTS / pair.engine).resolve()
    print(f"[run_pharaoh] pair: {run_roles.describe(pair)}")
    # Every stage is its own process; `cli_params_block` reads the block for this
    # pair from the environment, exactly as the QC-image setting travels.
    os.environ[PAIR_ENV_VAR] = pair.name

    he = str(Path(args.fixed).resolve())
    moving = str(Path(moving).resolve())
    for p in (he, moving):
        if not Path(p).exists():
            ap.error(f"image not found: {p}")
    if not engine.exists():
        ap.error(f"engine dir not found: {engine}")

    # ---- validate the integrated parameters file (read directly by every
    #      stage; no per-engine copy is generated) ----
    top_params = (SCRIPTS / "parameters.json").resolve()
    if not top_params.exists():
        ap.error(f"integrated parameters file not found: {top_params}")
    block_key = pair.params_block
    all_params = json.loads(top_params.read_text())
    if block_key not in all_params:
        ap.error(f"{top_params} has no '{block_key}' block")
    print(f"[run_pharaoh] parameters: {top_params} ['{block_key}']")
    configured_anchor_mode = stage4b_modes.mode_from_block(
        cli_params_block(pair.params_block).get("stage4b", {})
    )
    anchor_mode = args.stage4b_anchor_mode or configured_anchor_mode
    effective_block = parameter_block_with_cli_overrides(
        all_params[block_key], args.stage4b_anchor_mode
    )
    parameter_overrides = (
        {"stage4b.anchor_mode": anchor_mode}
        if args.stage4b_anchor_mode is not None else {}
    )
    if args.stage4b_anchor_mode is None:
        print(f"[run_pharaoh] Stage 4b anchor mode: {anchor_mode} (parameters.json)")
    else:
        print(
            f"[run_pharaoh] Stage 4b anchor mode: {anchor_mode} "
            f"(--stage4b-anchor-mode; parameters.json={configured_anchor_mode})"
        )

    # Run dirs live under version_4.8/outputs/ (not inside the engine folders).
    # The stage scripts read parameters.json from their own engine dir (stage 3
    # via its module dir; 4a/4b/4c/5 from the engine dir too), so the run
    # location is fully independent of where parameters are found.
    run_dir = Path(args.run_dir).resolve() if args.run_dir else \
        HERE / "outputs" / datetime.now().strftime("runs_%Y%m%d%H%M%S")
    starting_fresh_pipeline = args.force or not (run_dir / "images_info.json").exists()
    if not args.force:
        if starting_fresh_pipeline:
            check_missing_run_metadata(run_dir)
        else:
            check_resume_inputs(run_dir, pair, moving, he)
            check_resume_parameters(
                run_dir, block_key, effective_block, top_params,
                overrides=parameter_overrides,
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    if starting_fresh_pipeline:
        record_run_parameters(
            run_dir, block_key, effective_block, top_params,
            overrides=parameter_overrides,
        )
    if args.force:
        # A forced run starts the directory over: every stage's products go now, so a
        # forced run that stops early (--stop-after, a failure) cannot leave the
        # earlier run's later stages standing for a resume to skip over.
        invalidate_downstream(run_dir, "1", "--force")
    with monitor_current_process(run_dir, enabled=args.monitor, interval=args.monitor_interval):
        tiles_dir = run_dir / "tiles"
        pipeline_started = time.perf_counter()
        pipeline_started_at = datetime.now().isoformat(timespec="microseconds")
        if starting_fresh_pipeline:
            log_stage_event(
                run_dir,
                "pipeline_events",
                "pipeline_start",
                pair=mode,
                source="run_pharaoh",
                stage4b_anchor_mode=anchor_mode,
                stage4b_anchor_mode_source=(
                    "cli" if args.stage4b_anchor_mode is not None else "parameters"
                ),
            )
        if input_conversions:
            # Deliberately not a "*_events" list: those are scanned for the pipeline's own
            # start time, and this work finished before the pipeline started.
            _record_input_conversions(run_dir, input_conversions)
        if stage1_level_override:
            log_stage_event(
                run_dir,
                "pipeline_events",
                "stage2_relevel_retry",
                he_level=args.stage1_he_level,
                moving_level=args.stage1_moving_level,
            )

        print(f"[run_pharaoh] pair={mode}  engine={engine.name}  run_dir={run_dir}")
        qc_source = "--qc-images" if args.qc_images is not None else "parameters.json"
        print("[run_pharaoh] qc images: "
              + ", ".join(f"{group}={'on' if qc_images_enabled(group) else 'off'}"
                          for group in QC_IMAGE_GROUPS)
              + f"  (from {qc_source})")
        print(f"[run_pharaoh] fixed  H&E : {he}")
        print(f"[run_pharaoh] moving img : {moving}")

        stop = args.stop_after

        base_stage1_command = [
            PY, SCRIPTS / "cli_stage1.py", "--engine", engine, "--run-dir", run_dir,
            "--pair", mode, "--fixed", he, "--moving", moving,
        ]
        # Keep explicit display thresholds on the base command so Stage-2 level
        # retries use the same settings rather than silently returning to defaults.
        if args.stage1_fixed_threshold is not None:
            base_stage1_command.extend(
                ("--fixed-threshold", str(args.stage1_fixed_threshold))
            )
        if args.stage1_moving_threshold is not None:
            base_stage1_command.extend(
                ("--moving-threshold", str(args.stage1_moving_threshold))
            )
        if args.stage1_fixed_channel is not None:
            base_stage1_command.extend(
                ("--fixed-channel", str(args.stage1_fixed_channel))
            )
        if args.stage1_moving_channel is not None:
            base_stage1_command.extend(
                ("--moving-channel", str(args.stage1_moving_channel))
            )
        stage1_command = list(base_stage1_command)
        if args.stage1_he_level is not None:
            stage1_command.extend(("--he-level", str(args.stage1_he_level)))
        if args.stage1_moving_level is not None:
            stage1_command.extend(("--moving-level", str(args.stage1_moving_level)))
        # With stage1_extras on, the QC images become the completion marker, exactly as
        # the Stage-6 GIF does: a run directory that already holds images_info.json is
        # still missing them, so asking for them runs Stage 1 in its extras-only mode,
        # which draws them from the recorded levels, thresholds and orientation and
        # rewrites nothing -- not a full rerun, which re-chose all three under the
        # stages already on disk.
        stage1_marker = stage1_completion_marker(
            run_dir, pair, qc_images_enabled("stage1_extras"),
        )
        stage1_extras_only = (run_dir / "images_info.json").exists() and not args.force
        step("Stage 1 (read + masks)", stage1_marker, args.force,
             lambda: run(stage1_command + (["--extras-only"] if stage1_extras_only else [])),
             clear=(lambda: clear_stage_products(run_dir, "1")) if args.force else None)
        if stop == "1":
            return

        stage2_command = [
            PY, SCRIPTS / "cli_stage2_orb.py", "--engine", engine,
            "--run-dir", run_dir, "--pair", mode,
        ]
        if args.stage2_nomanual:
            # Do not end a run that has manual alignment switched off by inviting the
            # user to open the manual-alignment GUI.
            stage2_command.append("--no-manual-fallback")
        alignment_retries = 0
        relevel_plan = None
        while True:
            stage2_started_at = datetime.now()
            try:
                step(
                    "Stage 2 (ORB->SIFT alignment)",
                    initial_alignment_path(run_dir),
                    args.force,
                    lambda: run(stage2_command),
                    clear=(lambda: clear_stage_products(run_dir, "2")) if args.force else None,
                )
            except SystemExit:
                if not _current_stage2_alignment_failure(run_dir, stage2_started_at):
                    raise

                # The plan is computed from the *original* levels, so it is stable across
                # attempts even though Stage 1 has since rewritten images_info.json.
                if relevel_plan is None:
                    relevel_plan = _stage1_relevel_plan(
                        run_dir, mode, args.stage2_relevel_rounds
                    )
                levels = (relevel_plan[alignment_retries]
                          if alignment_retries < len(relevel_plan) else None)
                if levels is None:
                    if args.stage2_relevel_rounds == 0:
                        reason = "relevelling off (--stage2-relevel-rounds 0)"
                    elif relevel_plan:
                        reason = "every Stage-1 level pair in the relevel rounds was tried"
                    else:
                        reason = "no alternative Stage-1 pyramid levels remain"
                    if args.stage2_nomanual:
                        reason += "; manual alignment off (--stage2-nomanual)"
                    log_stage_event(
                        run_dir,
                        "pipeline_events",
                        "stage2_relevel_exhausted",
                        attempts=alignment_retries,
                        reason=reason,
                    )
                    print(f"[run_pharaoh] Stage-2 fallback stopped: {reason}", flush=True)
                    if failsafe_manual_alignment(args, run_dir, pair, moving, he):
                        break
                    raise

                alignment_retries += 1
                args.force = True
                invalidate_downstream(run_dir, "3", "Stage-2 relevel")
                log_stage_event(
                    run_dir,
                    "pipeline_events",
                    "stage2_relevel_retry",
                    attempt=alignment_retries,
                    **levels,
                )
                print(
                    f"[run_pharaoh] Stage-2 relevel attempt "
                    f"{alignment_retries}/{len(relevel_plan)} ({levels['direction']}): "
                    f"HE level {levels['previous_he_level']}->{levels['he_level']}, "
                    f"moving level {levels['previous_moving_level']}->"
                    f"{levels['moving_level']}",
                    flush=True,
                )
                retry_stage1_command = list(base_stage1_command)
                retry_stage1_command.extend((
                    "--he-level", str(levels["he_level"]),
                    "--moving-level", str(levels["moving_level"]),
                ))
                step(
                    "Stage 1 (read + masks)",
                    run_dir / "images_info.json",
                    True,
                    lambda: run(retry_stage1_command),
                    clear=lambda: clear_stage_products(run_dir, "1"),
                )
                continue

            if alignment_retries:
                log_stage_event(
                    run_dir,
                    "pipeline_events",
                    "stage2_relevel_succeeded",
                    attempts=alignment_retries,
                )
                print(
                    f"[run_pharaoh] automatic alignment recovered after "
                    f"{alignment_retries} higher-resolution attempt(s)",
                    flush=True,
                )
            break
        if stop == "2":
            return

        stage3_artifact = tiles_dir / f"{pair.token(run_roles.FIXED)}_tile_info.json"
        from run_artifacts import stage3_complete
        stage3_needed = args.force or not stage3_complete(run_dir)

        def run_stage3():
            step("Stage 3 (tiles)", stage3_artifact, stage3_needed,
                 lambda: run([PY, SCRIPTS / "cli_stage3.py", "--engine", engine,
                              "--run-dir", run_dir]),
                 clear=(lambda: clear_stage_products(run_dir, "3")) if args.force else None)

        if stage3_needed and args.stage3_slots_dir:
            print("\n########## Stage 3q (waiting for memory slot) ##########", flush=True)
            with resource_slot(
                args.stage3_slots_dir,
                args.stage3_max_concurrent,
                run_dir,
                args.stage3_poll_seconds,
                adaptive=args.stage3_adaptive,
                memory_per_job_gib=args.stage3_memory_per_job_gib,
                memory_reserve_gib=args.stage3_memory_reserve_gib,
                cpu_threshold_percent=args.stage3_cpu_threshold,
                queue_cpu_boost_per_waiter=args.stage3_queue_cpu_boost,
                queue_cpu_boost_max=args.stage3_max_queue_cpu_boost,
                cpu_bypass_after_seconds=args.stage3_cpu_bypass_after,
                admission_cooldown_seconds=args.stage3_admission_cooldown,
                label="Stage 3",
            ):
                run_stage3()
        else:
            run_stage3()
        if qc_images_enabled("stage3_debug"):
            roles_now = run_roles.read(run_dir)
            late_qc_step("Stage 3 (debug images)",
                         lambda: [p.name for p in qc_artifacts.stage3_debug_missing(run_dir, roles_now)],
                         [PY, SCRIPTS / "cli_stage3.py", "--engine", engine,
                          "--run-dir", run_dir, "--qc-only"])
        if stop == "3":
            return

        # ---- Stage 4: engine helper scripts, run from the engine dir. Stage 4a
        #      persists masks for the spatial Stage-4b subset only; that is its own
        #      default now, for every pair (the GUI's all-tile detection asks for
        #      `--all-tiles` itself). Stage 4b climbs the anchor ladder
        #      `stage4b.anchor_mode` names. Stage 4c writes the centroid table and, only
        #      when QC asks for them, the patch images. The subset and centroid-only
        #      flags used to be passed for `dapi-he` alone, a leftover of the engine 4.10
        #      ran `he-he` on, which made an `he-he` run mask and render everything. A
        #      batch wrapper can provide a shared slot directory to prevent memory-heavy
        #      Stage 4 work from overlapping across too many datasets. ----
        stage4_artifacts = (
            tiles_dir / "nuclei_mask_info.json",
            tiles_dir / "standout_nuclei.json",
            run_dir / "nuclei_patches" / "nuclei_centroids_global.json",
        )
        want_patch_images = patch_images_requested()
        missing_patch_images = want_patch_images and not patch_images_ready(run_dir)
        stage4_needed = args.force or not all(path.exists() for path in stage4_artifacts) or missing_patch_images

        def run_stage4():
            step("Stage 4a (nuclei masks)", stage4_artifacts[0], args.force,
                 lambda: run([PY, "4a_generate_nuclei_masks.py", tiles_dir], cwd=engine),
                 clear=(lambda: clear_stage_products(run_dir, "4a")) if args.force else None)
            climb_stage4b_ladder(args, run_dir, tiles_dir, engine, anchor_mode,
                                 stage4_artifacts[1])
            patch_command = [PY, "4c_get_nuclei_patches.py", tiles_dir]
            if want_patch_images and stage4_artifacts[2].exists() and not args.force:
                patch_command.append("--patches-only")
            elif not want_patch_images:
                patch_command.append("--centroids-only")
            step("Stage 4c (nuclei patches)", stage4_artifacts[2], args.force or missing_patch_images,
                 lambda: run(patch_command, cwd=engine),
                 clear=(lambda: clear_stage_products(run_dir, "4c")) if args.force else None)

        if stage4_needed and args.stage4_slots_dir:
            print(
                "\n########## Stage 4q (waiting for memory slot) ##########",
                flush=True,
            )
            with resource_slot(
                args.stage4_slots_dir,
                args.stage4_max_concurrent,
                run_dir,
                args.stage4_poll_seconds,
                adaptive=args.stage4_adaptive,
                memory_per_job_gib=args.stage4_memory_per_job_gib,
                memory_reserve_gib=args.stage4_memory_reserve_gib,
                cpu_threshold_percent=args.stage4_cpu_threshold,
                queue_cpu_boost_per_waiter=args.stage4_queue_cpu_boost,
                queue_cpu_boost_max=args.stage4_max_queue_cpu_boost,
                cpu_bypass_after_seconds=args.stage4_cpu_bypass_after,
                admission_cooldown_seconds=args.stage4_admission_cooldown,
            ):
                run_stage4()
        else:
            run_stage4()
        if qc_artifacts.stage4b_qc_missing(run_dir):
            level, all_masked = _last_stage4b_level(run_dir)
            late_qc_step("Stage 4b (QC images)",
                         lambda: qc_artifacts.stage4b_qc_missing(run_dir),
                         [PY, "4b_find_standout_nuclei.py", tiles_dir, "--qa-only",
                          "--anchor-mode", level] + (["--all-masked"] if all_masked else []),
                         cwd=engine,
                         incomplete=lambda: qc_artifacts.stage4b_qc_incomplete(run_dir))
        if stop == "4":
            return

        homography_name = f"{pair.homography_stem}.json"
        step("Stage 5 (final alignment)", run_dir / homography_name,
             args.force,
             lambda: run([PY, SCRIPTS / "cli_stage5.py", "--engine", engine,
                          "--run-dir", run_dir, "--pair", mode]),
             clear=(lambda: clear_stage_products(run_dir, "5", pair.homography_stem))
             if args.force else None)
        if stop == "5":
            return

        # ---- Stage 6: build the final overlay and Stage-2-vs-final GIF. ----
        overlay_prefix = "6"  # both dapi and he0 engines emit 6_* stage-6 outputs
        # The alternating GIF is a quality-control output. When stage6_overlays is off it is
        # never written, so the completion marker has to be the overlay instead. Reading the
        # effective setting here is also what makes `--qc-images` resume-safe: a run dir that
        # already holds the final overlay is still missing the GIF, so Stage 6 reruns to
        # produce it without needing --force.
        stage6_marker = (f"{overlay_prefix}_alternating_L*.gif"
                         if qc_images_enabled("stage6_overlays")
                         else f"{overlay_prefix}_overlay_final_L*.png")
        step_glob("Stage 6 (alignment outputs)", run_dir, stage6_marker,
                  args.force,
                  lambda: run([PY, SCRIPTS / "cli_stage6.py", "--engine", engine,
                               "--run-dir", run_dir, "--pair", mode]),
                  clear=(lambda: clear_stage_products(run_dir, "6")) if args.force else None)
        if stop == "6":
            return

        # ---- Stages 7-8: dense nucleus pairing across modalities, for the two mixed
        #      pairs (dapi-he and he-dapi); a same-modality pair has no such pairing. ----
        dense_stages = not pair.same_modality
        if dense_stages:
            step("Stage 7 (dense DAPI--H&E nucleus pairs)",
                 run_dir / "dense_nucleus_pairs" / "summary.json", args.force,
                 lambda: run([PY, SCRIPTS / "cli_stage7.py", "--run-dir", run_dir]),
                 clear=(lambda: clear_stage_products(run_dir, "7")) if args.force else None)
            late_qc_step("Stage 7 (QC images)",
                         lambda: qc_artifacts.stage7_qc_missing(run_dir),
                         [PY, SCRIPTS / "cli_stage7.py", "--run-dir", run_dir, "--qc-only"],
                         incomplete=lambda: qc_artifacts.stage7_qc_incomplete(run_dir))
            if stop == "7":
                return
            step("Stage 8 (high-confidence alignment refinement)",
                 run_dir / "stage8_alignment_refinement.json", args.force,
                 lambda: run([PY, SCRIPTS / "cli_stage8.py", "--run-dir", run_dir]),
                 clear=(lambda: clear_stage_products(run_dir, "8", pair.homography_stem))
                 if args.force else None)
        else:
            print(f"\n### Stages 7-8: SKIP (same-modality pair {mode}: no cross-modality "
                  "nucleus pairs to make)", flush=True)

        print(f"\n[run_pharaoh] DONE. Final homography: "
              f"{run_dir / homography_name}", flush=True)
        if dense_stages:
            print(f"[run_pharaoh] Dense nucleus pairs: "
                  f"{run_dir / 'dense_nucleus_pairs' / 'nucleus_pairs.csv.gz'}", flush=True)
            stage8_summary_path = run_dir / "stage8_alignment_refinement.json"
            if stage8_summary_path.exists():
                stage8_summary = json.loads(stage8_summary_path.read_text())
                outcome = (
                    "promoted"
                    if stage8_summary.get("promoted_to_canonical")
                    else f"Stage {stage8_summary.get('canonical_stage_after', 5)} kept"
                )
                print(f"[run_pharaoh] Stage 8 refinement: {outcome} ({stage8_summary_path})", flush=True)

        if starting_fresh_pipeline:
            total_elapsed = time.perf_counter() - pipeline_started
            timing = finalize_pipeline_timing(
                run_dir,
                total_elapsed_seconds=total_elapsed,
                started_at=pipeline_started_at,
                source="run_pharaoh",
                completed_through_stage=8 if dense_stages else 6,
            )
            stage7_elapsed = timing.get("stage7_elapsed_seconds")
            stage8_elapsed = timing.get("stage8_elapsed_seconds")
            if stage7_elapsed is not None and stage8_elapsed is not None:
                print(
                    f"[run_pharaoh] Timing: Stage 7={float(stage7_elapsed):.2f}s, "
                    f"Stage 8={float(stage8_elapsed):.2f}s, "
                    f"total={total_elapsed:.2f}s",
                    flush=True,
                )
            else:
                print(f"[run_pharaoh] Timing: total={total_elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
