#!/usr/bin/env python3
# ============================================================
# fail_safe_gui_manual_align.py - Work through manual_alignment_queue.json
#
# The last rung of the Stage-2 fallback. A parallel batch relevels a failed
# ORB/SIFT alignment at other pyramid levels first; whatever still fails is
# written to <output-root>/manual_alignment_queue.json for a person. This is the
# launcher that person runs.
#
# Per queued dataset, in its original RUN_DIR:
#
#   1. fail-safe Stage 1 (gui_stage1_orient.py) - both images are loaded from the
#      queue entry, so the only thing asked for is the moving image's
#      orientation (plus thresholds). No file dialogs, nothing to mis-pick.
#   2. Stage 2 manual alignment - the existing engine GUI, with its ORB
#      auto-align switched off (--no-auto-orb): the pair is here precisely
#      because ORB and SIFT rejected it, so re-offering that fit would suggest a
#      wrong answer rather than no answer. Writes initial_alignment.json.
#   3. resume - run_pharaoh.py on the same RUN_DIR. Stages 1 and 2 are skipped
#      because their artifacts now exist, so the run continues at Stage 3 and
#      finishes the dataset. Where the batch recorded the dataset's original
#      command in parallel_summary.json, that command is reused (minus the flags
#      that would recompute Stage 1/2 and destroy the manual alignment).
#
# All the interactive work happens first, for every dataset; the pipelines run
# afterwards. A person is not left waiting on Stage 3-8 of the first dataset
# before they can align the second.
#
# Usage:
#   python fail_safe_gui_manual_align.py <queue.json | output-root>
#   python fail_safe_gui_manual_align.py <queue.json> --name human_crc_p5
#   python fail_safe_gui_manual_align.py <queue.json> --no-resume
#   python fail_safe_gui_manual_align.py --run-dir RUN_DIR --pair dapi-he \
#       --moving DAPI --fixed HE
#
# Exit status is nonzero when a resumed pipeline fails, or when one could not be
# started because Stage-3+ outputs of an earlier run were in the way. A dataset
# whose window was closed without saving is reported as cancelled, not as a
# failure.
# ============================================================
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_roles  # noqa: E402
from run_artifacts import INITIAL_ALIGNMENT, initial_alignment_path
VERSION_ROOT = SCRIPTS.parent
RUN_PHARAOH = VERSION_ROOT / "run_pharaoh.py"
STAGE1_GUI = HERE / "gui_stage1_orient.py"
STAGE2_SCRIPT = "2_manual_alignment.py"
# Stage 2's in-window ORB suggestion is off here. Every dataset in the queue is
# one ORB and SIFT already failed on; a button offering that same registration
# can only hand the user a bad alignment to accept.
STAGE2_ARGS = ("--no-auto-orb",)

QUEUE_NAME = "manual_alignment_queue.json"
BATCH_SUMMARY_NAME = "parallel_summary.json"
SUMMARY_NAME = "fail_safe_manual_alignment_summary.json"

STAGE1_MARKER = "images_info.json"
STAGE2_MARKER = INITIAL_ALIGNMENT

# Stage-3 and later outputs. A queued dataset never reached them, so anything
# here belongs to an earlier run of the same directory -- computed from an
# alignment, and often from Stage-1 levels, that the manual pass has just
# replaced. run_pharaoh skips a stage whose output exists, so resuming over them
# would silently mix two coordinate systems.
DOWNSTREAM_ARTIFACTS = (
    "tiles",
    "nuclei_patches",
    "dense_nucleus_pairs",
    "dapi_to_he_homography_level0.json",
    "he0_to_he_homography_level0.json",
    "stage8_alignment_refinement.json",
)
DOWNSTREAM_GLOBS = ("6_overlay_final_L*.png",)

# Flags that would make the resumed run recompute Stage 1 or Stage 2 -- which is
# exactly the manual work just done -- or stop before Stage 3. `--stage1-*-level`
# is in the list because run_pharaoh turns a Stage-1 level override into a --force.
RESUME_DROP_FLAGS = {
    "--force": False,
    "--stop-after": True,
    "--stage1-he-level": True,
    "--stage1-fixed-level": True,
    "--stage1-moving-level": True,
    # The pair is re-stated from the case, so a command recorded by a 4.10 batch
    # (which spelled it --dapi/--hefg/--he) replays through the flags that exist.
    "--pair": True,
    "--moving": True,
    "--fixed": True,
    "--dapi": True,
    "--hefg": True,
    "--he0": True,
    "--he": True,
}

_print_lock = threading.Lock()


def log(message):
    with _print_lock:
        print(message, flush=True)


@dataclass
class ManualCase:
    name: str
    pair: str                      # a run_roles pair name
    moving: Path
    fixed: Path
    run_dir: Path
    resume_command: Optional[List[str]] = None
    record: dict = field(default_factory=dict)

    @property
    def engine(self) -> Path:
        return SCRIPTS / run_roles.PAIRS[self.pair].engine


# ---------------------------------------------------------------
# Loading the queue
# ---------------------------------------------------------------
def resolve_queue_path(target: Path) -> Path:
    target = target.expanduser().resolve()
    if target.is_dir():
        return target / QUEUE_NAME
    return target


def _recorded_commands(queue_path: Path) -> dict:
    """name -> the run_pharaoh command the batch used, if the batch left one."""
    summary_path = queue_path.parent / BATCH_SUMMARY_NAME
    if not summary_path.is_file():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    commands = {}
    for entry in summary.get("datasets", []):
        command = entry.get("command")
        if isinstance(entry, dict) and isinstance(command, list) and command:
            commands[entry.get("name")] = [str(part) for part in command]
    return commands


def strip_resume_flags(command: List[str]) -> List[str]:
    """Drop the flags that must not survive into a resumed run."""
    kept = []
    skip_value = False
    for token in command:
        if skip_value:
            skip_value = False
            continue
        flag = token.split("=", 1)[0]
        if flag in RESUME_DROP_FLAGS:
            skip_value = RESUME_DROP_FLAGS[flag] and "=" not in token
            continue
        kept.append(token)
    return kept


def build_resume_command(case: ManualCase, recorded: Optional[List[str]],
                         extra_args: List[str]) -> List[str]:
    """The command that finishes `case`, preferring the batch's own settings."""
    if recorded:
        command = strip_resume_flags(recorded)
        # The interpreter that ran the batch may no longer be the right one, and
        # the RUN_DIR is the one just aligned, whatever the record says.
        command[0] = sys.executable
        command = _with_option(command, "--run-dir", str(case.run_dir))
    else:
        command = [sys.executable, str(RUN_PHARAOH), "--run-dir", str(case.run_dir)]
    # The pair always comes from the case, never from the record.
    command += ["--pair", case.pair, "--moving", str(case.moving),
                "--fixed", str(case.fixed)]
    return command + list(extra_args)


def _with_option(command: List[str], flag: str, value: str) -> List[str]:
    """Set `flag` to `value`, whether the command already carries it or not."""
    out = []
    replaced = False
    index = 0
    while index < len(command):
        token = command[index]
        if token == flag:
            out.extend([flag, value])
            replaced = True
            index += 2
            continue
        if token.startswith(flag + "="):
            out.append(f"{flag}={value}")
            replaced = True
            index += 1
            continue
        out.append(token)
        index += 1
    if not replaced:
        out.extend([flag, value])
    return out


def load_cases(queue_path: Path, extra_args: List[str]) -> List[ManualCase]:
    """Read manual_alignment_queue.json into cases this launcher can run."""
    try:
        document = json.loads(queue_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"queue not found: {queue_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {queue_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("queue root must be a JSON object")
    entries = document.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("queue must contain a 'datasets' list")

    recorded = _recorded_commands(queue_path)
    cases = []
    for index, entry in enumerate(entries):
        label = f"datasets[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be a JSON object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        # 4.10 queues named the moving image by modality; 4.11 writes `pair`,
        # `moving` and `fixed`, and `dapi` is still read as `dapi-he`.
        if entry.get("moving"):
            pair_name = str(entry.get("pair") or run_roles.DEFAULT_PAIR)
            moving_field, fixed_field = "moving", "fixed"
        elif entry.get("dapi"):
            pair_name, moving_field, fixed_field = "dapi-he", "dapi", "he"
        else:
            raise ValueError(f"{label} must provide 'moving' (with 'pair'), or 'dapi'")
        pair = run_roles.pair_from_name(pair_name)
        run_dir = entry.get("run_dir")
        if not run_dir:
            raise ValueError(
                f"{label}.run_dir is missing; the queue must name the run "
                f"directory the failed alignment belongs to"
            )
        case = ManualCase(
            name=name,
            pair=pair.name,
            moving=Path(entry[moving_field]).expanduser(),
            fixed=(Path(entry[fixed_field]).expanduser()
                   if entry.get(fixed_field) else None),
            run_dir=Path(run_dir).expanduser().resolve(),
        )
        if case.fixed is None:
            raise ValueError(f"{label}.{fixed_field} is missing")
        case.resume_command = build_resume_command(
            case, recorded.get(name), extra_args
        )
        cases.append(case)
    return cases


def single_case(args, extra_args: List[str]) -> ManualCase:
    """One dataset named on the command line, without a queue file."""
    case = ManualCase(
        name=args.name[0] if args.name else Path(args.run_dir).name,
        pair=run_roles.pair_from_name(args.pair).name,
        moving=Path(args.moving).expanduser().resolve(),
        fixed=Path(args.fixed).expanduser().resolve(),
        run_dir=Path(args.run_dir).expanduser().resolve(),
    )
    case.resume_command = build_resume_command(case, None, extra_args)
    return case


# ---------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _wrote(path: Path, before: float) -> bool:
    """Whether a GUI actually (re)wrote `path`, rather than leaving an old one."""
    after = _mtime(path)
    return after > 0.0 and after != before


def stale_downstream(run_dir: Path, since: float) -> List[Path]:
    """Stage-3+ outputs older than the alignment they would be built on."""
    candidates = [run_dir / name for name in DOWNSTREAM_ARTIFACTS]
    for pattern in DOWNSTREAM_GLOBS:
        candidates.extend(run_dir.glob(pattern))
    return [path for path in candidates if path.exists() and _mtime(path) < since]


def remove_paths(paths: List[Path]):
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_manual_alignment(case: ManualCase) -> dict:
    """Stage 1 orientation, then Stage 2 manual alignment, for one dataset."""
    record = {
        "name": case.name,
        "pair": case.pair,
        "run_dir": str(case.run_dir),
        "fixed": str(case.fixed),
        "moving": str(case.moving),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    case.run_dir.mkdir(parents=True, exist_ok=True)

    info_path = case.run_dir / STAGE1_MARKER
    before = _mtime(info_path)
    stage1 = subprocess.run(
        [
            sys.executable, str(STAGE1_GUI),
            "--run-dir", str(case.run_dir),
            "--pair", case.pair,
            "--fixed", str(case.fixed),
            "--moving", str(case.moving),
            "--name", case.name,
        ],
        cwd=str(SCRIPTS),
    )
    if not _wrote(info_path, before):
        record.update(stage1="cancelled", status="cancelled",
                      detail=f"Stage 1 exited ({stage1.returncode}) without writing "
                             f"{STAGE1_MARKER}")
        return record
    record["stage1"] = "saved"

    alignment_path = case.run_dir / STAGE2_MARKER
    before = _mtime(alignment_path)
    stage2 = subprocess.run(
        [sys.executable, str(case.engine / STAGE2_SCRIPT), str(case.run_dir),
         *STAGE2_ARGS],
        cwd=str(case.engine),
    )
    if not _wrote(alignment_path, before):
        record.update(stage2="cancelled", status="cancelled",
                      detail=f"Stage 2 exited ({stage2.returncode}) without writing "
                             f"{STAGE2_MARKER}")
        return record
    record.update(stage2="saved", status="aligned")
    record["stale_downstream"] = [
        str(path) for path in stale_downstream(case.run_dir, _mtime(alignment_path))
    ]
    return record


def resume_pipeline(case: ManualCase, log_path: Path) -> dict:
    """Finish one aligned dataset: Stages 3-8 in its own RUN_DIR."""
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    log(f"[{case.name}] resuming: {' '.join(case.resume_command)}")
    log(f"[{case.name}] log: {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"command: {' '.join(case.resume_command)}\nstarted_at: {started_at}\n\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            case.resume_command,
            cwd=str(VERSION_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_file.write(line)
            log(f"[{case.name}] {line.rstrip()}")
        returncode = process.wait()
        elapsed = time.perf_counter() - started
        log_file.write(f"\nreturncode: {returncode}\nelapsed_seconds: {elapsed:.3f}\n")
    return {
        "command": case.resume_command,
        "log": str(log_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "returncode": returncode,
    }


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
def write_summary(path: Path, document: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False, 
        description="Manually align the datasets in manual_alignment_queue.json, "
                    "then finish their pipelines."
    )
    parser.add_argument(
        "queue", nargs="?", type=Path,
        help=f"{QUEUE_NAME}, or the batch output root that contains it",
    )
    parser.add_argument(
        "--name", action="append", default=[],
        help="only this dataset (repeatable); also names the ad-hoc --run-dir case",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=f"re-align datasets that already have a {STAGE2_MARKER}",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="stop after Stage 2 instead of finishing the pipelines",
    )
    parser.add_argument(
        "--resume-jobs", type=int, default=1,
        help="pipelines to finish concurrently after the manual work (default: 1)",
    )
    parser.add_argument(
        "--pharaoh-arg", action="append", default=[], metavar="ARG",
        help="extra argument appended to every resumed run_pharaoh.py command "
             "(repeatable)",
    )
    parser.add_argument(
        "--clean-downstream", action="store_true",
        help="delete Stage-3+ outputs left by an earlier run of the same RUN_DIR "
             "instead of refusing to resume over them",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be aligned and run, and change nothing",
    )
    single = parser.add_argument_group("single dataset (no queue file)")
    single.add_argument("--run-dir", help="RUN_DIR of one failed alignment")
    single.add_argument("--pair", default=run_roles.DEFAULT_PAIR,
                        help="what is registered onto what (default: "
                             f"{run_roles.DEFAULT_PAIR})")
    single.add_argument("--moving", help="moving image path")
    single.add_argument("--fixed", help="fixed image path")
    return parser


def resolve_cases(args, parser):
    """Either the queue's datasets, or the single dataset named on the CLI."""
    extra_args = list(args.pharaoh_arg)
    if args.run_dir:
        if not args.moving or not args.fixed:
            parser.error("--run-dir needs --moving and --fixed")
        if args.queue:
            parser.error("give either a queue file or --run-dir, not both")
        return None, [single_case(args, extra_args)]

    if not args.queue:
        parser.error("give a queue file (or an output root), or --run-dir")
    queue_path = resolve_queue_path(args.queue)
    try:
        cases = load_cases(queue_path, extra_args)
    except ValueError as exc:
        parser.error(str(exc))
    return queue_path, cases


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume_jobs < 1:
        parser.error("--resume-jobs must be at least 1")

    queue_path, cases = resolve_cases(args, parser)
    # A queue's results belong beside the queue; a single --run-dir case writes into
    # that run, so several of them under one batch root cannot overwrite each other's
    # summary (--failsafe-mode instant runs exactly that way).
    work_dir = queue_path.parent if queue_path else cases[0].run_dir

    if args.name and queue_path:
        wanted = set(args.name)
        unknown = wanted - {case.name for case in cases}
        if unknown:
            parser.error(f"not in the queue: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.name in wanted]

    skipped = []
    if not args.force:
        pending = []
        for case in cases:
            if initial_alignment_path(case.run_dir).is_file():
                skipped.append(case)
            else:
                pending.append(case)
        cases = pending

    for case in skipped:
        log(f"[skip] {case.name}: {initial_alignment_path(case.run_dir).name} already exists "
            f"(use --force to align it again)")
    if not cases:
        log("Nothing to align.")
        return 0

    log(f"{len(cases)} dataset(s) to align manually:")
    for index, case in enumerate(cases, start=1):
        log(f"  {index}. {case.name} [{case.pair}] -> {case.run_dir}")
        if args.dry_run:
            log(f"     stage 1: {STAGE1_GUI.name} --pair {case.pair} "
                f"--fixed {case.fixed} --moving {case.moving}")
            log(f"     stage 2: {case.engine.name}/{STAGE2_SCRIPT}")
            log(f"     resume : {' '.join(case.resume_command) if args.resume else '(off)'}")
    if args.dry_run:
        return 0

    summary_path = work_dir / SUMMARY_NAME
    summary = {
        "schema_version": 1,
        "kind": "pharaoh_fail_safe_manual_alignment_summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queue": str(queue_path) if queue_path else None,
        "resume": bool(args.resume),
        "resume_jobs": args.resume_jobs,
        "skipped": [case.name for case in skipped],
        "datasets": [],
    }

    # ---- interactive phase: every dataset, before any pipeline starts ----
    aligned = []
    for index, case in enumerate(cases, start=1):
        log(f"\n=== [{index}/{len(cases)}] {case.name} ===")
        case.record = run_manual_alignment(case)
        summary["datasets"].append(case.record)
        write_summary(summary_path, summary)
        if case.record["status"] == "aligned":
            log(f"[{case.name}] manual alignment saved -> {case.run_dir / STAGE2_MARKER}")
            aligned.append(case)
        else:
            log(f"[{case.name}] {case.record['detail']}")

    blocked = []
    if aligned and args.resume:
        # Only a resume can trip over old Stage-3+ outputs, so --no-resume has
        # nothing to block on.
        cleared = []
        for case in aligned:
            stale = [Path(path) for path in case.record.get("stale_downstream", [])]
            if not stale:
                cleared.append(case)
                continue
            if args.clean_downstream:
                remove_paths(stale)
                case.record["cleaned_downstream"] = [str(path) for path in stale]
                log(f"[{case.name}] removed {len(stale)} stale Stage-3+ artifact(s)")
                cleared.append(case)
                continue
            case.record["status"] = "stale_downstream"
            case.record["detail"] = (
                "Stage-3+ outputs from an earlier run would be reused over the new "
                "alignment; remove them, or re-run with --clean-downstream: "
                + ", ".join(str(path) for path in stale)
            )
            log(f"[{case.name}] not resumed. {case.record['detail']}")
            blocked.append(case)
        aligned = cleared
        write_summary(summary_path, summary)

    if not args.resume or not aligned:
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_summary(summary_path, summary)
        report(summary, summary_path)
        return 1 if blocked else 0

    # ---- compute phase: finish the aligned datasets ----
    log(f"\n=== finishing {len(aligned)} pipeline(s), "
        f"{args.resume_jobs} at a time ===")
    summary_lock = threading.Lock()

    def finish(case: ManualCase):
        result = resume_pipeline(case, work_dir / f"{case.run_dir.name}.failsafe.log")
        with summary_lock:
            case.record["resume"] = result
            case.record["status"] = (
                "completed" if result["returncode"] == 0 else "resume_failed"
            )
            write_summary(summary_path, summary)
        return result["returncode"]

    with ThreadPoolExecutor(max_workers=args.resume_jobs) as pool:
        returncodes = list(pool.map(finish, aligned))

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_summary(summary_path, summary)
    report(summary, summary_path)
    return 1 if blocked or any(code != 0 for code in returncodes) else 0


def report(summary: dict, summary_path: Path):
    log("\n=== fail-safe manual alignment ===")
    for record in summary["datasets"]:
        line = f"  {record['name']}: {record['status']}"
        resume = record.get("resume")
        if resume:
            line += f" (returncode {resume['returncode']})"
        elif record.get("detail"):
            line += f" - {record['detail']}"
        log(line)
    log(f"summary: {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
