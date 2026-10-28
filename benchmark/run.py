#!/usr/bin/env python3
"""Run one method config over one dataset manifest, one registration at a time.

Usually called through run_benchmark.sh. Directly:

    python run.py --config configs/methods/voltron.json --manifest manifests/keypoints.json --warmstart
    python run.py --config configs/methods/regwsi.json --manifest manifests/matrix.json --list-plan

Order is condition-major: for each condition in the config, every dataset in manifest
order, then the next condition. With --warmstart, each condition first runs its first
pending dataset once as a discarded warm-up (cold caches, package import, JVM / R start),
then the measured runs. Every run is its own process group; its process-tree RSS is
sampled and host swap and local disk are guarded (bench/resources.py). Resource settings
come from configs/resources.json and can be overridden on the command line.

Layout:  <runs_root>/<series>/<group>/<method>/<condition>/<dataset>/   measured runs
         <runs_root>/<series>/<group>/<method>/_warmup/<condition>/<dataset>/
         <runs_root>/<series>/<group>/<method>/_batch/   plan.json, summary.json, events.jsonl
<series> comes from --series (run_benchmark.sh's SERIES), <group> from the manifest. A run is complete when its
run_record.json says `success`; complete runs are skipped. An unsuccessful run keeps only its
logs and records (LOG_KEEP), and those are moved to <condition>/_attempts/ before a retry.
Create _batch/PAUSE to stop before the next launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import manifest  # noqa: E402
from bench import resources as guard_lib  # noqa: E402
from bench.artifacts import LOG_KEEP, TIMING_ONLY_KEEP, WARMUP_KEEP, keep_only  # noqa: E402
from bench.config import (MANIFESTS, METHODS, PACKAGES, ROOT as BENCH_ROOT, VALIS_DAPI_READER, environment,  # noqa: E402
                          method_config, read_json, resources, write_json)

MPS_BUDGET_MARKER = "MPS_BUDGET_EXCEEDED"
WRAPPERS = {m: BENCH_ROOT / "methods" / PACKAGES[m] / "wrapper.py" for m in METHODS}
def batch_stop_reason(reason, action: str):
    """Whether a run's stop reason ends the batch.

    A budget stop of the run itself -- process-tree RSS, or RegWSI's MPS allocation, which
    never reaches RSS -- is recorded and skipped in `skip` mode, exactly like a host swap or
    disk budget. Lost monitoring and anything unrecognised always stop the batch.
    """
    if not reason:
        return None
    if reason == "mps_budget":
        return reason if action == "stop" else None
    return reason if guard_lib.batch_must_stop(reason, action) else None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="Method config JSON (its \"method\" names the method)")
    p.add_argument("--method", choices=METHODS, help="Only needed without --config")
    p.add_argument("--manifest", type=Path, help="Dataset manifest JSON (its \"group\" names the group)")
    p.add_argument("--series", help="Output series: <runs_root>/<series>/... (run_benchmark.sh sets it)")
    p.add_argument("--group", help="Shorthand for --manifest manifests/<group>.json")
    p.add_argument("--warmstart", action="store_true",
                   help="Per condition, one discarded run on its first pending dataset first")
    p.add_argument("--datasets", nargs="+", help="Subset of the manifest's datasets")
    p.add_argument("--keep-artifacts", choices=("yes", "no"), default="yes",
                   help="no: after a successful run, delete its products and keep only the logs, "
                        "records and timings -- for repeat series that only measure time")
    p.add_argument("--conditions", nargs="+", help="Subset of the config's conditions")
    p.add_argument("--list-plan", action="store_true", help="Show what would run, and stop")
    p.add_argument("--dry-run", action="store_true", help="Print every wrapper command, and stop")
    p.add_argument("--rerun-completed", action="store_true", help="Repeat successful runs too")
    p.add_argument("--skip-failed", action="store_true",
                   help="Leave runs whose last attempt failed as they are")
    o = p.add_argument_group("resource overrides (default: configs/resources.json)")
    o.add_argument("--max-rss-gib", type=float, help="Per-run process-tree RSS budget; 0 = off")
    o.add_argument("--max-swap-gib", type=float, help="Whole-host swap budget; 0 = off")
    o.add_argument("--timeout", type=int, help="Per-run timeout, seconds")
    o.add_argument("--on-limit", choices=("skip", "stop"),
                   help="After a budget stop: record it and continue, or stop the batch")
    o.add_argument("--swap-settle", choices=("on", "off"),
                   help="Before each launch, wait for host swap to return near its starting level")
    o.add_argument("--settle-cap-gib", type=float, help="Absolute swap level a launch waits for")
    o.add_argument("--settle-tolerance-gib", type=float, help="Allowed swap above the batch's start")
    o.add_argument("--settle-timeout", type=float, help="Longest wait, seconds; then the batch pauses")
    o.add_argument("--cooldown", type=float, help="Seconds to pause after every run, 0 = none")
    args = p.parse_args()
    if not args.config and not args.method:
        p.error("give --config (or --method for its default config)")
    if bool(args.manifest) == bool(args.group):
        p.error("give exactly one of --manifest or --group")
    return args


def effective_limits(args) -> dict:
    limits = resources()
    settle = dict(limits.get("swap_settle") or {})
    for key, value in (("max_rss_gib", args.max_rss_gib), ("max_swap_gib", args.max_swap_gib),
                       ("timeout_seconds", args.timeout), ("resource_limit_action", args.on_limit),
                       ("cooldown_seconds", args.cooldown)):
        if value is not None:
            limits[key] = value
    for key, value in (("cap_gib", args.settle_cap_gib), ("tolerance_gib", args.settle_tolerance_gib),
                       ("timeout_seconds", args.settle_timeout)):
        if value is not None:
            settle[key] = value
    if args.swap_settle is not None:
        settle["enabled"] = args.swap_settle == "on"
    limits["swap_settle"] = settle
    limits.setdefault("cooldown_seconds", 0)
    return limits


# Methods that need no physical pixel size: PHARAOH works in pixels and its wrapper records the
# size only when given one; a dataset without he_mpp is skipped for every other method.
RUNS_WITHOUT_HE_MPP = ("pharaoh",)

class Batch:
    def __init__(self, args):
        self.args = args
        self.env = environment()
        self.limits = effective_limits(args)
        self.config = method_config(args.method or read_json(args.config)["method"], args.config)
        self.method = self.config["method"]
        if args.method and args.method != self.method:
            raise SystemExit(f"--method {args.method} but the config is for {self.method}")
        self.series = args.series or self.config.get("series", "main")
        self.manifest_path = (args.manifest or MANIFESTS / f"{args.group}.json").resolve()
        self.group, self.datasets = manifest.load_file(self.manifest_path, args.datasets)
        if self.method not in RUNS_WITHOUT_HE_MPP:   # no H&E pixel size: this method cannot run there
            for d in (d for d in self.datasets if d.get("he_mpp") is None):
                print(f"[run] {d['name']}: skipped for {self.method}, no H&E pixel size (he_mpp): "
                      f"{d.get('he_mpp_unknown')}", flush=True)
            self.datasets = [d for d in self.datasets if d.get("he_mpp") is not None]
        if args.group and self.group != args.group:
            raise SystemExit(f"{self.manifest_path} declares group {self.group!r}")
        self.root = Path(self.env["runs_root"]) / self.series / self.group / self.method
        self.batch_dir = self.root / "_batch"
        conditions = self.config["conditions"]
        if args.conditions:
            unknown = sorted(set(args.conditions) - set(conditions))
            if unknown:
                raise SystemExit(f"unknown condition(s) {unknown}; config has {sorted(conditions)}")
            conditions = {k: v for k, v in conditions.items() if k in args.conditions}
        self.conditions = conditions
        self.python = self.env["python"][PACKAGES[self.method]]
        if not Path(self.python).exists():
            raise SystemExit(f"{self.method} interpreter not found: {self.python}")

    # ---- paths and state -------------------------------------------------------------
    def run_dir(self, condition: str, dataset: dict) -> Path:
        return self.root / condition / dataset["name"]

    def warmup_dir(self, condition: str, dataset: dict) -> Path:
        return self.root / "_warmup" / condition / dataset["name"]

    @staticmethod
    def record_of(run_dir: Path):
        path = run_dir / "run_record.json"
        return read_json(path) if path.exists() else None

    def state(self, condition, dataset) -> str:
        record = self.record_of(self.run_dir(condition, dataset))
        if record is None:
            return "pending"
        return "completed" if record.get("outcome") == "success" else "failed"

    # ---- command ---------------------------------------------------------------------
    def pinned_levels(self, dataset):
        source = self.config.get("pinned_levels_from")
        if not source:
            return None
        series = source.get("series", self.series)  # by default the PHARAOH run of this series
        run = (Path(self.env["runs_root"]) / series / self.group / source["method"]
               / source["condition"] / dataset["name"])
        record = self.record_of(run)
        if not record or record.get("outcome") != "success" or not (run / "images_info.json").exists():
            return "unavailable"
        info = read_json(run / "images_info.json")
        return int(info["DAPI_level"]), int(info["HE_level"])

    def command(self, condition, dataset, run_dir):
        cmd = [self.python, str(WRAPPERS[self.method]),
               "--dapi", dataset["dapi"], "--he", dataset["he"], "--outdir", str(run_dir),
               "--dapi-mpp", repr(float(dataset["dapi_mpp"])),
               *(["--he-mpp", repr(float(dataset["he_mpp"]))] if dataset.get("he_mpp") is not None else []),
               *self.config.get("args", []), *self.conditions[condition]]
        if self.method == "pharaoh":
            cmd += ["--pharaoh-root", self.env["pharaoh_root"]]
        elif self.method == "voltron":
            cmd += ["--rscript", self.env["rscript"]]
        elif self.method == "regwsi":
            cmd += ["--memory-budget-gib", repr(float(self.limits["max_rss_gib"]))]
        elif PACKAGES[self.method] == "valis":
            # The method, not its config, decides how the DAPI reaches VALIS.
            cmd += ["--dapi-reader", VALIS_DAPI_READER[self.method],
                    "--input-cache", str(Path(self.env["cache_root"]) / "valis_inputs"),
                    "--cache-series", self.series]
        pinned = self.pinned_levels(dataset)
        if pinned == "unavailable":
            return None
        if pinned:
            cmd += ["--dapi-level", str(pinned[0]), "--he-level", str(pinned[1])]
        return cmd

    # ---- bookkeeping -----------------------------------------------------------------
    def event(self, **payload):
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        with (self.batch_dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": now(), **payload}) + "\n")

    def write_plan(self):
        write_json(self.batch_dir / "plan.json", {
            "written_at": now(), "method": self.method, "group": self.group, "series": self.series,
            "config": {k: v for k, v in self.config.items() if not k.startswith("_")},
            "config_path": self.config["_path"], "resources": self.limits,
            "environment": self.env, "warmstart": self.args.warmstart,
            "keep_artifacts": self.args.keep_artifacts,
            "order": "condition-major", "manifest": str(self.manifest_path),
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(WRAPPERS[self.method].read_bytes()).hexdigest(),
            "datasets": [d["name"] for d in self.datasets], "conditions": list(self.conditions)})

    def write_summary(self, state: str, **extra):
        rows = []
        for condition in self.config["conditions"]:
            for dataset in self.datasets:
                record = self.record_of(self.run_dir(condition, dataset))
                if record:
                    rows.append({k: record.get(k) for k in (
                        "dataset", "condition", "outcome", "reason", "wall_seconds",
                        "method_sec", "peak_rss_mib", "finished_at")})
        counts = {}
        for row in rows:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        write_json(self.batch_dir / "summary.json", {
            "updated_at": now(), "batch_state": state, "outcome_counts": counts,
            "runs": rows, **extra})

    @staticmethod
    def archive_attempt(run_dir: Path):
        """Keep an unsuccessful attempt's logs and records instead of overwriting them."""
        if not run_dir.exists():
            return
        keep_only(run_dir, LOG_KEEP)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        target = run_dir.parent / "_attempts" / f"{run_dir.name}_{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(target))

    # ---- launching -------------------------------------------------------------------
    def admit(self, condition, dataset, baseline):
        """Wait at the launch gates. Returns a stop reason, or None to go ahead."""
        limits, settle = self.limits, self.limits["swap_settle"]
        if (self.batch_dir / "PAUSE").exists():
            return "paused"
        if limits["resource_limit_action"] == "skip":
            gate = guard_lib.wait_for_host_recovery(
                limits["max_swap_gib"], limits["min_local_free_gib"],
                timeout=limits["recovery_timeout_seconds"], interval=limits["recovery_interval_seconds"])
            if not gate["ready"]:
                self.event(kind="gate_blocked", gate=gate)
                return "host_resources_not_recovered"
        if baseline is not None:
            limit_gib = min(settle["cap_gib"], baseline / guard_lib.GIB + settle["tolerance_gib"])
            current_gib = guard_lib.read_host_swap().allocated_bytes / guard_lib.GIB
            if current_gib > limit_gib:
                print(f"[wait ] host swap {current_gib:.1f} GiB > settle limit {limit_gib:.1f} GiB; "
                      f"waiting up to {settle['timeout_seconds']:.0f}s before {condition} "
                      f"{dataset['name']}", flush=True)
            settled = guard_lib.wait_for_swap_settle(
                baseline, tolerance_gib=settle["tolerance_gib"], cap_gib=settle["cap_gib"],
                timeout=settle["timeout_seconds"], interval=limits["recovery_interval_seconds"])
            self.event(kind="swap_settle", dataset=dataset["name"], condition=condition, **settled)
            if not settled["ready"]:
                return "swap_not_settled"
        return None

    def run_one(self, condition, dataset, cmd, run_dir: Path, kind: str) -> dict:
        """Launch one wrapper run under the guard and write its run_record.json."""
        if kind == "warmup":
            shutil.rmtree(run_dir, ignore_errors=True)
        else:
            self.archive_attempt(run_dir)
        run_dir.mkdir(parents=True)
        limits = self.limits
        guard = guard_lib.ResourceGuard(limits["max_rss_gib"], limits["max_swap_gib"],
                                        min_local_free_gib=limits["min_local_free_gib"])
        started_at, started = now(), time.monotonic()
        returncode, timed_out, launched, peak = None, False, False, None
        with (run_dir / "run.log").open("w") as log:
            log.write("$ " + shlex.join(cmd) + "\n\n")
            log.flush()
            if guard.sample(0):
                log.write(f"RESOURCE_GUARD_PRELAUNCH {json.dumps(guard.trigger)}\n")
            else:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix", cwd=str(BENCH_ROOT))
                launched = True
                sampler = guard_lib.TreeSampler(process.pid, limits["sample_interval_seconds"])
                sampler.start()
                try:
                    returncode, timed_out = guard_lib.watch(
                        process, sampler, guard, timeout=limits["timeout_seconds"],
                        interval=limits["sample_interval_seconds"])
                except BaseException:
                    guard_lib.kill_run_tree(process)
                    raise
                finally:
                    peak = sampler.stop()
                    if guard.trigger is None and (sampler.monitor_error or not sampler.peak_bytes):
                        guard.trigger = {"reason": "rss_monitor_unavailable",
                                         "error": sampler.monitor_error or "no RSS sample recorded"}
                    guard.sample(sampler.peak_bytes)
        reason = (guard.trigger or {}).get("reason")
        log_text = (run_dir / "run.log").read_text(errors="replace")
        if reason is None and MPS_BUDGET_MARKER in log_text:
            reason = "mps_budget"
        if launched and reason:
            guard_lib.kill_run_tree(process)
        contract_ok = (run_dir / "coordinate_contract.json").exists() and (run_dir / "run_metadata.json").exists()
        if reason:
            outcome = "resource_guard_stopped"
        elif timed_out:
            outcome = "timeout"
        elif returncode == 0 and contract_ok:
            outcome = "success"
        elif returncode == 0:
            outcome = "wrapper_export_error"
        elif returncode in (-9, 137):
            outcome = "external_sigkill"
        else:
            outcome = "registration_error"
        metadata = read_json(run_dir / "run_metadata.json") if contract_ok else {}
        record = {
            "kind": kind, "dataset": dataset["name"], "condition": condition, "method": self.method,
            "group": self.group, "series": self.series, "command": cmd,
            "started_at": started_at, "finished_at": now(),
            "wall_seconds": time.monotonic() - started, "launched": launched,
            "returncode": returncode, "timed_out": timed_out, "outcome": outcome,
            "reason": reason, "succeeded": outcome == "success",
            "method_sec": (metadata.get("timings") or {}).get("method_sec"),
            "peak_rss_mib": peak, "resource_guard": guard.report(),
            "error_tail": None if outcome == "success" else log_text.strip().splitlines()[-8:],
        }
        write_json(run_dir / "run_record.json", record)
        if kind == "warmup":  # a warm-up's transform is never scored; keep only its record
            keep_only(run_dir, WARMUP_KEEP)
        elif outcome != "success":  # no result to keep: only the logs and records
            keep_only(run_dir, LOG_KEEP)
        elif self.args.keep_artifacts == "no":  # timing-only series: only the timings stay
            keep_only(run_dir, TIMING_ONLY_KEEP)
        return record

    def report(self, record):
        label = "warmup" if record["kind"] == "warmup" else record["outcome"]
        extra = f" ({record['reason']})" if record["reason"] else ""
        if record["kind"] == "warmup" and record["outcome"] != "success":
            extra += f" [{record['outcome']}]"
        print(f"[{label}] {record['condition']:16s} {record['dataset']} {record['wall_seconds']:.0f}s "
              f"peak {record['peak_rss_mib'] or 0:.0f} MiB{extra}", flush=True)
        self.event(kind=record["kind"], **{k: record[k] for k in (
            "dataset", "condition", "outcome", "reason", "wall_seconds", "peak_rss_mib")})

    def must_stop(self, record):
        return batch_stop_reason(record["reason"], self.limits["resource_limit_action"])

    # ---- batch -----------------------------------------------------------------------
    def plan(self):
        """[(condition, [(dataset, state, todo), ...])] in condition-major order."""
        rows = []
        for condition in self.conditions:
            items = []
            for dataset in self.datasets:
                state = self.state(condition, dataset)
                todo = (state == "pending" or (state == "failed" and not self.args.skip_failed)
                        or (state == "completed" and self.args.rerun_completed))
                items.append((dataset, state, todo))
            rows.append((condition, items))
        return rows

    def run(self) -> int:
        rows = self.plan()
        counts = {}
        for _, items in rows:
            for _, state, todo in items:
                key = "to_launch" if todo else state
                counts[key] = counts.get(key, 0) + 1
        warmups = sum(1 for _, items in rows if any(todo for *_, todo in items)) if self.args.warmstart else 0
        print(f"[plan] {self.method} / {self.group} / series {self.series}: {len(self.conditions)} "
              f"conditions x {len(self.datasets)} datasets -> {self.root}")
        print(f"[plan] {counts}, warm-ups {warmups}")
        if self.args.list_plan:
            for condition, items in rows:
                for dataset, state, todo in items:
                    print(f"  {'*' if todo else ' '} {state:9s} {condition:16s} {dataset['name']}")
            return 0
        if self.args.dry_run:
            for condition, items in rows:
                for dataset, _, todo in items:
                    if todo:
                        cmd = self.command(condition, dataset, self.run_dir(condition, dataset))
                        print(shlex.join(cmd) if cmd else f"# {condition} {dataset['name']}: pinned levels unavailable")
            return 0

        limits = self.limits
        guard_lib.validate(limits["max_rss_gib"], limits["max_swap_gib"],
                           limits["sample_interval_seconds"], limits["min_local_free_gib"])
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.write_plan()
        settle = limits["swap_settle"]
        baseline = guard_lib.read_host_swap().allocated_bytes if settle.get("enabled") else None
        print(f"[batch] RSS {limits['max_rss_gib']} GiB, swap {limits['max_swap_gib']} GiB, "
              f"timeout {limits['timeout_seconds']}s, on limit: {limits['resource_limit_action']}, "
              f"swap settle {'on' if baseline is not None else 'off'}, cooldown {limits['cooldown_seconds']}s",
              flush=True)
        self.event(kind="batch_start", to_launch=counts.get("to_launch", 0), warmups=warmups,
                   swap_baseline_gib=None if baseline is None else baseline / guard_lib.GIB)
        self.write_summary("running")

        stop_reason = None
        for condition, items in rows:
            pending = [d for d, _, todo in items if todo]
            if not pending:
                continue
            launches = []
            if self.args.warmstart:
                launches.append(("warmup", pending[0]))
            launches += [("measured", d) for d in pending]
            for kind, dataset in launches:
                run_dir = (self.warmup_dir if kind == "warmup" else self.run_dir)(condition, dataset)
                cmd = self.command(condition, dataset, run_dir)
                if cmd is None:
                    print(f"[skip ] {condition} {dataset['name']}: pinned PHARAOH levels unavailable")
                    self.event(kind="skipped", dataset=dataset["name"], condition=condition,
                               reason="pinned_levels_unavailable")
                    continue
                stop_reason = self.admit(condition, dataset, baseline)
                if stop_reason:
                    break
                print(f"[{'warm ' if kind == 'warmup' else 'start'}] {condition:16s} {dataset['name']}",
                      flush=True)
                record = self.run_one(condition, dataset, cmd, run_dir, kind)
                self.report(record)
                if kind == "measured":
                    self.write_summary("running")
                stop_reason = self.must_stop(record)
                if stop_reason:
                    break
                if limits["cooldown_seconds"]:
                    time.sleep(limits["cooldown_seconds"])
            if stop_reason:
                break
        state = "stopped" if stop_reason else "finished"
        self.write_summary(state, stop_reason=stop_reason)
        self.event(kind="batch_end", state=state, stop_reason=stop_reason)
        print(f"[batch] {state}" + (f": {stop_reason}" if stop_reason else ""))
        return guard_lib.RESOURCE_GUARD_EXIT if stop_reason else 0


def main() -> int:
    return Batch(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
