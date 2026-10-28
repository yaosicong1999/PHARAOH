"""Per-run memory measurement and host protection. Never changes registration parameters.

* RSS is summed over the run's whole process tree, sampled every `interval` seconds.
* The swap budget is whole-host. On macOS it counts allocated swapfile capacity (what
  consumes local storage), elsewhere used swap. It is not attributable to one run.
* Both are sampled guards, not kernel limits; a run can briefly overshoot between samples.
* Each run gets its own process group, so a stop kills exactly that run's tree.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

GIB = 1024 ** 3
RESOURCE_GUARD_EXIT = 75
RECOVERABLE = frozenset({"rss_budget", "host_swap_budget", "host_disk_free_floor"})


# --------------------------------------------------------------------------------------
# Host readings
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SwapSnapshot:
    allocated_bytes: int
    used_bytes: int
    metric: str = "allocated"
    source: str = "macos_vm.swapusage"

    @property
    def budget_bytes(self) -> int:
        return self.allocated_bytes if self.metric == "allocated" else self.used_bytes


@dataclass(frozen=True)
class DiskSnapshot:
    free_bytes: int
    total_bytes: int
    path: str = "/"


def read_local_disk() -> DiskSnapshot:
    usage = shutil.disk_usage("/")
    return DiskSnapshot(usage.free, usage.total)


def parse_macos_swapusage(output: str) -> SwapSnapshot:
    values = {}
    for field in ("total", "used"):
        match = re.search(rf"\b{field}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)B?\b", output)
        if match is None:
            raise ValueError(f"vm.swapusage has no valid {field} value")
        values[field] = int(float(match[1]) * 1024 ** {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[match[2]])
    if values["used"] > values["total"]:
        raise ValueError("vm.swapusage used exceeds total")
    return SwapSnapshot(values["total"], values["used"])


def read_host_swap() -> SwapSnapshot:
    if sys.platform == "darwin":
        result = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                                capture_output=True, text=True, check=True, timeout=2)
        return parse_macos_swapusage(result.stdout)
    import psutil
    swap = psutil.swap_memory()
    return SwapSnapshot(swap.total, swap.used, metric="used", source="psutil.swap_memory")


# --------------------------------------------------------------------------------------
# Process-tree RSS
# --------------------------------------------------------------------------------------

class TreeSampler(threading.Thread):
    """Peak summed RSS of a process tree. A sampling error ends the thread and is kept in
    `monitor_error`, so the watcher fails closed instead of under-reporting."""

    def __init__(self, pid: int, interval: float):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_bytes = 0
        self.monitor_error = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            import psutil
            root = psutil.Process(self.pid)
            while not self._stop_event.is_set():
                try:
                    processes = [root] + root.children(recursive=True)
                except psutil.NoSuchProcess:
                    break
                total = 0
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue  # a child that exited normally is not lost monitoring
                self.peak_bytes = max(self.peak_bytes, total)
                self._stop_event.wait(self.interval)
        except Exception as exc:  # AccessDenied and friends must not under-report
            self.monitor_error = f"{type(exc).__name__}: {exc}"

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)
        return self.peak_bytes / 2**20 if self.peak_bytes else None


def memory_limit_gib(max_rss_gib, reserve_gib) -> float:
    """An explicit budget, or total RAM minus the reserve."""
    if max_rss_gib is not None:
        return float(max_rss_gib)
    import psutil
    return max(1.0, psutil.virtual_memory().total / GIB - float(reserve_gib))


# --------------------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------------------

def validate(max_rss_gib, max_swap_gib, interval, min_local_free_gib=0.0) -> None:
    for name, value in (("max_rss_gib", max_rss_gib), ("max_swap_gib", max_swap_gib),
                        ("min_local_free_gib", min_local_free_gib)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative; 0 disables it")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("sample interval must be finite and positive")


def batch_must_stop(reason, action: str) -> bool:
    """Loss of monitoring is never skippable; budgets are skippable in `skip` mode."""
    return bool(reason and (action == "stop" or reason not in RECOVERABLE))


class ResourceGuard:
    def __init__(self, max_rss_gib, max_swap_gib, *, min_local_free_gib=0.0,
                 swap_reader=read_host_swap, disk_reader=read_local_disk):
        validate(max_rss_gib, max_swap_gib, 1.0, min_local_free_gib)
        self.max_rss_gib, self.max_swap_gib = max_rss_gib, max_swap_gib
        self.min_local_free_gib = min_local_free_gib
        self.swap_reader, self.disk_reader = swap_reader, disk_reader
        self.initial_swap = self.latest_swap = None
        self.initial_disk = self.latest_disk = None
        self.min_disk_free_bytes = None
        self.peak_allocated_bytes = self.peak_used_bytes = self.peak_rss_bytes = 0
        self.trigger = None

    def _check(self, rss_bytes, snapshot=None) -> bool:
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        if snapshot is not None:
            self.initial_swap = self.initial_swap or snapshot
            self.latest_swap = snapshot
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, snapshot.allocated_bytes)
            self.peak_used_bytes = max(self.peak_used_bytes, snapshot.used_bytes)
        if self.trigger is not None:
            return True
        if self.max_rss_gib and rss_bytes >= self.max_rss_gib * GIB:
            self.trigger = {"reason": "rss_budget", "scope": "run_process_tree",
                            "limit_gib": self.max_rss_gib, "observed_gib": rss_bytes / GIB}
        elif self.max_swap_gib and snapshot is not None and snapshot.budget_bytes >= self.max_swap_gib * GIB:
            self.trigger = {"reason": "host_swap_budget", "scope": "whole_host",
                            "metric": snapshot.metric, "limit_gib": self.max_swap_gib,
                            "observed_gib": snapshot.budget_bytes / GIB}
        return self.trigger is not None

    def sample(self, rss_bytes: int) -> bool:
        try:
            disk = self.disk_reader()
            self.initial_disk = self.initial_disk or disk
            self.latest_disk = disk
            self.min_disk_free_bytes = (disk.free_bytes if self.min_disk_free_bytes is None
                                        else min(self.min_disk_free_bytes, disk.free_bytes))
        except Exception as exc:
            if self.min_local_free_gib and self.trigger is None:
                self.trigger = {"reason": "host_disk_monitor_unavailable", "scope": "whole_host",
                                "error": f"{type(exc).__name__}: {exc}"}
        if self._check(rss_bytes):
            return True
        if (self.min_local_free_gib and self.latest_disk is not None
                and self.latest_disk.free_bytes <= self.min_local_free_gib * GIB):
            self.trigger = {"reason": "host_disk_free_floor", "scope": "whole_host",
                            "limit_gib": self.min_local_free_gib,
                            "observed_gib": self.latest_disk.free_bytes / GIB}
            return True
        if not self.max_swap_gib:
            return False
        try:
            return self._check(rss_bytes, self.swap_reader())
        except Exception as exc:  # a configured protection must not silently vanish
            self.trigger = {"reason": "host_swap_monitor_unavailable", "scope": "whole_host",
                            "error": f"{type(exc).__name__}: {exc}"}
            return True

    def report(self) -> dict:
        start = self.initial_swap
        return {
            "max_rss_gib": self.max_rss_gib, "max_swap_gib": self.max_swap_gib,
            "min_local_free_gib": self.min_local_free_gib,
            "swap_metric": start.metric if start else None,
            "swap_start": asdict(start) if start else None,
            "swap_last": asdict(self.latest_swap) if self.latest_swap else None,
            "peak_swap_allocated_gib": self.peak_allocated_bytes / GIB if start else None,
            "peak_swap_allocated_delta_gib": (max(0, self.peak_allocated_bytes - start.allocated_bytes) / GIB
                                              if start else None),
            "peak_rss_mib": self.peak_rss_bytes / 2**20,
            "local_disk_start": asdict(self.initial_disk) if self.initial_disk else None,
            "min_local_free_gib_observed": (self.min_disk_free_bytes / GIB
                                            if self.min_disk_free_bytes is not None else None),
            "trigger": self.trigger,
        }


# --------------------------------------------------------------------------------------
# Launch gates and the watch loop
# --------------------------------------------------------------------------------------

def wait_for_host_recovery(max_swap_gib, min_local_free_gib, *, timeout, interval,
                           on_update=None) -> dict:
    """Before a launch: wait for host swap/disk to be under budget. Nothing is launched
    and no failure is recorded while waiting; monitor errors end the wait immediately."""
    started = time.monotonic()
    checks = 0
    while True:
        gate = ResourceGuard(0, max_swap_gib, min_local_free_gib=min_local_free_gib)
        blocked = gate.sample(0)
        checks += 1
        elapsed = time.monotonic() - started
        reason = (gate.trigger or {}).get("reason")
        timed_out = bool(blocked and elapsed >= timeout and reason in RECOVERABLE)
        report = {"ready": not blocked, "timed_out": timed_out, "elapsed_seconds": elapsed,
                  "checks": checks, "resource_guard": gate.report()}
        if not blocked or timed_out or reason not in RECOVERABLE:
            if on_update:
                on_update(report)
            return report
        if on_update and checks % 6 == 1:
            on_update(report)
        time.sleep(min(interval, max(0, timeout - elapsed)))


def wait_for_swap_settle(baseline_bytes, *, tolerance_gib, cap_gib, timeout, interval) -> dict:
    """Wait until host swap is back near the batch's starting level, so every run starts
    with the same share of the swap budget (macOS frees a stopped run's swap lazily)."""
    limit = min(cap_gib * GIB, baseline_bytes + tolerance_gib * GIB)
    started = time.monotonic()
    while True:
        try:
            current, error = read_host_swap().allocated_bytes, None
        except Exception as exc:
            current, error = None, f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        ready = error is None and current <= limit
        timed_out = error is None and not ready and elapsed >= timeout
        if ready or timed_out or error is not None:
            return {"ready": ready, "timed_out": timed_out, "elapsed_seconds": elapsed,
                    "baseline_gib": baseline_bytes / GIB, "limit_gib": limit / GIB,
                    "swap_allocated_gib": None if current is None else current / GIB,
                    "monitor_error": error}
        time.sleep(min(interval, max(0, timeout - elapsed)))


def kill_run_tree(process: subprocess.Popen) -> None:
    """SIGKILL this run's process group and any descendant that left it."""
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def watch(process, sampler: TreeSampler, guard: ResourceGuard, *, timeout, interval):
    """Wait for the run, enforcing the guard. Returns (returncode, timed_out)."""
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            return process.returncode, False
        if guard.max_rss_gib and not sampler.is_alive():
            guard.trigger = guard.trigger or {
                "reason": "rss_monitor_unavailable", "scope": "run_process_tree",
                "error": sampler.monitor_error or "RSS sampler ended while the run was active"}
            kill_run_tree(process)
            return process.returncode, False
        if guard.sample(sampler.peak_bytes):
            kill_run_tree(process)
            return process.returncode, False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kill_run_tree(process)
            return None, True
        try:
            return process.wait(timeout=min(interval, remaining)), False
        except subprocess.TimeoutExpired:
            pass
