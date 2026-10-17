"""Fair, resource-adaptive admission control for memory-heavy pipeline stages.

Written for Stage 4 and now shared with Stage 3, which measured as the larger
consumer of the two. Each stage gets its own slot directory so the two queue
independently; `label` only names the stage in the log lines.
"""

import ctypes
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from monitoring import sample_process_trees


GIB = float(1024 ** 3)


@dataclass(frozen=True)
class Stage4SchedulerConfig:
    max_concurrent: int
    adaptive: bool = False
    memory_per_job_gib: float = 5.0
    memory_reserve_gib: float = 4.0
    cpu_threshold_percent: float = 80.0
    queue_cpu_boost_per_waiter: float = 2.0
    queue_cpu_boost_max: float = 10.0
    cpu_bypass_after_seconds: float = 120.0
    admission_cooldown_seconds: float = 5.0
    poll_seconds: float = 0.5

    def validate(self):
        if self.max_concurrent < 1:
            raise ValueError("Stage-4 max_concurrent must be at least 1")
        if self.memory_per_job_gib <= 0:
            raise ValueError("Stage-4 memory_per_job_gib must be greater than 0")
        if self.memory_reserve_gib < 0:
            raise ValueError("Stage-4 memory_reserve_gib cannot be negative")
        if not 0 < self.cpu_threshold_percent <= 100:
            raise ValueError("Stage-4 cpu_threshold_percent must be in (0, 100]")
        if self.queue_cpu_boost_per_waiter < 0 or self.queue_cpu_boost_max < 0:
            raise ValueError("Stage-4 queue CPU boosts cannot be negative")
        if self.cpu_bypass_after_seconds < 0:
            raise ValueError("Stage-4 CPU bypass wait cannot be negative")
        if self.admission_cooldown_seconds < 0:
            raise ValueError("Stage-4 admission cooldown cannot be negative")
        if self.poll_seconds <= 0:
            raise ValueError("Stage-4 poll interval must be greater than 0")


@dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_gib: Optional[float]
    cpu_percent: Optional[float]
    cpu_ticks: Optional[List[int]]
    error: Optional[str] = None


class _DarwinCpuLoadInfo(ctypes.Structure):
    _fields_ = [("ticks", ctypes.c_uint32 * 4)]


class _DarwinVmStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_DARWIN_SYSTEM = None


def _darwin_system():
    global _DARWIN_SYSTEM
    if _DARWIN_SYSTEM is None:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        library.mach_host_self.restype = ctypes.c_uint
        library.host_statistics.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.host_statistics.restype = ctypes.c_int
        library.host_statistics64.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.host_statistics64.restype = ctypes.c_int
        library.host_page_size.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.host_page_size.restype = ctypes.c_int
        _DARWIN_SYSTEM = library
    return _DARWIN_SYSTEM


def _darwin_resources() -> Tuple[float, List[int]]:
    library = _darwin_system()
    host = library.mach_host_self()

    cpu_info = _DarwinCpuLoadInfo()
    cpu_count = ctypes.c_uint32(ctypes.sizeof(cpu_info) // 4)
    if library.host_statistics(
        host, 3, ctypes.byref(cpu_info), ctypes.byref(cpu_count)
    ) != 0:
        raise RuntimeError("host_statistics(HOST_CPU_LOAD_INFO) failed")

    vm_info = _DarwinVmStatistics64()
    vm_count = ctypes.c_uint32(ctypes.sizeof(vm_info) // 4)
    if library.host_statistics64(
        host, 4, ctypes.byref(vm_info), ctypes.byref(vm_count)
    ) != 0:
        raise RuntimeError("host_statistics64(HOST_VM_INFO64) failed")
    page_size = ctypes.c_uint32()
    if library.host_page_size(host, ctypes.byref(page_size)) != 0:
        raise RuntimeError("host_page_size failed")

    available_pages = (
        int(vm_info.free_count)
        + int(vm_info.inactive_count)
        + int(vm_info.speculative_count)
    )
    available_gib = available_pages * int(page_size.value) / GIB
    return available_gib, [int(value) for value in cpu_info.ticks]


def _linux_resources() -> Tuple[float, List[int]]:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields = value.split()
        if fields:
            memory[key] = int(fields[0])
    available_kib = memory.get("MemAvailable", memory.get("MemFree"))
    if available_kib is None:
        raise RuntimeError("MemAvailable is missing from /proc/meminfo")

    fields = Path("/proc/stat").read_text().splitlines()[0].split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("aggregate CPU counters are missing from /proc/stat")
    ticks = [int(value) for value in fields[1:]]
    return available_kib * 1024.0 / GIB, ticks


def _cpu_percent(current: List[int], previous: Optional[List[int]]) -> Optional[float]:
    if not previous or len(current) != len(previous):
        return None
    if sys.platform == "darwin":
        deltas = [
            (int(now) - int(before)) % (2 ** 32)
            for now, before in zip(current, previous)
        ]
        idle_indexes = {2}
    else:
        deltas = [max(0, int(now) - int(before)) for now, before in zip(current, previous)]
        idle_indexes = {3, 4}
    total = sum(deltas)
    if total <= 0:
        return None
    idle = sum(value for index, value in enumerate(deltas) if index in idle_indexes)
    return max(0.0, min(100.0, 100.0 * (total - idle) / total))


def read_system_resources(previous_ticks=None) -> ResourceSnapshot:
    try:
        if sys.platform == "darwin":
            available_gib, ticks = _darwin_resources()
        elif sys.platform.startswith("linux"):
            available_gib, ticks = _linux_resources()
        else:
            raise RuntimeError(f"adaptive resource sampling is unsupported on {sys.platform}")
        return ResourceSnapshot(
            available_memory_gib=round(float(available_gib), 6),
            cpu_percent=_cpu_percent(ticks, previous_ticks),
            cpu_ticks=ticks,
        )
    except Exception as exc:
        return ResourceSnapshot(None, None, None, str(exc))


def evaluate_admission(
    *,
    config: Stage4SchedulerConfig,
    active_count: int,
    active_rss_gib: Iterable[Optional[float]],
    available_memory_gib: Optional[float],
    cpu_percent: Optional[float],
    queue_depth: int,
    waited_seconds: float,
    seconds_since_admission: Optional[float],
) -> dict:
    """Pure admission policy used by the cross-process scheduler and tests."""
    active_rss = list(active_rss_gib)
    missing_active = max(0, int(active_count) - len(active_rss))
    remaining_growth_gib = missing_active * config.memory_per_job_gib
    for rss in active_rss:
        if rss is None:
            remaining_growth_gib += config.memory_per_job_gib
        else:
            remaining_growth_gib += max(0.0, config.memory_per_job_gib - float(rss))

    required_available_gib = (
        config.memory_reserve_gib
        + config.memory_per_job_gib
        + remaining_growth_gib
    )
    # An idle scheduler must never refuse the queue head forever. With no active job
    # there is nothing to protect the head from and nothing that will finish and free
    # memory, so a refusal here is permanent: on a 32 GiB machine with 13.0 GiB free,
    # a 10 GiB per-job estimate plus a 4 GiB reserve wants 14 GiB and the whole batch
    # stalls at queue_position=1 wait=memory. Admitting the head is what a sequential
    # run would do anyway, and the estimate is an estimate -- measured peaks run well
    # under it. The bypass applies only when nothing at all is running.
    memory_fits = (
        available_memory_gib is not None
        and available_memory_gib >= required_available_gib
    )
    memory_bypassed = (
        not memory_fits
        and int(active_count) == 0
        and waited_seconds >= config.admission_cooldown_seconds
    )
    memory_ok = memory_fits or memory_bypassed

    queue_boost = min(
        config.queue_cpu_boost_max,
        max(0, int(queue_depth) - 1) * config.queue_cpu_boost_per_waiter,
    )
    effective_cpu_threshold = min(
        100.0,
        config.cpu_threshold_percent + queue_boost,
    )
    cpu_bypassed = (
        active_count == 0
        and waited_seconds >= config.cpu_bypass_after_seconds
    )
    cpu_ok = (
        (cpu_percent is not None and cpu_percent <= effective_cpu_threshold)
        or cpu_bypassed
    )
    cooldown_ok = (
        active_count == 0
        or seconds_since_admission is None
        or seconds_since_admission >= config.admission_cooldown_seconds
    )
    hard_limit_ok = active_count < config.max_concurrent

    reasons = []
    if not hard_limit_ok:
        reasons.append("hard_limit")
    if not memory_ok:
        reasons.append("memory")
    if not cpu_ok:
        reasons.append("cpu_warmup" if cpu_percent is None else "cpu")
    if not cooldown_ok:
        reasons.append("cooldown")
    return {
        "admit": not reasons,
        "reasons": reasons,
        "active_count": int(active_count),
        "queue_depth": int(queue_depth),
        "available_memory_gib": available_memory_gib,
        "required_available_memory_gib": round(required_available_gib, 6),
        "required_available_gib": round(required_available_gib, 2),
        "estimated_active_growth_gib": round(remaining_growth_gib, 6),
        "cpu_percent": cpu_percent,
        "effective_cpu_threshold_percent": round(effective_cpu_threshold, 6),
        "cpu_bypassed_for_progress": bool(cpu_bypassed),
        "memory_bypassed_when_idle": bool(memory_bypassed),
        "cooldown_remaining_seconds": (
            round(max(0.0, config.admission_cooldown_seconds - seconds_since_admission), 6)
            if seconds_since_admission is not None and active_count > 0
            else 0.0
        ),
    }


def _load_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, data):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (ValueError, TypeError):
        return False
    except PermissionError:
        return True


class Stage4AdmissionController:
    def __init__(self, slot_dir, run_dir, config: Stage4SchedulerConfig, label="Stage 4"):
        config.validate()
        self.label = str(label)
        self.slot_dir = Path(slot_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.coordinator_path = self.slot_dir / "scheduler.lock"
        self.counter_path = self.slot_dir / "queue_counter.txt"
        self.state_path = self.slot_dir / "scheduler_state.json"
        self.status_path = self.slot_dir / "scheduler_status.json"
        self.ticket_path = None
        self.ticket_sequence = None
        self.ticket_created_epoch = None
        self.slot_handle = None
        self.slot_index = None
        self.admission = None
        self._last_report = None
        self._last_report_epoch = 0.0

    @contextmanager
    def _coordinator(self):
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        handle = self.coordinator_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _create_ticket_locked(self):
        try:
            sequence = int(self.counter_path.read_text().strip()) + 1
        except (FileNotFoundError, ValueError, OSError):
            sequence = 1
        self.counter_path.write_text(f"{sequence}\n", encoding="utf-8")
        self.ticket_sequence = sequence
        self.ticket_created_epoch = time.time()
        self.ticket_path = self.slot_dir / (
            f"queue_{sequence:09d}_{os.getpid()}.json"
        )
        _write_json_atomic(self.ticket_path, {
            "sequence": sequence,
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "queued_at": datetime.now().isoformat(timespec="microseconds"),
            "queued_epoch": self.ticket_created_epoch,
        })

    def _queue_locked(self) -> List[Tuple[Path, dict]]:
        queue = []
        for path in sorted(self.slot_dir.glob("queue_*.json")):
            data = _load_json(path, {})
            if not data or not _pid_alive(data.get("pid")):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            queue.append((path, data))
        return queue

    def _inspect_slots_locked(self):
        candidate = None
        active = []
        for slot_index in range(1, self.config.max_concurrent + 1):
            path = self.slot_dir / f"slot_{slot_index:02d}.lock"
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                active.append(_load_json(path, {"slot": slot_index}))
                continue
            if candidate is None:
                candidate = (slot_index, path, handle)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        return candidate, active

    def _active_rss_gib(self, active: List[dict]) -> List[Optional[float]]:
        pids = [int(item["pid"]) for item in active if item.get("pid")]
        if not pids:
            return []
        try:
            totals = sample_process_trees(pids)
        except RuntimeError:
            return [None for _pid in pids]
        return [
            totals.get(pid, (0.0, 0))[0] / 1024.0
            if totals.get(pid, (0.0, 0))[1] > 0
            else None
            for pid in pids
        ]

    def _resource_decision_locked(self, active, queue_depth, waited_seconds):
        state = _load_json(self.state_path, {})
        snapshot = read_system_resources(state.get("cpu_ticks"))
        now = time.time()
        raw_cpu_percent = snapshot.cpu_percent
        previous_smoothed = state.get("smoothed_cpu_percent")
        previous_sample_epoch = state.get("cpu_sample_epoch")
        cpu_sample_max_age = max(2.0, 4.0 * self.config.poll_seconds)
        cpu_sample_age = (
            max(0.0, now - float(previous_sample_epoch))
            if previous_sample_epoch is not None
            else None
        )
        cpu_sample_source = "unavailable"
        if raw_cpu_percent is not None:
            cpu_percent = float(raw_cpu_percent)
            if previous_smoothed is not None:
                cpu_percent = 0.35 * cpu_percent + 0.65 * float(previous_smoothed)
            state["smoothed_cpu_percent"] = cpu_percent
            state["cpu_sample_epoch"] = now
            cpu_sample_age = 0.0
            cpu_sample_source = "sample"
        elif (
            previous_smoothed is not None
            and cpu_sample_age is not None
            and cpu_sample_age <= cpu_sample_max_age
        ):
            cpu_percent = float(previous_smoothed)
            cpu_sample_source = "recent_cache"
        else:
            cpu_percent = None
        state["cpu_ticks"] = snapshot.cpu_ticks
        state["resource_error"] = snapshot.error
        last_admission = state.get("last_admission_epoch")
        seconds_since = now - float(last_admission) if last_admission else None
        decision = evaluate_admission(
            config=self.config,
            active_count=len(active),
            active_rss_gib=self._active_rss_gib(active),
            available_memory_gib=snapshot.available_memory_gib,
            cpu_percent=cpu_percent,
            queue_depth=queue_depth,
            waited_seconds=waited_seconds,
            seconds_since_admission=seconds_since,
        )
        decision["resource_error"] = snapshot.error
        decision["raw_cpu_percent"] = raw_cpu_percent
        decision["cpu_sample_source"] = cpu_sample_source
        decision["cpu_sample_age_seconds"] = (
            round(cpu_sample_age, 6) if cpu_sample_age is not None else None
        )
        if decision["admit"]:
            state["last_admission_epoch"] = now
        _write_json_atomic(self.state_path, state)
        return decision

    def _report_wait(self, position, decision=None):
        now = time.time()
        if decision is None:
            message = f"queue_position={position}"
        else:
            reasons = ",".join(decision["reasons"]) or "ready"
            memory = decision.get("available_memory_gib")
            cpu = decision.get("cpu_percent")
            memory_text = (
                f"{memory:.2f}GiB" if memory is not None else "unknown"
            )
            message = (
                f"queue_position={position} wait={reasons} "
                f"available_ram={memory_text} "
            )
            message += f"cpu={cpu:.1f}%" if cpu is not None else "cpu=warming-up"
        if message != self._last_report or now - self._last_report_epoch >= 30.0:
            print(f"[run_pharaoh] {self.label} scheduler: {message}", flush=True)
            self._last_report = message
            self._last_report_epoch = now

    def acquire(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._coordinator():
            self._create_ticket_locked()

        try:
            while self.slot_handle is None:
                with self._coordinator():
                    queue = self._queue_locked()
                    paths = [path for path, _data in queue]
                    if self.ticket_path not in paths:
                        self._create_ticket_locked()
                        queue = self._queue_locked()
                        paths = [path for path, _data in queue]
                    position = paths.index(self.ticket_path) + 1
                    if position != 1:
                        self._report_wait(position)
                    else:
                        candidate, active = self._inspect_slots_locked()
                        waited_seconds = time.time() - self.ticket_created_epoch
                        if candidate is None:
                            decision = {
                                "admit": False,
                                "reasons": ["hard_limit"],
                                "active_count": len(active),
                                "queue_depth": len(queue),
                            }
                        elif self.config.adaptive:
                            decision = self._resource_decision_locked(
                                active, len(queue), waited_seconds
                            )
                        else:
                            decision = {
                                "admit": True,
                                "reasons": [],
                                "active_count": len(active),
                                "queue_depth": len(queue),
                                "mode": "fixed",
                            }

                        if candidate is not None and decision["admit"]:
                            self.slot_index, slot_path, self.slot_handle = candidate
                            try:
                                self.ticket_path.unlink()
                            except FileNotFoundError:
                                pass
                            self.admission = dict(decision)
                            self.admission.update({
                                "mode": "adaptive" if self.config.adaptive else "fixed",
                                "slot": self.slot_index,
                                "max_concurrent": self.config.max_concurrent,
                                "queue_sequence": self.ticket_sequence,
                                "waited_seconds": round(waited_seconds, 6),
                                "admitted_at": datetime.now().isoformat(timespec="microseconds"),
                                "admitted_epoch": time.time(),
                                "config": asdict(self.config),
                            })
                            self.slot_handle.seek(0)
                            self.slot_handle.truncate()
                            json.dump(
                                {
                                    "state": "active",
                                    "pid": os.getpid(),
                                    "run_dir": str(self.run_dir),
                                    **self.admission,
                                },
                                self.slot_handle,
                                indent=2,
                            )
                            self.slot_handle.flush()
                            _write_json_atomic(
                                self.run_dir / "stage4_scheduler.json",
                                {"admission": self.admission},
                            )
                        else:
                            if candidate is not None:
                                fcntl.flock(candidate[2].fileno(), fcntl.LOCK_UN)
                                candidate[2].close()
                            self._report_wait(position, decision)
                        _write_json_atomic(self.status_path, {
                            "updated_at": datetime.now().isoformat(timespec="microseconds"),
                            "head_run_dir": str(self.run_dir),
                            "queue_depth": len(queue),
                            "decision": decision,
                        })
                if self.slot_handle is None:
                    time.sleep(max(0.05, self.config.poll_seconds))
        except BaseException:
            if self.slot_handle is not None:
                self.release()
            else:
                self.cancel_ticket()
            raise

        resource_text = ""
        if self.config.adaptive:
            resource_text = (
                f"; available_ram={self.admission.get('available_memory_gib')}GiB, "
                f"cpu={self.admission.get('cpu_percent')}%"
            )
            if self.admission.get("memory_bypassed_when_idle"):
                # Say so: the job was let through below the RAM the policy asked for.
                resource_text += (
                    f" (admitted below the {self.admission.get('required_available_gib')}"
                    f"GiB the policy wanted -- nothing else was running, so waiting "
                    f"could not have freed any)"
                )
        print(
            f"[run_pharaoh] {self.label} {self.admission['mode']} slot "
            f"{self.slot_index}/{self.config.max_concurrent} admitted after "
            f"{self.admission['waited_seconds']:.2f}s{resource_text}",
            flush=True,
        )
        return self.admission

    def cancel_ticket(self):
        if self.ticket_path is None:
            return
        with self._coordinator():
            try:
                self.ticket_path.unlink()
            except FileNotFoundError:
                pass

    def release(self):
        if self.slot_handle is None:
            self.cancel_ticket()
            return
        released_at = datetime.now().isoformat(timespec="microseconds")
        with self._coordinator():
            try:
                record_path = self.run_dir / "stage4_scheduler.json"
                record = _load_json(record_path, {"admission": self.admission})
                record["released_at"] = released_at
                record["stage4_elapsed_seconds"] = round(
                    time.time() - float(self.admission["admitted_epoch"]),
                    6,
                )
                _write_json_atomic(record_path, record)
                self.slot_handle.seek(0)
                self.slot_handle.truncate()
                json.dump(
                    {
                        "state": "available",
                        "released_at": released_at,
                        "previous_run_dir": str(self.run_dir),
                        "slot": self.slot_index,
                        "max_concurrent": self.config.max_concurrent,
                    },
                    self.slot_handle,
                    indent=2,
                )
                self.slot_handle.flush()
            finally:
                fcntl.flock(self.slot_handle.fileno(), fcntl.LOCK_UN)
                self.slot_handle.close()
                self.slot_handle = None
        print(
            f"[run_pharaoh] {self.label} slot {self.slot_index}/"
            f"{self.config.max_concurrent} released",
            flush=True,
        )


@contextmanager
def stage4_admission(slot_dir, run_dir, config: Stage4SchedulerConfig, label="Stage 4"):
    controller = Stage4AdmissionController(slot_dir, run_dir, config, label)
    admission = controller.acquire()
    try:
        yield admission
    finally:
        controller.release()
