"""Process-tree memory sampling and dependency-free PHARAOH monitor plots."""

import csv
import ctypes
from contextlib import contextmanager, redirect_stdout
import html
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_STAGE_START_RE = re.compile(
    r"^#{4,}\s*(Stage\s+\d+[a-z]?\s+\([^)]+\))\s*#{4,}$",
    re.IGNORECASE,
)
_STAGE_SKIP_RE = re.compile(
    r"^###\s*(Stage\s+\d+[a-z]?\s+\([^)]+\)):\s*SKIP\b",
    re.IGNORECASE,
)
_STAGE_KEY_RE = re.compile(r"^(Stage\s+\d+[a-z]?)\b", re.IGNORECASE)

_STAGE_COLORS = {
    "startup": "#64748b",
    "stage 1": "#2563eb",
    "stage 2": "#ea580c",
    "stage 3": "#16a34a",
    "stage 4q": "#475569",
    "stage 4a": "#7c3aed",
    "stage 4b": "#dc2626",
    "stage 4c": "#a16207",
    "stage 5": "#db2777",
    "stage 6": "#0891b2",
    "stage 7": "#4d7c0f",
    "stage 8": "#9333ea",
}
_FALLBACK_COLORS = (
    "#0369a1",
    "#c2410c",
    "#15803d",
    "#6d28d9",
    "#be123c",
    "#0f766e",
)


@dataclass(frozen=True)
class MemorySample:
    elapsed_seconds: float
    rss_mib: float
    process_count: int
    stage: str


class _DarwinProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


_DARWIN_LIBPROC = None


class StageTracker:
    """Track stage transitions parsed from the top-level pipeline output."""

    def __init__(self, started_perf: float):
        self.started_perf = float(started_perf)
        self._events = [(0.0, "Startup")]
        self._lock = threading.Lock()

    def observe_line(self, line: str):
        stripped = line.strip()
        match = _STAGE_START_RE.match(stripped) or _STAGE_SKIP_RE.match(stripped)
        if match is None:
            return
        label = _normalize_stage_label(match.group(1))
        elapsed = max(0.0, time.perf_counter() - self.started_perf)
        with self._lock:
            if self._events[-1][1] != label:
                self._events.append((elapsed, label))

    def current_stage(self) -> str:
        with self._lock:
            return self._events[-1][1]

    def intervals(self, duration_seconds: float) -> List[dict]:
        duration = max(0.0, float(duration_seconds))
        with self._lock:
            events = list(self._events)
        intervals = []
        for index, (start, label) in enumerate(events):
            end = events[index + 1][0] if index + 1 < len(events) else duration
            start = min(max(0.0, float(start)), duration)
            end = min(max(start, float(end)), duration)
            intervals.append({
                "stage": label,
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "elapsed_seconds": round(end - start, 6),
            })
        return intervals


def _normalize_stage_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip())
    match = re.match(r"stage\s+(\d+[a-z]?)\s*(.*)", label, re.IGNORECASE)
    if match is None:
        return label
    return f"Stage {match.group(1).lower()} {match.group(2)}".strip()


def _stage_key(label: str) -> str:
    if label.casefold() == "startup":
        return "startup"
    match = _STAGE_KEY_RE.match(label)
    return match.group(1).casefold() if match else label.casefold()


def _stage_color(label: str) -> str:
    key = _stage_key(label)
    if key in _STAGE_COLORS:
        return _STAGE_COLORS[key]
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(key))
    return _FALLBACK_COLORS[checksum % len(_FALLBACK_COLORS)]


def parse_ps_table(output: str) -> Dict[int, Tuple[int, int]]:
    """Parse ``ps`` PID/PPID/RSS output into KiB records."""
    table = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        table[pid] = (ppid, max(0, rss_kib))
    return table


def read_process_table() -> Dict[int, Tuple[int, int]]:
    """Read a portable-enough macOS/Linux PID, parent PID, and RSS snapshot."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not sample process memory with ps: {exc}") from exc
    return parse_ps_table(result.stdout)


def process_tree_rss(
    root_pids: Iterable[int],
    table: Dict[int, Tuple[int, int]],
) -> Dict[int, Tuple[float, int]]:
    """Return summed RSS MiB and process count for each requested process tree."""
    children = {}
    for pid, (ppid, _rss_kib) in table.items():
        children.setdefault(ppid, []).append(pid)

    totals = {}
    for root_pid in root_pids:
        stack = [int(root_pid)]
        visited = set()
        rss_kib = 0
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            record = table.get(pid)
            if record is not None:
                rss_kib += record[1]
            stack.extend(children.get(pid, ()))
        process_count = sum(1 for pid in visited if pid in table)
        totals[int(root_pid)] = (rss_kib / 1024.0, process_count)
    return totals


def _darwin_libproc():
    global _DARWIN_LIBPROC
    if _DARWIN_LIBPROC is None:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_listchildpids.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_listchildpids.restype = ctypes.c_int
        library.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_pidinfo.restype = ctypes.c_int
        _DARWIN_LIBPROC = library
    return _DARWIN_LIBPROC


def _darwin_children(pid: int) -> List[int]:
    library = _darwin_libproc()
    capacity = 4096
    buffer = (ctypes.c_int * capacity)()
    count = library.proc_listchildpids(
        int(pid), buffer, ctypes.sizeof(buffer)
    )
    if count < 0:
        return []
    return [int(child) for child in buffer[:count] if child > 0]


def _darwin_rss_bytes(pid: int):
    library = _darwin_libproc()
    info = _DarwinProcTaskInfo()
    received = library.proc_pidinfo(
        int(pid), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
    )
    if received != ctypes.sizeof(info):
        return None
    return int(info.pti_resident_size)


def _darwin_process_trees(root_pids: Iterable[int]) -> Dict[int, Tuple[float, int]]:
    totals = {}
    for root_pid in root_pids:
        stack = [int(root_pid)]
        visited = set()
        rss_bytes = 0
        process_count = 0
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            stack.extend(_darwin_children(pid))
            rss = _darwin_rss_bytes(pid)
            if rss is not None:
                rss_bytes += rss
                process_count += 1
        totals[int(root_pid)] = (rss_bytes / (1024.0 ** 2), process_count)
    return totals


def _linux_children(pid: int) -> List[int]:
    path = Path(f"/proc/{int(pid)}/task/{int(pid)}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, PermissionError, ValueError):
        return []


def _linux_rss_bytes(pid: int):
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None
    return None


def _linux_process_trees(root_pids: Iterable[int]) -> Dict[int, Tuple[float, int]]:
    totals = {}
    for root_pid in root_pids:
        stack = [int(root_pid)]
        visited = set()
        rss_bytes = 0
        process_count = 0
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            stack.extend(_linux_children(pid))
            rss = _linux_rss_bytes(pid)
            if rss is not None:
                rss_bytes += rss
                process_count += 1
        totals[int(root_pid)] = (rss_bytes / (1024.0 ** 2), process_count)
    return totals


def sample_process_trees(root_pids: Iterable[int]) -> Dict[int, Tuple[float, int]]:
    """Sample specific process trees without requiring a system-wide process list."""
    root_pids = list(root_pids)
    try:
        if sys.platform == "darwin":
            return _darwin_process_trees(root_pids)
        if sys.platform.startswith("linux") and Path("/proc").is_dir():
            return _linux_process_trees(root_pids)
        table = read_process_table()
        return process_tree_rss(root_pids, table)
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"could not sample process-tree memory: {exc}") from exc


def sample_running_processes(running) -> None:
    """Append one process-tree RSS sample to each RunningDataset-like object."""
    totals = sample_process_trees(item.process.pid for item in running)
    now = time.perf_counter()
    for item in running:
        rss_mib, process_count = totals.get(item.process.pid, (0.0, 0))
        item.monitor_samples.append(MemorySample(
            elapsed_seconds=max(0.0, now - item.started_perf),
            rss_mib=round(float(rss_mib), 6),
            process_count=int(process_count),
            stage=item.stage_tracker.current_stage(),
        ))


class _StageTrackingStream:
    """Forward stdout unchanged, observing only complete top-level banner lines.

    Stage subprocesses still inherit the native stdout descriptor directly: no
    extra pipe, buffering policy, or subprocess lifecycle is introduced.
    """

    def __init__(self, stream, tracker):
        self.stream = stream
        self.tracker = tracker
        self.pending = ""

    def write(self, text):
        result = self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.tracker.observe_line(line)
        return result

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def monitor_current_process(output_dir, *, enabled=False, interval=1.0):
    """Optional single-run monitor using the batch monitor's sampler and renderer.

    Report the current invocation (including resume skips), not historical run
    time. Finalize on success, early return, SystemExit, and KeyboardInterrupt.
    Sampling/output failures are diagnostic only and never change pipeline exit
    behavior. Input conversion before the run starts is outside this context.
    """
    if not enabled:
        yield
        return
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("monitor interval must be finite and greater than 0")
    output_dir = Path(output_dir)
    started = time.perf_counter()
    tracker = StageTracker(started)
    samples = []
    stop = threading.Event()
    sampling_error = []
    pid = os.getpid()
    status = "completed"

    def sample():
        try:
            rss, count = sample_process_trees([pid])[pid]
            if count < 1:
                raise RuntimeError("current process RSS is unavailable")
            samples.append(MemorySample(
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                rss_mib=round(float(rss), 6), process_count=int(count),
                stage=tracker.current_stage(),
            ))
        except Exception as exc:
            sampling_error.append(str(exc))
            stop.set()

    def sample_loop():
        while not stop.wait(interval):
            sample()

    worker = threading.Thread(target=sample_loop, name="pharaoh-memory-monitor", daemon=True)
    sample()  # Even a completely skipped/very short run gets a first sample.
    worker_started = False
    try:
        worker.start()
        worker_started = True
    except Exception as exc:
        sampling_error.append(str(exc))
        stop.set()
    try:
        print(f"[monitor] process-tree RSS every {interval:g}s -> {output_dir}", flush=True)
        with redirect_stdout(_StageTrackingStream(sys.stdout, tracker)):
            yield
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, SystemExit) and exc.code in (None, 0):
            status = "completed"
        else:
            status = "failed"
        raise
    finally:
        stop.set()
        if worker_started:
            worker.join()
        if not sampling_error:
            sample()
        elapsed = time.perf_counter() - started
        try:
            result = write_monitor_artifacts(
                output_dir=output_dir, sample_name=output_dir.name,
                samples=samples, stage_intervals=tracker.intervals(elapsed),
                duration_seconds=elapsed, sample_interval_seconds=interval,
            )
            result.update(status=status, duration_seconds=round(elapsed, 6))
            if sampling_error:
                result["error"] = sampling_error[0]
            (output_dir / "monitor_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"[monitor] plot: {result['plot']}", flush=True)
        except Exception as exc:
            print(f"[monitor] could not save monitor outputs: {exc}", file=sys.stderr)
        if sampling_error:
            print(f"[monitor] sampling stopped: {sampling_error[0]}", file=sys.stderr)


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(value))
    fraction = value / exponent
    for nice in (1.0, 2.0, 5.0, 10.0):
        if fraction <= nice:
            return nice * exponent
    return 10.0 * exponent


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


def _stage_legend_label(label: str) -> str:
    match = re.match(r"^(Stage\s+\d+[a-z]?)\s+\((.+)\)$", label)
    if match:
        return f"{match.group(1)} — {match.group(2)}"
    return label


def _downsample(samples: Sequence[MemorySample], limit: int = 2400):
    if len(samples) <= limit:
        return list(samples)
    stride = int(math.ceil(len(samples) / float(limit)))
    selected = list(samples[::stride])
    if selected[-1] != samples[-1]:
        selected.append(samples[-1])
    return selected


def render_memory_time_svg(
    *,
    sample_name: str,
    samples: Sequence[MemorySample],
    stage_intervals: Sequence[dict],
    duration_seconds: float,
    sample_interval_seconds: float,
) -> str:
    """Render a standalone memory/time SVG with stage-colored time zones."""
    width = 1200
    height = 680
    left = 88
    right = 34
    bottom = 70
    unique_stages = []
    for interval in stage_intervals:
        label = str(interval["stage"])
        if label not in unique_stages:
            unique_stages.append(label)

    legend_widths = [
        max(108, min(270, 42 + len(_stage_legend_label(label)) * 7))
        for label in unique_stages
    ]
    legend_rows = 1
    cursor = left
    usable_width = width - left - right
    for item_width in legend_widths:
        if cursor + item_width > left + usable_width:
            legend_rows += 1
            cursor = left
        cursor += item_width
    top = 104 + legend_rows * 24
    plot_width = width - left - right
    plot_height = height - top - bottom

    duration = max(float(duration_seconds), 0.001)
    peak_sample = max(samples, key=lambda item: item.rss_mib) if samples else None
    peak_mib = peak_sample.rss_mib if peak_sample is not None else 0.0
    display_gib = peak_mib >= 2048.0
    memory_divisor = 1024.0 if display_gib else 1.0
    memory_unit = "GiB" if display_gib else "MiB"
    y_max = _nice_ceiling(max(peak_mib / memory_divisor * 1.08, 1.0))

    def x_pos(seconds: float) -> float:
        return left + min(max(float(seconds), 0.0), duration) / duration * plot_width

    def y_pos(rss_mib: float) -> float:
        value = max(0.0, float(rss_mib) / memory_divisor)
        return top + plot_height - min(value, y_max) / y_max * plot_height

    escaped_name = html.escape(str(sample_name))
    subtitle = (
        f"Peak {peak_mib / memory_divisor:.2f} {memory_unit}"
        f"  •  Duration {_format_duration(duration)}"
        f"  •  Sample interval {sample_interval_seconds:g}s"
        "  •  RSS includes the full process tree"
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" '
        'viewBox="0 0 1200 680" role="img">',
        f"<title>{escaped_name} memory and pipeline time</title>",
        "<desc>Total resident memory over elapsed time. Background zones "
        "encode PHARAOH stages.</desc>",
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "sans-serif; fill: #172033; }",
        ".title { font-size: 22px; font-weight: 600; }",
        ".subtitle { font-size: 13px; fill: #526077; }",
        ".axis { font-size: 12px; fill: #526077; }",
        ".axis-title { font-size: 13px; font-weight: 600; }",
        ".zone-label { font-size: 12px; font-weight: 600; }",
        ".legend { font-size: 12px; }",
        "</style>",
        '<rect width="1200" height="680" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="34">{escaped_name} — memory vs time</text>',
        f'<text class="subtitle" x="{left}" y="58">{html.escape(subtitle)}</text>',
    ]

    legend_x = left
    legend_y = 84
    for label, item_width in zip(unique_stages, legend_widths):
        if legend_x + item_width > left + usable_width:
            legend_x = left
            legend_y += 24
        color = _stage_color(label)
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 11}" width="12" height="12" '
            f'fill="{color}" rx="2"/>'
        )
        parts.append(
            f'<text class="legend" x="{legend_x + 18}" y="{legend_y}">'
            f'{html.escape(_stage_legend_label(label))}</text>'
        )
        legend_x += item_width

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#ffffff" stroke="#cbd5e1"/>'
    )

    for interval_index, interval in enumerate(stage_intervals):
        label = str(interval["stage"])
        zone_x = x_pos(interval["start_seconds"])
        zone_end = x_pos(interval["end_seconds"])
        zone_width = max(0.0, zone_end - zone_x)
        color = _stage_color(label)
        parts.append(
            f'<rect x="{zone_x:.2f}" y="{top}" width="{zone_width:.2f}" '
            f'height="{plot_height}" fill="{color}" fill-opacity="0.13"/>'
        )
        if zone_width >= 58:
            parts.append(
                f'<text class="zone-label" x="{zone_x + zone_width / 2:.2f}" '
                f'y="{top + 20}" text-anchor="middle" fill="{color}">'
                f'{html.escape(_stage_key(label).title())}</text>'
            )
        if interval_index > 0:
            parts.append(
                f'<line x1="{zone_x:.2f}" y1="{top}" x2="{zone_x:.2f}" '
                f'y2="{top + plot_height}" stroke="{color}" stroke-opacity="0.55"/>'
            )

    for tick in range(7):
        value = y_max * tick / 6.0
        y = top + plot_height - plot_height * tick / 6.0
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#dbe3ee" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="axis" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f'{value:.1f}</text>'
        )

    for tick in range(9):
        value = duration * tick / 8.0
        x = left + plot_width * tick / 8.0
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#e5eaf1" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="axis" x="{x:.2f}" y="{top + plot_height + 24}" '
            f'text-anchor="middle">{html.escape(_format_duration(value))}</text>'
        )

    plotted = _downsample(samples)
    if plotted:
        points = " ".join(
            f"{x_pos(item.elapsed_seconds):.2f},{y_pos(item.rss_mib):.2f}"
            for item in plotted
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#172033" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if peak_sample is not None:
        peak_x = x_pos(peak_sample.elapsed_seconds)
        peak_y = y_pos(peak_sample.rss_mib)
        anchor = "end" if peak_x > left + plot_width * 0.78 else "start"
        label_x = peak_x - 9 if anchor == "end" else peak_x + 9
        label_y = max(top + 42, peak_y - 10)
        parts.extend([
            f'<circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="4.5" fill="#172033"/>',
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" '
            f'font-size="12" font-weight="600">Peak {peak_mib / memory_divisor:.2f} '
            f'{memory_unit}</text>',
        ])

    parts.extend([
        f'<text class="axis-title" x="{left + plot_width / 2:.2f}" y="{height - 18}" '
        'text-anchor="middle">Elapsed time</text>',
        f'<text class="axis-title" transform="translate(24 {top + plot_height / 2:.2f}) '
        f'rotate(-90)" text-anchor="middle">Total RSS ({memory_unit})</text>',
        "</svg>",
    ])
    return "\n".join(parts) + "\n"


def write_monitor_artifacts(
    *,
    output_dir: Path,
    sample_name: str,
    samples: Sequence[MemorySample],
    stage_intervals: Sequence[dict],
    duration_seconds: float,
    sample_interval_seconds: float,
) -> dict:
    """Write monitor CSV/SVG and return a JSON-ready result summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "monitor_memory_time.csv"
    svg_path = output_dir / "monitor_memory_time.svg"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("elapsed_seconds", "rss_mib", "process_count", "stage"))
        for sample in samples:
            writer.writerow((
                f"{sample.elapsed_seconds:.6f}",
                f"{sample.rss_mib:.6f}",
                sample.process_count,
                sample.stage,
            ))

    svg_path.write_text(
        render_memory_time_svg(
            sample_name=sample_name,
            samples=samples,
            stage_intervals=stage_intervals,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
        ),
        encoding="utf-8",
    )

    peak = max(samples, key=lambda item: item.rss_mib) if samples else None
    return {
        "metric": "sum_rss_process_tree",
        "sample_interval_seconds": float(sample_interval_seconds),
        "sample_count": len(samples),
        "peak_rss_mib": round(peak.rss_mib, 6) if peak is not None else None,
        "peak_elapsed_seconds": (
            round(peak.elapsed_seconds, 6) if peak is not None else None
        ),
        "peak_stage": peak.stage if peak is not None else None,
        "csv": str(csv_path.resolve()),
        "plot": str(svg_path.resolve()),
        "stage_intervals": list(stage_intervals),
    }
