# ============================================================
# cli_common.py - Shared helpers for the headless (CLI) pipeline
#
# The CLI pipeline reuses the compute in the two GUI "engines"
# (engine_dapi) without launching any Tk/Qt windows. Because the engine stage files have
# numeric names (e.g. "3_get_tiles.py") they cannot be imported with
# a normal `import`, so we load them by path with importlib.
# ============================================================
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


def load_module_from_path(mod_name: str, file_path):
    """Load a .py file (possibly with a numeric/illegal module name) as a module object."""
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # register so dataclasses / pickling inside the module behave
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_stage_event(run_dir, stage_key: str, event_name: str, **extra):
    """Append an event to <run_dir>/pipeline_times.json, mirroring the GUI stages' logging."""
    out_json = Path(run_dir) / "pipeline_times.json"
    now_str = datetime.now().isoformat(timespec="microseconds")
    data = {}
    if out_json.exists():
        try:
            data = load_json(out_json)
        except Exception:
            data = {}
    if stage_key not in data or not isinstance(data[stage_key], list):
        data[stage_key] = []
    rec = {"event": event_name, "time": now_str}
    rec.update(extra)
    data[stage_key].append(rec)
    save_json(out_json, data)


def _parse_iso_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _pipeline_start_from_events(data):
    """Return the latest explicit pipeline start, or the earliest known event."""
    explicit = []
    for rec in data.get("pipeline_events", []):
        if isinstance(rec, dict) and rec.get("event") == "pipeline_start":
            parsed = _parse_iso_time(rec.get("time"))
            if parsed is not None:
                explicit.append(parsed)
    if explicit:
        return max(explicit)

    all_times = []
    for key, records in data.items():
        if not key.endswith("_events") or not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            parsed = _parse_iso_time(rec.get("time"))
            if parsed is not None:
                all_times.append(parsed)
    return min(all_times) if all_times else None


def finalize_pipeline_timing(
    run_dir,
    *,
    stage7_elapsed_seconds=None,
    stage8_elapsed_seconds=None,
    total_elapsed_seconds=None,
    started_at=None,
    finished_at=None,
    source=None,
    completed_through_stage=7,
):
    """Write late-stage and total wall times while preserving event history.

    ``total_elapsed_seconds`` can be supplied by an end-to-end launcher for
    perf-counter precision. Otherwise it is calculated from the latest
    ``pipeline_start`` event, falling back to the earliest timestamp in older
    timing files that predate explicit pipeline events.
    """
    out_json = Path(run_dir) / "pipeline_times.json"
    data = {}
    if out_json.exists():
        try:
            data = load_json(out_json)
        except Exception:
            data = {}

    finished_dt = _parse_iso_time(finished_at) if finished_at else datetime.now()
    started_dt = _parse_iso_time(started_at) if started_at else _pipeline_start_from_events(data)
    if total_elapsed_seconds is None and started_dt is not None:
        total_elapsed_seconds = max(0.0, (finished_dt - started_dt).total_seconds())

    if stage7_elapsed_seconds is None:
        stage7_elapsed_seconds = data.get("stage7_elapsed_seconds")
    if stage8_elapsed_seconds is None:
        stage8_elapsed_seconds = data.get("stage8_elapsed_seconds")
    measured_stage_times = [
        float(value) for value in (stage7_elapsed_seconds, stage8_elapsed_seconds)
        if value is not None
    ]
    if measured_stage_times:
        # A standalone late-stage process can begin before its first timestamp
        # is logged (dependency imports happen first). The total cannot be
        # shorter than any measured stage duration.
        total_elapsed_seconds = max(
            float(total_elapsed_seconds or 0.0),
            max(measured_stage_times),
        )

    summary = {
        "started_at": started_dt.isoformat(timespec="microseconds") if started_dt else None,
        "finished_at": finished_dt.isoformat(timespec="microseconds"),
        "stage7_elapsed_seconds": (
            round(float(stage7_elapsed_seconds), 6)
            if stage7_elapsed_seconds is not None else None
        ),
        "stage8_elapsed_seconds": (
            round(float(stage8_elapsed_seconds), 6)
            if stage8_elapsed_seconds is not None else None
        ),
        "total_elapsed_seconds": (
            round(float(total_elapsed_seconds), 6)
            if total_elapsed_seconds is not None else None
        ),
        "completed_through_stage": int(completed_through_stage),
    }
    if source is not None:
        summary["source"] = str(source)

    data["stage7_elapsed_seconds"] = summary["stage7_elapsed_seconds"]
    data["stage8_elapsed_seconds"] = summary["stage8_elapsed_seconds"]
    data["total_elapsed_seconds"] = summary["total_elapsed_seconds"]
    data["pipeline_summary"] = summary
    save_json(out_json, data)
    return summary


def banner(msg: str):
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)
