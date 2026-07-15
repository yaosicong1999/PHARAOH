# ============================================================
# cli_common.py - Shared helpers for the headless (CLI) pipeline
#
# The CLI pipeline reuses the compute in the two GUI "engines"
# (engine_dapi = version_4.0, engine_he = version_4.0_he) without
# launching any Tk/Qt windows. Because the engine stage files have
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
    now_str = datetime.now().isoformat(timespec="seconds")
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


def banner(msg: str):
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)
