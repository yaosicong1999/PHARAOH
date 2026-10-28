"""Locations and configuration files of benchmark_new.

Everything a run depends on is declared in `configs/` and `manifests/`; nothing here has
a hidden default that a config file cannot override.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MANIFESTS = ROOT / "manifests"
METHODS = ("pharaoh", "voltron", "regwsi", "valis_tifffile", "valis_bioformats")
# The software behind each method: methods/<package>/wrapper.py, and the interpreters in
# environment.json. The two VALIS methods run the same wrapper and differ only in how the
# DAPI reaches VALIS (VALIS_DAPI_READER).
PACKAGES = {"pharaoh": "pharaoh", "voltron": "voltron", "regwsi": "regwsi",
            "valis_tifffile": "valis", "valis_bioformats": "valis"}
VALIS_DAPI_READER = {"valis_tifffile": "tifffile", "valis_bioformats": "bioformats"}
GROUPS = ("keypoints", "matrix", "experimental")
EVAL_MODES = ("keypoints", "matrix")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path, payload) -> None:
    """Atomic: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _expand(value):
    if isinstance(value, str) and value.startswith("~"):
        return os.path.expanduser(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def environment() -> dict:
    """configs/environment.json, or the file BENCH_ENVIRONMENT names (e.g. for a smoke test
    that must not write into the real run tree)."""
    env = _expand(read_json(os.environ.get("BENCH_ENVIRONMENT") or CONFIGS / "environment.json"))
    # Relative locations are relative to this folder, so the benchmark runs wherever it is cloned.
    for key in ("runs_root", "cache_root", "results_root", "pharaoh_root"):
        path = Path(env[key])
        env[key] = str(path if path.is_absolute() else (ROOT / path).resolve())
    return env


def resources() -> dict:
    """configs/resources.json, or the file BENCH_RESOURCES names (e.g. a test that must not
    wait on the swap-settle gate)."""
    return read_json(os.environ.get("BENCH_RESOURCES") or CONFIGS / "resources.json")


def method_config(method: str, config_path=None) -> dict:
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; expected one of {METHODS}")
    path = Path(config_path) if config_path else CONFIGS / "methods" / f"{method}.json"
    config = read_json(path)
    if config.get("method") != method:
        raise SystemExit(f"{path} is a config for {config.get('method')!r}, not {method!r}")
    config["_path"] = str(path.resolve())
    return config


def runs_dir(group: str, method: str, series: str = "main") -> Path:
    """<runs_root>/<series>/<group>/<method>/ -- `series` keeps ablations apart."""
    return Path(environment()["runs_root"]) / series / group / method
