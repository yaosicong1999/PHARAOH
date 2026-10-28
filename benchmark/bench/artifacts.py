"""What a run directory keeps, and the fingerprint of its transform.

Shared by the runner (run.py) and the series clean-up (clean_series.py). Deletion goes
through keep_only, which removes a symbolic link itself and never what it points to: VALIS's
_valis_input/ holds links to the original Xenium images.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .config import read_json

# A warm-up keeps only its record and log.
WARMUP_KEEP = {"run.log", "run_record.json"}
# An unsuccessful run keeps its logs, records and timings; its intermediate products go.
LOG_KEEP = {"run.log", "run_record.json", "run_metadata.json", "pipeline_times.json",
            "registration_time.json", "timings.tsv"}
# A successful run of a timing-only series also keeps what describes its transform.
TIMING_ONLY_KEEP = LOG_KEEP | {"coordinate_contract.json", "images_info.json", "artifact_sha256.json",
                               "artifact_check.json"}


def removable(run_dir: Path, keep) -> list[Path]:
    """The top-level entries of run_dir that keep_only would remove."""
    return [path for path in sorted(run_dir.iterdir())
            if not (path.name in keep or (path.suffix == ".log" and path.is_file() and not path.is_symlink()))
            and not path.name.startswith("._")]


def keep_only(run_dir: Path, keep) -> None:
    """Remove everything in run_dir but the named files (and any *.log)."""
    for path in removable(run_dir, keep):
        # On an external macOS volume a ._ AppleDouble sidecar vanishes together with
        # its file, so anything may already be gone by the time it is reached.
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def disk_bytes(path: Path) -> int:
    """Allocated bytes of a file or tree, links counted as links (never followed)."""
    if path.is_symlink() or not path.is_dir():
        return path.lstat().st_blocks * 512
    total = 0
    for child in _walk(path):
        total += child.lstat().st_blocks * 512
    return total


def _walk(root: Path):
    import os
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            yield Path(current) / name


def transform_files(run_dir: Path) -> tuple[dict, list[Path]]:
    """(coordinate contract, the files it names as the run's transform)."""
    c = read_json(run_dir / "coordinate_contract.json")
    artifact = c["artifact"]
    files = [run_dir / artifact["path"]]
    if artifact.get("postprocessing_path"):
        files.append(run_dir / artifact["postprocessing_path"])
    if artifact["kind"] == "valis_native":
        files += sorted((run_dir / artifact["path"]).parent.glob("*.npy"))
    if artifact["kind"] == "matrix" and (run_dir / "transform_level0.csv").exists():
        files.append(run_dir / "transform_level0.csv")
    return c, files


def artifact_sha256(run_dir: Path) -> dict:
    """sha256 and size of the run's transform files, and of its coordinate frames."""
    c, files = transform_files(run_dir)
    out = {}
    for f in files:
        h = hashlib.sha256()
        with f.open("rb") as handle:
            while block := handle.read(16 << 20):
                h.update(block)
        out[str(f.relative_to(run_dir))] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
    return {"kind": c["artifact"]["kind"],
            "frames_sha256": hashlib.sha256(json.dumps(c["frames"], sort_keys=True).encode()).hexdigest(),
            "files": out}


def recorded_fingerprint(run_dir: Path):
    """A fingerprint already written into the run (artifact_sha256.json, or the files of an
    artifact_check.json), or None."""
    path = run_dir / "artifact_sha256.json"
    if path.exists():
        return read_json(path)
    path = run_dir / "artifact_check.json"
    if path.exists():
        check = read_json(path)
        return {"files": {k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in check["files"].items()}}
    return None
