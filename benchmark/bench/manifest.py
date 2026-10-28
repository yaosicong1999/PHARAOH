"""Group manifests: manifests/<group>.json, each {"group": ..., "datasets": [...]}."""
from __future__ import annotations

import math

from pathlib import Path

from .config import GROUPS, MANIFESTS, ROOT, read_json

REQUIRED = ("name", "dapi", "he", "dapi_level0", "he_level0", "dapi_mpp", "he_mpp")
# Image and reference paths may be given relative to this folder (download_test_data.sh writes data/).
PATH_KEYS = ("dapi", "he", "transcripts", "nucleus_boundaries")
# he_mpp may be left null when the H&E image carries no pixel size (e.g. a plain JPEG); the entry
# then says why in "he_mpp_unknown", and running a method on it stops with that reason (run.py)
OPTIONAL_WITH_REASON = {"he_mpp": "he_mpp_unknown"}


def load(group: str, names=None) -> list[dict]:
    """The datasets of manifests/<group>.json."""
    if group not in GROUPS:
        raise SystemExit(f"unknown group {group!r}; expected one of {GROUPS}")
    found, datasets = load_file(MANIFESTS / f"{group}.json", names)
    if found != group:
        raise SystemExit(f"manifests/{group}.json declares group {found!r}")
    return datasets


def load_file(path, names=None) -> tuple[str, list[dict]]:
    """(group, datasets) of any manifest file, e.g. a hand-made subset of a group. Its
    "group" decides where runs are written and how they are evaluated."""
    payload = read_json(path)
    group = payload.get("group")
    if group not in GROUPS:
        raise SystemExit(f"{path}: \"group\" must be one of {GROUPS}, not {group!r}")
    datasets = payload["datasets"]
    seen = set()
    for entry in datasets:
        _absolute_paths(entry)
        missing = [key for key in REQUIRED if entry.get(key) is None
                   and not (key in OPTIONAL_WITH_REASON and entry.get(OPTIONAL_WITH_REASON[key]))]
        if missing:
            raise SystemExit(f"{path}/{entry.get('name')}: missing {', '.join(missing)}")
        for key in ("dapi_mpp", "he_mpp"):
            if entry.get(key) is None:
                continue                      # unknown, with its reason (checked above)
            if not (math.isfinite(float(entry[key])) and float(entry[key]) > 0):
                raise SystemExit(f"{path}/{entry['name']}: invalid {key}")
        if entry["name"] in seen:
            raise SystemExit(f"{path}: duplicate dataset {entry['name']}")
        seen.add(entry["name"])
    if names:
        unknown = sorted(set(names) - seen)
        if unknown:
            raise SystemExit(f"{path}: unknown dataset(s) {', '.join(unknown)}")
        datasets = [d for d in datasets if d["name"] in set(names)]
    return group, datasets


def _absolute_paths(entry: dict) -> None:
    """Resolve relative image and reference paths against this folder, in place."""
    for key in PATH_KEYS:
        if isinstance(entry.get(key), str) and not Path(entry[key]).is_absolute():
            entry[key] = str((ROOT / entry[key]).resolve())
    alignment = entry.get("alignment")
    if isinstance(alignment, dict) and isinstance(alignment.get("path"), str) \
            and not Path(alignment["path"]).is_absolute():
        alignment["path"] = str((ROOT / alignment["path"]).resolve())
