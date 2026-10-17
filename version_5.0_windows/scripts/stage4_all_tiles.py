"""Explicit full-GUI Stage-4 scope; the default CLI subset policy is unchanged."""
import hashlib
import json
from pathlib import Path

import run_roles

COMPLETION_FILE = "stage4b_all_tiles.json"


def shared_stage3_ids(tiles_dir, roles=None):
    tiles = Path(tiles_dir)
    roles = roles or run_roles.read(tiles.parent)
    moving = json.loads((tiles / f"{roles.moving.token}_tile_info.json").read_text())
    fixed = json.loads((tiles / f"{roles.fixed.token}_tile_info.json").read_text())
    return sorted(set(moving).intersection(fixed))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_all_tiles_completion(tiles_dir, tile_ids, write_qa):
    tiles = Path(tiles_dir)
    record = {
        "tile_ids": list(tile_ids), "write_qa": bool(write_qa),
        "anchors_sha256": fingerprint(tiles / "standout_nuclei.json"),
        "masks_sha256": fingerprint(tiles / "nuclei_mask_info.json"),
    }
    (tiles / COMPLETION_FILE).write_text(json.dumps(record, indent=2))


def all_tiles_detection_ready(tiles_dir):
    tiles = Path(tiles_dir)
    try:
        record = json.loads((tiles / COMPLETION_FILE).read_text())
        expected = shared_stage3_ids(tiles)
        return (bool(expected) and set(record["tile_ids"]) == set(expected)
                and record["write_qa"]
                and record["anchors_sha256"] == fingerprint(tiles / "standout_nuclei.json")
                and record["masks_sha256"] == fingerprint(tiles / "nuclei_mask_info.json")
                and all((tiles / f"{key}_standout.jpg").is_file() for key in expected))
    except (OSError, ValueError, KeyError, TypeError):
        return False
