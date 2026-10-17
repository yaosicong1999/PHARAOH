"""Deterministic spatial subsampling for DAPI Stage 4b."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MANIFEST_NAME = "spatially_selected_tiles.json"


def _tile_center(meta):
    if "cx" in meta and "cy" in meta:
        return float(meta["cx"]), float(meta["cy"])
    return (
        float(meta.get("x0", 0.0)) + float(meta.get("w", 0.0)) / 2.0,
        float(meta.get("y0", 0.0)) + float(meta.get("h", 0.0)) / 2.0,
    )


def select_spatially_uniform_tile_ids(
    tile_info, limit, eligible_ids=None, *, anchor_ids=None,
):
    """Select tiles with deterministic normalized farthest-point sampling.

    Without anchors, the first tile is nearest the candidate-set centroid. With
    anchors (for example, tiles already masked before a top-up), selection starts
    from those locations and therefore fills the largest remaining spatial gaps.
    Each subsequent tile maximizes its distance to the selected/anchored set.
    Normalizing each axis prevents a long slide dimension from dominating the
    selection.
    """
    eligible = set(tile_info) if eligible_ids is None else set(eligible_ids)
    anchors = set() if anchor_ids is None else set(anchor_ids).intersection(tile_info)
    keys = sorted(key for key in tile_info if key in eligible and key not in anchors)
    limit = int(limit)
    if limit < 1:
        raise ValueError("tile selection limit must be >= 1")
    if len(keys) <= limit:
        return keys

    centers = np.asarray([_tile_center(tile_info[key]) for key in keys], dtype=np.float64)
    anchor_keys = sorted(anchors)
    if anchor_keys:
        anchor_centers = np.asarray(
            [_tile_center(tile_info[key]) for key in anchor_keys], dtype=np.float64,
        )
        normalization_centers = np.vstack([centers, anchor_centers])
    else:
        anchor_centers = None
        normalization_centers = centers
    lo = normalization_centers.min(axis=0)
    span = np.maximum(normalization_centers.max(axis=0) - lo, 1.0)
    normalized = (centers - lo) / span
    if anchor_centers is not None:
        normalized_anchors = (anchor_centers - lo) / span
        minimum_distance2 = np.min(
            np.sum(
                (normalized[:, None, :] - normalized_anchors[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        selected = []
    else:
        centroid = normalized.mean(axis=0)
        first = int(np.argmin(np.sum((normalized - centroid) ** 2, axis=1)))
        selected = [first]
        minimum_distance2 = np.sum((normalized - normalized[first]) ** 2, axis=1)
        minimum_distance2[first] = -1.0

    while len(selected) < limit:
        next_index = int(np.argmax(minimum_distance2))
        selected.append(next_index)
        distance2 = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[selected] = -1.0
    return [keys[index] for index in selected]


def _moving_token(tiles_dir):
    """The moving side's token, from the run above `tiles_dir`; `dapi` when there is
    no run to ask (the unit tests build bare tile directories)."""
    try:
        import run_roles
        return run_roles.read(Path(tiles_dir).parent).moving.token
    except Exception:
        return "dapi"


def create_selection_manifest(tiles_dir, limit, eligible_ids=None, source="stage4b", *, persist=True):
    """Build the shared selection; a GUI preview can opt out of writing it."""
    tiles_dir = Path(tiles_dir)
    info_path = tiles_dir / f"{_moving_token(tiles_dir)}_tile_info.json"
    tile_info = json.loads(info_path.read_text())
    eligible = sorted(set(tile_info) if eligible_ids is None else set(eligible_ids).intersection(tile_info))
    selected = select_spatially_uniform_tile_ids(tile_info, int(limit), eligible)
    manifest = {
        "method": "deterministic_normalized_farthest_point",
        "source": str(source),
        "tile_info_path": str(info_path.resolve()),
        "available_stage3_tiles": int(len(tile_info)),
        "eligible_tiles": int(len(eligible)),
        "requested_tiles_for_nuclei_detection": int(limit),
        "selected_tile_count": int(len(selected)),
        "selected_tile_ids": selected,
        "selected_centers_in_dapi_tile_coordinates": {
            key: list(_tile_center(tile_info[key])) for key in selected
        },
    }
    path = tiles_dir / MANIFEST_NAME
    if persist:
        path.write_text(json.dumps(manifest, indent=2))
    return path, manifest


def load_selection_manifest(tiles_dir):
    path = Path(tiles_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data.get("selected_tile_ids"), list):
        raise ValueError(f"invalid tile-selection manifest: {path}")
    return data
