"""The coordinate contract every wrapper writes and every evaluation reads.

A run's public point map is DAPI level-0 pixels -> H&E level-0 pixels of the two files the
method was given. A method works in its own input frames (a pyramid level, a re-encoded
file); each frame records how it scales to level 0, separately in x and y. A package's
native artifact may run the other way (RegWSI stores an H&E -> DAPI pull field); the
contract records its direction and whether evaluation must invert it, and refuses a
direction and an inversion flag that disagree.

    DAPI level0 -> DAPI input frame -> native map (or its checked inverse)
                -> H&E input frame -> H&E level0
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "benchmark_new/1"
FILE_NAME = "coordinate_contract.json"
DIRECTIONS = ("dapi_to_he", "he_to_dapi")


def scale_xy(frame: dict) -> tuple[float, float]:
    """Level-0 pixels per input-frame pixel, x and y."""
    sx = frame["level0_width"] / frame["width"]
    sy = frame["level0_height"] / frame["height"]
    if not all(math.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError("invalid frame scaling")
    return sx, sy


def input_frame(path, level: int, width: int, height: int, level0_width: int,
                level0_height: int, mpp_level0: float, **extra) -> dict:
    """One modality's input frame: which level of which file, at what size."""
    if min(width, height) < 2:
        raise ValueError("input frame must be at least 2x2")
    if level is None or int(level) < 0:
        raise ValueError("the pyramid level must be recorded, never inferred from a tier")
    frame = {"image_path": str(Path(path).resolve()), "pyramid_level_0based": int(level),
             "width": int(width), "height": int(height),
             "level0_width": int(level0_width), "level0_height": int(level0_height),
             "mpp_level0": None if mpp_level0 is None else float(mpp_level0)}   # null: unknown
    frame["level0_per_frame_px"] = list(scale_xy(frame))
    frame.update(extra)
    return frame


def frame_from_pyramid(path, level: int, mpp_level0: float, *, size_wh=None, series: int = 0,
                       **extra) -> dict:
    """Read dimensions only, never pixels."""
    from .pyramid import level_sizes
    sizes = level_sizes(path, series)
    if not 0 <= level < len(sizes):
        raise ValueError(f"pyramid level {level} out of range for {path}")
    width, height = size_wh or sizes[level]
    return input_frame(path, level, width, height, *sizes[0], mpp_level0, **extra)


def write(outdir, method: str, frames: dict, artifact: dict, *, native=None) -> dict:
    direction = artifact["stored_direction"]
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown stored direction {direction!r}")
    if bool(artifact.get("requires_inverse", False)) != (direction == "he_to_dapi"):
        raise ValueError("stored direction and inversion flag disagree")
    payload = {"schema": SCHEMA, "method": method, "point_direction": "dapi_to_he",
               "frames": frames, "artifact": artifact, "native": native or {},
               "metric_frame": "he_level0",
               "invalid_point_policy": "NaN plus a validity flag; never edge-extrapolated"}
    (Path(outdir) / FILE_NAME).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def write_matrix(outdir, method: str, frames: dict, matrix_level0, *, native=None) -> dict:
    """Store a DAPI->H&E level-0 homography, re-expressed in the two input frames."""
    import numpy as np
    to_dapi_level0 = np.diag([*scale_xy(frames["dapi"]), 1.0])
    to_he_frame = np.diag([*(1.0 / np.asarray(scale_xy(frames["he"]))), 1.0])
    matrix = to_he_frame @ np.asarray(matrix_level0, dtype=float) @ to_dapi_level0
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
        raise ValueError("nonfinite or singular matrix")
    np.savetxt(Path(outdir) / "transform_frames.csv", matrix, delimiter=",", fmt="%.17g")
    np.savetxt(Path(outdir) / "transform_level0.csv", np.asarray(matrix_level0, dtype=float),
               delimiter=",", fmt="%.17g")
    return write(outdir, method, frames,
                 {"kind": "matrix", "path": "transform_frames.csv",
                  "stored_direction": "dapi_to_he", "requires_inverse": False},
                 native=native)


def read(run_dir) -> dict:
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / FILE_NAME).read_text())
    if payload.get("schema") != SCHEMA or payload.get("point_direction") != "dapi_to_he":
        raise ValueError(f"{run_dir}: unsupported coordinate contract")
    for modality in ("dapi", "he"):
        frame = payload["frames"][modality]
        if min(frame["width"], frame["height"], frame["level0_width"], frame["level0_height"]) < 2:
            raise ValueError("invalid frame dimensions")
        if not all(math.isclose(a, b, rel_tol=1e-12)
                   for a, b in zip(scale_xy(frame), frame["level0_per_frame_px"])):
            raise ValueError("frame scaling disagrees with its dimensions")
        mpp = frame["mpp_level0"]
        if mpp is not None and not (math.isfinite(mpp) and mpp > 0):   # None: unknown, allowed
            raise ValueError("invalid physical pixel size")
    artifact = payload["artifact"]
    if bool(artifact.get("requires_inverse", False)) != (artifact["stored_direction"] == "he_to_dapi"):
        raise ValueError("inconsistent artifact direction")
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts or not (run_dir / relative).is_file():
        raise ValueError("missing or unsafe artifact path")
    return payload
