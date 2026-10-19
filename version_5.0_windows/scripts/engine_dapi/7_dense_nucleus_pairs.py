#!/usr/bin/env python3
"""Stage 7: dense cross-modality nucleus-pair export.

The final Stage-5 transform maps original moving level-0 pixels to original fixed
level-0 pixels. This stage densifies the tissue-sampled Stage-3 tiles (from Stage
4a's detection cache when it is valid), restores every local centroid to original
level-0 coordinates, and performs a one-to-one assignment. The whole-slide chunk
scan that used to sit beside it was removed in 4.11: nothing ran it, and a census of
every nucleus on the slide is not what registration needs.

Outputs are written to ``<RUN_DIR>/dense_nucleus_pairs``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
_PROCESS_STARTED = time.perf_counter()
from pathlib import Path
from typing import Dict, Iterable, Tuple

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
from cli_common import finalize_pipeline_timing, log_stage_event
from spatial_tile_selection import select_spatially_uniform_tile_ids
from my_utils import qc_images_enabled
import physical_scale
import local_residual
from dense_nucleus_detection import (
    CACHE_META_NAME,
    CACHE_NPZ_NAME,
    CACHE_SCHEMA_VERSION,
    detection_params_snapshot,
    segment_dapi_centroids,
    segment_he_centroids,
)

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

DEFAULTS = {
    "scan_mode": "stage3_tiles",
    # um/px the `*_level0_px` gates below were tuned at, per modality; an image
    # scanned at another pixel size has its gates rescaled (scripts/physical_scale.py)
    "reference_mpp_dapi": physical_scale.REFERENCE_MPP["dapi"],
    "reference_mpp_he": physical_scale.REFERENCE_MPP["he"],
    "reuse_stage4a_detection_cache": True,
    # 0 means no cap: scan every Stage-3 tile. Stage 3 already sizes its own tile
    # count from the tissue it measured, so a second, unrelated cap here can only
    # silently discard work that was already done.
    "max_stage3_tiles": 0,
    "max_output_pairs": 0,
    "tile_dedup_radius_level0_px": 3.0,
    "assignment_mode": "greedy_one_to_one",
    "candidate_neighbors": 5,
    "area_cost_weight": 0.15,
    "dapi_threshold_offset": 0,
    "dapi_min_contrast_fraction": 0.002,
    "dapi_max_foreground_fraction": 0.65,
    "dapi_min_area_level0_px": 20.0,
    "dapi_max_area_level0_px": 5000.0,
    "he_n_smooth": 2,
    "he_intensity_threshold": 0.6,
    "he_upsample_factor": 1,
    "he_min_tissue_fraction": 0.002,
    "he_min_area_level0_px": 30.0,
    "he_max_area_level0_px": 5000.0,
    "candidate_distance_level0_px": 60.0,
    "max_pair_distance_level0_px": 35.0,
    "high_confidence_distance_level0_px": 15.0,
    "max_calibrated_area_ratio": 4.0,
    "high_confidence_area_ratio": 2.0,
    "calibrate_area_between_modalities": True,
    "write_nearest_candidates": True,
}

NUCLEUS_COLUMNS = [
    "x_work", "y_work", "x_level0", "y_level0",
    "area_work_px", "area_level0_px", "source_chunk",
]

STAGE4_ANCHOR_COLOR_BGR = (0, 255, 0)
STAGE7_ANCHOR_COLOR_BGR = (0, 0, 255)

TREE_QC_DIR = "tree_qc"
#: the classes an edge of the candidate search can end up in, and how each is drawn
TREE_EDGE_COLORS_BGR = {
    "high": (0, 200, 0),          # selected pair, high-confidence tier
    "medium": (0, 215, 255),      # selected pair, medium tier
    "candidate": (170, 170, 170),  # passed the distance/area gate, lost the one-to-one
    "rejected": (0, 0, 220),      # failed the gate
}
TREE_HE_NUCLEUS_BGR = (255, 90, 0)
TREE_PROJECTED_DAPI_BGR = (255, 0, 255)


from pyramid_region_reader import PyramidRegionReader


class Sides:
    """The two images by role: `moving` is projected into `fixed`'s frame and matched
    there. Every name in this file used to say dapi (moving) and he (fixed), which is
    what the default -- no roles -- still means; an `he-dapi` run swaps the tokens and
    modalities and nothing else. Tokens name files and columns, modalities choose
    detectors and area gates.
    """

    def __init__(self, roles=None):
        if roles is None:
            self.moving_token, self.fixed_token = "dapi", "he"
            self.moving_modality, self.fixed_modality = "dapi", "he"
            self.pair = "dapi-he"
            self.homography_stem = "dapi_to_he_homography_level0"
            self.roles = None
        else:
            self.moving_token, self.fixed_token = roles.moving.token, roles.fixed.token
            self.moving_modality = roles.moving.modality
            self.fixed_modality = roles.fixed.modality
            self.pair = roles.pair.name
            self.homography_stem = roles.pair.homography_stem
            self.roles = roles

    @property
    def tokens(self):
        return self.moving_token, self.fixed_token

    @property
    def mixed(self):
        return self.moving_modality != self.fixed_modality

    def image_path(self, role, info):
        """The image behind a role: from the roles block, else the legacy top-level key."""
        if self.roles is not None:
            return getattr(self.roles, role).path
        token = self.moving_token if role == "moving" else self.fixed_token
        return info[f"{token.upper()}_path"]

    def level(self, role, info, default=0):
        if self.roles is not None:
            return int(getattr(self.roles, role).level)
        token = self.moving_token if role == "moving" else self.fixed_token
        return int(info.get(f"{token.upper()}_level", default))

    def channel_index(self, role, info):
        if self.roles is not None:
            return getattr(self.roles, role).channel_index
        token = self.moving_token if role == "moving" else self.fixed_token
        value = info.get(f"{token.upper()}_channel_index")
        return None if value is None else int(value)

    def orientation(self, info):
        """The moving image's display orientation, for the anchor QA render."""
        token = self.moving_token.upper()
        if self.roles is not None:
            return int(self.roles.moving.orientation_case), (
                self.roles.moving.gui_affine or np.eye(3).tolist())
        return int(info.get(f"{token}_orientation_case", 0)), info.get(f"{token}_gui_affine", np.eye(3))


def sides_for(info):
    """Sides for a run's `images_info.json` record; dapi/he when it has no roles block
    the role reader can use (the synthetic runs the tests build)."""
    try:
        import run_roles
        return Sides(run_roles.read(info))
    except Exception:
        return Sides()


def area_gate_keys(modality):
    """The min/max level-0 area gates for nuclei of one modality."""
    return f"{modality}_min_area_level0_px", f"{modality}_max_area_level0_px"


def segment_centroids(modality, image, params, engine_dir):
    """The dense detector for one modality, on one tile image as read from disk."""
    if modality == "dapi":
        if image.ndim == 3:
            image = image[..., 0]
        return segment_dapi_centroids(image, params)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return segment_he_centroids(cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB), params, engine_dir)


def rect_to_fixed(tile_meta, fixed_token):
    """The fixed tile's rectified-to-image matrix (Stage 3 writes `M_rect_to_he` for the
    fixed side whatever its modality), or a plain translation for an unrectified tile."""
    meta = tile_meta.get("meta", {})
    for key in ("M_rect_to_he", f"M_rect_to_{fixed_token}"):
        if key in meta:
            return np.asarray(meta[key], dtype=np.float64)
    return np.array([
        [1.0, 0.0, float(tile_meta.get("x0", 0.0))],
        [0.0, 1.0, float(tile_meta.get("y0", 0.0))],
        [0.0, 0.0, 1.0],
    ])


def _to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating) and np.nanmax(a) <= 1.0:
        return np.clip(a * 255.0, 0, 255).astype(np.uint8)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (0.2, 99.8))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def apply_orientation_case(image: np.ndarray, case_id: int) -> np.ndarray:
    """Apply Stage 1's discrete DAPI display orientation."""
    if case_id == 0:
        return image
    if case_id == 1:
        return np.rot90(image, k=3)
    if case_id == 2:
        return np.rot90(image, k=2)
    if case_id == 3:
        return np.rot90(image, k=1)
    if case_id == 4:
        return np.flipud(image)
    if case_id == 5:
        return np.fliplr(image)
    if case_id == 6:
        return np.transpose(image, (1, 0, 2))
    if case_id == 7:
        return np.fliplr(np.rot90(image, k=1))
    raise ValueError(f"unknown orientation case: {case_id}")


def transform_display_points(points_level0, scale_xy, affine=None):
    points = np.asarray(points_level0, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    scale = np.asarray(scale_xy, dtype=np.float64).reshape(1, 2)
    points = points / scale
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape == (2, 3):
            affine = np.vstack([affine, [0.0, 0.0, 1.0]])
        homogeneous = np.column_stack([points, np.ones(len(points))])
        mapped = (affine @ homogeneous.T).T
        points = mapped[:, :2] / mapped[:, 2:3]
    return points


def render_anchor_overlay(
    base_bgr,
    stage4_points_level0,
    stage7_points_level0,
    scale_xy,
    *,
    orientation_case=0,
    orientation_affine=None,
    modality_label="image",
):
    """Draw Stage-4 anchor rings and Stage-7 high-confidence anchor dots."""
    # `np.rot90` / `flipud` / `fliplr` hand back views with negative strides, and
    # OpenCV's drawing calls refuse those ("Layout of the output array img is
    # incompatible with cv::Mat"). `cvtColor`, `resize` and `imwrite` accept them, which
    # is why only this renderer noticed: every other orientation-case consumer either
    # converts or writes. Turning `qc_images.stage7_anchors` on for a run whose Stage 1
    # rotated the DAPI crashed on the first `cv2.circle`.
    image = np.ascontiguousarray(
        apply_orientation_case(np.asarray(base_bgr).copy(), int(orientation_case))
    )
    stage4 = transform_display_points(stage4_points_level0, scale_xy, orientation_affine)
    stage7 = transform_display_points(stage7_points_level0, scale_xy, orientation_affine)
    h, w = image.shape[:2]

    def valid(points):
        if len(points) == 0:
            return points
        finite = np.isfinite(points).all(axis=1)
        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < w) &
            (points[:, 1] >= 0) & (points[:, 1] < h)
        )
        return points[finite & inside]

    stage4 = valid(stage4)
    stage7 = valid(stage7)
    for x, y in np.rint(stage4).astype(int):
        cv2.circle(image, (x, y), 7, STAGE4_ANCHOR_COLOR_BGR, 2, cv2.LINE_AA)
    for x, y in np.rint(stage7).astype(int):
        cv2.circle(image, (x, y), 3, STAGE7_ANCHOR_COLOR_BGR, -1, cv2.LINE_AA)

    legend_width = min(w - 20, 820)
    legend_height = min(h - 20, 112)
    if legend_width > 20 and legend_height > 20:
        legend = image.copy()
        cv2.rectangle(legend, (10, 10), (10 + legend_width, 10 + legend_height), (0, 0, 0), -1)
        image = cv2.addWeighted(legend, 0.65, image, 0.35, 0)
        cv2.putText(
            image, f"{modality_label}: Stage 4 anchors (green rings) = {len(stage4):,}",
            (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            image, f"Stage 7 high-confidence anchors (red dots) = {len(stage7):,}",
            (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return image, {"stage4_visible": int(len(stage4)), "stage7_visible": int(len(stage7))}


def _load_stage7_qa_base(run_dir: Path, info: Dict, modality: str, display_level: int,
                         token: str = None, path: str = None, channel_index=None):
    token = token or modality
    cache_name = f"6_cache_{token}_base_L{display_level}.png"
    cached = cv2.imread(str(run_dir / cache_name), cv2.IMREAD_COLOR)
    if cached is not None:
        return cached
    path = path or info["DAPI_path" if modality == "dapi" else "HE_path"]
    with PyramidRegionReader(
        Path(path), display_level, modality, channel_index=channel_index,
    ) as reader:
        raw = reader.read_region(0, 0, reader.width, reader.height)
    image8 = _to_uint8(raw)
    if image8.ndim == 2:
        return cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image8[..., :3], cv2.COLOR_RGB2BGR)


def write_stage7_anchor_qa_images(run_dir: Path, info: Dict, high_pairs: pd.DataFrame,
                                  sides: Sides = None):
    """Write moving- and fixed-side QA maps comparing Stage-4 and Stage-7 anchors.

    Everything below exists to produce those two PNGs: it reads the display-level DAPI
    and H&E images, warps them and renders both overlays. When the overlays are switched
    off, none of that work has a consumer, so it is skipped entirely rather than done and
    discarded. Only the anchor counts, which are cheap, are still reported.
    """
    sides = sides or Sides()
    m, f = sides.tokens
    stage4_path = run_dir / "nuclei_patches" / "nuclei_centroids_global.json"
    stage4_rows = json.loads(stage4_path.read_text()).get("data", [])
    stage4_dapi = np.asarray(
        [row[f"{m}_centroid_global"] for row in stage4_rows], dtype=np.float64
    ).reshape(-1, 2)
    stage4_he = np.asarray(
        [row[f"{f}_centroid_global"] for row in stage4_rows], dtype=np.float64
    ).reshape(-1, 2)
    stage7_dapi = high_pairs[[f"{m}_x_level0", f"{m}_y_level0"]].to_numpy(np.float64)
    stage7_he = high_pairs[[f"{f}_x_level0", f"{f}_y_level0"]].to_numpy(np.float64)

    display_level = sides.level("moving", info)
    if not qc_images_enabled("stage7_anchors"):
        print(f"[Stage7] qc_images.stage7_anchors is off; skipping the anchor QA render "
              f"(Stage4={len(stage4_rows):,}, Stage7={len(high_pairs):,})", flush=True)
        return {
            "dapi_anchor_overlay": None,
            "he_anchor_overlay": None,
            "display_level": display_level,
            "stage4_anchor_count": int(len(stage4_rows)),
            "stage7_high_confidence_anchor_count": int(len(high_pairs)),
            "dapi_visible_counts": None,
            "he_visible_counts": None,
            "colors_bgr": {"stage4": list(STAGE4_ANCHOR_COLOR_BGR),
                           "stage7": list(STAGE7_ANCHOR_COLOR_BGR)},
        }
    scale = (float(2 ** display_level), float(2 ** display_level))
    dapi_base = _load_stage7_qa_base(
        run_dir, info, sides.moving_modality, display_level, token=m,
        path=sides.image_path("moving", info) if sides.roles is not None else None,
        channel_index=sides.channel_index("moving", info))
    he_base = _load_stage7_qa_base(
        run_dir, info, sides.fixed_modality, display_level, token=f,
        path=sides.image_path("fixed", info) if sides.roles is not None else None,
        channel_index=sides.channel_index("fixed", info))
    orientation_case, orientation_affine = sides.orientation(info)
    dapi_overlay, dapi_counts = render_anchor_overlay(
        dapi_base,
        stage4_dapi,
        stage7_dapi,
        scale,
        orientation_case=orientation_case,
        orientation_affine=orientation_affine,
        modality_label=m.upper(),
    )
    he_overlay, he_counts = render_anchor_overlay(
        he_base, stage4_he, stage7_he, scale, modality_label=f.upper()
    )
    # The counts below are cheap and worth keeping in the summary even when the overlay
    # PNGs are not written, so the same dict shape is returned either way and the two
    # filename fields become null.
    write_overlays = True
    dapi_path = run_dir / f"7_{m}_anchors_L{display_level}.png"
    he_path = run_dir / f"7_{f}_anchors_L{display_level}.png"
    if write_overlays:
        if not cv2.imwrite(str(dapi_path), dapi_overlay):
            raise RuntimeError(f"could not write {dapi_path}")
        if not cv2.imwrite(str(he_path), he_overlay):
            raise RuntimeError(f"could not write {he_path}")
        print(
            f"[Stage7] wrote {dapi_path.name} and {he_path.name}: "
            f"Stage4={len(stage4_rows):,}, Stage7={len(high_pairs):,}",
            flush=True,
        )
    return {
        "dapi_anchor_overlay": dapi_path.name if write_overlays else None,
        "he_anchor_overlay": he_path.name if write_overlays else None,
        "display_level": display_level,
        "stage4_anchor_count": int(len(stage4_rows)),
        "stage7_high_confidence_anchor_count": int(len(high_pairs)),
        "dapi_visible_counts": dapi_counts,
        "he_visible_counts": he_counts,
        "colors_bgr": {"stage4": list(STAGE4_ANCHOR_COLOR_BGR), "stage7": list(STAGE7_ANCHOR_COLOR_BGR)},
    }


def write_stage7_tree_qa_images(run_dir: Path, keys, dapi: np.ndarray, he: np.ndarray,
                                predicted: np.ndarray, edges: Dict, high_pairs: pd.DataFrame,
                                he_info: Dict, he_level: int, sides: Sides = None) -> Dict:
    """One picture per Stage-3 tile of the nearest-neighbour candidate search.

    Drawn on the tile's rectified H&E image: its H&E nuclei as filled dots, the DAPI
    nuclei it holds projected through the final transform as hollow magenta circles,
    and every edge the KD-tree search considered from a projected DAPI nucleus to an
    H&E candidate, coloured by what became of it (`TREE_EDGE_COLORS_BGR`). Gated by
    `qc_images.stage7_tree`; written to `dense_nucleus_pairs/tree_qc/<tile>_tree.jpg`.
    A tile whose image is missing is skipped and counted, never fatal.
    """
    summary = {"dir": None, "tiles_written": 0, "tiles_skipped": 0, "files": []}
    if not qc_images_enabled("stage7_tree"):
        print("[Stage7] qc_images.stage7_tree is off; skipping the per-tile tree QC render",
              flush=True)
        return summary
    if not edges:
        print("[Stage7] no candidate edges to draw", flush=True)
        return summary
    sides = sides or Sides()
    m, f = sides.tokens
    out_dir = run_dir / "dense_nucleus_pairs" / TREE_QC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = run_dir / "tiles"
    he_scale = float(2 ** int(he_level))
    high = set(zip(
        (high_pairs[f"{m}_nucleus_id"].to_numpy(np.int64) - 1).tolist(),
        (high_pairs[f"{f}_nucleus_id"].to_numpy(np.int64) - 1).tolist(),
    ))
    dapi_chunk = dapi[:, 6].astype(np.int64)
    he_chunk = he[:, 6].astype(np.int64)
    e_dapi = np.asarray(edges["dapi"], dtype=np.int64)
    e_he = np.asarray(edges["he"], dtype=np.int64)
    e_gate = np.asarray(edges["gate"], dtype=bool)
    e_sel = np.asarray(edges["selected"], dtype=bool)
    edge_chunk = dapi_chunk[e_dapi]
    edge_class = np.where(
        e_sel,
        np.where([(int(d), int(h)) in high for d, h in zip(e_dapi, e_he)], "high", "medium"),
        np.where(e_gate, "candidate", "rejected"),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, key in enumerate(keys):
        hmeta = he_info.get(key, {})
        path = tiles_dir / str(hmeta.get("filename", f"{key}_{f}.png"))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            summary["tiles_skipped"] += 1
            continue
        he_to_rect = np.linalg.inv(rect_to_fixed(hmeta, f))

        def to_tile(level0_xy):
            return _perspective_xy((np.asarray(level0_xy, dtype=np.float64) + 0.5) / he_scale - 0.5, he_to_rect)

        def draw_point(fn, xy, *args):
            if np.isfinite(xy).all():
                fn(image, (int(round(xy[0])), int(round(xy[1]))), *args)

        he_here = np.flatnonzero(he_chunk == index)
        for xy in to_tile(he[he_here, 2:4]):
            draw_point(cv2.circle, xy, 3, TREE_HE_NUCLEUS_BGR, -1)
        edge_here = np.flatnonzero(edge_chunk == index)
        counts = {name: 0 for name in TREE_EDGE_COLORS_BGR}
        src = to_tile(predicted[e_dapi[edge_here]])
        dst = to_tile(he[e_he[edge_here], 2:4])
        for name in ("rejected", "candidate", "medium", "high"):   # important on top
            colour = TREE_EDGE_COLORS_BGR[name]
            thickness = 2 if name in ("high", "medium") else 1
            for k in np.flatnonzero(edge_class[edge_here] == name):
                a, b = src[k], dst[k]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    cv2.line(image, (int(round(a[0])), int(round(a[1]))),
                             (int(round(b[0])), int(round(b[1]))), colour, thickness, cv2.LINE_AA)
                counts[name] += 1
        dapi_here = np.flatnonzero(dapi_chunk == index)
        for xy in to_tile(predicted[dapi_here]):
            draw_point(cv2.circle, xy, 4, TREE_PROJECTED_DAPI_BGR, 1)

        header = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"{key}  {f} nuclei {len(he_here)}  projected {m} {len(dapi_here)}  "
                    f"edges {len(edge_here)}", (4, 15), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x = 4
        for name, colour in TREE_EDGE_COLORS_BGR.items():
            text = f"{name} {counts[name]}"
            cv2.putText(header, text, (x, 33), font, 0.42, colour, 1, cv2.LINE_AA)
            x += 12 + int(cv2.getTextSize(text, font, 0.42, 1)[0][0])
        cv2.putText(header, f"dot={f} nucleus  ring=projected {m}", (x + 8, 33), font, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{key}_tree.jpg"), np.vstack([header, image]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        summary["tiles_written"] += 1
        summary["files"].append(f"dense_nucleus_pairs/{TREE_QC_DIR}/{key}_tree.jpg")
    summary["dir"] = TREE_QC_DIR
    print(f"[Stage7] wrote {summary['tiles_written']} tree QC images to "
          f"dense_nucleus_pairs/{TREE_QC_DIR}/ ({summary['tiles_skipped']} tiles without an image)",
          flush=True)
    return summary


def _perspective_xy(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    denom = projected[:, 2]
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    good = np.abs(denom) > 1e-12
    out[good] = projected[good, :2] / denom[good, None]
    return out


def _local_area_scale(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Numerical homography Jacobian determinant at each input point."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    origin = _perspective_xy(points, matrix)
    dx = _perspective_xy(points + np.array([1.0, 0.0]), matrix) - origin
    dy = _perspective_xy(points + np.array([0.0, 1.0]), matrix) - origin
    return np.abs(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0])


def _stage3_tile_levels(run_dir: Path, sides: Sides = None) -> Tuple[int, int]:
    """(moving, fixed) Stage-3 tile levels, by token."""
    sides = sides or Sides()
    m, f = sides.tokens
    used = run_dir / "nuclei_centroids_global_used_for_stage5.json"
    if used.exists():
        try:
            meta = json.loads(used.read_text()).get("meta", {})
            return int(meta[f"{m}_tile_level"]), int(meta[f"{f}_tile_level"])
        except Exception:
            pass
    sampled = run_dir / "sampled_points.json"
    dapi_level = 1
    if sampled.exists():
        try:
            sampled_meta = json.loads(sampled.read_text())
            dapi_level = int(sampled_meta.get(f"{m}_tile_level", sampled_meta.get("dapi_tile_level", 1)))
        except Exception:
            pass
    # Stage 3 chooses the fixed level by preserving the Stage-1 level difference.
    info = json.loads((run_dir / "images_info.json").read_text())
    he_level = max(1, sides.level("fixed", info, 1) - (sides.level("moving", info, dapi_level) - dapi_level))
    return dapi_level, he_level


def dedup_radii(params: Dict):
    """The cross-tile dedup radius for the moving and the fixed side, each in its own
    level-0 pixels: `main` sets the per-side values from the physical scale, a caller
    without them (a test, an older parameter set) gets the one configured value twice."""
    base = float(params["tile_dedup_radius_level0_px"])
    return (float(params.get("tile_dedup_radius_moving_px", base)),
            float(params.get("tile_dedup_radius_fixed_px", base)))


def deduplicate_cross_tile_nuclei(data: np.ndarray, radius: float) -> np.ndarray:
    """Merge only near-identical detections originating in different tiles."""
    if len(data) < 2 or radius <= 0:
        return data
    pairs = cKDTree(data[:, 2:4]).query_pairs(float(radius), output_type="ndarray")
    if len(pairs) == 0:
        return data
    parent = np.arange(len(data), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for first, second in pairs:
        if int(data[first, 6]) != int(data[second, 6]):
            union(first, second)
    groups = {}
    for index in range(len(data)):
        groups.setdefault(find(index), []).append(index)
    keep = [max(indices, key=lambda i: data[i, 5]) for indices in groups.values()]
    return data[np.sort(np.asarray(keep, dtype=np.int64))]


def load_stage4a_detection_cache(
    run_dir: Path,
    params: Dict,
    dapi_info: Dict,
    he_info: Dict,
    keys,
    dapi_level: int,
    he_level: int,
    selection_method: str,
    sides: Sides = None,
):
    """Restore cached local Stage-7 detections to level-0 coordinates.

    Returns ``None`` whenever validation fails so older runs and partial Stage-4a
    runs continue through the original image-segmentation path. `dapi_*` here is
    the moving side and `he_*` the fixed side; the cache keys and area gates follow
    `sides` (tokens name the cache arrays, modalities pick the gates).
    """
    if not bool(params.get("reuse_stage4a_detection_cache", True)):
        return None
    sides = sides or Sides()
    m, f = sides.tokens
    m_min, m_max = area_gate_keys(sides.moving_modality)
    f_min, f_max = area_gate_keys(sides.fixed_modality)
    tiles_dir = run_dir / "tiles"
    meta_path = tiles_dir / CACHE_META_NAME
    cache_path = tiles_dir / CACHE_NPZ_NAME
    if not meta_path.exists() or not cache_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text())
        if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError("schema version differs")
        if int(metadata[f"{m}_tile_level"]) != int(dapi_level):
            raise ValueError(f"{m} tile level differs")
        if int(metadata[f"{f}_tile_level"]) != int(he_level):
            raise ValueError(f"{f} tile level differs")
        if metadata.get("detection_parameters") != detection_params_snapshot(params):
            raise ValueError("dense detection parameters differ")
        requested = set(keys)
        if not requested.issubset(set(metadata.get(f"{m}_successful_tile_ids", []))):
            raise ValueError(f"one or more requested {m} tiles were not cached")
        if not requested.issubset(set(metadata.get(f"{f}_successful_tile_ids", []))):
            raise ValueError(f"one or more requested {f} tiles were not cached")

        with np.load(cache_path, allow_pickle=False) as cache:
            tile_keys = [str(value) for value in cache["tile_keys"].tolist()]
            dapi_local = np.asarray(cache[f"{m}_components"], dtype=np.float64)
            he_local = np.asarray(cache[f"{f}_components"], dtype=np.float64)
        if dapi_local.ndim != 2 or dapi_local.shape[1] != 4:
            raise ValueError("invalid DAPI cache array")
        if he_local.ndim != 2 or he_local.shape[1] != 4:
            raise ValueError("invalid H&E cache array")
        cache_index = {key: index for index, key in enumerate(tile_keys)}
        if not requested.issubset(cache_index):
            raise ValueError("cache tile index is incomplete")
    except Exception as exc:
        print(f"[Stage7] Stage-4a detection cache ignored: {exc}", flush=True)
        return None

    dapi_scale = float(2 ** dapi_level)
    he_scale = float(2 ** he_level)
    dapi_batches, he_batches = [], []
    for source_chunk, key in enumerate(keys):
        index = cache_index[key]
        drows = dapi_local[dapi_local[:, 0].astype(np.int64) == index]
        if len(drows):
            xy_work = drows[:, 1:3] + np.array(
                [float(dapi_info[key]["x0"]), float(dapi_info[key]["y0"])]
            )
            area_work = drows[:, 3]
            area0 = area_work * dapi_scale * dapi_scale
            keep = (
                (area0 >= float(params[m_min]))
                & (area0 <= float(params[m_max]))
            )
            if np.any(keep):
                dapi_batches.append(
                    np.column_stack(
                        [
                            xy_work[keep],
                            (xy_work[keep] + 0.5) * dapi_scale - 0.5,
                            area_work[keep],
                            area0[keep],
                            np.full(int(keep.sum()), source_chunk, dtype=np.float64),
                        ]
                    )
                )

        hrows = he_local[he_local[:, 0].astype(np.int64) == index]
        if len(hrows):
            local_xy = hrows[:, 1:3]
            rect_to_he = rect_to_fixed(he_info[key], f)
            xy_work = _perspective_xy(local_xy, rect_to_he)
            area_work = hrows[:, 3]
            area0 = (
                area_work
                * _local_area_scale(local_xy, rect_to_he)
                * he_scale
                * he_scale
            )
            keep = (
                np.isfinite(xy_work).all(axis=1)
                & (area0 >= float(params[f_min]))
                & (area0 <= float(params[f_max]))
            )
            if np.any(keep):
                he_batches.append(
                    np.column_stack(
                        [
                            xy_work[keep],
                            (xy_work[keep] + 0.5) * he_scale - 0.5,
                            area_work[keep],
                            area0[keep],
                            np.full(int(keep.sum()), source_chunk, dtype=np.float64),
                        ]
                    )
                )

    dapi = (
        np.concatenate(dapi_batches)
        if dapi_batches
        else np.empty((0, len(NUCLEUS_COLUMNS)), dtype=np.float64)
    )
    he = (
        np.concatenate(he_batches)
        if he_batches
        else np.empty((0, len(NUCLEUS_COLUMNS)), dtype=np.float64)
    )
    dapi_before, he_before = len(dapi), len(he)
    radius_m, radius_f = dedup_radii(params)
    dapi = deduplicate_cross_tile_nuclei(dapi, radius_m)
    he = deduplicate_cross_tile_nuclei(he, radius_f)
    sampling = {
        "scan_mode": "stage3_tiles",
        "available_stage3_tiles": len(set(dapi_info).intersection(he_info)),
        "processed_stage3_tiles": len(keys),
        "tile_selection_method": selection_method,
        "tile_selection_manifest": None,
        "stage4b_selection_manifest_used": False,
        "selected_tile_ids": list(keys),
        f"{m}_tile_level": int(dapi_level),
        f"{f}_tile_level": int(he_level),
        f"{m}_detections_before_dedup": int(dapi_before),
        f"{f}_detections_before_dedup": int(he_before),
        "detection_source": "stage4a_exact_stage7_cache",
        "detection_cache": CACHE_NPZ_NAME,
        "detection_cache_metadata": CACHE_META_NAME,
    }
    print(
        f"[Stage7] reused Stage-4a detection cache: {m}={len(dapi):,}, "
        f"{f}={len(he):,}",
        flush=True,
    )
    return dapi, he, sampling


def scan_stage3_tiles(run_dir: Path, params: Dict, engine_dir: Path, sides: Sides = None):
    """Densify the already tissue-sampled Stage-3 tiles at nucleus resolution.

    Returns (moving nuclei, fixed nuclei, sampling metadata); the arrays are the
    `NUCLEUS_COLUMNS` rows in each side's own level-0 frame.
    """
    sides = sides or Sides()
    m, f = sides.tokens
    tiles_dir = run_dir / "tiles"
    dapi_info_path = tiles_dir / f"{m}_tile_info.json"
    he_info_path = tiles_dir / f"{f}_tile_info.json"
    if not dapi_info_path.exists() or not he_info_path.exists():
        raise FileNotFoundError(
            f"scan_mode='stage3_tiles' requires tiles/{m}_tile_info.json and tiles/{f}_tile_info.json"
        )
    dapi_info = json.loads(dapi_info_path.read_text())
    he_info = json.loads(he_info_path.read_text())
    available_keys = set(dapi_info).intersection(he_info)
    configured = int(params["max_stage3_tiles"])
    if configured <= 0 or len(available_keys) <= configured:
        # Stage 7 deliberately scans every Stage-3 tile.  The Stage-4b spatial
        # subset manifest is only for standout-nucleus selection in Stages 4/5.
        keys = sorted(available_keys)
        selection_method = "all_shared_stage3_tiles"
        print(f"[Stage7] scanning all {len(keys)} shared Stage-3 tiles", flush=True)
    else:
        keys = select_spatially_uniform_tile_ids(
            dapi_info, configured, available_keys
        )
        selection_method = "stage3_parameter_cap_deterministic_farthest_point"
        print(f"[Stage7] max_stage3_tiles={configured} caps the scan at {len(keys)} of "
              f"{len(available_keys)} shared Stage-3 tiles", flush=True)
    if not keys:
        raise RuntimeError("Stage-3 tile metadata has no shared DAPI/H&E tiles")
    dapi_level, he_level = _stage3_tile_levels(run_dir, sides)
    cached = load_stage4a_detection_cache(
        run_dir,
        params,
        dapi_info,
        he_info,
        keys,
        dapi_level,
        he_level,
        selection_method,
        sides,
    )
    if cached is not None:
        return cached
    m_min, m_max = area_gate_keys(sides.moving_modality)
    f_min, f_max = area_gate_keys(sides.fixed_modality)
    dapi_scale = float(2 ** dapi_level)
    he_scale = float(2 ** he_level)
    dapi_batches, he_batches = [], []
    started = time.perf_counter()
    failures = 0

    for number, key in enumerate(keys, 1):
        dmeta, hmeta = dapi_info[key], he_info[key]
        dapi_path = tiles_dir / str(dmeta.get("filename", f"{key}_{m}_u8.png"))
        if not dapi_path.exists():
            dapi_path = tiles_dir / f"{key}_{m}_u8.png"
        if not dapi_path.exists():
            dapi_path = tiles_dir / f"{key}_{m}.png"
        he_path = tiles_dir / str(hmeta.get("filename", f"{key}_{f}.png"))
        try:
            dapi_tile = cv2.imread(str(dapi_path), cv2.IMREAD_UNCHANGED)
            he_tile = cv2.imread(str(he_path), cv2.IMREAD_UNCHANGED)
            if dapi_tile is None or he_tile is None:
                raise RuntimeError("tile image could not be read")

            dparams = dict(params)
            dparams["_work_pixel_area_level0"] = dapi_scale * dapi_scale
            dcomp = segment_centroids(sides.moving_modality, dapi_tile, dparams, engine_dir)
            if len(dcomp):
                dxy_work = dcomp[:, :2] + np.array([float(dmeta["x0"]), float(dmeta["y0"])])
                darea0 = dcomp[:, 2] * dapi_scale * dapi_scale
                keep = (
                    (darea0 >= float(params[m_min])) &
                    (darea0 <= float(params[m_max]))
                )
                if np.any(keep):
                    dapi_batches.append(np.column_stack([
                        dxy_work[keep], (dxy_work[keep] + 0.5) * dapi_scale - 0.5,
                        dcomp[keep, 2], darea0[keep],
                        np.full(int(keep.sum()), number - 1, dtype=np.float64),
                    ]))

            rect_to_he = rect_to_fixed(hmeta, f)
            center = np.array([[he_tile.shape[1] / 2.0, he_tile.shape[0] / 2.0]])
            hparams = dict(params)
            hparams["_work_pixel_area_level0"] = float(_local_area_scale(center, rect_to_he)[0]) * he_scale * he_scale
            hcomp = segment_centroids(sides.fixed_modality, he_tile, hparams, engine_dir)
            if len(hcomp):
                hxy_work = _perspective_xy(hcomp[:, :2], rect_to_he)
                local_scale = _local_area_scale(hcomp[:, :2], rect_to_he)
                harea0 = hcomp[:, 2] * local_scale * he_scale * he_scale
                keep = (
                    np.isfinite(hxy_work).all(axis=1) &
                    (harea0 >= float(params[f_min])) &
                    (harea0 <= float(params[f_max]))
                )
                if np.any(keep):
                    he_batches.append(np.column_stack([
                        hxy_work[keep], (hxy_work[keep] + 0.5) * he_scale - 0.5,
                        hcomp[keep, 2], harea0[keep],
                        np.full(int(keep.sum()), number - 1, dtype=np.float64),
                    ]))
        except Exception as exc:
            failures += 1
            print(f"[Stage7] WARN sampled tile {key} skipped: {exc}", flush=True)

        if number == 1 or number % 20 == 0 or number == len(keys):
            dn = sum(len(batch) for batch in dapi_batches)
            hn = sum(len(batch) for batch in he_batches)
            print(
                f"[Stage7] sampled tiles {number}/{len(keys)}: {m} nuclei={dn:,}, "
                f"{f} nuclei={hn:,}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if failures == len(keys):
        raise RuntimeError("all sampled Stage-3 tiles failed")
    dapi = np.concatenate(dapi_batches) if dapi_batches else np.empty((0, len(NUCLEUS_COLUMNS)))
    he = np.concatenate(he_batches) if he_batches else np.empty((0, len(NUCLEUS_COLUMNS)))
    radius_m, radius_f = dedup_radii(params)
    dapi_before, he_before = len(dapi), len(he)
    dapi = deduplicate_cross_tile_nuclei(dapi, radius_m)
    he = deduplicate_cross_tile_nuclei(he, radius_f)
    metadata = {
        "scan_mode": "stage3_tiles",
        "available_stage3_tiles": len(set(dapi_info).intersection(he_info)),
        "processed_stage3_tiles": len(keys),
        "tile_selection_method": selection_method,
        "tile_selection_manifest": None,
        "stage4b_selection_manifest_used": False,
        "selected_tile_ids": keys,
        f"{m}_tile_level": dapi_level,
        f"{f}_tile_level": he_level,
        f"{m}_detections_before_dedup": dapi_before,
        f"{f}_detections_before_dedup": he_before,
        "detection_source": "stage3_tile_segmentation",
    }
    return dapi, he, metadata


def load_transform(path: Path) -> Dict:
    data = json.loads(Path(path).read_text())
    kind = str(data.get("transform_type", "")).lower().strip()
    if kind == local_residual.TRANSFORM_TYPE:
        # Keep the serialized model intact; the shared point transform validates it.
        return {"kind": local_residual.TRANSFORM_TYPE, "model": data}
    for key in ("homography_3x3", "affine_3x3", "H_mat", "H", "matrix_3x3"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"invalid {key} shape {matrix.shape}")
            return {"kind": "affine" if kind == "affine" else "homography", "matrix": matrix}
    if "affine_2x3" in data:
        affine = np.asarray(data["affine_2x3"], dtype=np.float64)
        if affine.shape != (2, 3):
            raise ValueError(f"invalid affine_2x3 shape {affine.shape}")
        return {"kind": "affine", "matrix": np.vstack([affine, [0.0, 0.0, 1.0]])}
    raise ValueError(f"unrecognized transform keys: {sorted(data.keys())}")


def transform_points(points: np.ndarray, transform: Dict, batch_size: int = 100_000) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(points)
    for start in range(0, len(points), batch_size):
        p = points[start:start + batch_size]
        if transform["kind"] == local_residual.TRANSFORM_TYPE:
            out[start:start + len(p)] = local_residual.forward_points(
                p, transform["model"]
            )
        else:
            homogeneous = np.column_stack([p, np.ones(len(p))])
            projected = (transform["matrix"] @ homogeneous.T).T
            denom = projected[:, 2]
            good = np.abs(denom) > 1e-12
            q = np.full((len(p), 2), np.nan, dtype=np.float64)
            q[good] = projected[good, :2] / denom[good, None]
            out[start:start + len(p)] = q
    return out


def _tree_query(tree: cKDTree, points: np.ndarray, upper: float, k: int = 1):
    try:
        return tree.query(points, k=k, distance_upper_bound=upper, workers=-1)
    except TypeError:  # scipy < 1.6
        return tree.query(points, k=k, distance_upper_bound=upper)


def match_reciprocal(
    dapi: np.ndarray,
    he: np.ndarray,
    predicted_he: np.ndarray,
    eligible_dapi: np.ndarray,
    params: Dict,
    edge_sink: Dict = None,
    tokens: Tuple[str, str] = ("dapi", "he"),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Match projected DAPI nuclei to H&E nuclei one-to-one.

    `edge_sink`, when a dict is passed, is filled with every candidate edge the
    search considered -- global `dapi` / `he` row indices, neighbour `rank`,
    `distance`, whether it passed the `gate`, whether it was `mutual` nearest, and
    whether it was `selected` as a pair -- for the per-tile tree QC image.
    """
    m, f = tokens
    candidate_radius = float(params["candidate_distance_level0_px"])
    max_distance = float(params["max_pair_distance_level0_px"])
    pred = predicted_he[eligible_dapi]
    tree_he = cKDTree(he[:, 2:4])
    distance, nearest_he = _tree_query(tree_he, pred, candidate_radius, k=1)
    valid = np.isfinite(distance) & (nearest_he < len(he))
    local_d = np.flatnonzero(valid)
    d_idx = eligible_dapi[local_d]
    h_idx = nearest_he[valid].astype(np.int64)
    dist = distance[valid]

    # H&E -> projected-DAPI reverse query establishes a scalable one-to-one
    # correspondence without building an N x M cost matrix.
    tree_pred = cKDTree(pred)
    _, reverse_local = _tree_query(tree_pred, he[:, 2:4], candidate_radius, k=1)
    mutual = reverse_local[h_idx] == local_d

    raw_ratio = np.maximum(
        dapi[d_idx, 5] / np.maximum(he[h_idx, 5], 1e-9),
        he[h_idx, 5] / np.maximum(dapi[d_idx, 5], 1e-9),
    )
    log_delta = np.log(np.maximum(he[h_idx, 5], 1e-9)) - np.log(np.maximum(dapi[d_idx, 5], 1e-9))
    area_offset = 0.0
    calibration_pool = mutual & (dist <= max_distance)
    if bool(params["calibrate_area_between_modalities"]) and int(calibration_pool.sum()) >= 20:
        area_offset = float(np.median(log_delta[calibration_pool]))
    calibrated_ratio = np.exp(np.abs(log_delta - area_offset))

    assignment_mode = str(params.get("assignment_mode", "reciprocal_nearest")).lower().strip()
    if assignment_mode == "greedy_one_to_one":
        neighbors = max(1, int(params["candidate_neighbors"]))
        edge_dist, edge_he = _tree_query(tree_he, pred, candidate_radius, k=neighbors)
        if neighbors == 1:
            edge_dist = edge_dist[:, None]
            edge_he = edge_he[:, None]
        edge_local_d = np.repeat(np.arange(len(pred), dtype=np.int64), neighbors)
        edge_rank = np.tile(np.arange(neighbors, dtype=np.int64), len(pred))
        edge_dist = np.asarray(edge_dist).reshape(-1)
        edge_he = np.asarray(edge_he).reshape(-1).astype(np.int64)
        edge_valid = np.isfinite(edge_dist) & (edge_he < len(he))
        edge_local_d = edge_local_d[edge_valid]
        edge_rank = edge_rank[edge_valid]
        edge_dist = edge_dist[edge_valid]
        edge_he = edge_he[edge_valid]
        edge_dapi = eligible_dapi[edge_local_d]
        edge_log_delta = (
            np.log(np.maximum(he[edge_he, 5], 1e-9)) -
            np.log(np.maximum(dapi[edge_dapi, 5], 1e-9))
        )
        edge_ratio = np.exp(np.abs(edge_log_delta - area_offset))
        gate = (
            (edge_dist <= max_distance) &
            (edge_ratio <= float(params["max_calibrated_area_ratio"]))
        )
        area_norm = max(math.log(float(params["max_calibrated_area_ratio"])), 1e-9)
        edge_cost = (
            edge_dist / max(max_distance, 1e-9) +
            float(params["area_cost_weight"]) * np.abs(edge_log_delta - area_offset) / area_norm
        )
        edge_mutual = (
            (edge_rank == 0) &
            (reverse_local[edge_he] == edge_local_d)
        )
        used_d = np.zeros(len(pred), dtype=bool)
        used_h = np.zeros(len(he), dtype=bool)
        selected = []

        def take_edges(indices):
            for edge_index in indices:
                ld = int(edge_local_d[edge_index])
                hh = int(edge_he[edge_index])
                if not used_d[ld] and not used_h[hh]:
                    used_d[ld] = True
                    used_h[hh] = True
                    selected.append(int(edge_index))

        # Preserve every gated reciprocal match first; fill the remaining
        # one-to-one slots with the lowest-cost alternative candidates.
        seeded = np.flatnonzero(gate & edge_mutual)
        take_edges(seeded[np.argsort(edge_cost[seeded])])
        remaining = np.flatnonzero(gate & ~edge_mutual)
        take_edges(remaining[np.argsort(edge_cost[remaining])])
        selected = np.asarray(selected, dtype=np.int64)
        if edge_sink is not None:
            chosen = np.zeros(len(edge_dapi), dtype=bool)
            chosen[selected] = True
            edge_sink.update({
                "dapi": edge_dapi, "he": edge_he, "rank": edge_rank, "distance": edge_dist,
                "gate": gate, "mutual": edge_mutual, "selected": chosen,
            })
        pair_local_d = edge_local_d[selected]
        pair_d = edge_dapi[selected]
        pair_h = edge_he[selected]
        pair_dist = edge_dist[selected]
        pair_log_delta = edge_log_delta[selected]
        pair_ratio = edge_ratio[selected]
        pair_mutual = edge_mutual[selected]
    elif assignment_mode == "reciprocal_nearest":
        gate = (
            mutual & (dist <= max_distance) &
            (calibrated_ratio <= float(params["max_calibrated_area_ratio"]))
        )
        pair_local_d = local_d[gate]
        pair_d = d_idx[gate]
        pair_h = h_idx[gate]
        pair_dist = dist[gate]
        pair_log_delta = log_delta[gate]
        pair_ratio = calibrated_ratio[gate]
        pair_mutual = np.ones(int(gate.sum()), dtype=bool)
        if edge_sink is not None:
            edge_sink.update({
                "dapi": d_idx, "he": h_idx, "rank": np.zeros(len(d_idx), dtype=np.int64),
                "distance": dist, "gate": gate, "mutual": mutual, "selected": gate.copy(),
            })
    else:
        raise ValueError("assignment_mode must be 'greedy_one_to_one' or 'reciprocal_nearest'")

    selected_set = set(zip(pair_local_d.tolist(), pair_h.tolist()))
    accepted = np.array(
        [(int(ld), int(hh)) in selected_set for ld, hh in zip(local_d, h_idx)],
        dtype=bool,
    )
    distance_score = np.exp(-np.square(dist / max(max_distance, 1e-9)))
    area_score = np.exp(-np.abs(log_delta - area_offset) / max(math.log(float(params["max_calibrated_area_ratio"])), 1e-9))
    confidence = 0.75 * distance_score + 0.25 * area_score
    high = accepted & mutual & (dist <= float(params["high_confidence_distance_level0_px"])) & (
        calibrated_ratio <= float(params["high_confidence_area_ratio"])
    )

    candidates = pd.DataFrame({
        f"{m}_nucleus_id": d_idx + 1,
        f"{f}_nucleus_id": h_idx + 1,
        f"{m}_x_level0": dapi[d_idx, 2],
        f"{m}_y_level0": dapi[d_idx, 3],
        f"predicted_{f}_x_level0": predicted_he[d_idx, 0],
        f"predicted_{f}_y_level0": predicted_he[d_idx, 1],
        f"{f}_x_level0": he[h_idx, 2],
        f"{f}_y_level0": he[h_idx, 3],
        "centroid_distance_level0_px": dist,
        f"{m}_area_level0_px": dapi[d_idx, 5],
        f"{f}_area_level0_px": he[h_idx, 5],
        "raw_area_ratio": raw_ratio,
        "calibrated_area_ratio": calibrated_ratio,
        "mutual_nearest": mutual,
        "accepted": accepted,
        "confidence": confidence,
        "confidence_tier": np.where(high, "high", np.where(accepted, "medium", "rejected")),
    })
    pair_distance_score = np.exp(-np.square(pair_dist / max(max_distance, 1e-9)))
    pair_area_score = np.exp(
        -np.abs(pair_log_delta - area_offset) /
        max(math.log(float(params["max_calibrated_area_ratio"])), 1e-9)
    )
    pair_confidence = 0.75 * pair_distance_score + 0.25 * pair_area_score
    pair_high = (
        pair_mutual &
        (pair_dist <= float(params["high_confidence_distance_level0_px"])) &
        (pair_ratio <= float(params["high_confidence_area_ratio"]))
    )
    pair_raw_ratio = np.maximum(
        dapi[pair_d, 5] / np.maximum(he[pair_h, 5], 1e-9),
        he[pair_h, 5] / np.maximum(dapi[pair_d, 5], 1e-9),
    )
    pairs = pd.DataFrame({
        f"{m}_nucleus_id": pair_d + 1,
        f"{f}_nucleus_id": pair_h + 1,
        f"{m}_x_level0": dapi[pair_d, 2],
        f"{m}_y_level0": dapi[pair_d, 3],
        f"predicted_{f}_x_level0": predicted_he[pair_d, 0],
        f"predicted_{f}_y_level0": predicted_he[pair_d, 1],
        f"{f}_x_level0": he[pair_h, 2],
        f"{f}_y_level0": he[pair_h, 3],
        "centroid_distance_level0_px": pair_dist,
        f"{m}_area_level0_px": dapi[pair_d, 5],
        f"{f}_area_level0_px": he[pair_h, 5],
        "raw_area_ratio": pair_raw_ratio,
        "calibrated_area_ratio": pair_ratio,
        "mutual_nearest": pair_mutual,
        "accepted": np.ones(len(pair_d), dtype=bool),
        "confidence": pair_confidence,
        "confidence_tier": np.where(pair_high, "high", "medium"),
    })
    pairs.insert(0, "pair_id", np.arange(1, len(pairs) + 1, dtype=np.int64))
    metrics = {
        "eligible_dapi_nuclei": int(len(eligible_dapi)),
        "dapi_with_nearest_candidate": int(len(candidates)),
        "reciprocal_nearest_candidates": int(mutual.sum()),
        "assignment_mode": assignment_mode,
        "accepted_pairs": int(len(pairs)),
        "high_confidence_pairs": int(pair_high.sum()),
        f"log_{f}_to_{m}_area_calibration": area_offset,
    }
    return candidates, pairs, metrics


def nucleus_dataframe(data: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(data, columns=NUCLEUS_COLUMNS)
    frame.insert(0, "nucleus_id", np.arange(1, len(frame) + 1, dtype=np.int64))
    if len(frame):
        frame["source_chunk"] = frame["source_chunk"].astype(np.int64)
    return frame


def write_csv_gz(frame: pd.DataFrame, path: Path):
    frame.to_csv(
        path, index=False, float_format="%.4f",
        compression={"method": "gzip", "compresslevel": 1},
    )
    print(f"[Stage7] wrote {path.name}: {len(frame):,} rows", flush=True)


def load_params(path: Path, pair: str = "dapi-he") -> Dict:
    data = json.loads(path.read_text())
    # Prefer the pair-keyed 4.11 block, falling back to 4.10's spelling only when the
    # current key is absent (not empty). Another mixed pair (he-dapi) reads its own
    # block, and dapi-he's when it has none: the gates are physical sizes by
    # modality and the calibration is symmetric, so nothing in them is directional.
    dapi_block = data.get("dapi-he", data.get("dapi", {}))
    block = dapi_block.get("stage7_dense_pairs", {})
    if pair != "dapi-he":
        own = data.get(pair, {}).get("stage7_dense_pairs")
        if isinstance(own, dict):
            block = own
        else:
            print(f"[Stage7] {pair} has no stage7_dense_pairs block; using dapi-he's", flush=True)
    params = dict(DEFAULTS)
    if isinstance(block, dict):
        params.update(block)
    # Do not inherit a cap from Stage 3's parameter names. That coupling made renaming a
    # Stage-3 key silently halve Stage 7's coverage: `n_tiles` became `fallback_n_tiles`,
    # the lookup missed, and Stage 7 fell back to its old default of 120 tiles out of 200,
    # cutting detected nuclei by 42% with nothing in the log to say so.
    params["scan_mode"] = str(params["scan_mode"]).lower().strip()
    if params["scan_mode"] != "stage3_tiles":
        raise ValueError("stage7 scan_mode must be 'stage3_tiles' (the whole-slide scan was "
                         "removed in 4.11)")
    if int(params["max_stage3_tiles"]) < 0:
        raise ValueError("stage7 max_stage3_tiles must be >= 0 (0 means every tile)")
    if int(params["max_output_pairs"]) < 0:
        raise ValueError("stage7 max_output_pairs must be >= 0 (0 means unlimited)")
    params["assignment_mode"] = str(params["assignment_mode"]).lower().strip()
    if params["assignment_mode"] not in {"greedy_one_to_one", "reciprocal_nearest"}:
        raise ValueError("stage7 assignment_mode must be 'greedy_one_to_one' or 'reciprocal_nearest'")
    if int(params["candidate_neighbors"]) < 1:
        raise ValueError("stage7 candidate_neighbors must be >= 1")
    if float(params["max_pair_distance_level0_px"]) > float(params["candidate_distance_level0_px"]):
        raise ValueError("max_pair_distance_level0_px cannot exceed candidate_distance_level0_px")
    return params


def main():
    parser = argparse.ArgumentParser(description="Dense DAPI--H&E nucleus pairing after final alignment")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parameters", default=str(PROJECT_ROOT / "parameters.json"))
    parser.add_argument("--qc-only", action="store_true",
                        help="re-draw the QC images (stage7_anchors, stage7_tree) for the "
                             "pairs already on disk by repeating the matching from the "
                             "cache; no CSV, summary or timing record is rewritten")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    # Dense pairing matches the moving image's nuclei to the fixed image's across
    # modalities: a DAPI detector on the DAPI side, an H&E detector on the H&E side,
    # whichever is moving. A same-modality pair has no such pairing to make.
    import run_roles
    roles = run_roles.read(info)
    sides = Sides(roles)
    if not sides.mixed:
        raise SystemExit(f"Stage 7 pairs nuclei across modalities and does not apply to "
                         f"a same-modality pair; this run is {roles.pair.name}")
    m, f = sides.tokens
    transform_path = run_dir / f"{sides.homography_stem}.json"
    if not transform_path.exists():
        raise FileNotFoundError(transform_path)
    # Stage 7 pairs against Stage 5's transform: once Stage 8 has promoted its own into
    # the canonical file, the Stage-5 one it backed up is the right input again, or a
    # rerun would pair against the transform those pairs produced.
    stage5_backup = run_dir / f"{sides.homography_stem}_stage5.json"
    try:
        canonical_stage = int(json.loads(transform_path.read_text()).get("stage", 5))
    except (OSError, ValueError, TypeError):
        canonical_stage = 5
    if canonical_stage == 8 and stage5_backup.exists():
        print(f"[Stage7] canonical transform is Stage 8's; pairing against the Stage-5 "
              f"backup {stage5_backup.name}", flush=True)
        transform_path = stage5_backup
    configured = load_params(Path(args.parameters), roles.pair.name)
    # The level-0-pixel gates are physical sizes at the reference pixel size; on an
    # image scanned at another size they are rescaled here, once, and every gate
    # below reads the scaled copy. The configured values are what the summary reports.
    scale = physical_scale.run_factors(info, {
        "dapi": float(configured["reference_mpp_dapi"]),
        "he": float(configured["reference_mpp_he"]),
    }, run_dir=run_dir)
    gate_spec = physical_scale.stage7_gates(sides.fixed_modality)
    params, scaled_gates = physical_scale.scale_gates(configured, scale["factor"], gate_spec)
    # The dedup radius is applied to each side in that side's own frame.
    base_radius = float(configured["tile_dedup_radius_level0_px"])
    params["tile_dedup_radius_moving_px"] = base_radius * float(scale["factor"].get(sides.moving_modality, 1.0))
    params["tile_dedup_radius_fixed_px"] = base_radius * float(scale["factor"].get(sides.fixed_modality, 1.0))
    for key, factor_key in (("tile_dedup_radius_moving_px", sides.moving_modality),
                            ("tile_dedup_radius_fixed_px", sides.fixed_modality)):
        if params[key] != base_radius:
            scaled_gates[key] = {"modality": factor_key, "power": 1, "before": base_radius,
                                 "after": params[key]}
    scale["gates"] = scaled_gates
    print(f"[Stage7] physical scale: {physical_scale.describe(scale)}"
          + ("" if scaled_gates else "; gates unchanged"), flush=True)
    for key, change in scaled_gates.items():
        print(f"[Stage7]   {key}: {change['before']:g} -> {change['after']:g} level-0 px",
              flush=True)
    out_dir = run_dir / "dense_nucleus_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A summary is the completion artifact; remove it before recomputation so a
    # failed run cannot look complete. A QC-only pass leaves it alone.
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.qc_only:
        summary_path.unlink()

    print(f"\n========== STAGE 7: DENSE {m.upper()}--{f.upper()} NUCLEUS PAIRS ==========", flush=True)
    print(f"[Stage7] run_dir={run_dir}  pair={roles.pair.name}  moving={m}  fixed={f}", flush=True)
    print(f"[Stage7] parameters={json.dumps(configured, sort_keys=True)}", flush=True)
    # Include dependency import time: from a launcher perspective that startup
    # cost is part of Stage 7, especially on the first cold run.
    started = _PROCESS_STARTED
    if not args.qc_only:
        log_stage_event(
            run_dir,
            "stage7_events",
            "cli_stage7_start",
            scan_mode=params["scan_mode"],
        )
    engine_dir = ENGINE_DIR
    transform = load_transform(transform_path)
    scan_mode = params["scan_mode"]

    if scan_mode == "stage3_tiles":
        print("[Stage7] scan_mode=stage3_tiles: densifying the existing tissue-sampled ROIs", flush=True)
        dapi, he, sampling_meta = scan_stage3_tiles(run_dir, params, engine_dir, sides)
        dapi_tile_level = int(sampling_meta[f"{m}_tile_level"])
        he_tile_level = int(sampling_meta[f"{f}_tile_level"])
        cached_detections = (
            sampling_meta.get("detection_source")
            == "stage4a_exact_stage7_cache"
        )
        dapi_meta = {
            "path": str(Path(sides.image_path("moving", info)).resolve()),
            "source": (
                "stage4a_cached_stage3_tile_detections"
                if cached_detections
                else "stage3_sampled_tiles"
            ),
            "tile_level": dapi_tile_level,
            "scale_to_level0": float(2 ** dapi_tile_level),
        }
        he_meta = {
            "path": str(Path(sides.image_path("fixed", info)).resolve()),
            "source": (
                "stage4a_cached_stage3_tile_detections"
                if cached_detections
                else "stage3_sampled_rectified_tiles"
            ),
            "tile_level": he_tile_level,
            "scale_to_level0": float(2 ** he_tile_level),
        }

    if len(dapi) == 0 or len(he) == 0:
        raise RuntimeError(f"segmentation found {m}={len(dapi)} and {f}={len(he)} nuclei; cannot pair")
    if not args.qc_only:
        write_csv_gz(nucleus_dataframe(dapi), out_dir / f"{m}_nuclei.csv.gz")
        write_csv_gz(nucleus_dataframe(he), out_dir / f"{f}_nuclei.csv.gz")

    predicted = transform_points(dapi[:, 2:4], transform)
    eligible_mask = np.isfinite(predicted).all(axis=1)
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) == 0:
        raise RuntimeError(f"the final transform projected no {m} nuclei inside the {f} image")
    candidate_edges = {}
    candidates, pairs, metrics = match_reciprocal(
        dapi, he, predicted, eligible, params, edge_sink=candidate_edges, tokens=sides.tokens)
    pairs_before_cap = len(pairs)
    max_pairs = int(params["max_output_pairs"])
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs.sort_values("confidence", ascending=False).head(max_pairs).copy().reset_index(drop=True)
        pairs["pair_id"] = np.arange(1, len(pairs) + 1, dtype=np.int64)
    selected_keys = pd.MultiIndex.from_frame(pairs[[f"{m}_nucleus_id", f"{f}_nucleus_id"]])
    candidate_keys = pd.MultiIndex.from_frame(candidates[[f"{m}_nucleus_id", f"{f}_nucleus_id"]])
    candidates["selected_for_pair_export"] = candidate_keys.isin(selected_keys)
    metrics["accepted_pairs_before_output_cap"] = int(pairs_before_cap)
    metrics["accepted_pairs"] = int(len(pairs))
    metrics["high_confidence_pairs"] = int((pairs["confidence_tier"] == "high").sum())
    high_pairs = pairs.loc[pairs["confidence_tier"] == "high"].copy()
    if args.qc_only:
        # The pairs on disk are the run's result; the pictures are drawn for them.
        saved = out_dir / "high_confidence_pairs.csv.gz"
        if not saved.is_file():
            raise SystemExit(f"--qc-only needs {saved}; run Stage 7 first")
        saved_pairs = pd.read_csv(saved)
        if len(saved_pairs) != len(high_pairs):
            print(f"[Stage7] WARN --qc-only: recomputed {len(high_pairs)} high-confidence "
                  f"pairs, the run saved {len(saved_pairs)}; drawing the saved ones",
                  flush=True)
        high_pairs = saved_pairs
    else:
        if bool(params["write_nearest_candidates"]):
            write_csv_gz(candidates, out_dir / "nearest_candidates.csv.gz")
        write_csv_gz(pairs, out_dir / "nucleus_pairs.csv.gz")
        write_csv_gz(high_pairs, out_dir / "high_confidence_pairs.csv.gz")
    anchor_qa = write_stage7_anchor_qa_images(run_dir, info, high_pairs, sides)
    tree_qa = {"dir": None, "tiles_written": 0, "tiles_skipped": 0}
    if scan_mode == "stage3_tiles":
        tree_qa = write_stage7_tree_qa_images(
            run_dir, list(sampling_meta["selected_tile_ids"]), dapi, he, predicted,
            candidate_edges, high_pairs,
            json.loads((run_dir / "tiles" / f"{f}_tile_info.json").read_text()), he_tile_level,
            sides,
        )
    qc_groups, qc_incomplete = {}, {}
    if qc_images_enabled("stage7_anchors"):
        qc_groups["stage7_anchors"] = [name for name in (anchor_qa["dapi_anchor_overlay"],
                                                         anchor_qa["he_anchor_overlay"]) if name]
    if qc_images_enabled("stage7_tree"):
        qc_groups["stage7_tree"] = list(tree_qa.get("files", []))
        if int(tree_qa.get("tiles_skipped", 0)) > 0:
            # Not "nothing to draw": the tiles are there, their Stage-3 images are not.
            # Recorded as incomplete so the next resume draws it once they are back.
            qc_incomplete["stage7_tree"] = (
                f"{tree_qa['tiles_skipped']} of "
                f"{tree_qa['tiles_skipped'] + tree_qa['tiles_written']} Stage-3 tiles have no "
                f"{f}_tile image to draw on")
    if qc_groups:
        import qc_artifacts
        qc_artifacts.write_stage7_manifest(run_dir, qc_groups, qc_incomplete)
        print(f"[Stage7] QC manifest: {qc_artifacts.STAGE7_MANIFEST} covers "
              + ", ".join(f"{g} ({len(v)} files"
                          f"{', INCOMPLETE: ' + qc_incomplete[g] if g in qc_incomplete else ''})"
                          for g, v in qc_groups.items()), flush=True)
    if args.qc_only:
        print("[Stage7] --qc-only: QC images redrawn; CSVs, summary and timing untouched",
              flush=True)
        return

    elapsed = time.perf_counter() - started
    summary = {
        "stage": 7,
        "mode": f"{m}_to_{f}_dense_nucleus_pairs",
        "pair": roles.pair.name,
        "moving_token": m,
        "fixed_token": f,
        "algorithm": (
            "stage4a_cached_exact_detectors_plus_reciprocal_nearest_neighbour"
            if sampling_meta.get("detection_source") == "stage4a_exact_stage7_cache"
            else f"{scan_mode}_fast_masks_plus_reciprocal_nearest_neighbour"
        ),
        "transform_path": str(transform_path),
        "transform_type": transform["kind"],
        "coordinate_system": "original image level-0 pixels (x, y)",
        f"{m}_image": dapi_meta,
        f"{f}_image": he_meta,
        "sampling": sampling_meta,
        f"{m}_nuclei": int(len(dapi)),
        f"{f}_nuclei": int(len(he)),
        **metrics,
        "elapsed_seconds": elapsed,
        "parameters": configured,
        "physical_scale": scale,
        "effective_gates": {**{key: params[key] for key in gate_spec if key in params},
                            "tile_dedup_radius_moving_px": params["tile_dedup_radius_moving_px"],
                            "tile_dedup_radius_fixed_px": params["tile_dedup_radius_fixed_px"]},
        "outputs": {
            f"{m}_nuclei": f"{m}_nuclei.csv.gz",
            f"{f}_nuclei": f"{f}_nuclei.csv.gz",
            "nearest_candidates": "nearest_candidates.csv.gz" if bool(params["write_nearest_candidates"]) else None,
            "pairs": "nucleus_pairs.csv.gz",
            "high_confidence_pairs": "high_confidence_pairs.csv.gz",
            "dapi_anchor_overlay": anchor_qa["dapi_anchor_overlay"],
            "he_anchor_overlay": anchor_qa["he_anchor_overlay"],
            "tree_qc": tree_qa["dir"],
        },
        "anchor_qa": anchor_qa,
        "tree_qc": tree_qa,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log_stage_event(
        run_dir,
        "stage7_events",
        "cli_stage7_done",
        elapsed_seconds=elapsed,
        accepted_pairs=int(len(pairs)),
        high_confidence_pairs=int(len(high_pairs)),
    )
    pipeline_timing = finalize_pipeline_timing(
        run_dir,
        stage7_elapsed_seconds=elapsed,
        source="stage7",
    )
    print(f"[Stage7] DONE: accepted={len(pairs):,}, high-confidence={len(high_pairs):,}, "
          f"elapsed={elapsed:.1f}s", flush=True)
    total_elapsed = pipeline_timing.get("total_elapsed_seconds")
    if total_elapsed is not None:
        print(
            f"[Stage7] pipeline timing: stage7={elapsed:.1f}s, "
            f"total={float(total_elapsed):.1f}s",
            flush=True,
        )
    print(f"[Stage7] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
