"""Shared fast nucleus detectors used by Stage 4a and Stage 7.

Stage 4a runs over the same Stage-3 tiles before Stage 7.  Keeping the dense
detectors here lets Stage 4a cache their local centroid/area output while its
worker processes already have the tile images in memory.  Stage 7 can then
perform only coordinate restoration, deduplication, and matching.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

CACHE_NPZ_NAME = "stage7_detection_cache.npz"
CACHE_META_NAME = "stage7_detection_cache.json"
CACHE_SCHEMA_VERSION = 1


DETECTION_PARAM_KEYS = (
    "dapi_threshold_offset",
    "dapi_min_contrast_fraction",
    "dapi_max_foreground_fraction",
    "he_n_smooth",
    "he_intensity_threshold",
    "he_upsample_factor",
    "he_min_tissue_fraction",
    "he_min_area_level0_px",
)

DETECTION_DEFAULTS = {
    "dapi_threshold_offset": 0,
    "dapi_min_contrast_fraction": 0.002,
    "dapi_max_foreground_fraction": 0.65,
    "he_n_smooth": 2,
    "he_intensity_threshold": 0.6,
    "he_upsample_factor": 1,
    "he_min_tissue_fraction": 0.002,
    "he_min_area_level0_px": 30.0,
}


def detection_params_snapshot(params):
    """Return the JSON-safe parameter subset that determines detections."""
    return {key: params[key] for key in DETECTION_PARAM_KEYS}


def _to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (0.2, 99.8))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip(
        (array.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255
    ).astype(np.uint8)


def segment_dapi_centroids(gray, params):
    """Return local ``x, y, area`` rows for a Stage-3 DAPI tile."""
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = gray[..., 0]
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    lo, hi = np.percentile(finite, (5.0, 99.8))
    dtype_max = (
        float(np.iinfo(gray.dtype).max)
        if np.issubdtype(gray.dtype, np.integer)
        else max(1.0, float(np.nanmax(finite)))
    )
    if hi - lo < float(params["dapi_min_contrast_fraction"]) * dtype_max:
        return np.empty((0, 3), dtype=np.float64)

    gray8 = np.clip(
        (gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-8)),
        0,
        255,
    ).astype(np.uint8)
    otsu, _ = cv2.threshold(
        gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = int(
        np.clip(
            round(float(otsu)) + int(params["dapi_threshold_offset"]), 0, 255
        )
    )
    mask = (gray8 > threshold).astype(np.uint8) * 255
    if float((mask > 0).mean()) > float(params["dapi_max_foreground_fraction"]):
        return np.empty((0, 3), dtype=np.float64)

    flood = mask.copy()
    padded = cv2.copyMakeBorder(
        flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    cv2.floodFill(padded, None, (0, 0), 255)
    holes = cv2.bitwise_not(padded[1:-1, 1:-1])
    mask = cv2.bitwise_or(mask, holes)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        [
            centroids[1:, 0],
            centroids[1:, 1],
            stats[1:, cv2.CC_STAT_AREA],
        ]
    ).astype(np.float64)


def he_nucleus_labels(rgb, params, engine_dir=None):
    """Watershed-split H&E nucleus labels for one tile, and the scale they are at.

    This is the detection `segment_he_centroids` has always run, factored out so the
    same result can also be written as a mask. It differs from Stage 4a's Macenko
    path in the two ways that matter on pigment-bearing slides: the hematoxylin
    channel comes from a *fixed* stain vector rather than one estimated per tile, and
    the cut is Otsu rather than a constant.

    Returns ``(labels, upsample)``; ``labels`` is None when the tile is skipped or
    holds nothing.
    """
    rgb8 = _to_uint8(rgb)
    upsample = max(1, int(params["he_upsample_factor"]))
    if rgb8.ndim != 3 or rgb8.shape[2] < 3:
        return None, upsample
    dark_fraction = float((np.min(rgb8[..., :3], axis=2) < 230).mean())
    if dark_fraction < float(params["he_min_tissue_fraction"]):
        return None, upsample

    if upsample > 1:
        rgb8 = cv2.resize(
            rgb8,
            None,
            fx=upsample,
            fy=upsample,
            interpolation=cv2.INTER_CUBIC,
        )

    optical_density = -np.log(
        (rgb8[..., :3].astype(np.float32) + 1.0) / 256.0
    )
    concentration = (
        0.650 * optical_density[..., 0]
        + 0.704 * optical_density[..., 1]
        + 0.286 * optical_density[..., 2]
    )
    lo, hi = np.percentile(concentration, (5.0, 99.8))
    if hi <= lo + 1e-8:
        return None, upsample
    hematoxylin8 = np.clip(
        (concentration - lo) * (255.0 / (hi - lo)), 0, 255
    ).astype(np.uint8)
    smooth = max(0, int(params["he_n_smooth"]))
    if smooth:
        kernel = 2 * smooth + 1
        hematoxylin8 = cv2.GaussianBlur(
            hematoxylin8, (kernel, kernel), 0
        )
    otsu, _ = cv2.threshold(
        hematoxylin8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(
        float(otsu), 255.0 * float(params["he_intensity_threshold"])
    )
    binary = (hematoxylin8 > threshold).astype(np.uint8) * 255

    if engine_dir is not None:
        engine_dir = str(Path(engine_dir))
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
    from my_utils import split_touching_objects_watershed

    work_pixel_area0 = float(params.get("_work_pixel_area_level0", 1.0))
    min_work_area = max(
        2,
        int(
            math.floor(
                float(params["he_min_area_level0_px"])
                / work_pixel_area0
                * upsample
                * upsample
            )
        ),
    )
    labels, _ = split_touching_objects_watershed(
        binary,
        min_area=min_work_area,
        opening_ksize=3,
        dist_frac=0.42,
        peak_min_dist=0,
    )
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0 or int(labels.max()) == 0:
        return None, upsample
    return labels, upsample


def segment_he_centroids(rgb, params, engine_dir=None):
    """Return local ``x, y, area`` rows for a Stage-3 H&E tile."""
    labels, upsample = he_nucleus_labels(rgb, params, engine_dir)
    if labels is None:
        return np.empty((0, 3), dtype=np.float64)

    from skimage.measure import regionprops_table

    props = regionprops_table(labels, properties=("area", "centroid"))
    if len(props["area"]) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        [
            np.asarray(props["centroid-1"], dtype=np.float64) / upsample,
            np.asarray(props["centroid-0"], dtype=np.float64) / upsample,
            np.asarray(props["area"], dtype=np.float64)
            / (upsample * upsample),
        ]
    )
