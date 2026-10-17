# ============================================================
# stage3_core.py - Stage 3: tile sampling and extraction, without a window
#
# Everything Stage 3 computes lives here and imports no GUI toolkit. `3_get_tiles.py`
# is the Tk window around it and `cli_stage3.py` the headless entry point; both run
# the same `Stage3Sampler`. (Split out of 3_get_tiles.py in 4.11 so the CLI process
# never loads a GUI toolkit.)
#
#
# Builds a tissue mask from the moving image, samples tile centroids (CVT), then
# crops the moving tiles and rectifies the fixed tiles into <RUN_DIR>/tiles/ with
# <moving>_tile_info.json / <fixed>_tile_info.json.
#
# Each side is read, masked and written by what it *is*: a DAPI side is a 16-bit
# intensity plane whose tissue is where the signal is; an H&E side is RGB whose
# tissue is where the hematoxylin is. The side's token from `run_roles` names its
# files, so `dapi-he` produces exactly the files it always did and `he-he` gets
# `*_hefg.png` / `*_hebg.png` from the same code that used to need a second engine.
# ============================================================
import os
import sys
import json
import re
import time
from datetime import datetime
import math
from pathlib import Path
import numpy as np
import cv2
_ENGINE_DIR = str(Path(__file__).resolve().parent)
if sys.path[:1] != [_ENGINE_DIR]:
    if _ENGINE_DIR in sys.path:
        sys.path.remove(_ENGINE_DIR)
    sys.path.insert(0, _ENGINE_DIR)
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from PIL import Image, ImageOps
print("[Stage3] imports done", flush=True)
Image.MAX_IMAGE_PIXELS = None

# -----------------------------
# Utilities
# -----------------------------
from my_utils import qc_images_enabled, scalar_mpp, choose_level_by_target_mpp
_VERSION_ROOT = str(Path(__file__).resolve().parents[1])
if _VERSION_ROOT not in sys.path:
    sys.path.insert(0, _VERSION_ROOT)
import run_roles
from run_artifacts import initial_alignment_path


LOW_SIGNAL_RESCUE_DEFAULTS = {
    "enabled": False,
    # Candidate acquisition components are inferred from nonzero DAPI support at the
    # Stage-3 sampling level.  Express the size floor relative to a sampled tile so
    # it scales with pyramid level and physical pixel size.
    "min_component_tile_area_factor": 4.0,
    "min_component_area_px": 256,
    "support_close_ksize": 3,
    # Rescue only components the global mask essentially abandoned.
    "max_existing_dense_fraction": 0.02,
    # Reject flat scanner background; a rescued component must have nucleus-like
    # local dynamic range before its own robust normalization is considered.
    "local_low_percentile": 5.0,
    "local_high_percentile": 99.5,
    "min_local_contrast_8bit": 8.0,
    "local_tissue_percentile": 45.0,
    "local_dense_percentile": 25.0,
    "blur_ksize": 13,
    "tissue_close_ksize": 25,
    "dense_close_ksize": 11,
    "min_tissue_area_px": 256,
    "min_tissue_component_fraction": 0.05,
    "minimum_sampled_tiles_per_component": 1,
}


def _rss_gib():
    """Resident size of this process tree, for the Stage-3 memory probes."""
    try:
        import psutil
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total / (1024 ** 3)
    except Exception:
        return float("nan")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def log_event(run_dir, event_name, stage="stage3", **extra):
    out_json = Path(run_dir) / "pipeline_times.json"
    now_str = datetime.now().isoformat(timespec="seconds")

    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    stage_key = f"{stage}_events"
    if stage_key not in data or not isinstance(data[stage_key], list):
        data[stage_key] = []

    rec = {
        "event": event_name,
        "time": now_str,
    }
    rec.update(extra)
    data[stage_key].append(rec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[LOG] {event_name} -> {out_json}", flush=True)

def load_stage3_params(script_path: Path, pair=None):
    """Read the shared pair block, with the same defaults for GUI and CLI.

    ``script_path`` remains for compatibility with older callers. Run entry points
    pass the pair recorded in images_info, not the launcher's optional environment.
    """
    del script_path

    stage3 = {
        "random_seed": 43,
        "n_tiles": 120,
        "tile_size": 600,
        "min_dist_factor": 1.5,
    }

    from my_utils import cli_params_block
    _blk = cli_params_block(pair)
    if isinstance(_blk.get("stage3", {}), dict):
        stage3.update(_blk.get("stage3", {}))

    # normalize types
    stage3["random_seed"] = int(stage3.get("random_seed", 43))
    stage3["n_tiles"]    = int(stage3.get("fallback_n_tiles", stage3.get("n_tiles", 120)))
    stage3["tile_size"]          = float(stage3.get("fallback_tile_size", stage3.get("tile_size", 600)))
    stage3["min_dist_factor"]    = float(stage3.get("min_dist_factor", 1.5))

    rescue = dict(LOW_SIGNAL_RESCUE_DEFAULTS)
    rescue.update(stage3.get("low_signal_component_rescue", {}) or {})
    rescue["enabled"] = bool(rescue["enabled"])
    for key in (
        "min_component_area_px", "support_close_ksize", "blur_ksize",
        "tissue_close_ksize", "dense_close_ksize", "min_tissue_area_px",
        "minimum_sampled_tiles_per_component",
    ):
        rescue[key] = int(rescue[key])
    for key in (
        "min_component_tile_area_factor", "max_existing_dense_fraction",
        "local_low_percentile", "local_high_percentile", "min_local_contrast_8bit",
        "local_tissue_percentile", "local_dense_percentile",
        "min_tissue_component_fraction",
    ):
        rescue[key] = float(rescue[key])
    if not 0 <= rescue["max_existing_dense_fraction"] <= 1:
        raise ValueError("max_existing_dense_fraction must be between 0 and 1")
    if not 0 <= rescue["min_tissue_component_fraction"] <= 1:
        raise ValueError("min_tissue_component_fraction must be between 0 and 1")
    for low, high in (
        ("local_low_percentile", "local_high_percentile"),
        (None, "local_tissue_percentile"),
        (None, "local_dense_percentile"),
    ):
        if low is not None and not 0 <= rescue[low] < rescue[high] <= 100:
            raise ValueError(f"expected 0 <= {low} < {high} <= 100")
        if low is None and not 0 <= rescue[high] <= 100:
            raise ValueError(f"expected 0 <= {high} <= 100")
    for key in ("support_close_ksize", "blur_ksize", "tissue_close_ksize", "dense_close_ksize"):
        rescue[key] = max(0, rescue[key])
        if rescue[key] and rescue[key] % 2 == 0:
            rescue[key] += 1
    stage3["low_signal_component_rescue"] = rescue

    return stage3

def load_dapi_lut_threshold_from_images_info(run_dir: Path, default=1000) -> int:
    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        print(f"[WARN] missing {info_path}, fallback threshold={default}", flush=True)
        return int(default)

    try:
        with open(info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read images_info.json: {e}, fallback threshold={default}", flush=True)
        return int(default)

    try:
        moving = run_roles.read(info).moving
        if moving.modality == "dapi" and moving.threshold is not None:
            return int(moving.threshold)
    except (KeyError, TypeError, ValueError):
        pass  # Also accept old threshold-only metadata without a full role record.
    for k in ("DAPI_LUT_threshold", "dapi_lut_threshold", "dapi_lut_thr", "lut_threshold"):
        if k in info:
            try:
                return int(info[k])
            except Exception:
                pass

    print(f"[WARN] images_info.json has no DAPI LUT threshold key, fallback threshold={default}", flush=True)
    return int(default)

def load_image_any(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

class StepTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
    def mark(self, name):
        now = time.perf_counter()
        print(f"[TIMER] {name:<30s}: {now - self.last:8.2f}s (total {now - self.t0:8.2f}s)", flush=True)
        self.last = now

# -----------------------------
# image helpers
# -----------------------------
def apply_orientation_case(img: np.ndarray, case_id: int) -> np.ndarray:
    """
    Apply orientation case to image (H,W) or (H,W,C).
    case_id definition must match your Stage1.
    """
    if img is None:
        return None
    if case_id == 0:
        return img
    if case_id == 1:   # rot90 CW
        return np.rot90(img, k=3)
    if case_id == 2:   # rot180
        return np.rot90(img, k=2)
    if case_id == 3:   # rot90 CCW
        return np.rot90(img, k=1)
    if case_id == 4:   # flip UD
        return np.flipud(img)
    if case_id == 5:   # flip LR
        return np.fliplr(img)
    if case_id == 6:   # transpose
        if img.ndim == 2:
            return img.T
        return np.transpose(img, (1, 0, 2))
    if case_id == 7:   # transverse (anti-diagonal): rot90 CCW + flip LR
        return np.fliplr(np.rot90(img, k=1))
    raise ValueError(f"Unknown case_id={case_id}")

def cv2_to_pil(img):
    """cv2 image -> PIL.Image (RGB or L)."""
    if img is None:
        return None
    if img.ndim == 2:
        return Image.fromarray(img)
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    return Image.fromarray(img)

def fit_to_tile(pil_img: Image.Image, size=(420, 420), bg=240):
    """Resize with aspect ratio and pad to fixed tile."""
    canvas = Image.new("RGB", size, (bg, bg, bg))
    if pil_img is None:
        return canvas
    if pil_img.mode not in ("RGB", "RGBA", "L"):
        pil_img = pil_img.convert("RGB")
    pil_contained = ImageOps.contain(pil_img, size)
    x = (size[0] - pil_contained.width) // 2
    y = (size[1] - pil_contained.height) // 2
    if pil_contained.mode == "RGBA":
        tmp = Image.new("RGBA", size, (bg, bg, bg, 255))
        tmp.paste(pil_contained, (x, y), pil_contained)
        return tmp.convert("RGB")
    canvas.paste(pil_contained, (x, y))
    return canvas

def normalize_uint16_to_uint8(img16: np.ndarray) -> np.ndarray:
    g = img16.astype(np.float32)
    mn, mx = float(np.min(g)), float(np.max(g))
    g = (g - mn) / (mx - mn + 1e-8)
    return (g * 255.0).astype(np.uint8)

def to_gray_uint8(img):
    """
    Accepts:
      - (H,W) uint8/uint16/float
      - (H,W,3) uint8/uint16/float
    Returns:
      - (H,W) uint8
    """
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 3:
            if img.dtype != np.uint8:
                img8 = normalize_uint16_to_uint8(img[..., 0])   # Use the first channel as grayscale preview.
                return img8
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., 0]
    if img.dtype == np.uint8:
        return img
    return normalize_uint16_to_uint8(img)

def draw_points_overlay(dapi_img, points_xy, tile_size=128, save_path=None):
    g8 = to_gray_uint8(dapi_img)
    base = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    half = int(tile_size) // 2
    for i, (x, y) in enumerate(points_xy):
        x, y = int(x), int(y)
        x0, y0 = x - half, y - half
        x1, y1 = x + half, y + half
        cv2.rectangle(base, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.circle(base, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(base, f"{i:02d}", (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    if save_path:
        cv2.imwrite(save_path, base)
    return base

# -----------------------------
# mask helpers
# -----------------------------
def apply_density_filter(mask_tissue_255: np.ndarray,
                         density_8u: np.ndarray,
                         mode="percentile",
                         p=40,
                         thr_fixed=30,
                         morph_close=0,
                         min_area=0):
    assert mask_tissue_255.shape == density_8u.shape
    tissue = (mask_tissue_255 > 0)
    if tissue.sum() == 0:
        return np.zeros_like(mask_tissue_255, dtype=np.uint8)

    vals = density_8u[tissue]
    thr = np.percentile(vals, p) if mode == "percentile" else thr_fixed
    keep = tissue & (density_8u >= thr)
    out = (keep.astype(np.uint8) * 255)

    if morph_close and morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)

    if min_area and min_area > 0:
        bw = (out > 0).astype(np.uint8)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        out2 = np.zeros_like(out)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out2[lab == i] = 255
        out = out2

    return out

def make_tissue_mask_from_dapi_gray(
    dapi_gray16: np.ndarray,
    blur_ksize=13,
    thr_mode="percentile",
    thr_percentile=45,
    thr_fixed=18,
    morph_close=25,
    morph_open=0,
    min_area=3000
):
    if dapi_gray16.ndim == 3:
        dapi_gray16 = dapi_gray16[..., 0]
    if dapi_gray16.ndim != 2:
        raise ValueError(f"dapi_gray16 must be 2D, got {dapi_gray16.shape}")

    g8 = normalize_uint16_to_uint8(dapi_gray16)
    blur_ksize = int(blur_ksize) | 1
    density = cv2.GaussianBlur(g8, (blur_ksize, blur_ksize), 0)

    if thr_mode == "otsu":
        _, bw = cv2.threshold(density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thr_mode == "fixed":
        _, bw = cv2.threshold(density, int(thr_fixed), 255, cv2.THRESH_BINARY)
    else:
        vals = density[density > 0]
        if len(vals) == 0:
            return np.zeros_like(density, dtype=np.uint8), density
        thr = np.percentile(vals, thr_percentile)
        _, bw = cv2.threshold(density, int(thr), 255, cv2.THRESH_BINARY)

    if morph_open and morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    if morph_close and morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    out = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255

    return out, density

def make_available_mask_by_eroding_valid_region(
    valid_mask_255: np.ndarray,
    radius: int,
    outside_is_invalid: bool = False,
):
    """
    Convert a valid-region mask into a sampling-availability mask.

    Parameters
    ----------
    valid_mask_255 : np.ndarray
        Binary mask where:
          255 = valid region
          0   = invalid region

    radius : int
        Erosion radius used to keep sampled tile centers away from boundaries.

    outside_is_invalid : bool
        If true, pixels beyond the decoded canvas are invalid and the erosion also
        contracts the image boundary. False preserves the 4.11/OpenCV morphology
        behaviour used by the H&E sampling path.

    Returns
    -------
    np.ndarray
        Availability mask where:
          0   = available for sampling
          255 = unavailable
    """
    valid = (valid_mask_255 > 0).astype(np.uint8)

    if radius > 0:
        ksize = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if outside_is_invalid:
            # A DAPI tile must fit inside the decoded canvas. OpenCV's morphology
            # default uses a favourable border value for erosion, which can otherwise
            # leave centres on row/column zero.
            valid = cv2.erode(
                valid,
                kernel,
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            valid = cv2.erode(valid, kernel, iterations=1)

    return np.where(valid > 0, 0, 255).astype(np.uint8)

# -----------------------------
# sampling helpers
# -----------------------------
def derive_he_mpp_from_homography(dapi_base_mpp, h_mat, dapi_level, he_level):
    """Recover the H&E level-0 pixel size from the Stage-2 homography.

    Used when the H&E slide carries no physical spacing of its own. The homography
    maps the *Stage-1 level* images onto each other -- DAPI level `dapi_level` to H&E
    level `he_level`, not level 0 to level 0 -- so both level factors come back out:

        he_mpp0 = dapi_mpp0 * 2**dapi_level / scale / 2**he_level

    where `scale` is the linear scale of the homography's 2x2 block. Getting those
    factors wrong is silent: on mouse_brain, dividing a level-0 DAPI spacing by a
    level-5-to-level-3 scale returns 0.1411 um/px instead of 0.5645, which still looks
    like a plausible slide and selects a tile level two steps too fine.

    Returns None when the inputs cannot support a derivation, so the caller can fall
    back to the assumed 40x spacing.
    """
    base = scalar_mpp(dapi_base_mpp)
    if not base:
        return None
    try:
        block = np.asarray(h_mat, dtype=np.float64)[:2, :2]
        scale = float(np.sqrt(abs(np.linalg.det(block))))
    except Exception:
        return None
    if not np.isfinite(scale) or scale <= 1e-9:
        return None
    derived = float(base) * (2.0 ** int(dapi_level)) / scale / (2.0 ** int(he_level))
    # Reject a value no brightfield slide could have rather than pass it downstream.
    if not np.isfinite(derived) or not (0.01 < derived < 20.0):
        return None
    return derived


def tile_size_px_for_footprint(footprint_um, mpp):
    """Tile side in pixels that covers `footprint_um` micrometres at `mpp`."""
    if not mpp or mpp <= 0:
        raise ValueError("mpp must be positive")
    return int(round(float(footprint_um) / float(mpp)))


def choose_tile_level(x, k,
                         anchor_sampling_tile=50,
                         min_tile_level=1,
                         max_tile_level=None):

    # Legacy fallback, used only when the physical pixel size is unknown. Tile size
    # determines the extracted footprint, not the pyramid offset.
    n = int(x) - 3

    if max_tile_level is not None:
        max_tile_level = int(max_tile_level)
        n = min(n, max_tile_level)
        min_tile_level = min(int(min_tile_level), max_tile_level)

    n = max(min_tile_level, n)

    return int(n)

def get_valid_coords(mask_available, max_points=250_000, seed=0):
    ys, xs = np.where(mask_available == 0)
    coords = np.column_stack([xs, ys]).astype(np.int32)
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
    return coords

def initialize_points(coords_valid, N, min_dist, seed=0):
    rng = np.random.default_rng(seed)
    points = []
    attempts = 0
    while len(points) < N and attempts < 500000:
        x, y = coords_valid[rng.integers(0, len(coords_valid))]
        if points:
            d = np.linalg.norm(np.asarray(points) - np.asarray([x, y]), axis=1)
            if np.any(d < min_dist):
                attempts += 1
                continue
        points.append([x, y])
        attempts += 1
    if len(points) < N:
        print(f"[WARN] only initialized {len(points)}/{N} points (min_dist too large or region too small).", flush=True)
    return np.asarray(points, dtype=np.float32)

def repel_too_close(points, coords_valid, min_sep, seed=0, max_rounds=20, max_tries=2000):
    """
    Soft constraint: only fix pairs closer than min_sep.
    """
    rng = np.random.default_rng(seed)
    points = points.copy()

    for r in range(max_rounds):
        tree = cKDTree(points)
        pairs = list(tree.query_pairs(min_sep))
        if not pairs:
            return points, True

        bad = np.zeros(len(points), dtype=np.int32)
        for i, j in pairs:
            bad[i] += 1
            bad[j] += 1
        order = np.argsort(-bad)

        moved = False
        for idx in order:
            if bad[idx] == 0:
                break

            for _ in range(max_tries):
                cand = coords_valid[rng.integers(0, len(coords_valid))]
                d, _ = tree.query(cand, k=1)
                if d >= min_sep:
                    points[idx] = cand
                    moved = True
                    break

        if not moved:
            return points, False

    return points, False

def cvt_masked(mask_available, N_POINTS=80, MIN_DIST=7, ITERATIONS=50, seed=0):
    rng = np.random.default_rng(seed)
    coords_valid = get_valid_coords(mask_available, seed=seed)
    if len(coords_valid) == 0:
        raise ValueError("No available pixels to sample from. (mask_available==0 is empty)")

    points = initialize_points(coords_valid, N_POINTS, MIN_DIST, seed=seed)
    if len(points) == 0:
        raise ValueError("Failed to initialize any points. Check MIN_DIST / mask size.")

    N = len(points)

    # soft separation: set as a fraction of MIN_DIST
    MIN_SEP = 0.55 * float(MIN_DIST)   # ~= (0.8/1.5)*MIN_DIST

    for it in range(ITERATIONS):
        tree = cKDTree(points)
        _, idxs = tree.query(coords_valid)

        new_points = points.copy()
        for i in range(N):
            region_idx = np.where(idxs == i)[0]
            if len(region_idx) == 0:
                K = 200
                cand = coords_valid[rng.integers(0, len(coords_valid), size=K)]
                d, _ = tree.query(cand, k=1)
                new_points[i] = cand[np.argmax(d)]
            else:
                sub = coords_valid[region_idx]
                centroid = sub.mean(axis=0)
                k = np.argmin(np.sum((sub - centroid) ** 2, axis=1))
                new_points[i] = sub[k]

        points, ok = repel_too_close(new_points, coords_valid, MIN_SEP, seed=seed + it + 1)
    return points.astype(np.int32)

# -----------------------------
# geometry helpers
# -----------------------------
def _tile_to_xywh(p):
    """
    Normalize tile spec to (x0, y0, w, h).
    Accepts:
      - dict with x0,y0,w,h
      - tuple/list (x0,y0,w,h)
    """
    if isinstance(p, dict):
        return float(p["x0"]), float(p["y0"]), float(p["w"]), float(p["h"])
    if isinstance(p, (list, tuple)) and len(p) == 4:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError(f"Unsupported tile format: {type(p)} {p}")


def projected_he_tile_max_side(tiles, h_mat, *, rescale_factor=1.0,
                               margin_ratio=0.2):
    """Return the largest rectified H&E tile dimension at one pyramid level."""
    matrix = np.asarray(h_mat, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"h_mat must be 3x3 or 2x3, got {matrix.shape}")

    largest_side = 0
    expansion = 1.0 + float(margin_ratio)
    for tile in tiles:
        x0, y0, width, height = _tile_to_xywh(tile)
        center_x = x0 + width / 2.0
        center_y = y0 + height / 2.0
        half_width = width * expansion / 2.0
        half_height = height * expansion / 2.0
        corners = np.array([
            [center_x - half_width, center_y - half_height, 1.0],
            [center_x + half_width, center_y - half_height, 1.0],
            [center_x + half_width, center_y + half_height, 1.0],
            [center_x - half_width, center_y + half_height, 1.0],
        ])
        projected = corners @ matrix.T
        denominators = projected[:, 2]
        if (not np.all(np.isfinite(projected))
                or np.any(np.abs(denominators) < 1e-9)):
            raise ValueError("cannot estimate H&E tile dimensions from invalid homography")

        points = projected[:, :2] / denominators[:, None]
        points *= float(rescale_factor)
        projected_width = (
            np.linalg.norm(points[0] - points[1])
            + np.linalg.norm(points[3] - points[2])
        ) / 2.0
        projected_height = (
            np.linalg.norm(points[1] - points[2])
            + np.linalg.norm(points[0] - points[3])
        ) / 2.0
        largest_side = max(
            largest_side,
            max(2, int(round(projected_width))),
            max(2, int(round(projected_height))),
        )

    return largest_side


def choose_he_tile_level(he_level, tiles, h_mat, dapi_tile_size, *,
                         margin_ratio=0.2, max_tile_size_factor=4.0,
                         min_tile_level=1, start_level=None):
    """Downshift from `start_level` until the projected H&E tiles fit the size cap.

    `start_level` is normally the level whose physical pixel size matches the DAPI tile
    level, so both sides of a pair are read at the same micrometres per pixel. Without it
    the start is three levels finer than the Stage-1 display level, which ties the tile
    resolution to a display choice rather than to the tissue.
    """
    he_level = int(he_level)
    finest_level = min(int(min_tile_level), he_level)
    default_start = max(finest_level, he_level - 3)
    read_level = default_start if start_level is None else max(finest_level, int(start_level))
    maximum_side = float(dapi_tile_size) * float(max_tile_size_factor)
    if maximum_side <= 0:
        raise ValueError("dapi_tile_size and max_tile_size_factor must be positive")

    while True:
        projected_side = projected_he_tile_max_side(
            tiles,
            h_mat,
            rescale_factor=2 ** (he_level - read_level),
            margin_ratio=margin_ratio,
        )
        if projected_side <= maximum_side:
            return read_level, projected_side
        read_level += 1


def centroids_to_tiles(points_xy, tile_size):
    half = tile_size / 2.0
    tiles = []
    for (x, y) in points_xy:
        tiles.append({"x0": float(x - half), "y0": float(y - half), "w": float(tile_size), "h": float(tile_size)})
    return tiles

def prune_tiles_out_of_he(
    tiles,
    h_mat,
    he_img_shape,
    rescale_factor=1.0,
    margin_ratio=0.0,
    min_overlap_ratio=0.15,
    require_center_inside=False,
):
    """
    Prune DAPI tiles that project mostly outside HE image.

    Args
    ----
    tiles : list[dict or tuple]
        DAPI tiles, each as (x0,y0,w,h) or {"x0","y0","w","h"} in DAPI coords.
    h_mat : 3x3 or 2x3
        DAPI -> HE transform.
    he_img_shape : tuple
        he_rgb.shape
    rescale_factor : float
        factor converting HE coords -> current he_img pixel coords
    margin_ratio : float
        optional expansion ratio before projection, same convention as save_he_tiles.
        Usually set to 0.0 for pruning based on the true DAPI tile footprint.
    min_overlap_ratio : float
        keep tile only if projected polygon overlap area / projected polygon area >= this.
    require_center_inside : bool
        if True, projected tile center must also be inside HE image.

    Returns
    -------
    kept_tiles, kept_indices, dropped_indices
    """
    H = np.asarray(h_mat, dtype=float)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"h_mat must be 3x3 or 2x3, got {H.shape}")

    h_img, w_img = he_img_shape[:2]
    img_poly = Polygon([(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)])

    kept_tiles = []
    kept_indices = []
    dropped_indices = []

    for idx, p in enumerate(tiles):
        x0, y0, w, h = _tile_to_xywh(p)
        x1 = x0 + w
        y1 = y0 + h

        # optional expand
        if margin_ratio and margin_ratio != 0:
            expand = 1.0 + float(margin_ratio)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            half_w = (w / 2.0) * expand
            half_h = (h / 2.0) * expand
            x0p, x1p = cx - half_w, cx + half_w
            y0p, y1p = cy - half_h, cy + half_h
        else:
            x0p, x1p, y0p, y1p = x0, x1, y0, y1

        corners = np.array([
            [x0p, y0p],
            [x1p, y0p],
            [x1p, y1p],
            [x0p, y1p],
        ], dtype=float)

        corners_h = np.hstack([corners, np.ones((4, 1), dtype=float)])
        proj = (H @ corners_h.T).T
        ww = proj[:, 2:3]

        if not np.all(np.isfinite(ww)):
            dropped_indices.append(idx)
            continue

        eps = 1e-9
        ww_safe = np.where(np.abs(ww) < eps, np.sign(ww) * eps + (ww == 0) * eps, ww)
        he_xy = proj[:, :2] / ww_safe
        he_xy = he_xy * float(rescale_factor)

        if not np.all(np.isfinite(he_xy)):
            dropped_indices.append(idx)
            continue

        try:
            tile_poly = Polygon(he_xy)
        except Exception:
            dropped_indices.append(idx)
            continue

        if (not tile_poly.is_valid) or tile_poly.area <= 1e-6:
            dropped_indices.append(idx)
            continue

        inter = tile_poly.intersection(img_poly)
        overlap_ratio = inter.area / tile_poly.area if tile_poly.area > 0 else 0.0

        keep = overlap_ratio >= float(min_overlap_ratio)

        if keep and require_center_inside:
            center = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0, 1.0]], dtype=float)
            cproj = (H @ center.T).T
            cw = cproj[:, 2:3]
            cw_safe = np.where(np.abs(cw) < eps, np.sign(cw) * eps + (cw == 0) * eps, cw)
            cxy = (cproj[:, :2] / cw_safe) * float(rescale_factor)
            cx_he, cy_he = float(cxy[0, 0]), float(cxy[0, 1])
            if not (0 <= cx_he < w_img and 0 <= cy_he < h_img):
                keep = False

        if keep:
            kept_tiles.append(p)
            kept_indices.append(idx)
        else:
            dropped_indices.append(idx)

    return kept_tiles, kept_indices, dropped_indices

# -----------------------------
# tile saving helpers
# -----------------------------
# -----------------------------
# One side, by modality
# -----------------------------
def read_side_image(path, level, modality, workers=None, channel_index=None):
    """Read one side the way its modality needs: DAPI as a 16-bit plane, H&E as RGB."""
    from my_utils import read_image

    extra = {} if workers is None else {"maxworkers": workers}
    if modality == "dapi":
        # One selected channel: replicating a 267-megapixel uint16 plane to RGB first costs
        # 1.0 GiB for nothing.
        channel = "dapi" if channel_index is None else int(channel_index)
        image, _ = read_image(path, keep_16bit=True, level=level, channel=channel,
                              force_rgb=False, **extra)
        return image[..., 0] if image.ndim == 3 else image
    image, _ = read_image(path, keep_16bit=True, level=level, channel="he", **extra)
    return image


def moving_tissue_masks(image, modality, threshold=None):
    """Return ``(tissue, dense, support)`` masks, with 255 meaning yes.

    The same two-step rule for both modalities -- a tissue region, then the part of
    it dense enough in nuclear signal -- with the signal chosen by what the image is.
    The support mask is geometric: it says which pixels belong to the decoded image
    canvas, not whether their measured intensity happens to be nonzero. In particular,
    zero is a legitimate DAPI value after acquisition, interpolation, gamma correction,
    or Poisson quantisation. Treating ``image > 0`` as support turns those legitimate
    zeros into one-pixel holes; eroding that mask then destroys otherwise dense tissue.

    Both current readers return a rectangular decoded canvas, so every pixel is in
    support. A future reader that exposes an acquisition-validity mask can replace this
    full mask without changing the tissue/density rules below. The density mask already
    excludes blank padding from centroid sampling.
    """
    if modality == "dapi":
        mask_tissue, density = make_tissue_mask_from_dapi_gray(
            image,
            blur_ksize=13,
            thr_mode="percentile",
            thr_percentile=45,
            morph_close=25,
            morph_open=0,
            min_area=3000,
        )
        mask_dense = apply_density_filter(
            mask_tissue,
            density,
            mode="percentile",
            p=25,
            morph_close=11,
        )
        support = np.full(mask_tissue.shape, 255, dtype=np.uint8)
        return mask_tissue, mask_dense, support

    from my_utils import enhance_hematoxylin_channel, extract_hematoxylin_channel

    rgb = np.asarray(image)[..., :3]
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    # Tissue: anything darker than the glass. The same closing and island floor the
    # DAPI rule uses, so the two sides of a pair are sampled by one geometry.
    _, not_white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    not_white = cv2.morphologyEx(not_white, cv2.MORPH_CLOSE, k, iterations=1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats((not_white > 0).astype(np.uint8), 8)
    mask_tissue = np.zeros_like(not_white)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 3000:
            mask_tissue[lab == i] = 255
    # Density: hematoxylin, blurred, inside the tissue. `threshold` is Stage 1's
    # H-channel cut for this side and marks the nuclei; the density filter keeps the
    # quarter of the tissue richest in them, as it does for DAPI intensity.
    h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(rgb))
    cut = 240 if threshold is None else int(threshold)
    _, dots = cv2.threshold(h_proc, cut, 255, cv2.THRESH_BINARY)
    density = cv2.GaussianBlur(dots, (13, 13), 0)
    mask_dense = apply_density_filter(
        mask_tissue, density, mode="percentile", p=25, morph_close=11,
    )
    support = np.full(mask_tissue.shape, 255, dtype=np.uint8)
    return mask_tissue, mask_dense, support


def rescue_low_signal_dapi_components(
    image, base_dense_mask, tile_size_sampling, config=None,
):
    """Recover locally informative DAPI components rejected by the global mask.

    Fragmented acquisitions can contain FOV islands whose nuclei are clear locally but
    whose absolute intensity is below the whole-canvas percentile threshold.  The
    primary mask remains untouched.  This supplementary path finds sizeable nonzero
    acquisition components, skips any already represented by the primary dense mask,
    requires real local contrast, and applies the same tissue/density percentile logic
    after robust normalization within that component.

    Returns ``(rescue_mask, component_labels, diagnostics)``.  Labels are nonzero only
    on rescued dense pixels and let the sampler guarantee that each rescued acquisition
    component receives the configured minimum number of tiles.
    """
    cfg = dict(LOW_SIGNAL_RESCUE_DEFAULTS)
    cfg.update(config or {})
    shape = np.asarray(base_dense_mask).shape
    empty = np.zeros(shape, dtype=np.uint8)
    empty_labels = np.zeros(shape, dtype=np.int32)
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)),
        "settings": cfg,
        "support_component_count": 0,
        "eligible_component_area_floor_px": None,
        "rescued_component_count": 0,
        "rescued_pixel_count": 0,
        "rescued_components": [],
        "skipped_components": {},
    }
    if not diagnostics["enabled"]:
        return empty, empty_labels, diagnostics

    g8 = normalize_uint16_to_uint8(np.asarray(image))
    support = (g8 > 0).astype(np.uint8)
    close_size = int(cfg["support_close_ksize"])
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    diagnostics["support_component_count"] = int(count - 1)
    area_floor = max(
        int(cfg["min_component_area_px"]),
        int(round(float(cfg["min_component_tile_area_factor"])
                  * float(tile_size_sampling) ** 2)),
    )
    diagnostics["eligible_component_area_floor_px"] = int(area_floor)

    rescue = np.zeros(shape, dtype=np.uint8)
    rescue_labels = np.zeros(shape, dtype=np.int32)
    skipped = {}

    def skip(reason):
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < area_floor:
            skip("below_component_area_floor")
            continue
        component = labels[y:y + height, x:x + width] == component_id
        base_crop = np.asarray(base_dense_mask)[y:y + height, x:x + width]
        existing_fraction = float(np.count_nonzero(base_crop[component]) / area)
        if existing_fraction > float(cfg["max_existing_dense_fraction"]):
            skip("already_represented_by_global_mask")
            continue

        gray_crop = g8[y:y + height, x:x + width]
        values = gray_crop[component]
        low = float(np.percentile(values, float(cfg["local_low_percentile"])))
        high = float(np.percentile(values, float(cfg["local_high_percentile"])))
        contrast = high - low
        if not np.isfinite(contrast) or contrast < float(cfg["min_local_contrast_8bit"]):
            skip("insufficient_local_contrast")
            continue

        local = np.zeros_like(gray_crop, dtype=np.uint8)
        scaled = np.clip(
            (gray_crop.astype(np.float32) - low) * (255.0 / max(contrast, 1e-6)),
            0.0, 255.0,
        ).astype(np.uint8)
        local[component] = scaled[component]
        blur_size = int(cfg["blur_ksize"])
        density = cv2.GaussianBlur(local, (blur_size, blur_size), 0)
        density_values = density[component & (density > 0)]
        if not len(density_values):
            skip("empty_local_density")
            continue
        tissue_threshold = float(np.percentile(
            density_values, float(cfg["local_tissue_percentile"])
        ))
        tissue = (component & (density > int(tissue_threshold))).astype(np.uint8) * 255
        close_size = int(cfg["tissue_close_ksize"])
        if close_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            )
            tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=1)
        tissue[~component] = 0

        local_count, local_labels, local_stats, _ = cv2.connectedComponentsWithStats(
            (tissue > 0).astype(np.uint8), 8
        )
        tissue_floor = max(
            int(cfg["min_tissue_area_px"]),
            int(round(area * float(cfg["min_tissue_component_fraction"]))),
        )
        cleaned = np.zeros_like(tissue)
        for local_id in range(1, local_count):
            if int(local_stats[local_id, cv2.CC_STAT_AREA]) >= tissue_floor:
                cleaned[local_labels == local_id] = 255
        if not np.any(cleaned):
            skip("no_local_tissue_after_area_gate")
            continue

        dense_threshold = float(np.percentile(
            density[cleaned > 0], float(cfg["local_dense_percentile"])
        ))
        dense = ((cleaned > 0) & (density >= dense_threshold)).astype(np.uint8) * 255
        close_size = int(cfg["dense_close_ksize"])
        if close_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            )
            dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, kernel, iterations=1)
        dense[~component] = 0
        rescued_pixels = int(np.count_nonzero(dense))
        if not rescued_pixels:
            skip("empty_local_dense_mask")
            continue

        rescue_id = len(diagnostics["rescued_components"]) + 1
        rescue_crop = rescue[y:y + height, x:x + width]
        np.maximum(rescue_crop, dense, out=rescue_crop)
        label_crop = rescue_labels[y:y + height, x:x + width]
        label_crop[dense > 0] = rescue_id
        diagnostics["rescued_components"].append({
            "rescue_id": int(rescue_id),
            "support_component_id": int(component_id),
            "bbox_xywh": [x, y, width, height],
            "support_area_px": area,
            "existing_dense_fraction": existing_fraction,
            "local_low_8bit": low,
            "local_high_8bit": high,
            "local_contrast_8bit": contrast,
            "local_tissue_threshold_8bit": tissue_threshold,
            "local_dense_threshold_8bit": dense_threshold,
            "rescued_dense_pixels": rescued_pixels,
        })

    diagnostics["skipped_components"] = skipped
    diagnostics["rescued_component_count"] = len(diagnostics["rescued_components"])
    diagnostics["rescued_pixel_count"] = int(np.count_nonzero(rescue))
    return rescue, rescue_labels, diagnostics


def ensure_component_sample_coverage(
    points, available_mask, component_labels, minimum_per_component=1, seed=43,
):
    """Deterministically replace redundant CVT points to cover rescued components."""
    points = np.asarray(points, dtype=np.int32).copy()
    labels = np.asarray(component_labels)
    required = max(0, int(minimum_per_component))
    present_regions = sorted(int(value) for value in np.unique(labels) if value > 0)
    diagnostics = {
        "minimum_per_component": required,
        "eligible_component_ids": present_regions,
        "forced_component_ids": [],
        "sample_count_by_component": {},
    }
    if required == 0 or not present_regions or not len(points):
        return points, diagnostics

    rng = np.random.default_rng(int(seed))
    for region in present_regions:
        for _ in range(required):
            point_regions = labels[points[:, 1], points[:, 0]]
            counts = {int(value): int(np.count_nonzero(point_regions == value))
                      for value in present_regions}
            if counts.get(region, 0) >= required:
                break
            ys, xs = np.where((labels == region) & (available_mask == 0))
            candidates = np.column_stack((xs, ys)).astype(np.int32)
            if not len(candidates):
                break
            if len(candidates) > 50_000:
                candidates = candidates[
                    rng.choice(len(candidates), 50_000, replace=False)
                ]
            tree = cKDTree(points.astype(np.float64))
            distance, _ = tree.query(candidates.astype(np.float64), k=1)
            replacement = candidates[int(np.argmax(distance))]

            if len(points) == 1:
                replace_index = 0
            else:
                nearest = tree.query(points.astype(np.float64), k=2)[0][:, 1]
                protected = np.asarray([
                    value > 0 and counts.get(int(value), 0) <= required
                    for value in point_regions
                ])
                choices = np.where(~protected)[0]
                if not len(choices):
                    break
                replace_index = int(choices[np.argmin(nearest[choices])])
            points[replace_index] = replacement
            diagnostics["forced_component_ids"].append(region)

    point_regions = labels[points[:, 1], points[:, 0]]
    diagnostics["sample_count_by_component"] = {
        str(region): int(np.count_nonzero(point_regions == region))
        for region in present_regions
    }
    return points, diagnostics


def save_moving_tiles(image, tiles, output_folder, rescale_factor, token, modality):
    """Crop the moving tiles: a DAPI side as `<key>_<token>_u8.png`, an H&E side as
    `<key>_<token>.png`, either way with `<token>_tile_info.json` beside them."""
    if modality == "dapi":
        return save_dapi_tiles_intensity(
            image, tiles, output_folder, rescale_factor=rescale_factor,
            prefix="tile", start_index=0, save_u16=False, save_u8_preview=True,
            token=token,
        )
    return save_rgb_tiles(image, tiles, output_folder, rescale_factor=rescale_factor,
                          token=token)


def prune_stale_tile_files(output_folder, token, keep_keys):
    """Remove this side's files for tile keys the writer did not produce this time.

    A re-extraction that samples fewer tiles used to leave the earlier run's extra tiles
    beside the new ones, and their keys in the merged `<token>_tile_info.json`; Stage
    4a lists tiles from the files on disk, so it masked and scored tiles the current
    sampling never chose. Everything named `<key>_<token>.*` or `<key>_<token>_*`
    (the tile, its u8/u16/LUT variants, its mask) goes when `<key>` is not kept.
    """
    keep = set(keep_keys)
    pattern = re.compile(r"^(tile_\d+)_" + re.escape(token) + r"(?:[._]|$)")
    removed = []
    for name in sorted(os.listdir(output_folder)):
        match = pattern.match(name)
        if match and match.group(1) not in keep:
            os.remove(os.path.join(output_folder, name))
            removed.append(name)
    if removed:
        print(f"[Stage3] removed {len(removed)} stale {token} tile file(s) from an earlier "
              f"extraction: {', '.join(removed[:4])}{', ...' if len(removed) > 4 else ''}",
              flush=True)
    return removed


def save_rgb_tiles(rgb, tiles, output_folder, rescale_factor=1.0, prefix="tile",
                   start_index=0, token="he"):
    """Plain RGB crops of a moving H&E side, the counterpart of the DAPI writer."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    h_img, w_img = rgb.shape[:2]
    out_meta = {}
    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)
        x0 = max(0, int(round(x0f * rescale_factor)))
        y0 = max(0, int(round(y0f * rescale_factor)))
        x1 = min(w_img, x0 + int(round(wf * rescale_factor)))
        y1 = min(h_img, y0 + int(round(hf * rescale_factor)))
        if x1 <= x0 or y1 <= y0:
            continue
        key = f"{prefix}_{i:03d}"
        filename = f"{key}_{token}.png"
        cv2.imwrite(os.path.join(output_folder, filename),
                    cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1, :3]),
                                 cv2.COLOR_RGB2BGR))
        out_meta[key] = {
            "x0": x0, "y0": y0, "w": int(x1 - x0), "h": int(y1 - y0),
            "cx": float((x0 + x1) / 2), "cy": float((y0 + y1) / 2),
            "type": "sampled", "id": int(i), "filename": filename,
        }
    json_path = os.path.join(output_folder, f"{token}_tile_info.json")
    with open(json_path, "w") as f:
        json.dump(out_meta, f, indent=4)
    prune_stale_tile_files(output_folder, token, out_meta)
    print(f"Saved {token} RGB tiles in '{output_folder}': {len(out_meta)}")
    return out_meta


def save_dapi_tiles_intensity(
    dapi_gray16,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="tile",
    start_index=0,
    save_u16=False,
    save_u8_preview=True,
    token="dapi",
):
    """
    Crop intensity DAPI tiles from uint16 grayscale image.
    Saves:
      - <key>_dapi_u16.png  (uint16 PNG)   [optional]
      - <key>_dapi_u8.png   (uint8 preview) [optional]
    """
    if dapi_gray16.ndim == 3:
        dapi_gray16 = dapi_gray16[..., 0]
    if dapi_gray16.dtype != np.uint16:
        dapi_gray16 = dapi_gray16.astype(np.uint16)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    h_img, w_img = dapi_gray16.shape[:2]

    out_meta = {}

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        x0 = int(round(x0f * rescale_factor))
        y0 = int(round(y0f * rescale_factor))
        w  = int(round(wf  * rescale_factor))
        h  = int(round(hf  * rescale_factor))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x0 + w)
        y1 = min(h_img, y0 + h)

        if x1 <= x0 or y1 <= y0:
            continue

        tile16 = dapi_gray16[y0:y1, x0:x1]

        key = f"{prefix}_{i:03d}"

        fn_u16 = None
        fn_u8 = None

        if save_u16:
            fn_u16 = f"{key}_{token}_u16.png"
            cv2.imwrite(os.path.join(output_folder, fn_u16), tile16)

        if save_u8_preview:
            fn_u8 = f"{key}_{token}_u8.png"
            tile8 = normalize_uint16_to_uint8(tile16)
            cv2.imwrite(os.path.join(output_folder, fn_u8), tile8)

        out_meta[key] = {
            "x0": x0, "y0": y0,
            "w": int(x1 - x0), "h": int(y1 - y0),
            "cx": float((x0 + x1) / 2), "cy": float((y0 + y1) / 2),
            "type": "sampled",
            "id": int(i),
            # `filename` is the tile a reader should open. The LUT writer overrides it
            # with its own tile when it runs; with LUT tiles off this stays the u8 one,
            # so the metadata has the same shape either way.
            "filename": fn_u8,
            f"filename_{token}_u16": fn_u16,
            f"filename_{token}_u8": fn_u8,
        }

    # Merge into the existing json for the keys extracted this time (the LUT writer's
    # fields, when it ran before us, survive); keys of tiles this extraction did not
    # produce are dropped, and their files with them.
    json_path = os.path.join(output_folder, f"{token}_tile_info.json")
    if os.path.exists(json_path):
        try:
            old = json.load(open(json_path, "r"))
        except Exception:
            old = {}
    else:
        old = {}

    merged = {}
    for k, v in out_meta.items():
        entry = dict(old.get(k, {}))
        entry.update(v)
        merged[k] = entry

    with open(json_path, "w") as f:
        json.dump(merged, f, indent=4)
    prune_stale_tile_files(output_folder, token, merged)

    print(f"Saved DAPI intensity tiles in '{output_folder}': {len(out_meta)}")
    return out_meta

def save_dapi_tiles(
    dapi_rgb,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="tile",          # file/key prefix
    start_index=0,          # in case you want to append
    token="dapi",
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_tiles = []
    output_dict = {}

    h_img, w_img = dapi_rgb.shape[:2]

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        x0 = int(round(x0f * rescale_factor))
        y0 = int(round(y0f * rescale_factor))
        w  = int(round(wf  * rescale_factor))
        h  = int(round(hf  * rescale_factor))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x0 + w)
        y1 = min(h_img, y0 + h)

        # skip invalid (can happen after clamp)
        if x1 <= x0 or y1 <= y0:
            continue

        tile_img = dapi_rgb[y0:y1, x0:x1]

        key = f"{prefix}_{i:03d}"
        filename = f"{key}_{token}.png"
        filepath = os.path.join(output_folder, filename)

        cv2.imwrite(filepath, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))

        info = {
            "x0": x0, "y0": y0,
            "w": x1 - x0, "h": y1 - y0,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
            "type": "sampled",
            "id": i,
            "filename": filename,
            "img": tile_img,
        }
        saved_tiles.append(info)
        output_dict[key] = {k: v for k, v in info.items() if k not in ("img", "img_rf")}

    print(f"Saved DAPI tiles in '{output_folder}': {len(saved_tiles)}")

    # Merge rather than replace: the intensity writer ran first and its filenames are
    # what the headless stages read.
    json_path = os.path.join(output_folder, f"{token}_tile_info.json")
    merged = {}
    if os.path.exists(json_path):
        try:
            merged = json.load(open(json_path, "r"))
        except Exception:
            merged = {}
    for key, value in output_dict.items():
        entry = dict(merged.get(key, {}))
        entry.update(value)
        merged[key] = entry
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=4)

    return saved_tiles

def save_he_tiles(
    he_rgb,
    tiles,
    h_mat,
    output_folder,
    rescale_factor=1.0,
    margin_ratio=0.2,
    prefix="tile",
    start_index=0,
    debug_first_n=0,
    mode="rectified",                 # "rectified" (default) or "bbox"
    rectify_interp=cv2.INTER_LINEAR,
    case_id=0,                        # NEW: must match your DAPI orientation case
    token="he",
):
    """
    Save HE tiles corresponding to DAPI tiles.

    Parameters
    ----------
    tiles : list
        Tile boxes defined in DAPI coordinates.
    h_mat : array-like
        Transform from DAPI coordinates to HE coordinates.
    rescale_factor : float
        Scale factor from HE coordinates to the current HE image level.

    Notes
    -----
    Each tile is either cropped by bbox or rectified into a canonical tile.
    Metadata are written to he_tile_info.json.
    """
    if mode not in ("rectified", "bbox"):
        raise ValueError(f"mode must be 'rectified' or 'bbox', got {mode}")
    os.makedirs(output_folder, exist_ok=True)

    H = np.asarray(h_mat, dtype=float)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"h_mat must be 3x3 homography (or 2x3 affine), got {H.shape}")

    h_img, w_img = he_rgb.shape[:2]
    rf = float(rescale_factor)

    def orient_quad_indices(case_id: int):
        """
        Return indices (len=4) to reorder [TL,TR,BR,BL] corners to match the same
        orientation you apply to DAPI tiles (apply_orientation_case / apply_orientation_to_tile).
        """
        pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=int)  # TL,TR,BR,BL in a 2x2 grid

        def apply_case_xy(x, y):
            if case_id == 0:   # identity
                return x, y
            if case_id == 1:   # rot90 CW
                return y, 1 - x
            if case_id == 2:   # rot180
                return 1 - x, 1 - y
            if case_id == 3:   # rot90 CCW
                return 1 - y, x
            if case_id == 4:   # flip UD (vertical)
                return x, 1 - y
            if case_id == 5:   # flip LR (horizontal)
                return 1 - x, y
            if case_id == 6:   # transpose
                return y, x
            if case_id == 7:   # anti-transpose
                return 1 - y, 1 - x
            raise ValueError(f"Unknown case_id={case_id}")

        pts2 = np.array([apply_case_xy(x, y) for x, y in pts], dtype=int)

        # identify TL/TR/BR/BL in the oriented grid
        s = pts2[:, 0] + pts2[:, 1]
        d = pts2[:, 0] - pts2[:, 1]
        tl = int(np.argmin(s))
        br = int(np.argmax(s))
        tr = int(np.argmax(d))
        bl = int(np.argmin(d))
        return np.array([tl, tr, br, bl], dtype=int)

    he_tiles = []
    output_dict = {}

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        # A) Original DAPI tile corners
        x0 = float(x0f)
        y0 = float(y0f)
        x1 = x0 + float(wf)
        y1 = y0 + float(hf)

        corners_dapi_tile = np.array(
            [[x0, y0],
             [x1, y0],
             [x1, y1],
             [x0, y1]], dtype=float
        )  # TL,TR,BR,BL

        # B) DAPI projection bbox (expanded for HE rectification), margin_ratio=0.2 means 1.2x larger
        expand = 1.0 + float(margin_ratio)  # e.g. 1.2
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = (float(wf) / 2.0) * expand
        half_h = (float(hf) / 2.0) * expand

        x0p = cx - half_w
        x1p = cx + half_w
        y0p = cy - half_h
        y1p = cy + half_h

        corners_dapi_proj = np.array(
            [[x0p, y0p],
             [x1p, y0p],
             [x1p, y1p],
             [x0p, y1p]], dtype=float
        )  # TL,TR,BR,BL for projection

        # ---- project to HE coords using homography ----
        corners_h = np.hstack([corners_dapi_proj, np.ones((4, 1), dtype=float)])  # (4,3)
        proj = (H @ corners_h.T).T  # (4,3)
        w = proj[:, 2:3]
        eps = 1e-9
        w_safe = np.where(np.abs(w) < eps, np.sign(w) * eps + (w == 0) * eps, w)
        corners_he = proj[:, :2] / w_safe  # (4,2) HE coords (pre-level adjust)

        # ---- convert to he_rgb pixel coords (level adjust) ----
        corners_he_px_raw = corners_he * rf  # same role as your original code

        # ---- bbox in he_rgb pixels ----
        xs = corners_he_px_raw[:, 0]
        ys = corners_he_px_raw[:, 1]
        min_x = int(np.floor(xs.min()))
        max_x = int(np.ceil(xs.max()))
        min_y = int(np.floor(ys.min()))
        max_y = int(np.ceil(ys.max()))

        # clamp bbox to image
        min_x_cl = max(0, min_x)
        min_y_cl = max(0, min_y)
        max_x_cl = min(w_img, max_x)
        max_y_cl = min(h_img, max_y)

        key = f"{prefix}_{i:03d}"

        if debug_first_n and (i - start_index) < debug_first_n:
            print(f"[DEBUG] {key} mode={mode}")
            print(" corners_dapi:\n", corners_dapi_proj)
            print(" corners_he (pre-rescale):\n", corners_he)
            print(" corners_he_px_raw (he_rgb px):\n", corners_he_px_raw)
            print(" bbox unclamped:", (min_x, min_y, max_x-min_x, max_y-min_y))
            print(" bbox clamped  :", (min_x_cl, min_y_cl, max_x_cl-min_x_cl, max_y_cl-min_y_cl))
            print(" he_rgb shape  :", he_rgb.shape)

        if max_x_cl <= min_x_cl or max_y_cl <= min_y_cl:
            continue

        # ----------------------------
        # Generate tile_img
        # ----------------------------
        M = None
        Minv = None

        if mode == "bbox":
            tile_img = he_rgb[min_y_cl:max_y_cl, min_x_cl:max_x_cl]
            out_w = int(max_x_cl - min_x_cl)
            out_h = int(max_y_cl - min_y_cl)

        else:  # rectified
            # src corners are already in DAPI order: TL,TR,BR,BL (after H + rf)
            src = corners_he_px_raw.astype(np.float32)

            def dist(a, b):
                return float(np.linalg.norm(a - b))
            width  = 0.5 * (dist(src[0], src[1]) + dist(src[3], src[2]))  # top & bottom
            height = 0.5 * (dist(src[1], src[2]) + dist(src[0], src[3]))  # right & left
            out_w = max(2, int(round(width)))
            out_h = max(2, int(round(height)))

            # --- canonical dst in TL,TR,BR,BL ---
            dst = np.array(
                [[0.0, 0.0],                 # TL
                 [out_w - 1.0, 0.0],          # TR
                 [out_w - 1.0, out_h - 1.0],  # BR
                 [0.0, out_h - 1.0]],         # BL
                dtype=np.float32
            )

            # --- apply SAME orientation convention as DAPI tile ---
            idx = orient_quad_indices(int(case_id))
            dst = dst[idx]

            # A mirrored Stage-2 transform gives `src` the opposite winding to `dst`,
            # and that is the mirror itself, not a defect: getPerspectiveTransform is
            # happy to return an orientation-reversing homography, and doing so is what
            # carries the flip into the rectified tile. Re-matching the winding by
            # swapping src's TR and BL leaves TL and BR fixed, so it reflects the tile
            # about its own diagonal -- silently undoing the flip and handing Stage 4b a
            # transposed H&E tile that its scale-and-shift search can never align. On
            # `cosmx_100_percent`, whose DAPI and H&E are mirror images, 127 of 140 tiles
            # matched their DAPI best under a transpose.

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = np.linalg.inv(M)

            # OpenCV remap rejects source dimensions >= SHRT_MAX. Rectifying
            # only the projected tile neighborhood also avoids processing the
            # complete H&E slide once for every saved tile.
            padding = 2
            crop_x0 = max(0, min_x_cl - padding)
            crop_y0 = max(0, min_y_cl - padding)
            crop_x1 = min(w_img, max_x_cl + padding)
            crop_y1 = min(h_img, max_y_cl + padding)
            he_local = he_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
            local_to_he = np.array([
                [1.0, 0.0, float(crop_x0)],
                [0.0, 1.0, float(crop_y0)],
                [0.0, 0.0, 1.0],
            ])

            tile_img = cv2.warpPerspective(
                he_local, M @ local_to_he, (out_w, out_h),
                flags=rectify_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        if tile_img.ndim == 2:
            # A DAPI fixed side: a 16-bit plane, written the way the moving DAPI tiles
            # are, so the same masker reads both.
            filename = f"{key}_{token}_u8.png"
            cv2.imwrite(os.path.join(output_folder, filename),
                        normalize_uint16_to_uint8(tile_img))
        else:
            filename = f"{key}_{token}.png"
            cv2.imwrite(
                os.path.join(output_folder, filename),
                cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR),
            )

        info = {
            "x0": int(min_x_cl),
            "y0": int(min_y_cl),
            "w": int(max_x_cl - min_x_cl),
            "h": int(max_y_cl - min_y_cl),
            "cx": float((min_x_cl + max_x_cl) / 2.0),
            "cy": float((min_y_cl + max_y_cl) / 2.0),
            "type": "sampled",
            "id": int(i),
            "filename": filename,
            "img": tile_img,
        }

        meta = {
            "mode": mode,
            "rescale_factor": float(rescale_factor),
            "margin_ratio": float(margin_ratio),
            "case_id": int(case_id),
            "dapi_corners_tile": corners_dapi_tile.tolist(),
            "dapi_corners_proj": corners_dapi_proj.tolist(),
            "proj_expand": float(1.0 + margin_ratio),
            "he_quad_px_raw": corners_he_px_raw.tolist(),     # TL,TR,BR,BL (DAPI order) in he_rgb coords
            "rectified_wh": [int(out_w), int(out_h)],
            "M_he_to_rect": None if M is None else M.tolist(),
            "M_rect_to_he": None if Minv is None else Minv.tolist(),
        }
        he_tiles.append({**info, "meta": meta, "img": tile_img})

        # JSON version (no image array)
        output_dict[key] = {k: v for k, v in info.items() if k != "img"}
        output_dict[key]["meta"] = meta

    print(f"Saved H&E tiles in '{output_folder}' (mode={mode}): {len(he_tiles)}")

    with open(os.path.join(output_folder, f"{token}_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
    prune_stale_tile_files(output_folder, token, output_dict)

    return he_tiles

# -----------------------------
# GUI classes
# -----------------------------


# -----------------------------
# The Stage-3 computation, without a window
# -----------------------------
class NullProgress:
    """A progress sink that discards every message (the CLI's stand-in for the Tk
    progress dialog's queue)."""
    def put(self, *args, **kwargs):
        pass


def sync_run_with_progress(self, title, worker_fn):
    """Run a progress-reporting worker to completion with no dialog: what a windowless
    caller (or a headless test of the window) uses in place of `_run_with_progress`."""
    print(f"[Stage3-CLI] {title}", flush=True)
    worker_fn(NullProgress())


class Stage3Sampler:
    """Stage 3 with no window: the state, the sampling and the extraction.

    `3_get_tiles.py`'s Tk window inherits this and adds panels, buttons and a progress
    dialog; `cli_stage3.py` uses it as it is. The two entry points therefore run the
    same code on the same state, which is what made them agree before the split, and
    the CLI process no longer imports a GUI toolkit to do it.
    """

    def __init__(self, run_dir: Path):
        self.initialize_sampling_state(run_dir)

    def initialize_sampling_state(self, run_dir: Path):
        """Plain computational state shared by the Tk and windowless entry points."""
        self.run_dir = Path(run_dir)
        self.roles = run_roles.read(self.run_dir)
        self.stage3 = load_stage3_params(Path(__file__), self.roles.params_block)
        self.has_sampling_outputs = False
        self.sampling_counter = 0
        self.current_tiles = None
        self.current_points_xy = None
        # Stage 2's H_mat maps original-image coordinates. Orientation is display-only.
        self.case_id = 0
        self.display_case_id = int(self.roles.moving.orientation_case)


    # --------------------------
    # sampling
    # --------------------------
    def _run_sampling_core(self):
        timer = StepTimer()

        self.sampling_counter += 1
        seed = int(self.stage3["random_seed"])
        print(f"[INFO] Sampling seed = {seed}", flush=True)

        info_path = self.run_dir / "images_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"missing {info_path}")

        with open(info_path, "r") as f:
            info = json.load(f)

        roles = run_roles.read(info)
        moving = roles.moving
        dapi_path = moving.path
        dapi_level = int(moving.level)
        print(f"[INFO] pair={roles.pair.name} moving={moving.token} ({moving.modality})",
              flush=True)
        print(f"[INFO] MOVING_PATH={dapi_path}", flush=True)
        print(f"[INFO] MOVING_LEVEL={dapi_level}", flush=True)

        dapi16 = read_side_image(
            dapi_path, dapi_level, moving.modality,
            channel_index=moving.channel_index,
        )
        timer.mark("Read moving image")

        dbg_gray_path = self.run_dir / f"3_dbg_{moving.token}_gray.png"
        if qc_images_enabled("stage3_debug"):
            cv2.imwrite(str(dbg_gray_path), to_gray_uint8(dapi16))
        timer.mark("Save gray preview")

        # Tile resolution and footprint are set in physical units, not pixels. The
        # pyramid level is the one whose micrometres per pixel is closest to
        # `target_tile_mpp`, so a tile means the same thing on both sides of the pair and
        # across platforms; the footprint is a fixed number of nuclear diameters.
        target_tile_mpp = float(self.stage3.get("target_tile_mpp", 0.425))
        nucleus_um = float(self.stage3.get("nucleus_diameter_um", 6.0))
        nuclei_across = float(self.stage3.get("tile_nuclei_across", 30.0))
        tile_footprint_um = nucleus_um * nuclei_across

        from my_utils import inspect_image_pyramid
        dapi_pyramid = inspect_image_pyramid(dapi_path)
        # Use the same Stage-2-based recovery as Stages 4-8. In particular, Case 4
        # must anchor the coarser side once and derive the other side; independently
        # assuming both images have the same pitch discards the relative scale that
        # Stage 2 already measured.
        stage3_physical_scale = None
        try:
            from physical_scale import run_factors
            stage3_physical_scale = run_factors(info, run_dir=self.run_dir)
        except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
            print(
                f"[WARN] Stage-3 physical-scale recovery unavailable ({exc}); "
                "using the legacy per-image fallback",
                flush=True,
            )
        self.stage3_physical_scale = stage3_physical_scale

        dapi_base_mpp = scalar_mpp(dapi_pyramid.get("base_mpp"))
        dapi_n_levels = len(dapi_pyramid.get("level_shapes") or []) or (dapi_level + 1)
        assumed_base_mpp = float(self.stage3.get("assumed_base_mpp_40x", 0.25))
        if dapi_base_mpp is None:
            recovered = (stage3_physical_scale or {}).get("mpp_by_role", {}).get("moving")
            recovered_source = (stage3_physical_scale or {}).get(
                "mpp_source_by_role", {}).get("moving")
            if recovered is not None:
                dapi_base_mpp = float(recovered)
                dapi_mpp_assumed = recovered_source != "ome_header"
                print(
                    f"[INFO] DAPI has no OME MPP; resolved {dapi_base_mpp:.4f} "
                    f"um/px from {recovered_source} for Stage 3.",
                    flush=True,
                )
            else:
                print(f"[WARN] DAPI has no physical pixel size in its OME metadata; assuming "
                      f"40x ({assumed_base_mpp:.4f} um/px) at level 0. Set "
                      f"stage3.moving_level_override if that is wrong for this slide.",
                      flush=True)
                dapi_base_mpp = assumed_base_mpp
                dapi_mpp_assumed = True
        else:
            dapi_mpp_assumed = False

        target_tile_size_at_extract = int(self.stage3.get("tile_size", self.stage3.get("fallback_tile_size", 400)))
        from role_parameters import role_parameter
        moving_override = role_parameter(self.stage3, "moving_level_override", "dapi_level_override")
        if moving_override not in [None, "None"]:
            dapi_tile_level = int(moving_override)
        else:
            dapi_tile_level = choose_level_by_target_mpp(
                dapi_base_mpp, dapi_n_levels, target_tile_mpp,
                min_level=0, max_level=dapi_level,
            )
            if dapi_tile_level is None:
                # Only reachable now if the pyramid itself is degenerate: with the 40x
                # assumption above, a missing OME spacing no longer lands here.
                print("[WARN] DAPI physical pixel size unusable; falling back to the "
                      "level-offset rule", flush=True)
                dapi_tile_level = choose_tile_level(
                    x=dapi_level,
                    k=target_tile_size_at_extract,
                    min_tile_level=1,
                    max_tile_level=dapi_level,
                )
        dapi_base_mpp = scalar_mpp(dapi_base_mpp)
        dapi_tile_mpp = (dapi_base_mpp * 2 ** dapi_tile_level) if dapi_base_mpp else None
        self.dapi_tile_mpp = dapi_tile_mpp
        # The H&E rule falls back to this when the H&E slide has no spacing of its own.
        self.dapi_base_mpp = dapi_base_mpp
        self.dapi_base_mpp_assumed = bool(dapi_mpp_assumed)
        self.tile_footprint_um = tile_footprint_um
        self.target_tile_mpp = target_tile_mpp
        print(f"[INFO] DAPI_EXTRACT_LEVEL={dapi_tile_level}"
              + (f" ({dapi_tile_mpp:.4f} um/px, target {target_tile_mpp:.4f})"
                 if dapi_tile_mpp else " (physical pixel size unknown)"), flush=True)

        if dapi_tile_mpp:
            derived = tile_size_px_for_footprint(tile_footprint_um, dapi_tile_mpp)
            print(f"[INFO] tile footprint {tile_footprint_um:.0f} um "
                  f"({nuclei_across:g} x {nucleus_um:g} um nuclei) -> {derived} px "
                  f"at {dapi_tile_mpp:.4f} um/px", flush=True)
            self.stage3["tile_size"] = float(derived)
            self.current_tile_size_at_extract = float(derived)
        else:
            print(f"[WARN] keeping the configured tile_size="
                  f"{self.stage3['tile_size']} px; footprint cannot be derived", flush=True)

        tile_size_at_extract = float(self.stage3["tile_size"])
        self.current_tile_size_at_extract = tile_size_at_extract
        n_tiles = int(self.stage3["n_tiles"])   # provisional; re-derived from the mask below
        min_dist_fac = float(self.stage3["min_dist_factor"])

        scale_extract_to_sampling = 2 ** (dapi_level - dapi_tile_level)
        tile_size_sampling = tile_size_at_extract / scale_extract_to_sampling
        min_dist = tile_size_sampling * min_dist_fac

        buffer_at_extract = float(self.stage3.get("buffer_px_at_extract", 0))
        buffer = buffer_at_extract / scale_extract_to_sampling

        print(f"[INFO] tile_size_at_extract={tile_size_at_extract}", flush=True)
        print(f"[INFO] TILE_SIZE@sampling={tile_size_sampling}", flush=True)
        print(f"[INFO] MIN_DIST@sampling={min_dist}", flush=True)

        mask_tissue, mask_dense, support_mask = moving_tissue_masks(
            dapi16, moving.modality, moving.threshold,
        )
        global_dense_pixels = int(np.count_nonzero(mask_dense))
        rescue_mask = np.zeros_like(mask_dense, dtype=np.uint8)
        rescue_component_labels = np.zeros_like(mask_dense, dtype=np.int32)
        rescue_diagnostics = {
            "enabled": False,
            "rescued_component_count": 0,
            "rescued_pixel_count": 0,
            "rescued_components": [],
        }

        # The zero-value/support bug is DAPI-specific. Keep H&E moving pairs on the
        # 4.11 mask, erosion and tile-budget path so he-he/he-dapi results remain
        # directly comparable; only a moving DAPI uses the separated geometric-support
        # and density constraints introduced in 4.12.
        seam_safe_radius = int(round(float(self.stage3.get("seam_safe_radius", 3))))
        mask_dense = mask_dense.astype(np.uint8)
        support_mask = support_mask.astype(np.uint8)
        if moving.modality == "dapi":
            rescue_cfg = self.stage3.get("low_signal_component_rescue", {})
            rescue_mask, rescue_component_labels, rescue_diagnostics = (
                rescue_low_signal_dapi_components(
                    dapi16, mask_dense, tile_size_sampling, rescue_cfg,
                )
            )
            if rescue_diagnostics["rescued_component_count"]:
                mask_dense = cv2.bitwise_or(mask_dense, rescue_mask)
            print(
                "[INFO] Stage-3 low-signal rescue: "
                f"enabled={rescue_diagnostics['enabled']} "
                f"components={rescue_diagnostics['rescued_component_count']} "
                f"pixels={rescue_diagnostics['rescued_pixel_count']}",
                flush=True,
            )
            valid_mask = cv2.bitwise_and(mask_dense, support_mask)

            # A complete DAPI tile must fit inside geometric support, so support is
            # contracted by half a tile plus the seam margin. Density only determines
            # where a useful centre may land and keeps the old small 10%-of-tile margin.
            tile_half_radius = int(np.ceil(tile_size_sampling / 2 + buffer))
            density_safe_radius = min(
                tile_half_radius,
                int(tile_size_sampling * 0.10),
            )
            support_safe_radius = tile_half_radius + max(0, seam_safe_radius)
            print(f"[INFO] DAPI density_safe_radius@sampling={density_safe_radius}", flush=True)
            print(f"[INFO] DAPI support_safe_radius@sampling={support_safe_radius}", flush=True)

            density_unavailable = make_available_mask_by_eroding_valid_region(
                mask_dense,
                radius=density_safe_radius,
                outside_is_invalid=True,
            )
            support_unavailable = make_available_mask_by_eroding_valid_region(
                support_mask,
                radius=support_safe_radius,
                outside_is_invalid=True,
            )
            # Availability uses 0=yes and 255=no. OR keeps a centre only when both
            # density and geometric support accept it.
            avail = cv2.bitwise_or(density_unavailable, support_unavailable)
            radius_summary = (
                f"density_safe_radius={density_safe_radius}, "
                f"support_safe_radius={support_safe_radius}"
            )
            budget_pixels = None  # filled from the final availability mask below
            budget_label = "sampleable centres"
        else:
            # Exact 4.11 H&E path: full nonvoid/support mask, its legacy seam erosion,
            # the small erosion of the combined dense mask, and dense-area budgeting.
            if seam_safe_radius > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * seam_safe_radius + 1, 2 * seam_safe_radius + 1),
                )
                support_mask = cv2.erode(support_mask, k, iterations=1)
            valid_mask = cv2.bitwise_and(mask_dense, support_mask)
            safe_radius = int(np.ceil(tile_size_sampling / 2 + buffer))
            safe_radius = min(safe_radius, int(tile_size_sampling * 0.10))
            print(f"[INFO] H&E legacy safe_radius@sampling={safe_radius}", flush=True)
            avail = make_available_mask_by_eroding_valid_region(
                valid_mask,
                radius=safe_radius,
                outside_is_invalid=False,
            )
            radius_summary = f"safe_radius={safe_radius}"
            budget_pixels = int(np.count_nonzero(mask_dense > 0))
            budget_label = "tissue"

        # dbg_support_path = self.run_dir / "3_dbg_support_mask.png"
        # cv2.imwrite(str(dbg_support_path), support_mask)
        dbg_density_path = self.run_dir / "3_dbg_density_mask.png"
        if qc_images_enabled("stage3_debug"):
            cv2.imwrite(str(dbg_density_path), valid_mask)
            if rescue_diagnostics.get("enabled"):
                cv2.imwrite(
                    str(self.run_dir / "3_dbg_low_signal_rescue_mask.png"),
                    rescue_mask,
                )

        print(f"[INFO] mask_tissue > 0: {np.count_nonzero(mask_tissue > 0)}", flush=True)
        print(f"[INFO] mask_dense  > 0: {np.count_nonzero(mask_dense > 0)}", flush=True)
        print(f"[INFO] support_mask > 0: {np.count_nonzero(support_mask > 0)}", flush=True)
        print(f"[INFO] valid_mask > 0: {np.count_nonzero(valid_mask > 0)}", flush=True)
        timer.mark("Make tissue+density mask")

        available_pixels = int(np.count_nonzero(avail == 0))
        if budget_pixels is None:
            budget_pixels = available_pixels
        print(f"[INFO] avail == 0: {available_pixels}", flush=True)

        # Only the file is optional; the GUI always displays the current in-memory array.
        avail_flipped = 255 - avail
        avail_path = self.run_dir / "3_dbg_available_after_erode.png"
        if qc_images_enabled("stage3_debug"):
            cv2.imwrite(str(avail_path), avail_flipped)
        timer.mark("Make available mask (eroded interior)")

        if available_pixels == 0:
            raise ValueError(
                "Stage 3 sampling mask is empty after erosion: "
                f"tissue={int(np.count_nonzero(mask_tissue > 0))}, "
                f"dense={int(np.count_nonzero(mask_dense > 0))}, "
                f"support={int(np.count_nonzero(support_mask > 0))}, "
                f"valid={int(np.count_nonzero(valid_mask > 0))}, "
                f"{radius_summary}. Check the tissue/density mask and geometric support."
            )

        # DAPI follows the final sampleable centre area. H&E deliberately retains 4.11's
        # pre-erosion dense-tissue area so the DAPI correction cannot alter he-he runs.
        sampling_mpp = dapi_tile_mpp * scale_extract_to_sampling if dapi_tile_mpp else None
        if sampling_mpp and budget_pixels:
            budget_area_um2 = budget_pixels * (sampling_mpp ** 2)
            tile_area_um2 = float(tile_footprint_um) ** 2
            sample_fraction = float(self.stage3.get("tissue_sample_fraction", 0.5))
            n_min = int(self.stage3.get("n_tiles_min", 100))
            n_max = int(self.stage3.get("n_tiles_max", 200))
            raw = budget_area_um2 * sample_fraction / tile_area_um2
            n_tiles = max(n_min, min(n_max, int(round(raw))))
            self.tissue_area_mm2 = budget_area_um2 / 1e6
            self.n_tiles_raw = float(raw)
            print(f"[INFO] {budget_label} {budget_area_um2/1e6:.2f} mm2 "
                  f"x {sample_fraction:g} / "
                  f"{tile_area_um2/1e6:.5f} mm2 per tile = {raw:.0f} -> "
                  f"n_tiles={n_tiles} (clamped to [{n_min}, {n_max}])", flush=True)
        else:
            print(f"[WARN] tissue area unavailable; keeping n_tiles={n_tiles}", flush=True)
        self.n_tiles_used = int(n_tiles)

        stage4_fraction = float(self.stage3.get("stage4_tile_fraction", 0.30))
        stage4_tiles = max(1, int(round(n_tiles * stage4_fraction)))
        self.stage4_tiles = stage4_tiles
        print(f"[INFO] TILE PLAN  level={dapi_tile_level}  "
              + (f"mpp={dapi_tile_mpp:.4f} um/px  " if dapi_tile_mpp else "mpp=unknown  ")
              + f"size={int(tile_size_at_extract)} px = {tile_footprint_um:.0f} um  "
              f"n_tiles={n_tiles}  stage4={stage4_tiles} ({stage4_fraction:.0%})", flush=True)

        points_xy = cvt_masked(
            avail,
            N_POINTS=n_tiles,
            MIN_DIST=min_dist,
            ITERATIONS=50,
            seed=seed,
        )
        coverage_diagnostics = {
            "minimum_per_component": 0,
            "eligible_component_ids": [],
            "forced_component_ids": [],
            "sample_count_by_component": {},
        }
        if moving.modality == "dapi" and rescue_diagnostics.get("enabled"):
            points_xy, coverage_diagnostics = ensure_component_sample_coverage(
                points_xy,
                avail,
                rescue_component_labels,
                minimum_per_component=int(
                    self.stage3["low_signal_component_rescue"]
                    ["minimum_sampled_tiles_per_component"]
                ),
                seed=seed,
            )
        timer.mark("CVT sampling")

        tiles = centroids_to_tiles(points_xy, tile_size=tile_size_sampling)

        self.current_dapi_tile_level = dapi_tile_level
        self.current_points_xy = points_xy
        self.current_tiles = tiles
        self.current_tile_size = tile_size_sampling
        self.current_dapi_level = dapi_level

        mask_diagnostics = {
            "pair": roles.pair.name,
            "moving_token": moving.token,
            "moving_modality": moving.modality,
            "sampling_level": int(dapi_level),
            "tile_level": int(dapi_tile_level),
            "tile_size_sampling_px": float(tile_size_sampling),
            "global_dense_pixel_count": global_dense_pixels,
            "combined_dense_pixel_count": int(np.count_nonzero(mask_dense)),
            "available_pixel_count": available_pixels,
            "low_signal_component_rescue": rescue_diagnostics,
            "rescued_component_sampling": coverage_diagnostics,
        }
        mask_diagnostics_path = self.run_dir / "stage3_mask_diagnostics.json"
        with open(mask_diagnostics_path, "w") as f:
            json.dump(mask_diagnostics, f, indent=2)

        sampled_points_json = self.run_dir / "sampled_points.json"
        with open(sampled_points_json, "w") as f:
            json.dump(
                {
                    "pair": roles.pair.name,
                    f"{moving.token}_level": dapi_level,
                    f"{moving.token}_tile_level": dapi_tile_level,
                    "dapi_level": dapi_level,
                    "dapi_tile_level": dapi_tile_level,
                    "random_seed": seed,
                    "tile_size_sampling": tile_size_sampling,
                    "stage3_mask_diagnostics": str(mask_diagnostics_path),
                    "low_signal_rescued_components": int(
                        rescue_diagnostics.get("rescued_component_count", 0)
                    ),
                    "points_xy": points_xy.tolist(),
                },
                f,
                indent=2,
            )
        timer.mark("Save points json")

        overlay_path = self.run_dir / "3_sampled_overlay.png"
        overlay_bgr = draw_points_overlay(
            dapi16,
            points_xy,
            tile_size=tile_size_sampling,
            save_path=str(overlay_path) if qc_images_enabled("stage3_debug") else None,
        )
        timer.mark("Save overlay")

        return {
            "points_xy": points_xy,
            "tiles": tiles,
            "dapi_level": dapi_level,
            "dapi_tile_level": dapi_tile_level,
            "random_seed": seed,
            "tile_size_sampling": tile_size_sampling,
            "sampled_points_json": sampled_points_json,
            "mask_diagnostics_path": mask_diagnostics_path,
            "avail_path": avail_path,
            "overlay_path": overlay_path,
            "avail_flipped": avail_flipped,
            "overlay_bgr": overlay_bgr,
        }

    run_sampling = _run_sampling_core

    def sample(self):
        """Sample the tile centroids and record it as the button does, without a window."""
        log_event(self.run_dir, "user_click_sample_centroids")
        result = self._run_sampling_core()
        self.has_sampling_outputs = True
        print("[DONE] sampling finished.", flush=True)
        log_event(
            self.run_dir,
            "system_ready_sample_centroids",
            n_points=int(len(result["points_xy"])),
            dapi_level=int(result["dapi_level"]),
            dapi_tile_level=int(result["dapi_tile_level"]),
            random_seed=int(result["random_seed"]),
            tile_size_sampling=float(result["tile_size_sampling"]),
            sampled_points_json=str(result["sampled_points_json"]),
            overlay_path=str(result["overlay_path"]),
        )
        return result

    on_sampling_clicked = sample

    # --------------------------
    # extraction
    # --------------------------
    def extract(self, progress=None):
        """Extract the current tiles and record it as the button does, without a window."""
        log_event(self.run_dir, "user_click_save_tiles")
        if not self.has_sampling_outputs:
            raise RuntimeError("Please run sampling first.")
        self.extract_tiles(progress if progress is not None else NullProgress())

    on_extract_clicked = extract

    def extract_tiles(self, q):
        """Crop the moving tiles and rectify the fixed ones into <RUN_DIR>/tiles/.

        `q` receives ("stage", (i, total, name)), ("progress", percent) and
        ("log", text) messages; the Tk window feeds them to its progress dialog and
        the CLI discards them.
        """
        timer0 = time.perf_counter()

        STAGE_TOTAL = 3
        def report_stage(i, name):
            q.put(("stage", (i, STAGE_TOTAL, name)))
        def tick(p, msg=None):
            q.put(("progress", p))
            if msg:
                q.put(("log", msg))

        # ---------- read inputs ----------
        report_stage(1, "Loading configs / inputs")
        info_path = self.run_dir / "images_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"missing {info_path}")
        with open(info_path, "r") as f:
            info = json.load(f)
        script_dir = Path(__file__).resolve().parent
        from my_utils import cli_params_block
        roles = run_roles.read(info)
        parameters = cli_params_block(roles.params_block)
        moving, fixed = roles.moving, roles.fixed
        DAPI_PATH = moving.path
        HE_PATH = fixed.path
        DAPI_LEVEL = int(moving.level)
        HE_LEVEL = int(fixed.level)
        from role_parameters import role_parameter
        DAPI_TILE_LEVEL_OVERRIDE = role_parameter(parameters["stage3"], "moving_level_override", "dapi_level_override")
        HE_TILE_LEVEL_OVERRIDE = role_parameter(parameters["stage3"], "fixed_level_override", "he_level_override")
        HE_TILE_LEVEL_OVERRIDE = self.__dict__.get("extraction_fixed_level", HE_TILE_LEVEL_OVERRIDE)

        lut_path = script_dir / "glasbey_inverted.lut"
        lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

        # ---------- read LUT threshold from images_info.json ----------
        dapi_lut_threshold = load_dapi_lut_threshold_from_images_info(self.run_dir, default=1000)
        q.put(("log", f"[PARAM] images_info.DAPI_LUT_threshold = {dapi_lut_threshold}"))

        sampled_path = self.run_dir / "sampled_points.json"
        if not sampled_path.exists():
            raise FileNotFoundError(f"missing {sampled_path}, run sampling first.")
        sampled = json.loads(sampled_path.read_text())
        points_xy = np.asarray(sampled["points_xy"], dtype=np.int32)

        # use tiles generated from latest sampling
        if not hasattr(self, "current_tiles") or self.current_tiles is None:
            raise RuntimeError("No tiles found. Please click 'Sample tile centroid' first.")
        tiles = self.current_tiles
        output_folder = self.run_dir / "tiles"
        ensure_dir(output_folder)

        tick(5, f"DAPI_LEVEL={DAPI_LEVEL}, HE_LEVEL={HE_LEVEL}")
        tick(6, f"DAPI_TILE_LEVEL_OVERRIDE={DAPI_TILE_LEVEL_OVERRIDE}, HE_TILE_LEVEL_OVERRIDE={HE_TILE_LEVEL_OVERRIDE}")
        tick(8, f"Output: {output_folder}")

        # ---------- imports from your project ----------
        report_stage(1, "Importing project utils")
        from my_utils import read_image, dapi_to_lut_rgb
        tick(12)

        # ------------------------------
        # Prepare alignment and HE image for tile pruning
        # ------------------------------
        report_stage(2, "Loading H&E / pruning out-of-H&E tiles")

        path_initial = initial_alignment_path(self.run_dir)
        if path_initial.exists():
            data = json.loads(path_initial.read_text())
            q.put(("log", f"[INFO] Using alignment from: {path_initial.name}"))
        else:
            raise FileNotFoundError(path_initial)

        h_mat = data["H_mat"]
        tick(20, "Loaded initial alignment")

        DAPI_EXTRACT_LEVEL = getattr(self, "current_dapi_tile_level", None)
        if DAPI_EXTRACT_LEVEL is None:
            DAPI_EXTRACT_LEVEL = 1

        if HE_TILE_LEVEL_OVERRIDE in [None, "None"]:
            # Match the H&E tile resolution to the DAPI tile resolution in
            # micrometres rather than in pyramid levels. The safety loop below still
            # downshifts if a projected tile would exceed the size cap.
            he_start_level = None
            he_target_mpp = float(getattr(self, "target_tile_mpp",
                                          parameters["stage3"].get("target_tile_mpp", 0.425)))
            he_assumed = float(parameters["stage3"].get("assumed_base_mpp_40x", 0.25))
            try:
                from my_utils import inspect_image_pyramid as _inspect
                he_pyramid = _inspect(HE_PATH)
                he_base_mpp = scalar_mpp(he_pyramid.get("base_mpp"))
                he_mpp_source = "ome"
                if he_base_mpp is None:
                    recovered = (getattr(self, "stage3_physical_scale", None) or {}).get(
                        "mpp_by_role", {}).get("fixed")
                    recovered_source = (getattr(
                        self, "stage3_physical_scale", None) or {}).get(
                            "mpp_source_by_role", {}).get("fixed")
                    if recovered is not None:
                        he_base_mpp = float(recovered)
                        he_mpp_source = str(recovered_source or "Stage-2 recovery")
                        print(
                            f"[INFO] H&E has no OME MPP; resolved {he_base_mpp:.4f} "
                            f"um/px from {he_mpp_source} for Stage 3.",
                            flush=True,
                        )

                    # Derive it rather than assume it. The Stage-2 homography maps
                    # the *Stage-1 level* images onto each other -- DAPI level
                    # DAPI_LEVEL to H&E level HE_LEVEL, not level 0 to level 0 -- so
                    # both level factors have to come back out:
                    #
                    #   he_mpp0 = dapi_mpp0 * 2**DAPI_LEVEL / scale / 2**HE_LEVEL
                    #
                    # On mouse_brain -- the one slide in the cohort with no H&E
                    # spacing -- this returns 0.5645 um/px against a true 0.5480
                    # (3%, because Stage 2 had no H&E spacing to match grids with).
                    # That is close enough to select the right level, where the 40x
                    # assumption of 0.25 would pick one twice too coarse.
                    if he_base_mpp is None:
                        derived = scale = None
                        dapi_mpp0 = getattr(self, "dapi_base_mpp", None)
                        if dapi_mpp0 and not getattr(self, "dapi_base_mpp_assumed", False):
                            derived = derive_he_mpp_from_homography(
                                dapi_mpp0, h_mat, DAPI_LEVEL, HE_LEVEL)
                            if derived is not None:
                                scale = float(dapi_mpp0) * (2.0 ** DAPI_LEVEL) / (
                                    derived * (2.0 ** HE_LEVEL))
                        # print, not q.put: the CLI queue is a null sink, and these lines
                        # are the only record of which rule chose the H&E level.
                        if derived is not None:
                            print(f"[INFO] H&E has no physical pixel size in its OME "
                                  f"metadata; derived {derived:.4f} um/px from the DAPI "
                                  f"spacing ({dapi_mpp0:.4f}) and the Stage-2 homography "
                                  f"scale ({scale:.4f}).", flush=True)
                            he_base_mpp = derived
                            he_mpp_source = "derived"
                        else:
                            print(f"[WARN] H&E has no physical pixel size in its OME "
                                  f"metadata and it could not be derived from the Stage-2 "
                                  f"homography; assuming 40x ({he_assumed:.4f} um/px) at "
                                  f"level 0. Set stage3.fixed_level_override if that is wrong "
                                  f"for this slide.", flush=True)
                            he_base_mpp = he_assumed
                            he_mpp_source = "assumed 40x"
                he_start_level = choose_level_by_target_mpp(
                    he_base_mpp,
                    len(he_pyramid.get("level_shapes") or []) or (HE_LEVEL + 1),
                    he_target_mpp,
                    min_level=0, max_level=HE_LEVEL,
                )
                if he_start_level is not None and he_base_mpp:
                    print(f"[INFO] H&E target {he_target_mpp:.4f} um/px -> start "
                          f"level {he_start_level} "
                          f"({he_base_mpp * 2 ** he_start_level:.4f} um/px, "
                          f"{he_mpp_source})", flush=True)
                elif he_start_level is None:
                    print(f"[WARN] H&E pyramid unusable for the {he_target_mpp:.4f} um/px "
                          f"rule; the level-offset rule chooses the H&E tile level "
                          f"instead.", flush=True)
            except Exception as exc:
                print(f"[WARN] H&E pyramid unreadable ({exc}); "
                      "using the level-offset rule", flush=True)
            he_read_level, projected_he_side = choose_he_tile_level(
                HE_LEVEL,
                self.__dict__.get("level_selection_tiles", tiles),
                h_mat,
                # The size cap must use the tile size actually in force, which the
                # sampling step derived from the footprint in micrometres, not the
                # fallback in the parameter file.
                float(getattr(self, "current_tile_size_at_extract",
                              self.stage3.get("tile_size",
                                              parameters["stage3"].get("fallback_tile_size", 400)))),
                margin_ratio=parameters["stage3"]["he_tile_margin_ratio"],
                start_level=he_start_level,
                # The floor of 1 guards the level-offset fallback from picking an
                # absurdly fine level off a display choice. When the micrometre rule
                # has a real measurement to work from, it decides -- otherwise the
                # floor silently overrides it, which cost mouse_brain a 2x drop in
                # H&E tile resolution (level 1 instead of 0) for no memory saving:
                # that slide's whole H&E level 0 is 0.60 GiB. The projected-size cap
                # still applies either way.
                min_tile_level=0 if he_start_level is not None else 1,
            )
            print(
                f"[INFO] H&E extraction level={he_read_level}, "
                f"largest projected tile side={projected_he_side}px, "
                f"limit={4 * float(getattr(self, 'current_tile_size_at_extract', 400)):g}px",
                flush=True,
            )
        elif isinstance(HE_TILE_LEVEL_OVERRIDE, int):
            he_read_level = int(HE_TILE_LEVEL_OVERRIDE)
        else:
            raise ValueError("HE_TILE_LEVEL_OVERRIDE must be None, 'None', or int")

        print(f"[TIMER] save phase start | RSS {_rss_gib():.2f} GiB", flush=True)
        # Both slides are 1024x1024 tiled, so their tiles decode in parallel. On the
        # CRC images this is the largest speedup available for the read: DAPI level 1
        # goes from 19.4s single-threaded to 2.5s at eight workers, H&E from 7.1s to
        # 1.5s. `None` lets tifffile pick, which lands in between.
        _read_workers = int(self.stage3.get("read_workers", 8)) or None
        _t_read_he = time.perf_counter()
        he_img2 = read_side_image(
            HE_PATH, he_read_level, fixed.modality,
            workers=_read_workers, channel_index=fixed.channel_index,
        )
        _d_read_he = time.perf_counter() - _t_read_he
        print(f"[TIMER] read H&E level {he_read_level} "
                      f"{getattr(he_img2, 'shape', None)}: {_d_read_he:.2f}s "
                      f"| RSS {_rss_gib():.2f} GiB", flush=True)
        tick(
            28,
            f"Now reading H&E tile from level {he_read_level} "
            f"with shape={getattr(he_img2, 'shape', None)}"
        )

        rescale_f = 2 ** (HE_LEVEL - he_read_level)

        # ---- prune tiles that fall mostly outside HE ----
        prune_flag = bool(parameters["stage3"].get("prune_out_of_he", True))
        min_he_overlap_ratio = float(parameters["stage3"].get("min_he_overlap_ratio", 0.15))

        _d_prune = 0.0
        if prune_flag:
            _t_prune = time.perf_counter()
            tiles_before = len(tiles)
            tiles, kept_idx, dropped_idx = prune_tiles_out_of_he(
                tiles=tiles,
                h_mat=h_mat,
                he_img_shape=he_img2.shape,
                rescale_factor=rescale_f,
                margin_ratio=0.0,  # Prefer pruning based on the true tile footprint.
                min_overlap_ratio=min_he_overlap_ratio,
                require_center_inside=False,
            )
            q.put(("log",
                   f"[INFO] prune_out_of_he=True | kept {len(tiles)}/{tiles_before} tiles | "
                   f"dropped {len(dropped_idx)} | min_he_overlap_ratio={min_he_overlap_ratio}"))

            # Keep point indices aligned with the pruned tile list.
            if len(kept_idx) < len(points_xy):
                points_xy = points_xy[np.array(kept_idx, dtype=int)]
            _d_prune = time.perf_counter() - _t_prune
            print(f"[TIMER] prune tiles out of H&E: {_d_prune:.2f}s | RSS {_rss_gib():.2f} GiB", flush=True)
        else:
            q.put(("log", "[INFO] prune_out_of_he=False, keeping all tiles"))

        tick(35, f"Tiles after prune: {len(tiles)}")

        # ------------------------------
        # Save DAPI
        # ------------------------------
        report_stage(2, "Saving DAPI tiles (intensity + LUT)")

        DAPI_EXTRACT_LEVEL = getattr(self, "current_dapi_tile_level", None)
        if DAPI_EXTRACT_LEVEL is None:
            if DAPI_TILE_LEVEL_OVERRIDE in [None, "None"]:
                DAPI_EXTRACT_LEVEL = 1
            elif isinstance(DAPI_TILE_LEVEL_OVERRIDE, int):
                DAPI_EXTRACT_LEVEL = int(DAPI_TILE_LEVEL_OVERRIDE)
            else:
                raise ValueError("DAPI_TILE_LEVEL_OVERRIDE in parameter.json['stage3'] must be None, 'None', or int")

        _t_read_dapi = time.perf_counter()
        # One selected channel: the next statement discards every other channel, and
        # replicating a 267-megapixel uint16 plane to RGB first costs 1.0 GiB for
        # nothing.
        dapi16_extract = read_side_image(
            DAPI_PATH, DAPI_EXTRACT_LEVEL, moving.modality,
            workers=_read_workers, channel_index=moving.channel_index,
        )
        _d_read_dapi = time.perf_counter() - _t_read_dapi
        print(f"[TIMER] read DAPI level {DAPI_EXTRACT_LEVEL} "
                      f"{getattr(dapi16_extract, 'shape', None)} {dapi16_extract.dtype}: "
                      f"{_d_read_dapi:.2f}s | RSS {_rss_gib():.2f} GiB", flush=True)
        tick(
            45,
            f"Now reading DAPI tile from level {DAPI_EXTRACT_LEVEL} "
            f"with shape={getattr(dapi16_extract, 'shape', None)} dtype={dapi16_extract.dtype}"
        )

        dapi_rescale_factor = 2 ** (DAPI_LEVEL - DAPI_EXTRACT_LEVEL)

        _t_intensity = time.perf_counter()
        save_moving_tiles(
            dapi16_extract, tiles, str(output_folder), dapi_rescale_factor,
            moving.token, moving.modality,
        )
        _d_intensity = time.perf_counter() - _t_intensity
        print(f"[TIMER] save DAPI intensity tiles: {_d_intensity:.2f}s | RSS {_rss_gib():.2f} GiB", flush=True)
        tick(58, "Saved DAPI intensity tiles (u16 + u8 preview)")

        # The LUT tiles are for a human to look at. Nothing in the headless pipeline
        # reads them -- Stage 4b and Stage 7 both open `*_dapi_u8.png` -- and mapping
        # the whole extract-level image is the most expensive step in this phase, so
        # both GUI and CLI follow the same pair's write_lut_tiles setting.
        write_lut_tiles = (bool(parameters["stage3"].get("write_lut_tiles", False))
                           and moving.modality == "dapi")
        _d_lut = 0.0
        _d_save_dapi = 0.0
        dapi_tiles = []
        if write_lut_tiles:
            _t_lut = time.perf_counter()
            dapi_rgb2 = dapi_to_lut_rgb(dapi16_extract, lut, threshold=dapi_lut_threshold)
            _d_lut = time.perf_counter() - _t_lut
            print(f"[TIMER] apply LUT to DAPI: {_d_lut:.2f}s", flush=True)
        else:
            q.put(("log", "[INFO] write_lut_tiles=False; skipping the DAPI LUT pass"))
        tick(66, "Applied LUT to DAPI" if write_lut_tiles else "Skipped DAPI LUT")

        if write_lut_tiles:
            _t_save_dapi = time.perf_counter()
            dapi_tiles = save_dapi_tiles(
                dapi_rgb2,
                tiles,
                str(output_folder),
                rescale_factor=dapi_rescale_factor,
                token=moving.token,
            )
            _d_save_dapi = time.perf_counter() - _t_save_dapi
            print(f"[TIMER] save DAPI LUT tiles: {_d_save_dapi:.2f}s", flush=True)
        tick(78, f"Saved DAPI LUT tiles: {len(dapi_tiles) if hasattr(dapi_tiles, '__len__') else 'done'}")

        # ------------------------------
        # Save HE
        # ------------------------------
        report_stage(3, "Saving H&E tiles")

        _t_save_he = time.perf_counter()
        he_tiles = save_he_tiles(
            he_img2,
            tiles,
            h_mat,
            str(output_folder),
            rescale_factor=rescale_f,
            mode="rectified",
            margin_ratio=parameters['stage3']['he_tile_margin_ratio'],
            case_id=self.case_id,
            token=fixed.token,
        )
        _d_save_he = time.perf_counter() - _t_save_he
        print(f"[TIMER] save H&E tiles: {_d_save_he:.2f}s | RSS {_rss_gib():.2f} GiB", flush=True)
        tick(95, f"Saved H&E tiles: {len(he_tiles) if hasattr(he_tiles, '__len__') else 'done'}")
        dt = time.perf_counter() - timer0
        print(f"[TIMER] save phase end | RSS {_rss_gib():.2f} GiB", flush=True)
        q.put(("log", f"[DONE] Extract finished in {dt:.2f}s"))

        # The H&E tile level is now chosen from micrometres per pixel, so it can no
        # longer be re-derived downstream as HE_level - (DAPI_level - dapi_tile_level).
        # Record what was actually read; Stage 4a/4c read this back.
        sampled_path = self.run_dir / "sampled_points.json"
        try:
            record = json.loads(sampled_path.read_text())
        except Exception:
            record = {}
        record["pair"] = roles.pair.name
        record[f"{fixed.token}_level"] = int(HE_LEVEL)
        record[f"{fixed.token}_tile_level"] = int(he_read_level)
        record["he_level"] = int(HE_LEVEL)
        record["he_tile_level"] = int(he_read_level)
        sampled_path.write_text(json.dumps(record, indent=2))

        log_event(
            self.run_dir,
            "system_ready_save_tiles",
            tiles_dir=str(output_folder),
            n_tiles=int(len(tiles)),
            dapi_level=int(DAPI_LEVEL),
            he_level=int(HE_LEVEL),
            dapi_tile_level=int(DAPI_EXTRACT_LEVEL),
            he_read_level=int(he_read_level),
            elapsed_sec=round(dt, 3),
            read_he_sec=round(_d_read_he, 3),
            prune_sec=round(_d_prune, 3),
            read_dapi_sec=round(_d_read_dapi, 3),
            save_dapi_intensity_sec=round(_d_intensity, 3),
            dapi_lut_sec=round(_d_lut, 3),
            save_dapi_tiles_sec=round(_d_save_dapi, 3),
            save_he_tiles_sec=round(_d_save_he, 3),
        )
