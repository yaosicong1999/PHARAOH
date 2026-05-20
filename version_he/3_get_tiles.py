import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon
import threading
import queue
import subprocess
import tkinter as tk
from scipy import ndimage as ndi
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageOps
Image.MAX_IMAGE_PIXELS = None

# =============================
# Utils
# =============================
def load_step3_params(script_path: Path):
    """
    Read parameters.json (same dir as this script) and return step3 params with defaults.
    """
    script_dir = script_path.resolve().parent
    params_path = script_dir / "parameters.json"

    # defaults (fallback if json missing or keys missing)
    step3 = {
        "n_tiles": 120,
        "tile_size": 600,
        "min_dist_factor": 1.5,
    }

    if params_path.exists():
        try:
            params = json.load(open(params_path, "r"))
            if isinstance(params, dict) and isinstance(params.get("step3", {}), dict):
                step3.update(params["step3"])
        except Exception as e:
            print(f"[WARN] failed to read parameters.json: {e}", flush=True)
    else:
        print(f"[INFO] parameters.json not found at {params_path}, using defaults", flush=True)

    # normalize types
    step3["n_tiles"]    = int(step3.get("n_tiles", 120))
    step3["tile_size"]          = float(step3.get("tile_size", 600))
    step3["min_dist_factor"]    = float(step3.get("min_dist_factor", 1.5))

    return step3

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def apply_orientation_case(img: np.ndarray, case_id: int) -> np.ndarray:
    """
    Apply orientation case to image (H,W) or (H,W,C).
    case_id definition must match your Step1.
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

def prune_unmatched_he0_tiles(output_folder: str):
    """
    Keep only tile IDs that exist in BOTH he0_tile_info.json and he_tile_info.json.
    Remove unmatched he0 tile entries and corresponding files.
    """
    he0_json = os.path.join(output_folder, "he0_tile_info.json")
    he_json = os.path.join(output_folder, "he_tile_info.json")

    if not os.path.exists(he0_json):
        raise FileNotFoundError(f"missing {he0_json}")
    if not os.path.exists(he_json):
        raise FileNotFoundError(f"missing {he_json}")

    with open(he0_json, "r") as f:
        he0_info = json.load(f)
    with open(he_json, "r") as f:
        he_info = json.load(f)

    he0_keys = set(he0_info.keys())
    he_keys = set(he_info.keys())

    keep_keys = he0_keys & he_keys
    drop_keys = sorted(he0_keys - he_keys)

    print(f"[INFO] he0 tiles total: {len(he0_keys)}", flush=True)
    print(f"[INFO] he  tiles total: {len(he_keys)}", flush=True)
    print(f"[INFO] keeping matched tiles: {len(keep_keys)}", flush=True)
    print(f"[INFO] dropping unmatched he0 tiles: {len(drop_keys)}", flush=True)

    # remove dropped HE0 image files
    for k in drop_keys:
        rec = he0_info.get(k, {})
        for fn_key in ["filename", "filename_he0_u16", "filename_he0_u8"]:
            fn = rec.get(fn_key, None)
            if fn:
                p = os.path.join(output_folder, fn)
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[DROP] removed {p}", flush=True)

    # rewrite he0 json with only matched keys
    he0_info_new = {k: he0_info[k] for k in sorted(keep_keys)}
    with open(he0_json, "w") as f:
        json.dump(he0_info_new, f, indent=4)

    # optional: also rewrite he json in sorted matched order
    he_info_new = {k: he_info[k] for k in sorted(keep_keys)}
    with open(he_json, "w") as f:
        json.dump(he_info_new, f, indent=4)

    return sorted(keep_keys)

class StepTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
    def mark(self, name):
        now = time.perf_counter()
        print(f"[TIMER] {name:<30s}: {now - self.last:8.2f}s (total {now - self.t0:8.2f}s)", flush=True)
        self.last = now

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

def load_image_any(path: Path):
    if path is None or (not path.exists()):
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

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
                img8 = normalize_uint16_to_uint8(img[..., 0])  # 简单点：取一通道再归一
                return img8
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., 0]
    if img.dtype == np.uint8:
        return img
    return normalize_uint16_to_uint8(img)

def draw_points_overlay(he0_img, points_xy, tile_size=128, save_path=None):
    g8 = to_gray_uint8(he0_img)
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

def apply_density_filter(mask_tissue_255: np.ndarray,
                         density_8u: np.ndarray,
                         mode="percentile",
                         p=40,
                         thr_fixed=30,
                         morph_close=0,
                         min_area=0,
                         keep_bright=True):
    assert mask_tissue_255.shape == density_8u.shape
    tissue = (mask_tissue_255 > 0)
    if tissue.sum() == 0:
        return np.zeros_like(mask_tissue_255, dtype=np.uint8)

    vals = density_8u[tissue]
    thr = np.percentile(vals, p) if mode == "percentile" else thr_fixed

    if keep_bright:
        keep = tissue & (density_8u >= thr)
    else:
        keep = tissue & (density_8u <= thr)

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



def make_available_mask_from_he0_threshold(
    he0_threshold_mask_255: np.ndarray,
    safe_radius: int,
    close_ksize: int = 3,
    open_ksize: int = 3,
    blur_ksize: int = 1,
    rethreshold: int = 10,
    min_area: int = 2000,
    fill_holes: bool = True,
):
    """
    Input:
        he0_threshold_mask_255:
            255 = H-channel-positive pixels
            0   = background
        safe_radius:
            required inward margin for tile center

    Returns:
        mask_available:
            0   = available
            255 = unavailable

        cleaned_mask_255:
            cleaned tissue-like region used to define availability

        dist_map:
            distance transform on cleaned_mask
    """
    m = (he0_threshold_mask_255 > 0).astype(np.uint8) * 255

    # 1) connect nearby positive dots
    if close_ksize > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    # 2) remove isolated noise
    if open_ksize > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    # 3) smooth slightly, then binarize again
    if blur_ksize > 0:
        blur_ksize = int(blur_ksize) | 1
        m_blur = cv2.GaussianBlur(m, (blur_ksize, blur_ksize), 0)
        _, m = cv2.threshold(m_blur, rethreshold, 255, cv2.THRESH_BINARY)

    # 4) fill holes
    if fill_holes:
        m = ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255

    # 5) remove small connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    # 6) distance transform => keep only interior pixels
    dist_map = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    available_bin = (dist_map >= float(safe_radius)).astype(np.uint8)

    # convention in your pipeline:
    # 0   = available
    # 255 = unavailable
    mask_available = np.where(available_bin > 0, 0, 255).astype(np.uint8)

    return mask_available, cleaned, dist_map
def make_tissue_mask_from_he0_gray(
    he0_gray16: np.ndarray,
    blur_ksize=13,
    thr_mode="percentile",
    thr_percentile=45,
    thr_fixed=18,
    morph_close=25,
    morph_open=0,
    min_area=3000
):
    # force 2D
    if he0_gray16.ndim == 3:
        he0_gray16 = he0_gray16[..., 0]
    if he0_gray16.ndim != 2:
        raise ValueError(f"he0_gray16 must be 2D, got {he0_gray16.shape}")

    g8 = normalize_uint16_to_uint8(he0_gray16)
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


def make_available_mask_by_eroding_valid_region(valid_mask_255: np.ndarray, radius: int):
    """
    valid_mask_255:
        255 = valid region
        0   = invalid region

    return mask_available:
        0   = available for sampling
        255 = unavailable
    """
    valid = (valid_mask_255 > 0).astype(np.uint8)

    if radius > 0:
        ksize = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        valid = cv2.erode(valid, kernel, iterations=1)

    return np.where(valid > 0, 0, 255).astype(np.uint8)


def choose_extract_level(x, k,
                         anchor_sampling_tile=50,
                         min_extract_level=1,
                         max_extract_level=None):

    level_shift = round(math.log2(k / anchor_sampling_tile))
    n = x - level_shift

    if max_extract_level is not None:
        n = min(n, max_extract_level)

    n = max(min_extract_level, n)

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

        # 统计冲突多的点优先挪走
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
                cand = coords_valid[np.random.randint(len(coords_valid), size=K)]
                d, _ = tree.query(cand, k=1)
                new_points[i] = cand[np.argmax(d)]
            else:
                sub = coords_valid[region_idx]
                centroid = sub.mean(axis=0)
                k = np.argmin(np.sum((sub - centroid) ** 2, axis=1))
                new_points[i] = sub[k]

        points, ok = repel_too_close(new_points, coords_valid, MIN_SEP, seed=seed + it + 1)
    return points.astype(np.int32)


def normalized_dispersion_index_corrected(points, mask, alpha=0.4, beta=0.4, gamma=0.2):
    points = np.array(points, dtype=float)
    N = len(points)
    if N < 2:
        return 0.0
    ys, xs = np.where(mask == 0)
    available_poly = Polygon(np.column_stack((xs, ys))).convex_hull
    A_available = available_poly.area
    hull = MultiPoint(points).convex_hull
    A_hull = hull.area
    dist = np.sqrt(np.sum((points[None, :, :] - points[:, None, :]) ** 2, axis=-1))
    dists = dist[np.triu_indices(N, k=1)]
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    coords = np.column_stack((xs, ys))
    dmax = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    term1 = mean_d / dmax
    term2 = A_hull / A_available
    term3 = std_d / mean_d if mean_d > 0 else 0
    return alpha * term1 + beta * term2 - gamma * term3

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

def save_he0_tiles(
    he0_rgb,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="tile",
    start_index=0,
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_tiles = []
    output_dict = {}

    h_img, w_img = he0_rgb.shape[:2]

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

        tile_img = he0_rgb[y0:y1, x0:x1]

        key = f"{prefix}_{i:03d}"
        filename = f"{key}_he0.png"
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
        output_dict[key] = {k: v for k, v in info.items() if k != "img"}

    print(f"Saved HE0 RGB tiles in '{output_folder}': {len(saved_tiles)}")

    with open(os.path.join(output_folder, "he0_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=4)

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
    case_id=0,                        # NEW: must match your HE0 orientation case
):
    """
    Coordinate conventions (matching your original pipeline):
    - tiles are in HE0 tile coordinate system (whatever level those tiles are defined in).
    - h_mat maps HE0 coords -> HE coords (in HE tile coord system).
    - rescale_factor converts HE coords -> he_rgb pixel coords (e.g. level mapping like 2**(HE_LEVEL-1)).

    Output:
    - Saves <prefix>_<id>_he.png for each tile (bbox crop OR rectified patch depending on mode)
    - Writes he_tile_info.json containing for each tile:
        * core bbox-like info (x0,y0,w,h,cx,cy,type,id,filename)
        * meta (he0 corners, he quad corners, rectification matrix for rectified mode)
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

    def signed_area(q):
        q = np.asarray(q, dtype=np.float32)
        x = q[:, 0]
        y = q[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def orient_quad_indices(case_id: int):
        """
        Return indices (len=4) to reorder [TL,TR,BR,BL] corners to match the same
        orientation you apply to HE0 tiles (apply_orientation_case / apply_orientation_to_tile).
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

        # ==========================================
        # A) HE0 tile bbox (unchanged)
        # ==========================================
        x0 = float(x0f)
        y0 = float(y0f)
        x1 = x0 + float(wf)
        y1 = y0 + float(hf)

        corners_he0_tile = np.array(
            [[x0, y0],
             [x1, y0],
             [x1, y1],
             [x0, y1]], dtype=float
        )  # TL,TR,BR,BL

        # ==========================================
        # B) HE0 projection bbox (expanded for HE rectification)
        #    margin_ratio=0.2 means 1.2x larger
        # ==========================================
        expand = 1.0 + float(margin_ratio)  # e.g. 1.2
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = (float(wf) / 2.0) * expand
        half_h = (float(hf) / 2.0) * expand

        x0p = cx - half_w
        x1p = cx + half_w
        y0p = cy - half_h
        y1p = cy + half_h

        corners_he0_proj = np.array(
            [[x0p, y0p],
             [x1p, y0p],
             [x1p, y1p],
             [x0p, y1p]], dtype=float
        )  # TL,TR,BR,BL for projection

        # ---- project to HE coords using homography ----
        corners_h = np.hstack([corners_he0_proj, np.ones((4, 1), dtype=float)])  # (4,3)
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
            print(" corners_he0:\n", corners_he0_proj)
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
            # src corners are already in HE0 order: TL,TR,BR,BL (after H + rf)
            src = corners_he_px_raw.astype(np.float32)

            # --- choose output size (natural: avg opposite edges) ---
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

            # --- apply SAME orientation convention as HE0 tile ---
            idx = orient_quad_indices(int(case_id))
            dst = dst[idx]

            # --- fix mirror (winding mismatch) while keeping point correspondence ---
            if signed_area(src) * signed_area(dst) < 0:
                # swap TR <-> BL (indices 1 and 3) in src to match dst winding
                src = src[[0, 3, 2, 1]]

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = np.linalg.inv(M)

            tile_img = cv2.warpPerspective(
                he_rgb, M, (out_w, out_h),
                flags=rectify_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        filename = f"{key}_he.png"
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
            "he0_corners_tile": corners_he0_tile.tolist(),  # 原 tile（不扩）
            "he0_corners_proj": corners_he0_proj.tolist(),  # 用于投影到 HE 的扩张框
            "proj_expand": float(1.0 + margin_ratio),
            "he_quad_px_raw": corners_he_px_raw.tolist(),     # TL,TR,BR,BL (HE0 order) in he_rgb coords
            "rectified_wh": [int(out_w), int(out_h)],
            "M_he_to_rect": None if M is None else M.tolist(),
            "M_rect_to_he": None if Minv is None else Minv.tolist(),
        }
        he_tiles.append({**info, "meta": meta, "img": tile_img})

        # JSON version (no image array)
        output_dict[key] = {k: v for k, v in info.items() if k != "img"}
        output_dict[key]["meta"] = meta

    print(f"Saved H&E tiles in '{output_folder}' (mode={mode}): {len(he_tiles)}")

    with open(os.path.join(output_folder, "he_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=4)

    return he_tiles

def centroids_to_tiles(points_xy, tile_size):
    half = tile_size / 2.0
    tiles = []
    for (x, y) in points_xy:
        tiles.append({"x0": float(x - half), "y0": float(y - half), "w": float(tile_size), "h": float(tile_size)})
    return tiles

# =============================
# GUI App
# =============================
class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, title="Working..."):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)

        self._t0 = time.perf_counter()
        self._running = True
        self._stage_i = 0
        self._stage_total = 0
        self._stage_name = "Starting..."
        self._suffix = ""

        self.var = tk.DoubleVar(value=0.0)

        # 顶部状态行
        self.lbl = tk.Label(self, text="", anchor="w")
        self.lbl.pack(side="top", fill="x", padx=12, pady=(12, 6))

        self.pb = ttk.Progressbar(
            self, orient="horizontal", mode="determinate",
            maximum=100.0, variable=self.var, length=420
        )
        self.pb.pack(side="top", fill="x", padx=12, pady=(0, 10))

        self.txt = tk.Text(self, height=12, width=70)
        self.txt.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt.configure(state="disabled")

        self.btn_close = tk.Button(self, text="Close", state="disabled", command=self.destroy)
        self.btn_close.pack(side="bottom", pady=(0, 12))

        self.transient(parent)
        self.grab_set()

        # start auto elapsed update
        self._tick_elapsed()

    def start_elapsed(self):
        def tick():
            if not self._running:
                return
            elapsed = time.perf_counter() - self._t0
            self.lbl.config(
                text=f"{self.lbl.cget('text').split(' | ')[0]} | Elapsed: {elapsed:.1f}s"
            )
            self.after(200, tick)

        tick()

    def stop_elapsed(self):
        self._running = False
        # freeze final header (once)
        try:
            self.lbl.config(text=self._format_header())
        except tk.TclError:
            pass

    def _format_header(self):
        elapsed = time.perf_counter() - self._t0
        suffix = self._suffix or ""
        if self._stage_total > 0 and self._stage_i > 0:
            return f"STAGE {self._stage_i}/{self._stage_total}  {self._stage_name}   Elapsed: {elapsed:,.1f}s{suffix}"
        else:
            return f"{self._stage_name}   Elapsed: {elapsed:,.1f}s{suffix}"

    def _tick_elapsed(self):
        if not self._running:
            return
        try:
            self.lbl.config(text=self._format_header())
            self.after(200, self._tick_elapsed)
        except tk.TclError:
            return

    def set_stage(self, i: int, total: int, name: str):
        self._stage_i = int(i)
        self._stage_total = int(total)
        self._stage_name = str(name)
        self.lbl.config(text=self._format_header())

    def set_status(self, s: str):
        # 兼容旧接口：仅更新 stage_name
        self._stage_name = str(s)
        self.lbl.config(text=self._format_header())

    def set_progress(self, p: float):
        self.var.set(float(p))

    def log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s.rstrip() + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def enable_close(self):
        self.btn_close.config(state="normal")

    def mark_done(self):
        self._suffix = " (Done)"
        self.lbl.config(text=self._format_header())

    def mark_failed(self):
        self._suffix = " (Failed)"
        self.lbl.config(text=self._format_header())

class Step3SamplingApp(tk.Tk):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

        self.title("Step 3 — CVT sampling")

        # 3 panels
        self.tile_size = (420, 420)

        # runtime state
        self.has_sampling_outputs = False

        self.sampling_counter = 0
        # ---- load step3 params from parameters.json (same folder as this script)
        self.step3 = load_step3_params(Path(__file__))
        print(f"[PARAM] step3 = {self.step3}", flush=True)

        # orientation case (from images_info.json)
        self.case_id = 0
        info_path = self.run_dir / "images_info.json"
        if info_path.exists():
            try:
                info = json.load(open(info_path, "r"))
                self.case_id = int(info.get("HE0_orientation_case", 0))
            except Exception as e:
                print(f"[WARN] failed to read HE0_orientation_case: {e}", flush=True)
        print(f"[INFO] HE0_orientation_case={self.case_id}", flush=True)

        # -------- layout
        top = tk.Frame(self)
        top.pack(side="top", fill="x", padx=12, pady=(10, 6))

        tk.Label(
            top,
            text=f"RUN_DIR: {self.run_dir}",
            anchor="w",
            font=("Helvetica", 12, "bold"),
        ).pack(side="left", fill="x", expand=True)

        mid = tk.Frame(self)
        mid.pack(side="top", fill="x", padx=12, pady=(6, 6))

        self.panel_left  = self._make_image_panel(mid, "HE0 Image")
        self.panel_mid   = self._make_image_panel(mid, "Available Mask for Tile Centroids")
        self.panel_right = self._make_image_panel(mid, "Sampled Tile Centroids")

        self.panel_left.grid(row=0, column=0, padx=8, sticky="n")
        self.panel_mid.grid(row=0, column=1, padx=8, sticky="n")
        self.panel_right.grid(row=0, column=2, padx=8, sticky="n")

        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)

        bottom = tk.Frame(self)
        bottom.pack(side="top", fill="x", padx=12, pady=(2, 12))

        self.btn_sampling = tk.Button(
            bottom, text="Sample tile centroids",
            command=self.on_sampling_clicked
        )
        self.btn_sampling.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_pilot = tk.Button(
            bottom, text="Tile pilot examination",
            command=self.on_pilot_clicked,
            state="disabled"
        )
        self.btn_pilot.pack(side="left", fill="x", expand=True, padx=(6, 6))

        self.btn_extract = tk.Button(
            bottom, text="Extract current tiles",
            command=self.on_extract_clicked,
            state="disabled"
        )
        self.btn_extract.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # load initial images
        self._load_initial_left()
        self._set_placeholder(self.panel_mid, "Not Available Now")
        self._set_placeholder(self.panel_right, "Not Available Now")

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

    def _make_image_panel(self, parent, title):
        frame = tk.Frame(parent)
        tk.Label(frame, text=title, font=("Helvetica", 12, "bold")).pack(side="top", pady=(0, 6))
        lbl = tk.Label(frame)
        lbl.pack(side="top")
        frame._img_label = lbl
        return frame

    def _set_panel_image(self, panel_frame, cv_img, apply_orientation=True):
        if apply_orientation:
            cv_img = apply_orientation_case(cv_img, self.case_id)
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil, size=self.tile_size)
        tk_img = ImageTk.PhotoImage(tile)
        panel_frame._img_label.configure(image=tk_img)
        panel_frame._img_label.image = tk_img

    def _set_placeholder(self, panel_frame, text="placeholder"):
        w, h = self.tile_size  # (W,H)
        img = np.full((h, w, 3), 240, np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (80, 80, 80)

        # --- compute centered position ---
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2 + th  # y is baseline in cv2.putText

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # IMPORTANT: placeholder must NOT be rotated/flipped
        self._set_panel_image(panel_frame, img, apply_orientation=False)

    def _load_initial_left(self):
        img_dir = self.run_dir / "1_he0_threshold_mask.png"
        img = load_image_any(img_dir)
        if img is None:
            self._set_placeholder(self.panel_left, "missing HE0")
            messagebox.showwarning("Missing", f"Can't find {img_dir.name} in RUN_DIR.")
            return

        self._set_panel_image(self.panel_left, img, apply_orientation=False)

    # --------------------------
    # Button callbacks
    # --------------------------
    def on_sampling_clicked(self):
        try:
            timer = StepTimer()

            self.sampling_counter += 1
            seed = int(time.time() * 1000) % (2 ** 31 - 1)
            print(f"[INFO] Sampling seed = {seed}", flush=True)

            ## read reading level
            info_path = self.run_dir / "images_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"missing {info_path}")
            info = json.load(open(info_path, "r"))
            HE0_PATH = info["HE0_path"]
            HE0_LEVEL = int(info["HE0_level"])
            print(f"[INFO] HE0_PATH={HE0_PATH}", flush=True)
            print(f"[INFO] HE0_LEVEL={HE0_LEVEL}", flush=True)
            from my_utils import read_image
            he016, _ = read_image(HE0_PATH, keep_16bit=True, level=HE0_LEVEL, channel="he")
            timer.mark("Read HE0")
            cv2.imwrite(str(self.run_dir / "3_dbg_he0_gray.png"), normalize_uint16_to_uint8(he016))
            timer.mark("Save gray preview")

            # choose HE0 extract level first
            target_tile_size_at_extract = int(self.step3.get("tile_size", 400))
            if "he0_level_override" in self.step3 and self.step3["he0_level_override"] not in [None, "None"]:
                HE0_EXTRACT_LEVEL = int(self.step3["he0_level_override"])
            else:
                HE0_EXTRACT_LEVEL = choose_extract_level(
                    x=HE0_LEVEL,
                    k=target_tile_size_at_extract,
                    min_extract_level=1,
                    max_extract_level=HE0_LEVEL,
                )
            print(f"[INFO] HE0_EXTRACT_LEVEL={HE0_EXTRACT_LEVEL}", flush=True)

            # tile size + radius in sampling coords
            # target tile size is defined at extract level
            tile_size_at_extract = float(self.step3["tile_size"])   # e.g. 600 px at extract level
            N_TILES = int(self.step3["n_tiles"])
            min_dist_fac = float(self.step3["min_dist_factor"])
            scale_extract_to_sampling = 2 ** (HE0_LEVEL - HE0_EXTRACT_LEVEL)
            TILE_SIZE = tile_size_at_extract / scale_extract_to_sampling
            MIN_DIST = TILE_SIZE * min_dist_fac
            # buffer also defined in extract-level pixels, then mapped to sampling level
            buffer_at_extract = float(self.step3.get("buffer_px_at_extract", 0))
            buffer = buffer_at_extract / scale_extract_to_sampling
            print(f"[INFO] tile_size_at_extract={tile_size_at_extract}", flush=True)
            print(f"[INFO] TILE_SIZE@sampling={TILE_SIZE}", flush=True)
            print(f"[INFO] MIN_DIST@sampling={MIN_DIST}", flush=True)

            he0_thr_path = self.run_dir / "1_he0_threshold_mask.png"
            he0_thr = cv2.imread(str(he0_thr_path), cv2.IMREAD_GRAYSCALE)
            if he0_thr is None:
                raise FileNotFoundError(f"missing {he0_thr_path}")

            safe_radius = int(np.ceil(TILE_SIZE / 2 + buffer))
            print(f"[INFO] safe_radius@sampling={safe_radius}", flush=True)

            avail, cleaned_mask, dist_map = make_available_mask_from_he0_threshold(
                he0_thr,
                safe_radius=safe_radius,
                close_ksize=7,
                open_ksize=3,
                blur_ksize=5,
                rethreshold=10,
                min_area=2000,
                fill_holes=True,
            )

            cv2.imwrite(str(self.run_dir / "3_dbg_he0_cleaned_mask.png"), cleaned_mask)

            # save distance transform preview
            if dist_map.max() > 0:
                dist_vis = (dist_map / dist_map.max() * 255).astype(np.uint8)
            else:
                dist_vis = np.zeros_like(dist_map, dtype=np.uint8)
            cv2.imwrite(str(self.run_dir / "3_dbg_he0_distance_map.png"), dist_vis)

            avail_flipped = 255 - avail
            avail_path = self.run_dir / "3_dbg_available_after_erode.png"
            cv2.imwrite(str(avail_path), avail_flipped)

            print(f"[INFO] cleaned_mask > 0: {np.count_nonzero(cleaned_mask > 0)}", flush=True)
            print(f"[INFO] avail == 0: {np.count_nonzero(avail == 0)}", flush=True)
            timer.mark("Make available mask from he0 threshold")
            points_xy = cvt_masked(
                avail,
                N_POINTS=N_TILES,
                MIN_DIST=MIN_DIST,
                ITERATIONS=50,
                seed=seed
            )
            timer.mark("CVT sampling")
            self.current_he0_extract_level = HE0_EXTRACT_LEVEL

            # ndi_score = normalized_dispersion_index_corrected(points_xy, avail)
            # print("NDI Score:", ndi_score)
            tiles = centroids_to_tiles(points_xy, tile_size=TILE_SIZE)

            self.current_points_xy = points_xy
            self.current_tiles = tiles
            self.current_tile_size = TILE_SIZE
            self.current_he0_level = HE0_LEVEL

            # save points
            out_json = self.run_dir / "sampled_points.json"
            json.dump(
                {"he0_level": HE0_LEVEL,
                 "he0_extract_level": HE0_EXTRACT_LEVEL,
                 "tile_size_sampling": TILE_SIZE,
                 "points_xy": points_xy.tolist(),
                },
                open(out_json, "w"),
                indent=2
            )
            timer.mark("Save points json")

            overlay_bgr = draw_points_overlay(
                he016, points_xy,
                tile_size=TILE_SIZE,
                save_path=str(self.run_dir / "3_sampled_overlay.png")
            )
            timer.mark("Save overlay")

            # ---- update GUI: mid & right images
            mid_img = load_image_any(avail_path)
            right_img = load_image_any(self.run_dir / "3_sampled_overlay.png")
            if mid_img is None:
                mid_img = avail_flipped  # fallback (single channel)
            if right_img is None:
                right_img = overlay_bgr

            self._set_panel_image(self.panel_mid, mid_img)
            self._set_panel_image(self.panel_right, right_img)

            # enable pilot/extract button
            self.btn_pilot.config(state="normal")
            self.btn_extract.config(state="normal")
            self.has_sampling_outputs = True
            print("[DONE] sampling finished.", flush=True)

        except Exception as e:
            messagebox.showerror("Sampling failed", str(e))
            raise

    def on_pilot_clicked(self):
        """
        Launch 3b.py for tile pilot examination.
        Pass RUN_DIR as argv[1].
        """
        try:
            script_dir = Path(__file__).resolve().parent
            pilot_script = script_dir / "3b_tile_pilot.py"

            if not pilot_script.exists():
                messagebox.showerror("Missing", f"Cannot find {pilot_script}")
                return

            # Use same python executable (important for env)
            py = sys.executable

            cmd = [py, str(pilot_script), str(self.run_dir)]
            print("[INFO] launching:", " ".join(cmd), flush=True)

            # Non-blocking
            subprocess.Popen(cmd, cwd=str(script_dir))

        except Exception as e:
            messagebox.showerror("Failed to launch 3b.py", str(e))

    def on_extract_clicked(self):
        if not self.has_sampling_outputs:
            messagebox.showwarning("Not ready", "Please run sampling first.")
            return

        def worker(q):
            timer0 = time.perf_counter()

            STAGE_TOTAL = 4
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
            info = json.load(open(info_path, "r"))
            parameters = json.load(open("parameters.json", "r"))

            HE0_PATH = info["HE0_path"]
            HE_PATH = info["HE_path"]
            HE0_LEVEL = int(info["HE0_level"])
            HE_LEVEL = int(info["HE_level"])
            HE0_TILE_LEVEL_OVERRIDE = parameters["step3"]["he0_level_override"]
            HE_TILE_LEVEL_OVERRIDE = parameters["step3"]["he_level_override"]

            # ---------- read LUT threshold from images_info.json ----------
            sampled_path = self.run_dir / "sampled_points.json"
            if not sampled_path.exists():
                raise FileNotFoundError(f"missing {sampled_path}, run sampling first.")
            sampled = json.load(open(sampled_path, "r"))
            points_xy = np.asarray(sampled["points_xy"], dtype=np.int32)

            # use tiles generated from latest sampling
            if not hasattr(self, "current_tiles") or self.current_tiles is None:
                raise RuntimeError("No tiles found. Please click 'Sample tile centroid' first.")
            tiles = self.current_tiles
            output_folder = self.run_dir / "tiles"
            ensure_dir(output_folder)

            tick(5, f"HE0_LEVEL={HE0_LEVEL}, HE_LEVEL={HE_LEVEL}")
            tick(6, f"HE0_TILE_LEVEL_OVERRIDE={HE0_TILE_LEVEL_OVERRIDE}, HE_TILE_LEVEL_OVERRIDE={HE_TILE_LEVEL_OVERRIDE}")
            tick(8, f"Output: {output_folder}")

            # ---------- imports from your project ----------
            report_stage(2, "Importing project utils")
            from my_utils import read_image
            tick(12)

            report_stage(3, "Saving HE0 tiles")

            HE0_EXTRACT_LEVEL = getattr(self, "current_he0_extract_level", None)
            if HE0_EXTRACT_LEVEL is None:
                if HE0_TILE_LEVEL_OVERRIDE in [None, "None"]:
                    HE0_EXTRACT_LEVEL = 1
                elif isinstance(HE0_TILE_LEVEL_OVERRIDE, int):
                    HE0_EXTRACT_LEVEL = int(HE0_TILE_LEVEL_OVERRIDE)
                else:
                    raise ValueError("HE0_TILE_LEVEL_OVERRIDE in parameter.json['step3'] must be None, 'None', or int")

            he0_rgb_extract, _ = read_image(
                HE0_PATH,
                keep_16bit=False,
                level=HE0_EXTRACT_LEVEL,
                channel="he"
            )
            tick(
                20,
                f"Now reading HE0 RGB tile from level {HE0_EXTRACT_LEVEL} "
                f"with shape={getattr(he0_rgb_extract, 'shape', None)} dtype={he0_rgb_extract.dtype}"
            )

            he0_rescale_factor = 2 ** (HE0_LEVEL - HE0_EXTRACT_LEVEL)

            he0_tiles = save_he0_tiles(
                he0_rgb_extract,
                tiles,
                str(output_folder),
                rescale_factor=he0_rescale_factor,
                prefix="tile",
                start_index=0,
            )
            tick(55, f"Saved HE0 RGB tiles: {len(he0_tiles)}")

            # ------------------------------
            # 6. Save HE tiles using transformation
            # ------------------------------
            report_stage(4, "Saving HE tiles")
            path_clicked = self.run_dir / "clicked_blob_initial_alignment.json"
            path_manual = self.run_dir / "manual_initial_alignment.json"
            if path_clicked.exists():
                data = json.load(open(path_clicked, "r"))
                q.put(("log", f"[INFO] Using alignment from: {path_clicked.name}"))
            elif path_manual.exists():
                data = json.load(open(path_manual, "r"))
                q.put(("log", f"[INFO] Using alignment from: {path_manual.name}"))
            else:
                raise FileNotFoundError(
                    "Neither clicked_blob_initial_alignment.json nor manual_initial_alignment.json found.")

            h_mat = data["H_mat"]
            tick(72, "Loaded initial alignment")

            HE0_EXTRACT_LEVEL = getattr(self, "current_he0_extract_level", None)
            if HE0_EXTRACT_LEVEL is None:
                HE0_EXTRACT_LEVEL = 1
            if HE_TILE_LEVEL_OVERRIDE in [None, "None"]:
                he_read_level = HE_LEVEL - (HE0_LEVEL - HE0_EXTRACT_LEVEL)
                he_read_level = max(1, he_read_level)
            elif isinstance(HE_TILE_LEVEL_OVERRIDE, int):
                he_read_level = int(HE_TILE_LEVEL_OVERRIDE)
            else:
                raise ValueError("HE_TILE_LEVEL_OVERRIDE must be None, 'None', or int")
            he_img2, _ = read_image(
                HE_PATH,
                keep_16bit=False,
                level=he_read_level,
                channel="he"
            )
            tick(
                78,
                f"Now reading H&E tile from level {he_read_level} "
                f"with shape={getattr(he_img2, 'shape', None)}"
            )

            rescale_f = 2 ** (HE_LEVEL - he_read_level)
            he_tiles = save_he_tiles(
                he_img2,
                tiles,
                h_mat,
                str(output_folder),
                rescale_factor=rescale_f,
                mode="rectified",
                margin_ratio=parameters['step3']['he_tile_margin_ratio'],
                case_id=self.case_id,
            )
            matched_keys = prune_unmatched_he0_tiles(str(output_folder))
            tick(95, f"Saved matched HE0/HE tiles: {len(matched_keys)}")

            # done
            dt = time.perf_counter() - timer0
            q.put(("log", f"[DONE] Extract finished in {dt:.2f}s"))

        # run with progress dialog
        self._run_with_progress("Extracting tiles...", worker)

    def _run_with_progress(self, title, worker_fn):
        dlg = ProgressDialog(self, title=title)
        q = queue.Queue()

        def pump():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "stage":
                        i, total, name = payload
                        dlg.set_stage(i, total, name)
                        dlg.log(f"[STAGE] {i}/{total} {name}")
                    elif kind == "progress":
                        dlg.set_progress(payload)
                    elif kind == "log":
                        dlg.log(payload)
                    elif kind == "done":
                        dlg.mark_done()
                        dlg.stop_elapsed()
                        dlg.set_progress(100)
                        dlg.enable_close()
                        return
                    elif kind == "error":
                        dlg.log("[ERROR] " + str(payload))
                        dlg.mark_failed()
                        dlg.stop_elapsed()
                        dlg.enable_close()
                        return
            except queue.Empty:
                pass
            self.after(100, pump)

        def bg():
            try:
                worker_fn(q)
                q.put(("done", None))
            except Exception as e:
                q.put(("error", str(e)))

        threading.Thread(target=bg, daemon=True).start()
        pump()

def main():
    if len(sys.argv) < 2:
        print("Usage: python 3.py <RUN_DIR>")
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] RUN_DIR not found: {run_dir}")
        sys.exit(2)

    app = Step3SamplingApp(run_dir)
    app.mainloop()


if __name__ == "__main__":
    main()