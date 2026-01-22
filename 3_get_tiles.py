import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon
import threading
import queue

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageOps

Image.MAX_IMAGE_PIXELS = None


# =============================
# Utils
# =============================
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


# =============================
# CVT sampling pipeline (来自你 test.py 的思路，内嵌进来)
# =============================
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
    # force 2D
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


def make_available_mask_boundary_only(mask_tissue255: np.ndarray, boundary_radius: int = 0):
    """
    只做“图像边界 buffer”，不做 tissue 内缩：
      available = tissue ∩ inner_image_box
    输出约定：0=available, 255=unavailable
    """
    h, w = mask_tissue255.shape[:2]
    tissue = (mask_tissue255 > 0)

    if boundary_radius <= 0:
        inner = np.ones((h, w), dtype=bool)
    else:
        inner = np.zeros((h, w), dtype=bool)
        inner[boundary_radius:h - boundary_radius, boundary_radius:w - boundary_radius] = True

    available = tissue & inner
    return np.where(available, 0, 255).astype(np.uint8)


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


def enforce_min_distances(points, coords_valid, min_dist, seed=0):
    rng = np.random.default_rng(seed)
    if len(points) < 2:
        return points
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(min_dist))
    if not pairs:
        return points
    for (i, j) in pairs:
        points[j] = coords_valid[rng.integers(0, len(coords_valid))]
    return points


def cvt_masked(mask_available, N_POINTS=80, MIN_DIST=7, ITERATIONS=50, seed=0):
    coords_valid = get_valid_coords(mask_available, seed=seed)
    if len(coords_valid) == 0:
        raise ValueError("No available pixels to sample from. (mask_available==0 is empty)")

    points = initialize_points(coords_valid, N_POINTS, MIN_DIST, seed=seed)
    if len(points) == 0:
        raise ValueError("Failed to initialize any points. Check MIN_DIST / mask size.")

    N = len(points)

    for it in range(ITERATIONS):
        tree = cKDTree(points)
        _, idxs = tree.query(coords_valid)

        new_points = points.copy()
        for i in range(N):
            region_idx = np.where(idxs == i)[0]
            if len(region_idx) == 0:
                new_points[i] = coords_valid[np.random.randint(len(coords_valid))]
            else:
                sub = coords_valid[region_idx]
                centroid = sub.mean(axis=0)
                k = np.argmin(np.sum((sub - centroid) ** 2, axis=1))
                new_points[i] = sub[k]

        points = enforce_min_distances(new_points, coords_valid, MIN_DIST, seed=seed + it + 1)

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


def save_dapi_tiles(
    dapi_rgb,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="tile",          # file/key prefix
    start_index=0,          # in case you want to append
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
        filename = f"{key}_dapi.png"
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

    with open(os.path.join(output_folder, "dapi_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=4)

    return saved_tiles


def save_he_tiles(
    he_rgb,
    tiles,
    h_mat,
    output_folder,
    rescale_factor=1.0,
    margin_ratio=0.1,
    prefix="tile",
    start_index=0,
    debug_first_n=0,   # >0 就打印前 n 个 tile 的 corners/transform
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    he_tiles = []
    output_dict = {}
    H = np.array(h_mat, dtype=float)
    h_img, w_img = he_rgb.shape[:2]

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        # enlarge around center in DAPI space (still in tile coordinate system)
        mw = float(wf) * (1.0 + float(margin_ratio))
        mh = float(hf) * (1.0 + float(margin_ratio))
        x0_centered = float(x0f) - (mw - float(wf)) / 2.0
        y0_centered = float(y0f) - (mh - float(hf)) / 2.0
        x1_centered = x0_centered + mw
        y1_centered = y0_centered + mh

        corners = np.array(
            [
                [x0_centered, y0_centered],
                [x1_centered, y0_centered],
                [x1_centered, y1_centered],
                [x0_centered, y1_centered],
            ],
            dtype=float,
        )

        # affine: (x',y') = A*[x,y] + t
        transformed = np.dot(H[:, :2], corners.T).T + H[:, 2]

        if debug_first_n and (i - start_index) < debug_first_n:
            print(f"[DEBUG] tile {i}:")
            print("H:\n", H)
            print("corners (DAPI coords):\n", corners)
            print("transformed (HE coords, before rescale):\n", transformed)
            print("HE image shape:", he_rgb.shape)

        xs = transformed[:, 0] * float(rescale_factor)
        ys = transformed[:, 1] * float(rescale_factor)

        min_x = int(np.floor(xs.min()))
        max_x = int(np.ceil(xs.max()))
        min_y = int(np.floor(ys.min()))
        max_y = int(np.ceil(ys.max()))

        # clamp
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w_img, max_x)
        max_y = min(h_img, max_y)

        if max_x <= min_x or max_y <= min_y:
            continue

        tile_img = he_rgb[min_y:max_y, min_x:max_x]

        key = f"{prefix}_{i:03d}"
        filename = f"{key}_he.png"
        cv2.imwrite(os.path.join(output_folder, filename), cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))

        info = {
            "x0": min_x, "y0": min_y,
            "w": max_x - min_x, "h": max_y - min_y,
            "cx": (min_x + max_x) / 2, "cy": (min_y + max_y) / 2,
            "type": "sampled",
            "id": i,
            "filename": filename,
            "img": tile_img,
        }
        he_tiles.append(info)
        output_dict[key] = {k: v for k, v in info.items() if k != "img"}

    print(f"Saved H&E tiles in '{output_folder}': {len(he_tiles)}")

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

        self.title("Step 3 — CVT sampling draft")

        # 3 panels
        self.tile_size = (420, 420)

        # runtime state
        self.has_sampling_outputs = False

        self.sampling_counter = 0

        # orientation case (from images_info.json)
        self.case_id = 0
        info_path = self.run_dir / "images_info.json"
        if info_path.exists():
            try:
                info = json.load(open(info_path, "r"))
                self.case_id = int(info.get("DAPI_orientation_case", 0))
            except Exception as e:
                print(f"[WARN] failed to read DAPI_orientation_case: {e}", flush=True)
        print(f"[INFO] DAPI_orientation_case={self.case_id}", flush=True)

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

        self.panel_left  = self._make_image_panel(mid, "DAPI Image(LUT-ed)")
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
        img_dir = self.run_dir / "1_dapi_lut.png"

        img = load_image_any(img_dir)

        if img is None:
            self._set_placeholder(self.panel_left, "missing DAPI-luted")
            messagebox.showwarning("Missing", f"Can't find {img_dir.name} in RUN_DIR.")
            return

        self._set_panel_image(self.panel_left, img, apply_orientation=False)

    # --------------------------
    # Button callbacks
    # --------------------------
    def on_sampling_clicked(self):
        """
        这里直接跑你 test.py 的 CVT sampling pipeline
        """
        try:
            timer = StepTimer()

            self.sampling_counter += 1
            seed = int(time.time() * 1000) % (2 ** 31 - 1)
            print(f"[INFO] Sampling seed = {seed}", flush=True)

            info_path = self.run_dir / "images_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"missing {info_path}")

            info = json.load(open(info_path, "r"))
            DAPI_PATH = info["DAPI_path"]
            DAPI_LEVEL = int(info["DAPI_level"])
            print(f"[INFO] DAPI_PATH={DAPI_PATH}", flush=True)
            print(f"[INFO] DAPI_LEVEL={DAPI_LEVEL}", flush=True)

            # 读 dapi（沿用你的 my_utils.read_image）
            from my_utils import read_image
            dapi16, _ = read_image(DAPI_PATH, keep_16bit=True, level=DAPI_LEVEL)
            timer.mark("Read DAPI")

            # debug preview
            cv2.imwrite(str(self.run_dir / "3_dbg_dapi_gray.png"), normalize_uint16_to_uint8(dapi16))
            timer.mark("Save gray preview")

            # tissue mask + density
            mask_tissue, density = make_tissue_mask_from_dapi_gray(
                dapi16,
                blur_ksize=13,
                thr_mode="percentile",
                thr_percentile=45,
                morph_close=25,
                morph_open=0,
                min_area=3000
            )
            cv2.imwrite(str(self.run_dir / "3_dbg_tissue_mask.png"), mask_tissue)

            mask_dense = apply_density_filter(
                mask_tissue, density,
                mode="percentile",
                p=40,
                morph_close=11
            )
            cv2.imwrite(str(self.run_dir / "3_dbg_density_mask.png"), mask_dense)
            timer.mark("Make tissue+density mask")

            # boundary-only available mask
            TILE_SIZE = 600
            TILE_SIZE = TILE_SIZE / (2 ** (DAPI_LEVEL - 1))  # 你之前的写法，先沿用
            buffer = 10 / (2 ** (DAPI_LEVEL - 1))
            boundary_radius = int(np.ceil(TILE_SIZE * np.sqrt(2) / 2)) + int(buffer)

            avail = make_available_mask_boundary_only(mask_dense, boundary_radius=boundary_radius)
            avail_path = self.run_dir / "3_dbg_available_after_erode.png"
            cv2.imwrite(str(avail_path), avail)
            timer.mark("Make available mask (boundary)")

            # CVT
            MIN_DIST = TILE_SIZE * 1.5
            points_xy = cvt_masked(
                avail,
                N_POINTS=120,
                MIN_DIST=MIN_DIST,
                ITERATIONS=50,
                seed=seed
            )
            timer.mark("CVT sampling")
            # ndi_score = normalized_dispersion_index_corrected(points_xy, avail)
            # print("NDI Score:", ndi_score)

            tiles = centroids_to_tiles(points_xy, tile_size=TILE_SIZE)

            # 保存到 self，供 Extract 按钮使用
            self.current_points_xy = points_xy
            self.current_tiles = tiles
            self.current_tile_size = TILE_SIZE
            self.current_dapi_level = DAPI_LEVEL

            # save points
            out_json = self.run_dir / "sampled_points.json"
            json.dump({"dapi_level": DAPI_LEVEL, "points_xy": points_xy.tolist()},
                      open(out_json, "w"), indent=2)
            timer.mark("Save points json")

            # overlay —— 用 raw dapi16 + raw points_xy 画
            overlay_bgr = draw_points_overlay(
                dapi16, points_xy,
                tile_size=TILE_SIZE,
                save_path=str(self.run_dir / "3_sampled_overlay.png")
            )
            timer.mark("Save overlay")

            # ---- update GUI: mid & right images
            mid_img = load_image_any(avail_path)
            right_img = load_image_any(self.run_dir / "3_sampled_overlay.png")
            if mid_img is None:
                mid_img = avail  # fallback (single channel)
            if right_img is None:
                right_img = overlay_bgr

            self._set_panel_image(self.panel_mid, mid_img)
            self._set_panel_image(self.panel_right, right_img)

            # enable extract button
            self.btn_extract.config(state="normal")
            self.has_sampling_outputs = True

            print("[DONE] sampling finished.", flush=True)

        except Exception as e:
            messagebox.showerror("Sampling failed", str(e))
            raise

    def on_extract_clicked(self):
        if not self.has_sampling_outputs:
            messagebox.showwarning("Not ready", "Please run sampling first.")
            return

        def worker(q):
            timer0 = time.perf_counter()

            STAGE_TOTAL = 3  # 你现在的 extract 基本就 3 个大阶段：load/import/dapi/he（可自行改）

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

            DAPI_PATH = info["DAPI_path"]
            HE_PATH = info["HE_path"]
            DAPI_LEVEL = int(info["DAPI_level"])
            HE_LEVEL = int(info["HE_level"])

            lut_path = "glasbey_inverted.lut"
            lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)


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

            tick(5, f"DAPI_LEVEL={DAPI_LEVEL}, HE_LEVEL={HE_LEVEL}")
            tick(8, f"Output: {output_folder}")

            # ---------- imports from your project ----------
            report_stage(1, "Importing project utils")
            from my_utils import read_image, dapi_to_lut_rgb
            tick(12)

            report_stage(2, "Saving DAPI tiles")
            dapi_img2, _ = read_image(DAPI_PATH, keep_16bit=True, level=1)
            tick(20, f"Read DAPI level=1 shape={getattr(dapi_img2, 'shape', None)}")

            # lut 可能是 None：你可以在这里 fallback 到你默认 lut
            if lut is None:
                raise RuntimeError("lut is None; please load/build LUT (info['DAPI_lut'] or your default LUT).")

            dapi_rgb2 = dapi_to_lut_rgb(dapi_img2, lut, threshold=500)
            tick(30, "Applied LUT to DAPI")

            dapi_tiles = save_dapi_tiles(
                dapi_rgb2, tiles, str(output_folder),
                rescale_factor=2 ** (DAPI_LEVEL - 1)
            )
            tick(55, f"Saved DAPI tiles: {len(dapi_tiles) if hasattr(dapi_tiles, '__len__') else 'done'}")

            # overlays（你这边用 dapi_rgb 还是 dapi_rgb2：按你之前逻辑）
            # 你原代码里用的是 dapi_rgb（可能是 level=DAPI_LEVEL 的 LUT 图）
            # 这里如果没有 dapi_rgb，就用 dapi_rgb2 先顶着
            dapi_rgb_for_overlay = dapi_rgb2
            tick(65, "Saved DAPI overlay")

            # ------------------------------
            # 6. Save HE tiles using transformation
            # ------------------------------
            report_stage(3, "Saving HE tiles")

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

            he_img2, _ = read_image(HE_PATH, keep_16bit=True, level=1)
            tick(78, f"Read HE level=1 shape={getattr(he_img2, 'shape', None)}")

            he_tiles = save_he_tiles(
                he_img2, tiles, h_mat, str(output_folder),
                rescale_factor=2 ** (HE_LEVEL - 1),
                margin_ratio=0.2
            )
            tick(95, f"Saved HE tiles: {len(he_tiles) if hasattr(he_tiles, '__len__') else 'done'}")

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