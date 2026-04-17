"""
Stage 6 final alignment viewer.

This GUI displays:
1. DAPI image at the selected display level
2. H&E image at the selected display level
3. DAPI warped into H&E space
4. Cell centroids transformed into H&E space

It supports both homography-based and TPS-based registration results.
All registration parameters are stored in level-0 coordinates, while
the GUI renders images at a downsampled display level.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
from ome_types import from_tiff
import pandas as pd
from my_utils import read_image, dapi_to_lut_rgb, plot_cell_centroid

Image.MAX_IMAGE_PIXELS = None


# =============================
# CONFIG
# =============================
TILE_SIZE = (420, 420)
BG_COLOR = (240, 240, 240)


# =============================
# Geometry / transform helpers
# =============================
def apply_orientation_case(img, case_id):
    """
    Apply the stored DAPI orientation case to an image.

    Parameters
    ----------
    img : ndarray
        2D or 3D image.
    case_id : int
        Orientation code in [0, 7].

    Returns
    -------
    ndarray
        Re-oriented image for display.
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
    if case_id == 7:   # transverse
        return np.fliplr(np.rot90(img, k=1))
    raise ValueError(case_id)


# =============================
# Image / display helpers
# =============================
def add_watermark(img, text):
    """
    Add a simple text watermark to a PIL image.

    Used for the alternating GIF that compares:
    - final alignment
    - manual initial alignment
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    padding = 20
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return img

def build_stage_to_morph_from_ome(dapi_ome_tif_path: Path) -> np.ndarray:
    """
    Construct a 3x3 transform from stage coordinates (microns)
    to DAPI morphology image coordinates at level 0 using OME metadata.
    """
    md = from_tiff(path=str(dapi_ome_tif_path))
    origin_x = float(md.plates[0].well_origin_x)
    origin_y = float(md.plates[0].well_origin_y)
    physical_size_x = float(md.images[0].pixels.physical_size_x)  # micron / pixel
    physical_size_y = float(md.images[0].pixels.physical_size_y)

    H = np.array([
        [1.0 / physical_size_x, 0.0, origin_x],
        [0.0, 1.0 / physical_size_y, origin_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    print("[DEBUG] stage_to_morph from OME:",
          "origin=", (origin_x, origin_y),
          "px_size=", (physical_size_x, physical_size_y), flush=True)
    print("[DEBUG] stage_to_morph=\n", H, flush=True)
    return H

def cv2_to_pil(img):
    if img is None:
        return None
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fit_to_tile(pil_img, size=TILE_SIZE, bg=BG_COLOR):
    canvas = Image.new("RGB", size, bg)
    if pil_img is None:
        return canvas
    pil_img = ImageOps.contain(pil_img, size)
    x = (size[0] - pil_img.width) // 2
    y = (size[1] - pil_img.height) // 2
    canvas.paste(pil_img, (x, y))
    return canvas

def draw_points(img, pts, color=(0, 255, 0), r=4, max_points=6000, name=""):
    """
    Draw 2D points on an image.

    Parameters
    ----------
    img : ndarray
        Target image.
    pts : (N, 2) array
        Point coordinates in the same coordinate system as `img`.
    max_points : int
        Maximum number of points to draw directly. If exceeded,
        points are uniformly subsampled for GUI responsiveness.
    """
    out = img.copy()
    h, w = img.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        print(f"[DEBUG] draw_points({name}): empty", flush=True)
        return out

    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
        print(f"[DEBUG] draw_points({name}): subsample -> {len(pts)}", flush=True)

    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    print(f"[DEBUG] draw_points({name}) img=(h={h}, w={w}) pts={len(pts)} valid={valid.sum()}", flush=True)

    for (x, y) in pts[valid]:
        cv2.circle(out, (int(x), int(y)), int(r), color, -1)

    return out

def draw_points_two_groups(
    img,
    pts_selected,
    pts_other,
    color_selected=(0, 255, 0),   # green
    color_other=(0, 165, 255),    # orange
    r_selected=5,
    r_other=4,
    max_points=6000,
    name=""
):
    """
    Draw two groups of points on an image.

    Parameters
    ----------
    img : ndarray
        Target image.
    pts_selected : (N, 2) array
        Selected / balanced points.
    pts_other : (M, 2) array
        Non-selected / imbalanced points.
    """
    out = img.copy()
    h, w = img.shape[:2]

    def _prepare(pts, group_name):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            print(f"[DEBUG] draw_points_two_groups({name}:{group_name}): empty", flush=True)
            return pts

        if len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
            print(
                f"[DEBUG] draw_points_two_groups({name}:{group_name}): subsample -> {len(pts)}",
                flush=True
            )
        return pts

    pts_selected = _prepare(pts_selected, "selected")
    pts_other = _prepare(pts_other, "other")

    def _draw(pts, color, radius, group_name):
        if pts.size == 0:
            return
        valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        print(
            f"[DEBUG] draw_points_two_groups({name}:{group_name}) "
            f"img=(h={h}, w={w}) pts={len(pts)} valid={valid.sum()}",
            flush=True
        )
        for (x, y) in pts[valid]:
            cv2.circle(out, (int(x), int(y)), int(radius), color, -1)

    # draw orange first, then green on top
    _draw(pts_other, color_other, r_other, "other")
    _draw(pts_selected, color_selected, r_selected, "selected")

    return out


# =============================
# transformation helpers
# =============================
def load_transform_mode(path: Path):
    data = json.load(open(path, "r"))
    transform_type = str(data.get("transform_type", "")).strip().lower()

    if transform_type == "affine":
        return "affine"
    elif transform_type == "homography":
        return "homography"
    elif transform_type in {"tps", "local_tps"}:
        return "tps"

    # backward compatibility
    if "inverse_tps" in data and "forward_tps" in data:
        return "tps"
    elif "homography_3x3" in data:
        return "homography"
    elif "affine_2x3" in data or "affine_3x3" in data:
        return "affine"
    elif "initial_homography_3x3" in data:
        return "tps"

    raise ValueError(f"Unknown transform json format. Keys: {list(data.keys())}")

def load_affine(path: Path):
    data = json.load(open(path, "r"))
    print(f"[DEBUG] affine json keys: {list(data.keys())}", flush=True)

    A2 = None
    A3 = None

    if "affine_3x3" in data:
        A3 = np.array(data["affine_3x3"], dtype=np.float32)
        print(f"[DEBUG] affine_3x3 shape={A3.shape}", flush=True)

    if "affine_2x3" in data:
        A2 = np.array(data["affine_2x3"], dtype=np.float32)
        print(f"[DEBUG] affine_2x3 shape={A2.shape}", flush=True)

    # support stage5 affine output saved as homography_3x3
    if A3 is None and "homography_3x3" in data:
        A3 = np.array(data["homography_3x3"], dtype=np.float32)
        print(f"[DEBUG] homography_3x3 used as affine_3x3, shape={A3.shape}", flush=True)

    if A3 is not None and A3.shape == (3, 3):
        A2 = A3[:2, :].copy().astype(np.float32)
        return A2, A3.astype(np.float32)

    if A2 is not None and A2.shape == (2, 3):
        A3 = np.vstack([A2, [0, 0, 1]]).astype(np.float32)
        return A2, A3

    raise ValueError(
        f"[ERROR] Invalid affine shapes.\n"
        f"affine_2x3={None if A2 is None else A2.shape}\n"
        f"affine_3x3={None if A3 is None else A3.shape}\n"
        f"Raw affine_2x3 value head={str(data.get('affine_2x3'))[:200]}"
    )

def load_homography(path: Path):
    """
    Load perspective transform (homography) from json.
    Returns:
      H2 (2x3), H3 (3x3)
    Notes:
      - For true perspective warp, ALWAYS use H3 with cv2.warpPerspective.
      - H2 is provided only for backward compatibility; it drops the 3rd row.
    """
    data = json.load(open(path, "r"))
    print(f"[DEBUG] homography json keys: {list(data.keys())}", flush=True)

    H3 = None

    # ---- accept a few possible keys ----
    for k in ("homography_3x3", "H_mat", "H", "matrix_3x3"):
        if k in data:
            H3 = np.array(data[k], dtype=np.float32)
            print(f"[DEBUG] {k} shape={H3.shape}", flush=True)
            break

    if H3 is None:
        raise ValueError(
            f"[ERROR] No homography found in json.\n"
            f"Expected one of keys: homography_3x3 / H_mat / H / matrix_3x3\n"
            f"Got keys: {list(data.keys())}"
        )

    if H3.shape != (3, 3):
        raise ValueError(
            f"[ERROR] Invalid homography shape: {H3.shape}, expected (3,3).\n"
            f"Raw value head={str(H3)[:200]}"
        )

    # Optional sanity: ensure bottom-right not 0 (scale ambiguity is OK, but all-zero is bad)
    if abs(float(H3[2, 2])) < 1e-12:
        print("[WARN] H[2,2] is ~0; homography scale may be unusual. Still returning H.", flush=True)

    # Provide 2x3 "compat" affine-like slice (NOT a true perspective transform)
    H2 = H3[:2, :].copy().astype(np.float32)

    return H2, H3

def load_tps(path: Path, direction="forward"):
    data = json.load(open(path, "r"))
    print(f"[DEBUG] TPS json keys: {list(data.keys())}", flush=True)

    if direction == "forward":
        key = "forward_tps"
    elif direction == "inverse":
        key = "inverse_tps"
    else:
        raise ValueError("direction must be 'forward' or 'inverse'")

    if key not in data:
        raise ValueError(f"[ERROR] Missing {key} in json. Keys: {list(data.keys())}")

    d = data[key]

    ctrl = np.array(d["control_points_src"], dtype=np.float64)
    w = np.array(d["weights"], dtype=np.float64)
    A2x3 = np.array(d["affine_2x3"], dtype=np.float64)

    if ctrl.ndim != 2 or ctrl.shape[1] != 2:
        raise ValueError(f"[ERROR] ctrl shape invalid: {ctrl.shape}")
    if w.shape[0] != ctrl.shape[0] or w.shape[1] != 2:
        raise ValueError(f"[ERROR] weights shape invalid: {w.shape}, ctrl={ctrl.shape}")
    if A2x3.shape != (2, 3):
        raise ValueError(f"[ERROR] affine_2x3 invalid: {A2x3.shape}")

    a = np.zeros((3, 2), dtype=np.float64)
    a[0, :] = A2x3[:, 0]
    a[1, :] = A2x3[:, 1]
    a[2, :] = A2x3[:, 2]

    return {
        "ctrl": ctrl,
        "w": w,
        "a": a,
    }

def transform_coordinates(coords_xy, homography_3x3):
    """
    coords_xy: (N,2) float
    homography_3x3: (3,3)
    return: (N,2)
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(coords_xy, np.asarray(homography_3x3, dtype=np.float32))
    return out[:, 0, :]

def transform_xy_affine(xy: np.ndarray, A2x3: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(xy, A2x3)
    return out[:, 0, :]

def transform_xy_perspective(xy: np.ndarray, H3x3: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    H = np.asarray(H3x3, dtype=np.float32)

    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy_h = np.concatenate([xy, ones], axis=1)
    proj = (H @ xy_h.T).T
    w = np.clip(proj[:, 2:3], 1e-8, None)
    return (proj[:, :2] / w).astype(np.float32)

def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r2, dtype=np.float64)
    mask = r2 > 0
    out[mask] = r2[mask] * np.log(r2[mask])
    return out

def apply_tps_points(xy: np.ndarray, model: dict) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    ctrl = model["ctrl"]
    w = model["w"]
    a = model["a"]

    diff = xy[:, None, :] - ctrl[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    K = _tps_kernel(r2)
    P = np.concatenate([np.ones((xy.shape[0], 1)), xy], axis=1)
    return (K @ w + P @ a).astype(np.float32)

def infer_scale_from_info(info: dict):
    """
    Try to infer micron->HE pixel scale from images_info.json.
    You can customize the key to your real pipeline.
    """
    for k in ["micron_to_he_scale", "microns_to_he_scale", "scale", "he_scale"]:
        if k in info:
            return float(info[k])
    return None


# =============================
# Cell coordinate conversion
# =============================
def cells_to_he_pixels(cell_csv_gz: Path, transform_json: Path, dapi_ome_tif: Path):
    """
    Read cells.csv.gz (assumed stage micron coords) and convert:
      stage(micron) -> DAPI(level0 px) -> HE(level0 px)

    Supports:
      - affine
      - homography
      - tps
    """
    print(f"[INFO] reading cells: {cell_csv_gz}", flush=True)
    df = pd.read_csv(cell_csv_gz, index_col=0)

    need_cols = {"x_centroid", "y_centroid"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"cells.csv.gz missing columns {need_cols}. Got cols={list(df.columns)[:20]}...")

    xy_stage = df[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    print(f"[DEBUG] cells rows={len(df)} xy(stage micron) min={xy_stage.min(axis=0)} max={xy_stage.max(axis=0)}", flush=True)

    # stage(micron) -> dapi(level0 px)
    H_stage2dapi = build_stage_to_morph_from_ome(dapi_ome_tif)
    xy_dapi = transform_xy_perspective(xy_stage, H_stage2dapi)
    print(f"[DEBUG] xy_dapi(level0 px) min={xy_dapi.min(axis=0)} max={xy_dapi.max(axis=0)}", flush=True)

    mode = load_transform_mode(transform_json)
    print(f"[INFO] cell transform mode = {mode}", flush=True)

    if mode == "affine":
        A2, _ = load_affine(transform_json)
        xy_he = transform_xy_affine(xy_dapi, A2)
    elif mode == "homography":
        _, H3 = load_homography(transform_json)
        xy_he = transform_xy_perspective(xy_dapi, H3)
    elif mode == "tps":
        tps_fwd = load_tps(transform_json, direction="forward")
        xy_he = apply_tps_points(xy_dapi, tps_fwd)
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")

    print(f"[DEBUG] xy_he(level0 px) min={xy_he.min(axis=0)} max={xy_he.max(axis=0)}", flush=True)

    df.loc[:, "x_centroid"] = xy_he[:, 0]
    df.loc[:, "y_centroid"] = xy_he[:, 1]
    return df

# =============================
# GUI application
# =============================
class FinalAlignmentApp(tk.Tk):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.title("Stage 6 — Final Alignment Viewer")
        self.run_dir = run_dir

        # -------- metadata --------
        self.info = json.load(open(run_dir / "images_info.json"))
        self.case_id = int(self.info.get("DAPI_orientation_case", 0))

        # -------- display settings --------
        self.display_level = int(self.info["DAPI_level"])
        self.display_scale = float(2 ** self.display_level)
        print(f"[INFO] DISPLAY_LEVEL = {self.display_level}", flush=True)
        print(f"[INFO] DISPLAY_SCALE = {self.display_scale}", flush=True)

        # -------- cache paths --------
        self.cache = {
            # base
            "dapi_base": self.run_dir / f"6_cache_dapi_base_L{self.display_level}.png",
            "he_base": self.run_dir / f"6_cache_he_base_L{self.display_level}.png",

            # after load (with keypoints)
            "dapi_kp": self.run_dir / f"6_cache_dapi_kp_L{self.display_level}.png",
            "he_kp": self.run_dir / f"6_cache_he_kp_L{self.display_level}.png",

            # overlays
            "overlay": self.run_dir / f"6_overlay_final_L{self.display_level}.png",
            "manual": self.run_dir / f"6_overlay_manual_L{self.display_level}.png",
            "alternating": self.run_dir / f"6_alternating_L{self.display_level}.gif",
            "cells": self.run_dir / f"6_cells_centroids_L{self.display_level}.png",
        }

        # -------- state --------
        self.alignment_loaded = False
        self.H3 = None
        self.warped_dapi = None
        self.he_dapi_overlay = None

        self.dapi_pts0 = None
        self.he_pts0 = None
        self.selected_indices = None
        self.other_indices = None

        self.cells_df = None
        self.cells_pts_lvl2 = None

        # -------- load base images --------
        # DAPI
        dapi_lut_thr = int(self.info.get("DAPI_LUT_threshold", 300))
        print(f"[INFO] Using DAPI_LUT_threshold={dapi_lut_thr} (from images_info.json)", flush=True)

        self.dapi_rgb = self._load_or_build_dapi_base()
        self.he_rgb, self.he16 = self._load_or_build_he_base()

        print("[DEBUG] DAPI image shape:", self.dapi_rgb.shape, flush=True)
        print("[DEBUG] HE   image shape:", self.he_rgb.shape, flush=True)

        self.cells_df = None
        self.cells_pts_lvl2 = None
        mid = tk.Frame(self)
        mid.pack(padx=10, pady=10)

        self.panels = []
        titles = [
            "DAPI (LUT)",
            "H&E",
            "DAPI Overlay on H&E",
            "Cell centroids on H&E"
        ]

        for i, t in enumerate(titles):
            f = tk.Frame(mid)
            tk.Label(f, text=t, font=("Helvetica", 11, "bold")).pack()
            lbl = tk.Label(f)
            lbl.pack()
            f.grid(row=0, column=i, padx=6)
            f.lbl = lbl
            self.panels.append(f)

        # click panels to enlarge (always bind; handler decides what to show)
        for i in range(4):
            self.panels[i].lbl.bind("<Button-1>", lambda e, ii=i: self.on_panel_click(ii))

        self.panels[0].lbl.configure(cursor="hand2")
        self.panels[1].lbl.configure(cursor="hand2")
        self.panels[2].lbl.configure(cursor="hand2")
        self.panels[3].lbl.configure(cursor="hand2")

        btns = tk.Frame(self)
        btns.pack(fill="x", padx=10, pady=6)

        tk.Button(
            btns, text="Load alignment",
            command=self.load_alignment_and_keypoints
        ).pack(side="left", expand=True, fill="x")

        tk.Button(
            btns, text="Toggle H&E / Overlay",
            command=self.toggle_floating
        ).pack(side="left", expand=True, fill="x")

        tk.Button(
            btns, text="Load cell data (cells.csv.gz)",
            command=self.load_cell_data
        ).pack(side="left", expand=True, fill="x")

        # init display: panel1/2 show base images; panel3/4 placeholders
        self._panel3_show_overlay = True
        self.refresh_images_initial()
        self.minsize(self.winfo_width(), self.winfo_height())

    # --------------------------
    # initial refresh (no alignment)
    # --------------------------
    def refresh_images_initial(self):
        """
        Initialize the 4-panel view before alignment is loaded.

        Panels:
        0. DAPI
        1. H&E
        2. Final overlay if cached, otherwise placeholder
        3. Cell overlay if cached, otherwise placeholder
        """
        # panel1/2 base
        self._set_panel(0, apply_orientation_case(self.dapi_rgb, self.case_id))
        self._set_panel(1, self.he_rgb)

        # panel3: try cached overlay first
        p = self.cache["overlay"]
        img3 = self._imread(p)
        if img3 is not None:
            self.he_dapi_overlay = img3
            self._panel3_tkimg_he = self._make_tkimg(self.he_rgb)
            self._panel3_tkimg_overlay = self._make_tkimg(img3)
            self._panel3_show_overlay = True
            self._update_panel3_fast()
        else:
            self._set_placeholder(2, "Click 'Load alignment'")

        # panel4: try cached cells overlay first
        p = self.cache["cells"]
        img4 = self._imread(p)
        if img4 is not None:
            self._set_panel(3, img4)
        else:
            self._set_placeholder(3, "Click 'Load cell data'")

    # --------------------------
    # refresh after alignment loaded
    # --------------------------
    def refresh_images_after_alignment(self):
        """
        Refresh all panels after alignment has been loaded.
        Cached keypoint panels are reused when available.
        """
        # panel0: DAPI kp
        p = self.cache["dapi_kp"]
        pts_sel = self.dapi_pts0[
                      self.selected_indices] / self.display_scale if self.selected_indices is not None else self.dapi_pts0 / self.display_scale
        pts_other = self.dapi_pts0[
                        self.other_indices] / self.display_scale if self.other_indices is not None else np.empty((0, 2),
                                                                                                                 dtype=np.float32)

        dapi_kp = draw_points_two_groups(
            self.dapi_rgb,
            pts_selected=pts_sel,
            pts_other=pts_other,
            color_selected=(0, 255, 0),
            color_other=(0, 165, 255),
            r_selected=6,
            r_other=5,
            name="DAPI nuclei"
        )
        self._imsave(p, dapi_kp)
        self._set_panel(0, apply_orientation_case(dapi_kp, self.case_id))

        # panel1: HE kp
        p = self.cache["he_kp"]
        pts_sel = self.he_pts0[
                      self.selected_indices] / self.display_scale if self.selected_indices is not None else self.he_pts0 / self.display_scale
        pts_other = self.he_pts0[
                        self.other_indices] / self.display_scale if self.other_indices is not None else np.empty((0, 2),
                                                                                                                 dtype=np.float32)

        he_kp = draw_points_two_groups(
            self.he_rgb,
            pts_selected=pts_sel,
            pts_other=pts_other,
            color_selected=(0, 255, 0),
            color_other=(0, 165, 255),
            r_selected=6,
            r_other=5,
            name="HE nuclei"
        )
        self._imsave(p, he_kp)
        self._set_panel(1, he_kp)
        # panel2/panel3 keep as is...
        self._update_panel3_fast()

        # panel4: cells (unchanged)
        p = self.cache["cells"]
        if p.exists():
            img4 = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img4 is None:
                self._set_placeholder(3, "Failed to load cells overlay")
            else:
                self._set_panel(3, img4)
        else:
            self._set_placeholder(3, "Click 'Load cell data'")

    def load_alignment_and_keypoints(self):
        """
        Load nuclei keypoints and the final DAPI-to-H&E transform,
        then build display-space overlays.

        This function:
        1. loads nuclei centroids in level-0 coordinates
        2. loads either homography or TPS transform
        3. warps DAPI into H&E display space
        4. builds final overlay and manual-alignment overlay
        5. caches panel images for fast GUI refresh
        """

        # ---- local TPS helpers ----
        def load_tps(path: Path, direction="forward"):
            """
            direction:
              - 'forward' : dapi -> he
              - 'inverse' : he   -> dapi
            """
            data = json.load(open(path, "r"))
            print(f"[DEBUG] TPS json keys: {list(data.keys())}", flush=True)

            if direction == "forward":
                key = "forward_tps"
            elif direction == "inverse":
                key = "inverse_tps"
            else:
                raise ValueError("direction must be 'forward' or 'inverse'")

            if key not in data:
                raise ValueError(f"[ERROR] Missing {key} in json. Keys: {list(data.keys())}")

            d = data[key]

            ctrl = np.array(d["control_points_src"], dtype=np.float64)
            w = np.array(d["weights"], dtype=np.float64)
            A2x3 = np.array(d["affine_2x3"], dtype=np.float64)

            if ctrl.ndim != 2 or ctrl.shape[1] != 2:
                raise ValueError(f"[ERROR] ctrl shape invalid: {ctrl.shape}")
            if w.shape[0] != ctrl.shape[0] or w.shape[1] != 2:
                raise ValueError(f"[ERROR] weights shape invalid: {w.shape}, ctrl={ctrl.shape}")
            if A2x3.shape != (2, 3):
                raise ValueError(f"[ERROR] affine_2x3 invalid: {A2x3.shape}")

            # internal TPS affine format: (3,2)
            a = np.zeros((3, 2), dtype=np.float64)
            a[0, :] = A2x3[:, 0]
            a[1, :] = A2x3[:, 1]
            a[2, :] = A2x3[:, 2]

            return {
                "ctrl": ctrl,
                "w": w,
                "a": a,
            }

        def _tps_kernel(r2: np.ndarray) -> np.ndarray:
            out = np.zeros_like(r2, dtype=np.float64)
            mask = r2 > 0
            out[mask] = r2[mask] * np.log(r2[mask])
            return out

        def apply_tps(xy: np.ndarray, model: dict) -> np.ndarray:
            xy = np.asarray(xy, dtype=np.float64)
            ctrl = model["ctrl"]
            w = model["w"]
            a = model["a"]

            diff = xy[:, None, :] - ctrl[None, :, :]
            r2 = np.sum(diff * diff, axis=2)
            K = _tps_kernel(r2)

            P = np.concatenate([np.ones((xy.shape[0], 1)), xy], axis=1)  # (N,3)
            return K @ w + P @ a

        def load_transform_mode(path: Path):
            data = json.load(open(path, "r"))

            transform_type = str(data.get("transform_type", "")).strip().lower()

            if transform_type == "affine":
                return "affine"
            elif transform_type == "homography":
                return "homography"
            elif transform_type in {"tps", "local_tps"}:
                return "tps"

            # fallback for older files
            if "inverse_tps" in data and "forward_tps" in data:
                return "tps"
            elif "homography_3x3" in data:
                return "homography"
            elif "initial_homography_3x3" in data:
                return "tps"

            raise ValueError(f"[ERROR] Unknown transform json format. Keys: {list(data.keys())}")

        def build_tps_remap_for_display(tps_model, out_shape_hw, display_scale):
            """
            Build map_x, map_y in DISPLAY space for cv2.remap.

            Parameters
            ----------
            tps_model : dict
                inverse TPS model: HE(level0) -> DAPI(level0)
            out_shape_hw : (H, W)
                output display image shape, typically he_rgb.shape[:2]
            display_scale : float
                level0_px / display_px
            """
            H, W = out_shape_hw
            s = float(display_scale)

            gx, gy = np.meshgrid(
                np.arange(W, dtype=np.float32),
                np.arange(H, dtype=np.float32)
            )
            pts_disp = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

            # display HE px -> level0 HE px
            pts_he_lvl0 = pts_disp * s

            # inverse TPS: HE(level0) -> DAPI(level0)
            pts_dapi_lvl0 = apply_tps(pts_he_lvl0, tps_model)

            # level0 DAPI px -> display DAPI px
            pts_dapi_disp = pts_dapi_lvl0 / s

            map_x = pts_dapi_disp[:, 0].reshape(H, W).astype(np.float32)
            map_y = pts_dapi_disp[:, 1].reshape(H, W).astype(np.float32)

            return map_x, map_y

        try:
            # ---- load nuclei centroids (level 0) ----
            nuclei_path = self.run_dir / "nuclei_patches/nuclei_centroids_global.json"
            nuclei = json.load(open(nuclei_path))['data']
            self.dapi_pts0 = np.array([x["dapi_centroid_global"] for x in nuclei], np.float32)
            self.he_pts0   = np.array([x["he_centroid_global"]   for x in nuclei], np.float32)
            print(f"[INFO] Loaded nuclei: {len(self.dapi_pts0)}", flush=True)

            # ---- load final transform ----
            tf_path = self.run_dir / "dapi_to_he_homography_level0.json"
            mode = load_transform_mode(tf_path)

            tf_data = json.load(open(tf_path, "r"))
            sampling = tf_data.get("sampling", {})
            selected_idx = sampling.get("selected_indices_from_original", None)

            n_pts = len(self.dapi_pts0)
            if selected_idx is None:
                # no balanced sampling info -> treat all as selected
                self.selected_indices = np.arange(n_pts, dtype=int)
                self.other_indices = np.array([], dtype=int)
                print("[INFO] No selected_indices_from_original found; treating all points as selected.", flush=True)
            else:
                self.selected_indices = np.array(selected_idx, dtype=int)
                self.selected_indices = self.selected_indices[
                    (self.selected_indices >= 0) & (self.selected_indices < n_pts)
                ]

                mask = np.zeros(n_pts, dtype=bool)
                mask[self.selected_indices] = True
                self.other_indices = np.where(~mask)[0]

                print(
                    f"[INFO] Balanced sampling: selected={len(self.selected_indices)} "
                    f"other={len(self.other_indices)} total={n_pts}",
                    flush=True
                )

            # ---- warp DAPI into H&E display space ----
            s = float(self.display_scale)
            if mode == "affine":
                _, self.H3 = load_affine(tf_path)

                S = np.array([[s, 0, 0],
                              [0, s, 0],
                              [0, 0, 1]], dtype=np.float32)
                S_inv = np.array([[1 / s, 0, 0],
                                  [0, 1 / s, 0],
                                  [0, 0, 1]], dtype=np.float32)

                H_disp = (S_inv @ self.H3 @ S).astype(np.float32)

                self.warped_dapi = cv2.warpPerspective(
                    self.dapi_rgb,
                    H_disp,
                    (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            elif mode == "homography":
                _, self.H3 = load_homography(tf_path)

                S = np.array([[s, 0, 0],
                              [0, s, 0],
                              [0, 0, 1]], dtype=np.float32)
                S_inv = np.array([[1 / s, 0, 0],
                                  [0, 1 / s, 0],
                                  [0, 0, 1]], dtype=np.float32)

                H_disp = (S_inv @ self.H3 @ S).astype(np.float32)

                self.warped_dapi = cv2.warpPerspective(
                    self.dapi_rgb,
                    H_disp,
                    (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            elif mode == "tps":
                # TPS path: for image warp we must use inverse TPS (HE -> DAPI)
                self.tps_inv = load_tps(tf_path, direction="inverse")

                map_x, map_y = build_tps_remap_for_display(
                    self.tps_inv,
                    out_shape_hw=self.he_rgb.shape[:2],
                    display_scale=self.display_scale
                )

                self.warped_dapi = cv2.remap(
                    self.dapi_rgb,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )

            else:
                raise ValueError(f"Unknown transform mode: {mode}")

            # ---- save final overlay ----
            self.he_dapi_overlay = cv2.addWeighted(self.he_rgb, 0.7, self.warped_dapi, 0.8, 0)
            cv2.imwrite(str(self.cache["overlay"]), self.he_dapi_overlay)

            # ---- build manual-initial-alignment overlay ----
            data_manual = json.load(open(self.run_dir / "manual_initial_alignment.json", "r"))
            H3m = np.array(data_manual['H_mat_level_0'], dtype=np.float32)
            Sm = np.array([[s, 0, 0],
                          [0, s, 0],
                          [0, 0, 1]], dtype=np.float32)
            Sm_inv = np.array([[1 / s, 0, 0],
                              [0, 1 / s, 0],
                              [0, 0, 1]], dtype=np.float32)
            Hm_disp = (Sm_inv @ H3m @ Sm).astype(np.float32)
            warped_dapi_manual = cv2.warpPerspective(
                self.dapi_rgb,
                Hm_disp,
                (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            he_dapi_overlay_manual = cv2.addWeighted(self.he_rgb, 0.7, warped_dapi_manual, 0.8, 0)
            cv2.imwrite(str(self.cache["manual"]), he_dapi_overlay_manual)

            # ---- compare against manual initial alignment ----
            img1_w = add_watermark(Image.fromarray(self.he_dapi_overlay[..., ::-1]), "Final alignment")
            img2_w = add_watermark(Image.fromarray(he_dapi_overlay_manual[..., ::-1]), "Manual initial alignment")
            img1_w.save(
                str(self.cache["alternating"]),
                save_all=True,
                append_images=[img2_w],
                duration=1000,
                loop=0
            )

            # ---- build cached keypoint panels ----
            # 1) DAPI kp
            p_dapi_kp = self.cache["dapi_kp"]
            dapi_kp = self._imread(p_dapi_kp)
            if dapi_kp is None:
                pts = self.dapi_pts0 / self.display_scale
                dapi_kp = draw_points(self.dapi_rgb, pts, color=(0, 255, 0), r=6, name="DAPI nuclei")
                self._imsave(p_dapi_kp, dapi_kp)

            # 2) HE kp
            p_he_kp = self.cache["he_kp"]
            he_kp = self._imread(p_he_kp)
            if he_kp is None:
                pts = self.he_pts0 / self.display_scale
                he_kp = draw_points(self.he_rgb, pts, color=(0, 255, 0), r=6, name="HE nuclei")
                self._imsave(p_he_kp, he_kp)

            # ---- cache panel images for quick toggle ----
            self._panel3_tkimg_he = self._make_tkimg(self.he_rgb)
            self._panel3_tkimg_overlay = self._make_tkimg(self.he_dapi_overlay)
            self._panel3_show_overlay = True

            self.alignment_loaded = True
            self.refresh_images_after_alignment()

        except Exception as e:
            messagebox.showerror("Load alignment failed", str(e), parent=self)
            raise

    # --------------------------
    # panel helpers (same as before)
    # --------------------------
    def _imread(self, p: Path):
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return img
        return None

    def _imsave(self, p: Path, img_bgr: np.ndarray):
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img_bgr)

    def _load_or_build_dapi_base(self):
        """
        Load cached display-level DAPI image if available,
        otherwise read the DAPI image at DISPLAY_LEVEL,
        apply LUT coloring, and cache the result.
        """
        img = self._imread(self.cache["dapi_base"])
        if img is not None:
            return img

        dapi_lut_thr = int(self.info.get("DAPI_LUT_threshold", 300))
        dapi16, _ = read_image(self.info["DAPI_path"], keep_16bit=True, level=self.display_level, channel="dapi")
        print(dapi16.max())
        lut = np.fromfile("glasbey_inverted.lut", dtype=np.uint8).reshape(256, 3)
        print(dapi_lut_thr)
        rgb = dapi_to_lut_rgb(dapi16, lut, threshold=dapi_lut_thr)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._imsave(self.cache["dapi_base"], bgr)
        return bgr

    def _load_or_build_he_base(self):
        """
        Load cached display-level H&E image if available,
        otherwise read H&E at DISPLAY_LEVEL, normalize to uint8,
        and cache the display image.
        """
        img = self._imread(self.cache["he_base"])
        if img is not None:            return img, None

        he16, _ = read_image(self.info["HE_path"], keep_16bit=False, level=self.display_level,  channel="he")
        he8 = cv2.normalize(he16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if he8.ndim == 2:
            bgr = cv2.cvtColor(he8, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(he8, cv2.COLOR_RGB2BGR)
        self._imsave(self.cache["he_base"], bgr)
        return bgr, he16

    def _make_tkimg(self, cv_img):
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil)
        return ImageTk.PhotoImage(tile)

    def _update_panel3_fast(self):
        if self.he_dapi_overlay is None:
            self._set_placeholder(2, "Overlay Not Available")
            return
        if not hasattr(self, "_panel3_tkimg_he") or self._panel3_tkimg_he is None:
            self._panel3_tkimg_he = self._make_tkimg(self.he_rgb)
        if not hasattr(self, "_panel3_tkimg_overlay") or self._panel3_tkimg_overlay is None:
            self._panel3_tkimg_overlay = self._make_tkimg(self.he_dapi_overlay)
        tkimg = self._panel3_tkimg_overlay if self._panel3_show_overlay else self._panel3_tkimg_he
        self.panels[2].lbl.configure(image=tkimg)
        self.panels[2].lbl.image = tkimg

    def _set_panel(self, idx, cv_img):
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil)
        tkimg = ImageTk.PhotoImage(tile)
        self.panels[idx].lbl.configure(image=tkimg)
        self.panels[idx].lbl.image = tkimg

    def _set_placeholder(self, idx, text):
        img = np.full((TILE_SIZE[1], TILE_SIZE[0], 3), BG_COLOR, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        color = (80, 80, 80)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (TILE_SIZE[0] - tw) // 2
        y = (TILE_SIZE[1] + th) // 2
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        self._set_panel(idx, img)

    def toggle_floating(self):
        """Toggle panel 3 between H&E and final DAPI-on-H&E overlay."""
        if self.he_dapi_overlay is None:
            messagebox.showinfo(
                "Not available",
                "No overlay available.\nIf you want to generate one, click 'Load alignment'.",
                parent=self
            )
            return

        self._panel3_show_overlay = not self._panel3_show_overlay
        self._update_panel3_fast()

    def load_cell_data(self):
        """
        Load a cells.csv.gz file, transform cell centroids into H&E space,
        and generate a cached cell-centroid overlay for panel 4.
        """
        # --- pick file ---
        self.focus_force()
        self.update()

        path = filedialog.askopenfilename(
            parent=self,
            title="Select cells.csv.gz",
            filetypes=[
                ("cells.csv.gz (recommended)", "*.csv.gz"),
                ("Gzipped CSV", "*.gz"),
                ("All files", "*.*"),
            ],
            initialfile="cells.csv.gz",
            defaultextension=".csv.gz",
        )

        if not path:
            return

        path = Path(path)
        if path.name != "cells.csv.gz":
            messagebox.showerror(
                "Invalid file",
                "You must select a file named exactly:\n\ncells.csv.gz",
                parent=self
            )
            return

        # --- load & transform ---
        try:
            transformation_json = self.run_dir / "dapi_to_he_homography_level0.json"
            df = cells_to_he_pixels(
                path,
                transform_json=transformation_json,
                dapi_ome_tif=Path(self.info["DAPI_path"]),
            )
            print("[DEBUG] HE image level2 shape:", self.he_rgb.shape, flush=True)
            print("[DEBUG] first 5 transformed cells (he_l0):\n", df[["x_centroid", "y_centroid"]].head(), flush=True)

            # downsample to DISPLAY_LEVEL for GUI overlay
            df_gui = df.copy()
            df_gui["x_centroid"] /= self.display_scale
            df_gui["y_centroid"] /= self.display_scale

            self.cells_df = df  # keep level0 px
            self.cells_pts_lvl2 = df_gui[["x_centroid", "y_centroid"]].to_numpy(np.float32)

            # --- plot + save (uses he16 at DISPLAY_LEVEL) ---
            out_png = self.cache["cells"]
            print(f"[INFO] plotting cell centroids -> {out_png}", flush=True)
            if self.he16 is None:
                self.he16, _ = read_image(self.info["HE_path"], keep_16bit=True, level=self.display_level)
            plot_cell_centroid(
                df_gui,
                he=self.he16,
                color="red",
                save_name=str(out_png),
                save_fig=True,
                dot_size=max(1, 5 / (2 ** max(self.display_level - 2, 0)))
            )
            messagebox.showinfo(
                "Loaded",
                f"Loaded and transformed:\n{path}\n\nSaved:\n{out_png}",
                parent=self
            )
            img4 = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
            if img4 is not None:
                self._set_panel(3, img4)

        except Exception as e:
            messagebox.showerror("Load failed", str(e), parent=self)
            raise

    def show_large_view(self, title: str, bgr_img: np.ndarray):
        """
        Show a larger window for the given BGR image.
        """
        if bgr_img is None:
            return

        h, w = bgr_img.shape[:2]

        # limit window size to screen
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        max_w = int(sw * 0.85)
        max_h = int(sh * 0.85)

        scale = min(1.0, max_w / w, max_h / h)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if scale < 1.0:
            pil = pil.resize((disp_w, disp_h), Image.BILINEAR)

        tk_img = ImageTk.PhotoImage(pil)

        win = tk.Toplevel(self)
        win.title(title)
        win.resizable(True, True)

        lbl = tk.Label(win, image=tk_img, bg="black")
        lbl.image = tk_img
        lbl.pack(expand=True, fill="both")

        # center
        win.update_idletasks()
        x = (sw - disp_w) // 2
        y = (sh - disp_h) // 2
        win.geometry(f"{disp_w}x{disp_h}+{x}+{y}")

        win.bind("<Escape>", lambda e: win.destroy())

    def on_panel_click(self, idx: int):
        """
        Open a larger view of the selected panel.

        Panel mapping
        -------------
        0 : DAPI
        1 : H&E
        2 : Overlay or H&E toggle view
        3 : Cell-centroid overlay
        """
        if idx == 0:
            if self.alignment_loaded and self.dapi_pts0 is not None:
                pts_sel = self.dapi_pts0[self.selected_indices] / self.display_scale if self.selected_indices is not None else self.dapi_pts0 / self.display_scale
                pts_other = self.dapi_pts0[self.other_indices] / self.display_scale if self.other_indices is not None else np.empty((0, 2), dtype=np.float32)

                img = draw_points_two_groups(
                    self.dapi_rgb,
                    pts_selected=pts_sel,
                    pts_other=pts_other,
                    color_selected=(0, 255, 0),
                    color_other=(0, 165, 255),
                    r_selected=6,
                    r_other=5,
                    name="DAPI nuclei (large)"
                )
            else:
                img = self.dapi_rgb.copy()
            img = apply_orientation_case(img, self.case_id)
            self.show_large_view("DAPI", img)
            return

        if idx == 1:
            if self.alignment_loaded and self.he_pts0 is not None:
                pts_sel = self.he_pts0[self.selected_indices] / self.display_scale if self.selected_indices is not None else self.he_pts0 / self.display_scale
                pts_other = self.he_pts0[self.other_indices] / self.display_scale if self.other_indices is not None else np.empty((0, 2), dtype=np.float32)

                img = draw_points_two_groups(
                    self.he_rgb,
                    pts_selected=pts_sel,
                    pts_other=pts_other,
                    color_selected=(0, 255, 0),
                    color_other=(0, 165, 255),
                    r_selected=6,
                    r_other=5,
                    name="HE nuclei (large)"
                )
            else:
                img = self.he_rgb.copy()
            self.show_large_view("H&E", img)
            return

        if idx == 2:
            if self.he_dapi_overlay is None:
                messagebox.showinfo(
                    "Not available",
                    "No cached overlay found (6_overlay.png).\nClick 'Load alignment' to generate one.",
                    parent=self
                )
                return
            img = self.he_dapi_overlay if self._panel3_show_overlay else self.he_rgb
            title = "Overlay" if self._panel3_show_overlay else "H&E"
            self.show_large_view(title, img)
            return

        if idx == 3:
            p = self.cache["cells"]
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    messagebox.showwarning("Failed", f"Cannot read: {p}", parent=self)
                    return
                self.show_large_view("Cells on H&E", img)
            else:
                messagebox.showinfo("Not loaded", "Cells overlay not available yet.\nClick 'Load cell data' first.",
                                    parent=self)
            return

# =============================
# main
# =============================
def main():
    if len(sys.argv) < 2:
        print("Usage: python 6_final_alignment.py <RUN_DIR>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    app = FinalAlignmentApp(run_dir)
    app.mainloop()


if __name__ == "__main__":
    main()