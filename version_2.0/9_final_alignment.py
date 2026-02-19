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
DISPLAY_LEVEL = 4
DISPLAY_SCALE = 2 ** DISPLAY_LEVEL
TILE_SIZE = (420, 420)
BG_COLOR = (240, 240, 240)

# =============================
# Utils
# =============================
def apply_orientation_case(img, case_id):
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

def add_watermark(img, text):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    # Position (top-left with padding)
    padding = 20
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return img

def build_stage_to_morph_from_ome(dapi_ome_tif_path: Path) -> np.ndarray:
    """
    Build 3x3 homography: stage(micron) -> morphology pixel (level0)
    using OME-TIFF metadata.
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
    Draw points on image.
    pts: (N,2) float32 in the SAME coordinate system as img.
    """
    out = img.copy()
    h, w = img.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        print(f"[DEBUG] draw_points({name}): empty", flush=True)
        return out

    # optional subsample for GUI speed
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

    # --- pick one ---
    if A3 is not None and A3.shape == (3, 3):
        A2 = A3[:2, :].copy().astype(np.float32)
        return A2, A3.astype(np.float32)

    if A2 is not None and A2.shape == (2, 3):
        A3 = np.vstack([A2, [0, 0, 1]]).astype(np.float32)
        return A2, A3

    # --- HARD FAIL with helpful info ---
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

def transform_xy_affine(xy: np.ndarray, A2x3: np.ndarray) -> np.ndarray:
    """
    xy: (N,2) float32
    A2x3: (2,3) float32
    return: (N,2) float32
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(xy, A2x3)  # affine transform
    return out[:, 0, :]

def transform_xy_perspective(xy: np.ndarray, H3x3: np.ndarray) -> np.ndarray:
    """
    xy:    (N,2) float32
    H3x3:  (3,3) float32 homography
    return:(N,2) float32

    Applies perspective transform:
      [x', y', w']^T = H * [x, y, 1]^T
      (x', y') = (x'/w', y'/w')
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    H = np.asarray(H3x3, dtype=np.float32)

    if H.shape != (3, 3):
        raise ValueError(f"H3x3 must be (3,3), got {H.shape}")

    # ---- homogeneous coordinates ----
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy_h = np.concatenate([xy, ones], axis=1)          # (N,3)

    # ---- apply homography ----
    proj = (H @ xy_h.T).T                               # (N,3)

    # ---- normalize by w ----
    w = proj[:, 2:3]
    eps = 1e-8
    out = proj[:, 0:2] / np.maximum(w, eps)

    return out.astype(np.float32)

def transform_coordinates(coords_xy, homography_3x3):
    """
    coords_xy: (N,2) float
    homography_3x3: (3,3)
    return: (N,2)
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(coords_xy, np.asarray(homography_3x3, dtype=np.float32))
    return out[:, 0, :]

def infer_scale_from_info(info: dict):
    """
    Try to infer micron->HE pixel scale from images_info.json.
    You can customize the key to your real pipeline.
    """
    for k in ["micron_to_he_scale", "microns_to_he_scale", "scale", "he_scale"]:
        if k in info:
            return float(info[k])
    return None

def cells_to_he_pixels(cell_csv_gz: Path, affine_json: Path, dapi_ome_tif: Path):
    """
    Read cells.csv.gz (assumed stage micron coords) and convert:
      stage(micron) -> DAPI(level0 pixel, morphology_focus) -> HE(level0 pixel)
    using:
      stage_to_morph from OME metadata (DAPI OME-TIFF)
      + dapi_to_he_affine_level0.json (affine)
    """
    print(f"[INFO] reading cells: {cell_csv_gz}", flush=True)
    df = pd.read_csv(cell_csv_gz, index_col=0)

    need_cols = {"x_centroid", "y_centroid"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"cells.csv.gz missing columns {need_cols}. Got cols={list(df.columns)[:20]}...")

    xy_stage = df[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    print(f"[DEBUG] cells rows={len(df)} xy(stage micron) min={xy_stage.min(axis=0)} max={xy_stage.max(axis=0)}", flush=True)

    # stage(micron) -> dapi(level0 px) using OME metadata of morphology_focus
    H_stage2dapi = build_stage_to_morph_from_ome(dapi_ome_tif)  # 3x3
    xy_dapi = transform_coordinates(xy_stage, H_stage2dapi)
    print(f"[DEBUG] xy_dapi(level0 px) min={xy_dapi.min(axis=0)} max={xy_dapi.max(axis=0)}", flush=True)

    # # dapi(level0 px) -> he(level0 px) using affine
    # A2, _ = load_affine(affine_json)
    # ones = np.ones((xy_dapi.shape[0], 1), dtype=np.float32)
    # xy1 = np.concatenate([xy_dapi.astype(np.float32), ones], axis=1)   # (N,3)
    # xy_he = (xy1 @ A2.T).astype(np.float32)                            # (N,2)
    # print(f"[DEBUG] xy_he(level0 px) min={xy_he.min(axis=0)} max={xy_he.max(axis=0)}", flush=True)

    # dapi(level0 px) -> he(level0 px) using homography
    _, H3 = load_homography(affine_json)
    xy = xy_dapi.astype(np.float32)  # (N,2)
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy1 = np.concatenate([xy, ones], axis=1)  # (N,3)
    # apply homography
    proj = (xy1 @ H3.T).astype(np.float32)  # (N,3)
    # normalize by w
    w = proj[:, 2:3]
    eps = 1e-8
    xy_he = proj[:, 0:2] / np.maximum(w, eps)  # (N,2)
    print(
        f"[DEBUG] xy_he(level0 px) "
        f"min={xy_he.min(axis=0)} max={xy_he.max(axis=0)}",
        flush=True
    )

    df.loc[:, "x_centroid"] = xy_he[:, 0]
    df.loc[:, "y_centroid"] = xy_he[:, 1]
    return df

# =============================
# GUI App
# =============================
class FinalAlignmentApp(tk.Tk):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.title("Step 9 — Final Alignment Viewer")
        self.run_dir = run_dir

        # -------- metadata --------
        self.info = json.load(open(run_dir / "images_info.json"))
        self.case_id = int(self.info.get("DAPI_orientation_case", 0))

        # -------- config --------
        self.display_level = DISPLAY_LEVEL
        self.display_scale = float(2 ** self.display_level)
        self.cache = {
            # base
            "dapi_base": self.run_dir / f"9_cache_dapi_base_L{self.display_level}.png",
            "he_base": self.run_dir / f"9_cache_he_base_L{self.display_level}.png",

            # after load (with keypoints)
            "dapi_kp": self.run_dir / f"9_cache_dapi_kp_L{self.display_level}.png",
            "he_kp": self.run_dir / f"9_cache_he_kp_L{self.display_level}.png",

            # overlays
            "overlay": self.run_dir / f"9_overlay_final_L{self.display_level}.png",
            "manual": self.run_dir / f"9_overlay_manual_L{self.display_level}.png",
            "alternating": self.run_dir / f"9_alternating_L{self.display_level}.gif",
            "cells": self.run_dir / f"9_cells_centroids_L{self.display_level}.png",
        }

        # -------- state (alignment not loaded initially) --------
        self.alignment_loaded = False
        self.H3 = None
        self.warped_dapi = None
        self.he_dapi_overlay = None

        self.dapi_pts0 = None
        self.he_pts0 = None

        # -------- load base images ONLY (no alignment) --------
        # DAPI
        dapi_lut_thr = int(self.info.get("DAPI_LUT_threshold", 300))
        print(f"[INFO] Using DAPI_LUT_threshold={dapi_lut_thr} (from images_info.json)", flush=True)

        self.dapi_rgb = self._load_or_build_dapi_base()
        self.he_rgb, self.he16 = self._load_or_build_he_base()

        print("[DEBUG] DAPI image shape:", self.dapi_rgb.shape, flush=True)
        print("[DEBUG] HE   image shape:", self.he_rgb.shape, flush=True)

        # -------- cell state --------
        self.cells_df = None
        self.cells_pts_lvl2 = None

        # -------- layout --------
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
            btns, text="Load keypoints + alignment matrix",
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
    # after alignment loaded
    # --------------------------
    def refresh_images_after_alignment(self):
        # panel0: DAPI kp (prefer cache)
        p = self.cache["dapi_kp"]
        dapi_kp = self._imread(p)
        if dapi_kp is None:
            pts = self.dapi_pts0 / self.display_scale
            dapi_kp = draw_points(self.dapi_rgb, pts, color=(0, 255, 0), r=6, name="DAPI nuclei")
            self._imsave(p, dapi_kp)
        self._set_panel(0, apply_orientation_case(dapi_kp, self.case_id))

        # panel1: HE kp (prefer cache)
        p = self.cache["he_kp"]
        he_kp = self._imread(p)
        if he_kp is None:
            pts = self.he_pts0 / self.display_scale
            he_kp = draw_points(self.he_rgb, pts, color=(0, 255, 0), r=6, name="HE nuclei")
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
        Click to load nuclei keypoints + homography and build overlay.
        """
        try:
            # -------- load nuclei points (LEVEL 0 global px) --------
            nuclei_path = self.run_dir / "nuclei_patches/nuclei_centroids_global.json"
            nuclei = json.load(open(nuclei_path))
            self.dapi_pts0 = np.array([x["dapi_centroid_global"] for x in nuclei], np.float32)
            self.he_pts0   = np.array([x["he_centroid_global"]   for x in nuclei], np.float32)
            print(f"[INFO] Loaded nuclei: {len(self.dapi_pts0)}", flush=True)

            # -------- load homography level0 --------
            _, self.H3 = load_homography(self.run_dir / "dapi_to_he_homography_level0.json")
            # -------- build display homography --------
            s = float(self.display_scale)
            S = np.array([[s, 0, 0],
                          [0, s, 0],
                          [0, 0, 1]], dtype=np.float32)
            S_inv = np.array([[1 / s, 0, 0],
                              [0, 1 / s, 0],
                              [0, 0, 1]], dtype=np.float32)
            H_disp = (S_inv @ self.H3 @ S).astype(np.float32)
            # -------- warp + overlay --------
            self.warped_dapi = cv2.warpPerspective(
                self.dapi_rgb,
                H_disp,
                (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            self.he_dapi_overlay = cv2.addWeighted(self.he_rgb, 0.7, self.warped_dapi, 0.8, 0)
            cv2.imwrite(str(self.cache["overlay"]), self.he_dapi_overlay)

            # -------- load homography level0 --------
            data_manual = json.load(open(self.run_dir / "manual_initial_alignment.json", "r"))
            H3m = np.array(data_manual['H_mat_level_0'], dtype=np.float32)
            # -------- build display homography --------
            Sm = np.array([[s, 0, 0],
                          [0, s, 0],
                          [0, 0, 1]], dtype=np.float32)
            Sm_inv = np.array([[1 / s, 0, 0],
                              [0, 1 / s, 0],
                              [0, 0, 1]], dtype=np.float32)
            Hm_disp = (Sm_inv @ H3m @ Sm).astype(np.float32)
            # -------- warp + overlay --------
            warped_dapi_manual = cv2.warpPerspective(
                self.dapi_rgb,
                Hm_disp,
                (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            he_dapi_overlay_manual = cv2.addWeighted(self.he_rgb, 0.7, warped_dapi_manual, 0.8, 0)
            cv2.imwrite(str(self.cache["manual"]), he_dapi_overlay_manual)

            img1_w = add_watermark(Image.fromarray(self.he_dapi_overlay[..., ::-1]), "Final alignment")
            img2_w = add_watermark(Image.fromarray(he_dapi_overlay_manual[..., ::-1]), "Manual initial alignment")
            img1_w.save(
                str(self.cache["alternating"]),
                save_all=True,
                append_images=[img2_w],
                duration=1000,
                loop=0
            )
            # -------- build & save keypoint panels (cache) --------
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

            # -------- cache Tk images for fast toggle --------
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
        img = self._imread(self.cache["dapi_base"])
        if img is not None:
            return img

        dapi_lut_thr = int(self.info.get("DAPI_LUT_threshold", 300))
        dapi16, _ = read_image(self.info["DAPI_path"], keep_16bit=True, level=self.display_level, channel="dapi")
        lut = np.fromfile("glasbey_inverted.lut", dtype=np.uint8).reshape(256, 3)
        rgb = dapi_to_lut_rgb(dapi16, lut, threshold=dapi_lut_thr)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._imsave(self.cache["dapi_base"], bgr)
        return bgr

    def _load_or_build_he_base(self):
        img = self._imread(self.cache["he_base"])
        if img is not None:            return img, None

        he16, _ = read_image(self.info["HE_path"], keep_16bit=False, level=self.display_level,  channel="he")
        print(f"he16 shape is {he16.shape}")
        print("channels equal?", np.all(he16[..., 0] == he16[..., 1]) and np.all(he16[..., 1] == he16[..., 2]))
        print("min/max per channel:", [(he16[..., c].min(), he16[..., c].max()) for c in range(3)])
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
            # affine_json = self.run_dir / "dapi_to_he_affine_level0.json"
            perspective_json = self.run_dir / "dapi_to_he_homography_level0.json"
            df = cells_to_he_pixels(
                path,
                affine_json=perspective_json,
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
                dot_size=5 / 2 ** (2 ** (DISPLAY_LEVEL - 2))
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

    def _open_cells_panel(self):
        if self.cells_overlay_path.exists():
            img = cv2.imread(str(self.cells_overlay_path), cv2.IMREAD_COLOR)
            if img is None:
                messagebox.showwarning("Failed", f"Cannot read: {self.cells_overlay_path}", parent=self)
                return
            self.show_large_view("Cells on H&E", img)
        else:
            messagebox.showinfo("Not loaded", "Cells overlay not available yet.\nClick 'Load cell data' first.", parent=self)

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
        idx: 0..3 for the 4 panels
        Behavior depends on whether alignment is loaded.
        """
        # panel 0: DAPI
        if idx == 0:
            if self.alignment_loaded and self.dapi_pts0 is not None:
                pts = self.dapi_pts0 / self.display_scale
                img = draw_points(self.dapi_rgb, pts, color=(0, 255, 0), r=6, name="DAPI nuclei (large)")
            else:
                img = self.dapi_rgb.copy()
            img = apply_orientation_case(img, self.case_id)
            self.show_large_view("DAPI", img)
            return

        # panel 1: HE
        if idx == 1:
            if self.alignment_loaded and self.he_pts0 is not None:
                pts = self.he_pts0 / self.display_scale
                img = draw_points(self.he_rgb, pts, color=(0, 255, 0), r=6, name="HE nuclei (large)")
            else:
                img = self.he_rgb.copy()
            self.show_large_view("H&E", img)
            return

        # panel 2: Overlay / HE toggle
        if idx == 2:
            if self.he_dapi_overlay is None:
                messagebox.showinfo(
                    "Not available",
                    "No cached overlay found (9_overlay.png).\nClick 'Load alignment' to generate one.",
                    parent=self
                )
                return
            img = self.he_dapi_overlay if self._panel3_show_overlay else self.he_rgb
            title = "Overlay" if self._panel3_show_overlay else "H&E"
            self.show_large_view(title, img)
            return

        # panel 3: Cells
        if idx == 3:
            p = self.cache["cells"]
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    messagebox.showwarning("Failed", f"Cannot read: {self.cells_overlay_path}", parent=self)
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
        print("Usage: python 9_final_alignment.py <RUN_DIR>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    app = FinalAlignmentApp(run_dir)
    app.mainloop()


if __name__ == "__main__":
    main()