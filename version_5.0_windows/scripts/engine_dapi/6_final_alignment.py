# ============================================================
# 6_final_alignment.py - Stage 6: Final alignment viewer
#
# Displays DAPI, H&E, DAPI warped onto H&E, and transformed cell
# centroids using the final registration result. Supports
# homography- and TPS-based transforms; parameters are stored in
# level-0 coordinates while the GUI renders at a downsampled level.
# Usage: python 6_final_alignment.py <RUN_DIR>
# ============================================================

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps
from my_utils import read_image, plot_cell_centroid
_VERSION_ROOT = str(Path(__file__).resolve().parents[1])
if _VERSION_ROOT not in sys.path:
    sys.path.insert(0, _VERSION_ROOT)
# Keep existing helper imports available to callers of this viewer module.
from stage6_outputs import (
    AlignmentOutputs, add_watermark, draw_points, load_affine, load_homography,
)
import local_residual

Image.MAX_IMAGE_PIXELS = None


# -----------------------------
# CONFIG
# -----------------------------
TILE_SIZE = (420, 420)
BG_COLOR = (240, 240, 240)


# -----------------------------
# Geometry / transform helpers
# -----------------------------
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


# -----------------------------
# Image / display helpers
# -----------------------------

def build_stage_to_morph_from_ome(dapi_ome_tif_path: Path) -> np.ndarray:
    """
    Construct a 3x3 transform from stage coordinates (microns)
    to DAPI morphology image coordinates at level 0 using OME metadata.
    """
    from ome_types import from_tiff

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


# -----------------------------
# transformation helpers
# -----------------------------
def load_transform_mode(path: Path):
    data = json.load(open(path, "r"))
    transform_type = str(data.get("transform_type", "")).strip().lower()

    if transform_type == "affine":
        return "affine"
    elif transform_type == "homography":
        return "homography"
    elif transform_type == local_residual.TRANSFORM_TYPE:
        return local_residual.TRANSFORM_TYPE
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


# -----------------------------
# Cell coordinate conversion
# -----------------------------
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
    import pandas as pd

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
    elif mode == local_residual.TRANSFORM_TYPE:
        xy_he = local_residual.forward_points(
            xy_dapi, json.loads(Path(transform_json).read_text())
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")

    print(f"[DEBUG] xy_he(level0 px) min={xy_he.min(axis=0)} max={xy_he.max(axis=0)}", flush=True)

    df.loc[:, "x_centroid"] = xy_he[:, 0]
    df.loc[:, "y_centroid"] = xy_he[:, 1]
    return df

# -----------------------------
# GUI application
# -----------------------------
class FinalAlignmentApp(tk.Tk, AlignmentOutputs):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.title("Stage 6 — Final Alignment Viewer")
        AlignmentOutputs.__init__(
            self, run_dir,
            lut_path=Path(__file__).resolve().parent / "glasbey_inverted.lut",
        )
        self.prepare_base_images()

        print("[DEBUG] DAPI image shape:", self.dapi_rgb.shape, flush=True)
        print("[DEBUG] H&E   image shape:", self.he_rgb.shape, flush=True)

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
        # panel0: moving kp
        p = self.cache["moving_kp"]
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

        # panel1: fixed kp
        p = self.cache["fixed_kp"]
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
            name="H&E nuclei"
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
        """Run the shared writer, then refresh the interactive viewer."""
        try:
            AlignmentOutputs.load_alignment_and_keypoints(self)
            self._panel3_tkimg_he = self._make_tkimg(self.he_rgb)
            self._panel3_tkimg_overlay = self._make_tkimg(self.he_dapi_overlay)
            self._panel3_show_overlay = True
            self.refresh_images_after_alignment()
        except Exception as e:
            messagebox.showerror("Load alignment failed", str(e), parent=self)
            raise

    # --------------------------
    # GUI-only panel helpers
    # --------------------------
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
            transformation_json = self.run_dir / f"{self.roles.homography_stem}.json"
            df = cells_to_he_pixels(
                path,
                transform_json=transformation_json,
                dapi_ome_tif=Path(self.roles.moving.path),
            )
            print("[DEBUG] H&E image level2 shape:", self.he_rgb.shape, flush=True)
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
                fixed = self.roles.fixed
                channel = (
                    fixed.channel_index
                    if fixed.modality == "dapi" and fixed.channel_index is not None
                    else fixed.modality
                )
                self.he16, _ = read_image(
                    fixed.path, keep_16bit=True,
                    level=self.display_level, channel=channel,
                )
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
                    name="H&E nuclei (large)"
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

# -----------------------------
# main
# -----------------------------
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
