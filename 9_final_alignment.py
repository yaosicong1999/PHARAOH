import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps
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

def transform_xy_affine(xy: np.ndarray, A2x3: np.ndarray) -> np.ndarray:
    """
    xy: (N,2) float32
    A2x3: (2,3) float32
    return: (N,2) float32
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(xy, A2x3)  # affine transform
    return out[:, 0, :]

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

    # dapi(level0 px) -> he(level0 px) using affine
    A2, _ = load_affine(affine_json)
    ones = np.ones((xy_dapi.shape[0], 1), dtype=np.float32)
    xy1 = np.concatenate([xy_dapi.astype(np.float32), ones], axis=1)   # (N,3)
    xy_he = (xy1 @ A2.T).astype(np.float32)                            # (N,2)
    print(f"[DEBUG] xy_he(level0 px) min={xy_he.min(axis=0)} max={xy_he.max(axis=0)}", flush=True)

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
        self.show_floating = True

        # -------- metadata --------
        self.info = json.load(open(run_dir / "images_info.json"))
        self.case_id = int(self.info.get("DAPI_orientation_case", 0))

        # -------- load images (LEVEL 2, NO ORIENTATION) --------
        dapi16, _ = read_image(self.info["DAPI_path"], keep_16bit=True, level=DISPLAY_LEVEL)
        lut = np.fromfile("glasbey_inverted.lut", dtype=np.uint8).reshape(256, 3)
        self.dapi_rgb = dapi_to_lut_rgb(dapi16, lut, threshold=300)
        self.dapi_rgb = cv2.cvtColor(self.dapi_rgb, cv2.COLOR_RGB2BGR)  # ✅ 加这一行：统一内部用BGR

        he16, _ = read_image(self.info["HE_path"], keep_16bit=True, level=DISPLAY_LEVEL)
        # normalize to uint8
        he8 = cv2.normalize(he16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if he8.ndim == 2:
            he_bgr = cv2.cvtColor(he8, cv2.COLOR_GRAY2BGR)
        elif he8.ndim == 3 and he8.shape[2] == 3:
            he_bgr = cv2.cvtColor(he8, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unexpected HE shape: {he8.shape}")

        self.he_rgb = he_bgr
        self.he16 = he16
        print("[DEBUG] DAPI image level-2 shape:", self.dapi_rgb.shape, flush=True)
        print("[DEBUG] HE   image level-2 shape:", self.he_rgb.shape, flush=True)

        # -------- load nuclei points (LEVEL 0) --------
        nuclei = json.load(open(run_dir / "nuclei_patches/nuclei_centroids_global.json"))
        self.dapi_pts0 = np.array([x["dapi_centroid_global"] for x in nuclei], np.float32)
        self.he_pts0   = np.array([x["he_centroid_global"]   for x in nuclei], np.float32)

        # -------- affine --------
        self.A2, self.A3 = load_affine(run_dir / "dapi_to_he_affine_level0.json")  # A2: (2,3)

        # ---------- precompute floating overlay (NO orientation) ----------
        H_disp = self.A2.copy()
        H_disp[:, 2] /= DISPLAY_SCALE

        self.warped_dapi = cv2.warpAffine(
            self.dapi_rgb,
            H_disp,
            (self.he_rgb.shape[1], self.he_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        print("[DEBUG] he_rgb first pixel (BGR):", self.he_rgb[0, 0], flush=True)
        print("[DEBUG] dapi_rgb first pixel (BGR):", self.dapi_rgb[0, 0], flush=True)
        self.he_dapi_overlay = cv2.addWeighted(self.he_rgb, 0.7, self.warped_dapi, 0.8, 0)

        out_path = self.run_dir / "9_overlay.png"
        cv2.imwrite(str(out_path), self.he_dapi_overlay)
        print(f"[DEBUG] wrote overlay to {out_path}", flush=True)

        # -------- cell state --------
        self.cells_df = None
        self.cells_pts_lvl2 = None

        # -------- layout --------
        mid = tk.Frame(self)
        mid.pack(padx=10, pady=10)

        self.panels = []
        titles = [
            "Nuclei on DAPI (LUT)",
            "Nuclei on H&E",
            "DAPI Overlay on H&E",
            "Cells on H&E (from cells.csv.gz)"
        ]

        for i, t in enumerate(titles):
            f = tk.Frame(mid)
            tk.Label(f, text=t, font=("Helvetica", 11, "bold")).pack()
            lbl = tk.Label(f)
            lbl.pack()
            f.grid(row=0, column=i, padx=6)
            f.lbl = lbl
            self.panels.append(f)

        btns = tk.Frame(self)
        btns.pack(fill="x", padx=10, pady=6)

        tk.Button(
            btns, text="Toggle H&E / Overlay",
            command=self.toggle_floating
        ).pack(side="left", expand=True, fill="x")

        tk.Button(
            btns, text="Load cell data (cells.csv.gz)",
            command=self.load_cell_data
        ).pack(side="left", expand=True, fill="x")

        self._panel3_tkimg_he = self._make_tkimg(self.he_rgb)
        self._panel3_tkimg_overlay = self._make_tkimg(self.he_dapi_overlay)
        self.cells_overlay_path = self.run_dir / "9_cells_centroids.png"

        # init current display
        self._panel3_show_overlay = True
        self.refresh_images()
        self.minsize(self.winfo_width(), self.winfo_height())


    # =============================
    def refresh_images(self):
        # -------- panel 1 (DAPI + nuclei) --------
        dapi_pts_lvl2 = self.dapi_pts0 / DISPLAY_SCALE
        img1 = draw_points(self.dapi_rgb, dapi_pts_lvl2, color=(0, 255, 0), r=6, name="DAPI nuclei")
        img1 = apply_orientation_case(img1, self.case_id)
        self._set_panel(0, img1)

        # -------- panel 2 (HE + nuclei) --------
        he_pts_lvl2 = self.he_pts0 / DISPLAY_SCALE
        img2 = draw_points(self.he_rgb, he_pts_lvl2, color=(255, 0, 0), r=6, name="HE nuclei")
        self._set_panel(1, img2)

        # -------- panel 3 (toggle overlay) --------
        self._update_panel3_fast()

        # -------- panel 4 (cells) --------
        if self.cells_overlay_path.exists():
            img4 = cv2.imread(str(self.cells_overlay_path), cv2.IMREAD_COLOR)
            if img4 is None:
                self._set_placeholder(3, "Failed to load cells overlay")
            else:
                self._set_panel(3, img4)
        else:
            self._set_placeholder(3, "Cell Info Not Loaded")

    def _make_tkimg(self, cv_img):
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil)
        return ImageTk.PhotoImage(tile)

    def _update_panel3_fast(self):
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

        # ---- compute text size ----
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x = (TILE_SIZE[0] - tw) // 2
        y = (TILE_SIZE[1] + th) // 2  # 注意：putText 用的是 baseline

        cv2.putText(
            img,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        self._set_panel(idx, img)
    def toggle_floating(self):
        self._panel3_show_overlay = not self._panel3_show_overlay
        self._update_panel3_fast()

    # =============================
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
            affine_json = self.run_dir / "dapi_to_he_affine_level0.json"
            df = cells_to_he_pixels(
                path,
                affine_json=affine_json,
                dapi_ome_tif=Path(self.info["DAPI_path"]),
            )
            print("[DEBUG] HE image level2 shape:", self.he_rgb.shape, flush=True)
            print("[DEBUG] first 5 transformed cells (he_l0):\n", df[["x_centroid", "y_centroid"]].head(), flush=True)

            # downsample to DISPLAY_LEVEL for GUI overlay
            df_gui = df.copy()
            df_gui.loc[:, "x_centroid"] = df_gui["x_centroid"].astype(np.float32) / DISPLAY_SCALE
            df_gui.loc[:, "y_centroid"] = df_gui["y_centroid"].astype(np.float32) / DISPLAY_SCALE

            self.cells_df = df  # keep level0 px
            self.cells_pts_lvl2 = df_gui[["x_centroid", "y_centroid"]].to_numpy(np.float32)

            # --- plot + save (uses he16 at DISPLAY_LEVEL) ---
            out_png = self.run_dir / "9_cells_centroids.png"
            print(f"[INFO] plotting cell centroids -> {out_png}", flush=True)

            # plot_cell_centroid expects df with columns x_centroid/y_centroid
            out_png = self.run_dir / "9_cells_centroids.png"
            plot_cell_centroid(
                df_gui,
                he=self.he16,
                color="red",
                save_name=str(out_png),
                save_fig=True,
                dot_size=5/2**(2**(DISPLAY_LEVEL-2))
            )
            messagebox.showinfo(
                "Loaded",
                f"Loaded and transformed:\n{path}\n\nSaved:\n{out_png}",
                parent=self
            )
            self.refresh_images()

        except Exception as e:
            messagebox.showerror("Load failed", str(e), parent=self)
            raise


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