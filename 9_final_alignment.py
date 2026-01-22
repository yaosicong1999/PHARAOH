import os
import sys
import json
from pathlib import Path

import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps

from my_utils import read_image, dapi_to_lut_rgb

Image.MAX_IMAGE_PIXELS = None


# =============================
# CONFIG
# =============================
DISPLAY_LEVEL = 2
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


def cv2_to_pil(img):
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def fit_to_tile(pil_img, size=TILE_SIZE, bg=BG_COLOR):
    canvas = Image.new("RGB", size, bg)
    pil_img = ImageOps.contain(pil_img, size)
    x = (size[0] - pil_img.width) // 2
    y = (size[1] - pil_img.height) // 2
    canvas.paste(pil_img, (x, y))
    return canvas


def draw_points(img, pts, color=(0, 255, 0), r=40, name=""):
    """
    Draw points on image WITHOUT changing coordinate system.
    Includes hard debug prints.
    """
    out = img.copy()
    h, w = img.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    print(f"        valid pts: {valid.sum()} / {len(pts)}", flush=True)

    for (x, y) in pts[valid]:
        cv2.circle(out, (int(x), int(y)), r, color, -1)

    return out


def load_affine(path: Path):
    data = json.load(open(path))
    for key in ("H_mat", "H", "affine", "affine_2x3", "affine_3x3"):
        if key in data:
            H = np.asarray(data[key], dtype=np.float32)
            break
    else:
        raise KeyError(f"No affine matrix found in {path}")

    if H.shape == (3, 3):
        H = H[:2, :]
    if H.shape != (2, 3):
        raise ValueError(f"Invalid affine shape: {H.shape}")
    return H


# =============================
# GUI App
# =============================
class FinalAlignmentApp(tk.Tk):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.title("Step 6 — Final Alignment Viewer")
        self.run_dir = run_dir
        self.show_floating = True

        # -------- metadata --------
        info = json.load(open(run_dir / "images_info.json"))
        self.case_id = int(info.get("DAPI_orientation_case", 0))

        # -------- load images (LEVEL 2, NO ORIENTATION) --------
        dapi16, _ = read_image(info["DAPI_path"], keep_16bit=True, level=DISPLAY_LEVEL)
        lut = np.fromfile("glasbey_inverted.lut", dtype=np.uint8).reshape(256, 3)
        self.dapi_rgb = dapi_to_lut_rgb(dapi16, lut, threshold=300)

        he16, _ = read_image(info["HE_path"], keep_16bit=True, level=DISPLAY_LEVEL)
        he8 = cv2.normalize(he16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if he8.ndim == 2:
            he8 = cv2.cvtColor(he8, cv2.COLOR_GRAY2BGR)
        self.he_rgb = he8

        print("[DEBUG] DAPI image level-2 shape:", self.dapi_rgb.shape)
        print("[DEBUG] HE   image level-2 shape:", self.he_rgb.shape)

        # -------- load nuclei points (LEVEL 0) --------
        nuclei = json.load(open(run_dir / "nuclei_patches/nuclei_centroids_global.json"))
        self.dapi_pts0 = np.array([x["dapi_centroid_global"] for x in nuclei], np.float32)
        self.he_pts0   = np.array([x["he_centroid_global"]   for x in nuclei], np.float32)

        # -------- affine --------
        self.H = load_affine(run_dir / "dapi_to_he_affine_level0.json")

        # ---------- precompute floating overlay (NO orientation) ----------
        H_disp = self.H.copy()
        H_disp[:, 2] /= DISPLAY_SCALE

        self.warped_dapi = cv2.warpAffine(
            self.dapi_rgb,
            H_disp,
            (self.he_rgb.shape[1], self.he_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        self.he_dapi_overlay = cv2.addWeighted(
            self.he_rgb, 0.7,
            self.warped_dapi, 0.8,
            0
        )
        out_path = self.run_dir / "9_overlay.png"
        overlay_ds4 = cv2.resize(
            self.he_dapi_overlay,
            dsize=None,
            fx=0.25,
            fy=0.25,
            interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(str(out_path), overlay_ds4)
        print(
            f"[DEBUG] wrote overlay (downsampled 4x) to {out_path} | "
            f"shape {overlay_ds4.shape}",
            flush=True
        )
        # -------- layout --------
        mid = tk.Frame(self)
        mid.pack(padx=10, pady=10)

        self.panels = []
        titles = [
            "Nuclei on DAPI (LUT)",
            "Nuclei on H&E",
            "DAPI floating on H&E",
            "Cell centroid aligned (TBD)"
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
            btns, text="Hide / Show floating DAPI",
            command=self.toggle_floating
        ).pack(side="left", expand=True, fill="x")

        tk.Button(
            btns, text="Load cell data",
            command=lambda: messagebox.showinfo("TODO", "Not implemented")
        ).pack(side="left", expand=True, fill="x")

        # =============================
        # Precompute static panels
        # =============================
        # ---- panel 1: DAPI + points + orientation ----
        dapi_pts_lvl2 = self.dapi_pts0 / DISPLAY_SCALE
        img1 = draw_points(self.dapi_rgb, dapi_pts_lvl2, color=(0, 255, 0))
        self.panel1_img = apply_orientation_case(img1, self.case_id)
        # ---- panel 2: HE + points (NO orientation) ----
        he_pts_lvl2 = self.he_pts0 / DISPLAY_SCALE
        self.panel2_img = draw_points(self.he_rgb, he_pts_lvl2, color=(255, 0, 0))
        # ---- panel 3 variants (NO recompute later) ----
        self.panel3_overlay = self.he_dapi_overlay
        self.panel3_he = self.he_rgb

        self.refresh_images()
        self.minsize(self.winfo_width(), self.winfo_height())

    # =============================
    def refresh_images(self):
        # panel 1
        self._set_panel(0, self.panel1_img)

        # panel 2
        self._set_panel(1, self.panel2_img)

        # panel 3 (only switch reference)
        if self.show_floating:
            self._set_panel(2, self.panel3_overlay)
        else:
            self._set_panel(2, self.panel3_he)

        self._set_placeholder(3, "Not Loaded")
    def _set_panel(self, idx, cv_img):
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil)
        tkimg = ImageTk.PhotoImage(tile)
        self.panels[idx].lbl.configure(image=tkimg)
        self.panels[idx].lbl.image = tkimg

    def _set_placeholder(self, idx, text):
        img = np.full((TILE_SIZE[1], TILE_SIZE[0], 3), BG_COLOR, np.uint8)
        cv2.putText(img, text, (50, TILE_SIZE[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        self._set_panel(idx, img)

    def toggle_floating(self):
        self.show_floating = not self.show_floating
        self.refresh_images()


# =============================
# main
# =============================
def main():
    if len(sys.argv) < 2:
        print("Usage: python 9_final_alignment.py <RUN_DIR>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    app = FinalAlignmentApp(run_dir)
    app.mainloop()


if __name__ == "__main__":
    main()