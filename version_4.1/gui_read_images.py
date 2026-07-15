# ============================================================
# gui_read_images.py - Unified Stage 1 (DAPI / H&E-FG + H&E)
#
# One window, two columns:
#   - LEFT  column: the fixed H&E image (always).
#   - RIGHT column: the MOVING image. Its top has two stacked buttons,
#       [ Select DAPI Image ]
#       [ Select H&E-FG Image  ]
#     Pick one to set the mode. DAPI is read as fluorescence + LUT;
#     H&E-FG is read as a second brightfield H&E (hematoxylin threshold).
#     Both support rotate/flip orientation.
#
# Each column shows the original image and its threshold/LUT preview.
# On save, writes images_info.json (with a "mode" field) and the level
# images (+ the H&E-FG threshold mask that Stage 3 uses), so Stages 2-6 can
# be dispatched to engine_dapi/ or engine_he/.
#
# Usage: python gui_read_images.py <RUN_DIR>
# ============================================================
import json
import sys
import tkinter as tk
import tkinter.messagebox as messagebox
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from my_utils import (
    read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel, dapi_to_lut_rgb,
)

Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = (320, 320)
HERE = Path(__file__).resolve().parent
LUT = np.fromfile(str(HERE / "glasbey_inverted.lut"), dtype=np.uint8).reshape(256, 3)

ORIENTATION_CASES = {
    0: np.array([[1, 0], [0, 1]]),
    1: np.array([[0, -1], [1, 0]]),
    2: np.array([[-1, 0], [0, -1]]),
    3: np.array([[0, 1], [-1, 0]]),
    4: np.array([[1, 0], [0, -1]]),
    5: np.array([[-1, 0], [0, 1]]),
    6: np.array([[0, 1], [1, 0]], np.float32),
    7: np.array([[0, -1], [-1, 0]], np.float32),
}


def infer_orientation_case(gui_affine, tol=1e-4):
    A = np.array(gui_affine, dtype=np.float32)[:2, :2]
    for cid, ref in ORIENTATION_CASES.items():
        if np.allclose(A, ref, atol=tol):
            return cid
    return 0


def make_na_tile(size=TILE_SIZE, bg=240):
    return Image.new("RGB", size, (bg, bg, bg))


def to_fixed_tile(img_np, size=TILE_SIZE, bg=240):
    if img_np is None:
        return make_na_tile(size, bg)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.dtype == np.uint16:
        img_np = (img_np / 256).astype(np.uint8)
    pil = ImageOps.contain(Image.fromarray(img_np), size)
    canvas = Image.new("RGB", size, (bg, bg, bg))
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    return canvas


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn) * 255 if mx > mn else np.zeros_like(img)
    return img.astype(np.uint8)


class UnifiedStage1(tk.Tk):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.withdraw()                    # stay hidden until sized + centered
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        name = run_dir.name
        self.run_id = name.replace("runs_", "", 1) if name.startswith("runs_") else name
        self.title("STEP 1: Unified image selection (DAPI / H&E-FG + H&E)")
        # No fixed geometry: the window auto-sizes to its content, so it starts
        # as just the two selector buttons and grows as images are loaded.

        # ---- HE state ----
        self.he_orig = None
        self.he_h_proc = None
        self.he_mask_img = None
        self.he_level = None
        self.he_path = None

        # ---- moving state ----
        self.moving_mode = None            # None | "dapi" | "he0"
        self.moving_path = None
        self.moving_level = None
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = None
        self.dapi_img_view = None
        self.dapi_lut_img = None
        self.he0_orig = None
        self.he0_h_proc = None
        self.he0_mask_img = None

        self._imgrefs = {}                 # keep PhotoImage refs alive
        self._build_ui()
        self._fit_window()                 # size + center while still hidden
        self.deiconify()                   # now show it, already centered (no top-left flash)
        self._raise_to_front()             # pop in front of the pipeline window

    # -----------------------------
    # UI
    # -----------------------------
    def _panel(self, row, col):
        # empty label, hidden until an image is loaded (no placeholder, no space)
        lbl = tk.Label(self)
        lbl.grid(row=row, column=col, padx=5, pady=5)
        lbl.grid_remove()
        return lbl

    def _set_tile(self, label, pil_or_np):
        pil = pil_or_np if isinstance(pil_or_np, Image.Image) else to_fixed_tile(pil_or_np)
        tkimg = ImageTk.PhotoImage(pil)
        label.configure(image=tkimg)
        label.grid()                       # reveal the panel now that it has content
        self._imgrefs[id(label)] = tkimg

    def _build_ui(self):
        # uniform group keeps both columns equal width, so the H&E button and
        # the DAPI/H&E-FG buttons are the same width even before any image loads
        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=1, uniform="cols")

        # ---- selectors ----
        tk.Button(self, text="Select H&E Image", command=self.select_he)\
            .grid(row=0, column=0, sticky="we", padx=5, pady=5)

        mv_btns = tk.Frame(self)
        mv_btns.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        self.btn_dapi = tk.Button(mv_btns, text="Select DAPI Image", command=self.on_select_dapi)
        self.btn_he0 = tk.Button(mv_btns, text="Select H&E-FG Image", command=self.on_select_he0)
        self.btn_dapi.pack(side="top", fill="x", pady=(0, 3))
        self.btn_he0.pack(side="top", fill="x")

        # ---- image panels: original + threshold/LUT preview ----
        self.he_orig_label = self._panel(1, 0)
        self.he_mask_label = self._panel(2, 0)
        self.mv_orig_label = self._panel(1, 1)
        self.mv_proc_label = self._panel(2, 1)

        # ---- HE slider (built when an H&E image is selected) ----
        self.he_thr = tk.IntVar(value=240)
        self.he_slider_frame = tk.Frame(self)
        self.he_slider_frame.grid(row=3, column=0, padx=5, pady=6)
        self.he_slider_frame.grid_remove()

        # ---- moving slider (built when a DAPI/H&E-FG image is selected) ----
        self.mv_slider_frame = tk.Frame(self)
        self.mv_slider_frame.grid(row=3, column=1, padx=5, pady=6)
        self.mv_slider_frame.grid_remove()
        self.mv_thr = tk.IntVar(value=300)

        # ---- moving orientation buttons (shown once a moving image is loaded) ----
        # 2x2 grid so this row stays within the tile column width -> the window
        # width is the same whether DAPI/H&E-FG or H&E is loaded first.
        self.orient_frame = tk.Frame(self)
        self.orient_frame.grid(row=4, column=1, padx=5, pady=5, sticky="we")
        self.orient_frame.columnconfigure(0, weight=1, uniform="ori")
        self.orient_frame.columnconfigure(1, weight=1, uniform="ori")
        self.btn_rot_cw = tk.Button(self.orient_frame, text="Rotate CW",
                                    command=lambda: self._orient("rot_cw"))
        self.btn_rot_ccw = tk.Button(self.orient_frame, text="Rotate CCW",
                                     command=lambda: self._orient("rot_ccw"))
        self.btn_flip_v = tk.Button(self.orient_frame, text="Flip V",
                                    command=lambda: self._orient("flip_v"))
        self.btn_flip_h = tk.Button(self.orient_frame, text="Flip H",
                                    command=lambda: self._orient("flip_h"))
        self.btn_rot_cw.grid(row=0, column=0, sticky="we", padx=2, pady=2)
        self.btn_rot_ccw.grid(row=0, column=1, sticky="we", padx=2, pady=2)
        self.btn_flip_v.grid(row=1, column=0, sticky="we", padx=2, pady=2)
        self.btn_flip_h.grid(row=1, column=1, sticky="we", padx=2, pady=2)
        self.orient_frame.grid_remove()

        # ---- confirm (shown once both images are ready) ----
        self.confirm_btn = tk.Button(self, text="Confirm & Save", command=self.confirm_and_save,
                                     state=tk.DISABLED)
        self.confirm_btn.grid(row=5, column=0, columnspan=2, sticky="we", padx=10, pady=10)
        self.confirm_btn.grid_remove()

    def _rebuild_moving_slider(self, label, lo, hi, step, default):
        for w in self.mv_slider_frame.winfo_children():
            w.destroy()
        self.mv_thr = tk.IntVar(value=default)
        tk.Label(self.mv_slider_frame, text=label).pack(anchor="w")
        tk.Scale(self.mv_slider_frame, from_=lo, to=hi, orient=tk.HORIZONTAL, resolution=step,
                 showvalue=True, length=TILE_SIZE[0] - 40, variable=self.mv_thr,
                 command=lambda v: self.update_moving(int(v))).pack(fill="x")
        self.mv_slider_frame.grid()

    def _raise_to_front(self):
        """Bring the window to the foreground when it launches."""
        self.lift()
        self.attributes("-topmost", True)
        self.after(400, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def _fit_window(self):
        """Resize the window to fit its content and center it on the screen."""
        # Once any image is loaded, keep both columns tile-width so the H&E and
        # moving columns stay balanced even if only one side has an image yet.
        if self.he_orig is not None or self.moving_mode is not None:
            col_w = TILE_SIZE[0] + 12
            self.columnconfigure(0, weight=1, minsize=col_w, uniform="cols")
            self.columnconfigure(1, weight=1, minsize=col_w, uniform="cols")
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(0, (self.winfo_screenheight() - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _set_orientation_enabled(self, enabled):
        if enabled:
            self.orient_frame.grid()
        else:
            self.orient_frame.grid_remove()
        st = tk.NORMAL if enabled else tk.DISABLED
        for b in (self.btn_rot_cw, self.btn_rot_ccw, self.btn_flip_v, self.btn_flip_h):
            b.config(state=st)

    def _update_confirm_state(self):
        ready = self.he_mask_img is not None and (
            (self.moving_mode == "dapi" and self.dapi_lut_img is not None) or
            (self.moving_mode == "he0" and self.he0_mask_img is not None)
        )
        if ready:
            self.confirm_btn.grid()
            self.confirm_btn.config(state=tk.NORMAL)
        else:
            self.confirm_btn.config(state=tk.DISABLED)

    # -----------------------------
    # H&E (fixed)
    # -----------------------------
    def select_he(self):
        path = filedialog.askopenfilename(title="Select H&E Image")
        if not path:
            return
        self.he_path = path
        self.he_orig, self.he_level = read_image(path, channel="he")
        self.he_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.he_orig))
        if not self.he_slider_frame.winfo_children():
            tk.Label(self.he_slider_frame, text="H&E Threshold").pack(anchor="w")
            tk.Scale(self.he_slider_frame, from_=150, to=255, orient=tk.HORIZONTAL, resolution=5,
                     showvalue=True, length=TILE_SIZE[0] - 40, variable=self.he_thr,
                     command=lambda v: self.update_he(int(v))).pack(fill="x")
        self.he_slider_frame.grid()
        self._set_tile(self.he_orig_label, self.he_orig)
        self.update_he(int(self.he_thr.get()))
        self._fit_window()

    def update_he(self, threshold):
        if self.he_h_proc is None:
            return
        _, m = cv2.threshold(self.he_h_proc, int(threshold), 255, cv2.THRESH_BINARY)
        self.he_mask_img = m.astype(np.uint8)
        self._set_tile(self.he_mask_label, self.he_mask_img)
        self._update_confirm_state()

    # -----------------------------
    # Moving: DAPI
    # -----------------------------
    def on_select_dapi(self):
        path = filedialog.askopenfilename(title="Select DAPI Image")
        if not path:
            return
        self.moving_mode = "dapi"
        self.moving_path = path
        img, self.moving_level = read_image(path, keep_16bit=True, force_rgb=False, channel="dapi")
        self.dapi_img_view = img.copy()
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = self.dapi_img_view.shape[:2]
        self.he0_orig = None; self.he0_mask_img = None
        self._rebuild_moving_slider("DAPI LUT Threshold", 0, 2000, 10, 300)
        self._set_orientation_enabled(True)
        self.update_moving(int(self.mv_thr.get()))
        self._fit_window()

    # -----------------------------
    # Moving: H&E-FG
    # -----------------------------
    def on_select_he0(self):
        path = filedialog.askopenfilename(title="Select H&E-FG Image")
        if not path:
            return
        self.moving_mode = "he0"
        self.moving_path = path
        self.he0_orig, self.moving_level = read_image(path, channel="he")
        self.he0_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.he0_orig))
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = self.he0_orig.shape[:2]
        self.dapi_img_view = None; self.dapi_lut_img = None
        self._rebuild_moving_slider("H&E-FG Threshold", 150, 255, 5, 240)
        self._set_orientation_enabled(True)
        self.update_moving(int(self.mv_thr.get()))
        self._fit_window()

    def update_moving(self, threshold):
        if self.moving_mode == "dapi":
            if self.dapi_img_view is None:
                return
            self.dapi_lut_img = dapi_to_lut_rgb(self.dapi_img_view, LUT, threshold=int(threshold))
            gray = self.dapi_img_view[..., 0] if self.dapi_img_view.ndim == 3 else self.dapi_img_view
            self._set_tile(self.mv_orig_label, normalize_to_uint8(gray))
            self._set_tile(self.mv_proc_label, cv2.cvtColor(self.dapi_lut_img, cv2.COLOR_BGR2RGB))
        elif self.moving_mode == "he0":
            if self.he0_h_proc is None:
                return
            _, m = cv2.threshold(self.he0_h_proc, int(threshold), 255, cv2.THRESH_BINARY)
            self.he0_mask_img = m.astype(np.uint8)
            self._set_tile(self.mv_orig_label, self.he0_orig)
            self._set_tile(self.mv_proc_label, self.he0_mask_img)
        self._update_confirm_state()

    # -----------------------------
    # Orientation (moving)
    # -----------------------------
    def _orient(self, kind):
        if self.moving_mode is None or self.moving_gui_shape is None:
            return
        H, W = self.moving_gui_shape
        if kind == "rot_cw":
            M = np.array([[0, -1, H - 1], [1, 0, 0], [0, 0, 1]], np.float32)
        elif kind == "rot_ccw":
            M = np.array([[0, 1, 0], [-1, 0, W - 1], [0, 0, 1]], np.float32)
        elif kind == "flip_v":
            M = np.array([[1, 0, 0], [0, -1, H - 1], [0, 0, 1]], np.float32)
        else:  # flip_h
            M = np.array([[-1, 0, W - 1], [0, 1, 0], [0, 0, 1]], np.float32)
        self.moving_gui_affine = M @ self.moving_gui_affine

        def xform(arr):
            if kind == "rot_cw":
                return np.rot90(arr, k=3)
            if kind == "rot_ccw":
                return np.rot90(arr, k=1)
            if kind == "flip_v":
                return np.flipud(arr)
            return np.fliplr(arr)

        if self.moving_mode == "dapi":
            self.dapi_img_view = xform(self.dapi_img_view)
        else:
            self.he0_orig = xform(self.he0_orig)
            self.he0_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.he0_orig))
        if kind in ("rot_cw", "rot_ccw"):
            self.moving_gui_shape = (W, H)
        self.update_moving(int(self.mv_thr.get()))

    # -----------------------------
    # Save
    # -----------------------------
    def confirm_and_save(self):
        if self.he_mask_img is None or self.he_orig is None:
            messagebox.showerror("Error", "Load & threshold the H&E image first.")
            return
        if self.moving_mode is None:
            messagebox.showerror("Error", "Select a DAPI or H&E-FG image first.")
            return
        rd = self.run_dir
        Image.fromarray(self.he_orig).save(rd / "1_he_level_image.png")
        cv2.imwrite(str(rd / "1_he_threshold_mask.png"), self.he_mask_img)

        info = {
            "RUN_ID": self.run_id,
            "mode": self.moving_mode,
            "HE_path": str(self.he_path),
            "HE_level": int(self.he_level),
            "HE_threshold": int(self.he_thr.get()),
        }

        if self.moving_mode == "dapi":
            cv2.imwrite(str(rd / "1_dapi_lut.png"), self.dapi_lut_img)
            info.update({
                "DAPI_path": str(self.moving_path),
                "DAPI_level": int(self.moving_level),
                "DAPI_gui_affine": self.moving_gui_affine.tolist(),
                "DAPI_orientation_case": int(infer_orientation_case(self.moving_gui_affine)),
                "DAPI_LUT_threshold": int(self.mv_thr.get()),
            })
        else:
            Image.fromarray(self.he0_orig).save(rd / "1_he0_level_image.png")
            cv2.imwrite(str(rd / "1_he0_threshold_mask.png"), self.he0_mask_img)
            info.update({
                "HE0_path": str(self.moving_path),
                "HE0_level": int(self.moving_level),
                "HE0_gui_affine": self.moving_gui_affine.tolist(),
                "HE0_orientation_case": int(infer_orientation_case(self.moving_gui_affine)),
                "HE0_threshold": int(self.mv_thr.get()),
            })

        with open(rd / "images_info.json", "w") as f:
            json.dump(info, f, indent=2)
        messagebox.showinfo("Saved",
                            f"Stage 1 saved (mode={self.moving_mode}) to:\n{rd}\n\n"
                            f"Return to the pipeline and run Stage 2.")
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gui_read_images.py <RUN_DIR>")
        sys.exit(2)
    app = UnifiedStage1(Path(sys.argv[1]).resolve())
    app.mainloop()
