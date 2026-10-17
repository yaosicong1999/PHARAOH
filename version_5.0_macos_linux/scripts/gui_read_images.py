# ============================================================
# gui_read_images.py - Unified Stage 1 (all moving/fixed DAPI and H&E pairs)
#
# One window, two columns:
#   - LEFT column: Moving (foreground), with two stacked buttons,
#       [ Select DAPI Image ]
#       [ Select H&E Image (Foreground) ]
#   - RIGHT column: Fixed (background), with two stacked buttons,
#       [ Select H&E Image ]
#       [ Select DAPI Image (Background) ]
#     Pick one to set the mode. DAPI is read as fluorescence + LUT;
#     H&E-FG is read as a second brightfield H&E (hematoxylin threshold).
#     Both support rotate/flip orientation.
#
# Each column shows the original image and its threshold/LUT preview.
# On save, writes images_info.json (with pair + roles) and the level
# images (+ the H&E-FG threshold mask that Stage 3 uses), so Stages 2-6 can
# be dispatched to engine_dapi/, the one engine.
#
# Usage: python gui_read_images.py <RUN_DIR>
# ============================================================
import json
import argparse
import sys
import tkinter as tk
import tkinter.messagebox as messagebox
from pathlib import Path
from tkinter import filedialog
import run_roles

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from my_utils import (
    read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel,
    dapi_to_lut_rgb, resolve_dapi_lut_threshold, select_paired_image_levels,
    resolve_dapi_channel,
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
    def __init__(self, run_dir: Path, forced_modality=None, *, pair=None, forced_mode=None):
        requested = pair or forced_modality or forced_mode
        forced_pair = run_roles.coerce_pair(requested) if requested else None
        if forced_pair is not None and not forced_pair.implemented:
            raise ValueError(f"GUI pair {forced_pair.name} is not enabled")
        super().__init__()
        self.withdraw()                    # stay hidden until sized + centered
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        name = run_dir.name
        self.run_id = name.replace("runs_", "", 1) if name.startswith("runs_") else name
        self.forced_modality = forced_pair.moving if forced_pair else None
        self.forced_fixed_modality = forced_pair.fixed if forced_pair else None
        self.fixed_modality = self.forced_fixed_modality or "he"
        self.fixed_channel_selection = None
        self.fixed_dapi_lut = None
        self.fixed_dapi_window = None
        mode_label = forced_pair.name if forced_pair else "Moving / Fixed"
        self.title(f"STEP 1: Image selection ({mode_label})")
        # No fixed geometry: the window auto-sizes to its content, so it starts
        # as the two selector columns and grows as images are loaded.

        # ---- HE state ----
        self.he_orig = None
        self.he_h_proc = None
        self.he_mask_img = None
        self.he_level = None
        self.he_path = None

        # ---- moving state ----
        self.moving_modality = None            # None | "dapi" | "he"
        self.moving_path = None
        self.moving_level = None
        self.moving_channel_selection = None
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = None
        self.dapi_img_view = None
        self.dapi_lut_img = None
        self.moving_he_orig = None
        self.moving_he_h_proc = None
        self.moving_he_mask_img = None

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

    def _ask_image_path(self, title):
        """Where a selector button gets its path.

        Overridden by the fail-safe Stage 1, which is handed both images up front
        and only asks the user for the moving image's orientation.
        """
        return filedialog.askopenfilename(title=title)

    def _resolve_dapi_channel(self, path, role):
        return resolve_dapi_channel(path)

    def _build_ui(self):
        # uniform group keeps both columns equal width, so the H&E button and
        # the DAPI/H&E-FG buttons are the same width even before any image loads
        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=1, uniform="cols")

        # ---- selectors: moving always left, fixed always right ----
        self.moving_selectors = tk.Frame(self)
        self.moving_selectors.grid(row=0, column=0, sticky="new", padx=5, pady=5)
        self.moving_heading = tk.Label(self.moving_selectors, text="Moving (foreground)")
        self.moving_heading.pack(side="top", fill="x", pady=(0, 6))
        self.btn_dapi = tk.Button(self.moving_selectors, text="Select DAPI Image",
                                  command=self.on_select_dapi)
        self.btn_moving_he = tk.Button(self.moving_selectors, text="Select H&E Image (Foreground)",
                                 command=self.on_select_moving_he)
        if self.forced_modality in (None, "dapi"):
            self.btn_dapi.pack(side="top", fill="x", pady=(0, 3))
        if self.forced_modality in (None, "he"):
            self.btn_moving_he.pack(side="top", fill="x")

        self.fixed_selectors = tk.Frame(self)
        self.fixed_selectors.grid(row=0, column=1, sticky="new", padx=5, pady=5)
        self.fixed_heading = tk.Label(self.fixed_selectors, text="Fixed (background)")
        self.fixed_heading.pack(side="top", fill="x", pady=(0, 6))
        self.btn_he = tk.Button(self.fixed_selectors, text="Select H&E Image",
                                command=self.select_he)
        if self.forced_fixed_modality in (None, "he"):
            self.btn_he.pack(side="top", fill="x", pady=(0, 3))
        self.btn_dapi_bg = tk.Button(self.fixed_selectors,
                                     text="Select DAPI Image (Background)",
                                     command=self.select_fixed_dapi)
        if self.forced_fixed_modality in (None, "dapi"):
            self.btn_dapi_bg.pack(side="top", fill="x")

        # ---- image panels: original + threshold/LUT preview ----
        self.mv_orig_label = self._panel(1, 0)
        self.mv_proc_label = self._panel(2, 0)
        self.he_orig_label = self._panel(1, 1)
        self.he_mask_label = self._panel(2, 1)

        # ---- HE slider (built when an H&E image is selected) ----
        self.he_thr = tk.IntVar(value=240)
        self.he_slider_frame = tk.Frame(self)
        self.he_slider_frame.grid(row=3, column=1, padx=5, pady=6)
        self.he_slider_frame.grid_remove()

        # ---- moving slider (built when a DAPI/H&E-FG image is selected) ----
        self.mv_slider_frame = tk.Frame(self)
        self.mv_slider_frame.grid(row=3, column=0, padx=5, pady=6)
        self.mv_slider_frame.grid_remove()
        self.mv_thr = tk.IntVar(value=300)

        # ---- moving orientation buttons (shown once a moving image is loaded) ----
        # 2x2 grid so this row stays within the tile column width -> the window
        # width is the same whether DAPI/H&E-FG or H&E is loaded first.
        self.orient_frame = tk.Frame(self)
        self.orient_frame.grid(row=4, column=0, padx=5, pady=5, sticky="we")
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
        if self.he_orig is not None or self.moving_modality is not None:
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
            (self.moving_modality == "dapi" and self.dapi_lut_img is not None) or
            (self.moving_modality == "he" and self.moving_he_mask_img is not None)
        )
        if ready:
            self.confirm_btn.grid()
            self.confirm_btn.config(state=tk.NORMAL)
        else:
            self.confirm_btn.config(state=tk.DISABLED)

    # -----------------------------
    # H&E (fixed)
    # -----------------------------
    def _paired_stage1_levels(self):
        if self.he_path is None or self.moving_path is None:
            return None, None
        return select_paired_image_levels(
            self.moving_path, self.he_path,
            requested_moving_level=None, requested_fixed_level=None,
        )

    def _ensure_paired_levels(self):
        """Replace provisional preview levels using the CLI's joint selector.

        Reloading the moving side preserves the user's orientation through
        _reload_moving_level(). There is no separate GUI level policy.
        """
        moving_level, fixed_level = self._paired_stage1_levels()
        if moving_level is None or fixed_level is None:
            raise ValueError("Select both moving and fixed images before saving.")
        if fixed_level != self.he_level:
            self._reload_he_level(fixed_level)
        if moving_level != self.moving_level:
            self._reload_moving_level(moving_level)
        print(f"[Stage1-GUI] CLI auto levels: moving={self.moving_level}, "
              f"fixed={self.he_level}", flush=True)

    def _reload_he_level(self, level):
        # Historical he_* widget/state names represent the fixed side.
        if self.fixed_modality == "dapi":
            self.fixed_channel_selection = self._resolve_dapi_channel(self.he_path, "fixed")
            self.he_orig, self.he_level = read_image(
                self.he_path, keep_16bit=True, force_rgb=False,
                channel=int(self.fixed_channel_selection["index"]), level=level,
            )
            self.he_h_proc = None
            from gui_fail_safe.dapi_lut_window import auto_lut_window
            self.fixed_dapi_window = auto_lut_window(self.he_orig)
            self._fixed_slider(
                "Fixed DAPI LUT Threshold", 0, self.fixed_dapi_window.slider_max,
                self.fixed_dapi_window.slider_step, self.fixed_dapi_window.cutoff,
            )
            self.update_he(int(self.he_thr.get()))
            return
        self.he_orig, self.he_level = read_image(
            self.he_path, channel="he", level=level
        )
        self.he_h_proc = enhance_hematoxylin_channel(
            extract_hematoxylin_channel(self.he_orig)
        )
        self._set_tile(self.he_orig_label, self.he_orig)
        self.update_he(int(self.he_thr.get()))

    def _reload_moving_level(self, level):
        orientation_case = infer_orientation_case(self.moving_gui_affine)
        if self.moving_modality == "dapi":
            self.moving_channel_selection = self._resolve_dapi_channel(self.moving_path, "moving")
            image, self.moving_level = read_image(
                self.moving_path,
                keep_16bit=True,
                force_rgb=False,
                channel=int(self.moving_channel_selection["index"]),
                level=level,
            )
            self.dapi_img_view = image.copy()
        else:
            self.moving_he_orig, self.moving_level = read_image(
                self.moving_path, channel="he", level=level
            )
            self.moving_he_h_proc = enhance_hematoxylin_channel(
                extract_hematoxylin_channel(self.moving_he_orig)
            )

        image = self.dapi_img_view if self.moving_modality == "dapi" else self.moving_he_orig
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = image.shape[:2]
        orientation_steps = {
            0: (),
            1: ("rot_cw",),
            2: ("rot_cw", "rot_cw"),
            3: ("rot_ccw",),
            4: ("flip_v",),
            5: ("flip_h",),
            6: ("rot_cw", "flip_h"),
            7: ("rot_cw", "flip_v"),
        }
        for step in orientation_steps[orientation_case]:
            self._orient(step)
        if not orientation_steps[orientation_case]:
            self.update_moving(int(self.mv_thr.get()))

    def select_he(self):
        path = self._ask_image_path("Select H&E Image")
        if not path:
            return
        self.fixed_modality = "he"
        self.fixed_channel_selection = None
        self.fixed_dapi_lut = None
        self.fixed_dapi_window = None
        self.he_path = path
        moving_level, he_level = self._paired_stage1_levels()
        self.he_orig, self.he_level = read_image(
            path, channel="he", level=he_level
        )
        self.he_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.he_orig))
        self._fixed_slider("H&E Threshold", 150, 255, 5, 240)
        self.he_slider_frame.grid()
        self._set_tile(self.he_orig_label, self.he_orig)
        self.update_he(int(self.he_thr.get()))
        if moving_level is not None and moving_level != self.moving_level:
            self._reload_moving_level(moving_level)
        self._fit_window()

    def _fixed_slider(self, label, lo, hi, step, default):
        for child in self.he_slider_frame.winfo_children():
            child.destroy()
        self.he_thr.set(default)
        tk.Label(self.he_slider_frame, text=label).pack(anchor="w")
        tk.Scale(self.he_slider_frame, from_=lo, to=hi, orient=tk.HORIZONTAL,
                 resolution=step, showvalue=True, length=TILE_SIZE[0] - 40,
                 variable=self.he_thr,
                 command=lambda v: self.update_he(int(v))).pack(fill="x")
        self.he_slider_frame.grid()

    def select_fixed_dapi(self):
        path = self._ask_image_path("Select DAPI Image (Background)")
        if not path:
            return
        self.fixed_modality = "dapi"
        self.he_path = path
        moving_level, fixed_level = self._paired_stage1_levels()
        self._reload_he_level(fixed_level)
        if moving_level is not None and moving_level != self.moving_level:
            self._reload_moving_level(moving_level)
        self._fit_window()

    def update_he(self, threshold):
        if self.fixed_modality == "dapi":
            if self.he_orig is None:
                return
            from gui_fail_safe.dapi_lut_window import clip_to_window
            self.fixed_dapi_lut = dapi_to_lut_rgb(
                clip_to_window(self.he_orig, self.fixed_dapi_window.high),
                LUT, threshold=int(threshold),
            )
            self.he_mask_img = self.fixed_dapi_lut
            self._set_tile(self.he_orig_label, normalize_to_uint8(self.he_orig))
            self._set_tile(self.he_mask_label,
                           cv2.cvtColor(self.fixed_dapi_lut, cv2.COLOR_BGR2RGB))
            self._update_confirm_state()
            return
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
        path = self._ask_image_path("Select DAPI Image")
        if not path:
            return
        self.moving_modality = "dapi"
        self.moving_path = path
        moving_level, he_level = self._paired_stage1_levels()
        if he_level is not None and he_level != self.he_level:
            self._reload_he_level(he_level)
        self.moving_channel_selection = self._resolve_dapi_channel(path, "moving")
        img, self.moving_level = read_image(
            path,
            keep_16bit=True,
            force_rgb=False,
            channel=int(self.moving_channel_selection["index"]),
            level=moving_level,
        )
        self.dapi_img_view = img.copy()
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = self.dapi_img_view.shape[:2]
        self.moving_he_orig = None; self.moving_he_mask_img = None
        self._rebuild_moving_slider(*self._dapi_slider_spec())
        self._set_orientation_enabled(True)
        self.update_moving(int(self.mv_thr.get()))
        self._fit_window()

    def _dapi_slider_spec(self):
        """(label, lo, hi, step, default) for the DAPI LUT threshold slider."""
        default = resolve_dapi_lut_threshold(self.dapi_img_view)
        dtype = np.asarray(self.dapi_img_view).dtype
        if np.issubdtype(dtype, np.integer) and np.iinfo(dtype).max <= 255:
            return ("DAPI LUT Threshold", 0, int(np.iinfo(dtype).max), 1, default)
        return ("DAPI LUT Threshold", 0, max(2000, default * 2), 10, default)

    # -----------------------------
    # Moving: H&E-FG
    # -----------------------------
    def on_select_moving_he(self):
        path = self._ask_image_path("Select H&E Image (Foreground)")
        if not path:
            return
        self.moving_modality = "he"
        self.moving_path = path
        moving_level, he_level = self._paired_stage1_levels()
        if he_level is not None and he_level != self.he_level:
            self._reload_he_level(he_level)
        self.moving_he_orig, self.moving_level = read_image(
            path, channel="he", level=moving_level
        )
        self.moving_he_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.moving_he_orig))
        self.moving_gui_affine = np.eye(3, dtype=np.float32)
        self.moving_gui_shape = self.moving_he_orig.shape[:2]
        self.dapi_img_view = None; self.dapi_lut_img = None
        self._rebuild_moving_slider("H&E-FG Threshold", 150, 255, 5, 240)
        self._set_orientation_enabled(True)
        self.update_moving(int(self.mv_thr.get()))
        self._fit_window()

    def update_moving(self, threshold):
        if self.moving_modality == "dapi":
            if self.dapi_img_view is None:
                return
            self.dapi_lut_img = self._dapi_lut_image(int(threshold))
            gray = self.dapi_img_view[..., 0] if self.dapi_img_view.ndim == 3 else self.dapi_img_view
            self._set_tile(self.mv_orig_label, normalize_to_uint8(gray))
            self._set_tile(self.mv_proc_label, cv2.cvtColor(self.dapi_lut_img, cv2.COLOR_BGR2RGB))
        elif self.moving_modality == "he":
            if self.moving_he_h_proc is None:
                return
            _, m = cv2.threshold(self.moving_he_h_proc, int(threshold), 255, cv2.THRESH_BINARY)
            self.moving_he_mask_img = m.astype(np.uint8)
            self._set_tile(self.mv_orig_label, self.moving_he_orig)
            self._set_tile(self.mv_proc_label, self.moving_he_mask_img)
        self._update_confirm_state()

    def _dapi_lut_image(self, threshold):
        """The LUT picture shown for the DAPI image, and saved as 1_dapi_lut.png."""
        return dapi_to_lut_rgb(self.dapi_img_view, LUT, threshold=int(threshold))

    # -----------------------------
    # Orientation (moving)
    # -----------------------------
    def _orient(self, kind):
        if self.moving_modality is None or self.moving_gui_shape is None:
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

        if self.moving_modality == "dapi":
            self.dapi_img_view = xform(self.dapi_img_view)
        else:
            self.moving_he_orig = xform(self.moving_he_orig)
            self.moving_he_h_proc = enhance_hematoxylin_channel(extract_hematoxylin_channel(self.moving_he_orig))
        if kind in ("rot_cw", "rot_ccw"):
            self.moving_gui_shape = (W, H)
        self.update_moving(int(self.mv_thr.get()))

    # -----------------------------
    # Save
    # -----------------------------
    def _on_saved(self, info):
        """Hook that runs after images_info.json is written. Nothing by default."""

    def _saved_message(self):
        return (f"Stage 1 saved (pair={run_roles.pair_for(self.moving_modality, self.fixed_modality).name}) to:\n{self.run_dir}\n\n"
                f"Return to the pipeline and run Stage 2: Manual Alignment.")

    def confirm_and_save(self):
        if self.he_mask_img is None or self.he_orig is None:
            messagebox.showerror("Error", "Load and configure the fixed image first.")
            return
        if self.moving_modality is None:
            messagebox.showerror("Error", "Select a DAPI or H&E-FG image first.")
            return
        self._ensure_paired_levels()
        rd = self.run_dir
        pair = run_roles.pair_for(self.moving_modality, self.fixed_modality)
        moving_channel = (
            {
                "channel_index": int(self.moving_channel_selection["index"]),
                "channel_name": str(self.moving_channel_selection.get("name") or ""),
                "channel_source": str(self.moving_channel_selection.get("source") or ""),
            }
            if self.moving_modality == "dapi" and self.moving_channel_selection
            else {}
        )
        moving = run_roles.image_role(pair, "moving", self.moving_path, self.moving_level,
            threshold=self.mv_thr.get(), orientation_case=infer_orientation_case(self.moving_gui_affine),
            gui_affine=self.moving_gui_affine.tolist(), **moving_channel)
        fixed_channel = (
            {
                "channel_index": int(self.fixed_channel_selection["index"]),
                "channel_name": str(self.fixed_channel_selection.get("name") or ""),
                "channel_source": str(self.fixed_channel_selection.get("source") or ""),
            } if self.fixed_modality == "dapi" and self.fixed_channel_selection else {}
        )
        fixed = run_roles.image_role(pair, "fixed", self.he_path, self.he_level,
                                    threshold=self.he_thr.get(), **fixed_channel)
        if self.fixed_modality == "dapi":
            cv2.imwrite(str(rd / pair.stage1_image("fixed")),
                        np.ascontiguousarray(self.he_orig))
            cv2.imwrite(str(rd / f"1_{fixed.token}_lut.png"), self.fixed_dapi_lut)
        else:
            Image.fromarray(self.he_orig).save(rd / pair.stage1_image("fixed"))
            cv2.imwrite(str(rd / f"1_{fixed.token}_threshold_mask.png"), self.he_mask_img)
        info = {"RUN_ID": self.run_id, **run_roles.flat_keys(pair, moving, fixed),
                "roles": run_roles.roles_block(pair, moving, fixed)}

        if self.moving_modality == "dapi":
            dapi_gray = (
                self.dapi_img_view[..., 0]
                if self.dapi_img_view.ndim == 3
                else self.dapi_img_view
            )
            # Registration consumes this intensity-preserving image. The LUT
            # artifact remains available for display and overlay rendering.
            cv2.imwrite(
                str(rd / pair.stage1_image("moving")),
                np.ascontiguousarray(dapi_gray),
            )
            cv2.imwrite(str(rd / f"1_{moving.token}_lut.png"), self.dapi_lut_img)
        else:
            Image.fromarray(self.moving_he_orig).save(rd / pair.stage1_image("moving"))
            cv2.imwrite(str(rd / f"1_{moving.token}_threshold_mask.png"), self.moving_he_mask_img)

        with open(rd / "images_info.json", "w") as f:
            json.dump(info, f, indent=2)
        self._on_saved(info)
        messagebox.showinfo("Saved", self._saved_message())
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    run_roles.add_pair_arguments(parser)
    args = parser.parse_args()
    pair = run_roles.pair_from_args(args) if args.pair or args.mode else None
    app = UnifiedStage1(Path(args.run_dir).resolve(), pair=pair)
    app.mainloop()
