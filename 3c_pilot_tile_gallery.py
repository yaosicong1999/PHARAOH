from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np
import cv2
import os
import json
import sys


# -----------------------------
# Orientation (same as your 4)
# -----------------------------
def apply_orientation_to_tile(img, case_id):
    if case_id == 0:
        return img
    if case_id == 1:      # rot90 CW
        return np.rot90(img, k=3)
    if case_id == 2:      # rot180
        return np.rot90(img, k=2)
    if case_id == 3:      # rot90 CCW
        return np.rot90(img, k=1)
    if case_id == 4:      # flip vertical
        return np.flipud(img)
    if case_id == 5:      # flip horizontal
        return np.fliplr(img)
    if case_id == 6:      # rot90 CW + flip H (transpose)
        if img.ndim == 3:
            return np.transpose(np.rot90(img, k=3), (1, 0, 2))
        else:
            return np.transpose(np.rot90(img, k=3))
    if case_id == 7:      # rot90 CW + flip V
        return np.flipud(np.rot90(img, k=3))
    raise ValueError(f"Unknown orientation case: {case_id}")


def pad_to_fixed_size(img_pil, display_size=(256, 256), bg_color=(255, 255, 255)):
    img_pil = ImageOps.contain(img_pil, display_size)
    canvas = Image.new("RGB", display_size, bg_color)
    x = (display_size[0] - img_pil.width) // 2
    y = (display_size[1] - img_pil.height) // 2
    canvas.paste(img_pil, (x, y))
    return canvas


def _placeholder_img(display_size=(256, 256), text="NOT AVAILABLE NOW"):
    img = np.full((display_size[1], display_size[0], 3), 240, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (display_size[0] - tw) // 2
    y = (display_size[1] + th) // 2
    cv2.putText(img, text, (x, y), font, font_scale, (120, 120, 120), thickness, cv2.LINE_AA)
    return Image.fromarray(img)


def load_optional(path, display_size=(256, 256), bg_color=(255, 255, 255),
                  is_mask=False, case_id=None):
    """
    path: file path to load
    is_mask: read as grayscale and convert to RGB
    case_id: if not None, apply orientation
    """
    if path and os.path.exists(path):
        if is_mask:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read mask: {path}")
            if case_id is not None:
                img = apply_orientation_to_tile(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if case_id is not None:
                img = apply_orientation_to_tile(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
    else:
        pil = _placeholder_img(display_size, text="NOT AVAILABLE NOW")

    return pad_to_fixed_size(pil, display_size=display_size, bg_color=bg_color)


def make_dapi_mask_from_u8_step5_like(
    dapi_u8: np.ndarray,
    thr_offset: int = 0,
    min_area_factor: float = 10e-5,
    connectivity: int = 8,
    invert: bool = True,
    upscale: int = 2,
):
    """
    Replicate 5_generate_nuclei_masks.py DAPI logic.
    Input: uint8 grayscale (H,W)
    Output: uint8 mask 0/255 (white bg, black nuclei if invert=True)
    """
    from my_utils import upsample_tile, fill_holes_binary, remove_small_components

    if dapi_u8 is None:
        return None
    if dapi_u8.ndim == 3:
        dapi_u8 = cv2.cvtColor(dapi_u8, cv2.COLOR_RGB2GRAY)
    if dapi_u8.dtype != np.uint8:
        dapi_u8 = dapi_u8.astype(np.uint8)

    # upscale
    img = upsample_tile(dapi_u8, upscale)

    # 1) Otsu + offset
    otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(int(otsu_thr) + int(thr_offset), 0, 255))
    _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

    # 2) fill holes
    mask = fill_holes_binary(mask)

    # 3) remove small components
    min_area = min_area_factor * (img.shape[0] ** 2)
    min_area = min_area * (upscale ** 2)
    mask, _ = remove_small_components(
        mask, min_area=int(min_area), connectivity=int(connectivity)
    )

    # 4) invert (white bg, black nuclei)
    if invert:
        mask = 255 - mask

    # downsample back to tile size (for display alignment)
    if upscale != 1:
        mask = cv2.resize(
            mask,
            (dapi_u8.shape[1], dapi_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask.astype(np.uint8)


def make_he_mask_from_rgb_threshold(
    he_rgb: np.ndarray,
    intensity_threshold: float,
    upscale: int = 2,
    n_smooth: int = 2,
):
    """
    HE mask like step5, but intensity_threshold is controllable.
    Returns uint8 mask 0/255 with white background and black nuclei.
    """
    from my_utils import segment_super_dark_nuclei_full

    if he_rgb is None:
        return None
    if he_rgb.ndim == 2:
        he_rgb = cv2.cvtColor(he_rgb, cv2.COLOR_GRAY2RGB)

    thr = float(intensity_threshold)
    thr = min(1.0, max(0.0, thr))

    _, mask_dark = segment_super_dark_nuclei_full(
        he_rgb, upsample_scale=upscale, n_smooth=n_smooth, intensity_threshold=thr
    )

    bw = (mask_dark.astype(np.uint8) * 255)  # nuclei=255
    return bw

def show_pilot_gallery(
    run_dir: Path,
    pilot_dir: Path,
    case_id: int,
    display_size=(256, 256),
    bg_color=(255, 255, 255),
):
    root = tk.Tk()
    root.title("STEP 3c: Pilot Tile Gallery")

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("Gallery.TButton", font=("Helvetica", 12), padding=(6, 6))

    idx = [0]

    # -------------------------
    # Layout
    # -------------------------
    tk.Label(root, text="DAPI (LUT)", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="DAPI (Intensity u8)", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="DAPI mask", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E (rectified/bbox)", font=("Helvetica", 15)).grid(row=0, column=3)
    tk.Label(root, text="H&E mask", font=("Helvetica", 15)).grid(row=0, column=4)

    dapi_label = tk.Label(root)
    dapi_int_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)
    he_rect_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    dapi_label.grid(row=1, column=0, padx=8, pady=8)
    dapi_int_label.grid(row=1, column=1, padx=8, pady=8)
    dapi_mask_label.grid(row=1, column=2, padx=8, pady=8)
    he_rect_label.grid(row=1, column=3, padx=8, pady=8)
    he_mask_label.grid(row=1, column=4, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # -------------------------
    # Load pilot meta
    # -------------------------
    lut_json = pilot_dir / "pilot_dapi_tile_info_lut.json"
    int_json = pilot_dir / "pilot_dapi_tile_info_intensity.json"
    he_rect_json = pilot_dir / "pilot_he_tile_info.json"  # produced by 3b

    if not lut_json.exists():
        messagebox.showerror("Missing", f"Cannot find {lut_json}")
        root.destroy()
        return

    dapi_lut_info = json.load(open(lut_json, "r"))
    dapi_int_info = json.load(open(int_json, "r")) if int_json.exists() else {}
    he_rect_info = json.load(open(he_rect_json, "r")) if he_rect_json.exists() else {}

    keys = sorted(dapi_lut_info.keys())

    # -------------------------
    # Cutoff controls state
    # -------------------------
    dapi_thr_var = tk.StringVar(value="30")  # THR_OFFSET, default 100
    he_thr_var = tk.StringVar(value="0.70")  # intensity_threshold, default 0.8

    _last_good = {"dapi": "30", "he": "0.70"}

    # -------------------------
    # Persist parameters to pilot_output_parameters.json
    # -------------------------
    params_path = pilot_dir / "pilot_output_parameters.json"

    def _load_existing_params():
        if params_path.exists():
            try:
                return json.load(open(params_path, "r"))
            except Exception:
                return {}
        return {}

    def _save_pilot_params(dapi_thr_offset: int, he_intensity_thr: float):
        data = _load_existing_params()
        # keep a clean structure (append / update)
        data["mask_preview"] = {
            "dapi_thr_offset": int(dapi_thr_offset),
            "he_intensity_threshold": float(he_intensity_thr),
            "dapi_thr_offset_range": [-100, 100],
            "dapi_thr_offset_step": 5,
            "he_intensity_threshold_range": [0.0, 1.0],
            "he_intensity_threshold_step": 0.01,
        }
        with open(params_path, "w") as f:
            json.dump(data, f, indent=2)

    def _parse_dapi_thr():
        s = dapi_thr_var.get().strip()
        try:
            v = int(float(s))  # allow "10.0"
        except Exception:
            raise ValueError("DAPI THR_OFFSET must be a number (e.g. -20, 0, 30).")
        if not (-100 <= v <= 100):
            raise ValueError("DAPI THR_OFFSET must be between -100 and 100 (relative to Otsu).")
        # snap to step=5 (optional but recommended)
        if v % 5 != 0:
            v = int(round(v / 5) * 5)
        return v

    def _parse_he_thr():
        s = he_thr_var.get().strip()
        try:
            v = float(s)
        except Exception:
            raise ValueError("HE intensity_threshold must be a float in [0, 1].")
        if not (0.0 <= v <= 1.0):
            raise ValueError("HE intensity_threshold must be within [0, 1].")
        # normalize display to 2 decimals
        return float(f"{v:.2f}")

    def _apply_controls(which=None):
        """
        Validate spinbox values, write back normalized string, refresh images.
        """
        try:
            if which in (None, "dapi"):
                v = _parse_dapi_thr()
                dapi_thr_var.set(str(v))
                _last_good["dapi"] = str(v)

            if which in (None, "he"):
                v = _parse_he_thr()
                he_thr_var.set(f"{v:.2f}")
                _last_good["he"] = f"{v:.2f}"

            # save to pilot_output_parameters.json
            _save_pilot_params(
                dapi_thr_offset=int(dapi_thr_var.get()),
                he_intensity_thr=float(he_thr_var.get()),
            )

            update_images()

        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            # rollback
            if which in (None, "dapi"):
                dapi_thr_var.set(_last_good["dapi"])
            if which in (None, "he"):
                he_thr_var.set(_last_good["he"])
    # -------------------------
    # Image updater
    # -------------------------
    def update_images():
        k = keys[idx[0]]

        # ---- DAPI LUT ----
        dapi_fn = dapi_lut_info[k].get("filename", None)
        dapi_path = str(pilot_dir / dapi_fn) if dapi_fn else ""
        dapi_pil = load_optional(dapi_path, display_size, bg_color, is_mask=False, case_id=case_id)

        # ---- DAPI intensity u8 ----
        int_fn = dapi_int_info.get(k, {}).get("filename_dapi_u8", None)
        int_path = str(pilot_dir / int_fn) if int_fn else ""
        int_pil = load_optional(int_path, display_size, bg_color, is_mask=False, case_id=case_id)

        # ---- DAPI mask (generated, step5-like) ----
        dapi_mask_pil = _placeholder_img(display_size, text="NOT AVAILABLE NOW")
        if int_fn and os.path.exists(int_path):
            dapi_u8 = cv2.imread(int_path, cv2.IMREAD_GRAYSCALE)
            if dapi_u8 is not None:
                if case_id is not None:
                    dapi_u8 = apply_orientation_to_tile(dapi_u8, case_id)

                dapi_thr = _parse_dapi_thr()  # 你的输入框：>=0 int

                dapi_mask = make_dapi_mask_from_u8_step5_like(
                    dapi_u8,
                    thr_offset=dapi_thr,
                    invert=True,
                    upscale=2,
                )

                if dapi_mask is not None:
                    dapi_mask_rgb = cv2.cvtColor(dapi_mask, cv2.COLOR_GRAY2RGB)
                    dapi_mask_pil = pad_to_fixed_size(
                        Image.fromarray(dapi_mask_rgb),
                        display_size,
                        bg_color,
                    )

        # ---- HE rectified/bbox (read file) ----
        he_fn = he_rect_info.get(k, {}).get("filename", None)
        he_path = str(pilot_dir / he_fn) if he_fn else ""
        he_rect_pil = load_optional(he_path, display_size, bg_color, is_mask=False, case_id=None)

        # ---- HE mask (generated from HE image) ----
        he_mask_pil = _placeholder_img(display_size, text="NOT AVAILABLE NOW")
        if he_fn and os.path.exists(he_path):
            he_bgr = cv2.imread(he_path, cv2.IMREAD_COLOR)
            if he_bgr is not None:
                he_rgb = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)
                he_thr = _parse_he_thr()

                he_mask = make_he_mask_from_rgb_threshold(
                    he_rgb,
                    intensity_threshold=he_thr,
                    upscale=2,
                    n_smooth=2,
                )
                if he_mask is not None:
                    he_mask_rgb = cv2.cvtColor(he_mask, cv2.COLOR_GRAY2RGB)
                    he_mask_pil = pad_to_fixed_size(Image.fromarray(he_mask_rgb), display_size, bg_color)


        # ---- render ----
        imgs = [
            ImageTk.PhotoImage(dapi_pil),
            ImageTk.PhotoImage(int_pil),
            ImageTk.PhotoImage(dapi_mask_pil),
            ImageTk.PhotoImage(he_rect_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]
        labels = [dapi_label, dapi_int_label, dapi_mask_label, he_rect_label, he_mask_label]
        for lbl, im in zip(labels, imgs):
            lbl.configure(image=im)
            lbl.image = im

        info_label.configure(text=f"{idx[0] + 1}/{len(keys)} | {k}")

    # -------------------------
    # Navigation
    # -------------------------
    def next_tile():
        idx[0] = (idx[0] + 1) % len(keys)
        update_images()

    def prev_tile():
        idx[0] = (idx[0] - 1) % len(keys)
        update_images()

    btn_prev = ttk.Button(root, text="⟨ Previous", command=prev_tile)
    btn_next = ttk.Button(root, text="Next ⟩", command=next_tile)
    btn_reload = ttk.Button(root, text="⟳ Refresh", style="Gallery.TButton", command=lambda: _apply_controls(None))

    btn_prev.grid(row=3, column=0, pady=(0, 6))
    btn_next.grid(row=3, column=2, pady=(0, 6))
    btn_reload.grid(row=3, column=4, pady=(0, 6))

    # -------------------------
    # Controls row (below nav)
    # -------------------------
    ctrl = tk.Frame(root)
    ctrl.grid(row=4, column=0, columnspan=5, sticky="ew", padx=8, pady=(2, 10))

    tk.Label(ctrl, text="DAPI THR_OFFSET:", font=("Helvetica", 12)).grid(row=0, column=0, sticky="e")
    sp_dapi = ttk.Spinbox(
        ctrl,
        from_=-100, to=100,
        increment=1,
        textvariable=dapi_thr_var,
        width=10,
        command=lambda: _apply_controls("dapi"),  # arrow click triggers
    )
    sp_dapi.grid(row=0, column=1, sticky="w", padx=(6, 18))

    tk.Label(ctrl, text="HE intensity_threshold:", font=("Helvetica", 12)).grid(row=0, column=2, sticky="e")
    sp_he = ttk.Spinbox(
        ctrl,
        from_=0.0, to=1.0,
        increment=0.01,
        format="%.2f",  # display nicely
        textvariable=he_thr_var,
        width=10,
        command=lambda: _apply_controls("he"),
    )
    sp_he.grid(row=0, column=3, sticky="w", padx=(6, 18))

    btn_apply = ttk.Button(ctrl, text="Apply", command=lambda: _apply_controls(None))
    btn_apply.grid(row=0, column=4, sticky="w")

    # Enter / focus-out triggers validate+refresh too
    sp_dapi.bind("<Return>", lambda e: _apply_controls("dapi"))
    sp_he.bind("<Return>", lambda e: _apply_controls("he"))
    sp_dapi.bind("<FocusOut>", lambda e: _apply_controls("dapi"))
    sp_he.bind("<FocusOut>", lambda e: _apply_controls("he"))

    # init
    update_images()
    root.mainloop()

def main():
    """
    Usage:
      python 3c_pilot_tile_gallery.py <RUN_DIR>
      or
      python 3c_pilot_tile_gallery.py <RUN_DIR> <PILOT_DIR>
    """
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 3c_pilot_tile_gallery.py <RUN_DIR> [PILOT_DIR]")

    run_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        pilot_dir = Path(sys.argv[2]).resolve()
    else:
        pilot_dir = run_dir / "pilot_tiles"

    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    info = json.load(open(info_path, "r"))
    case_id = int(info.get("DAPI_orientation_case", 0))

    if not pilot_dir.exists():
        raise FileNotFoundError(pilot_dir)

    show_pilot_gallery(
        run_dir=run_dir,
        pilot_dir=pilot_dir,
        case_id=case_id,
        display_size=(256, 256),
        bg_color=(255, 255, 255),
    )


if __name__ == "__main__":
    main()