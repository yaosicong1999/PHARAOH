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
# Orientation (same as your step4)
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


def make_he_like_mask_from_rgb_threshold(
    rgb_img: np.ndarray,
    intensity_threshold: float,
    upscale: int = 2,
    n_smooth: int = 2,
):
    """
    Unified mask logic for both HE0 and HE.
    Returns uint8 mask 0/255 with nuclei as white.
    """
    from my_utils import segment_super_dark_nuclei_full

    if rgb_img is None:
        return None
    if rgb_img.ndim == 2:
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)

    thr = float(intensity_threshold)
    thr = min(1.0, max(0.0, thr))

    _, mask_dark = segment_super_dark_nuclei_full(
        rgb_img,
        upsample_scale=upscale,
        n_smooth=n_smooth,
        intensity_threshold=thr
    )

    bw = (mask_dark.astype(np.uint8) * 255)
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
    tk.Label(root, text="HE0", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="HE0 mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E mask", font=("Helvetica", 15)).grid(row=0, column=3)

    he0_label = tk.Label(root)
    he0_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    he0_label.grid(row=1, column=0, padx=8, pady=8)
    he0_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=4)

    # -------------------------
    # Load pilot meta
    # -------------------------
    he0_json = pilot_dir / "pilot_he0_tile_info.json"
    he_json = pilot_dir / "pilot_he_tile_info.json"

    if not he0_json.exists():
        messagebox.showerror("Missing", f"Cannot find {he0_json}")
        root.destroy()
        return

    he0_info = json.load(open(he0_json, "r"))
    he_info = json.load(open(he_json, "r")) if he_json.exists() else {}

    keys = sorted(he0_info.keys())

    # -------------------------
    # Threshold controls
    # -------------------------
    he0_thr_var = tk.StringVar(value="0.70")
    he_thr_var = tk.StringVar(value="0.70")

    _last_good = {"he0": "0.70", "he": "0.70"}

    # -------------------------
    # Persist params
    # -------------------------
    params_path = pilot_dir / "pilot_output_parameters.json"

    def _load_existing_params():
        if params_path.exists():
            try:
                return json.load(open(params_path, "r"))
            except Exception:
                return {}
        return {}

    def _save_pilot_params(he0_thr: float, he_thr: float):
        data = _load_existing_params()
        data["mask_preview"] = {
            "he0_intensity_threshold": float(he0_thr),
            "he_intensity_threshold": float(he_thr),
            "he0_intensity_threshold_range": [0.0, 1.0],
            "he0_intensity_threshold_step": 0.01,
            "he_intensity_threshold_range": [0.0, 1.0],
            "he_intensity_threshold_step": 0.01,
        }
        with open(params_path, "w") as f:
            json.dump(data, f, indent=2)

    def _parse_he0_thr():
        s = he0_thr_var.get().strip()
        try:
            v = float(s)
        except Exception:
            raise ValueError("HE0 intensity_threshold must be a float in [0, 1].")
        if not (0.0 <= v <= 1.0):
            raise ValueError("HE0 intensity_threshold must be within [0, 1].")
        return float(f"{v:.2f}")

    def _parse_he_thr():
        s = he_thr_var.get().strip()
        try:
            v = float(s)
        except Exception:
            raise ValueError("HE intensity_threshold must be a float in [0, 1].")
        if not (0.0 <= v <= 1.0):
            raise ValueError("HE intensity_threshold must be within [0, 1].")
        return float(f"{v:.2f}")

    def _apply_controls(which=None):
        try:
            if which in (None, "he0"):
                v = _parse_he0_thr()
                he0_thr_var.set(f"{v:.2f}")
                _last_good["he0"] = f"{v:.2f}"

            if which in (None, "he"):
                v = _parse_he_thr()
                he_thr_var.set(f"{v:.2f}")
                _last_good["he"] = f"{v:.2f}"

            _save_pilot_params(
                he0_thr=float(he0_thr_var.get()),
                he_thr=float(he_thr_var.get()),
            )

            update_images()

        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            if which in (None, "he0"):
                he0_thr_var.set(_last_good["he0"])
            if which in (None, "he"):
                he_thr_var.set(_last_good["he"])

    # -------------------------
    # Image updater
    # -------------------------
    def update_images():
        k = keys[idx[0]]

        # ---- HE0 image ----
        he0_fn = he0_info[k].get("filename", None)
        he0_path = str(pilot_dir / he0_fn) if he0_fn else ""
        he0_pil = load_optional(
            he0_path,
            display_size,
            bg_color,
            is_mask=False,
            case_id=case_id
        )

        # ---- HE0 mask ----
        he0_mask_pil = _placeholder_img(display_size, text="NOT AVAILABLE NOW")
        if he0_fn and os.path.exists(he0_path):
            he0_bgr = cv2.imread(he0_path, cv2.IMREAD_COLOR)
            if he0_bgr is not None:
                he0_rgb = cv2.cvtColor(he0_bgr, cv2.COLOR_BGR2RGB)
                if case_id is not None:
                    he0_rgb = apply_orientation_to_tile(he0_rgb, case_id)

                he0_thr = _parse_he0_thr()
                he0_mask = make_he_like_mask_from_rgb_threshold(
                    he0_rgb,
                    intensity_threshold=he0_thr,
                    upscale=2,
                    n_smooth=2,
                )
                if he0_mask is not None:
                    he0_mask_rgb = cv2.cvtColor(he0_mask, cv2.COLOR_GRAY2RGB)
                    he0_mask_pil = pad_to_fixed_size(
                        Image.fromarray(he0_mask_rgb),
                        display_size,
                        bg_color,
                    )

        # ---- HE image ----
        he_fn = he_info.get(k, {}).get("filename", None)
        he_path = str(pilot_dir / he_fn) if he_fn else ""
        he_pil = load_optional(
            he_path,
            display_size,
            bg_color,
            is_mask=False,
            case_id=None
        )

        # ---- HE mask ----
        he_mask_pil = _placeholder_img(display_size, text="NOT AVAILABLE NOW")
        if he_fn and os.path.exists(he_path):
            he_bgr = cv2.imread(he_path, cv2.IMREAD_COLOR)
            if he_bgr is not None:
                he_rgb = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)
                he_thr = _parse_he_thr()

                he_mask = make_he_like_mask_from_rgb_threshold(
                    he_rgb,
                    intensity_threshold=he_thr,
                    upscale=2,
                    n_smooth=2,
                )
                if he_mask is not None:
                    he_mask_rgb = cv2.cvtColor(he_mask, cv2.COLOR_GRAY2RGB)
                    he_mask_pil = pad_to_fixed_size(
                        Image.fromarray(he_mask_rgb),
                        display_size,
                        bg_color,
                    )

        # ---- render ----
        imgs = [
            ImageTk.PhotoImage(he0_pil),
            ImageTk.PhotoImage(he0_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]
        labels = [he0_label, he0_mask_label, he_label, he_mask_label]

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
    btn_reload.grid(row=3, column=3, pady=(0, 6))

    # -------------------------
    # Controls row
    # -------------------------
    ctrl = tk.Frame(root)
    ctrl.grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=(2, 10))

    tk.Label(ctrl, text="HE0 intensity_threshold:", font=("Helvetica", 12)).grid(row=0, column=0, sticky="e")
    sp_he0 = ttk.Spinbox(
        ctrl,
        from_=0.0, to=1.0,
        increment=0.01,
        format="%.2f",
        textvariable=he0_thr_var,
        width=10,
        command=lambda: _apply_controls("he0"),
    )
    sp_he0.grid(row=0, column=1, sticky="w", padx=(6, 18))

    tk.Label(ctrl, text="HE intensity_threshold:", font=("Helvetica", 12)).grid(row=0, column=2, sticky="e")
    sp_he = ttk.Spinbox(
        ctrl,
        from_=0.0, to=1.0,
        increment=0.01,
        format="%.2f",
        textvariable=he_thr_var,
        width=10,
        command=lambda: _apply_controls("he"),
    )
    sp_he.grid(row=0, column=3, sticky="w", padx=(6, 18))

    btn_apply = ttk.Button(ctrl, text="Apply", command=lambda: _apply_controls(None))
    btn_apply.grid(row=0, column=4, sticky="w")

    sp_he0.bind("<Return>", lambda e: _apply_controls("he0"))
    sp_he.bind("<Return>", lambda e: _apply_controls("he"))
    sp_he0.bind("<FocusOut>", lambda e: _apply_controls("he0"))
    sp_he.bind("<FocusOut>", lambda e: _apply_controls("he"))

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
    case_id = int(info.get("HE0_orientation_case", 0))

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