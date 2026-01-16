# 0_gallery.py
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import os
import subprocess
import threading
import json
import sys
import time
import traceback

ORIENTATION_CASES = {
    0: np.array([[ 1,  0],
                 [ 0,  1]], np.float32),  # identity

    1: np.array([[ 0, -1],
                 [ 1,  0]], np.float32),  # rot90 CW

    2: np.array([[-1,  0],
                 [ 0, -1]], np.float32),  # rot180

    3: np.array([[ 0,  1],
                 [-1,  0]], np.float32),  # rot90 CCW

    4: np.array([[ 1,  0],
                 [ 0, -1]], np.float32),  # flip vertical (up-down)

    5: np.array([[-1,  0],
                 [ 0,  1]], np.float32),  # flip horizontal (left-right)

    6: np.array([[ 0,  1],
                 [ 1,  0]], np.float32),  # rot90 CW then flip H  (== transpose)

    7: np.array([[ 0, -1],
                 [-1,  0]], np.float32),  # rot90 CW then flip V  (== anti-transpose)
}
def apply_orientation_to_patch(img, case_id):
    """
    img: np.ndarray (H,W) or (H,W,3)
    case_id: int in [0..7]
    """
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



def show_nucleus_patch_in_memory(
        tile_id,
        nucleus_id,
        output_folder,
        case_id,
        display_size=(256, 256),
        bg_color=(255, 255, 255)
):
    assert len(tile_id) == len(nucleus_id)

    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )
    root.title("STEP 8: DAPI & H&E Patch Gallery")
    idx = [0]
    # ---------- Labels ----------
    tk.Label(root, text="DAPI", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="DAPI nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)

    dapi_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    dapi_label.grid(row=1, column=0, padx=8, pady=8)
    dapi_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # ---------- Utils ----------
    def pad_to_fixed_size(img_pil):
        img_pil = ImageOps.contain(img_pil, display_size)
        canvas = Image.new("RGB", display_size, bg_color)
        x = (display_size[0] - img_pil.width) // 2
        y = (display_size[1] - img_pil.height) // 2
        canvas.paste(img_pil, (x, y))
        return canvas

    def load_optional(path, is_mask=False, case_id=None):
        if os.path.exists(path):
            if is_mask:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read mask: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.full((display_size[1], display_size[0], 3), 240, np.uint8)
            text = "NOT AVAILABLE NOW"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = (display_size[0] - text_w) // 2
            y = (display_size[1] + text_h) // 2
            cv2.putText(
                img,
                text,
                (x, y),
                font,
                font_scale,
                (120, 120, 120),
                thickness,
                cv2.LINE_AA
            )
        return pad_to_fixed_size(Image.fromarray(img))


    # ---------- Core ----------
    def update_images():
        i = idx[0]
        tile_name = tile_id[i]
        nucleus_name = nucleus_id[i]

        # ----------------------------
        # resolve base & filenames
        # ----------------------------
        base = output_folder
        dapi_img_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_patch_overlay.png"
        he_img_name = f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png"
        # ----------------------------
        # DAPI image (from disk)
        # ----------------------------
        dapi_pil = load_optional(
            os.path.join(output_folder, dapi_img_name),
            is_mask=False,
            case_id=case_id
        )
        # ----------------------------
        # DAPI mask
        # ----------------------------
        dapi_mask_pil = load_optional(
            os.path.join(output_folder, f"{tile_name}_nucleus_{nucleus_name}_dapi_mask_overlay.png"),
            is_mask=True,
            case_id=case_id
        )
        # ----------------------------
        # HE image (from disk)
        # ----------------------------
        he_pil = load_optional(
            os.path.join(output_folder, he_img_name),
            is_mask=False
        )
        # ----------------------------
        # HE mask
        # ----------------------------
        he_mask_pil = load_optional(
            os.path.join(output_folder, f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png"),
            is_mask=True
        )
        # ----------------------------
        # Tk images
        # ----------------------------
        imgs = [
            ImageTk.PhotoImage(dapi_pil),
            ImageTk.PhotoImage(dapi_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]

        labels = [
            dapi_label,
            dapi_mask_label,
            he_label,
            he_mask_label,
        ]

        for lbl, im in zip(labels, imgs):
            lbl.config(image=im)
            lbl.image = im

        info_label.config(
            text=f"{i + 1}/{len(tile_id)} | {tile_name.split('_', 1)[0].capitalize()}"
        )

    def next_patch():
        idx[0] = (idx[0] + 1) % len(tile_id)
        update_images()

    def prev_patch():
        idx[0] = (idx[0] - 1) % len(tile_id)
        update_images()

    # ---------- Buttons ----------
    btn_prev = ttk.Button(root, text="⟨ Previous", command=prev_patch)
    btn_next = ttk.Button(root, text="Next ⟩", command=next_patch)
    btn_reload = ttk.Button(
        root,
        text="⟳ Refresh",
        style="Gallery.TButton",
        command=update_images
    )
    btn_prev.grid(row=3, column=0, sticky="w")
    btn_reload.grid(row=3, column=1, columnspan=2)
    btn_next.grid(row=3, column=3, sticky="e")
    update_images()
    root.mainloop()

def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 8_nucleus_gallery.py <patches_folder>")
    output_folder = sys.argv[1]

    tiles_folder = sys.argv[1]
    # tiles_folder = '/Users/sicongy/PycharmProjects/iStar/tkinter_version_v3/runs_202512182051/tiles/'
    output_folder = os.path.join(tiles_folder, "../nuclei_patches/")
    with open(os.path.join(output_folder, "nuclei_centroids_global.json"), "r") as f:
        nuclei_info = json.load(f)
    with open(os.path.join(output_folder, "../images_info.json"), "r") as f:
        case_id = json.load(f)['DAPI_orientation_case']

    tile_id = [x['tile_id'] for x in nuclei_info]
    nucleus_id = [x['nucleus_id'] for x in nuclei_info]

    show_nucleus_patch_in_memory(
        tile_id=tile_id,
        nucleus_id=nucleus_id,
        output_folder=output_folder,
        case_id=case_id
    )

if __name__ == "__main__":
    main()