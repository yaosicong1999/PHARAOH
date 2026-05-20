import json
import cv2
import os
import sys
import time
import numpy as np
from glob import glob
from PIL import Image
from my_utils import (
    read_image,
    segment_super_dark_nuclei_full,
    read_crop_patch,
)
import math
from pathlib import Path

def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

def apply_homography_xy(M3x3, x, y):
    M3x3 = np.asarray(M3x3, dtype=float)
    p = np.array([float(x), float(y), 1.0], dtype=float)
    q = M3x3 @ p
    w = q[2] if abs(q[2]) > 1e-12 else 1e-12
    return float(q[0] / w), float(q[1] / w)

def extract_level_to_read_level(extract_level: int) -> int:
    """
    convert step3 extract level convention to read_image level convention:
      extract 1 -> read 0
      extract 2 -> read 1
      extract 3 -> read 2
      ...
    """
    return max(0, int(extract_level) - 1)

def he_point_tile_to_image_coords(he_info, x_tile, y_tile):
    """
    Return point coordinates in the SAME coordinate system as the loaded HE image
    (i.e. HE_READ_LEVEL coordinates).

    If rectified:
        (x_tile, y_tile) is in rectified coords -> project back using M_rect_to_he
    Else:
        (x_tile, y_tile) is tile-local bbox coords -> x0/y0 add
    """
    meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}

    if meta.get("mode", None) == "rectified" and meta.get("M_rect_to_he", None) is not None:
        Minv = np.asarray(meta["M_rect_to_he"], dtype=float)
        x_img, y_img = apply_homography_xy(Minv, x_tile, y_tile)
        return x_img, y_img

    x_img = float(he_info["x0"]) + float(x_tile)
    y_img = float(he_info["y0"]) + float(y_tile)
    return x_img, y_img

def ensure_gray_uint16(x):
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., 0]
    return x.astype(np.uint16)

def stretch_to_uint8_percentile(raw16, p_low=1, p_high=99.7):
    raw16 = raw16.astype(np.float32)
    lo, hi = np.percentile(raw16, [p_low, p_high])
    if hi <= lo:
        return np.zeros(raw16.shape, np.uint8)
    vis8 = (raw16 - lo) * 255.0 / (hi - lo)
    return np.clip(vis8, 0, 255).astype(np.uint8)

def load_initial_affine_and_compute_he_patch(
    run_dir,
    images_info,
    he0_patch_len=100,
    margin=1.1
):
    run_dir = Path(run_dir)

    path_clicked = run_dir / "../clicked_blob_initial_alignment.json"
    path_manual  = run_dir / "../manual_initial_alignment.json"

    if path_clicked.exists():
        data = json.load(open(path_clicked, "r"))
        src = path_clicked.name
    elif path_manual.exists():
        data = json.load(open(path_manual, "r"))
        src = path_manual.name
    else:
        raise FileNotFoundError(
            "Neither clicked_blob_initial_alignment.json nor manual_initial_alignment.json found."
        )

    # ---- parse affine ----
    if "affine_2x3" in data:
        M = np.array(data["affine_2x3"], dtype=float)
    elif "matrix" in data:
        M = np.array(data["matrix"], dtype=float)
    elif "H_mat" in data:
        M = np.array(data["H_mat"], dtype=float)
    elif "affine_3x3" in data:
        M = np.array(data["affine_3x3"], dtype=float)[:2, :]
    else:
        raise ValueError("Cannot find affine matrix in alignment json")

    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]

    # ---- raw scale from affine (at alignment levels) ----
    sx = math.sqrt(a * a + b * b)
    sy = math.sqrt(c * c + d * d)

    # ---- pyramid level correction ----
    level_he   = images_info["HE_level"]
    level_he0 = images_info["HE0_level"]
    scale_level = 2 ** (level_he - level_he0)

    sx0 = sx * scale_level
    sy0 = sy * scale_level
    s0 = max(sx0, sy0)

    he_patch_len = int(math.ceil(he0_patch_len * s0 * margin))
    if he_patch_len % 2 == 1:
        he_patch_len += 1

    print(
        f"[INFO] Initial alignment from {src}\n"
        f"       affine scale: sx={sx:.3f}, sy={sy:.3f}\n"
        f"       level correction: 2^({level_he}-{level_he0}) = {scale_level:.3f}\n"
        f"       level0 scale: sx0={sx0:.3f}, sy0={sy0:.3f}\n"
        f"       HE0={he0_patch_len} -> HE≈{he_patch_len} (margin={margin})",
        flush=True
    )

    return he_patch_len

def extract_patch_and_mark_point(
    img,
    x_global,
    y_global,
    tile_id,
    nucleus_id,
    type=None,
    patch_length=60,
    out_dir="",
    save_patch=True,
    save_overlay=True,
):
    """
    Crop patch around (x_global, y_global), save patch and overlay.
    type must be "he0" or "he".
    """
    half = patch_length // 2
    H, W = img.shape[:2]

    x_global = int(np.clip(x_global, 0, W - 1))
    y_global = int(np.clip(y_global, 0, H - 1))

    y0 = max(0, y_global - half)
    x0 = max(0, x_global - half)

    crop = read_crop_patch(img, x0, y0, patch_length, patch_length)
    if crop.size == 0:
        raise ValueError(f"Empty crop at (x_global={x_global}, y_global={y_global})")

    if crop.ndim == 2:
        h, w = crop.shape
    else:
        h, w = crop.shape[:2]

    cx0 = int(np.clip(x_global - x0, 0, w - 1))
    cy0 = int(np.clip(y_global - y0, 0, h - 1))

    # make uint8 RGB for saving
    crop_vis = crop
    if crop_vis.ndim == 2:
        crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2RGB)
    if crop_vis.dtype != np.uint8:
        crop_vis = np.clip(crop_vis / 256.0, 0, 255).astype(np.uint8)

    if type not in ("he0", "he"):
        raise ValueError("type must be 'he0' or 'he'")

    os.makedirs(out_dir, exist_ok=True)

    if save_patch:
        patch_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png"
        cv2.imwrite(patch_path, cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

    if save_overlay:
        overlay = crop_vis.copy()
        cv2.circle(overlay, (cx0, cy0), 2, (255, 0, 0), -1)
        overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    info = {
        "centroid_global": [float(x_global), float(y_global)],
        "point_patch": [int(cx0), int(cy0)],
        "crop_box": [int(x0), int(y0), int(x0 + patch_length), int(y0 + patch_length)],
    }
    return float(x_global), float(y_global), crop_vis, info

def load_pilot_mask_params(out_folder: str, default_he0_thr=0.6, default_he_thr=0.6):
    path = os.path.join(out_folder, "../pilot_tiles", "pilot_output_parameters.json")
    if not os.path.exists(path):
        return float(default_he0_thr), float(default_he_thr)

    try:
        data = json.load(open(path, "r"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Use defaults.", flush=True)
        return float(default_he0_thr), float(default_he_thr)

    src = data.get("mask_preview", data) if isinstance(data, dict) else {}

    he0_thr = src.get("he0_intensity_threshold", src.get("he_intensity_threshold", default_he0_thr))
    he_thr = src.get("he_intensity_threshold", default_he_thr)

    try:
        he0_thr = float(he0_thr)
    except Exception:
        he0_thr = float(default_he0_thr)
    he0_thr = max(0.0, min(1.0, he0_thr))

    try:
        he_thr = float(he_thr)
    except Exception:
        he_thr = float(default_he_thr)
    he_thr = max(0.0, min(1.0, he_thr))

    return he0_thr, he_thr


def process_he0(image_file, upscale=2, intensity_threshold=0.6, dot_r=4):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=2,
            intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he0_patch.png", "_he0_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)

        mask_u8 = (mask_dark.astype(np.uint8) * 255)
        h, w = mask_u8.shape[:2]
        cx, cy = w // 2, h // 2
        overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), int(dot_r), (255, 0, 0), -1)
        overlay_save_path = image_file.replace("_he0_patch.png", "_he0_mask_overlay.png")
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return f"Processed: {image_file} with int_thr {intensity_threshold} mask: {mask_save_path} overlay: {overlay_save_path}"

    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

def process_he(image_file, upscale=2, intensity_threshold=0.6, dot_r=4):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile, upsample_scale=upscale, n_smooth=2, intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he_patch.png", "_he_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        # ---- save overlay with center red dot ----
        mask_u8 = (mask_dark.astype(np.uint8) * 255)  # uint8 0/255
        h, w = mask_u8.shape[:2]
        cx, cy = w // 2, h // 2
        overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), int(dot_r), (255, 0, 0), -1)
        overlay_save_path = image_file.replace("_he_patch.png", "_he_mask_overlay.png")
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return f"Processed: {image_file} with int_thr {intensity_threshold} mask: {mask_save_path}  overlay: {overlay_save_path}"

    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

# ==========================================================
# MAIN
# ==========================================================
def main(run_dir):
    start_time = time.time()

    print("[INFO] Loading metadata")

    scripts_dir = os.getcwd()
    with open(f"{scripts_dir}/parameters.json") as f:
        parameters = json.load(f)
    os.chdir(run_dir)
    with open("standout_nuclei.json") as f:
        nuclei = json.load(f)
    with open("he0_tile_info.json") as f:
        he0_tiles = json.load(f)
    with open("he_tile_info.json") as f:
        he_tiles = json.load(f)
    with open("../sampled_points.json") as f:
        sampled_info = json.load(f)
    with open("../images_info.json") as f:
        images_info = json.load(f)

    HE0_PATCH_LEN = 200
    HE_PATCH_LEN = load_initial_affine_and_compute_he_patch(
        run_dir,
        images_info,
        he0_patch_len=HE0_PATCH_LEN,
        margin=1.1
    )
    HE_PATH = images_info["HE_path"]
    HE0_PATH = images_info["HE0_path"]
    HE0_LEVEL = int(images_info["HE0_level"])
    HE_LEVEL = int(images_info["HE_level"])
    HE0_EXTRACT_LEVEL = int(sampled_info["he0_extract_level"])
    # step3 extract level -> actual read_image level
    HE0_READ_LEVEL = extract_level_to_read_level(HE0_EXTRACT_LEVEL)
    # keep HE at the same effective scale as HE0, then also shift by -1
    HE_EXTRACT_LEVEL = HE_LEVEL - (HE0_LEVEL - HE0_EXTRACT_LEVEL)
    HE_EXTRACT_LEVEL = max(1, int(HE_EXTRACT_LEVEL))
    HE_READ_LEVEL = extract_level_to_read_level(HE_EXTRACT_LEVEL)
    print(f"[INFO] HE0_LEVEL={HE0_LEVEL}, HE_LEVEL={HE_LEVEL}", flush=True)
    print(f"[INFO] HE0_EXTRACT_LEVEL={HE0_EXTRACT_LEVEL} -> HE0_READ_LEVEL={HE0_READ_LEVEL}", flush=True)
    print(f"[INFO] HE_EXTRACT_LEVEL={HE_EXTRACT_LEVEL} -> HE_READ_LEVEL={HE_READ_LEVEL}", flush=True)
    print("[INFO] Loading full-res images. This could take about 1 min ⚠️", flush=True)

    he0_scale_to_level0 = 2 ** HE0_READ_LEVEL
    he_scale_to_level0 = 2 ** HE_READ_LEVEL

    he_img, *_ = read_image(HE_PATH, keep_16bit=True, level=HE_READ_LEVEL, channel="he")
    global lut
    lut_path = images_info.get(
        "HE0_LUT",
        f"{scripts_dir}/glasbey_inverted.lut",
    )
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)
    he0_img, *_ = read_image(HE0_PATH, keep_16bit=True, level=HE0_READ_LEVEL, channel="he0")

    t_after_io = time.time()
    io_sec = t_after_io - start_time
    print(f"[INFO] Image loading time: {fmt_time(io_sec)}", flush=True)
    loop_t0 = time.time()
    print("[ETA_START] loop", flush=True)

    out_dir = "../nuclei_patches"
    os.makedirs(out_dir, exist_ok=True)
    output_coord_record = []
    total = len(nuclei)
    print(f"[INFO] Refining {total} nuclei centroids", flush=True)

    he0_scale_extract_to_read = 2 ** (HE0_EXTRACT_LEVEL - HE0_READ_LEVEL)
    he_scale_extract_to_read = 2 ** (HE_EXTRACT_LEVEL - HE_READ_LEVEL)

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        he0_info = he0_tiles[tile_id]

        xA_tile, yA_tile = n["original"]["he0"]

        xA_global = int(round((float(he0_info["x0"]) + float(xA_tile)) * he0_scale_extract_to_read))
        yA_global = int(round((float(he0_info["y0"]) + float(yA_tile)) * he0_scale_extract_to_read))

        output_xA_global, output_yA_global, _, _ = extract_patch_and_mark_point(
            he0_img,
            xA_global, yA_global,
            tile_id, nucleus_id,
            type="he0",
            patch_length=HE0_PATCH_LEN,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )

        output_coord_record.append({
            "tile_id": tile_id,
            "nucleus_id": nucleus_id,
            "mode": n.get("mode", "nuclei_pair"),
            "he0_centroid_global": [
                float(output_xA_global * he0_scale_to_level0),
                float(output_yA_global * he0_scale_to_level0),
            ],
            "he_centroid_global": None,
        })

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        he_info = he_tiles[tile_id]

        if "original" in n and "he" in n["original"] and n["original"]["he"] is not None:
            xB_tile, yB_tile = n["original"]["he"]
        else:
            meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}
            if meta.get("mode") == "rectified" and meta.get("rectified_wh") is not None:
                out_w, out_h = meta["rectified_wh"]
                xB_tile, yB_tile = (out_w / 2.0, out_h / 2.0)
            else:
                xB_tile, yB_tile = (float(he_info["w"]) / 2.0, float(he_info["h"]) / 2.0)

        xB_img, yB_img = he_point_tile_to_image_coords(he_info, xB_tile, yB_tile)

        xB_global = float(xB_img) * he_scale_extract_to_read
        yB_global = float(yB_img) * he_scale_extract_to_read

        output_xB_global, output_yB_global, _, _ = extract_patch_and_mark_point(
            he_img,
            xB_global, yB_global,
            tile_id, nucleus_id,
            type="he",
            patch_length=HE_PATCH_LEN,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )

        output_coord_record[i - 1]["he_centroid_global"] = [
            float(output_xB_global * he_scale_to_level0),
            float(output_yB_global * he_scale_to_level0),
        ]

    out_json = os.path.join(out_dir, "nuclei_centroids_global.json")
    with open(out_json, "w") as f:
        json.dump(output_coord_record, f, indent=2)
    print(f"[INFO] Saved centroids to {out_json}", flush=True)

    pilot_he0_thr, pilot_he_thr = load_pilot_mask_params(
        out_dir,
        default_he0_thr=0.6,
        default_he_thr=0.6,
    )
    out_dir = Path(out_dir)
    for f in glob(str(out_dir / "*_he0_patch.png")):
        print(process_he0(f, intensity_threshold=pilot_he0_thr))
    for f in glob(str(out_dir / "*_he_patch.png")):
        print(process_he(f, intensity_threshold=pilot_he_thr))
    print(f"[INFO] Saved nuclei masks", flush=True)

    print("[DONE]", flush=True)

# ==========================================================
# Entry
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 7_get_nuclei_patches.py <RUN_DIR>")
    run_dir = sys.argv[1]
    main(run_dir)