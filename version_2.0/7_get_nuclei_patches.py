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
    dapi_to_lut_rgb,
    segment_super_dark_nuclei_full,
    read_crop_patch,
    upsample_tile,
    fill_holes_binary,
    remove_small_components

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

def he_point_tile_to_global(he_info, x_tile, y_tile, S):
    """
    If rectified: (x_tile,y_tile) is in rectified coords -> project back using M_rect_to_he (global level=1 coords)
    Else: (x_tile,y_tile) is tile-local bbox coords -> x0/y0 add (global level=1 coords)
    Return: (x_global_level0, y_global_level0)
    """
    meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}

    if meta.get("mode", None) == "rectified" and meta.get("M_rect_to_he", None) is not None:
        Minv = np.asarray(meta["M_rect_to_he"], dtype=float)
        x1, y1 = apply_homography_xy(Minv, x_tile, y_tile)  # HE level=1 global coords
        return x1 * S, y1 * S

    # bbox / non-rectified fallback: tile-local -> level=1 global -> level0
    x1 = float(he_info["x0"]) + float(x_tile)
    y1 = float(he_info["y0"]) + float(y_tile)
    return x1 * S, y1 * S

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
    dapi_patch_len=100,
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
    level_dapi = images_info["DAPI_level"]
    scale_level = 2 ** (level_he - level_dapi)

    sx0 = sx * scale_level
    sy0 = sy * scale_level
    s0 = max(sx0, sy0)

    he_patch_len = int(math.ceil(dapi_patch_len * s0 * margin))
    if he_patch_len % 2 == 1:
        he_patch_len += 1

    print(
        f"[INFO] Initial alignment from {src}\n"
        f"       affine scale: sx={sx:.3f}, sy={sy:.3f}\n"
        f"       level correction: 2^({level_he}-{level_dapi}) = {scale_level:.3f}\n"
        f"       level0 scale: sx0={sx0:.3f}, sy0={sy0:.3f}\n"
        f"       DAPI={dapi_patch_len} -> HE≈{he_patch_len} (margin={margin})",
        flush=True
    )

    return he_patch_len

def get_dapi_mask(dapi_rgb, threshold=20):
    dapi_mask = (np.any(dapi_rgb > threshold, axis=-1)).astype(np.uint8) * 255
    dapi_mask_flipped = 255 - dapi_mask
    return dapi_mask_flipped

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
    save_raw_dapi=True,
):
    """
    NO refine. Only:
      - crop patch around (x_global, y_global)
      - for DAPI: also save a raw visualization (uint16 -> uint8)
      - save overlay with a dot at the center point
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

    # point in patch coords
    if crop.ndim == 2:
        h, w = crop.shape
    else:
        h, w = crop.shape[:2]
    cx0 = int(np.clip(x_global - x0, 0, w - 1))
    cy0 = int(np.clip(y_global - y0, 0, h - 1))

    # ---- prepare visualization crop (RGB) ----
    crop_raw16 = None
    crop_vis = crop

    if type == "dapi":
        # crop is uint16 grayscale usually
        crop_raw16 = ensure_gray_uint16(crop)
        crop_vis = dapi_to_lut_rgb(crop_raw16, lut, threshold=1000)  # RGB uint8
    elif type == "he":
        # crop likely RGB already (uint8/uint16 depends on your read_image)
        # make sure it's uint8 RGB for saving/overlay
        if crop_vis.ndim == 2:
            crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2RGB)
        if crop_vis.dtype != np.uint8:
            # simple compress if uint16 (you can replace with better mapping if needed)
            crop_vis = np.clip(crop_vis / 256.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError("type must be 'dapi' or 'he'")

    os.makedirs(out_dir, exist_ok=True)

    # ---- save patch (no dot) ----
    if save_patch:
        patch_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png"
        cv2.imwrite(patch_path, cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

        # raw DAPI visualization (optional)
        if type == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_norm_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw.png"
            cv2.imwrite(raw_norm_path, raw_vis8)

    # ---- save overlay (dot) ----
    if save_overlay:
        overlay = crop_vis.copy()
        cv2.circle(overlay, (cx0, cy0), 2, (255, 0, 0), -1)  # RGB blue dot
        overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if type == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_overlay = cv2.cvtColor(raw_vis8, cv2.COLOR_GRAY2BGR)
            cv2.circle(raw_overlay, (cx0, cy0), 2, (0, 0, 255), -1)  # red dot
            raw_overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw_overlay.png"
            cv2.imwrite(raw_overlay_path, raw_overlay)

    info = {
        "centroid_global": [float(x_global), float(y_global)],
        "point_patch": [int(cx0), int(cy0)],
        "crop_box": [int(x0), int(y0), int(x0 + patch_length), int(y0 + patch_length)],
    }
    return float(x_global), float(y_global), crop_vis, info

def load_pilot_mask_params(out_folder: str, default_dapi_offset=0, default_he_thr=0.6):
    """
    Try read:
      <out_folder>/pilot_tiles/pilot_output_parameters.json

    Supports JSON layout:
      1) {"mask_preview": {"dapi_thr_offset": 25, "he_intensity_threshold": 0.5}}
      2) {"dapi_thr_offset": 25, "he_intensity_threshold": 0.5}

    Returns:
      (dapi_thr_offset:int, he_intensity_threshold:float)
    """
    path = os.path.join(out_folder, "../pilot_tiles", "pilot_output_parameters.json")
    if not os.path.exists(path):
        return int(default_dapi_offset), float(default_he_thr)
    try:
        data = json.load(open(path, "r"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Use defaults.", flush=True)
        return int(default_dapi_offset), float(default_he_thr)

    # prefer nested structure
    src = data.get("mask_preview", data) if isinstance(data, dict) else {}

    dapi_off = src.get("dapi_thr_offset", default_dapi_offset)
    he_thr = src.get("he_intensity_threshold", default_he_thr)

    # sanitize
    try:
        dapi_off = int(float(dapi_off))
    except Exception:
        dapi_off = int(default_dapi_offset)
    dapi_off = max(-100, min(100, dapi_off))  # keep consistent with your UI

    try:
        he_thr = float(he_thr)
    except Exception:
        he_thr = float(default_he_thr)
    he_thr = max(0.0, min(1.0, he_thr))

    return dapi_off, he_thr

def process_dapi(
    dapi_file,
    THR_OFFSET=0,
    min_area_factor=10e-5,
    CONNECTIVITY=8,
    invert=True,
    upscale=2,
    dot_r=4
):
    """
    Input:  *_dapi_u8.png  (uint8 grayscale)
    Output: *_dapi_mask.png (binary mask, uint8 0/255)
    """
    try:
        if not os.path.exists(dapi_file):
            return f"Skipped (DAPI not found): {dapi_file}"

        base_name = os.path.basename(dapi_file)
        if base_name.endswith("_dapi_u8.png"):
            base_name = base_name.replace("_dapi_u8.png", "")
        elif base_name.endswith("_dapi_raw.png"):
            base_name = base_name.replace("_dapi_raw.png", "")
        else:
            return f"Skipped (not a valid dapi file): {dapi_file}"
        folder_name = os.path.dirname(dapi_file)

        # -----------------------------
        # LOAD u8 GRAYSCALE
        # -----------------------------
        img = cv2.imread(dapi_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed: {dapi_file}, Error: cv2.imread returned None"
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = upsample_tile(img, upscale)

        # -----------------------------
        # 1) Otsu + optional offset
        # -----------------------------
        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(int(otsu_thr) + int(THR_OFFSET), 0, 255))
        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        fg_ratio0 = float((mask > 0).mean())

        # -----------------------------
        # 2) fill holes
        # -----------------------------
        mask = fill_holes_binary(mask)
        fg_ratio1 = float((mask > 0).mean())

        # -----------------------------
        # 3) remove tiny blobs
        # -----------------------------
        MIN_AREA = min_area_factor * img.shape[0] ** 2
        MIN_AREA = MIN_AREA * upscale ** 2
        mask, cc_info = remove_small_components(
            mask, min_area=int(MIN_AREA), connectivity=int(CONNECTIVITY)
        )
        fg_ratio2 = float((mask > 0).mean())
        # optional invert (match your previous 255 - mask)
        if invert:
            mask = 255 - mask

        mask_save_path = os.path.join(folder_name, f"{base_name}_dapi_mask.png")
        cv2.imwrite(mask_save_path, mask)

        kept = None
        total = None
        if isinstance(cc_info, dict):
            kept = cc_info.get("kept", None)
            total = cc_info.get("total", None)
        extra = f" fg {fg_ratio0:.3f}->{fg_ratio1:.3f}->{fg_ratio2:.3f}"
        if kept is not None and total is not None:
            extra += f" cc kept {kept}/{total}"

        # ---- save overlay with center red dot ----
        mask_u8 = mask  # uint8 0/255
        h, w = mask_u8.shape[:2]
        cx, cy = w // 2, h // 2
        overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), int(dot_r), (255, 0, 0), -1)  # RGB red
        overlay_save_path = mask_save_path.replace("_dapi_mask.png", "_dapi_mask_overlay.png")
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return f"Processed: {os.path.basename(dapi_file)} otsu={int(otsu_thr)} used={thr}{extra} saved: {mask_save_path}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"

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
    with open("dapi_tile_info.json") as f:
        dapi_tiles = json.load(f)
    with open("he_tile_info.json") as f:
        he_tiles = json.load(f)
    with open("../images_info.json") as f:
        images_info = json.load(f)

    DAPI_PATCH_LEN = 200
    HE_PATCH_LEN = load_initial_affine_and_compute_he_patch(
        run_dir,
        images_info,
        dapi_patch_len=DAPI_PATCH_LEN,
        margin=1.1
    )
    HE_PATH = images_info["HE_path"]
    DAPI_PATH = images_info["DAPI_path"]
    print("[INFO] Loading full-res images. This could take about 1 min ⚠️", flush=True)

    he_img, *_ = read_image(HE_PATH, keep_16bit=True, level=0, channel="he")

    global lut
    lut_path = images_info.get(
        "DAPI_LUT",
        f"{scripts_dir}/glasbey_inverted.lut",
    )
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)
    dapi_img, *_ = read_image(DAPI_PATH, keep_16bit=True, level=0, channel="dapi")

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

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        dapi_info = dapi_tiles[tile_id]

        DAPI_TILE_LEVEL_OVERRIDE = parameters["step3"]["dapi_level_override"]
        if DAPI_TILE_LEVEL_OVERRIDE == "None":
            LEVEL_DIFF = 1
        elif isinstance(DAPI_TILE_LEVEL_OVERRIDE, int):
            LEVEL_DIFF = DAPI_TILE_LEVEL_OVERRIDE
        else:
            raise ValueError("DAPI_TILE_LEVEL_OVERRIDE in parameter.json['step3'] must be 'None' or int")

        S = 2 ** LEVEL_DIFF
        # ---- choose tile coords ----
        xA_tile, yA_tile = n["original"]["dapi"]
        xA_global = int(round(dapi_info["x0"] * S + xA_tile * S))
        yA_global = int(round(dapi_info["y0"] * S + yA_tile * S))

        output_xA_global, output_yA_global, _, _ = extract_patch_and_mark_point(
            dapi_img,
            xA_global, yA_global,
            tile_id, nucleus_id,
            type="dapi",
            patch_length=DAPI_PATCH_LEN,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )

        output_coord_record.append({
            "tile_id": tile_id,
            "nucleus_id": nucleus_id,
            "mode": n.get("mode", "nuclei_pair"),
            "dapi_centroid_global": [float(output_xA_global), float(output_yA_global)],
            "he_centroid_global": None,
        })
        print(f"[PROGRESS] DAPI {i}/{total}", flush=True)

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        he_info = he_tiles[tile_id]

        DAPI_LEVEL = images_info["DAPI_level"]
        HE_LEVEL = images_info["HE_level"]
        HE_TILE_LEVEL_OVERRIDE = parameters["step3"]["he_level_override"]
        if HE_TILE_LEVEL_OVERRIDE == "None":
            if HE_LEVEL <= DAPI_LEVEL:
                LEVEL_DIFF = 1
            else:
                LEVEL_DIFF = 1 + HE_LEVEL - DAPI_LEVEL
        elif isinstance(HE_TILE_LEVEL_OVERRIDE, int):
            LEVEL_DIFF = HE_TILE_LEVEL_OVERRIDE
        else:
            raise ValueError("DAPI_TILE_LEVEL_OVERRIDE must be 'None' or int")
        S = 2 ** LEVEL_DIFF

        if "original" in n and "he" in n["original"] and n["original"]["he"] is not None:
            xB_tile, yB_tile = n["original"]["he"]
        else:
            meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}
            if meta.get("mode") == "rectified" and meta.get("rectified_wh") is not None:
                out_w, out_h = meta["rectified_wh"]  # NOTE: [W, H]
                xB_tile, yB_tile = (out_w / 2.0, out_h / 2.0)   # center in rectified coords
            else:
                xB_tile, yB_tile = (float(he_info["w"]) / 2.0, float(he_info["h"]) / 2.0)

        xB_global, yB_global = he_point_tile_to_global(he_info, xB_tile, yB_tile, S)

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

        output_coord_record[i - 1]["he_centroid_global"] = [float(output_xB_global), float(output_yB_global)]
        print(f"[PROGRESS] HE {i}/{total}", flush=True)

    out_json = os.path.join(out_dir, "nuclei_centroids_global.json")
    with open(out_json, "w") as f:
        json.dump(output_coord_record, f, indent=2)
    print(f"[INFO] Saved centroids to {out_json}", flush=True)


    pilot_dapi_offset, pilot_he_thr = load_pilot_mask_params(
        out_dir,
        default_dapi_offset=0,
        default_he_thr=0.6,
    )
    out_dir = Path(out_dir)
    for f in glob(str(out_dir / "*_dapi_raw.png")):
        print(process_dapi(f,THR_OFFSET=pilot_dapi_offset))
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