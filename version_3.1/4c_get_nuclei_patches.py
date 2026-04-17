import json
import cv2
import os
import sys
import time
import numpy as np
from glob import glob
from PIL import Image
import math
from pathlib import Path
from my_utils import (
    read_image,
    dapi_to_lut_rgb,
    segment_super_dark_nuclei_full,
    read_crop_patch,
    upsample_tile,
    fill_holes_binary,
    remove_small_components,
)


# ==========================================================
# Progress / timing helpers
# ==========================================================
def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

def print_progress(stage_name, i, total):
    print(f"[PROGRESS] {stage_name} {i}/{total}", flush=True)

def print_eta_start(stage_name):
    print(f"[ETA_START] {stage_name}", flush=True)


# ==========================================================
# Geometry and coordinate conversion helpers
# ==========================================================
def apply_homography_xy(M3x3, x, y):
    M3x3 = np.asarray(M3x3, dtype=float)
    p = np.array([float(x), float(y), 1.0], dtype=float)
    q = M3x3 @ p
    w = q[2] if abs(q[2]) > 1e-12 else 1e-12
    return float(q[0] / w), float(q[1] / w)

def tile_level_to_patch_level(tile_level: int) -> int:
    """
    Convert step3 extract level convention to read_image level convention:
      extract 1 -> read 0
      extract 2 -> read 1
      extract 3 -> read 2
      ...
    """
    return max(0, int(tile_level) - 1)

def he_point_tile_to_image_coords(he_info, x_tile, y_tile):
    """
    Return point coordinates in the same coordinate system as the loaded HE image
    (i.e. HE_patch_level coordinates).

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

def load_initial_affine_and_compute_he_patch(
    run_dir,
    images_info,
    dapi_patch_len=100,
    margin=1.1
):
    run_dir = Path(run_dir)

    path_manual = run_dir / "../manual_initial_alignment.json"
    if path_manual.exists():
        data = json.load(open(path_manual, "r"))
        src = path_manual.name
    else:
        raise FileNotFoundError("No manual_initial_alignment.json found.")

    if "affine_2x3" in data:
        M = np.array(data["affine_2x3"], dtype=float)
    elif "affine_3x3" in data:
        M = np.array(data["affine_3x3"], dtype=float)[:2, :]
    elif "H_mat" in data:
        M = np.array(data["H_mat"], dtype=float)
    elif "matrix" in data:
        M = np.array(data["matrix"], dtype=float)
    else:
        raise ValueError("Cannot find affine matrix in alignment json")

    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]

    sx = math.sqrt(a * a + b * b)
    sy = math.sqrt(c * c + d * d)

    level_he = images_info["HE_level"]
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


# ==========================================================
# Patch extraction and visualization helpers
# ==========================================================
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

def get_dot_radius(img, base_size=200, base_r=4):
    h, w = img.shape[:2]
    scale = min(h, w) / base_size
    r = int(round(base_r * scale))
    return max(1, r)

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
    save_raw_dapi=True
):
    """
    No refine. Only:
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

    if crop.ndim == 2:
        h, w = crop.shape
    else:
        h, w = crop.shape[:2]
    cx0 = int(np.clip(x_global - x0, 0, w - 1))
    cy0 = int(np.clip(y_global - y0, 0, h - 1))

    crop_raw16 = None
    crop_vis = crop

    if type == "dapi":
        crop_raw16 = ensure_gray_uint16(crop)
        crop_vis = dapi_to_lut_rgb(crop_raw16, lut, threshold=1000)
    elif type == "he":
        if crop_vis.ndim == 2:
            crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2RGB)
        if crop_vis.dtype != np.uint8:
            crop_vis = np.clip(crop_vis / 256.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError("type must be 'dapi' or 'he'")

    os.makedirs(out_dir, exist_ok=True)

    if save_patch:
        patch_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png"
        cv2.imwrite(patch_path, cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

        if type == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_norm_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw.png"
            cv2.imwrite(raw_norm_path, raw_vis8)

    if save_overlay:
        overlay = crop_vis.copy()
        cv2.circle(overlay, (cx0, cy0), int(get_dot_radius(overlay)), (255, 0, 0), -1)
        overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if type == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_overlay = cv2.cvtColor(raw_vis8, cv2.COLOR_GRAY2BGR)
            cv2.circle(raw_overlay, (cx0, cy0), int(get_dot_radius(overlay)), (0, 0, 255), -1)
            raw_overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw_overlay.png"
            cv2.imwrite(raw_overlay_path, raw_overlay)

    info = {
        "centroid_global": [float(x_global), float(y_global)],
        "centroid_local": [int(cx0), int(cy0)],
        "point_patch": [int(cx0), int(cy0)],
        "crop_box": [int(x0), int(y0), int(x0 + patch_length), int(y0 + patch_length)],
    }
    return float(x_global), float(y_global), crop_vis, info


# ==========================================================
# Patch-level mask generation helpers
# ==========================================================
def load_stage4c_params_from_parameters_json(
    out_folder: str,
    default_dapi_patch_len=200,
    default_dapi_offset=0,
    default_dapi_min_area_factor=1e-4,
    default_dapi_upscale=1,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=1,
):
    """
    Read ../../parameters.json and return stage4c params.
    Falls back to the provided defaults if the file or keys are missing.
    """
    path = os.path.join(out_folder, "../../parameters.json")
    print(f"[INFO] loading stage4c parameters from: {path}", flush=True)

    params = {
        "dapi_patch_len": int(default_dapi_patch_len),
        "dapi_thr_offset": int(default_dapi_offset),
        "dapi_mask_min_area_factor": float(default_dapi_min_area_factor),
        "dapi_mask_upscale_factor": int(default_dapi_upscale),
        "he_mask_n_smooth": int(default_he_n_smooth),
        "he_mask_intensity_threshold": float(default_he_thr),
        "he_mask_upscale_factor": int(default_he_upscale),
    }

    if not os.path.exists(path):
        print(f"[INFO] parameters from: {path} do not exist", flush=True)
        return params
    try:
        with open(path, "r") as f:
            data = json.load(f)
            print(f"[INFO] loaded parameters from: {path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Use defaults.", flush=True)
        return params

    stage4c = data.get("stage4c", {}) if isinstance(data, dict) else {}

    try:
        params["dapi_patch_len"] = int(
            stage4c.get("dapi_patch_len", params["dapi_patch_len"])
        )
    except Exception:
        pass

    try:
        params["dapi_thr_offset"] = int(
            float(stage4c.get("dapi_thr_offset", params["dapi_thr_offset"]))
        )
    except Exception:
        pass

    try:
        params["dapi_mask_min_area_factor"] = float(
            stage4c.get("dapi_mask_min_area_factor", params["dapi_mask_min_area_factor"])
        )
    except Exception:
        pass

    try:
        params["dapi_mask_upscale_factor"] = int(
            stage4c.get("dapi_mask_upscale_factor", params["dapi_mask_upscale_factor"])
        )
    except Exception:
        pass

    try:
        params["he_mask_n_smooth"] = int(
            stage4c.get("he_mask_n_smooth", params["he_mask_n_smooth"])
        )
    except Exception:
        pass

    try:
        params["he_mask_intensity_threshold"] = float(
            stage4c.get("he_mask_intensity_threshold", params["he_mask_intensity_threshold"])
        )
    except Exception:
        pass

    try:
        params["he_mask_upscale_factor"] = int(
            stage4c.get("he_mask_upscale_factor", params["he_mask_upscale_factor"])
        )
    except Exception:
        pass

    return params

def load_stage4c_effective_params(
    out_folder: str,
    default_dapi_patch_len=200,
    default_dapi_offset=0,
    default_dapi_min_area_factor=1e-4,
    default_dapi_upscale=1,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=1,
):
    """
    Resolve Stage 4c parameters with this priority:
      1) parameters.json -> stage4c
      2) pilot_output_parameters.json overrides:
           - dapi_thr_offset
           - he_mask_intensity_threshold
      3) hard-coded defaults
    """
    params = load_stage4c_params_from_parameters_json(
        out_folder,
        default_dapi_patch_len=default_dapi_patch_len,
        default_dapi_offset=default_dapi_offset,
        default_dapi_min_area_factor=default_dapi_min_area_factor,
        default_dapi_upscale=default_dapi_upscale,
        default_he_n_smooth=default_he_n_smooth,
        default_he_thr=default_he_thr,
        default_he_upscale=default_he_upscale,
    )

    pilot_path = os.path.join(out_folder, "../pilot_tiles", "pilot_output_parameters.json")
    if not os.path.exists(pilot_path):
        return params

    try:
        with open(pilot_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {pilot_path}: {e}. Use parameters.json/defaults.", flush=True)
        return params

    src = data.get("mask_preview", data) if isinstance(data, dict) else {}

    dapi_off = src.get("dapi_thr_offset", params["dapi_thr_offset"])
    he_thr = src.get("he_intensity_threshold", params["he_mask_intensity_threshold"])

    try:
        dapi_off = int(float(dapi_off))
    except Exception:
        dapi_off = params["dapi_thr_offset"]
    dapi_off = max(-100, min(100, dapi_off))

    try:
        he_thr = float(he_thr)
    except Exception:
        he_thr = params["he_mask_intensity_threshold"]
    he_thr = max(0.0, min(1.0, he_thr))

    params["dapi_thr_offset"] = dapi_off
    params["he_mask_intensity_threshold"] = he_thr
    return params

def process_dapi(
    dapi_file,
    THR_OFFSET=0,
    min_area_factor=10e-5,
    CONNECTIVITY=8,
    invert=True,
    upscale=1
):
    """
    Input:  *_dapi_u8.png or *_dapi_raw.png
    Output: *_dapi_mask.png and *_dapi_mask_overlay.png
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

        img = cv2.imread(dapi_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed: {dapi_file}, Error: cv2.imread returned None"
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = upsample_tile(img, upscale)

        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(int(otsu_thr) + int(THR_OFFSET), 0, 255))
        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        fg_ratio0 = float((mask > 0).mean())

        mask = fill_holes_binary(mask)
        fg_ratio1 = float((mask > 0).mean())

        min_area = min_area_factor * img.shape[0] ** 2
        min_area = min_area * upscale ** 2
        mask, cc_info = remove_small_components(
            mask, min_area=int(min_area), connectivity=int(CONNECTIVITY)
        )
        fg_ratio2 = float((mask > 0).mean())

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

        mask_u8 = mask
        h, w = mask_u8.shape[:2]
        cx, cy = w // 2, h // 2
        overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), int(get_dot_radius(overlay)), (255, 0, 0), -1)
        overlay_save_path = mask_save_path.replace("_dapi_mask.png", "_dapi_mask_overlay.png")
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return f"Processed: {os.path.basename(dapi_file)} otsu={int(otsu_thr)}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"

def process_he(image_file, upscale=1, n_smooth=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        _, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=int(n_smooth),
            intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he_patch.png", "_he_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)

        mask_u8 = mask_dark.astype(np.uint8) * 255
        h, w = mask_u8.shape[:2]
        cx, cy = w // 2, h // 2
        overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), int(get_dot_radius(overlay)), (255, 0, 0), -1)
        overlay_save_path = image_file.replace("_he_patch.png", "_he_mask_overlay.png")
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return f"Processed: {image_file} with int_thr {intensity_threshold}"

    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

# ==========================================================
# Metadata loading and image context setup
# ==========================================================
def load_run_metadata(run_dir):
    print("[INFO] Loading metadata", flush=True)

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
    with open("../sampled_points.json") as f:
        sampled_info = json.load(f)
    with open("../images_info.json") as f:
        images_info = json.load(f)

    return {
        "scripts_dir": scripts_dir,
        "parameters": parameters,
        "nuclei": nuclei,
        "dapi_tiles": dapi_tiles,
        "he_tiles": he_tiles,
        "sampled_info": sampled_info,
        "images_info": images_info,
    }

def prepare_image_context(run_dir, meta, dapi_patch_len=200):
    images_info = meta["images_info"]
    sampled_info = meta["sampled_info"]
    scripts_dir = meta["scripts_dir"]

    he_patch_len = load_initial_affine_and_compute_he_patch(
        run_dir,
        images_info,
        dapi_patch_len=dapi_patch_len,
        margin=1.1
    )

    he_path = images_info["HE_path"]
    dapi_path = images_info["DAPI_path"]
    dapi_level = int(images_info["DAPI_level"])
    he_level = int(images_info["HE_level"])
    dapi_tile_level = int(sampled_info["dapi_tile_level"])

    dapi_patch_level = tile_level_to_patch_level(dapi_tile_level)
    he_tile_level = he_level - (dapi_level - dapi_tile_level)
    he_tile_level = max(1, int(he_tile_level))
    he_patch_level = tile_level_to_patch_level(he_tile_level)

    print(f"[INFO] DAPI_LEVEL={dapi_level}, HE_LEVEL={he_level}", flush=True)
    print(f"[INFO] DAPI_tile_level={dapi_tile_level} -> DAPI_patch_level={dapi_patch_level}", flush=True)
    print(f"[INFO] HE_tile_level={he_tile_level} -> HE_patch_level={he_patch_level}", flush=True)
    print("[INFO] Loading full-res images. This could take about 1 min ⚠️", flush=True)

    dapi_scale_to_level0 = 2 ** dapi_patch_level
    he_scale_to_level0 = 2 ** he_patch_level

    he_img, *_ = read_image(he_path, keep_16bit=True, level=he_patch_level, channel="he")

    global lut
    lut_path = images_info.get("DAPI_LUT", f"{scripts_dir}/glasbey_inverted.lut")
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

    dapi_img, *_ = read_image(dapi_path, keep_16bit=True, level=dapi_patch_level, channel="dapi")

    dapi_scale_tile_to_patch = 2 ** (dapi_tile_level - dapi_patch_level)
    he_scale_tile_to_patch = 2 ** (he_tile_level - he_patch_level)

    return {
        "he_patch_len": he_patch_len,
        "dapi_patch_len": dapi_patch_len,
        "dapi_img": dapi_img,
        "he_img": he_img,
        "dapi_scale_to_level0": dapi_scale_to_level0,
        "he_scale_to_level0": he_scale_to_level0,
        "dapi_scale_tile_to_patch": dapi_scale_tile_to_patch,
        "he_scale_tile_to_patch": he_scale_tile_to_patch,
        "dapi_tile_level": dapi_tile_level,
        "dapi_patch_level": dapi_patch_level,
        "he_tile_level": he_tile_level,
        "he_patch_level": he_patch_level,
    }

def save_global_centroids(out_dir, output_coord_record, image_ctx):
    out_json = os.path.join(out_dir, "nuclei_centroids_global.json")

    out = {
        "meta": {
            "dapi_tile_level": int(image_ctx.get("dapi_tile_level", -1)),
            "dapi_patch_level": int(image_ctx.get("dapi_patch_level", -1)),
            "he_tile_level": int(image_ctx.get("he_tile_level", -1)),
            "he_patch_level": int(image_ctx.get("he_patch_level", -1)),
        },
        "data": output_coord_record
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved centroids to {out_json}", flush=True)

# ==========================================================
# Main pipeline stages
# ==========================================================
def extract_all_dapi_patches(nuclei, dapi_tiles, dapi_img, out_dir, dapi_patch_len, dapi_scale_tile_to_patch, dapi_scale_to_level0):
    output_coord_record = []
    total = len(nuclei)

    print_eta_start("DAPI_PATCH")
    dapi_patch_t0 = time.time()
    print(f"[INFO] Extracting DAPI patches for {total} nuclei", flush=True)

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        dapi_info = dapi_tiles[tile_id]

        xA_tile, yA_tile = n["original"]["dapi"]

        xA_global = int(round((float(dapi_info["x0"]) + float(xA_tile)) * dapi_scale_tile_to_patch))
        yA_global = int(round((float(dapi_info["y0"]) + float(yA_tile)) * dapi_scale_tile_to_patch))

        output_xA_global, output_yA_global, _, dapi_patch_info = extract_patch_and_mark_point(
            dapi_img,
            xA_global,
            yA_global,
            tile_id,
            nucleus_id,
            type="dapi",
            patch_length=dapi_patch_len,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )

        output_coord_record.append({
            "tile_id": tile_id,
            "nucleus_id": nucleus_id,
            "mode": n.get("mode", "nuclei_pair"),
            "dapi_centroid_global": [
                float(output_xA_global * dapi_scale_to_level0),
                float(output_yA_global * dapi_scale_to_level0),
            ],
            "dapi_centroid_local": dapi_patch_info["centroid_local"],
            "he_centroid_global": None,
            "he_centroid_local": None,
        })

        elapsed = time.time() - dapi_patch_t0
        avg = elapsed / i if i > 0 else 0.0
        eta = avg * (total - i)
        print_progress("DAPI_PATCH", i, total)
        print(
            f"[INFO] DAPI patch progress: {i}/{total} | "
            f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}",
            flush=True,
        )

    return output_coord_record

def extract_all_he_patches(nuclei, he_tiles, he_img, out_dir, he_patch_len, he_scale_tile_to_patch, he_scale_to_level0,
                           output_coord_record):
    total = len(nuclei)

    print_eta_start("HE_PATCH")
    he_patch_t0 = time.time()
    print(f"[INFO] Extracting HE patches for {total} nuclei", flush=True)

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
        xB_global = float(xB_img) * he_scale_tile_to_patch
        yB_global = float(yB_img) * he_scale_tile_to_patch

        output_xB_global, output_yB_global, _, he_patch_info = extract_patch_and_mark_point(
            he_img,
            xB_global,
            yB_global,
            tile_id,
            nucleus_id,
            type="he",
            patch_length=he_patch_len,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )

        output_coord_record[i - 1]["he_centroid_global"] = [
            float(output_xB_global * he_scale_to_level0),
            float(output_yB_global * he_scale_to_level0),
        ]
        output_coord_record[i - 1]["he_centroid_local"] = he_patch_info["centroid_local"]

        elapsed = time.time() - he_patch_t0
        avg = elapsed / i if i > 0 else 0.0
        eta = avg * (total - i)
        print_progress("HE_PATCH", i, total)
        print(
            f"[INFO] HE patch progress: {i}/{total} | "
            f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}",
            flush=True,
        )

    return output_coord_record

def generate_all_dapi_masks(out_dir, dapi_thr_offset, dapi_mask_min_area_factor, dapi_mask_upscale_factor):
    dapi_files = glob(str(Path(out_dir) / "*_dapi_raw.png"))
    total = len(dapi_files)

    print_eta_start("DAPI_MASK")
    dapi_mask_t0 = time.time()
    print(f"[INFO] Generating DAPI masks for {total} patches", flush=True)

    for i, f in enumerate(dapi_files, 1):
        print(
            process_dapi(
                f,
                THR_OFFSET=dapi_thr_offset,
                min_area_factor=dapi_mask_min_area_factor,
                upscale=dapi_mask_upscale_factor,
            ),
            flush=True,
        )

        elapsed = time.time() - dapi_mask_t0
        avg = elapsed / i if i > 0 else 0.0
        eta = avg * (total - i)
        print_progress("DAPI_MASK", i, total)
        print(
            f"[INFO] DAPI mask progress: {i}/{total} | "
            f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}",
            flush=True,
        )

def generate_all_he_masks(out_dir, he_mask_intensity_threshold, he_mask_n_smooth, he_mask_upscale_factor):
    he_files = glob(str(Path(out_dir) / "*_he_patch.png"))
    total = len(he_files)

    print_eta_start("HE_MASK")
    he_mask_t0 = time.time()
    print(f"[INFO] Generating HE masks for {total} patches", flush=True)

    for i, f in enumerate(he_files, 1):
        print(
            process_he(
                f,
                upscale=he_mask_upscale_factor,
                n_smooth=he_mask_n_smooth,
                intensity_threshold=he_mask_intensity_threshold,
            ),
            flush=True,
        )

        elapsed = time.time() - he_mask_t0
        avg = elapsed / i if i > 0 else 0.0
        eta = avg * (total - i)
        print_progress("HE_MASK", i, total)
        print(
            f"[INFO] HE mask progress: {i}/{total} | "
            f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta)}",
            flush=True,
        )

def main(run_dir):
    start_time = time.time()
    run_dir = Path(run_dir).resolve()

    meta = load_run_metadata(run_dir)

    out_dir = run_dir / "../nuclei_patches"
    os.makedirs(out_dir, exist_ok=True)

    stage4c_params = load_stage4c_effective_params(
        out_dir,
        default_dapi_patch_len=200,
        default_dapi_offset=0,
        default_dapi_min_area_factor=1e-4,
        default_dapi_upscale=1,
        default_he_n_smooth=2,
        default_he_thr=0.6,
        default_he_upscale=1,
    )

    print(
        "[INFO] Effective stage4c params: "
        f"dapi_patch_len={stage4c_params['dapi_patch_len']}, "
        f"dapi_thr_offset={stage4c_params['dapi_thr_offset']}, "
        f"dapi_mask_min_area_factor={stage4c_params['dapi_mask_min_area_factor']}, "
        f"dapi_mask_upscale_factor={stage4c_params['dapi_mask_upscale_factor']}, "
        f"he_mask_n_smooth={stage4c_params['he_mask_n_smooth']}, "
        f"he_mask_intensity_threshold={stage4c_params['he_mask_intensity_threshold']}, "
        f"he_mask_upscale_factor={stage4c_params['he_mask_upscale_factor']}",
        flush=True,
    )

    image_ctx = prepare_image_context(
        run_dir,
        meta,
        dapi_patch_len=int(stage4c_params["dapi_patch_len"]),
    )

    io_sec = time.time() - start_time
    print(f"[INFO] Image loading time: {fmt_time(io_sec)}", flush=True)

    nuclei = meta["nuclei"]
    dapi_tiles = meta["dapi_tiles"]
    he_tiles = meta["he_tiles"]

    output_coord_record = extract_all_dapi_patches(
        nuclei=nuclei,
        dapi_tiles=dapi_tiles,
        dapi_img=image_ctx["dapi_img"],
        out_dir=out_dir,
        dapi_patch_len=image_ctx["dapi_patch_len"],
        dapi_scale_tile_to_patch=image_ctx["dapi_scale_tile_to_patch"],
        dapi_scale_to_level0=image_ctx["dapi_scale_to_level0"],
    )

    output_coord_record = extract_all_he_patches(
        nuclei=nuclei,
        he_tiles=he_tiles,
        he_img=image_ctx["he_img"],
        out_dir=out_dir,
        he_patch_len=image_ctx["he_patch_len"],
        he_scale_tile_to_patch=image_ctx["he_scale_tile_to_patch"],
        he_scale_to_level0=image_ctx["he_scale_to_level0"],
        output_coord_record=output_coord_record
    )

    save_global_centroids(out_dir, output_coord_record, image_ctx)

    print(stage4c_params["dapi_mask_upscale_factor"])
    generate_all_dapi_masks(
        out_dir=out_dir,
        dapi_thr_offset=stage4c_params["dapi_thr_offset"],
        dapi_mask_min_area_factor=stage4c_params["dapi_mask_min_area_factor"],
        dapi_mask_upscale_factor=stage4c_params["dapi_mask_upscale_factor"],
    )

    generate_all_he_masks(
        out_dir=out_dir,
        he_mask_intensity_threshold=stage4c_params["he_mask_intensity_threshold"],
        he_mask_n_smooth=stage4c_params["he_mask_n_smooth"],
        he_mask_upscale_factor=stage4c_params["he_mask_upscale_factor"],
    )
    print("[INFO] Saved nuclei masks", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 4c_get_nuclei_patches.py <RUN_DIR>")
    run_dir = sys.argv[1]
    main(run_dir)