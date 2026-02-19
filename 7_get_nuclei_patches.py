import json
import cv2
import os
import sys
import time
import numpy as np
from my_utils import (
    read_image,
    dapi_to_lut_rgb,
    segment_super_dark_nuclei_full,
    open_ome_level_lazy,
    read_crop_patch
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
        print("rectified")
        Minv = np.asarray(meta["M_rect_to_he"], dtype=float)
        print(Minv)
        x1, y1 = apply_homography_xy(Minv, x_tile, y_tile)  # HE level=1 global coords
        return x1 * S, y1 * S

    # bbox / non-rectified fallback: tile-local -> level=1 global -> level0
    x1 = float(he_info["x0"]) + float(x_tile)
    y1 = float(he_info["y0"]) + float(y_tile)
    return x1 * S, y1 * S

def get_tile_hw(tile_info, default_hw=1024):
    meta = tile_info.get("meta", {}) if isinstance(tile_info, dict) else {}
    if meta.get("mode") == "rectified" and meta.get("rectified_wh") is not None:
        out_w, out_h = meta["rectified_wh"]
        return int(out_h), int(out_w)  # return H, W

    # 否则走原逻辑（bbox / 普通 tile）
    for kw in [("w", "h"), ("W", "H"), ("width", "height"), ("tile_w", "tile_h")]:
        if kw[0] in tile_info and kw[1] in tile_info:
            return int(tile_info[kw[1]]), int(tile_info[kw[0]])

    for k in ["tile_size", "size", "tile_len", "tile_side"]:
        if k in tile_info:
            s = int(tile_info[k])
            return s, s
    return int(default_hw), int(default_hw)


def get_tile_center_xy(tile_info, default_hw=1024):
    H, W = get_tile_hw(tile_info, default_hw=default_hw)
    return (W / 2.0, H / 2.0)

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

def get_nucleus_centroid_from_mask_at_xy(
    mask,
    x,
    y,
    return_overlay=False,
    highlight_color=(0, 0, 255),
    alpha=0.5
):
    H, W = mask.shape
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    binary = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    label = labels[y, x]   # IMPORTANT: (row=y, col=x)
    if label == 0:
        return None
    cx, cy = centroids[label]
    x0 = stats[label, cv2.CC_STAT_LEFT]
    y0 = stats[label, cv2.CC_STAT_TOP]
    w  = stats[label, cv2.CC_STAT_WIDTH]
    h  = stats[label, cv2.CC_STAT_HEIGHT]
    area = stats[label, cv2.CC_STAT_AREA]
    result = {
        "centroid": (float(cx), float(cy)),
        "area": int(area),
        "bbox": (x0, y0, x0 + w, y0 + h),
        "label": int(label),
    }
    if return_overlay:
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blob_mask = (labels == label)
        overlay[blob_mask] = cv2.addWeighted(
            overlay[blob_mask], 1 - alpha,
            np.full_like(overlay[blob_mask], highlight_color),
            alpha, 0
        )
        # mark input point
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
        # mark centroid
        cv2.circle(overlay, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        result["overlay"] = overlay
    return result

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
        "status": "no_refine",
        "centroid_global": [float(x_global), float(y_global)],
        "point_patch": [int(cx0), int(cy0)],
        "crop_box": [int(x0), int(y0), int(x0 + patch_length), int(y0 + patch_length)],
    }
    return float(x_global), float(y_global), crop_vis, info


# ==========================================================
# MAIN
# ==========================================================
def main(run_dir, do_refine=True):
    start_time = time.time()

    print("[INFO] Loading metadata")

    os.chdir(run_dir)

    with open("standout_nuclei.json") as f:
        nuclei = json.load(f)
    with open("dapi_tile_info.json") as f:
        dapi_tiles = json.load(f)
    with open("he_tile_info.json") as f:
        he_tiles = json.load(f)
    with open("../images_info.json") as f:
        images_info = json.load(f)
    DAPI_PATCH_LEN = 500
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
    # tif_he, he_img = open_ome_level_lazy(HE_PATH, series=0, level=0)

    global lut
    lut_path = images_info.get(
        "DAPI_LUT",
        "../../glasbey_inverted.lut",
    )
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

    dapi_img, *_ = read_image(DAPI_PATH, keep_16bit=True, level=0, channel="dapi")
    # dapi_rgb = dapi_to_lut_rgb(dapi_img, lut, threshold=300)
    # tif_dapi, dapi_img = open_ome_level_lazy(DAPI_PATH, series=0, level=0)

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

        LEVEL_DIFF = 1
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

        LEVEL_DIFF = 1
        S = 2 ** LEVEL_DIFF

        if "original" in n and "he" in n["original"] and n["original"]["he"] is not None:
            xB_tile, yB_tile = n["original"]["he"]
        else:
            meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}
            if meta.get("mode") == "rectified" and meta.get("rectified_wh") is not None:
                out_w, out_h = meta["rectified_wh"]  # NOTE: [W, H]
                xB_tile, yB_tile = (out_w / 2.0, out_h / 2.0)   # center in rectified coords
            else:
                # bbox tile fallback: center in bbox-local coords
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

    print(f"[INFO] Saved refined centroids to {out_json}", flush=True)
    print("[DONE]", flush=True)

# ==========================================================
# Entry
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 7_refine_nuclei_centroids.py <RUN_DIR>")
    run_dir = sys.argv[1]
    main(run_dir)