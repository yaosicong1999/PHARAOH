
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
    read_he_patch
)
import math
from pathlib import Path

def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

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

def refine_nucleus_center_from_patch(
    img,
    x_global,
    y_global,
    x_tile,
    y_tile,
    tile_id,
    nucleus_id,
    type=None,
    patch_length=60,
    intensity_threshold=0.5,
    n_smooth=1,
    upsample_scale=1,
    out_dir="",
    save_patch=True,
    save_overlay=True,
    save_raw_dapi=True,          # <-- 新增
    raw_dapi_ext="png"           # <-- 新增: "png" or "tif"
):
    half = patch_length // 2
    H, W = img.shape[:2]
    # ===============================
    # (1) Crop H&E patch
    # ===============================
    y0 = max(0, y_global - half)
    y1 = min(H, y_global + half)
    x0 = max(0, x_global - half)
    x1 = min(W, x_global + half)
    crop = read_he_patch(img, x0, y0, patch_length, patch_length)

    if crop.size == 0:
        raise ValueError(
            f"Empty H&E crop at (x_global={x_global}, y_global={y_global})"
        )
    # DAPI: 同时保留 raw + 生成用于显示/后续处理的 LUT 版
    crop_raw = None
    if type == "dapi":
        crop_raw = crop.copy()  # 这里的 crop 还是 uint16 / 原始值
        crop = dapi_to_lut_rgb(crop_raw, lut, threshold=1000)  # crop 变成 RGB uint8
        print("LUT-ed dapi crop!")

    # ===============================
    # ========= default case: do not refine =========
    # ===============================
    h, w = crop.shape[:2]
    cx0 = int(np.clip(x_global - x0, 0, w - 1))
    cy0 = int(np.clip(y_global - y0, 0, h - 1))
    status = "keep_original"
    dapi_mask_overlay_from_func = None
    he_mask_overlay_from_func = None
    centroid_patch = np.array([cx0, cy0], dtype=np.float32)
    if type == "he":
        print("    ## HE detected, get HE masking...")

        _, mask_dark = segment_super_dark_nuclei_full(
            crop,
            upsample_scale=upsample_scale,
            n_smooth=n_smooth,
            intensity_threshold=intensity_threshold
        )

        # nucleus=1
        mask_fg = (mask_dark == 0).astype(np.uint8)

        # ✅ 关键：转成 nucleus=0 / background=255 的 mask，才能直接复用你的 func
        he_mask_8 = (255 - mask_fg * 255).astype(np.uint8)

        res = get_nucleus_centroid_from_mask_at_xy(
            he_mask_8,
            cx0, cy0,
            return_overlay=True,
            highlight_color=(0, 0, 255),  # BGR：红色高亮 blob（你想要红的话）
            alpha=0.5
        )

        if res is not None:
            centroid_patch = np.array(res["centroid"])
            status = "hit_L0"
            he_mask_overlay_from_func = res.get("overlay")
    elif type == "dapi":
        print("    ** DAPI detected, get DAPI masking...")
        mask_fg = get_dapi_mask(crop, threshold=20)

        res = get_nucleus_centroid_from_mask_at_xy(
            mask_fg,
            cx0, cy0,
            return_overlay=True,  # ✅ 开启 overlay
            highlight_color=(0, 0, 255),  # 蓝色高亮 blob（BGR）
            alpha=0.5
        )

        dapi_mask_overlay_from_func = None
        if res is not None:
            centroid_patch = np.array(res["centroid"])
            status = "hit_L0"
            dapi_mask_overlay_from_func = res.get("overlay")
        else:
            dapi_mask_overlay_from_func = None
    else:
        raise ValueError(
            f"Type must be either he or dapi!"
        )
    # -------------------------------
    # L1: identity-preserving fallback to tile-level masking
    # -------------------------------
    if status != "hit_L0":
        print("    ---> Fall back to try using the tile-level nuclei mask.")
        with open("nuclei_mask_info.json", "r") as f:
            mask_scale_data = json.load(f)
        MASK_SCALE = mask_scale_data['mask_scale'][type]
        mask = cv2.imread(f"{tile_id}_{type}_mask.png", cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask from {f'{tile_id}_{type}_mask.png'}")
        res = get_nucleus_centroid_from_mask_at_xy(
            mask,
            x_tile * MASK_SCALE, y_tile * MASK_SCALE,
            return_overlay=True
        )
        if res is not None:
            print("    ** Successfully use the tile-level nuclei mask.")
            centroid_tile = np.array(res['centroid']) / MASK_SCALE
            centroid_patch = centroid_tile - np.array([x_tile, y_tile]) + np.array([cx0, cy0])
            status = "fallback_L1"
        else:
            print("    ** Cannot use the tile-level nuclei mask. Use the default coordinates")
    # ===============================
    # (4) Map back to global coords
    # ===============================
    centroid_patch = centroid_patch / float(upsample_scale)
    output_x_global = centroid_patch[0] + x0
    output_y_global = centroid_patch[1] + y0
    info = {
        "status": status,
        "centroid_patch": centroid_patch.tolist(),
        "centroid_global": [float(output_x_global), float(output_y_global)],
        "crop_box": (x0, y0, x1, y1),
    }
    print(f"    Old centroids in the patch-level is {np.array([cx0, cy0])}")
    print(f"    New centroids in the patch-level is {centroid_patch}")
    print(f"    Old centroids in the global-level is {np.array([x_global, y_global])}")
    print(f"    New centroids in the global-level is {np.array([float(output_x_global), float(output_y_global)])}")
    # ===============================
    # (5) Save outputs
    # ===============================

    # ---------- 5.1 Save patch images (no centroid) ----------
    if save_patch:
        # ---- save RGB patch (LUT-ed DAPI or HE) ----
        patch_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png"
        cv2.imwrite(patch_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # ---- save mask (no centroid) ----
        mask_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_mask.png"
        if type == "he":
            mask_to_save = (255 - mask_fg * 255).astype(np.uint8)
        elif type == "dapi":
            mask_to_save = mask_fg.astype(np.uint8)
        else:
            raise ValueError(f"Unknown type: {type}")

        cv2.imwrite(mask_path, mask_to_save)

        # ---- save raw DAPI patch (uint16) ----
        if type == "dapi" and save_raw_dapi and (crop_raw is not None):
            raw_patch_path = (
                f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw_patch.{raw_dapi_ext.lower()}"
            )
            raw16 = ensure_gray_uint16(crop_raw)
            raw_vis8 = stretch_to_uint8_percentile(raw16, p_low=1, p_high=99.7)
            raw_norm_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw.png"
            cv2.imwrite(raw_norm_path, raw_vis8)

    # ---------- 5.2 Save overlay images (with centroid) ----------
    if save_overlay:
        cx, cy = int(round(centroid_patch[0])), int(round(centroid_patch[1]))

        # ---- overlay on RGB patch ----
        patch_overlay = crop.copy()
        cv2.circle(patch_overlay, (cx, cy), 2, (255, 0, 0), -1)

        patch_overlay_path = (
            f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        )
        cv2.imwrite(
            patch_overlay_path,
            cv2.cvtColor(patch_overlay, cv2.COLOR_RGB2BGR)
        )

        # ---- overlay on mask ----
        if type == "he":
            mask_vis = (255 - mask_fg * 255).astype(np.uint8)
        elif type == "dapi":
            mask_vis = mask_fg.astype(np.uint8)
        else:
            raise ValueError(f"Unknown type: {type}")

        # ---- overlay on mask ----
        mask_overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_mask_overlay.png"

        if type == "dapi" and dapi_mask_overlay_from_func is not None:
            cv2.imwrite(mask_overlay_path, dapi_mask_overlay_from_func)

        elif type == "he" and he_mask_overlay_from_func is not None:
            cv2.imwrite(mask_overlay_path, he_mask_overlay_from_func)

        else:
            # fallback（保持你原来的）
            mask_overlay = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            cv2.circle(mask_overlay, (cx, cy), 2, (0, 0, 255), -1)
            cv2.imwrite(mask_overlay_path, mask_overlay)

        # ---- overlay on raw DAPI (uint16) ----
        if type == "dapi" and save_raw_dapi and (crop_raw is not None):
            cx, cy = int(round(centroid_patch[0])), int(round(centroid_patch[1]))
            raw_overlay = cv2.cvtColor(raw_vis8, cv2.COLOR_GRAY2BGR)
            cv2.circle(raw_overlay, (cx, cy), 2, (0, 0, 255), -1)  # 红点
            raw_overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_dapi_raw_overlay.png"
            cv2.imwrite(raw_overlay_path, raw_overlay)

    return output_x_global, output_y_global, crop, info


# ==========================================================
# MAIN
# ==========================================================
def main(run_dir):
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
    DAPI_PATCH_LEN = 100
    HE_PATCH_LEN = load_initial_affine_and_compute_he_patch(
        run_dir,
        images_info,
        dapi_patch_len=DAPI_PATCH_LEN,
        margin=1.1
    )
    HE_PATH = images_info["HE_path"]
    DAPI_PATH = images_info["DAPI_path"]

    print("[INFO] Loading full-res images. This could take about 1 min ⚠️", flush=True)


    # he_img, *_ = read_image(HE_PATH, keep_16bit=True, level=0)
    tif_he, he_img = open_ome_level_lazy(HE_PATH, series=0, level=0)

    global lut
    lut_path = images_info.get(
        "DAPI_LUT",
        "../../glasbey_inverted.lut",
    )
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

    # dapi_img, *_ = read_image(DAPI_PATH, keep_16bit=True, level=0)
    # dapi_rgb = dapi_to_lut_rgb(dapi_img, lut, threshold=300)
    tif_dapi, dapi_img = open_ome_level_lazy(DAPI_PATH, series=0, level=0)

    out_dir = "../nuclei_patches"
    os.makedirs(out_dir, exist_ok=True)
    output_coord_record = []
    total = len(nuclei)
    print(f"[INFO] Refining {total} nuclei centroids", flush=True)

    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n["nucleus_id"]
        dapi_info = dapi_tiles[tile_id]

        LEVEL_DIFF = 1
        S = 2 ** LEVEL_DIFF

        xA_tile, yA_tile = n["original"]["dapi"]
        xA_global = int(round(dapi_info["x0"] * S + xA_tile * S))
        yA_global = int(round(dapi_info["y0"] * S + yA_tile * S))

        output_xA_global, output_yA_global, _, _ = refine_nucleus_center_from_patch(
            # dapi_rgb,
            dapi_img,
            xA_global, yA_global,
            xA_tile, yA_tile,
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
            "dapi_centroid_global": [float(output_xA_global), float(output_yA_global)],
            "he_centroid_global": None,  # 先占位
        })
        print(f"[PROGRESS] DAPI {i}/{total}", flush=True)


    for i, n in enumerate(nuclei, 1):
        tile_id = n["tile"]
        nucleus_id = n["nucleus_id"]
        he_info = he_tiles[tile_id]

        LEVEL_DIFF = 1
        S = 2 ** LEVEL_DIFF

        xB_tile, yB_tile = n["original"]["he"]
        xB_global = int(round(he_info["x0"] * S + xB_tile * S))
        yB_global = int(round(he_info["y0"] * S + yB_tile * S))

        output_xB_global, output_yB_global, _, _ = refine_nucleus_center_from_patch(
            he_img,
            xB_global, yB_global,
            xB_tile, yB_tile,
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
    main(sys.argv[1])
