
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
)

def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

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
        cv2.circle(overlay, (int(cx), int(cy)), 2, (255, 0, 0), -1)
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
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(
            f"Empty H&E crop at (x_global={x_global}, y_global={y_global})"
        )
    # ===============================
    # ========= default case: do not refine =========
    # ===============================
    h, w = crop.shape[:2]
    cx0 = int(np.clip(x_global - x0, 0, w - 1))
    cy0 = int(np.clip(y_global - y0, 0, h - 1))
    status = "keep_original"
    centroid_patch = np.array([cx0, cy0], dtype=np.float32)
    if type == "he":
        print("    ## HE detected, get HE masking...")
        # ===============================
        # (2) Segment super-dark nuclei (L0)
        # ===============================
        _, mask_dark = segment_super_dark_nuclei_full(
            crop,
            upsample_scale=upsample_scale,
            n_smooth=n_smooth,
            intensity_threshold=intensity_threshold
        )
        # nucleus = 1
        mask_fg = (mask_dark == 0).astype(np.uint8)
        # ===============================
        # (3) Connected components (L0)
        # ===============================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_fg, connectivity=8
        )
        if num_labels > 1:
            label_at_center = labels[cy0, cx0]
            if label_at_center != 0:
                centroid_patch = centroids[label_at_center]
                status = "hit_L0"
    elif type == "dapi":
        print("    ** DAPI detected, get DAPI masking...")
        mask_fg = get_dapi_mask(crop, threshold=20)
        res = get_nucleus_centroid_from_mask_at_xy(
            mask_fg,
            cx0, cy0)
        if res is not None:
            centroid_patch = np.array(res['centroid'])
            status = "hit_L0"
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
    if save_patch:
        cv2.imwrite(
            f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png",
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        )
        if type == "he":
            cv2.imwrite(
                f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_mask.png",
                (255-mask_fg * 255).astype(np.uint8)
            )
        elif type == "dapi":
            cv2.imwrite(
                f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_mask.png",
                mask_fg.astype(np.uint8)
            )
    if save_overlay:
        overlay = crop.copy()
        cx, cy = int(round(centroid_patch[0])), int(round(centroid_patch[1]))
        cv2.circle(
            overlay,
            (cx, cy),
            2,
            (255, 0, 0),
            -1
        )
        out_overlay = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if type == "he":
            overlay = (255-mask_fg * 255).astype(np.uint8).copy()
        elif type == "dapi":
            overlay = mask_fg.astype(np.uint8).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        cx, cy = int(round(centroid_patch[0])), int(round(centroid_patch[1]))
        cv2.circle(
            overlay,
            (cx, cy),
            2,
            (255, 0, 0),
            -1
        )
        out_overlay = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_mask_overlay.png"
        cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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

    HE_PATH = images_info["HE_path"]
    DAPI_PATH = images_info["DAPI_path"]

    print("[INFO] Loading full-res images. This could take about 1 min ⚠️", flush=True)

    he_img, *_ = read_image(HE_PATH, keep_16bit=True, level=0)

    lut_path = images_info.get(
        "DAPI_LUT",
        "/Users/sicongy/Documents/GitHub/rotation_1/LUT/glasbey_inverted.lut",
    )
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

    dapi_img, *_ = read_image(DAPI_PATH, keep_16bit=True, level=0)
    dapi_rgb = dapi_to_lut_rgb(dapi_img, lut, threshold=300)

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
            dapi_rgb,
            xA_global, yA_global,
            xA_tile, yA_tile,
            tile_id, nucleus_id,
            type="dapi",
            patch_length=100,
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
            patch_length=60,
            out_dir=out_dir,
            save_patch=True,
            save_overlay=True,
        )
        output_coord_record[i - 1]["he_centroid_global"] = [
            float(output_xB_global),
            float(output_yB_global),
        ]

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

    # main("/Users/sicongy/PycharmProjects/iStar/tkinter_version_v3/runs_202512182051/tiles/")
