import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
import os
import glob
from scipy import ndimage
import time
import json
import sys

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

def inverse_orientation_point(x, y, H, W, case_id):
    if case_id == 0:      # identity
        return x, y
    if case_id == 1:      # rot90 CW
        return y, W - 1 - x
    if case_id == 2:      # rot180
        return W - 1 - x, H - 1 - y
    if case_id == 3:      # rot90 CCW
        return H - 1 - y, x
    if case_id == 4:      # flip vertical
        return x, H - 1 - y
    if case_id == 5:      # flip horizontal
        return W - 1 - x, y
    if case_id == 6:      # transpose
        return y, x
    if case_id == 7:      # anti-transpose
        return W - 1 - y, H - 1 - x

def inverse_affine_point(xa, ya, scale, tx, ty):
    x = (xa - tx) / scale
    y = (ya - ty) / scale
    return x, y

def apply_orientation_to_tile(img, case_id):
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

# ==================================================
# Basic utils
# ==================================================
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img < 128).astype(np.uint8)


def pad_or_crop(mask, target_shape):
    """
    Return:
      out: (H,W)
      paste_x0, paste_y0: input (cropped) pasted into out at this offset
      crop_x0, crop_y0: crop start in the original input mask
    Mapping:
      x_out = (x_in - crop_x0) + paste_x0
      y_out = (y_in - crop_y0) + paste_y0
    """
    H, W = target_shape
    h, w = mask.shape
    crop_x0 = 0
    crop_y0 = 0
    # center-crop if needed
    if h > H:
        crop_y0 = (h - H) // 2
        mask = mask[crop_y0:crop_y0 + H, :]
        h = H
    if w > W:
        crop_x0 = (w - W) // 2
        mask = mask[:, crop_x0:crop_x0 + W]
        w = W
    out = np.zeros((H, W), dtype=mask.dtype)
    paste_y0 = (H - h) // 2
    paste_x0 = (W - w) // 2
    out[paste_y0:paste_y0 + h, paste_x0:paste_x0 + w] = mask
    return out, paste_x0, paste_y0, crop_x0, crop_y0



def warp(mask, scale, tx, ty, out_shape):
    H, W = out_shape
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, 0.0, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)


def aligned_to_he_mask_img(
        xB_aligned, yB_aligned,
        final, H, W,
        paste_x0, paste_y0,
        crop_x0, crop_y0
):
    cx = W / 2
    cy = H / 2
    x_maskB = (xB_aligned - final["tx"] - cx) / final["scale"] + cx
    y_maskB = (yB_aligned - final["ty"] - cy) / final["scale"] + cy
    x_raw = (x_maskB - paste_x0) + crop_x0
    y_raw = (y_maskB - paste_y0) + crop_y0
    return x_raw, y_raw

def iou(A, B):
    inter = np.sum((A > 0) & (B > 0))
    uni = np.sum((A > 0) | (B > 0))
    return inter / (uni + 1e-6)


def downsample(mask, ds):
    H, W = mask.shape
    return cv2.resize(mask, (W // ds, H // ds),
                      interpolation=cv2.INTER_NEAREST)


# ==================================================
# Parallel Stage 1 helpers
# ==================================================
def eval_one(s, tx, ty, A, B, H, W):
    B_s = cv2.resize(B, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    B_s, _, _, _, _ = pad_or_crop(B_s, (H, W))
    B_st = warp(B_s, 1.0, tx, ty, (H, W))
    score = iou(A, B_st)
    return score, s, tx, ty


def coarse_search_ds_parallel(maskA, maskB,
                              ds=4,
                              scale_range=(1.3, 2.0),
                              scale_step=0.05,
                              shift_range=60,
                              shift_step=10,
                              n_jobs=None):

    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    A = downsample(maskA, ds)
    B = downsample(maskB, ds)
    H, W = A.shape

    scales = np.arange(scale_range[0], scale_range[1] + 1e-9, scale_step)
    shifts = range(-shift_range, shift_range + 1, shift_step)

    tasks = (
        delayed(eval_one)(s, tx, ty, A, B, H, W)
        for s in scales
        for tx in shifts
        for ty in shifts
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=0
    )(tasks)

    best_score, best_s, best_tx, best_ty = max(results, key=lambda x: x[0])

    return {
        "scale": best_s,
        "tx": best_tx * ds,
        "ty": best_ty * ds,
        "score_ds": best_score
    }


# ==================================================
# Stage 2: local refine on full resolution (unchanged)
# ==================================================

def refine_eval_one(args):
    s, tx, ty, maskA, maskB, H, W = args
    B_s = cv2.resize(maskB, None, fx=s, fy=s,
                     interpolation=cv2.INTER_NEAREST)
    B_s, _, _, _, _ = pad_or_crop(B_s, (H, W))
    B_st = warp(B_s, 1.0, tx, ty, (H, W))
    score = iou(maskA, B_st)
    return score, s, tx, ty

def refine_full(maskA, maskB, init,
                          shift_delta=20,
                          shift_step=5,
                          n_jobs=None):

    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    H, W = maskA.shape
    s0, tx0, ty0 = init["scale"], init["tx"], init["ty"]

    scales = [s0 * 0.985, s0, s0 * 1.015]
    shifts_x = range(tx0 - shift_delta, tx0 + shift_delta + 1, shift_step)
    shifts_y = range(ty0 - shift_delta, ty0 + shift_delta + 1, shift_step)

    tasks = [
        (s, tx, ty, maskA, maskB, H, W)
        for s in scales
        for tx in shifts_x
        for ty in shifts_y
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=0
    )(
        delayed(refine_eval_one)(t) for t in tasks
    )

    score, s, tx, ty = max(results, key=lambda x: x[0])
    return {"scale": s, "tx": tx, "ty": ty, "score": score}
def find_isolated_components(mask, min_area=10, radii=(3,5,7), area_iqr_k=1.5):
    H, W = mask.shape
    x_lo, x_hi = W/6, 5*W/6
    y_lo, y_hi = H/6, 5*H/6

    labeled, _ = ndimage.label(mask > 0)
    objects = ndimage.find_objects(labeled)

    candidates = []

    for lab, slc in enumerate(objects, start=1):
        comp = (labeled[slc] == lab)
        area = comp.sum()
        if area < min_area:
            continue

        ys, xs = np.where(comp)
        cy = ys.mean() + slc[0].start
        cx = xs.mean() + slc[1].start
        if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
            continue

        full_local = (mask[slc] > 0)
        isolated = True
        for r in radii:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
            dil = cv2.dilate(comp.astype(np.uint8), k)
            ring = dil & (~comp)
            if np.any(ring & full_local & (~comp)):
                isolated = False
                break

        if isolated:
            candidates.append({
                "comp": comp,
                "bbox": slc,
                "area": area,
                "centroid": (cx, cy)
            })

    if not candidates:
        return []

    areas = np.array([c["area"] for c in candidates])
    q1, q3 = np.percentile(areas, [25, 75])
    max_area = q3 + area_iqr_k*(q3-q1)

    return [c for c in candidates if c["area"] <= max_area]


def touch(comp_bool, other_bool, radius=3):
    """
    comp_bool: bool mask (ROI)
    other_bool: bool mask (ROI)
    """
    comp_u8 = comp_bool.astype(np.uint8)
    other_u8 = other_bool.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dil = cv2.dilate(comp_u8, k)
    return np.any((dil > 0) & (other_u8 > 0))

def pair_isolated_cells(
    A_cells,
    B_cells,
    A_iso_mask,
    B_iso_mask,
    min_area=60,
    min_sym_cov=0.4,
    top_k=3,
    debug=False,
    touch_radius=3
):
    H, W = A_iso_mask.shape
    results = []

    # debug counters
    n_total = 0
    n_bbox = 0
    n_inter = 0
    n_sym = 0
    n_touchA = 0
    n_touchB = 0

    for a in A_cells:
        if a["area"] < min_area:
            continue
        ay0, ay1 = a["bbox"][0].start, a["bbox"][0].stop
        ax0, ax1 = a["bbox"][1].start, a["bbox"][1].stop

        for b in B_cells:
            if b["area"] < min_area:
                continue
            n_total += 1

            by0, by1 = b["bbox"][0].start, b["bbox"][0].stop
            bx0, bx1 = b["bbox"][1].start, b["bbox"][1].stop

            # intersection bbox just for quick reject
            y0, y1 = max(ay0, by0), min(ay1, by1)
            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            if y0 >= y1 or x0 >= x1:
                continue
            n_bbox += 1

            # intersection overlap
            a_loc = a["comp"][y0-ay0:y1-ay0, x0-ax0:x1-ax0].astype(bool)
            b_loc = b["comp"][y0-by0:y1-by0, x0-bx0:x1-bx0].astype(bool)
            inter = np.sum(a_loc & b_loc)
            if inter == 0:
                continue
            n_inter += 1

            covA = inter / (a["area"] + 1e-6)
            covB = inter / (b["area"] + 1e-6)
            sym = min(covA, covB)
            if sym < min_sym_cov:
                continue
            n_sym += 1

            # ==================================================
            # NEW: ROI = union(bboxA, bboxB) + padding
            # ==================================================
            pad = touch_radius + 2
            ry0 = max(0, min(ay0, by0) - pad)
            ry1 = min(H, max(ay1, by1) + pad)
            rx0 = max(0, min(ax0, bx0) - pad)
            rx1 = min(W, max(ax1, bx1) + pad)

            # roi masks for all isolated comps
            A_iso_roi = (A_iso_mask[ry0:ry1, rx0:rx1] > 0)
            B_iso_roi = (B_iso_mask[ry0:ry1, rx0:rx1] > 0)

            # build full a_roi and b_roi (NOT just intersection part)
            a_roi = np.zeros((ry1-ry0, rx1-rx0), dtype=bool)
            b_roi = np.zeros((ry1-ry0, rx1-rx0), dtype=bool)

            # paste A comp into ROI
            a_y0 = ay0 - ry0
            a_x0 = ax0 - rx0
            a_roi[a_y0:a_y0 + a["comp"].shape[0], a_x0:a_x0 + a["comp"].shape[1]] |= a["comp"].astype(bool)

            # paste B comp into ROI
            b_y0 = by0 - ry0
            b_x0 = bx0 - rx0
            b_roi[b_y0:b_y0 + b["comp"].shape[0], b_x0:b_x0 + b["comp"].shape[1]] |= b["comp"].astype(bool)

            # other masks (exclude itself)  —— 注意这里是 bool 的 ~，不会踩 uint8 坑
            A_other = A_iso_roi & (~a_roi)
            B_other = B_iso_roi & (~b_roi)

            # bidirectional non-touch
            if touch(a_roi, B_other, radius=touch_radius):
                n_touchA += 1
                continue
            if touch(b_roi, A_other, radius=touch_radius):
                n_touchB += 1
                continue

            f1 = 2 * covA * covB / (covA + covB + 1e-6)

            results.append({
                "A_centroid": a["centroid"],
                "B_centroid": b["centroid"],
                "A_comp": a,
                "areaA": a["area"],
                "areaB": b["area"],
                "covA": float(covA),
                "covB": float(covB),
                "sym_cov": float(sym),
                "f1": float(f1)
            })

    results.sort(key=lambda d: d["f1"], reverse=True)

    if debug:
        print(f"  [DEBUG pairing] total={n_total}, bbox_ok={n_bbox}, inter>0={n_inter}, pass_sym={n_sym}, fail_touchA={n_touchA}, fail_touchB={n_touchB}, kept={len(results)}")

    return results[:top_k]



# ==================================================
# Main (single pair)
# ==================================================
if __name__ == "__main__":
    base_dir = sys.argv[1]
    print(f"[INFO] Using output folder: {base_dir}")
    t_global_start = time.perf_counter()
    with open(os.path.join(base_dir, "../images_info.json"), "r") as f:
        case_id = json.load(f)['DAPI_orientation_case']
    with open(os.path.join(base_dir, "nuclei_mask_info.json"), "r") as f:
        mask_scale_data = json.load(f)
    DAPI_MASK_SCALE = mask_scale_data['mask_scale']['dapi']
    HE_MASK_SCALE = mask_scale_data['mask_scale']['he']

    dapi_files = sorted(
        glob.glob(os.path.join(base_dir, "*_dapi_mask.png"))
    )
    total = len(dapi_files)
    print(f"[INFO] Found {total} tiles for standout nuclei detection")

    all_nuclei_records = []
    for idx, dapi_path in enumerate(dapi_files, 1):
        t_patch_start = time.perf_counter()
        # -------- derive prefix --------
        fname = os.path.basename(dapi_path)
        prefix = fname.replace("_dapi_mask.png", "")
        he_path = os.path.join(base_dir, f"{prefix}_he_mask.png")
        out_path = os.path.join(base_dir, f"{prefix}_aligned.jpg")
        if not os.path.exists(he_path):
            print(f"[SKIP] HE mask not found for {prefix}")
            continue

        print(f"\n=== Processing {prefix} ({idx}/{len(dapi_files)}) ===")
        nucleus_id = 0
        # -------- read masks --------
        maskA = read_mask(dapi_path)
        maskA = apply_orientation_to_tile(maskA, case_id)
        maskB_raw = read_mask(he_path)
        maskB, paste_x0, paste_y0, crop_x0, crop_y0 = pad_or_crop(maskB_raw, maskA.shape)
        # -------- Stage 1 (parallel coarse) --------
        coarse = coarse_search_ds_parallel(
            maskA, maskB,
            ds=4,
            scale_range=(1.3, 2.0),
            scale_step=0.015,
            shift_range=100,
            shift_step=10
        )
        print("  Coarse:", coarse)
        # -------- Stage 2 (parallel refine) --------
        final = refine_full(
            maskA, maskB,
            coarse,
            shift_delta=20,
            shift_step=4
        )
        print("  Final:", final)
        # -------- Final warp --------
        H, W = maskA.shape
        B_final = warp(maskB, final["scale"], final["tx"], final["ty"], (H, W))
        # -------- Save aligned result --------
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[..., 0] = (maskA > 0) * 255   # red: DAPI
        overlay[..., 1] = (B_final > 0) * 255 # green: HE
        cv2.imwrite(out_path, overlay)
        print(f"  Saved -> {out_path}")
        # -------- Isolated components --------
        A_iso = find_isolated_components(maskA, min_area=60)
        B_iso = find_isolated_components(B_final, min_area=60)
        print(f"Isolated A: {len(A_iso)}, Isolated B: {len(B_iso)}")
        # ---------- build isolated masks ----------
        A_iso_mask = np.zeros_like(maskA, dtype=np.uint8)
        for c in A_iso:
            slc = c["bbox"]
            A_iso_mask[slc][c["comp"]] = 1
        B_iso_mask = np.zeros_like(maskA, dtype=np.uint8)
        for c in B_iso:
            slc = c["bbox"]
            B_iso_mask[slc][c["comp"]] = 1
        pairs = pair_isolated_cells(
            A_iso,
            B_iso,
            A_iso_mask,
            B_iso_mask,
            min_area=60,
            min_sym_cov=0.4,
            top_k=3,
            debug=True
        )
        if len(pairs) == 0:
            print("  [DEBUG] No valid A/B pairs found after filtering")
        for p in pairs:
            H, W = maskA.shape
            xA_aligned, yA_aligned = p["A_centroid"]
            xA_dapi, yA_dapi = inverse_orientation_point(
                xA_aligned, yA_aligned, H, W, case_id
            )
            DAPI_MASK_SCALE = 2
            xA_img = xA_dapi / DAPI_MASK_SCALE
            yA_img = yA_dapi / DAPI_MASK_SCALE

            xB_aligned, yB_aligned = p["B_centroid"]
            xB_mask, yB_mask = aligned_to_he_mask_img(
                xB_aligned, yB_aligned,
                final, H, W,
                paste_x0, paste_y0,
                crop_x0, crop_y0
            )
            xB_img = xB_mask / HE_MASK_SCALE
            yB_img = yB_mask / HE_MASK_SCALE

            print(
                "DAPI:", np.round(p["A_centroid"], 2),
                "DAPI original:", np.round(np.array([xA_img, yA_img]), 2),
                "HE:", np.round(p["B_centroid"], 2),
                "HE original:", np.round(np.array([xB_img, yB_img]), 2),
                "Overlapping ratio A: ", np.round(p["covA"], 2),
                "Overlapping ratio B: ", np.round(p["covB"], 2),
                "F1-score: ", np.round(p["f1"], 2)
            )
            record = {
                "tile": prefix,
                "nucleus_id": nucleus_id,
                "aligned": {
                    "dapi": [float(xA_aligned), float(yA_aligned)],
                    "he": [float(xB_aligned), float(yB_aligned)],
                },
                "original": {
                    "dapi": [float(xA_img), float(yA_img)],
                    "he": [float(xB_img), float(yB_img)],
                },
                "area": {
                    "dapi": int(p["areaA"]),
                    "he": int(p["areaB"]),
                },
                "bbox_aligned": {
                    "y0": int(p["A_comp"]["bbox"][0].start),
                    "y1": int(p["A_comp"]["bbox"][0].stop),
                    "x0": int(p["A_comp"]["bbox"][1].start),
                    "x1": int(p["A_comp"]["bbox"][1].stop),
                },
                "metrics": {
                    "covA": float(p["covA"]),
                    "covB": float(p["covB"]),
                    "f1": float(p["f1"]),
                }
            }
            all_nuclei_records.append(record)
            nucleus_id = nucleus_id + 1

        # -------- Visualization --------
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[..., 0] = (maskA > 0) * 120
        vis[..., 1] = (B_final > 0) * 120
        alpha = 0.6
        for p in pairs:
            comp = p["A_comp"]["comp"]
            slc = p["A_comp"]["bbox"]
            for c in [0, 1]:
                vis_ch = vis[slc[0], slc[1], c]
                vis_ch[comp] = (
                    alpha * 255 + (1 - alpha) * vis_ch[comp]
                ).astype(np.uint8)

        cv2.imwrite(os.path.join(base_dir, f"{prefix}_standout.jpg"), vis)
        # ================================
        # PATCH TIMER (END)
        # ================================
        t_patch_end = time.perf_counter()
        print(f"[TIME] {prefix}: {t_patch_end - t_patch_start:.2f} sec")
        print(f"[PROGRESS] TILES {idx}/{total}", flush=True)

    json_path = os.path.join(base_dir, "standout_nuclei.json")
    with open(json_path, "w") as f:
        json.dump(all_nuclei_records, f, indent=2)
    print(f"[INFO] Saved standout nuclei info -> {json_path}")
    # ================================
    # OVERALL TIMER (END)
    # ================================
    t_global_end = time.perf_counter()
    print("\n==============================")
    print(f"[TOTAL TIME] {t_global_end - t_global_start:.2f} sec")
    print("==============================")
    print("[INFO] Found ...", flush=True)
    print("[DONE]", flush=True)