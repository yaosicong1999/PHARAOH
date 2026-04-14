import warnings
import os
import sys
from glob import glob
import numpy as np
from PIL import Image
import cv2
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from my_utils import upsample_tile, segment_super_dark_nuclei_full, fill_holes_binary, remove_small_components
warnings.filterwarnings('ignore')

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
    upscale=2
):
    """
    Input:  *_dapi_u8.png  (uint8 grayscale)
    Output: *_dapi_mask.png (binary mask, uint8 0/255)
    """
    try:
        if not os.path.exists(dapi_file):
            return f"Skipped (DAPI not found): {dapi_file}"

        base_name = os.path.basename(dapi_file).replace("_dapi_u8.png", "")
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

        return f"Processed: {os.path.basename(dapi_file)} otsu={int(otsu_thr)} used={thr}{extra} saved: {mask_save_path}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"

def process_he(image_file, upscale=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile, upsample_scale=upscale, n_smooth=2, intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he.png", "_he_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        return f"Processed: {image_file}"
    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

def main():
    t0 = time.perf_counter()
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 5_generate_nuclei_masks.py <output_folder>")
    out_folder = sys.argv[1]
    print(f"[INFO] Using output folder: {out_folder}")

    # ---------------- Load pilot preview params (optional) ----------------
    pilot_dapi_offset, pilot_he_thr = load_pilot_mask_params(
        out_folder,
        default_dapi_offset=0,
        default_he_thr=0.6,
    )
    print(
        f"[INFO] Using pilot params (if any): dapi_thr_offset={pilot_dapi_offset}, he_intensity_threshold={pilot_he_thr}",
        flush=True,
    )


    dapi_images = glob(os.path.join(out_folder, "*_dapi_u8.png"))
    he_images = glob(os.path.join(out_folder, "*_he.png"))

    # ---------------- DAPI ----------------f
    t_dapi_start = time.perf_counter()
    dapi_mask_upscale = 2
    print(f"[INFO] Starting DAPI processing: {len(dapi_images)} tiles from {out_folder}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_dapi, upscale=dapi_mask_upscale, THR_OFFSET=pilot_dapi_offset)
        futures = [executor.submit(func, f) for f in dapi_images]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            _ = future.result()
            done += 1
            print(f"[PROGRESS] DAPI {done}/{total}", flush=True)
    t_dapi_end = time.perf_counter()

    # ---------------- H&E ----------------
    t_he_start = time.perf_counter()
    he_mask_upscale = 2
    print(f"[INFO] Starting H&E processing: {len(he_images)} tiles", flush=True)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_he, upscale=he_mask_upscale, intensity_threshold=pilot_he_thr)
        futures = [executor.submit(func, f) for f in he_images]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            msg = future.result()  # <- 别丢掉
            done += 1
            if msg.startswith("Failed:"):
                n_fail += 1
                print("[HE FAIL]", msg, flush=True)  # 打印前几个也行
            print(f"[PROGRESS] H&E {done}/{total}", flush=True)
    print(f"[INFO] H&E done. failed={n_fail}/{len(he_images)}", flush=True)
    t_he_end = time.perf_counter()

    # ---------------- Summary ----------------
    t1 = time.perf_counter()
    print("\n================ Timing Summary ================ ")
    print(f"DAPI stage time : {t_dapi_end - t_dapi_start:.2f} s")
    print(f"H&E stage time  : {t_he_end - t_he_start:.2f} s")
    print(f"Total time     : {t1 - t0:.2f} s")
    print("================================================\n")
    print("[DONE] Nuclei masking finished", flush=True)

    mask_info = {
        "mask_scale": {
            "dapi": dapi_mask_upscale,
            "he": he_mask_upscale
        }
    }
    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}")



if __name__ == "__main__":
    main()