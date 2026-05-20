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


def load_pilot_mask_params(out_folder: str, default_he0_thr=0.6, default_he_thr=0.6):
    """
    Try read:
      <out_folder>/../pilot_tiles/pilot_output_parameters.json

    Supports JSON layout:
      1) {
           "mask_preview": {
             "he0_intensity_threshold": 0.74,
             "he_intensity_threshold": 0.57
           }
         }
      2) {
           "he0_intensity_threshold": 0.74,
           "he_intensity_threshold": 0.57
         }

    Returns:
      (he0_intensity_threshold: float, he_intensity_threshold: float)
    """
    path = os.path.join(out_folder, "../pilot_tiles", "pilot_output_parameters.json")
    if not os.path.exists(path):
        return float(default_he0_thr), float(default_he_thr)

    try:
        data = json.load(open(path, "r"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Use defaults.", flush=True)
        return float(default_he0_thr), float(default_he_thr)

    src = data.get("mask_preview", data) if isinstance(data, dict) else {}

    he0_thr = src.get("he0_intensity_threshold", default_he0_thr)
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


def process_he0(image_file, upscale=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=2,
            intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he0.png", "_he0_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        return f"Processed: {image_file}"
    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"


def process_he(image_file, upscale=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=2,
            intensity_threshold=float(intensity_threshold)
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
    pilot_he0_thr, pilot_he_thr = load_pilot_mask_params(
        out_folder,
        default_he0_thr=0.6,
        default_he_thr=0.6,
    )
    print(
        f"[INFO] Using pilot params (if any): "
        f"he0_intensity_threshold={pilot_he0_thr}, "
        f"he_intensity_threshold={pilot_he_thr}",
        flush=True,
    )

    he0_images = glob(os.path.join(out_folder, "*_he0.png"))
    he_images = glob(os.path.join(out_folder, "*_he.png"))

    # ---------------- HE0 ----------------
    t_he0_start = time.perf_counter()
    he0_mask_upscale = 2

    print(f"[INFO] Starting HE0 processing: {len(he0_images)} tiles from {out_folder}", flush=True)
    n_fail_he0 = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_he0, upscale=he0_mask_upscale, intensity_threshold=pilot_he0_thr)
        futures = [executor.submit(func, f) for f in he0_images]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            msg = future.result()
            done += 1
            if msg.startswith("Failed:"):
                n_fail_he0 += 1
                print("[HE0 FAIL]", msg, flush=True)
            print(f"[PROGRESS] HE0 {done}/{total}", flush=True)
    print(f"[INFO] HE0 done. failed={n_fail_he0}/{len(he0_images)}", flush=True)
    t_he0_end = time.perf_counter()

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
            msg = future.result()
            done += 1
            if msg.startswith("Failed:"):
                n_fail += 1
                print("[HE FAIL]", msg, flush=True)
            print(f"[PROGRESS] H&E {done}/{total}", flush=True)
    print(f"[INFO] H&E done. failed={n_fail}/{len(he_images)}", flush=True)
    t_he_end = time.perf_counter()

    # ---------------- Summary ----------------
    t1 = time.perf_counter()
    print("\n================ Timing Summary ================ ")
    print(f"HE0 stage time  : {t_he0_end - t_he0_start:.2f} s")
    print(f"H&E stage time  : {t_he_end - t_he_start:.2f} s")
    print(f"Total time      : {t1 - t0:.2f} s")
    print("================================================\n")
    print("[DONE] Nuclei masking finished", flush=True)

    mask_info = {
        "mask_scale": {
            "he0": he0_mask_upscale,
            "he": he_mask_upscale
        },
        "mask_parameters": {
            "he0_intensity_threshold": float(pilot_he0_thr),
            "he_intensity_threshold": float(pilot_he_thr),
        }
    }

    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}")


if __name__ == "__main__":
    main()