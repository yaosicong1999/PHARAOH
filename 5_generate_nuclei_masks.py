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
from my_utils import upsample_tile, segment_super_dark_nuclei_full
warnings.filterwarnings('ignore')



def process_dapi(dapi_file, scale=2):
    try:
        base_name = os.path.basename(dapi_file).replace("_dapi.png", "")
        folder_name = os.path.dirname(dapi_file)
        if not os.path.exists(dapi_file):
            return f"Skipped (DAPI not found): {dapi_file}"

        # Load DAPI
        dapi_rgb = cv2.imread(dapi_file)
        dapi_rgb = cv2.cvtColor(dapi_rgb, cv2.COLOR_BGR2RGB)

        # Upsample 4x
        dapi_up = upsample_tile(dapi_rgb, scale=scale)

        # Threshold to binary mask
        threshold = 20
        dapi_mask_up = (np.any(dapi_up > threshold, axis=-1)).astype(np.uint8) * 255
        dapi_mask_up_flipped = 255 - dapi_mask_up

        # Save mask (same folder)
        mask_save_path = os.path.join(folder_name, f"{base_name}_dapi_mask.png")
        cv2.imwrite(mask_save_path, dapi_mask_up_flipped)

        return f"Processed: {dapi_file}, saved as: {mask_save_path}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"


def process_he(image_file, scale=2):
    try:
        rgb_tile = np.array(Image.open(image_file))
        # Segment nuclei (no plotting)
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile, upsample_scale=scale, n_smooth=2, intensity_threshold=0.7)
        # Save upsampled mask
        mask_save_path = image_file.replace("_he.png", "_he_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        # Downsample mask back to original size
        # mask_down = resize(mask_dark.astype(float),
        #                    (rgb_tile.shape[0], rgb_tile.shape[1]),
        #                    order=0,  # nearest-neighbor
        #                    preserve_range=True,
        #                    anti_aliasing=False).astype(bool)
        # mask_down_save_path = image_file.replace("_he.png", "_he_mask_down.png")
        # cv2.imwrite(mask_down_save_path, (mask_down * 255).astype(np.uint8))
        return f"Processed: {image_file}"
    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

def main():
    t0 = time.perf_counter()
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 5_generate_nuclei_masks.py <output_folder>")
    out_folder = sys.argv[1]
    print(f"[INFO] Using output folder: {out_folder}")
    dapi_images = glob(os.path.join(out_folder, "*_dapi.png"))
    he_images = glob(os.path.join(out_folder, "*_he.png"))

    # ---------------- DAPI ----------------
    t_dapi_start = time.perf_counter()
    dapi_mask_scale = 2
    print(f"[INFO] Starting DAPI processing: {len(dapi_images)} tiles from {out_folder}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_dapi, scale=dapi_mask_scale)
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
    he_mask_scale = 2
    print(f"[INFO] Starting H&E processing: {len(he_images)} tiles", flush=True)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_he, scale=he_mask_scale)
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
            "dapi": dapi_mask_scale,
            "he": he_mask_scale
        }
    }
    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}")



if __name__ == "__main__":
    main()