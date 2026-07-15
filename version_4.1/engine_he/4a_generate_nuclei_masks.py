# ============================================================
# 4a_generate_nuclei_masks.py - Stage 4a: Nuclei mask generation
#
# Generates HE0 and HE nuclei masks (super-dark segmentation) for
# every tile in parallel; writes nuclei_mask_info.json.
# Usage: python 4a_generate_nuclei_masks.py <TILES_DIR>
# ============================================================
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


def load_stage4a_params_from_parameters_json(
    out_folder: str,
    default_he0_n_smooth=2,
    default_he0_thr=0.6,
    default_he0_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
):
    """
    Read ../../parameters.json and return stage4a params.
    Falls back to the provided defaults if the file or keys are missing.
    """
    from my_utils import cli_params_block

    params = {
        "he0_mask_n_smooth": int(default_he0_n_smooth),
        "he0_mask_intensity_threshold": float(default_he0_thr),
        "he0_mask_upscale_factor": int(default_he0_upscale),
        "he_mask_n_smooth": int(default_he_n_smooth),
        "he_mask_intensity_threshold": float(default_he_thr),
        "he_mask_upscale_factor": int(default_he_upscale),
    }

    data = cli_params_block()

    stage4a = data.get("stage4a", {}) if isinstance(data, dict) else {}
    for k in params:
        if k in stage4a:
            try:
                params[k] = type(params[k])(stage4a[k])
            except Exception:
                pass
    return params


def load_stage4a_effective_params(
    out_folder: str,
    default_he0_n_smooth=2,
    default_he0_thr=0.6,
    default_he0_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
):
    """
    Resolve Stage 4a parameters with this priority:
      1) parameters.json -> stage4a
      2) pilot_output_parameters.json overrides:
           - he0_intensity_threshold
           - he_intensity_threshold
      3) hard-coded defaults
    """
    params = load_stage4a_params_from_parameters_json(
        out_folder,
        default_he0_n_smooth=default_he0_n_smooth,
        default_he0_thr=default_he0_thr,
        default_he0_upscale=default_he0_upscale,
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
    for pilot_key, param_key in (
        ("he0_intensity_threshold", "he0_mask_intensity_threshold"),
        ("he_intensity_threshold", "he_mask_intensity_threshold"),
    ):
        if pilot_key in src:
            try:
                params[param_key] = max(0.0, min(1.0, float(src[pilot_key])))
            except Exception:
                pass
    return params


def process_he0(image_file, upscale=2, n_smooth=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=int(n_smooth),
            intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace("_he0.png", "_he0_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        return f"Processed: {image_file}"
    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"


def process_he(image_file, upscale=2, n_smooth=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        labeled_mask, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=int(n_smooth),
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
        raise RuntimeError("Usage: python 4a_generate_nuclei_masks.py <output_folder>")
    out_folder = sys.argv[1]
    print(f"[INFO] Using output folder: {out_folder}")

    # ---------------- Resolve effective params (parameters.json + pilot) ----------------
    p = load_stage4a_effective_params(out_folder)
    he0_mask_upscale = int(p["he0_mask_upscale_factor"])
    he0_n_smooth = int(p["he0_mask_n_smooth"])
    he0_thr = float(p["he0_mask_intensity_threshold"])
    he_mask_upscale = int(p["he_mask_upscale_factor"])
    he_n_smooth = int(p["he_mask_n_smooth"])
    he_thr = float(p["he_mask_intensity_threshold"])
    print(
        "[INFO] Effective stage4a params: "
        f"he0_mask_n_smooth={he0_n_smooth}, he0_mask_intensity_threshold={he0_thr}, "
        f"he0_mask_upscale_factor={he0_mask_upscale}, "
        f"he_mask_n_smooth={he_n_smooth}, he_mask_intensity_threshold={he_thr}, "
        f"he_mask_upscale_factor={he_mask_upscale}",
        flush=True,
    )

    he0_images = glob(os.path.join(out_folder, "*_he0.png"))
    he_images = glob(os.path.join(out_folder, "*_he.png"))

    # ---------------- HE0 ----------------
    t_he0_start = time.perf_counter()
    print(f"[INFO] Starting HE0 processing: {len(he0_images)} tiles from {out_folder}", flush=True)
    n_fail_he0 = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_he0, upscale=he0_mask_upscale, n_smooth=he0_n_smooth, intensity_threshold=he0_thr)
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
    print(f"[INFO] Starting H&E processing: {len(he_images)} tiles", flush=True)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(process_he, upscale=he_mask_upscale, n_smooth=he_n_smooth, intensity_threshold=he_thr)
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
            "he0_mask_n_smooth": he0_n_smooth,
            "he0_intensity_threshold": float(he0_thr),
            "he_mask_n_smooth": he_n_smooth,
            "he_intensity_threshold": float(he_thr),
        }
    }

    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}")


if __name__ == "__main__":
    main()
