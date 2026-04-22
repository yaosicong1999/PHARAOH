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


# -----------------------------
# Utilities
# -----------------------------
def load_stage4a_params_from_parameters_json(
    out_folder: str,
    default_dapi_offset=0,
    default_dapi_min_area_factor=10e-5,
    default_dapi_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
):
    """
    Read ../../parameters.json and return stage4a params.
    Falls back to the provided defaults if the file or keys are missing.
    """
    path = os.path.join(out_folder, "../../parameters.json")
    print(f"[INFO] loading parameters from: {path}", flush=True)

    params = {
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

    stage4a = data.get("stage4a", {}) if isinstance(data, dict) else {}

    try:
        params["dapi_thr_offset"] = int(
            float(stage4a.get("dapi_thr_offset", params["dapi_thr_offset"]))
        )
    except Exception:
        pass

    try:
        params["dapi_mask_min_area_factor"] = float(
            stage4a.get("dapi_mask_min_area_factor", params["dapi_mask_min_area_factor"])
        )
    except Exception:
        pass

    try:
        params["dapi_mask_upscale_factor"] = int(
            stage4a.get("dapi_mask_upscale_factor", params["dapi_mask_upscale_factor"])
        )
    except Exception:
        pass

    try:
        params["he_mask_n_smooth"] = int(
            stage4a.get("he_mask_n_smooth", params["he_mask_n_smooth"])
        )
    except Exception:
        pass

    try:
        params["he_mask_intensity_threshold"] = float(
            stage4a.get("he_mask_intensity_threshold", params["he_mask_intensity_threshold"])
        )
    except Exception:
        pass

    try:
        params["he_mask_upscale_factor"] = int(
            stage4a.get("he_mask_upscale_factor", params["he_mask_upscale_factor"])
        )
    except Exception:
        pass

    return params

def load_stage4a_effective_params(
    out_folder: str,
    default_dapi_offset=0,
    default_dapi_min_area_factor=10e-5,
    default_dapi_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
):
    """
    Resolve Stage 4a parameters with this priority:
      1) parameters.json -> stage4a
      2) pilot_output_parameters.json overrides:
           - dapi_thr_offset
           - he_mask_intensity_threshold
      3) hard-coded defaults
    """
    params = load_stage4a_params_from_parameters_json(
        out_folder,
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

def run_parallel(file_list, worker_fn, stage_name, max_workers=8, print_fail=False):
    """
    Run a single-file worker function over a list of files in parallel and
    print progress in the format expected by the gallery launcher.
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, f) for f in file_list]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            msg = future.result()
            results.append(msg)
            done += 1
            if print_fail and str(msg).startswith("Failed:"):
                print(f"[{stage_name} FAIL] {msg}", flush=True)
            print(f"[PROGRESS] {stage_name} {done}/{total}", flush=True)
    return results

# -----------------------------
# process tiles
# -----------------------------
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

        # load grayscale tile
        img = cv2.imread(dapi_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed: {dapi_file}, Error: cv2.imread returned None"
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = upsample_tile(img, upscale)

        # 1) Otsu + optional offset
        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(int(otsu_thr) + int(THR_OFFSET), 0, 255))
        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        fg_ratio0 = float((mask > 0).mean())
        # 2) fill holes
        mask = fill_holes_binary(mask)
        fg_ratio1 = float((mask > 0).mean())
        # 3) remove tiny blobs
        MIN_AREA = min_area_factor * img.shape[0] ** 2
        MIN_AREA = MIN_AREA * (
                    upscale ** 2)  # Adjust area threshold after upsampling.
        mask, cc_info = remove_small_components(
            mask, min_area=int(MIN_AREA), connectivity=int(CONNECTIVITY)
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

        return f"Processed: {os.path.basename(dapi_file)} otsu={int(otsu_thr)} used={thr}{extra} saved: {mask_save_path}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"

def process_he(image_file, upscale=2, n_smooth=2, intensity_threshold=0.6):
    try:
        rgb_tile = np.array(Image.open(image_file))
        _, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=int(n_smooth),
            intensity_threshold=float(intensity_threshold),
        )
        mask_save_path = image_file.replace("_he.png", "_he_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)
        return f"Processed: {os.path.basename(image_file)}"
    except Exception as e:
        return f"Failed: {os.path.basename(image_file)}, Error: {e}"


def main():
    t0 = time.perf_counter()
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 5_generate_nuclei_masks.py <output_folder>")
    out_folder = sys.argv[1]
    print(f"[INFO] Using output folder: {out_folder}", flush=True)

    stage4a_params = load_stage4a_effective_params(
        out_folder,
        default_dapi_offset=0,
        default_dapi_min_area_factor=10e-5,
        default_dapi_upscale=2,
        default_he_n_smooth=2,
        default_he_thr=0.6,
        default_he_upscale=2,
    )

    print(
        "[INFO] Effective stage4a params: "
        f"dapi_thr_offset={stage4a_params['dapi_thr_offset']}, "
        f"dapi_mask_min_area_factor={stage4a_params['dapi_mask_min_area_factor']}, "
        f"dapi_mask_upscale_factor={stage4a_params['dapi_mask_upscale_factor']}, "
        f"he_mask_n_smooth={stage4a_params['he_mask_n_smooth']}, "
        f"he_mask_intensity_threshold={stage4a_params['he_mask_intensity_threshold']}, "
        f"he_mask_upscale_factor={stage4a_params['he_mask_upscale_factor']}",
        flush=True,
    )

    dapi_images = sorted(glob(os.path.join(out_folder, "*_dapi_u8.png")))
    he_images = sorted(glob(os.path.join(out_folder, "*_he.png")))

    # ---------------- DAPI ----------------
    t_dapi_start = time.perf_counter()
    print("[ETA_START]", flush=True)
    print(f"[INFO] Starting DAPI processing: {len(dapi_images)} tiles", flush=True)
    dapi_mask_upscale = int(stage4a_params["dapi_mask_upscale_factor"])
    dapi_min_area_factor = float(stage4a_params["dapi_mask_min_area_factor"])
    print(f"[INFO] Starting DAPI processing: {len(dapi_images)} tiles", flush=True)

    dapi_results = run_parallel(
        file_list=dapi_images,
        worker_fn=partial(
            process_dapi,
            upscale=dapi_mask_upscale,
            THR_OFFSET=stage4a_params["dapi_thr_offset"],
            min_area_factor=dapi_min_area_factor,
        ),
        stage_name="DAPI",
        max_workers=8,
        print_fail=True,
    )

    n_dapi_fail = sum(str(msg).startswith("Failed:") for msg in dapi_results)
    print(f"[INFO] DAPI done. failed={n_dapi_fail}/{len(dapi_images)}", flush=True)
    t_dapi_end = time.perf_counter()

    # ---------------- H&E ----------------
    t_he_start = time.perf_counter()
    print("[ETA_START]", flush=True)
    print(f"[INFO] Starting H&E processing: {len(he_images)} tiles", flush=True)
    he_mask_upscale = int(stage4a_params["he_mask_upscale_factor"])
    he_mask_n_smooth = int(stage4a_params["he_mask_n_smooth"])
    he_mask_intensity_threshold = float(stage4a_params["he_mask_intensity_threshold"])
    print(f"[INFO] Starting H&E processing: {len(he_images)} tiles", flush=True)

    he_results = run_parallel(
        file_list=he_images,
        worker_fn=partial(
            process_he,
            upscale=he_mask_upscale,
            n_smooth=he_mask_n_smooth,
            intensity_threshold=he_mask_intensity_threshold,
        ),
        stage_name="H&E",
        max_workers=8,
        print_fail=True,
    )
    n_he_fail = sum(str(msg).startswith("Failed:") for msg in he_results)
    print(f"[INFO] H&E done. failed={n_he_fail}/{len(he_images)}", flush=True)
    t_he_end = time.perf_counter()

    # ---------------- Summary ----------------
    t1 = time.perf_counter()
    print("\n================ Timing Summary ================")
    print(f"DAPI stage time : {t_dapi_end - t_dapi_start:.2f} s")
    print(f"H&E stage time  : {t_he_end - t_he_start:.2f} s")
    print(f"Total time      : {t1 - t0:.2f} s")
    print("================================================\n")
    print("[DONE] Nuclei masking finished", flush=True)

    mask_info = {
        "mask_scale": {
            "dapi": dapi_mask_upscale,
            "he": he_mask_upscale,
        },
        "effective_params": {
            "dapi_thr_offset": stage4a_params["dapi_thr_offset"],
            "dapi_mask_min_area_factor": stage4a_params["dapi_mask_min_area_factor"],
            "dapi_mask_upscale_factor": stage4a_params["dapi_mask_upscale_factor"],
            "he_mask_n_smooth": stage4a_params["he_mask_n_smooth"],
            "he_mask_intensity_threshold": stage4a_params["he_mask_intensity_threshold"],
            "he_mask_upscale_factor": stage4a_params["he_mask_upscale_factor"],
        },
    }
    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}", flush=True)

if __name__ == "__main__":
    main()