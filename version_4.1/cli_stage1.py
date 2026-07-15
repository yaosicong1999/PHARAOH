# ============================================================
# cli_stage1.py - Stage 1 (headless): read images + build masks
#
# Replaces the interactive Tk selection window of
#   engine_dapi/1_read_dapi_he.py  /  engine_he/1_read_he0_he.py
# with a non-interactive run that uses the same compute helpers and
# the GUI's *default* thresholds (HE=240, DAPI LUT=300) and an
# identity orientation (case 0), then writes the exact same artifacts.
#
# Usage:
#   python cli_stage1.py --engine <engine_dir> --run-dir <run_dir>
#       --mode {dapi,he} --he <he_path> --moving <dapi_or_he0_path>
#       [--stage1-file <name>] [--he-threshold 240] [--moving-threshold N]
# ============================================================
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cli_common import load_module_from_path, save_json, log_stage_event, banner

Image.MAX_IMAGE_PIXELS = None

HE_DEFAULT_THRESHOLD = 240      # matches he_val = tk.IntVar(value=240)
DAPI_LUT_DEFAULT_THRESHOLD = 300  # matches dapi_val = tk.IntVar(value=300)
BLOB_MIN_AREA = 2000


def count_components(mask_u8, min_area=BLOB_MIN_AREA):
    if mask_u8 is None:
        return 0
    m = (mask_u8 > 0).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.sum(areas >= min_area))


def run_dapi_mode(engine_dir, run_dir, s1, he_path, dapi_path, he_thr, dapi_thr):
    # my_utils is importable because we chdir'd into the engine dir
    from my_utils import (
        read_image, extract_hematoxylin_channel,
        enhance_hematoxylin_channel, dapi_to_lut_rgb,
    )

    print("[Stage1-CLI] reading H&E ...", flush=True)
    he_orig, he_level = read_image(he_path, channel="he")
    he_h = extract_hematoxylin_channel(he_orig)
    he_h_proc = enhance_hematoxylin_channel(he_h)
    _, he_mask = cv2.threshold(he_h_proc, int(he_thr), 255, cv2.THRESH_BINARY)
    he_mask = he_mask.astype(np.uint8)
    he_dense = s1.create_blob_mask_from_dot_mask(he_mask)

    print("[Stage1-CLI] reading DAPI ...", flush=True)
    dapi_img, dapi_level = read_image(dapi_path, keep_16bit=True, force_rgb=False, channel="dapi")
    dapi_lut = dapi_to_lut_rgb(dapi_img, s1.lut, threshold=int(dapi_thr))
    dapi_mask = s1.create_blob_mask_from_luted_dapi(dapi_lut)

    # ---- save artifacts (identical filenames to the GUI) ----
    Image.fromarray(he_orig).save(run_dir / "1_he_level_image.png")
    cv2.imwrite(str(run_dir / "1_dapi_lut.png"), dapi_lut)
    cv2.imwrite(str(run_dir / "1_confirmed_he_dense_mask.png"), he_dense)
    cv2.imwrite(str(run_dir / "1_confirmed_dapi_mask.png"), dapi_mask)

    info = {
        "RUN_ID": run_dir.name.replace("runs_", "", 1),
        "HE_path": str(he_path),
        "HE_level": int(he_level),
        "DAPI_path": str(dapi_path),
        "DAPI_level": int(dapi_level),
        "DAPI_gui_affine": np.eye(3, dtype=np.float32).tolist(),
        "DAPI_orientation_case": 0,
        "HE_threshold": int(he_thr),
        "DAPI_LUT_threshold": int(dapi_thr),
        "blob_count_min_area": BLOB_MIN_AREA,
        "HE_blob_count": count_components(he_dense),
        "DAPI_blob_count": count_components(dapi_mask),
    }
    save_json(run_dir / "images_info.json", info)
    return info


def run_he_mode(engine_dir, run_dir, s1, he_path, he0_path, he_thr, he0_thr):
    from my_utils import (
        read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel,
    )

    def he_like(path, thr):
        img, level = read_image(path, channel="he")
        h = extract_hematoxylin_channel(img)
        h_proc = enhance_hematoxylin_channel(h)
        _, mask = cv2.threshold(h_proc, int(thr), 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        dense = s1.create_blob_mask_from_dot_mask(mask)
        return img, level, mask, dense

    print("[Stage1-CLI] reading H&E (fixed) ...", flush=True)
    he_orig, he_level, he_mask, he_dense = he_like(he_path, he_thr)
    print("[Stage1-CLI] reading H&E0 (moving) ...", flush=True)
    he0_orig, he0_level, he0_mask, he0_dense = he_like(he0_path, he0_thr)

    Image.fromarray(he_orig).save(run_dir / "1_he_level_image.png")
    Image.fromarray(he0_orig).save(run_dir / "1_he0_level_image.png")
    cv2.imwrite(str(run_dir / "1_he_threshold_mask.png"), he_mask)
    cv2.imwrite(str(run_dir / "1_he0_threshold_mask.png"), he0_mask)
    cv2.imwrite(str(run_dir / "1_confirmed_he_dense_mask.png"), he_dense)
    cv2.imwrite(str(run_dir / "1_confirmed_he0_dense_mask.png"), he0_dense)

    info = {
        "RUN_ID": run_dir.name.replace("runs_", "", 1),
        "HE_path": str(he_path),
        "HE_level": int(he_level),
        "HE0_path": str(he0_path),
        "HE0_level": int(he0_level),
        "HE0_gui_affine": np.eye(3, dtype=np.float32).tolist(),
        "HE0_orientation_case": 0,
        "HE_threshold": int(he_thr),
        "HE0_threshold": int(he0_thr),
        "blob_count_min_area": BLOB_MIN_AREA,
        "HE_blob_count": count_components(he_dense),
        "HE0_blob_count": count_components(he0_dense),
    }
    save_json(run_dir / "images_info.json", info)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["dapi", "he"])
    ap.add_argument("--he", required=True, help="fixed H&E image path")
    ap.add_argument("--moving", required=True, help="moving image path (DAPI or HE0)")
    ap.add_argument("--stage1-file", default=None,
                    help="engine stage1 filename (default per mode)")
    ap.add_argument("--he-threshold", type=int, default=HE_DEFAULT_THRESHOLD)
    ap.add_argument("--moving-threshold", type=int, default=None)
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run "inside" the engine dir so `import my_utils` and the relative
    # LUT path used by the stage1 module both resolve.
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    stage1_file = args.stage1_file or (
        "1_read_dapi_he.py" if args.mode == "dapi" else "1_read_he0_he.py"
    )
    s1 = load_module_from_path("engine_stage1", engine_dir / stage1_file)

    banner(f"STAGE 1 (headless, mode={args.mode}) -> {run_dir}")
    log_stage_event(run_dir, "stage1_events", "cli_stage1_start", mode=args.mode)

    if args.mode == "dapi":
        moving_thr = args.moving_threshold if args.moving_threshold is not None else DAPI_LUT_DEFAULT_THRESHOLD
        info = run_dapi_mode(engine_dir, run_dir, s1, args.he, args.moving,
                             args.he_threshold, moving_thr)
    else:
        moving_thr = args.moving_threshold if args.moving_threshold is not None else HE_DEFAULT_THRESHOLD
        info = run_he_mode(engine_dir, run_dir, s1, args.he, args.moving,
                           args.he_threshold, moving_thr)

    log_stage_event(run_dir, "stage1_events", "cli_stage1_done", **{
        k: info[k] for k in info if k.endswith("_blob_count")
    })
    print(f"[Stage1-CLI] wrote images_info.json  (blobs: "
          f"{[ (k, info[k]) for k in info if k.endswith('_blob_count') ]})", flush=True)


if __name__ == "__main__":
    main()
