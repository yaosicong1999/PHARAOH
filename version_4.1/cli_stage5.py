# ============================================================
# cli_stage5.py - Stage 5 (headless): final alignment from nuclei pairs
#
# The engine's alignment math lives in a nested closure inside the Tk
# gallery (calculate_alignment_transform), so it cannot be imported.
# This is a faithful headless transcription of that compute for the
# configured transform modes (homography / affine), reading the same
# nuclei_patches/nuclei_centroids_global.json and writing the same
# <moving>_to_he_homography_level0.json (+ .csv).
#
# transform_type = "tps"/"local_tps" is not reproduced here (both
# engines' parameters.json are configured for "homography"); set the
# mode to homography/affine to use the CLI.
#
# Usage:
#   python cli_stage5.py --engine <dir> --run-dir <dir> --mode {dapi,he}
# ============================================================
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from cli_common import load_json, save_json, log_stage_event, banner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["dapi", "he"])
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    moving = "dapi" if args.mode == "dapi" else "he0"
    src_key = f"{moving}_centroid_global"
    dst_key = "he_centroid_global"

    banner(f"STAGE 5 (headless alignment, mode={args.mode})")
    log_stage_event(run_dir, "stage5_events", "cli_stage5_start", mode=args.mode)

    # ---- params: read directly from the integrated version_4.0_cli/parameters.json ----
    block = "dapi" if args.mode == "dapi" else "he0fg"
    params = load_json(engine_dir.parent / "parameters.json").get(block, {}).get("stage5", {})
    transform_type = str(params.get("transform_mode", "homography")).lower().strip()
    balance_points_bool = bool(params.get("balance_points_bool", False))

    if transform_type in ("tps", "local_tps"):
        raise NotImplementedError(
            f"transform_mode='{transform_type}' is not supported by the CLI; "
            f"set parameters.json['stage5']['transform_mode'] to 'homography' or 'affine'."
        )
    if transform_type not in ("homography", "affine"):
        raise ValueError(f"invalid transform_mode: {transform_type}")

    # ---- load nuclei centroid pairs ----
    info_path = run_dir / "nuclei_patches" / "nuclei_centroids_global.json"
    data = load_json(info_path)
    if isinstance(data, dict) and "data" in data:
        nuclei_info = data["data"]
        nuclei_meta = data.get("meta", {})
    else:
        nuclei_info = data
        nuclei_meta = {}

    n = len(nuclei_info)
    min_needed = 3 if transform_type == "affine" else 4
    if n < min_needed:
        raise RuntimeError(f"need >= {min_needed} nucleus pairs, got {n}")

    src_all = np.array([x[src_key] for x in nuclei_info], dtype=np.float32)
    dst_all = np.array([x[dst_key] for x in nuclei_info], dtype=np.float32)

    if balance_points_bool:
        print("[Stage5-CLI][WARN] balance_points_bool=true is not reproduced in the "
              "CLI; using all points.", flush=True)
    keep_idx = np.arange(len(src_all), dtype=int)
    src = src_all[keep_idx]
    dst = dst_all[keep_idx]

    ransac_thr, maxIters, confidence = 8.0, 10000, 0.995

    if transform_type == "affine":
        A, inliers = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr,
            maxIters=maxIters, confidence=confidence, refineIters=10,
        )
        if A is None:
            raise RuntimeError("affine RANSAC failed (estimateAffine2D returned None)")
        H = np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64)
        method_str = "cv2.estimateAffine2D(RANSAC) -> 3x3"
    else:
        H, inliers = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr,
            maxIters=maxIters, confidence=confidence,
        )
        if H is None:
            raise RuntimeError("homography RANSAC failed (findHomography returned None)")
        method_str = "cv2.findHomography(RANSAC)"

    # reprojection error
    src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
    proj = (H.astype(np.float64) @ src_h.T).T
    proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
    err = np.linalg.norm(proj_xy - dst, axis=1)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if inliers is not None:
        mask = inliers.ravel().astype(bool)
        err_in = err[mask]
        med_err = float(np.median(err_in)) if len(err_in) else float("nan")
        mean_err = float(np.mean(err_in)) if len(err_in) else float("nan")
    else:
        med_err = float(np.median(err))
        mean_err = float(np.mean(err))

    out = {
        "from": f"{moving}_level0",
        "to": "he_level0",
        "method": method_str,
        "transform_type": transform_type,
        "sampling": {
            "type": "all_points",
            "balance_points_bool": bool(balance_points_bool),
            "num_points_before_sampling": int(len(src_all)),
            "num_points_after_sampling": int(len(src)),
            "selected_indices_from_original": keep_idx.tolist(),
        },
        "ransacReprojThreshold": float(ransac_thr),
        "maxIters": int(maxIters),
        "confidence": float(confidence),
        "num_points": int(len(src)),
        "num_inliers": int(inlier_count),
        "inlier_median_reproj_error_px": med_err,
        "inlier_mean_reproj_error_px": mean_err,
        "homography_3x3": H.tolist(),
    }

    out_path = run_dir / f"{moving}_to_he_homography_level0.json"
    save_json(out_path, out)
    np.savetxt(run_dir / f"{moving}_to_he_homography_level0.csv",
               np.array(H, dtype=np.float64), delimiter=",", fmt="%.10f")
    save_json(run_dir / "nuclei_centroids_global_used_for_stage5.json",
              {"meta": nuclei_meta, "manual_override_applied": False, "data": nuclei_info})

    print(f"[Stage5-CLI] {method_str}: {inlier_count}/{len(src)} inliers, "
          f"median reproj err {med_err:.3f}px -> {out_path.name}", flush=True)
    log_stage_event(run_dir, "stage5_events", "cli_stage5_done",
                    num_inliers=int(inlier_count), num_points=int(len(src)))


if __name__ == "__main__":
    main()
