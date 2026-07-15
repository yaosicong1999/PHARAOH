# ============================================================
# cli_stage2_orb.py - Stage 2 (headless): automatic ORB alignment
#
# Replaces the interactive manual-alignment GUI. Runs ORB feature
# registration moving -> H&E on the same low-level images the GUI
# loads, and writes manual_initial_alignment.json in the exact schema
# the GUI's "Save" produces (keys H_mat, H_mat_level_0), plus an
# overlay preview.
#
# Because the CLI stage-1 uses identity orientation (DAPI/HE0 gui_affine
# = I), the ORB homography (moving-level pixels -> HE-level pixels) IS
# H_mat directly -- no gui-affine subtraction is needed.
#
# NOTE: moving (DAPI fluorescence or a 2nd H&E) and the fixed H&E can be
# cross-modal; ORB may misalign. That is accepted for this CLI variant.
#
# Usage:
#   python cli_stage2_orb.py --engine <dir> --run-dir <dir> --mode {dapi,he}
# ============================================================
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from cli_common import load_json, save_json, log_stage_event, banner


def _prep(img_bgr, invert=False):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if invert:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def orb_homography(src_path, dst_path, invert_src, max_dim=1600, n_features=8000,
                   ratio=0.75, ransac_thresh=5.0):
    """Return (H_3x3 mapping src-image px -> dst-image px, n_inliers)."""
    src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    dst = cv2.imread(str(dst_path), cv2.IMREAD_COLOR)
    if src is None or dst is None:
        raise FileNotFoundError(f"could not read {src_path} or {dst_path}")

    def scale_for(img):
        h, w = img.shape[:2]
        return min(1.0, max_dim / float(max(h, w)))

    ss, sd = scale_for(src), scale_for(dst)
    src_s = cv2.resize(src, None, fx=ss, fy=ss, interpolation=cv2.INTER_AREA) if ss < 1 else src
    dst_s = cv2.resize(dst, None, fx=sd, fy=sd, interpolation=cv2.INTER_AREA) if sd < 1 else dst

    g_src = _prep(src_s, invert=invert_src)
    g_dst = _prep(dst_s, invert=False)

    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(g_src, None)
    k2, d2 = orb.detectAndCompute(g_dst, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise RuntimeError("too few ORB features detected")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"only {len(good)} good ORB matches (need >=4)")

    pts_src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_s, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    if H_s is None:
        raise RuntimeError("findHomography returned None")
    n_inl = int(mask.sum()) if mask is not None else 0

    # lift scaled-space homography back to full-file pixels
    S_src = np.diag([ss, ss, 1.0])
    S_dst_inv = np.diag([1.0 / sd, 1.0 / sd, 1.0])
    H_full = S_dst_inv @ H_s.astype(np.float64) @ S_src
    if abs(H_full[2, 2]) > 1e-12:
        H_full = H_full / H_full[2, 2]
    return H_full, n_inl


def save_overlay(src_path, dst_path, H, out_path, invert_src):
    bg = cv2.imread(str(dst_path), cv2.IMREAD_COLOR)
    fg = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bg is None or fg is None:
        return
    warped = cv2.warpPerspective(fg, H.astype(np.float64), (bg.shape[1], bg.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = cv2.addWeighted(bg, 0.7, warped, 0.8, 0)
    cv2.imwrite(str(out_path), out)


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

    info = load_json(run_dir / "images_info.json")

    if args.mode == "dapi":
        src_path = run_dir / "1_dapi_lut.png"
        moving_level = int(info["DAPI_level"])
        invert_src = True   # LUT'd DAPI is bright; invert to match H&E polarity
    else:
        src_path = run_dir / "1_he0_level_image.png"
        moving_level = int(info["HE0_level"])
        invert_src = False  # both brightfield H&E, same polarity
    dst_path = run_dir / "1_he_level_image.png"
    he_level = int(info["HE_level"])

    banner(f"STAGE 2 (headless ORB, mode={args.mode})")
    log_stage_event(run_dir, "stage2_events", "cli_stage2_orb_start", mode=args.mode)

    H_mat, n_inl = orb_homography(src_path, dst_path, invert_src=invert_src)
    print(f"[Stage2-CLI] ORB homography: {n_inl} inliers", flush=True)

    sd = float(2 ** moving_level)
    sh = float(2 ** he_level)
    S_d = np.diag([sd, sd, 1.0])
    S_h = np.diag([sh, sh, 1.0])
    H_mat_level_0 = (S_h @ H_mat @ np.linalg.inv(S_d))

    data = {
        "active_mode": "original",
        "mask_pose": None,
        "original_pose": None,
        "auto_alignment": {"method": "ORB", "n_inliers": int(n_inl)},
        "H_mat": H_mat.tolist(),
        "H_mat_level_0": H_mat_level_0.tolist(),
    }
    save_json(run_dir / "manual_initial_alignment.json", data)
    print("[Stage2-CLI] wrote manual_initial_alignment.json", flush=True)

    save_overlay(src_path, dst_path, H_mat,
                 run_dir / "2_manual_overlay_original.png", invert_src)

    log_stage_event(run_dir, "stage2_events", "cli_stage2_orb_done", inliers=int(n_inl))


if __name__ == "__main__":
    main()
