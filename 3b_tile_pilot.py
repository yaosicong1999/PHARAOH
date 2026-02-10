import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import subprocess

# =============================
# Params
# =============================
def load_initial_alignment(run_dir: Path):
    """
    Load DAPI->HE alignment matrix from RUN_DIR.

    Priority:
      1) clicked_blob_initial_alignment.json
      2) manual_initial_alignment.json

    Returns:
      H_mat as list (2x3 or 3x3)
    """
    path_clicked = run_dir / "clicked_blob_initial_alignment.json"
    path_manual  = run_dir / "manual_initial_alignment.json"

    if path_clicked.exists():
        data = json.load(open(path_clicked, "r"))
        print(f"[INFO] Using alignment from {path_clicked.name}", flush=True)
    elif path_manual.exists():
        data = json.load(open(path_manual, "r"))
        print(f"[INFO] Using alignment from {path_manual.name}", flush=True)
    else:
        raise FileNotFoundError(
            "No initial alignment found.\n"
            "Expected one of:\n"
            "  - clicked_blob_initial_alignment.json\n"
            "  - manual_initial_alignment.json"
        )

    if "H_mat" not in data:
        raise KeyError("Alignment json missing key 'H_mat'")

    return data["H_mat"]

def load_dapi_lut_threshold_from_images_info(run_dir: Path, default=1000) -> int:
    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        print(f"[WARN] missing {info_path}, fallback threshold={default}", flush=True)
        return int(default)

    try:
        info = json.load(open(info_path, "r"))
    except Exception as e:
        print(f"[WARN] failed to read images_info.json: {e}, fallback threshold={default}", flush=True)
        return int(default)

    # primary key you want
    if "DAPI_LUT_threshold" in info:
        try:
            return int(info["DAPI_LUT_threshold"])
        except Exception:
            pass

    # optional compatibility keys
    for k in ("dapi_lut_threshold", "lut_threshold", "DAPI_LUT_thr"):
        if k in info:
            try:
                return int(info[k])
            except Exception:
                pass

    print(f"[WARN] images_info.json has no DAPI LUT threshold key, fallback threshold={default}", flush=True)
    return int(default)

def load_step3_params(script_path: Path):
    script_dir = script_path.resolve().parent
    params_path = script_dir / "parameters.json"
    step3 = {}
    if params_path.exists():
        try:
            params = json.load(open(params_path, "r"))
            if isinstance(params, dict) and isinstance(params.get("step3", {}), dict):
                step3.update(params["step3"])
        except Exception as e:
            print(f"[WARN] failed to read parameters.json: {e}", flush=True)
    else:
        print(f"[INFO] parameters.json not found at {params_path}, using defaults", flush=True)

    # normalize
    step3["number_of_tiles"] = int(step3.get("n_tiles", 120))
    step3["tile_size"] = float(step3.get("tile_size", 600))
    step3["min_dist_factor"] = float(step3.get("min_dist_factor", 1.5))
    step3["pilot_k"] = int(step3.get("pilot_k", 10))
    return step3

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =============================
# Tile helpers (same contract as 3.py)
# =============================
def centroids_to_tiles(points_xy, tile_size):
    half = tile_size / 2.0
    tiles = []
    for (x, y) in points_xy:
        tiles.append({"x0": float(x - half), "y0": float(y - half), "w": float(tile_size), "h": float(tile_size)})
    return tiles

def _tile_to_xywh(p):
    if isinstance(p, dict):
        return float(p["x0"]), float(p["y0"]), float(p["w"]), float(p["h"])
    if isinstance(p, (list, tuple)) and len(p) == 4:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError(f"Unsupported tile format: {type(p)} {p}")

def normalize_uint16_to_uint8(img16: np.ndarray) -> np.ndarray:
    g = img16.astype(np.float32)
    mn, mx = float(np.min(g)), float(np.max(g))
    g = (g - mn) / (mx - mn + 1e-8)
    return (g * 255.0).astype(np.uint8)

def save_dapi_tiles_intensity(
    dapi_gray16,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="pilot",
    start_index=0,
    save_u16=False,
    save_u8_preview=True,
):
    if dapi_gray16.ndim == 3:
        dapi_gray16 = dapi_gray16[..., 0]
    if dapi_gray16.dtype != np.uint16:
        dapi_gray16 = dapi_gray16.astype(np.uint16)

    os.makedirs(output_folder, exist_ok=True)
    h_img, w_img = dapi_gray16.shape[:2]

    out_meta = {}
    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        x0 = int(round(x0f * rescale_factor))
        y0 = int(round(y0f * rescale_factor))
        w  = int(round(wf  * rescale_factor))
        h  = int(round(hf  * rescale_factor))

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w_img, x0 + w); y1 = min(h_img, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue

        tile16 = dapi_gray16[y0:y1, x0:x1]
        key = f"{prefix}_{i:03d}"

        fn_u16 = None
        fn_u8 = None
        if save_u16:
            fn_u16 = f"{key}_dapi_u16.png"
            cv2.imwrite(os.path.join(output_folder, fn_u16), tile16)
        if save_u8_preview:
            fn_u8 = f"{key}_dapi_u8.png"
            cv2.imwrite(os.path.join(output_folder, fn_u8), normalize_uint16_to_uint8(tile16))

        out_meta[key] = {
            "x0": x0, "y0": y0,
            "w": int(x1 - x0), "h": int(y1 - y0),
            "cx": float((x0 + x1) / 2), "cy": float((y0 + y1) / 2),
            "id": int(i),
            "filename_dapi_u16": fn_u16,
            "filename_dapi_u8": fn_u8,
        }

    with open(os.path.join(output_folder, "pilot_dapi_tile_info_intensity.json"), "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"[OK] Saved intensity DAPI pilot tiles: {len(out_meta)} -> {output_folder}")
    return out_meta

def save_dapi_tiles(
    dapi_rgb,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="pilot",
    start_index=0,
):
    os.makedirs(output_folder, exist_ok=True)
    h_img, w_img = dapi_rgb.shape[:2]

    out_meta = {}
    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        x0 = int(round(x0f * rescale_factor))
        y0 = int(round(y0f * rescale_factor))
        w  = int(round(wf  * rescale_factor))
        h  = int(round(hf  * rescale_factor))

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w_img, x0 + w); y1 = min(h_img, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue

        tile = dapi_rgb[y0:y1, x0:x1]
        key = f"{prefix}_{i:03d}"
        fn = f"{key}_dapi.png"
        cv2.imwrite(os.path.join(output_folder, fn), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        out_meta[key] = {
            "x0": x0, "y0": y0, "w": int(x1-x0), "h": int(y1-y0),
            "cx": float((x0+x1)/2), "cy": float((y0+y1)/2),
            "id": int(i),
            "filename": fn,
        }

    with open(os.path.join(output_folder, "pilot_dapi_tile_info_lut.json"), "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"[OK] Saved LUT DAPI pilot tiles: {len(out_meta)} -> {output_folder}")
    return out_meta

def save_he_tiles(
    he_rgb,
    tiles,
    h_mat,
    output_folder,
    rescale_factor=1.0,
    margin_ratio=0.1,
    prefix="pilot",
    start_index=0,
    debug_first_n=0,
    mode="rectified",                 # "rectified" or "bbox"
    rectify_interp=cv2.INTER_LINEAR,
    case_id=0,                        # must match DAPI orientation case
):
    """
    Save HE tiles by mapping DAPI tile corners -> HE via homography/affine.

    - tiles are defined in DAPI coordinate system (the same system as h_mat expects)
    - h_mat maps DAPI -> HE (3x3 homography or 2x3 affine)
    - rescale_factor converts HE coords -> he_rgb pixel coords (e.g. 2**(HE_LEVEL-1))

    mode:
      - "bbox": axis-aligned bbox crop of the projected quad
      - "rectified": warpPerspective to rectify the quad to a rectangle

    Output:
      - <prefix>_<id>_he.png
      - he_tile_info.json with meta corners + warp matrices (if rectified)
    """
    if mode not in ("rectified", "bbox"):
        raise ValueError(f"mode must be 'rectified' or 'bbox', got {mode}")
    os.makedirs(output_folder, exist_ok=True)

    H = np.asarray(h_mat, dtype=float)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"h_mat must be 3x3 homography (or 2x3 affine), got {H.shape}")

    he_rgb = np.asarray(he_rgb)
    h_img, w_img = he_rgb.shape[:2]
    rf = float(rescale_factor)

    def signed_area(q):
        q = np.asarray(q, dtype=np.float32)
        x = q[:, 0]
        y = q[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def orient_quad_indices(case_id: int):
        """
        Return indices to reorder [TL,TR,BR,BL] into the orientation-applied order.
        This matches your orientation convention used on DAPI tiles.
        """
        pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=int)  # TL,TR,BR,BL

        def apply_case_xy(x, y):
            if case_id == 0:   # identity
                return x, y
            if case_id == 1:   # rot90 CW
                return y, 1 - x
            if case_id == 2:   # rot180
                return 1 - x, 1 - y
            if case_id == 3:   # rot90 CCW
                return 1 - y, x
            if case_id == 4:   # flip UD
                return x, 1 - y
            if case_id == 5:   # flip LR
                return 1 - x, y
            if case_id == 6:   # transpose
                return y, x
            if case_id == 7:   # anti-transpose
                return 1 - y, 1 - x
            raise ValueError(f"Unknown case_id={case_id}")

        pts2 = np.array([apply_case_xy(x, y) for x, y in pts], dtype=int)

        # find TL/TR/BR/BL in the oriented 2x2 grid
        s = pts2[:, 0] + pts2[:, 1]
        d = pts2[:, 0] - pts2[:, 1]
        tl = int(np.argmin(s))
        br = int(np.argmax(s))
        tr = int(np.argmax(d))
        bl = int(np.argmin(d))
        return np.array([tl, tr, br, bl], dtype=int)

    he_tiles = []
    output_dict = {}

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        # ---- add margin in DAPI coords ----
        mw = float(wf) * (1.0 + float(margin_ratio))
        mh = float(hf) * (1.0 + float(margin_ratio))
        x0c = float(x0f) - (mw - float(wf)) / 2.0
        y0c = float(y0f) - (mh - float(hf)) / 2.0
        x1c = x0c + mw
        y1c = y0c + mh

        corners_dapi = np.array(
            [[x0c, y0c],
             [x1c, y0c],
             [x1c, y1c],
             [x0c, y1c]], dtype=float
        )  # TL,TR,BR,BL in DAPI coords

        # ---- project to HE coords ----
        corners_h = np.hstack([corners_dapi, np.ones((4, 1), dtype=float)])  # (4,3)
        proj = (H @ corners_h.T).T  # (4,3)
        w = proj[:, 2:3]
        eps = 1e-9
        w_safe = np.where(np.abs(w) < eps, np.sign(w) * eps + (w == 0) * eps, w)
        corners_he = proj[:, :2] / w_safe  # (4,2) in HE coord system

        # ---- to he_rgb pixel coords ----
        corners_he_px_raw = corners_he * rf  # (4,2) in he_rgb pixel coords

        # ---- bbox ----
        xs = corners_he_px_raw[:, 0]
        ys = corners_he_px_raw[:, 1]
        min_x = int(np.floor(xs.min()))
        max_x = int(np.ceil(xs.max()))
        min_y = int(np.floor(ys.min()))
        max_y = int(np.ceil(ys.max()))

        min_x_cl = max(0, min_x)
        min_y_cl = max(0, min_y)
        max_x_cl = min(w_img, max_x)
        max_y_cl = min(h_img, max_y)

        key = f"{prefix}_{i:03d}"

        if debug_first_n and (i - start_index) < debug_first_n:
            print(f"[DEBUG] {key} mode={mode}")
            print(" corners_dapi:\n", corners_dapi)
            print(" corners_he (pre-rescale):\n", corners_he)
            print(" corners_he_px_raw (he_rgb px):\n", corners_he_px_raw)
            print(" bbox unclamped:", (min_x, min_y, max_x-min_x, max_y-min_y))
            print(" bbox clamped  :", (min_x_cl, min_y_cl, max_x_cl-min_x_cl, max_y_cl-min_y_cl))
            print(" he_rgb shape  :", he_rgb.shape)

        if max_x_cl <= min_x_cl or max_y_cl <= min_y_cl:
            continue

        M = None
        Minv = None

        if mode == "bbox":
            tile_img = he_rgb[min_y_cl:max_y_cl, min_x_cl:max_x_cl]
            out_w = int(max_x_cl - min_x_cl)
            out_h = int(max_y_cl - min_y_cl)
        else:
            # src quad in full image coords
            src = corners_he_px_raw.astype(np.float32)

            def dist(a, b):
                return float(np.linalg.norm(a - b))

            width  = 0.5 * (dist(src[0], src[1]) + dist(src[3], src[2]))
            height = 0.5 * (dist(src[1], src[2]) + dist(src[0], src[3]))
            out_w = max(2, int(round(width)))
            out_h = max(2, int(round(height)))

            # canonical dst (TL,TR,BR,BL)
            dst = np.array(
                [[0.0, 0.0],
                 [out_w - 1.0, 0.0],
                 [out_w - 1.0, out_h - 1.0],
                 [0.0, out_h - 1.0]],
                dtype=np.float32
            )

            # apply same orientation as DAPI tile so visual “direction” matches
            idx_ord = orient_quad_indices(int(case_id))
            dst = dst[idx_ord]

            # fix mirror if winding mismatch
            if signed_area(src) * signed_area(dst) < 0:
                src = src[[0, 3, 2, 1]]

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = np.linalg.inv(M)

            tile_img = cv2.warpPerspective(
                he_rgb, M, (out_w, out_h),
                flags=rectify_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        filename = f"{key}_he.png"
        cv2.imwrite(
            os.path.join(output_folder, filename),
            cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
            if tile_img.ndim == 3 and tile_img.shape[2] == 3
            else tile_img
        )

        info = {
            "x0": int(min_x_cl),
            "y0": int(min_y_cl),
            "w": int(max_x_cl - min_x_cl),
            "h": int(max_y_cl - min_y_cl),
            "cx": float((min_x_cl + max_x_cl) / 2.0),
            "cy": float((min_y_cl + max_y_cl) / 2.0),
            "type": "pilot",
            "id": int(i),
            "filename": filename,
        }
        meta = {
            "mode": mode,
            "rescale_factor": float(rescale_factor),
            "margin_ratio": float(margin_ratio),
            "case_id": int(case_id),
            "dapi_corners": corners_dapi.tolist(),
            "he_quad_px_raw": corners_he_px_raw.tolist(),  # TL,TR,BR,BL (DAPI order) in he_rgb coords
            "rectified_wh": [int(out_w), int(out_h)],
            "M_he_to_rect": None if M is None else M.tolist(),
            "M_rect_to_he": None if Minv is None else Minv.tolist(),
        }

        output_dict[key] = {**info, "meta": meta}
        he_tiles.append({**info, "meta": meta})

    print(f"[OK] Saved H&E tiles (mode={mode}): {len(he_tiles)} -> {output_folder}", flush=True)

    with open(os.path.join(output_folder, "pilot_he_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=2)

    return he_tiles


# =============================
# Farthest point sampling (cover most area)
# =============================
def farthest_point_subset(points_xy: np.ndarray, k: int, start="centroid"):
    """
    Greedy FPS: iteratively pick point that maximizes distance to selected set.
    O(N^2) but N~120 so fine.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    n = len(pts)
    if k >= n:
        return np.arange(n, dtype=int)

    if start == "random":
        s0 = np.random.randint(n)
    else:
        c = pts.mean(axis=0)
        s0 = int(np.argmin(np.sum((pts - c) ** 2, axis=1)))

    selected = [s0]
    # min distance to selected set for each point
    dmin = np.linalg.norm(pts - pts[s0], axis=1)

    for _ in range(1, k):
        # pick the point farthest from current selected set (max dmin)
        idx = int(np.argmax(dmin))
        selected.append(idx)
        # update dmin
        dnew = np.linalg.norm(pts - pts[idx], axis=1)
        dmin = np.minimum(dmin, dnew)

    return np.array(selected, dtype=int)

# =============================
# Main
# =============================
def main():
    if len(sys.argv) < 2:
        print("Usage: python 3b.py <RUN_DIR>")
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] RUN_DIR not found: {run_dir}")
        sys.exit(2)

    step3 = load_step3_params(Path(__file__))
    pilot_k = int(step3.get("pilot_k", 10))

    info_path = run_dir / "images_info.json"
    sampled_path = run_dir / "sampled_points.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"missing {sampled_path} (run Step3 sampling first)")

    info = json.load(open(info_path, "r"))
    sampled = json.load(open(sampled_path, "r"))

    DAPI_PATH = info["DAPI_path"]
    DAPI_LEVEL = int(info["DAPI_level"])
    points_xy = np.asarray(sampled["points_xy"], dtype=np.int32)

    # tile size conversion: parameters.json tile_size is in level=1 coords
    tile_size_lvl1 = float(step3["tile_size"])
    TILE_SIZE = tile_size_lvl1 / (2 ** (DAPI_LEVEL - 1))  # in DAPI_LEVEL pixel space

    # pick k farthest points
    sel_idx = farthest_point_subset(points_xy, pilot_k, start="centroid")
    sel_pts = points_xy[sel_idx]

    pilot_dir = run_dir / "pilot_tiles"
    ensure_dir(pilot_dir)

    # save selection record
    with open(pilot_dir / "pilot_selection.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "dapi_level": DAPI_LEVEL,
                "pilot_k": int(pilot_k),
                "tile_size_lvl1": tile_size_lvl1,
                "tile_size_in_dapi_level": float(TILE_SIZE),
                "selected_indices": sel_idx.tolist(),
                "selected_points_xy": sel_pts.tolist(),
            },
            f,
            indent=2,
        )
    print(f"[OK] selected {len(sel_idx)}/{len(points_xy)} pilot points -> {pilot_dir/'pilot_selection.json'}")

    # build tiles in DAPI_LEVEL coords
    tiles = centroids_to_tiles(sel_pts, tile_size=TILE_SIZE)

    # read DAPI (level=1) for extraction (same as your 3.py extract)
    from my_utils import read_image, dapi_to_lut_rgb

    # intensity
    dapi16_lvl1, _ = read_image(DAPI_PATH, keep_16bit=True, level=1)
    if dapi16_lvl1.ndim == 3:
        dapi16_lvl1 = dapi16_lvl1[..., 0]

    # LUT
    lut_path = Path(__file__).resolve().parent / "glasbey_inverted.lut"
    lut = np.fromfile(str(lut_path), dtype=np.uint8).reshape(256, 3)
    dapi_lut_threshold = load_dapi_lut_threshold_from_images_info(run_dir,default=step3.get("dapi_lut_threshold", 1000))
    print(f"[INFO] DAPI_LUT_threshold (from images_info.json) = {dapi_lut_threshold}", flush=True)
    dapi_rgb = dapi_to_lut_rgb(dapi16_lvl1, lut, threshold=int(dapi_lut_threshold))

    # rescale from DAPI_LEVEL tile coords -> level=1 pixel coords
    rescale_factor = 2 ** (DAPI_LEVEL - 1)

    save_dapi_tiles_intensity(
        dapi16_lvl1,
        tiles,
        str(pilot_dir),
        rescale_factor=rescale_factor,
        prefix="pilot",
        start_index=0,
        save_u16=False,
        save_u8_preview=True,
    )

    save_dapi_tiles(
        dapi_rgb,
        tiles,
        str(pilot_dir),
        rescale_factor=rescale_factor,
        prefix="pilot",
        start_index=0,
    )

    print("[DONE] pilot extraction finished.")

    # -----------------------------
    # Extract pilot HE tiles
    # -----------------------------
    HE_PATH = info["HE_path"]
    HE_LEVEL = int(info["HE_level"])

    h_mat = load_initial_alignment(run_dir)

    # read HE at level=1 (same style as step3)
    he_lvl1, _ = read_image(HE_PATH, keep_16bit=True, level=1)

    # rescale_factor maps HE_LEVEL coords -> level=1 pixels
    he_rescale = 2 ** (HE_LEVEL - 1)

    # IMPORTANT: tiles are in DAPI_LEVEL coords, h_mat is DAPI->HE, so pass same tiles
    save_he_tiles(
        he_lvl1,
        tiles,
        h_mat,
        str(pilot_dir),
        rescale_factor=he_rescale,
        mode="rectified",          # or "bbox"
        case_id=int(info.get("DAPI_orientation_case", 0)),
        prefix="pilot",
        start_index=0,
    )

    # -----------------------------
    # Launch 3c.py automatically
    # -----------------------------
    script_dir = Path(__file__).resolve().parent
    script_3c = script_dir / "3c_pilot_tile_gallery.py"

    if not script_3c.exists():
        print(f"[WARN] 3c script not found: {script_3c}", flush=True)
    else:
        py = sys.executable  # use same env python
        cmd = [py, str(script_3c), str(run_dir), str(pilot_dir)]
        print("[INFO] launching 3c:", " ".join(cmd), flush=True)

        # Non-blocking: open new process and return immediately
        subprocess.Popen(cmd, cwd=str(script_dir))

if __name__ == "__main__":
    main()