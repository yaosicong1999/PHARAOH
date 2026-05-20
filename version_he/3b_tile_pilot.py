import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import subprocess


# =============================
# Params / IO
# =============================
def load_initial_alignment(run_dir: Path):
    """
    Load HE0->HE alignment matrix from RUN_DIR.

    Priority:
      1) clicked_blob_initial_alignment.json
      2) manual_initial_alignment.json

    Returns:
      H_mat as list (2x3 or 3x3)
    """
    path_clicked = run_dir / "clicked_blob_initial_alignment.json"
    path_manual = run_dir / "manual_initial_alignment.json"

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

    step3["n_tiles"] = int(step3.get("n_tiles", 120))
    step3["tile_size"] = float(step3.get("tile_size", 600))
    step3["min_dist_factor"] = float(step3.get("min_dist_factor", 1.5))
    step3["pilot_k"] = int(step3.get("pilot_k", 10))
    step3["he_tile_margin_ratio"] = float(step3.get("he_tile_margin_ratio", 0.1))
    return step3


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def prune_unmatched_he0_pilot_tiles(output_folder: str):
    """
    Keep only pilot tile IDs that exist in BOTH
      - pilot_he0_tile_info.json
      - pilot_he_tile_info.json

    Remove unmatched HE0 pilot tile files and rewrite both jsons.
    """
    he0_json = os.path.join(output_folder, "pilot_he0_tile_info.json")
    he_json = os.path.join(output_folder, "pilot_he_tile_info.json")

    if not os.path.exists(he0_json):
        raise FileNotFoundError(f"missing {he0_json}")
    if not os.path.exists(he_json):
        raise FileNotFoundError(f"missing {he_json}")

    with open(he0_json, "r") as f:
        he0_info = json.load(f)
    with open(he_json, "r") as f:
        he_info = json.load(f)

    he0_keys = set(he0_info.keys())
    he_keys = set(he_info.keys())

    keep_keys = sorted(he0_keys & he_keys)
    drop_keys = sorted(he0_keys - he_keys)

    print(f"[INFO] pilot he0 tiles total: {len(he0_keys)}", flush=True)
    print(f"[INFO] pilot he  tiles total: {len(he_keys)}", flush=True)
    print(f"[INFO] pilot keeping matched tiles: {len(keep_keys)}", flush=True)
    print(f"[INFO] pilot dropping unmatched he0 tiles: {len(drop_keys)}", flush=True)

    # delete unmatched HE0 files
    for k in drop_keys:
        rec = he0_info.get(k, {})
        for fn_key in ["filename", "filename_he0_u16", "filename_he0_u8"]:
            fn = rec.get(fn_key, None)
            if fn:
                p = os.path.join(output_folder, fn)
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[DROP] removed {p}", flush=True)

    he0_info_new = {k: he0_info[k] for k in keep_keys}
    he_info_new = {k: he_info[k] for k in keep_keys}

    with open(he0_json, "w") as f:
        json.dump(he0_info_new, f, indent=2)

    with open(he_json, "w") as f:
        json.dump(he_info_new, f, indent=2)

    return keep_keys

# =============================
# Tile helpers
# =============================
def centroids_to_tiles(points_xy, tile_size):
    half = tile_size / 2.0
    tiles = []
    for (x, y) in points_xy:
        tiles.append({
            "x0": float(x - half),
            "y0": float(y - half),
            "w": float(tile_size),
            "h": float(tile_size),
        })
    return tiles


def _tile_to_xywh(p):
    if isinstance(p, dict):
        return float(p["x0"]), float(p["y0"]), float(p["w"]), float(p["h"])
    if isinstance(p, (list, tuple)) and len(p) == 4:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError(f"Unsupported tile format: {type(p)} {p}")


def save_he0_tiles(
    he0_rgb,
    tiles,
    output_folder,
    rescale_factor=1.0,
    prefix="pilot",
    start_index=0,
):
    """
    Save HE0 RGB tiles directly.
    """
    os.makedirs(output_folder, exist_ok=True)
    h_img, w_img = he0_rgb.shape[:2]

    out_meta = {}

    for i, p in enumerate(tiles, start=start_index):
        x0f, y0f, wf, hf = _tile_to_xywh(p)

        x0 = int(round(x0f * rescale_factor))
        y0 = int(round(y0f * rescale_factor))
        w = int(round(wf * rescale_factor))
        h = int(round(hf * rescale_factor))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x0 + w)
        y1 = min(h_img, y0 + h)

        if x1 <= x0 or y1 <= y0:
            continue

        tile = he0_rgb[y0:y1, x0:x1]
        key = f"{prefix}_{i:03d}"
        fn = f"{key}_he0.png"

        cv2.imwrite(os.path.join(output_folder, fn), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        out_meta[key] = {
            "x0": x0,
            "y0": y0,
            "w": int(x1 - x0),
            "h": int(y1 - y0),
            "cx": float((x0 + x1) / 2.0),
            "cy": float((y0 + y1) / 2.0),
            "id": int(i),
            "type": "pilot",
            "filename": fn,
        }

    with open(os.path.join(output_folder, "pilot_he0_tile_info.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"[OK] Saved HE0 pilot tiles: {len(out_meta)} -> {output_folder}", flush=True)
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
    case_id=0,
):
    """
    Save HE tiles by mapping HE0 tile corners -> HE via homography/affine.

    - tiles are defined in HE0 coordinate system
    - h_mat maps HE0 -> HE
    - rescale_factor converts HE coords -> he_rgb pixel coords

    Output:
      - <prefix>_<id>_he.png
      - pilot_he_tile_info.json
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

    def orient_quad_indices(case_id_: int):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=int)  # TL,TR,BR,BL

        def apply_case_xy(x, y):
            if case_id_ == 0:
                return x, y
            if case_id_ == 1:   # rot90 CW
                return y, 1 - x
            if case_id_ == 2:   # rot180
                return 1 - x, 1 - y
            if case_id_ == 3:   # rot90 CCW
                return 1 - y, x
            if case_id_ == 4:   # flip UD
                return x, 1 - y
            if case_id_ == 5:   # flip LR
                return 1 - x, y
            if case_id_ == 6:   # transpose
                return y, x
            if case_id_ == 7:   # anti-transpose
                return 1 - y, 1 - x
            raise ValueError(f"Unknown case_id={case_id_}")

        pts2 = np.array([apply_case_xy(x, y) for x, y in pts], dtype=int)

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

        expand = 1.0 + float(margin_ratio)
        cx = x0f + wf / 2.0
        cy = y0f + hf / 2.0
        half_w = (wf / 2.0) * expand
        half_h = (hf / 2.0) * expand

        x0p = cx - half_w
        x1p = cx + half_w
        y0p = cy - half_h
        y1p = cy + half_h

        corners_he0 = np.array(
            [[x0p, y0p],
             [x1p, y0p],
             [x1p, y1p],
             [x0p, y1p]],
            dtype=float
        )  # TL,TR,BR,BL

        corners_h = np.hstack([corners_he0, np.ones((4, 1), dtype=float)])
        proj = (H @ corners_h.T).T
        w = proj[:, 2:3]
        eps = 1e-9
        w_safe = np.where(np.abs(w) < eps, np.sign(w) * eps + (w == 0) * eps, w)
        corners_he = proj[:, :2] / w_safe
        corners_he_px_raw = corners_he * rf

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
            print(f"[DEBUG] {key} mode={mode}", flush=True)
            print(" corners_he0:\n", corners_he0, flush=True)
            print(" corners_he:\n", corners_he, flush=True)
            print(" corners_he_px_raw:\n", corners_he_px_raw, flush=True)

        if max_x_cl <= min_x_cl or max_y_cl <= min_y_cl:
            continue

        M = None
        Minv = None

        if mode == "bbox":
            tile_img = he_rgb[min_y_cl:max_y_cl, min_x_cl:max_x_cl]
            out_w = int(max_x_cl - min_x_cl)
            out_h = int(max_y_cl - min_y_cl)
        else:
            src = corners_he_px_raw.astype(np.float32)

            def dist(a, b):
                return float(np.linalg.norm(a - b))

            width = 0.5 * (dist(src[0], src[1]) + dist(src[3], src[2]))
            height = 0.5 * (dist(src[1], src[2]) + dist(src[0], src[3]))
            out_w = max(2, int(round(width)))
            out_h = max(2, int(round(height)))

            dst = np.array(
                [[0.0, 0.0],
                 [out_w - 1.0, 0.0],
                 [out_w - 1.0, out_h - 1.0],
                 [0.0, out_h - 1.0]],
                dtype=np.float32
            )

            idx_ord = orient_quad_indices(int(case_id))
            dst = dst[idx_ord]

            if signed_area(src) * signed_area(dst) < 0:
                src = src[[0, 3, 2, 1]]

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = np.linalg.inv(M)

            tile_img = cv2.warpPerspective(
                he_rgb,
                M,
                (out_w, out_h),
                flags=rectify_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        filename = f"{key}_he.png"
        cv2.imwrite(os.path.join(output_folder, filename), cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))

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
            "he0_corners": corners_he0.tolist(),
            "he_quad_px_raw": corners_he_px_raw.tolist(),
            "rectified_wh": [int(out_w), int(out_h)],
            "M_he_to_rect": None if M is None else M.tolist(),
            "M_rect_to_he": None if Minv is None else Minv.tolist(),
        }

        output_dict[key] = {**info, "meta": meta}
        he_tiles.append({**info, "meta": meta})

    with open(os.path.join(output_folder, "pilot_he_tile_info.json"), "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"[OK] Saved H&E pilot tiles: {len(he_tiles)} -> {output_folder}", flush=True)
    return he_tiles


# =============================
# Farthest point sampling
# =============================
def farthest_point_subset(points_xy: np.ndarray, k: int, start="centroid"):
    """
    Greedy FPS. O(N^2), but N is small.
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
    dmin = np.linalg.norm(pts - pts[s0], axis=1)

    for _ in range(1, k):
        idx = int(np.argmax(dmin))
        selected.append(idx)
        dnew = np.linalg.norm(pts - pts[idx], axis=1)
        dmin = np.minimum(dmin, dnew)

    return np.array(selected, dtype=int)


# =============================
# Main
# =============================
def main():
    if len(sys.argv) < 2:
        print("Usage: python 3b_tile_pilot.py <RUN_DIR>")
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

    HE0_PATH = info["HE0_path"]
    HE0_LEVEL = int(info["HE0_level"])
    HE_LEVEL = int(info["HE_level"])
    points_xy = np.asarray(sampled["points_xy"], dtype=np.int32)

    tile_size_lvl1 = float(step3["tile_size"])
    TILE_SIZE = tile_size_lvl1 / (2 ** (HE0_LEVEL - 1))

    sel_idx = farthest_point_subset(points_xy, pilot_k, start="centroid")
    sel_pts = points_xy[sel_idx]

    pilot_dir = run_dir / "pilot_tiles"
    ensure_dir(pilot_dir)

    with open(pilot_dir / "pilot_selection.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "he0_level": HE0_LEVEL,
                "pilot_k": int(pilot_k),
                "tile_size_lvl1": tile_size_lvl1,
                "tile_size_in_he0_level": float(TILE_SIZE),
                "selected_indices": sel_idx.tolist(),
                "selected_points_xy": sel_pts.tolist(),
            },
            f,
            indent=2,
        )
    print(f"[OK] selected {len(sel_idx)}/{len(points_xy)} pilot points -> {pilot_dir/'pilot_selection.json'}", flush=True)

    tiles = centroids_to_tiles(sel_pts, tile_size=TILE_SIZE)

    from my_utils import read_image

    parameter_path = Path(__file__).resolve().parent / "parameters.json"
    parameters = json.load(open(parameter_path, "r"))

    HE0_TILE_LEVEL_OVERRIDE = parameters["step3"].get("he0_level_override", "None")
    HE_TILE_LEVEL_OVERRIDE = parameters["step3"].get("he_level_override", "None")

    # -----------------------------
    # Read HE0 image for pilot tiles
    # -----------------------------
    if HE0_TILE_LEVEL_OVERRIDE in [None, "None"]:
        he0_read_level = 1
    elif isinstance(HE0_TILE_LEVEL_OVERRIDE, int):
        he0_read_level = int(HE0_TILE_LEVEL_OVERRIDE)
    else:
        raise ValueError("HE0_TILE_LEVEL_OVERRIDE must be None, 'None', or int")

    he0_rgb, _ = read_image(
        HE0_PATH,
        keep_16bit=False,
        level=he0_read_level,
        channel="he"
    )

    he0_rescale = 2 ** (HE0_LEVEL - he0_read_level)

    save_he0_tiles(
        he0_rgb,
        tiles,
        str(pilot_dir),
        rescale_factor=he0_rescale,
        prefix="pilot",
        start_index=0,
    )

    # -----------------------------
    # Read HE image for pilot tiles
    # -----------------------------
    HE_PATH = info["HE_path"]
    h_mat = load_initial_alignment(run_dir)

    if HE_TILE_LEVEL_OVERRIDE in [None, "None"]:
        if HE_LEVEL <= HE0_LEVEL:
            he_read_level = 1
        else:
            he_read_level = 1 + HE_LEVEL - HE0_LEVEL
            he_read_level = max(1, he_read_level)
    elif isinstance(HE_TILE_LEVEL_OVERRIDE, int):
        he_read_level = int(HE_TILE_LEVEL_OVERRIDE)
    else:
        raise ValueError("HE_TILE_LEVEL_OVERRIDE must be None, 'None', or int")

    he_rgb, _ = read_image(
        HE_PATH,
        keep_16bit=False,
        level=he_read_level,
        channel="he"
    )

    he_rescale = 2 ** (HE_LEVEL - he_read_level)

    save_he_tiles(
        he_rgb,
        tiles,
        h_mat,
        str(pilot_dir),
        rescale_factor=he_rescale,
        mode="rectified",  # or "bbox"
        case_id=int(info.get("HE0_orientation_case", 0)),
        prefix="pilot",
        start_index=0,
    )

    matched_keys = prune_unmatched_he0_pilot_tiles(str(pilot_dir))
    print(f"[OK] pilot matched HE0/HE tiles: {len(matched_keys)}", flush=True)

    print("[DONE] pilot extraction finished.", flush=True)

    # -----------------------------
    # Launch 3c automatically
    # -----------------------------
    script_dir = Path(__file__).resolve().parent
    script_3c = script_dir / "3c_pilot_tile_gallery.py"

    if not script_3c.exists():
        print(f"[WARN] 3c script not found: {script_3c}", flush=True)
    else:
        py = sys.executable
        cmd = [py, str(script_3c), str(run_dir), str(pilot_dir)]
        print("[INFO] launching 3c:", " ".join(cmd), flush=True)
        subprocess.Popen(cmd, cwd=str(script_dir))


if __name__ == "__main__":
    main()