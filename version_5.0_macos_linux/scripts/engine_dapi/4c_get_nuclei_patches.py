# ============================================================
# 4c_get_nuclei_patches.py - Stage 4c: Nucleus patch extraction
#
# Crops a patch from each side around every standout nucleus and writes
# nuclei_patches/nuclei_centroids_global.json (gates Stages 5-6). A side's token
# from `run_roles` names its patches and its centroid keys; its modality decides
# how the patch is rendered and which patch-level masker runs on it.
# Usage: python 4c_get_nuclei_patches.py <TILES_DIR>
# ============================================================
import json
import cv2
import os
import sys
import time
import numpy as np
from PIL import Image
import math
import argparse
from pathlib import Path
from my_utils import (
    dapi_to_lut_rgb,
    segment_super_dark_nuclei_full,
    read_crop_patch,
    upsample_tile,
    fill_holes_binary,
    remove_small_components,
    qc_images_enabled,
)


# CLI Stage 5 consumes only coordinates. The GUI also consumes these patches/masks;
# optional centre-marker overlays have their own QC group in both entry points.
WRITE_PATCH_OVERLAYS = qc_images_enabled("stage4c_overlays")

_VERSION_ROOT = str(Path(__file__).resolve().parents[1])
if _VERSION_ROOT not in sys.path:
    sys.path.insert(0, _VERSION_ROOT)
import run_roles
from run_artifacts import initial_alignment_path
from nucleus_patch_geometry import GEOMETRY_DEFAULTS, resolve_geometry, local_centroid
from pyramid_region_reader import PyramidRegionReader
from patch_qc import MANIFEST, centroid_digest

# The two sides as this stage names them; `load_run_metadata` fills it from the run.
# The default is the `dapi-he` spelling so a function called on its own reads and
# writes what it always did.
SIDES = {
    "pair": "dapi-he",
    "moving": {"token": "dapi", "modality": "dapi"},
    "fixed": {"token": "he", "modality": "he"},
}


def set_sides(roles):
    SIDES["pair"] = roles.pair.name
    for role in ("moving", "fixed"):
        image = roles.role(role)
        SIDES[role] = {"token": image.token, "modality": image.modality}


def tok(side):
    return SIDES[side]["token"]


# -----------------------------
# Progress / timing helpers
# -----------------------------
def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

def print_progress(stage_name, i, total):
    print(f"[PROGRESS] {stage_name} {i}/{total}", flush=True)

def print_eta_start(stage_name):
    print(f"[ETA_START] {stage_name}", flush=True)


# -----------------------------
# Geometry and coordinate conversion helpers
# -----------------------------
def apply_homography_xy(M3x3, x, y):
    M3x3 = np.asarray(M3x3, dtype=float)
    p = np.array([float(x), float(y), 1.0], dtype=float)
    q = M3x3 @ p
    w = q[2] if abs(q[2]) > 1e-12 else 1e-12
    return float(q[0] / w), float(q[1] / w)


def stage3_he_tile_level(sampled_info, images_info, dapi_tile_level):
    """The H&E level Stage 3 actually read.

    Stage 3 records it because the level is chosen from micrometres per pixel and can no
    longer be re-derived as HE_level - (DAPI_level - dapi_tile_level). That subtraction is
    kept only for runs made before the level was recorded.
    """
    recorded = sampled_info.get("he_tile_level")
    if recorded is not None:
        return max(0, int(recorded))
    return max(1, int(images_info["HE_level"])
               - (int(images_info["DAPI_level"]) - int(dapi_tile_level)))

def tile_level_to_patch_level(tile_level: int) -> int:
    """Legacy policy: read one pyramid level finer than the extracted tile.

    Both stages already use zero-based levels; this is not an indexing correction.
    """
    return max(0, int(tile_level) - 1)

def he_point_tile_to_image_coords(he_info, x_tile, y_tile):
    """
    Return point coordinates in the same coordinate system as the loaded HE image
    (i.e. HE_patch_level coordinates).

    If rectified:
        (x_tile, y_tile) is in rectified coords -> project back using M_rect_to_he
    Else:
        (x_tile, y_tile) is tile-local bbox coords -> x0/y0 add
    """
    meta = he_info.get("meta", {}) if isinstance(he_info, dict) else {}

    if meta.get("mode", None) == "rectified" and meta.get("M_rect_to_he", None) is not None:
        Minv = np.asarray(meta["M_rect_to_he"], dtype=float)
        x_img, y_img = apply_homography_xy(Minv, x_tile, y_tile)
        return x_img, y_img

    x_img = float(he_info["x0"]) + float(x_tile)
    y_img = float(he_info["y0"]) + float(y_tile)
    return x_img, y_img

def load_initial_affine_and_compute_he_patch(
    run_dir,
    images_info,
    dapi_patch_len=100,
    margin=1.1,
    return_scale=False,
):
    run_dir = Path(run_dir)

    path_initial = initial_alignment_path(run_dir.parent)
    if path_initial.exists():
        data = json.loads(path_initial.read_text())
        src = path_initial.name
    else:
        raise FileNotFoundError(path_initial)

    if "affine_2x3" in data:
        M = np.array(data["affine_2x3"], dtype=float)
    elif "affine_3x3" in data:
        M = np.array(data["affine_3x3"], dtype=float)[:2, :]
    elif "H_mat" in data:
        M = np.array(data["H_mat"], dtype=float)
    elif "matrix" in data:
        M = np.array(data["matrix"], dtype=float)
    else:
        raise ValueError("Cannot find affine matrix in alignment json")

    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]

    sx = math.sqrt(a * a + b * b)
    sy = math.sqrt(c * c + d * d)

    level_he = images_info[f"{tok('fixed').upper()}_level"]
    level_dapi = images_info[f"{tok('moving').upper()}_level"]
    scale_level = 2 ** (level_he - level_dapi)

    sx0 = sx * scale_level
    sy0 = sy * scale_level
    s0 = max(sx0, sy0)

    if return_scale:
        return s0

    he_patch_len = int(math.ceil(dapi_patch_len * s0 * margin))
    if he_patch_len % 2 == 1:
        he_patch_len += 1

    print(
        f"[INFO] Initial alignment from {src}\n"
        f"       affine scale: sx={sx:.3f}, sy={sy:.3f}\n"
        f"       level correction: 2^({level_he}-{level_dapi}) = {scale_level:.3f}\n"
        f"       level0 scale: sx0={sx0:.3f}, sy0={sy0:.3f}\n"
        f"       DAPI={dapi_patch_len} -> HE={he_patch_len} (margin={margin})",
        flush=True
    )

    return he_patch_len


# -----------------------------
# Patch extraction and visualization helpers
# -----------------------------
def ensure_gray_uint16(x):
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., 0]
    return x.astype(np.uint16)

def stretch_to_uint8_percentile(raw16, p_low=1, p_high=99.7):
    raw16 = raw16.astype(np.float32)
    lo, hi = np.percentile(raw16, [p_low, p_high])
    if hi <= lo:
        return np.zeros(raw16.shape, np.uint8)
    vis8 = (raw16 - lo) * 255.0 / (hi - lo)
    return np.clip(vis8, 0, 255).astype(np.uint8)

def get_dot_radius(img, base_size=200, base_r=4):
    h, w = img.shape[:2]
    scale = min(h, w) / base_size
    r = int(round(base_r * scale))
    return max(1, r)

def extract_patch_and_mark_point(
    img,
    x_global,
    y_global,
    tile_id,
    nucleus_id,
    type=None,
    patch_length=60,
    out_dir="",
    save_patch=True,
    save_overlay=True,
    save_raw_dapi=True,
    modality=None,
    centroid_local=None,
):
    """
    No refine. Only:
      - crop patch around (x_global, y_global)
      - for DAPI: also save a raw visualization (uint16 -> uint8)
      - save overlay with a dot at the center point

    `type` is the side's token and names the files; `modality` decides how the patch
    is rendered and defaults to the token, which is right for `dapi-he`.
    """
    if modality is None:
        modality = type
    half = patch_length // 2
    is_region_reader = hasattr(img, "read_region")
    H, W = (img.height, img.width) if is_region_reader else img.shape[:2]
    x_global, y_global = float(x_global), float(y_global)
    local = centroid_local if centroid_local is not None else (half, half)
    x0 = math.floor(x_global - float(local[0]))
    y0 = math.floor(y_global - float(local[1]))
    rx0, ry0 = max(0, x0), max(0, y0)
    rx1, ry1 = min(W, x0 + patch_length), min(H, y0 + patch_length)
    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError(f"Patch outside the source image at ({x_global}, {y_global})")
    read_started = time.perf_counter()
    raw = (img.read_region(rx0, ry0, rx1, ry1) if is_region_reader
           else read_crop_patch(img, rx0, ry0, rx1 - rx0, ry1 - ry0))
    read_sec = time.perf_counter() - read_started
    shape = (patch_length, patch_length) + raw.shape[2:]
    fill = (np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0) if modality == "he" else 0
    crop = np.full(shape, fill, dtype=raw.dtype)
    crop[ry0-y0:ry1-y0, rx0-x0:rx1-x0] = raw
    cx0, cy0 = x_global - x0, y_global - y0

    crop_raw16 = None
    crop_vis = crop

    if modality == "dapi":
        crop_raw16 = ensure_gray_uint16(crop)
        crop_vis = dapi_to_lut_rgb(crop_raw16, lut, threshold=1000)
    elif modality == "he":
        if crop_vis.ndim == 2:
            crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2RGB)
        if crop_vis.dtype != np.uint8:
            crop_vis = np.clip(crop_vis / 256.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError("modality must be 'dapi' or 'he'")

    os.makedirs(out_dir, exist_ok=True)

    if save_patch:
        patch_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch.png"
        cv2.imwrite(patch_path, cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

        if modality == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_norm_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_raw.png"
            cv2.imwrite(raw_norm_path, raw_vis8)

    if save_overlay:
        overlay = crop_vis.copy()
        cv2.circle(overlay, (round(cx0), round(cy0)), int(get_dot_radius(overlay)), (255, 0, 0), -1)
        overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_patch_overlay.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if modality == "dapi" and save_raw_dapi and (crop_raw16 is not None):
            raw_vis8 = stretch_to_uint8_percentile(crop_raw16, p_low=1, p_high=99.7)
            raw_overlay = cv2.cvtColor(raw_vis8, cv2.COLOR_GRAY2BGR)
            cv2.circle(raw_overlay, (round(cx0), round(cy0)), int(get_dot_radius(overlay)), (0, 0, 255), -1)
            raw_overlay_path = f"{out_dir}/{tile_id}_nucleus_{nucleus_id}_{type}_raw_overlay.png"
            cv2.imwrite(raw_overlay_path, raw_overlay)

    info = {
        "centroid_global": [float(x_global), float(y_global)],
        "centroid_local": [int(cx0), int(cy0)],
        "point_patch": [int(cx0), int(cy0)],
        "crop_box": [int(x0), int(y0), int(x0 + patch_length), int(y0 + patch_length)],
        "read_sec": read_sec,
    }
    return float(x_global), float(y_global), crop_vis, info


# -----------------------------
# Patch-level mask generation helpers
# -----------------------------
def load_stage4c_params_from_parameters_json(
    out_folder: str,
    default_dapi_patch_len=200,
    default_dapi_offset=0,
    default_dapi_min_area_factor=1e-4,
    default_dapi_upscale=1,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=1,
):
    """
    Read ../../parameters.json and return stage4c params.
    Falls back to the provided defaults if the file or keys are missing.
    """
    from my_utils import cli_params_block
    from role_parameters import role_parameter

    params = {
        "dapi_patch_len": int(default_dapi_patch_len),
        "dapi_thr_offset": int(default_dapi_offset),
        "dapi_mask_min_area_factor": float(default_dapi_min_area_factor),
        "dapi_mask_upscale_factor": int(default_dapi_upscale),
        "he_mask_n_smooth": int(default_he_n_smooth),
        "he_mask_intensity_threshold": float(default_he_thr),
        "he_mask_upscale_factor": int(default_he_upscale),
    }

    data = cli_params_block(SIDES["pair"])

    stage4c = data.get("stage4c", {}) if isinstance(data, dict) else {}
    params["_raw_block"] = dict(stage4c)
    params.update({key: stage4c.get(key, value) for key, value in GEOMETRY_DEFAULTS.items()})

    try:
        params["dapi_patch_len"] = int(
            role_parameter(stage4c, "moving_patch_len", "dapi_patch_len", params["dapi_patch_len"])
        )
    except Exception:
        pass

    try:
        params["dapi_thr_offset"] = int(
            float(stage4c.get("dapi_thr_offset", params["dapi_thr_offset"]))
        )
    except Exception:
        pass

    try:
        params["dapi_mask_min_area_factor"] = float(
            stage4c.get("dapi_mask_min_area_factor", params["dapi_mask_min_area_factor"])
        )
    except Exception:
        pass

    try:
        params["dapi_mask_upscale_factor"] = int(
            stage4c.get("dapi_mask_upscale_factor", params["dapi_mask_upscale_factor"])
        )
    except Exception:
        pass

    try:
        params["he_mask_n_smooth"] = int(
            stage4c.get("he_mask_n_smooth", params["he_mask_n_smooth"])
        )
    except Exception:
        pass

    try:
        params["he_mask_intensity_threshold"] = float(
            stage4c.get("he_mask_intensity_threshold", params["he_mask_intensity_threshold"])
        )
    except Exception:
        pass

    try:
        params["he_mask_upscale_factor"] = int(
            stage4c.get("he_mask_upscale_factor", params["he_mask_upscale_factor"])
        )
    except Exception:
        pass

    return params

def load_stage4c_effective_params(
    out_folder: str,
    default_dapi_patch_len=200,
    default_dapi_offset=0,
    default_dapi_min_area_factor=1e-4,
    default_dapi_upscale=1,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=1,
):
    """
    Resolve Stage 4c parameters with this priority:
      1) parameters.json -> stage4c
      2) pilot_output_parameters.json overrides:
           - dapi_thr_offset
           - he_mask_intensity_threshold
      3) hard-coded defaults
    """
    params = load_stage4c_params_from_parameters_json(
        out_folder,
        default_dapi_patch_len=default_dapi_patch_len,
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
    from pilot_mask_parameters import apply_token_overrides
    apply_token_overrides(params, src)
    return params

def process_dapi(
    dapi_file,
    THR_OFFSET=0,
    min_area_factor=10e-5,
    CONNECTIVITY=8,
    invert=True,
    upscale=1,
    token="dapi",
    centroid_local=None,
):
    """
    Input:  *_<token>_u8.png or *_<token>_raw.png
    Output: *_<token>_mask.png and *_<token>_mask_overlay.png
    """
    try:
        if not os.path.exists(dapi_file):
            return f"Skipped (DAPI not found): {dapi_file}"

        base_name = os.path.basename(dapi_file)
        if base_name.endswith(f"_{token}_u8.png"):
            base_name = base_name[:-len(f"_{token}_u8.png")]
        elif base_name.endswith(f"_{token}_raw.png"):
            base_name = base_name[:-len(f"_{token}_raw.png")]
        else:
            return f"Skipped (not a valid {token} file): {dapi_file}"

        folder_name = os.path.dirname(dapi_file)

        img = cv2.imread(dapi_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed: {dapi_file}, Error: cv2.imread returned None"
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = upsample_tile(img, upscale)

        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(int(otsu_thr) + int(THR_OFFSET), 0, 255))
        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        fg_ratio0 = float((mask > 0).mean())

        mask = fill_holes_binary(mask)
        fg_ratio1 = float((mask > 0).mean())

        min_area = min_area_factor * img.shape[0] ** 2
        min_area = min_area * upscale ** 2
        mask, cc_info = remove_small_components(
            mask, min_area=int(min_area), connectivity=int(CONNECTIVITY)
        )
        fg_ratio2 = float((mask > 0).mean())

        if invert:
            mask = 255 - mask

        mask_save_path = os.path.join(folder_name, f"{base_name}_{token}_mask.png")
        cv2.imwrite(mask_save_path, mask)

        kept = None
        total = None
        if isinstance(cc_info, dict):
            kept = cc_info.get("cc_kept", None)
            total = cc_info.get("cc_total", None)

        extra = f" fg {fg_ratio0:.3f}->{fg_ratio1:.3f}->{fg_ratio2:.3f}"
        if kept is not None and total is not None:
            extra += f" cc kept {kept}/{total}"

        if WRITE_PATCH_OVERLAYS:
            mask_u8 = mask
            h, w = mask_u8.shape[:2]
            cx, cy = w // 2, h // 2
            if centroid_local is not None:
                cx, cy = (int(round(float(v) * upscale)) for v in centroid_local)
            overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            cv2.circle(overlay, (cx, cy), int(get_dot_radius(overlay)), (255, 0, 0), -1)
            overlay_save_path = mask_save_path[:-len("_mask.png")] + "_mask_overlay.png"
            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return f"Processed: {os.path.basename(dapi_file)} otsu={int(otsu_thr)}"

    except Exception as e:
        return f"Failed: {dapi_file}, Error: {e}"

def process_he(image_file, upscale=1, n_smooth=2, intensity_threshold=0.6, token="he", centroid_local=None):
    try:
        rgb_tile = np.array(Image.open(image_file))
        _, mask_dark = segment_super_dark_nuclei_full(
            rgb_tile,
            upsample_scale=upscale,
            n_smooth=int(n_smooth),
            intensity_threshold=float(intensity_threshold)
        )
        mask_save_path = image_file.replace(f"_{token}_patch.png", f"_{token}_mask.png")
        cv2.imwrite(mask_save_path, mask_dark.astype(np.uint8) * 255)

        if WRITE_PATCH_OVERLAYS:
            mask_u8 = mask_dark.astype(np.uint8) * 255
            h, w = mask_u8.shape[:2]
            cx, cy = w // 2, h // 2
            if centroid_local is not None:
                cx, cy = (int(round(float(v) * upscale)) for v in centroid_local)
            overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            cv2.circle(overlay, (cx, cy), int(get_dot_radius(overlay)), (255, 0, 0), -1)
            overlay_save_path = image_file.replace(f"_{token}_patch.png", f"_{token}_mask_overlay.png")
            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return f"Processed: {image_file} with int_thr {intensity_threshold}"

    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"

# -----------------------------
# Metadata loading and image context setup
# -----------------------------
def load_run_metadata(run_dir):
    print("[INFO] Loading metadata", flush=True)

    scripts_dir = str(Path(__file__).resolve().parent)
    from my_utils import cli_params_block
    roles = run_roles.read(Path(run_dir).resolve().parent)
    parameters = cli_params_block(roles.params_block)
    set_sides(roles)
    print(f"[INFO] pair {roles.pair.name}: moving={tok('moving')} fixed={tok('fixed')}",
          flush=True)
    with open(Path(run_dir) / "standout_nuclei.json") as f:
        nuclei = json.load(f)
    with open(Path(run_dir) / f"{tok('moving')}_tile_info.json") as f:
        dapi_tiles = json.load(f)
    with open(Path(run_dir) / f"{tok('fixed')}_tile_info.json") as f:
        he_tiles = json.load(f)
    with open(Path(run_dir).parent / "sampled_points.json") as f:
        sampled_info = json.load(f)
    with open(Path(run_dir).parent / "images_info.json") as f:
        images_info = json.load(f)

    return {
        "scripts_dir": scripts_dir,
        "parameters": parameters,
        "nuclei": nuclei,
        "dapi_tiles": dapi_tiles,
        "he_tiles": he_tiles,
        "sampled_info": sampled_info,
        "images_info": images_info,
        "roles": roles,
    }


def _moving_tile_level(sampled_info):
    return int(sampled_info.get(f"{tok('moving')}_tile_level",
                                sampled_info.get("dapi_tile_level")))


def _fixed_tile_level(sampled_info, images_info, moving_tile_level):
    recorded = sampled_info.get(f"{tok('fixed')}_tile_level")
    if recorded is None:
        return stage3_he_tile_level(sampled_info, images_info, moving_tile_level)
    return max(0, int(recorded))



def save_global_centroids(out_dir, output_coord_record, image_ctx):
    out_json = os.path.join(out_dir, "nuclei_centroids_global.json")

    out = {
        "meta": {
            f"{tok('moving')}_tile_level": int(image_ctx.get("dapi_tile_level", -1)),
            f"{tok('moving')}_patch_level": int(image_ctx.get("dapi_patch_level", -1)),
            f"{tok('fixed')}_tile_level": int(image_ctx.get("he_tile_level", -1)),
            f"{tok('fixed')}_patch_level": int(image_ctx.get("he_patch_level", -1)),
            "pair": SIDES["pair"],
            "patch_geometry": image_ctx,
        },
        "data": output_coord_record
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved centroids to {out_json}", flush=True)


def build_centroids_without_patches(run_dir, meta, dapi_patch_len=200, image_ctx=None):
    """Build the Stage-5 coordinate table without loading full-resolution images.

    The headless Stage 5 consumes only the level-0 centroids. Stage-3 metadata
    already contains every transform needed to restore those coordinates, so
    reading two large source levels and encoding ten PNGs per pair is unnecessary
    in an automatic run. GUI and CLI rendering consume this same coordinate table.
    """
    images_info = meta["images_info"]
    sampled_info = meta["sampled_info"]
    nuclei = meta["nuclei"]
    dapi_tiles = meta["dapi_tiles"]
    he_tiles = meta["he_tiles"]

    dapi_tile_level = _moving_tile_level(sampled_info)
    he_tile_level = _fixed_tile_level(sampled_info, images_info, dapi_tile_level)
    dapi_patch_level = tile_level_to_patch_level(dapi_tile_level)
    he_patch_level = tile_level_to_patch_level(he_tile_level)
    dapi_tile_to_patch = float(2 ** (dapi_tile_level - dapi_patch_level))
    he_tile_to_patch = float(2 ** (he_tile_level - he_patch_level))
    dapi_patch_to_level0 = float(2 ** dapi_patch_level)
    he_patch_to_level0 = float(2 ** he_patch_level)

    he_patch_len = (image_ctx["he_patch_len"] if image_ctx is not None else
                   load_initial_affine_and_compute_he_patch(
                       run_dir, images_info, dapi_patch_len=int(dapi_patch_len), margin=1.1))
    records = []
    for n in nuclei:
        tile_id = n["tile"]
        nucleus_id = n.get("nucleus_id", 0)
        dapi_info = dapi_tiles[tile_id]
        he_info = he_tiles[tile_id]

        x_dapi_tile, y_dapi_tile = n["original"][tok("moving")]
        x_dapi_patch = (float(dapi_info["x0"]) + float(x_dapi_tile)) * dapi_tile_to_patch
        y_dapi_patch = (float(dapi_info["y0"]) + float(y_dapi_tile)) * dapi_tile_to_patch

        x_he_tile, y_he_tile = n["original"][tok("fixed")]
        x_he_image, y_he_image = he_point_tile_to_image_coords(
            he_info, x_he_tile, y_he_tile
        )
        # Keep sub-pixel centroids: only the patch crop needs integer pixels, and it
        # floors its own origin (extract_patch_and_mark_point, local_centroid).
        x_he_patch = float(x_he_image) * he_tile_to_patch
        y_he_patch = float(y_he_image) * he_tile_to_patch

        records.append({
            "tile_id": tile_id,
            "nucleus_id": nucleus_id,
            "mode": n.get("mode", "nuclei_pair"),
            f"{tok('moving')}_centroid_global": [
                float(x_dapi_patch * dapi_patch_to_level0),
                float(y_dapi_patch * dapi_patch_to_level0),
            ],
            f"{tok('moving')}_centroid_local": [int(dapi_patch_len // 2), int(dapi_patch_len // 2)],
            f"{tok('fixed')}_centroid_global": [
                float(x_he_patch * he_patch_to_level0),
                float(y_he_patch * he_patch_to_level0),
            ],
            f"{tok('fixed')}_centroid_local": [int(he_patch_len // 2), int(he_patch_len // 2)],
        })

    image_ctx = image_ctx if image_ctx is not None else {
        "dapi_tile_level": dapi_tile_level,
        "dapi_patch_level": dapi_patch_level,
        "he_tile_level": he_tile_level,
        "he_patch_level": he_patch_level,
    }
    return records, image_ctx

# -----------------------------
# Main pipeline stages
# -----------------------------

def patch_geometry(run_dir, meta, params):
    moving_level = _moving_tile_level(meta["sampled_info"])
    fixed_level = _fixed_tile_level(meta["sampled_info"], meta["images_info"], moving_level)
    scale0 = load_initial_affine_and_compute_he_patch(
        run_dir, meta["images_info"], return_scale=True,
    )
    alignment = json.loads(initial_alignment_path(Path(run_dir).resolve().parent).read_text())
    return resolve_geometry(
        meta["roles"], {"moving": moving_level, "fixed": fixed_level}, params, scale0,
        level0_homography=alignment.get("H_mat_level_0"),
    )


def render_patch_images(meta, records, context, out_dir, params):
    """One bounded-memory renderer for CLI QC and the interactive gallery."""
    global lut
    roles = meta["roles"]
    lut_path = meta["images_info"].get("DAPI_LUT", str(Path(__file__).parent / "glasbey_inverted.lut"))
    if any(side.modality == "dapi" for side in (roles.moving, roles.fixed)):
        lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)
    files = []
    timings = {"roi_read_sec": 0.0, "reader_open_sec": 0.0}
    started = time.perf_counter()
    # Open/read one side at a time; never decode either entire slide level.
    for role, legacy in (("moving", "dapi"), ("fixed", "he")):
        side = roles.role(role)
        level = context[f"{legacy}_patch_level"]
        length = context[f"{legacy}_patch_len"]
        print(f"[INFO] {side.token}: patch level {level}, side {length}px; ROI reads only", flush=True)
        opened = time.perf_counter()
        with PyramidRegionReader(
            Path(side.path), level, side.modality,
            channel_index=side.channel_index,
        ) as reader:
            timings["reader_open_sec"] += time.perf_counter() - opened
            for index, row in enumerate(records):
                token = side.token
                prefix = f"{row['tile_id']}_nucleus_{row['nucleus_id']}_{token}"
                global_xy = row[f"{token}_centroid_global"]
                local_xy = row[f"{token}_centroid_local"]
                if not all(0 <= float(v) < length for v in local_xy):
                    raise ValueError("Patch size differs from the saved centroid table; "
                                     "rerun Stage 4c explicitly to change patch geometry.")
                _, _, _, crop_info = extract_patch_and_mark_point(
                    reader, float(global_xy[0]) / 2 ** level, float(global_xy[1]) / 2 ** level,
                    row["tile_id"], row["nucleus_id"], type=token, modality=side.modality,
                    patch_length=length, out_dir=str(out_dir), save_overlay=WRITE_PATCH_OVERLAYS,
                    centroid_local=local_xy,
                )
                timings["roi_read_sec"] += crop_info["read_sec"]
                from pilot_mask_parameters import side_value
                def value(key):
                    return side_value(params, token, side.modality, key)
                if side.modality == "dapi":
                    result = process_dapi(
                        str(out_dir / f"{prefix}_raw.png"),
                        THR_OFFSET=value("thr_offset"),
                        min_area_factor=value("mask_min_area_factor"),
                        upscale=value("mask_upscale_factor"), token=token,
                        centroid_local=local_xy,
                    )
                    suffixes = ["patch", "raw", "mask"]
                    if WRITE_PATCH_OVERLAYS:
                        suffixes += ["patch_overlay", "raw_overlay", "mask_overlay"]
                else:
                    result = process_he(
                        str(out_dir / f"{prefix}_patch.png"),
                        upscale=value("mask_upscale_factor"),
                        n_smooth=value("mask_n_smooth"),
                        intensity_threshold=value("mask_intensity_threshold"), token=token,
                        centroid_local=local_xy,
                    )
                    suffixes = ["patch", "mask"]
                    if WRITE_PATCH_OVERLAYS:
                        suffixes += ["patch_overlay", "mask_overlay"]
                if not result.startswith("Processed:"):
                    raise RuntimeError(result)
                files.extend(f"{prefix}_{suffix}.png" for suffix in suffixes)
                print_progress(f"{token}_PATCH", index + 1, len(records))
    missing = [name for name in files if not (out_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing patch outputs: {missing[:5]}")
    timings["render_total_sec"] = time.perf_counter() - started
    print(f"[TIMER] Patch generation: {timings}", flush=True)
    return files, timings


def main(run_dir, centroids_only=False, patches_only=False):
    if centroids_only and patches_only:
        raise ValueError("--centroids-only and --patches-only cannot be combined")
    global WRITE_PATCH_OVERLAYS
    WRITE_PATCH_OVERLAYS = qc_images_enabled("stage4c_overlays")
    run_dir = Path(run_dir).resolve()
    meta = load_run_metadata(run_dir)
    out_dir = run_dir.parent / "nuclei_patches"
    out_dir.mkdir(exist_ok=True)
    params = load_stage4c_effective_params(
        out_dir, default_dapi_patch_len=200, default_dapi_offset=0,
        default_dapi_min_area_factor=1e-4, default_dapi_upscale=1,
        default_he_n_smooth=2, default_he_thr=0.6, default_he_upscale=1,
    )
    context = patch_geometry(run_dir, meta, params)
    context["standout_sha256"] = centroid_digest(run_dir / "standout_nuclei.json")
    centroid_path = out_dir / "nuclei_centroids_global.json"
    if patches_only:
        # A resume QC top-up must not replace reviewed/dropped/manual coordinates.
        saved = json.loads(centroid_path.read_text())
        records = saved["data"]
        saved_geometry = saved.get("meta", {}).get("patch_geometry")
        if saved_geometry is not None:
            # A late QC request renders the saved crop geometry. It cannot change
            # the coordinate origin of already-reviewed points.
            context = saved_geometry
        for role, legacy in (("moving", "dapi"), ("fixed", "he")):
            key = f"{tok(role)}_patch_level"
            if int(saved["meta"][key]) != context[f"{legacy}_patch_level"]:
                raise ValueError("Saved patch levels differ from requested levels; "
                                 "rerun Stage 4c explicitly to change geometry.")
    else:
        # Centroids are level-0 floats, independent of QC and requested display
        # resolution; only local crop coordinates vary.
        records, context = build_centroids_without_patches(
            run_dir, meta, dapi_patch_len=context["dapi_patch_len"], image_ctx=context,
        )
        for row in records:
            for role, legacy in (("moving", "dapi"), ("fixed", "he")):
                token = tok(role)
                row[f"{token}_centroid_local"] = local_centroid(
                    row[f"{token}_centroid_global"], context[f"{legacy}_patch_level"],
                    context[f"{legacy}_patch_len"],
                )
    print(f"[INFO] Patch geometry: {context}", flush=True)
    if centroids_only:
        save_global_centroids(out_dir, records, context)
        print(f"[INFO] Centroid-only mode: {len(records)} pairs; no pixel reads or PNGs", flush=True)
    else:
        files, timings = render_patch_images(meta, records, context, out_dir, params)
        if not patches_only:
            save_global_centroids(out_dir, records, context)
        manifest = {
            "geometry": context, "timing": timings, "overlays": WRITE_PATCH_OVERLAYS, "files": files,
            "centroids_sha256": centroid_digest(centroid_path), "pair_count": len(records),
            "standout_sha256": context["standout_sha256"],
        }
        (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print("[DONE]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument(
        "--centroids-only",
        action="store_true",
        help="headless fast path: write the Stage-5 coordinate table without gallery PNGs",
    )
    parser.add_argument("--patches-only", action="store_true",
                        help="add gallery/QC images from the saved centroid table without changing it")
    args = parser.parse_args()
    main(args.run_dir, centroids_only=args.centroids_only, patches_only=args.patches_only)
