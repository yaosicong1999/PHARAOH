# ============================================================
# 4b_find_standout_nuclei.py - Stage 4b: Standout nuclei detection
#
# Aligns the two sides' masks per tile (3-phase) and writes
# standout_nuclei.json with anchors. One process runs one anchor level:
#   nuclei_pairs -> individual isolated nuclei, paired by overlap
#                   (binary Dice search); the only anchor whose two
#                   coordinates are measured independently
#   density      -> locally conserved nuclear-density architecture
#                   (density-NCC search), for adjacent sections where
#                   no nucleus exists on both sides
# A level that comes up short records the shortfall in pipeline_times.json
# and fails; run_pharaoh.py climbs the ladder `stage4b.anchor_mode` names
# (see scripts/stage4b_modes.py). Tile centres are no longer shipped.
# Usage: python 4b_find_standout_nuclei.py <TILES_DIR>
#            [--anchor-mode nuclei_pairs|density] [--all-masked]
#            [--all-tiles] [--skip-qa-images] [--write-centroids]
# ============================================================
from fractions import Fraction
import argparse
import cv2
import numpy as np
from joblib import Parallel, delayed, cpu_count
import os
import glob
from scipy import ndimage
import time
import json
import sys
from pathlib import Path

ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION_ROOT = os.path.dirname(ENGINE_ROOT)
if VERSION_ROOT not in sys.path:
    sys.path.insert(0, VERSION_ROOT)
# Python places the executing script's directory in sys.path already. Move it
# explicitly to the front after adding VERSION_ROOT so ``from my_utils`` always
# resolves to engine_dapi/my_utils rather than scripts/my_utils.
if ENGINE_ROOT in sys.path:
    sys.path.remove(ENGINE_ROOT)
sys.path.insert(0, ENGINE_ROOT)
from spatial_tile_selection import create_selection_manifest, load_selection_manifest
from stage4_tile_budget import stage4b_tile_count
import stage4b_modes
from stage4_all_tiles import COMPLETION_FILE, shared_stage3_ids, write_all_tiles_completion
import nucleus_masks
import run_roles

# The two sides of the pair as this stage names them. `load_runtime_context` sets
# them from the run; the default is the `dapi-he` spelling so a function called on
# its own -- the unit tests do this -- still reads the files it always read.
# "A" is the moving mask (oriented by the moving side's case), "B" the fixed one.
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


def mask_name(prefix, side):
    return nucleus_masks.mask_filename(prefix, SIDES[side]["token"])


def raw_tile_name(prefix, side):
    return nucleus_masks.tile_filename(prefix, SIDES[side]["token"], SIDES[side]["modality"])


def prefix_of(mask_path, side="moving"):
    """The tile id a side's mask file belongs to."""
    suffix = f"_{SIDES[side]['token']}{nucleus_masks.MASK_SUFFIX}"
    name = os.path.basename(mask_path)
    return name[:-len(suffix)] if name.endswith(suffix) else name

# -----------------------------
# Formatting / config helpers
# -----------------------------
def fmt_sec(sec):
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def load_stage4b_params(base_dir, pair=None):
    from my_utils import cli_params_block

    # -------- defaults --------
    params = {
        # The ladder the launcher climbs (see stage4b_modes). One Stage-4b process
        # runs one level of it; `level` and the `score_metric` it searches with are
        # set by `resolve_level`, never read from the file.
        "anchor_mode": stage4b_modes.DEFAULT_MODE,
        # The density rung is an architecture matcher, so its blur is defined in
        # physical units.  `load_runtime_context` converts this to mask pixels.  The
        # old pixel value remains as a visible fallback for legacy images without an
        # MPP or tile-level record; the nuclei-pairs rung never reads either value.
        "density_sigma_um": 8.0,
        "density_sigma_px": 8.0,
        # Density alignment is meaningful only where both serial sections carry
        # enough coherent tissue.  The effective foreground floor is the larger
        # of this absolute floor and the selected tiles' quantile below.  These
        # gates are consulted by the density level only; the nuclei-pairs path is
        # deliberately unchanged.
        "density_min_foreground_fraction": 0.02,
        "density_rich_tissue_quantile": 0.50,
        "density_max_foreground_ratio": 4.0,
        "density_min_largest_component_fraction": 0.03,
        "architecture_grid_size": 3,
        "architecture_max_anchors_per_tile": 3,
        "architecture_min_region_mass_fraction": 0.04,
        "architecture_min_region_ncc": 0.10,
        "architecture_min_anchor_distance_fraction": 0.15,
        # A serial-section tile is accepted from its local architecture evidence.
        # The whole-tile NCC is retained only as a weak sanity gate: demanding the
        # same 0.4 used by the nuclei-pairs rung discards valid local anchors whenever
        # adjacent sections differ outside those anchor neighbourhoods.
        "density_min_full_ncc": 0.05,
        "density_min_anchors_per_tile": 2,
        "density_min_anchor_median_ncc": 0.15,
        # Stage 3 already determines the geometric B-to-A scale from its fixed-tile
        # margin (normally 1.2).  Serial-section density is too repetitive to
        # re-estimate that scale independently in every tile, so search narrowly
        # around the known geometry and reject boundary-sized local corrections.
        "density_scale_range_frac": 0.05,
        "density_shift_frac": 0.25,
        "density_shift_step_frac": 0.025,
        "density_max_shift_fraction": 0.24,

        "good_nuclei_min": 2,
        "min_good_tiles": 40,
        "fallback_score_thr": 0.30,
        "min_fallback_tiles": 20,

        "pair_top_k": 3,
        "pairs_to_take_per_tile": 2,
        # None: the shared rule (`stage4_tile_budget`), i.e. the tiles Stage 4a masked.
        "n_tiles": None,

        "phase1": {
            "n_tiles_for_scale_estimation": 50,
            "ds": 4,
            "scale_step": 0.05,
            "shift_frac": 0.20,
            "shift_step_frac": 0.05,
        },

        "phase2": {
            "scale_step": 0.02,
            "shift_frac": 0.40,
            "shift_step_frac": 0.05,
            "median_scale_frac": 0.10,
        },

        "phase3": {
            "scale_range_frac": 0.02,
            "scale_step_frac": 0.005,
            "shift_range_frac": 0.03,
            "shift_step_frac": 0.01,
        },
    }

    data = cli_params_block(pair)

    stage4b = data.get("stage4b", {})

    # -------- flat params --------
    params["anchor_mode"] = stage4b_modes.mode_from_block(stage4b)
    cli_mode = os.environ.get(stage4b_modes.ANCHOR_MODE_ENV_VAR, "").strip()
    if cli_mode:
        params["anchor_mode"] = stage4b_modes.normalize_mode(cli_mode)
        print(
            f"[INFO] stage4b.anchor_mode overridden by run_pharaoh CLI: "
            f"{params['anchor_mode']}",
            flush=True,
        )
    if "score_metric" in stage4b:
        print("[WARNING] stage4b.score_metric is no longer read: the anchor level "
              "decides the score (nuclei_pairs -> binary Dice, density -> density NCC)",
              flush=True)
    for k in ("good_nuclei_min", "min_good_tiles", "min_fallback_tiles",
              "pair_top_k", "pairs_to_take_per_tile", "n_tiles",
              "architecture_grid_size", "architecture_max_anchors_per_tile",
              "density_min_anchors_per_tile"):
        if k in stage4b:
            params[k] = int(stage4b[k])
    for k in ("density_sigma_um", "density_sigma_px",
              "density_min_foreground_fraction",
              "density_rich_tissue_quantile", "density_max_foreground_ratio",
              "density_min_largest_component_fraction",
              "architecture_min_region_mass_fraction",
              "architecture_min_region_ncc",
              "architecture_min_anchor_distance_fraction",
              "density_min_full_ncc", "density_min_anchor_median_ncc",
              "density_scale_range_frac", "density_shift_frac",
              "density_shift_step_frac", "density_max_shift_fraction"):
        if k in stage4b:
            params[k] = float(stage4b[k])
    # Backward compatibility with the initial descriptive key used during
    # development of this split.
    if "n_tiles" not in stage4b and "n_tiles_for_nuclei_detection" in stage4b:
        params["n_tiles"] = int(stage4b["n_tiles_for_nuclei_detection"])

    if "fallback_score_thr" in stage4b:
        params["fallback_score_thr"] = float(stage4b["fallback_score_thr"])

    # -------- nested phases --------
    for phase in ("phase1", "phase2", "phase3"):
        if phase in stage4b:
            for k, v in stage4b[phase].items():
                params[phase][k] = v
    phase1_config = stage4b.get("phase1", {})
    if (
        isinstance(phase1_config, dict)
        and "n_tiles_for_scale_estimation" not in phase1_config
        and "n_tiles" in phase1_config
    ):
        params["phase1"]["n_tiles_for_scale_estimation"] = int(phase1_config["n_tiles"])

    if params["n_tiles"] is not None and int(params["n_tiles"]) < 1:
        raise ValueError("stage4b n_tiles must be >= 1")
    if not 0.0 <= params["density_min_foreground_fraction"] <= 1.0:
        raise ValueError("stage4b density_min_foreground_fraction must be in [0, 1]")
    if not 0.0 <= params["density_rich_tissue_quantile"] <= 1.0:
        raise ValueError("stage4b density_rich_tissue_quantile must be in [0, 1]")
    if params["density_max_foreground_ratio"] < 1.0:
        raise ValueError("stage4b density_max_foreground_ratio must be >= 1")
    if not 0.0 <= params["density_min_largest_component_fraction"] <= 1.0:
        raise ValueError(
            "stage4b density_min_largest_component_fraction must be in [0, 1]"
        )
    if params["density_sigma_um"] <= 0:
        raise ValueError("stage4b density_sigma_um must be > 0")
    if params["density_min_anchors_per_tile"] < 1:
        raise ValueError("stage4b density_min_anchors_per_tile must be >= 1")
    if not -1.0 <= params["density_min_full_ncc"] <= 1.0:
        raise ValueError("stage4b density_min_full_ncc must be in [-1, 1]")
    if not -1.0 <= params["density_min_anchor_median_ncc"] <= 1.0:
        raise ValueError(
            "stage4b density_min_anchor_median_ncc must be in [-1, 1]"
        )
    for key in (
        "density_scale_range_frac", "density_shift_frac",
        "density_shift_step_frac", "density_max_shift_fraction",
    ):
        if not 0.0 < params[key] < 1.0:
            raise ValueError(f"stage4b {key} must be in (0, 1)")

    if params["pairs_to_take_per_tile"] > params["pair_top_k"]:
        print(
            f"[WARNING] pairs_to_take_per_tile ({params['pairs_to_take_per_tile']}) "
            f"> pair_top_k ({params['pair_top_k']}); effective output per tile is limited by pair_top_k.",
            flush=True
        )

    return params


def resolve_level(params, level=None):
    """Pin the level this process runs and the score it searches with.

    `level` is the launcher's `--anchor-mode`; without it, the configured mode's
    first rung (`auto` starts at nuclei pairs). `auto` itself is not a level.
    """
    if level is None:
        level = stage4b_modes.first_level(params.get("anchor_mode", stage4b_modes.DEFAULT_MODE))
    else:
        level = stage4b_modes.normalize_mode(level)
    if level not in stage4b_modes.LEVELS:
        raise ValueError(
            f"--anchor-mode must name a level, one of {', '.join(stage4b_modes.LEVELS)}"
        )
    params["level"] = level
    params["score_metric"] = stage4b_modes.SCORE_METRIC[level]
    if level == "density":
        # These are density-only effective settings.  The nuclei-pairs dictionaries
        # and execution path remain byte-for-byte unchanged.
        params["phase2"] = dict(params["phase2"])
        params["phase2"].update({
            "median_scale_frac": float(params["density_scale_range_frac"]),
            "shift_frac": float(params["density_shift_frac"]),
            "shift_step_frac": float(params["density_shift_step_frac"]),
        })
    return params

# -----------------------------
# Orientation / coordinate transform helpers
# -----------------------------
def inverse_orientation_point(x, y, H, W, case_id):
    if case_id == 0:
        return x, y
    if case_id == 1:
        return y, W - 1 - x
    if case_id == 2:
        return W - 1 - x, H - 1 - y
    if case_id == 3:
        return H - 1 - y, x
    if case_id == 4:
        return x, H - 1 - y
    if case_id == 5:
        return W - 1 - x, y
    if case_id == 6:
        return y, x
    if case_id == 7:
        return W - 1 - y, H - 1 - x
    raise ValueError(f"Unknown orientation case: {case_id}")

def apply_orientation_to_tile(img, case_id):
    """
    img: np.ndarray (H,W) or (H,W,3)
    case_id: int in [0..7]
    """
    if case_id == 0:
        return img
    if case_id == 1:      # rot90 CW
        return np.rot90(img, k=3)
    if case_id == 2:      # rot180
        return np.rot90(img, k=2)
    if case_id == 3:      # rot90 CCW
        return np.rot90(img, k=1)
    if case_id == 4:      # flip vertical
        return np.flipud(img)
    if case_id == 5:      # flip horizontal
        return np.fliplr(img)
    if case_id == 6:      # rot90 CW + flip H (transpose)
        if img.ndim == 3:
            return np.transpose(np.rot90(img, k=3), (1, 0, 2))
        else:
            return np.transpose(np.rot90(img, k=3))
    if case_id == 7:      # rot90 CW + flip V
        return np.flipud(np.rot90(img, k=3))

    raise ValueError(f"Unknown orientation case: {case_id}")

def mask_to_tile(x_mask, mask_scale):
    """Tile coordinate of a mask coordinate. Stage 4a upsamples each tile by
    ``mask_scale`` pixel-centre aligned, so mask pixel m covers tile (m + 0.5) / s - 0.5."""
    return (float(x_mask) + 0.5) / float(mask_scale) - 0.5


def nearest_resize_source(x_scaled, raw_len, base_scale):
    """Source coordinate of a position in a mask that scale_and_pad resized.

    cv2.resize with INTER_NEAREST shows source pixel floor(x * k), k = raw/resized.
    With k = p/q in lowest terms, floor(x * k) lies on average (q - 1) / (2q)
    below x * k, so a centroid measured in the resized mask maps back to
    k * x - (q - 1) / (2q): 0 at k = 1, 0.25 at k = 1/2, nearly 0.5 for a general k.
    """
    new_len = int(round(raw_len * base_scale))       # the size scale_and_pad resized to
    k = Fraction(int(raw_len), new_len)
    return float(k) * x_scaled - (k.denominator - 1) / (2.0 * k.denominator)


def aligned_to_he_mask_img(x_aligned, y_aligned,
                           final, H, W,
                           paste_x0, paste_y0,
                           base_scale, raw_shape):
    """
    x_aligned, y_aligned: in aligned canvas (same as maskA space)
    final: dict with {"scale","tx","ty"} applied to maskB (scaled+pad canvas) -> aligned
    paste_x0,y0: where scaled HE mask was pasted into canvas
    base_scale: scale used to create scaled HE mask from original HE mask

    Returns:
      x_he_mask, y_he_mask in ORIGINAL HE mask pixel coordinates (maskB_raw coords)
    """
    # 1) invert final warp: aligned -> (scaled+pad canvas) coords
    s = float(final["scale"])
    tx = float(final["tx"])
    ty = float(final["ty"])

    # 1) invert the center-based scale+translation used by warp().
    cx = W // 2
    cy = H // 2
    x_canvas = (x_aligned - tx - cx) / s + cx
    y_canvas = (y_aligned - ty - cy) / s + cy
    # 2) remove padding offset: canvas -> scaled HE mask coords
    x_scaled = x_canvas - paste_x0
    y_scaled = y_canvas - paste_y0
    # 3) undo base scale: scaled HE mask -> original HE mask coords, inverting the
    # nearest-neighbour resize exactly per axis (raw_shape = original mask shape)
    x_raw = nearest_resize_source(x_scaled, raw_shape[1], base_scale)
    y_raw = nearest_resize_source(y_scaled, raw_shape[0], base_scale)

    return float(x_raw), float(y_raw)

def dapi_aligned_to_original(xA_aligned, yA_aligned, H, W, case_id, dapi_mask_scale):
    # aligned(DAPI after apply_orientation) -> original DAPI mask coords
    xA_dapi, yA_dapi = inverse_orientation_point(xA_aligned, yA_aligned, H, W, case_id)
    # original DAPI tile coords: the mask is the tile upsampled pixel-centre aligned
    # (skimage rescale), so mask pixel m sits at tile coordinate (m + 0.5) / s - 0.5
    return (float(mask_to_tile(xA_dapi, dapi_mask_scale)),
            float(mask_to_tile(yA_dapi, dapi_mask_scale)))

def he_aligned_to_original(xB_aligned, yB_aligned, final, H, W, paste_x0, paste_y0, base_scale, he_mask_scale,
                           raw_shape):
    # aligned -> original HE mask coords
    xB_mask, yB_mask = aligned_to_he_mask_img(
        xB_aligned, yB_aligned,
        final, H, W,
        paste_x0, paste_y0,
        base_scale, raw_shape
    )
    # original HE tile coords, with the same pixel-centre convention as the DAPI side
    return float(mask_to_tile(xB_mask, he_mask_scale)), float(mask_to_tile(yB_mask, he_mask_scale))

# -----------------------------
# Basic mask / warp / score helpers
# -----------------------------
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img < 128).astype(np.uint8)

def scale_and_pad(mask, target_shape, interpolation=cv2.INTER_NEAREST):
    H, W = target_shape
    h, w = mask.shape
    # --- 1) isotropic scale so mask fits inside target ---
    scale = min(H / h, W / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    mask_scaled = cv2.resize(
        mask, (new_w, new_h), interpolation=interpolation
    )
    # --- 2) center pad (no crop) ---
    out = np.zeros((H, W), dtype=mask.dtype)
    paste_y0 = (H - new_h) // 2
    paste_x0 = (W - new_w) // 2
    # safety check (should never fail if scale=min(...))
    if paste_y0 < 0 or paste_x0 < 0:
        raise ValueError(
            f"Scaled mask {mask_scaled.shape} larger than target {target_shape}"
        )
    out[paste_y0:paste_y0 + new_h, paste_x0:paste_x0 + new_w] = mask_scaled
    return out, scale, paste_x0, paste_y0

def warp(mask, scale, tx, ty, out_shape, interpolation=cv2.INTER_NEAREST):
    H, W = out_shape
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, 0.0, float(scale))
    M[0, 2] += float(tx)
    M[1, 2] += float(ty)
    return cv2.warpAffine(mask, M, (W, H), flags=interpolation)

def dice_score(A, B):
    Afg = (A > 0)
    Bfg = (B > 0)
    inter = np.sum(Afg & Bfg)
    sizeA = np.sum(Afg)
    sizeB = np.sum(Bfg)
    return 2.0 * float(inter) / float(sizeA + sizeB + 1e-6)

def dice_score_integer_translation(Afg, Bfg, tx, ty, sizeA=None):
    """Score an integer translation without allocating a warped mask.

    This is pixel-equivalent to ``dice_score(A, warp(B, 1, tx, ty, A.shape))``
    for the integer shifts used by the grid searches.  Only the overlapping
    source/destination slices are inspected; foreground shifted outside the
    canvas is excluded exactly as it is by ``cv2.warpAffine``.
    """
    Afg = np.asarray(Afg, dtype=bool)
    Bfg = np.asarray(Bfg, dtype=bool)
    if Afg.shape != Bfg.shape:
        raise ValueError(f"mask shape mismatch: {Afg.shape} != {Bfg.shape}")

    H, W = Afg.shape
    tx = int(tx)
    ty = int(ty)
    if sizeA is None:
        sizeA = int(np.count_nonzero(Afg))

    dst_x0 = max(0, tx)
    dst_x1 = min(W, W + tx)
    dst_y0 = max(0, ty)
    dst_y1 = min(H, H + ty)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return 0.0

    src_x0 = dst_x0 - tx
    src_x1 = dst_x1 - tx
    src_y0 = dst_y0 - ty
    src_y1 = dst_y1 - ty

    A_overlap = Afg[dst_y0:dst_y1, dst_x0:dst_x1]
    B_overlap = Bfg[src_y0:src_y1, src_x0:src_x1]
    inter = int(np.count_nonzero(A_overlap & B_overlap))
    sizeB = int(np.count_nonzero(B_overlap))
    return 2.0 * float(inter) / float(int(sizeA) + sizeB + 1e-6)

# -----------------------------
# Serial-section scoring: nuclear-density fields
#
# Adjacent sections share no nucleus, so overlap of individual components says
# nothing about their alignment. What they do share is architecture -- where the
# nuclei crowd and where they thin -- and a blurred mask measures exactly that.
# Pearson correlation of the two density fields is the score; the cost is one
# warp per translation instead of the slice-only Dice, so this path is slower
# and is the `density` level's alone: `auto` reaches it only after nuclei
# pairs, topped up, came up short.
# -----------------------------
def nuclear_density(mask, sigma_px=8.0):
    """Convert discrete nuclei to a continuous local-density field."""
    field = (np.asarray(mask) > 0).astype(np.float32)
    sigma_px = float(sigma_px)
    if sigma_px <= 0:
        return field
    return cv2.GaussianBlur(
        field, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px,
        # A zero border creates the same dark frame in both fields. At architecture
        # scales that shared frame can dominate Pearson NCC and manufacture an
        # attractive score unrelated to tissue. Reflection keeps the score driven by
        # tissue rather than by a shared tile boundary.
        borderType=cv2.BORDER_REFLECT_101,
    )


def density_ncc(A, B, support_fraction=0.01):
    """Pearson NCC over the union of non-trivial nuclear-density support."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    amax = float(A.max()) if A.size else 0.0
    bmax = float(B.max()) if B.size else 0.0
    if amax <= 0 or bmax <= 0:
        return -1.0
    support = (A > support_fraction * amax) | (B > support_fraction * bmax)
    if int(support.sum()) < 16:
        return -1.0
    a = A[support].astype(np.float64)
    b = B[support].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return -1.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _density_tissue_measurements(maskA, maskB):
    """Measure whether a mask pair contains coherent, information-rich tissue.

    Foreground amount alone is not enough: a tile filled with many unrelated tiny
    fragments can still have a respectable coverage.  The largest connected
    foreground component, expressed as a fraction of all foreground, supplies the
    complementary architecture check.  This is intentionally a pre-alignment
    measurement so an attractive NCC peak cannot make a poor tile look eligible.
    """
    measurements = {}
    foreground = []
    largest = []
    for mask in (maskA, maskB):
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        count = int(np.count_nonzero(binary))
        fraction = float(count) / float(max(1, binary.size))
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count > 0 and n_labels > 1:
            largest_fraction = float(stats[1:, cv2.CC_STAT_AREA].max()) / float(count)
        else:
            largest_fraction = 0.0
        foreground.append(fraction)
        largest.append(largest_fraction)

    lo = min(foreground)
    hi = max(foreground)
    measurements.update({
        "foreground_fraction_a": float(foreground[0]),
        "foreground_fraction_b": float(foreground[1]),
        "foreground_ratio": float(hi / max(lo, 1e-12)),
        "largest_component_fraction_a": float(largest[0]),
        "largest_component_fraction_b": float(largest[1]),
    })
    return measurements


def _apply_density_tissue_gate(measurements, params):
    """Apply the density-only rich-tissue thresholds to precomputed measurements."""
    quality = dict(measurements)
    min_foreground = float(params.get(
        "density_effective_min_foreground_fraction",
        params.get("density_min_foreground_fraction", 0.02),
    ))
    max_ratio = float(params.get("density_max_foreground_ratio", 4.0))
    min_component = float(params.get(
        "density_min_largest_component_fraction", 0.03
    ))
    reasons = []
    for side in ("a", "b"):
        value = quality[f"foreground_fraction_{side}"]
        if value < min_foreground:
            reasons.append(f"foreground_{side}={value:.3f}<{min_foreground:.3f}")
    if quality["foreground_ratio"] > max_ratio:
        reasons.append(
            f"foreground_ratio={quality['foreground_ratio']:.2f}>{max_ratio:.2f}"
        )
    for side in ("a", "b"):
        value = quality[f"largest_component_fraction_{side}"]
        if value < min_component:
            reasons.append(f"component_{side}={value:.3f}<{min_component:.3f}")

    quality.update({
        "min_foreground_fraction_required": min_foreground,
        "max_foreground_ratio_allowed": max_ratio,
        "min_largest_component_fraction_required": min_component,
        "passes": not reasons,
        "rejection_reasons": reasons,
    })
    return quality


def density_tissue_quality(maskA, maskB, params):
    """Return density-level tissue quality; the nuclei-pairs level never calls it."""
    return _apply_density_tissue_gate(
        _density_tissue_measurements(maskA, maskB), params
    )


def calibrate_density_tissue_gate(dapi_files, base_dir, case_id, params):
    """Calibrate the rich-tissue floor from this run and cache every tile's result.

    Absolute nuclear coverage is modality- and dataset-dependent.  Taking the richer
    half of the available tiles makes the density level local-tissue-led without
    hard-coding the H&E coverage seen in one run.  The absolute floor still protects
    a uniformly empty run, while the component and side-balance gates remain fixed.
    """
    if params.get("level") != "density":
        return {}

    measurements_by_tile = {}
    for moving_path in dapi_files:
        prefix = prefix_of(moving_path)
        fixed_path = os.path.join(base_dir, mask_name(prefix, "fixed"))
        if not os.path.exists(fixed_path):
            continue
        maskA = apply_orientation_to_tile(read_mask(moving_path), case_id)
        maskB_raw = read_mask(fixed_path)
        maskB, _, _, _ = scale_and_pad(maskB_raw, maskA.shape)
        measurements_by_tile[prefix] = _density_tissue_measurements(maskA, maskB)

    absolute_floor = float(params.get("density_min_foreground_fraction", 0.02))
    quantile = float(params.get("density_rich_tissue_quantile", 0.50))
    coverages = [
        min(m["foreground_fraction_a"], m["foreground_fraction_b"])
        for m in measurements_by_tile.values()
    ]
    adaptive_floor = float(np.quantile(coverages, quantile)) if coverages else 0.0
    effective_floor = max(absolute_floor, adaptive_floor)
    params["density_effective_min_foreground_fraction"] = effective_floor
    quality_by_tile = {
        prefix: _apply_density_tissue_gate(measurements, params)
        for prefix, measurements in measurements_by_tile.items()
    }
    params["_density_tissue_quality_by_tile"] = quality_by_tile
    rich = sum(bool(quality["passes"]) for quality in quality_by_tile.values())
    print(
        f"[INFO] Density rich-tissue gate: {rich}/{len(quality_by_tile)} tiles pass; "
        f"foreground floor=max({absolute_floor:.3f}, q{quantile:.2f}="
        f"{adaptive_floor:.3f})={effective_floor:.3f}, "
        f"side ratio<={params.get('density_max_foreground_ratio', 4.0):.2f}, "
        f"largest component>="
        f"{params.get('density_min_largest_component_fraction', 0.03):.3f}",
        flush=True,
    )
    return quality_by_tile


def density_acceptance(
    tissue_quality,
    final_full_score,
    anchors,
    params,
    *,
    alignment=None,
    mask_shape=None,
):
    """Assess one serial-section tile from local architecture evidence.

    Whole-tile density NCC is intentionally only a weak sanity gate. Adjacent
    sections can conserve several local structures while differing elsewhere in the
    tile, so the old same-section-style ``NCC > fallback_score_thr`` rule threw away
    the very local anchors this level exists to find.
    """
    anchors = list(anchors or [])
    local_ncc = []
    for anchor in anchors:
        metrics = anchor.get("metrics", anchor) if isinstance(anchor, dict) else {}
        value = metrics.get("local_density_ncc")
        if value is not None and np.isfinite(float(value)):
            local_ncc.append(float(value))

    tissue_passes = (
        True if tissue_quality is None else bool(tissue_quality.get("passes", False))
    )
    min_full_ncc = float(params.get("density_min_full_ncc", 0.05))
    min_anchors = int(params.get("density_min_anchors_per_tile", 2))
    min_median_ncc = float(params.get("density_min_anchor_median_ncc", 0.15))
    median_ncc = float(np.median(local_ncc)) if local_ncc else -1.0

    reasons = []
    if not tissue_passes:
        reasons.append("tissue_gate")
    if float(final_full_score) < min_full_ncc:
        reasons.append(f"full_ncc={float(final_full_score):.3f}<{min_full_ncc:.3f}")
    if len(anchors) < min_anchors:
        reasons.append(f"anchors={len(anchors)}<{min_anchors}")
    if median_ncc < min_median_ncc:
        reasons.append(f"median_local_ncc={median_ncc:.3f}<{min_median_ncc:.3f}")

    shift_fraction = None
    scale_relative_error = None
    if alignment is not None and mask_shape is not None:
        short_side = float(min(mask_shape))
        shift_fraction = max(
            abs(float(alignment.get("tx", 0.0))),
            abs(float(alignment.get("ty", 0.0))),
        ) / max(short_side, 1.0)
        max_shift_fraction = float(params.get("density_max_shift_fraction", 0.24))
        if shift_fraction > max_shift_fraction:
            reasons.append(
                f"shift_fraction={shift_fraction:.3f}>{max_shift_fraction:.3f}"
            )
        expected_scale = params.get("density_expected_scale")
        if expected_scale is not None:
            scale_relative_error = abs(
                float(alignment.get("scale", expected_scale)) / float(expected_scale) - 1.0
            )
            max_scale_error = float(params.get("density_scale_range_frac", 0.05)) * 1.5
            if scale_relative_error > max_scale_error:
                reasons.append(
                    f"scale_relative_error={scale_relative_error:.3f}>"
                    f"{max_scale_error:.3f}"
                )

    return {
        "passes": not reasons,
        "rejection_reasons": reasons,
        "full_ncc": float(final_full_score),
        "min_full_ncc": min_full_ncc,
        "n_anchors": int(len(anchors)),
        "min_anchors": min_anchors,
        "median_local_ncc": median_ncc,
        "min_median_local_ncc": min_median_ncc,
        "shift_fraction": shift_fraction,
        "max_shift_fraction": float(params.get("density_max_shift_fraction", 0.24)),
        "scale_relative_error": scale_relative_error,
    }


def density_result_is_good(result, score_thr=None, params=None):
    """The one density tile-selection rule used by progress, QC and final output."""
    acceptance = result.get("density_acceptance")
    if acceptance is not None:
        return bool(acceptance.get("passes", False))

    # Compatibility for callers/results created before density_acceptance was saved.
    if params is None:
        quality = result.get("density_tissue_quality")
        rich_tissue = True if quality is None else bool(quality.get("passes", False))
        return (
            rich_tissue
            and result.get("n_architecture_anchors", 0) > 0
            and result.get("final_full_score", -1.0) > float(score_thr)
        )
    acceptance = density_acceptance(
        result.get("density_tissue_quality"),
        result.get("final_full_score", -1.0),
        result.get("architecture_anchors_mapped", []),
        params,
        alignment=result.get("final"),
        mask_shape=(result.get("H", 0), result.get("W", 0)),
    )
    return bool(acceptance["passes"])


def check_score_metric(score_metric):
    if score_metric not in ("binary_dice", "density_ncc"):
        raise ValueError(f"Unknown Stage-4b score metric: {score_metric!r}")
    return score_metric


def prepare_score_fields(maskA, maskB, score_metric, density_sigma_px):
    """The two fields a metric compares, and how a warp should interpolate them."""
    if check_score_metric(score_metric) == "binary_dice":
        return (
            (np.asarray(maskA) > 0).astype(np.uint8),
            (np.asarray(maskB) > 0).astype(np.uint8),
            cv2.INTER_NEAREST,
        )
    return (
        nuclear_density(maskA, density_sigma_px),
        nuclear_density(maskB, density_sigma_px),
        cv2.INTER_LINEAR,
    )


def prepared_similarity(A, B, score_metric):
    if check_score_metric(score_metric) == "binary_dice":
        return dice_score(A, B)
    return density_ncc(A, B)


def score_full(maskA, maskB, params, score_metric="binary_dice", density_sigma_px=8.0):
    H, W = maskA.shape
    if check_score_metric(score_metric) == "binary_dice":
        # The path every `dapi-he` run has taken: the raw mask, warped nearest.
        B_w = warp(maskB, float(params["scale"]), int(params["tx"]), int(params["ty"]), (H, W))
        return dice_score(maskA, B_w)
    A_score, B_score, interpolation = prepare_score_fields(
        maskA, maskB, score_metric, density_sigma_px
    )
    B_w = warp(
        B_score, float(params["scale"]), int(params["tx"]), int(params["ty"]),
        (H, W), interpolation=interpolation,
    )
    return prepared_similarity(A_score, B_w, score_metric)

def downsample(mask, ds, interpolation=cv2.INTER_NEAREST):
    H, W = mask.shape
    return cv2.resize(mask, (max(1, W // ds), max(1, H // ds)),
                      interpolation=interpolation)

# -----------------------------
# Phase 1: median scale estimation
# -----------------------------
def robust_median_scale_from_firstN(
    dapi_files,
    base_dir,
    case_id,
    N=20,
    ds=4,
    scale_range=(0.8, 1.8),
    scale_step=0.05,
    shift_frac=0.8,
    shift_step_frac=0.05,
    score_metric="binary_dice",
    density_sigma_px=8.0,
):
    """Run coarse on the first N tiles and return the median scale."""
    def _fmt_sec(sec):
        sec = int(round(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"

    scales = []
    use = min(N, len(dapi_files))
    t_phase1_start = time.perf_counter()

    print(f"[INFO] Phase-1: coarse on first {use}/{len(dapi_files)} tiles to estimate median scale", flush=True)

    for i in range(use):
        dapi_path = dapi_files[i]
        fname = os.path.basename(dapi_path)
        prefix = prefix_of(fname)
        he_path = os.path.join(base_dir, mask_name(prefix, "fixed"))
        done = i + 1
        if not os.path.exists(he_path):
            elapsed = time.perf_counter() - t_phase1_start
            avg = elapsed / done if done > 0 else 0.0
            eta = avg * (use - done)

            print(f"[PROGRESS] TILES {done}/{use}", flush=True)
            print(
                f"[INFO] Phase-1 {prefix}: skipped (HE mask not found) | "
                f"Elapsed: {_fmt_sec(elapsed)} | ETA: {_fmt_sec(eta)}",
                flush=True,
            )
            continue

        maskA = read_mask(dapi_path)
        maskA = apply_orientation_to_tile(maskA, case_id)
        maskB_raw = read_mask(he_path)
        maskB, _, _, _ = scale_and_pad(maskB_raw, maskA.shape)

        coarse = coarse_search_ds_parallel_cached(
            maskA,
            maskB,
            ds=ds,
            scale_range=scale_range,
            scale_step=scale_step,
            shift_frac=shift_frac,
            shift_step_frac=shift_step_frac,
            score_metric=score_metric,
            density_sigma_px=density_sigma_px,
        )
        s = float(coarse["scale"])
        scales.append(s)

        elapsed = time.perf_counter() - t_phase1_start
        avg = elapsed / done if done > 0 else 0.0
        eta = avg * (use - done)

        print(f"[PROGRESS] TILES {done}/{use}", flush=True)
        print(
            f"[INFO] Phase-1 {prefix}: "
            f"Elapsed: {_fmt_sec(elapsed)} | ETA: {_fmt_sec(eta)}",
            flush=True,
        )

    if len(scales) == 0:
        med = 0.5 * (scale_range[0] + scale_range[1])
        print(f"[WARN] Phase-1 got 0 valid scales, fallback median={med:.4f}", flush=True)
        return float(med)

    med = float(np.median(np.asarray(scales, dtype=np.float32)))
    total_elapsed = time.perf_counter() - t_phase1_start
    print(
        f"[INFO] Phase-1 median scale from {len(scales)} tiles = {med:.4f} | "
        f"Elapsed: {_fmt_sec(total_elapsed)}",
        flush=True,
    )
    return med

def make_scale_range_around_median(med, frac=0.15, hard_min=0.5, hard_max=3.0):
    """
    Return (lo, hi) where lo=med*(1-frac), hi=med*(1+frac), clamped.
    frac=0.15 means ±15%.
    """
    lo = max(hard_min, med * (1.0 - frac))
    hi = min(hard_max, med * (1.0 + frac))
    return float(lo), float(hi)

# ----------------------------------------------------------
# Phase 2 / 3: tile alignment
# ----------------------------------------------------------
def coarse_search_ds_parallel_cached(
    maskA, maskB,
    ds=4,
    scale_range=(0.9, 1.8),
    scale_step=0.04,
    shift_frac=0.5,
    shift_step_frac=0.05,
    n_jobs=None,
    score_metric="binary_dice",
    density_sigma_px=8.0,
):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)
    check_score_metric(score_metric)
    dice = score_metric == "binary_dice"

    # ---- ds-space fields. Dice keeps the raw masks, nearest-sampled, exactly as
    #      before. Density NCC blurs first and area-averages down, so the search
    #      rewards architecture rather than component overlap. ----
    if dice:
        A = downsample(maskA, ds)
        B = downsample(maskB, ds)
        interpolation = cv2.INTER_NEAREST
    else:
        A_field, B_field, interpolation = prepare_score_fields(
            maskA, maskB, score_metric, density_sigma_px
        )
        A = downsample(A_field, ds, interpolation=cv2.INTER_AREA)
        B = downsample(B_field, ds, interpolation=cv2.INTER_AREA)
    Hds, Wds = A.shape

    # ---- compute shift range from fraction (FULL-RES semantics) ----
    H0, W0 = maskA.shape
    L = min(H0, W0)

    shift_range_px = int(round(shift_frac * L))
    shift_step_px = max(1, int(round(shift_step_frac * L)))

    shift_range_ds = max(1, shift_range_px // ds)
    shift_step_ds = max(1, shift_step_px // ds)

    shifts = list(range(-shift_range_ds, shift_range_ds + 1, shift_step_ds))

    # ---- scales ----
    scales = np.arange(scale_range[0], scale_range[1] + 1e-9, scale_step)

    # ---- cache scaled B (clipped) ----
    B_cache = {
        float(s): warp(B, float(s), 0, 0, (Hds, Wds), interpolation=interpolation)
        for s in scales
    }
    A_fg = A > 0
    sizeA = int(np.count_nonzero(A_fg))

    def eval_one_scale(s):
        B_s = B_cache[float(s)]
        B_s_fg = B_s > 0

        best_score = -1.0
        best_tx_ds = 0
        best_ty_ds = 0

        for tx_ds in shifts:
            for ty_ds in shifts:
                if dice:
                    score = dice_score_integer_translation(
                        A_fg, B_s_fg, tx_ds, ty_ds, sizeA=sizeA
                    )
                else:
                    B_st = warp(B_s, 1.0, tx_ds, ty_ds, (Hds, Wds),
                                interpolation=interpolation)
                    score = prepared_similarity(A, B_st, score_metric)

                if score > best_score:
                    best_score = score
                    best_tx_ds = tx_ds
                    best_ty_ds = ty_ds

        return best_score, float(s), int(best_tx_ds), int(best_ty_ds)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(eval_one_scale)(s) for s in scales
    )

    best_score, best_s, best_tx_ds, best_ty_ds = max(results, key=lambda x: x[0])

    return {
        "scale": best_s,
        "tx": int(best_tx_ds * ds),
        "ty": int(best_ty_ds * ds),
        "score_ds": float(best_score),
        "shift_frac": shift_frac,
        "shift_step_frac": shift_step_frac,
    }

def refine_full(
    maskA, maskB, init,
    scale_range_frac=0.02,
    scale_step_frac=0.005,
    shift_range_frac=0.05,
    shift_step_frac=0.01,
    n_jobs=None,
    score_metric="binary_dice",
    density_sigma_px=8.0,
):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)
    check_score_metric(score_metric)
    dice = score_metric == "binary_dice"

    H, W = maskA.shape
    L = min(H, W)

    s0 = float(init["scale"])
    tx0 = int(round(init["tx"]))
    ty0 = int(round(init["ty"]))

    # -----------------------------
    # Scale grid
    # -----------------------------
    n_scale = int(np.floor(scale_range_frac / scale_step_frac))
    rel_factors = np.linspace(
        1.0 - n_scale * scale_step_frac,
        1.0 + n_scale * scale_step_frac,
        2 * n_scale + 1
    )
    scales = [float(s0 * r) for r in rel_factors]

    # -----------------------------
    # Shift grid
    # -----------------------------
    shift_range_px = int(round(shift_range_frac * L))
    shift_step_px = max(1, int(round(shift_step_frac * L)))

    shifts_x = list(range(tx0 - shift_range_px, tx0 + shift_range_px + 1, shift_step_px))
    shifts_y = list(range(ty0 - shift_range_px, ty0 + shift_range_px + 1, shift_step_px))

    # -----------------------------
    # Cache scaled B
    # -----------------------------
    if dice:
        A_score, B_score, interpolation = maskA, maskB, cv2.INTER_NEAREST
    else:
        A_score, B_score, interpolation = prepare_score_fields(
            maskA, maskB, score_metric, density_sigma_px
        )
    B_scale_cache = {
        s: warp(B_score, s, 0, 0, (H, W), interpolation=interpolation) for s in scales
    }
    A_fg = A_score > 0
    sizeA = int(np.count_nonzero(A_fg))

    def eval_one_scale(s):
        B_s = B_scale_cache[s]
        B_s_fg = B_s > 0

        best_score = -1.0
        best_tx = tx0
        best_ty = ty0

        for tx in shifts_x:
            for ty in shifts_y:
                if dice:
                    score = dice_score_integer_translation(
                        A_fg, B_s_fg, tx, ty, sizeA=sizeA
                    )
                else:
                    B_st = warp(B_s, 1.0, tx, ty, (H, W), interpolation=interpolation)
                    score = prepared_similarity(A_score, B_st, score_metric)

                if score > best_score:
                    best_score = score
                    best_tx = tx
                    best_ty = ty

        return best_score, float(s), int(best_tx), int(best_ty)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(eval_one_scale)(s) for s in scales
    )

    score, s, tx, ty = max(results, key=lambda x: x[0])

    return {
        "scale": float(s),
        "tx": int(tx),
        "ty": int(ty),
        "score": float(score),
        "scale_range_frac": float(scale_range_frac),
        "scale_step_frac": float(scale_step_frac),
        "shift_range_frac": float(shift_range_frac),
        "shift_step_frac": float(shift_step_frac),
    }


# -----------------------------------------
# Isolated nuclei detection / pairing
# -----------------------------------------
def find_isolated_components(mask, min_area=10, radii=(3,5,7), area_iqr_k=1.5):
    H, W = mask.shape
    x_lo, x_hi = W/6, 5*W/6
    y_lo, y_hi = H/6, 5*H/6

    labeled, _ = ndimage.label(mask > 0)
    objects = ndimage.find_objects(labeled)

    candidates = []

    for lab, slc in enumerate(objects, start=1):
        comp = (labeled[slc] == lab)
        area = comp.sum()
        if area < min_area:
            continue

        ys, xs = np.where(comp)
        cy = ys.mean() + slc[0].start
        cx = xs.mean() + slc[1].start
        if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
            continue

        full_local = (mask[slc] > 0)
        isolated = True
        for r in radii:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
            dil = cv2.dilate(comp.astype(np.uint8), k)
            ring = dil & (~comp)
            if np.any(ring & full_local & (~comp)):
                isolated = False
                break

        if isolated:
            candidates.append({
                "comp": comp,
                "bbox": slc,
                "area": area,
                "centroid": (cx, cy)
            })

    if not candidates:
        return []

    areas = np.array([c["area"] for c in candidates])
    q1, q3 = np.percentile(areas, [25, 75])
    max_area = q3 + area_iqr_k*(q3-q1)

    return [c for c in candidates if c["area"] <= max_area]

def touch(comp_bool, other_bool, radius=3):
    """
    comp_bool: bool mask (ROI)
    other_bool: bool mask (ROI)
    """
    comp_u8 = comp_bool.astype(np.uint8)
    other_u8 = other_bool.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dil = cv2.dilate(comp_u8, k)
    return np.any((dil > 0) & (other_u8 > 0))

def pair_isolated_cells(
    A_cells,
    B_cells,
    A_iso_mask,
    B_iso_mask,
    min_area=60,
    min_sym_cov=0.4,
    top_k=3,
    debug=False,
    touch_radius=3
):
    H, W = A_iso_mask.shape
    results = []

    # debug counters
    n_total = 0
    n_bbox = 0
    n_inter = 0
    n_sym = 0
    n_touchA = 0
    n_touchB = 0

    for a in A_cells:
        if a["area"] < min_area:
            continue
        ay0, ay1 = a["bbox"][0].start, a["bbox"][0].stop
        ax0, ax1 = a["bbox"][1].start, a["bbox"][1].stop

        for b in B_cells:
            if b["area"] < min_area:
                continue
            n_total += 1

            by0, by1 = b["bbox"][0].start, b["bbox"][0].stop
            bx0, bx1 = b["bbox"][1].start, b["bbox"][1].stop

            # intersection bbox just for quick reject
            y0, y1 = max(ay0, by0), min(ay1, by1)
            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            if y0 >= y1 or x0 >= x1:
                continue
            n_bbox += 1

            # intersection overlap
            a_loc = a["comp"][y0-ay0:y1-ay0, x0-ax0:x1-ax0].astype(bool)
            b_loc = b["comp"][y0-by0:y1-by0, x0-bx0:x1-bx0].astype(bool)
            inter = np.sum(a_loc & b_loc)
            if inter == 0:
                continue
            n_inter += 1

            covA = inter / (a["area"] + 1e-6)
            covB = inter / (b["area"] + 1e-6)
            sym = min(covA, covB)
            if sym < min_sym_cov:
                continue
            n_sym += 1

            # Build a local ROI around the union of the two component boxes.
            pad = touch_radius + 2
            ry0 = max(0, min(ay0, by0) - pad)
            ry1 = min(H, max(ay1, by1) + pad)
            rx0 = max(0, min(ax0, bx0) - pad)
            rx1 = min(W, max(ax1, bx1) + pad)

            # roi masks for all isolated comps
            A_iso_roi = (A_iso_mask[ry0:ry1, rx0:rx1] > 0)
            B_iso_roi = (B_iso_mask[ry0:ry1, rx0:rx1] > 0)

            # build full a_roi and b_roi (NOT just intersection part)
            a_roi = np.zeros((ry1-ry0, rx1-rx0), dtype=bool)
            b_roi = np.zeros((ry1-ry0, rx1-rx0), dtype=bool)

            # paste A comp into ROI
            a_y0 = ay0 - ry0
            a_x0 = ax0 - rx0
            a_roi[a_y0:a_y0 + a["comp"].shape[0], a_x0:a_x0 + a["comp"].shape[1]] |= a["comp"].astype(bool)

            # paste B comp into ROI
            b_y0 = by0 - ry0
            b_x0 = bx0 - rx0
            b_roi[b_y0:b_y0 + b["comp"].shape[0], b_x0:b_x0 + b["comp"].shape[1]] |= b["comp"].astype(bool)

            # Exclude the current component itself. Keep everything in bool space.
            A_other = A_iso_roi & (~a_roi)
            B_other = B_iso_roi & (~b_roi)

            # bidirectional non-touch
            if touch(a_roi, B_other, radius=touch_radius):
                n_touchA += 1
                continue
            if touch(b_roi, A_other, radius=touch_radius):
                n_touchB += 1
                continue

            f1 = 2 * covA * covB / (covA + covB + 1e-6)

            results.append({
                "A_centroid": a["centroid"],
                "B_centroid": b["centroid"],
                "A_comp": a,
                "areaA": a["area"],
                "areaB": b["area"],
                "covA": float(covA),
                "covB": float(covB),
                "sym_cov": float(sym),
                "f1": float(f1)
            })

    results.sort(key=lambda d: d["f1"], reverse=True)

    if debug:
        print(f"  [DEBUG pairing] total={n_total}, bbox_ok={n_bbox}, inter>0={n_inter}, pass_sym={n_sym}, fail_touchA={n_touchA}, fail_touchB={n_touchB}, kept={len(results)}")

    return results[:top_k]


# -----------------------------------------
# Serial-section landmarks: conserved nuclear architecture
# -----------------------------------------
def architecture_grid_cells(
    maskA,
    aligned_maskB,
    density_sigma_px=8.0,
    grid_size=3,
    min_region_mass_fraction=0.04,
    min_region_ncc=0.10,
):
    """The density fields and every grid cell's statistics, whether or not it qualifies.

    This is the whole of what `nuclear_architecture_anchors` looks at; it is split out
    so the density QC image can show the cells that were rejected as well as the ones
    that became anchors. Returns (densityA, densityB, joint, cells); `joint` is None and
    `cells` empty when either side has no nuclei at all.
    """
    densityA = nuclear_density(maskA, density_sigma_px)
    densityB = nuclear_density(aligned_maskB, density_sigma_px)
    maxA = float(densityA.max()) if densityA.size else 0.0
    maxB = float(densityB.max()) if densityB.size else 0.0
    if maxA <= 0 or maxB <= 0:
        return densityA, densityB, None, []

    joint = np.sqrt(
        np.clip((densityA / maxA) * (densityB / maxB), 0.0, None)
    ).astype(np.float32)
    total_mass = float(joint.sum())
    if total_mass <= 1e-12:
        return densityA, densityB, joint, []

    H, W = joint.shape
    grid_size = max(1, int(grid_size))
    y_edges = np.linspace(0, H, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, W, grid_size + 1, dtype=int)
    cells = []
    for row in range(grid_size):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(grid_size):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            weights = joint[y0:y1, x0:x1]
            mass = float(weights.sum())
            mass_fraction = mass / total_mass
            passes_mass = not (mass_fraction < float(min_region_mass_fraction) or mass <= 1e-12)
            local_ncc = density_ncc(densityA[y0:y1, x0:x1], densityB[y0:y1, x0:x1])
            passes_ncc = local_ncc >= float(min_region_ncc)
            centroid = None
            if mass > 1e-12:
                yy, xx = np.indices(weights.shape, dtype=np.float64)
                centroid = (
                    float(x0 + np.sum(xx * weights) / mass),
                    float(y0 + np.sum(yy * weights) / mass),
                )
            ncc_weight = max(0.0, 0.5 * (local_ncc + 1.0))
            cells.append({
                "grid_cell": (int(row), int(col)),
                "bounds": (x0, y0, x1, y1),
                "density_mass_fraction": float(mass_fraction),
                "local_density_ncc": float(local_ncc),
                "centroid": centroid,
                "strength": float(mass_fraction * ncc_weight),
                "passes_mass": bool(passes_mass),
                "passes_ncc": bool(passes_ncc),
            })
    return densityA, densityB, joint, cells


def nuclear_architecture_anchors(
    maskA,
    aligned_maskB,
    density_sigma_px=8.0,
    grid_size=3,
    max_anchors=3,
    min_region_mass_fraction=0.04,
    min_region_ncc=0.10,
    min_anchor_distance_fraction=0.15,
    return_details=False,
):
    """Derive stable landmarks from locally conserved nuclear architecture.

    Both masks must be in the same aligned canvas. Each candidate is the
    density-weighted centroid of a grid cell, using the geometric mean of the
    two sides' normalised density fields. The product suppresses structures
    found in only one physical section. Candidates are ranked by conserved
    density mass and local NCC, then spatially de-duplicated.

    With `return_details` the fields and every cell come back too, for the QC image.
    """
    densityA, densityB, joint, cells = architecture_grid_cells(
        maskA, aligned_maskB, density_sigma_px, grid_size,
        min_region_mass_fraction, min_region_ncc,
    )
    details = {"densityA": densityA, "densityB": densityB, "joint": joint, "cells": cells}
    candidates = [
        {
            "centroid": cell["centroid"],
            "grid_cell": cell["grid_cell"],
            "density_mass_fraction": cell["density_mass_fraction"],
            "local_density_ncc": cell["local_density_ncc"],
            "strength": cell["strength"],
        }
        for cell in cells if cell["passes_mass"] and cell["passes_ncc"]
    ]
    selected = []
    if candidates:
        H, W = np.asarray(maskA).shape
        max_anchors = max(1, int(max_anchors))
        candidates.sort(key=lambda item: item["strength"], reverse=True)
        min_distance = float(min_anchor_distance_fraction) * min(H, W)
        for candidate in candidates:
            cx, cy = candidate["centroid"]
            if any(np.hypot(cx - old["centroid"][0], cy - old["centroid"][1]) < min_distance
                   for old in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_anchors:
                break
    return (selected, details) if return_details else selected


DENSITY_QC_DIR = "density_qc"


def _heatmap(field, colormap=cv2.COLORMAP_INFERNO):
    field = np.asarray(field, dtype=np.float32)
    peak = float(field.max()) if field.size else 0.0
    if peak <= 0:
        scaled = np.zeros(field.shape, dtype=np.uint8)
    else:
        scaled = np.clip(field * (255.0 / peak), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, colormap)


def write_density_qc_image(base_dir, prefix, maskA, aligned_maskB, details, selected,
                           final_score, score_thr, max_panel_width=420,
                           density_quality=None, acceptance=None):
    """One picture of what the density level saw in a tile.

    Four panels: the two masks on the aligned canvas (A red, B green); A's density
    field; B's; and the joint field with the grid, each cell's local NCC and mass
    fraction (green when the cell qualifies, red when it does not), each cell's
    density centroid as a small cross, and the anchors that were selected as numbered
    yellow circles. The header says whether the tile as a whole is kept -- rich tissue,
    score above the threshold, and at least one anchor -- which is
    `select_final_records`' rule.
    Written to `<tiles_dir>/density_qc/<tile>_density.jpg`; returns the path.
    """
    maskA = np.asarray(maskA)
    H, W = maskA.shape
    scale = min(1.0, float(max_panel_width) / max(W, 1))

    def shrink(image):
        if scale >= 1.0:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def pt(x, y):
        return int(round(x * scale)), int(round(y * scale))

    masks = np.zeros((H, W, 3), dtype=np.uint8)
    masks[..., 2] = (maskA > 0) * 255
    masks[..., 1] = (np.asarray(aligned_maskB) > 0) * 255
    joint_field = details.get("joint")
    if joint_field is None:
        joint_field = np.zeros((H, W), dtype=np.float32)
    panels = [
        shrink(masks),
        shrink(_heatmap(details["densityA"])),
        shrink(_heatmap(details["densityB"])),
        shrink(_heatmap(joint_field, cv2.COLORMAP_VIRIDIS)),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    joint = panels[3]
    for cell in details.get("cells", []):
        x0, y0, x1, y1 = cell["bounds"]
        ok = cell["passes_mass"] and cell["passes_ncc"]
        color = (0, 220, 0) if ok else (0, 0, 230)
        cv2.rectangle(joint, pt(x0, y0), (pt(x1, y1)[0] - 1, pt(x1, y1)[1] - 1), color, 1)
        label = f"ncc {cell['local_density_ncc']:.2f} m {cell['density_mass_fraction']:.2f}"
        cv2.putText(joint, label, (pt(x0, y0)[0] + 3, pt(x0, y0)[1] + 13), font, 0.38,
                    color, 1, cv2.LINE_AA)
        if cell["centroid"] is not None:
            cv2.drawMarker(joint, pt(*cell["centroid"]), (210, 210, 210),
                           cv2.MARKER_CROSS, 8, 1)
    for index, anchor in enumerate(selected):
        for panel in (panels[0], joint):
            centre = pt(*anchor["centroid"])
            cv2.circle(panel, centre, 9, (0, 255, 255), 2)
            cv2.putText(panel, str(index), (centre[0] + 10, centre[1] - 6), font, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)

    titles = ["masks: A red, B green (aligned)", "density A", "density B (aligned)",
              "joint field, grid, anchors"]
    columns = []
    for panel, title in zip(panels, titles):
        strip = np.zeros((18, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(strip, title, (4, 13), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        columns.append(np.vstack([strip, panel]))
    body = np.hstack(columns)
    tissue_ok = density_quality is None or bool(density_quality.get("passes", False))
    kept = (
        bool(acceptance.get("passes", False)) if acceptance is not None
        else tissue_ok and (final_score > score_thr) and len(selected) > 0
    )
    tissue_summary = ""
    if density_quality is not None:
        tissue_summary = (
            f"  fg={density_quality['foreground_fraction_a']:.2f}/"
            f"{density_quality['foreground_fraction_b']:.2f}"
            f"  comp={density_quality['largest_component_fraction_a']:.2f}/"
            f"{density_quality['largest_component_fraction_b']:.2f}"
            f"  tissue={'PASS' if tissue_ok else 'FAIL'}"
        )
    header = np.zeros((24, body.shape[1], 3), dtype=np.uint8)
    if acceptance is None:
        acceptance_summary = f"score>{score_thr:.2f}"
    else:
        acceptance_summary = (
            f"full>{acceptance['min_full_ncc']:.2f}  "
            f"local_med={acceptance['median_local_ncc']:.2f}>"
            f"{acceptance['min_median_local_ncc']:.2f}  "
            f"anchors>={acceptance['min_anchors']}"
        )
    cv2.putText(
        header,
        f"{prefix}  level=density  score={final_score:.3f} ({acceptance_summary})  "
        f"anchors={len(selected)}{tissue_summary}  {'KEPT' if kept else 'DROPPED'}",
        (4, 17), font, 0.5, (0, 220, 0) if kept else (0, 0, 230), 1, cv2.LINE_AA,
    )
    out_dir = Path(base_dir) / DENSITY_QC_DIR
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{prefix}_density.jpg"
    cv2.imwrite(str(path), np.vstack([header, body]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def map_architecture_anchors(anchors, final, H, W, case_id, paste_x0, paste_y0,
                             base_scale, dapi_mask_scale, he_mask_scale, raw_shape):
    """Architecture anchors in each side's original mask coordinates, keyed by token
    like the nucleus pairs so Stage 4c and Stage 5 read them the same way."""
    mapped = []
    for anchor_id, anchor in enumerate(anchors):
        x_aligned, y_aligned = anchor["centroid"]
        xA_img, yA_img = dapi_aligned_to_original(
            x_aligned, y_aligned, H, W, case_id, dapi_mask_scale
        )
        xB_img, yB_img = he_aligned_to_original(
            x_aligned, y_aligned, final, H, W,
            paste_x0, paste_y0, base_scale, he_mask_scale, raw_shape
        )
        mapped.append({
            "anchor_id": int(anchor_id),
            # nucleus_id keeps Stage 4c/5 filename and schema compatibility.
            "nucleus_id": int(anchor_id),
            "aligned": {
                SIDES["moving"]["token"]: [float(x_aligned), float(y_aligned)],
                SIDES["fixed"]["token"]: [float(x_aligned), float(y_aligned)],
            },
            "original": {
                SIDES["moving"]["token"]: [float(xA_img), float(yA_img)],
                SIDES["fixed"]["token"]: [float(xB_img), float(yB_img)],
            },
            "metrics": {
                "density_mass_fraction": float(anchor["density_mass_fraction"]),
                "local_density_ncc": float(anchor["local_density_ncc"]),
                "strength": float(anchor["strength"]),
            },
            "grid_cell": [int(anchor["grid_cell"][0]), int(anchor["grid_cell"][1])],
        })
    return mapped


# -----------------------------------------
# wrappers for main
# -----------------------------------------
def load_tile_orientation_case(base_dir, roles):
    """Read the coordinate convention of the tile files, not the Stage-1 display.

    New Stage-3 runs record case 0 in the fixed-side tile metadata because every saved
    Stage-2 homography maps original moving pixels to fixed pixels.  Reading the case
    from the tile artifacts also keeps old, already
    extracted GUI runs resumable: their H&E tiles carry the legacy non-zero case and
    still need the matching moving-mask orientation.
    """
    base_dir = Path(base_dir)
    tile_info_path = base_dir / f"{roles.fixed.token}_tile_info.json"
    try:
        tile_info = json.loads(tile_info_path.read_text())
        cases = {
            int(entry["meta"]["case_id"])
            for entry in tile_info.values()
            if isinstance(entry, dict)
            and isinstance(entry.get("meta"), dict)
            and entry["meta"].get("case_id") is not None
        }
        if len(cases) > 1:
            raise ValueError(
                f"mixed tile orientation cases in {tile_info_path}: {sorted(cases)}"
            )
        if cases:
            return cases.pop()
    except OSError:
        pass

    # Legacy tile sets predating explicit tile-coordinate metadata were written in the
    # Stage-1 orientation, so retain that convention when resuming them.
    return int(roles.moving.orientation_case)


def resolve_density_sigma_pixels(base_dir, roles, moving_mask_scale, params):
    """Convert the architecture blur from micrometres to moving-mask pixels.

    Stage 3 can read a pyramid level and Stage 4a can then upscale its mask, so a
    fixed pixel sigma changes biological meaning across datasets.  The density rung
    should instead see the same tissue scale everywhere.  If physical metadata is
    unavailable, retain the configured pixel sigma and record that fallback.
    """
    configured_px = float(params["density_sigma_px"])
    params["density_sigma_px_configured"] = configured_px
    params["density_sigma_source"] = "configured_px_fallback"
    if params.get("level") != "density":
        return configured_px

    try:
        from my_utils import inspect_image_pyramid

        run_dir = Path(base_dir).resolve().parent
        sampling = json.loads((run_dir / "sampled_points.json").read_text())
        level_key = f"{SIDES['moving']['token']}_tile_level"
        tile_level = int(sampling.get(level_key, sampling.get("dapi_tile_level", 0)))
        pyramid = inspect_image_pyramid(roles.moving.path)
        shapes = pyramid["level_shapes"]
        base_mpp = pyramid.get("base_mpp")
        if isinstance(base_mpp, (list, tuple, np.ndarray)):
            base_mpp_x = float(base_mpp[0])
        else:
            base_mpp_x = float(base_mpp)
        if not np.isfinite(base_mpp_x) or base_mpp_x <= 0:
            raise ValueError(f"invalid base MPP {base_mpp!r}")
        if tile_level < 0 or tile_level >= len(shapes):
            raise ValueError(f"tile level {tile_level} outside {len(shapes)} levels")
        base_width = float(shapes[0][1])
        level_width = float(shapes[tile_level][1])
        level_downsample = base_width / level_width
        mask_mpp = base_mpp_x * level_downsample / float(moving_mask_scale)
        sigma_um = float(params["density_sigma_um"])
        sigma_px = sigma_um / mask_mpp
        if not np.isfinite(sigma_px) or sigma_px <= 0:
            raise ValueError(f"invalid converted sigma {sigma_px}")
        params.update({
            "density_sigma_px": float(sigma_px),
            "density_mask_mpp": float(mask_mpp),
            "density_tile_level": int(tile_level),
            "density_level_downsample": float(level_downsample),
            "density_sigma_source": "physical_um",
        })
        print(
            f"[INFO] Density architecture sigma: {sigma_um:.3f} um / "
            f"{mask_mpp:.5f} um/px = {sigma_px:.2f} mask px "
            f"(moving tile level {tile_level}, mask x{moving_mask_scale:g})",
            flush=True,
        )
        return float(sigma_px)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        params["density_sigma_fallback_reason"] = str(exc)
        print(
            f"[WARN] Density architecture sigma uses configured "
            f"{configured_px:.2f} px because physical conversion failed: {exc}",
            flush=True,
        )
        return configured_px


def load_runtime_context(base_dir, all_tiles=False, all_masked=False, level=None):
    print(f"[INFO] Using output folder: {base_dir}", flush=True)

    roles = run_roles.read(os.path.join(base_dir, os.pardir))
    set_sides(roles)
    display_case_id = int(roles.moving.orientation_case)
    case_id = load_tile_orientation_case(base_dir, roles)
    print(f"[INFO] pair {roles.pair.name}: A={SIDES['moving']['token']} "
          f"B={SIDES['fixed']['token']} tile_case={case_id} "
          f"display_case={display_case_id}", flush=True)

    with open(os.path.join(base_dir, "nuclei_mask_info.json"), "r") as f:
        mask_scale_data = json.load(f)

    scales = mask_scale_data["mask_scale"]
    dapi_mask_scale = scales[SIDES["moving"]["token"]]
    he_mask_scale = scales[SIDES["fixed"]["token"]]

    stage4b_params = resolve_level(load_stage4b_params(base_dir, pair=roles.params_block), level)
    resolve_density_sigma_pixels(
        base_dir, roles, dapi_mask_scale, stage4b_params
    )
    print(f"[INFO] Effective stage4b params: {json.dumps(stage4b_params, indent=2)}", flush=True)
    print(f"[Stage4b] anchor level: {stage4b_params['level']} "
          f"(configured stage4b.anchor_mode={stage4b_params.get('anchor_mode', stage4b_modes.DEFAULT_MODE)}"
          f"{'' if level is None else ', --anchor-mode override'})", flush=True)

    all_dapi_files = sorted(
        path
        for path in glob.glob(os.path.join(base_dir, mask_name("*", "moving")))
        if not os.path.basename(path).startswith("._")
    )
    file_by_tile = {prefix_of(path): path for path in all_dapi_files}
    eligible_ids = [
        tile_id for tile_id in file_by_tile
        if os.path.exists(os.path.join(base_dir, mask_name(tile_id, "fixed")))
    ]
    if stage4b_params.get("n_tiles"):
        requested = int(stage4b_params["n_tiles"])
    else:
        # The rule Stage 4a masked by: a fraction of the Stage-3 pool, not of the masks.
        try:
            pool = len(shared_stage3_ids(base_dir, roles))
        except (OSError, ValueError):
            pool = len(eligible_ids)
        requested = stage4b_tile_count(available_tiles=pool, pair=roles.params_block)
        print(f"[INFO] stage4b.n_tiles unset: scoring {requested} tiles, "
              f"stage3.stage4_tile_fraction of the {pool} Stage-3 tile pairs, the rule "
              f"Stage 4a masked by", flush=True)
    if all_tiles:
        expected = set(shared_stage3_ids(base_dir, roles))
        missing = sorted(expected - set(eligible_ids))
        if missing:
            raise ValueError(f"Run Nuclei Masking for all tiles first; missing mask pairs: {missing[:5]}")
        if not expected:
            raise ValueError("No Stage-3 tile pairs to process")
        eligible_ids = sorted(expected)
        requested = len(eligible_ids)
    elif all_masked:
        # After a top-up the launcher wants every masked tile scored, not a fresh
        # `n_tiles` draw from the enlarged pool: the point of masking more tiles is
        # to score more tiles.
        requested = max(requested, len(eligible_ids))
        print(f"[INFO] --all-masked: scoring all {len(eligible_ids)} masked tile pairs",
              flush=True)
    selection_manifest = load_selection_manifest(base_dir)
    # A manifest is reusable only if it was drawn from the same pool. Without the
    # `eligible_tiles` check, `min(requested, len(eligible_ids))` hides a pool that has
    # grown: Stage 4b's tile top-up masks 40 more tiles, the pool goes 60 -> 100 -> 140,
    # and every comparison still holds because the cap is 60 either way. On
    # `cosmx_100_percent` that made two top-up rounds produce byte-identical Stage-4b
    # output -- same tiles, same scores, same one good tile -- after masking 80 tiles
    # for nothing. The top-up only ever worked where Stage 4a had masked fewer than
    # `n_tiles` to begin with, so the selection was forced to grow.
    manifest_is_valid = (
        isinstance(selection_manifest, dict)
        and int(selection_manifest.get("requested_tiles_for_nuclei_detection", -1))
        == requested
        and int(selection_manifest.get("eligible_tiles", -1)) == len(eligible_ids)
        and len(selection_manifest.get("selected_tile_ids", []))
        == min(requested, len(eligible_ids))
        and set(selection_manifest.get("selected_tile_ids", [])).issubset(eligible_ids)
    )
    if manifest_is_valid:
        manifest_path = os.path.join(base_dir, "spatially_selected_tiles.json")
        print("[INFO] Reusing Stage-4a spatial tile-selection manifest", flush=True)
    else:
        manifest_path, selection_manifest = create_selection_manifest(
            base_dir, requested, eligible_ids=eligible_ids, source="stage4b"
        )
    dapi_files = [
        file_by_tile[tile_id]
        for tile_id in selection_manifest["selected_tile_ids"]
        if tile_id in file_by_tile
    ]
    print(
        f"[INFO] Found {len(all_dapi_files)} masked tiles; selected "
        f"{len(dapi_files)} spatially uniform tiles for nuclei detection",
        flush=True,
    )
    print(f"[INFO] Tile-selection manifest: {manifest_path}", flush=True)

    return {
        "case_id": case_id,
        "dapi_mask_scale": dapi_mask_scale,
        "he_mask_scale": he_mask_scale,
        "stage4b_params": stage4b_params,
        "dapi_files": dapi_files,
        "selection_manifest": selection_manifest,
    }

def stage3_he_tile_margin(base_dir, default=0.2):
    """The slack Stage 3 left around each H&E tile, from the parameters it ran under."""
    from my_utils import cli_params_block

    # The copy beside the run wins, because it is the one the run used. Its block is
    # named by pair; `dapi` is accepted too, for copies made before the blocks were.
    path = os.path.join(base_dir, "..", "..", "..", "scripts", "parameters.json")
    for candidate in (os.path.join(base_dir, "../../parameters.json"), path):
        try:
            with open(candidate) as fh:
                data = json.load(fh)
            block = data.get(SIDES["pair"]) or data.get("dapi") or {}
            return float(block["stage3"]["he_tile_margin_ratio"])
        except Exception:
            continue
    try:
        return float(cli_params_block(SIDES["pair"])["stage3"]["he_tile_margin_ratio"])
    except Exception:
        return float(default)


def phase1_scale_centre(he_tile_margin_ratio):
    """The scale Phase 1 is looking for, which Stage 3 has already fixed.

    Both mask comparisons start with ``scale_and_pad(maskB_raw, maskA.shape)``, which
    resizes the H&E mask isotropically onto the DAPI mask's canvas. That divides the two
    tiles' pixel counts out, so what is left for Phase 1 to find is only the ratio of the
    *fields* the tiles cover -- and Stage 3 sets that deliberately, cutting the H&E tile
    over ``1 + he_tile_margin_ratio`` times the DAPI tile's field to leave slack. The
    centre is therefore that constant, and nothing about the tiles' sizes in pixels.

    The measurement says the same thing. Across the thirteen local datasets the H&E tile
    ranges from 788 to 940 px against a fixed 848 px DAPI tile, and Phase 1 lands on
    **1.22 in twelve of them** regardless; `cosmx_100_percent`, at 936 px, lands on 1.18.
    A centre of 1.2 is within 1.7% of all fourteen. The one dataset that disagrees is
    `artery_P3D` at 1.42, whose masks are the wrong size for a reason `TODO.md` item 5
    records and no scale window can repair.

    An earlier version of this multiplied by ``dapi_side / he_side`` as well, which
    double-counts what `scale_and_pad` has already done: it varies (1.08-1.29 over those
    same thirteen) where the truth does not, and on CosMx it gives 0.959 against an
    observed 1.18, putting the answer outside its own window.
    """
    return 1.0 + float(he_tile_margin_ratio)


def run_phase1_median_scale(dapi_files, base_dir, case_id, stage4b_params):
    p1 = stage4b_params["phase1"]

    phase1_files = dapi_files
    if stage4b_params["level"] == "density":
        quality_by_tile = calibrate_density_tissue_gate(
            dapi_files, base_dir, case_id, stage4b_params
        )
        phase1_files = [
            path for path in dapi_files
            if quality_by_tile.get(prefix_of(path), {}).get("passes", False)
        ]
        required = int(stage4b_params["min_fallback_tiles"])
        if len(phase1_files) < required:
            _log_insufficient_tiles(
                base_dir, "density", len(dapi_files), len(phase1_files), required
            )
            raise RuntimeError(
                "Not enough rich-tissue tiles for density alignment: "
                f"{len(phase1_files)} < {required}."
            )
        print(
            f"[INFO] Phase-1 density scale uses rich tissue only: "
            f"{len(phase1_files)}/{len(dapi_files)} eligible tiles",
            flush=True,
        )

    centre = phase1_scale_centre(stage3_he_tile_margin(base_dir))
    if stage4b_params["level"] == "density":
        # Stage 3 constructed the fixed tile with this field-of-view margin, then
        # scale_and_pad fitted that larger tile onto A's canvas.  Its inverse is the
        # known B-to-A geometric scale.  Broad serial-section density is repetitive
        # enough that estimating scale from NCC can prefer a different gland pattern
        # (1.02 in the ANHIR lung pair instead of the geometric 1.20).
        stage4b_params["density_expected_scale"] = float(centre)
        print(
            f"[INFO] Phase-1 density scale fixed by Stage-3 geometry: {centre:.4f}; "
            "NCC estimates translation only within a narrow scale tolerance",
            flush=True,
        )
        return float(centre)
    half = float(p1.get("scale_centre_frac", 0.25))
    lo, hi = centre * (1.0 - half), centre * (1.0 + half)
    print(f"[INFO] Phase-1 scale window from Stage 3's tile margin: centre {centre:.4f} "
          f"+/-{half:.0%} -> ({lo:.4f}, {hi:.4f})", flush=True)

    med_scale = robust_median_scale_from_firstN(
        dapi_files=phase1_files,
        base_dir=base_dir,
        case_id=case_id,
        N=int(p1["n_tiles_for_scale_estimation"]),
        ds=int(p1["ds"]),
        scale_range=(lo, hi),
        scale_step=float(p1["scale_step"]),
        shift_frac=float(p1["shift_frac"]),
        shift_step_frac=float(p1["shift_step_frac"]),
        score_metric=stage4b_params["score_metric"],
        density_sigma_px=float(stage4b_params["density_sigma_px"]),
    )
    return med_scale

def collect_phase23_files(dapi_files, base_dir):
    phase23_files = []
    for dapi_path in dapi_files:
        prefix = prefix_of(dapi_path)
        he_path = os.path.join(base_dir, mask_name(prefix, "fixed"))
        if os.path.exists(he_path):
            phase23_files.append(dapi_path)
    return phase23_files

def process_single_tile(
    dapi_path,
    base_dir,
    case_id,
    dapi_mask_scale,
    he_mask_scale,
    stage4b_params,
    new_scale_range,
    write_qa=True,
    write_density_qa=False,
):
    prefix = prefix_of(dapi_path)
    he_path = os.path.join(base_dir, mask_name(prefix, "fixed"))
    out_path = os.path.join(base_dir, f"{prefix}_aligned.jpg")

    maskA = read_mask(dapi_path)
    maskA = apply_orientation_to_tile(maskA, case_id)
    maskB_raw = read_mask(he_path)
    maskB, base_scale, paste_x0, paste_y0 = scale_and_pad(maskB_raw, maskA.shape)

    p1 = stage4b_params["phase1"]
    p2 = stage4b_params["phase2"]
    p3 = stage4b_params["phase3"]
    level = stage4b_params["level"]
    score_metric = stage4b_params["score_metric"]
    density_sigma_px = float(stage4b_params["density_sigma_px"])

    H, W = maskA.shape
    density_quality = None
    if level == "density":
        density_quality = stage4b_params.get(
            "_density_tissue_quality_by_tile", {}
        ).get(prefix)
        if density_quality is None:
            density_quality = density_tissue_quality(maskA, maskB, stage4b_params)

    run_alignment = level != "density" or bool(density_quality["passes"])
    if run_alignment:
        coarse = coarse_search_ds_parallel_cached(
            maskA,
            maskB,
            ds=int(p1["ds"]),
            scale_range=new_scale_range,
            scale_step=float(p2["scale_step"]),
            shift_frac=float(p2["shift_frac"]),
            shift_step_frac=float(p2["shift_step_frac"]),
            score_metric=score_metric,
            density_sigma_px=density_sigma_px,
        )

        final = refine_full(
            maskA,
            maskB,
            coarse,
            scale_range_frac=float(p3["scale_range_frac"]),
            scale_step_frac=float(p3["scale_step_frac"]),
            shift_range_frac=float(p3["shift_range_frac"]),
            shift_step_frac=float(p3["shift_step_frac"]),
            score_metric=score_metric,
            density_sigma_px=density_sigma_px,
        )

        coarse_full_score = score_full(
            maskA, maskB, coarse, score_metric, density_sigma_px
        )
        final_full_score = score_full(
            maskA, maskB, final, score_metric, density_sigma_px
        )
        B_final = warp(maskB, final["scale"], final["tx"], final["ty"], (H, W))
    else:
        # Keep a diagnostic record but do not let a sparse/fragmented tile search
        # the large scale-shift space and manufacture an attractive density NCC.
        coarse = {
            "scale": 1.0, "tx": 0, "ty": 0, "score_ds": -1.0,
            "shift_frac": float(p2["shift_frac"]),
            "shift_step_frac": float(p2["shift_step_frac"]),
            "skipped": "density_tissue_gate",
        }
        final = {
            "scale": 1.0, "tx": 0, "ty": 0, "score": -1.0,
            "scale_range_frac": float(p3["scale_range_frac"]),
            "scale_step_frac": float(p3["scale_step_frac"]),
            "shift_range_frac": float(p3["shift_range_frac"]),
            "shift_step_frac": float(p3["shift_step_frac"]),
            "skipped": "density_tissue_gate",
        }
        coarse_full_score = -1.0
        final_full_score = -1.0
        B_final = maskB

    if write_qa:
        # Save red/green alignment overlay for interactive inspection.
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[..., 0] = (maskA > 0) * 255
        overlay[..., 1] = (B_final > 0) * 255
        cv2.imwrite(out_path, overlay)

    # Architecture landmarks belong to the density level. On the pairs level's
    # Dice-aligned canvas they would only resample a transform the pairs already
    # measure better, and a pairs run that is short is answered by the launcher
    # with more tiles or with the density level, never with a weaker anchor.
    architecture_anchors = []
    density_acceptance_info = None
    if level == "density":
        density_candidates, density_details = nuclear_architecture_anchors(
            maskA,
            B_final,
            density_sigma_px=density_sigma_px,
            grid_size=int(stage4b_params["architecture_grid_size"]),
            max_anchors=int(stage4b_params["architecture_max_anchors_per_tile"]),
            min_region_mass_fraction=float(
                stage4b_params["architecture_min_region_mass_fraction"]),
            min_region_ncc=float(stage4b_params["architecture_min_region_ncc"]),
            min_anchor_distance_fraction=float(
                stage4b_params["architecture_min_anchor_distance_fraction"]),
            return_details=True,
        )
        if density_quality["passes"]:
            architecture_anchors = density_candidates
        density_acceptance_info = density_acceptance(
            density_quality,
            final_full_score,
            architecture_anchors,
            stage4b_params,
            alignment=final,
            mask_shape=(H, W),
        )
        if write_density_qa:
            # `qc_images.stage4b_density`: what this tile's density fields looked like
            # and which grid cells became anchors, in tiles/density_qc/.
            write_density_qc_image(
                base_dir, prefix, maskA, B_final, density_details, architecture_anchors,
                float(final_full_score), float(stage4b_params["fallback_score_thr"]),
                density_quality=density_quality,
                acceptance=density_acceptance_info,
            )
    architecture_mapped = map_architecture_anchors(
        architecture_anchors, final, H, W, case_id, paste_x0, paste_y0,
        base_scale, dapi_mask_scale, he_mask_scale, maskB_raw.shape,
    )

    # Isolated nuclei
    A_iso = find_isolated_components(maskA, min_area=60)
    B_iso = find_isolated_components(B_final, min_area=60)

    A_iso_mask = np.zeros_like(maskA, dtype=np.uint8)
    for c in A_iso:
        slc = c["bbox"]
        A_iso_mask[slc][c["comp"]] = 1

    B_iso_mask = np.zeros_like(maskA, dtype=np.uint8)
    for c in B_iso:
        slc = c["bbox"]
        B_iso_mask[slc][c["comp"]] = 1

    pairs = pair_isolated_cells(
        A_iso,
        B_iso,
        A_iso_mask,
        B_iso_mask,
        min_area=60,
        min_sym_cov=0.4,
        top_k=int(stage4b_params["pair_top_k"]),
    )

    pairs_mapped = []
    for nucleus_id, p in enumerate(pairs):
        xA_aligned, yA_aligned = p["A_centroid"]
        xB_aligned, yB_aligned = p["B_centroid"]

        xA_img, yA_img = dapi_aligned_to_original(
            xA_aligned, yA_aligned, H, W, case_id, dapi_mask_scale
        )
        xB_img, yB_img = he_aligned_to_original(
            xB_aligned, yB_aligned,
            final, H, W,
            paste_x0, paste_y0,
            base_scale,
            he_mask_scale,
            maskB_raw.shape,
        )

        pairs_mapped.append({
            "nucleus_id": int(nucleus_id),
            "aligned": {
                SIDES["moving"]["token"]: [float(xA_aligned), float(yA_aligned)],
                SIDES["fixed"]["token"]: [float(xB_aligned), float(yB_aligned)],
            },
            "original": {
                SIDES["moving"]["token"]: [float(xA_img), float(yA_img)],
                SIDES["fixed"]["token"]: [float(xB_img), float(yB_img)],
            },
            "area": {
                SIDES["moving"]["token"]: int(p["areaA"]),
                SIDES["fixed"]["token"]: int(p["areaB"]),
            },
            "metrics": {
                "f1": float(p["f1"]),
                "covA": float(p["covA"]),
                "covB": float(p["covB"]),
            },
            "bbox_aligned": {
                "y0": int(p["A_comp"]["bbox"][0].start),
                "y1": int(p["A_comp"]["bbox"][0].stop),
                "x0": int(p["A_comp"]["bbox"][1].start),
                "x1": int(p["A_comp"]["bbox"][1].stop),
            },
        })

    if write_qa:
        # Save standout visualization for the Stage-4 tile gallery.
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[..., 0] = (maskA > 0) * 120
        vis[..., 1] = (B_final > 0) * 120
        alpha = 0.6
        for p in pairs:
            comp = p["A_comp"]["comp"]
            slc = p["A_comp"]["bbox"]
            for c in [0, 1]:
                vis_ch = vis[slc[0], slc[1], c]
                vis_ch[comp] = (alpha * 255 + (1 - alpha) * vis_ch[comp]).astype(np.uint8)
        for anchor in architecture_anchors:
            ax, ay = anchor["centroid"]
            cv2.circle(vis, (int(round(ax)), int(round(ay))), 7, (255, 255, 0), 2)

        cv2.imwrite(os.path.join(base_dir, f"{prefix}_standout.jpg"), vis)

    return {
        "tile": prefix,
        "coarse": coarse,
        "final": final,
        "coarse_full_score": float(coarse_full_score),
        "final_full_score": float(final_full_score),
        "n_pairs": int(len(pairs)),
        "pairs_mapped": pairs_mapped,
        "n_architecture_anchors": int(len(architecture_mapped)),
        "architecture_anchors_mapped": architecture_mapped,
        "level": level,
        "score_metric": score_metric,
        "H": int(H),
        "W": int(W),
        "paste_x0": int(paste_x0),
        "paste_y0": int(paste_y0),
        "base_scale": float(base_scale),
        "n_isolated_a": int(len(A_iso)),
        "n_isolated_b": int(len(B_iso)),
        "density_tissue_quality": density_quality,
        "density_acceptance": density_acceptance_info,
    }

def run_phase23_alignment_and_pairing(
    phase23_files,
    base_dir,
    case_id,
    dapi_mask_scale,
    he_mask_scale,
    stage4b_params,
    new_scale_range,
    write_qa=True,
    write_density_qa=False,
):
    good_nuclei_min = int(stage4b_params["good_nuclei_min"])
    fallback_score_thr = float(stage4b_params["fallback_score_thr"])
    serial = stage4b_params["level"] == "density"
    score_metric = stage4b_params["score_metric"]

    fallback_candidates = []
    good_tile_count = 0

    phase23_total = len(phase23_files)
    t_phase23_start = time.perf_counter()
    print(f"[INFO] Phase-2/3 valid tiles: {phase23_total}", flush=True)

    for idx, dapi_path in enumerate(phase23_files, 1):
        t_patch_start = time.perf_counter()
        prefix = prefix_of(dapi_path)
        print(f"\n=== Processing {prefix} ({idx}/{phase23_total}) ===", flush=True)

        result = process_single_tile(
            dapi_path=dapi_path,
            base_dir=base_dir,
            case_id=case_id,
            dapi_mask_scale=dapi_mask_scale,
            he_mask_scale=he_mask_scale,
            stage4b_params=stage4b_params,
            new_scale_range=new_scale_range,
            write_qa=write_qa,
            write_density_qa=write_density_qa,
        )

        print("  Coarse:", result["coarse"], flush=True)
        print("  Final:", result["final"], flush=True)
        print(
            f"  [SCORE:{score_metric}] coarse_ds={result['coarse']['score_ds']:.4f} | "
            f"coarse_full_score={result['coarse_full_score']:.4f} | "
            f"final_full_score={result['final_full_score']:.4f}",
            flush=True,
        )
        print(
            f"  Isolated A: {result['n_isolated_a']}, "
            f"Isolated B: {result['n_isolated_b']}",
            flush=True,
        )
        if serial:
            print(f"  Nuclear-architecture anchors: {result['n_architecture_anchors']}",
                  flush=True)
            quality = result["density_tissue_quality"]
            if not quality["passes"]:
                print(
                    "  [DENSITY] alignment skipped by tissue gate: "
                    + "; ".join(quality["rejection_reasons"]),
                    flush=True,
                )
            acceptance = result["density_acceptance"]
            print(
                f"  [DENSITY] local median NCC="
                f"{acceptance['median_local_ncc']:.3f}, "
                f"full NCC={acceptance['full_ncc']:.3f}, "
                f"accepted={'yes' if acceptance['passes'] else 'no'}"
                + (
                    ""
                    if acceptance["passes"]
                    else " (" + "; ".join(acceptance["rejection_reasons"]) + ")"
                ),
                flush=True,
            )

        if result["n_pairs"] == 0:
            print("  [DEBUG] No valid A/B pairs found after filtering", flush=True)

        if serial:
            good = density_result_is_good(
                result, fallback_score_thr, params=stage4b_params
            )
        else:
            good = (result["n_pairs"] >= good_nuclei_min
                    and result["final_full_score"] > fallback_score_thr)
        if good:
            good_tile_count += 1

        fallback_candidates.append(result)

        t_patch_end = time.perf_counter()
        phase23_elapsed = time.perf_counter() - t_phase23_start
        phase23_avg = phase23_elapsed / idx if idx > 0 else 0.0
        phase23_eta = phase23_avg * (phase23_total - idx)

        print(f"[TIME] {prefix}: {t_patch_end - t_patch_start:.2f} sec", flush=True)
        print(
            f"[INFO] Phase-2/3 progress: {idx}/{phase23_total} | "
            f"Elapsed: {fmt_sec(phase23_elapsed)} | ETA: {fmt_sec(phase23_eta)}",
            flush=True,
        )
        print(f"[PROGRESS] TILES {idx}/{phase23_total}", flush=True)

    return good_tile_count, fallback_candidates


INSUFFICIENT_TILES_EVENT = "stage4b_insufficient_tiles"
ANCHORS_SELECTED_EVENT = "stage4b_anchors_selected"


def _log_stage4b_event(tiles_dir, event, **fields):
    """Record a Stage-4b outcome where the launcher can find it (pipeline_times.json)."""
    if not tiles_dir:
        return
    try:
        run_dir = Path(tiles_dir).resolve().parent
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from cli_common import log_stage_event

        log_stage_event(run_dir, "stage4b_events", event, **fields)
    except Exception as exc:                      # never mask the real outcome
        print(f"[Stage4b] could not record {event}: {exc}", flush=True)


def _log_insufficient_tiles(tiles_dir, level, scored_tiles, good_tiles, required_tiles):
    """Record a tile shortfall.

    Stage 4b can fail for want of tiles or for want of anything else, and the two need
    different responses: the first is answered by masking more of what Stage 3 already
    extracted or by the next rung of the ladder, the second is not. `run_pharaoh.py`
    reads this event to tell them apart, and `level` to know which rung fell short.
    """
    _log_stage4b_event(
        tiles_dir, INSUFFICIENT_TILES_EVENT, level=str(level),
        scored_tiles=int(scored_tiles), good_tiles=int(good_tiles),
        required_tiles=int(required_tiles),
    )


def select_final_records(
    fallback_candidates,
    case_id,
    dapi_mask_scale,
    he_mask_scale,
    stage4b_params,
    tiles_dir=None,
):
    """The anchors this level ships, or a recorded shortfall and a failure.

    A level either meets its tile quota or it does not; there is no third
    outcome. The tile-centre floor that used to fill a short pairs run is gone:
    a centre carried the tile's estimated shift and no tissue correspondence,
    and the launcher now answers a shortfall with more tiles or with the density
    level, both better evidence than a centre. Nor does a density run keep the
    few pairs a short pairs run found: mixing two alignment estimates into one
    homography fit is worse than either alone.
    """
    scored = fallback_candidates
    good_nuclei_min = int(stage4b_params["good_nuclei_min"])
    min_good_tiles = int(stage4b_params["min_good_tiles"])
    fallback_score_thr = float(stage4b_params["fallback_score_thr"])
    min_fallback_tiles = int(stage4b_params["min_fallback_tiles"])
    pairs_to_take_per_tile = int(stage4b_params["pairs_to_take_per_tile"])
    level = stage4b_params["level"]
    score_metric = stage4b_params["score_metric"]
    density_sigma_px = float(stage4b_params["density_sigma_px"])
    meta = {
        "anchor_mode": stage4b_params["anchor_mode"],
        "level": level,
        "alignment_score_metric": score_metric,
        "case_id": int(case_id),
        "mask_scale": {
            SIDES["moving"]["token"]: float(dapi_mask_scale),
            SIDES["fixed"]["token"]: float(he_mask_scale),
        },
    }

    if level == "density":
        # Adjacent sections: the landmarks are the conserved-architecture anchors,
        # and a tile counts when its density alignment held and it found some.
        tiles = [
            t for t in scored
            if density_result_is_good(
                t, fallback_score_thr, params=stage4b_params
            )
        ]
        print(f"[INFO] density level: {len(tiles)} of {len(scored)} scored tiles pass "
              f"rich tissue + local architecture evidence "
              f"(required >= {min_fallback_tiles})", flush=True)
        if len(tiles) < min_fallback_tiles:
            _log_insufficient_tiles(tiles_dir, level, len(scored), len(tiles), min_fallback_tiles)
            raise RuntimeError(
                "Not enough tiles with reliable density alignment and architecture "
                f"anchors: {len(tiles)} < {min_fallback_tiles}."
            )
        records = [
            {
                "tile": t["tile"],
                "mode": "nuclear_architecture_anchor",
                "anchor_id": int(anchor["anchor_id"]),
                "nucleus_id": int(anchor["nucleus_id"]),
                "aligned": anchor["aligned"],
                "original": anchor["original"],
                "grid_cell": anchor["grid_cell"],
                "metrics": {
                    "final_full_score": float(t["final_full_score"]),
                    "n_pairs": int(t["n_pairs"]),
                    "tissue_quality": t.get("density_tissue_quality"),
                    "density_acceptance": t.get("density_acceptance"),
                    **anchor["metrics"],
                },
                "meta": {
                    **meta,
                    "selection_mode": "density",
                    "biological_unit": "local_nuclear_architecture",
                    "density_sigma_px": density_sigma_px,
                    "density_sigma_um": float(stage4b_params["density_sigma_um"]),
                    "density_sigma_source": stage4b_params.get(
                        "density_sigma_source", "configured_px_fallback"
                    ),
                    "density_mask_mpp": stage4b_params.get("density_mask_mpp"),
                },
            }
            for t in tiles
            for anchor in t["architecture_anchors_mapped"]
        ]
    else:
        tiles = [
            t for t in scored
            if t["n_pairs"] >= good_nuclei_min and t["final_full_score"] > fallback_score_thr
        ]
        print(f"[INFO] nuclei_pairs level: {len(tiles)} of {len(scored)} scored tiles are "
              f"good (n_pairs>={good_nuclei_min}, score>{fallback_score_thr}; "
              f"required >= {min_good_tiles})", flush=True)
        if len(tiles) < min_good_tiles:
            _log_insufficient_tiles(tiles_dir, level, len(scored), len(tiles), min_good_tiles)
            raise RuntimeError(
                f"Not enough good tiles: good_tiles(n_pairs>={good_nuclei_min})="
                f"{len(tiles)} < {min_good_tiles}."
            )
        records = [
            {
                "tile": t["tile"],
                "mode": "nuclei_pair",
                "nucleus_id": int(pm["nucleus_id"]),
                "aligned": pm["aligned"],
                "original": pm["original"],
                "area": pm["area"],
                "bbox_aligned": pm["bbox_aligned"],
                "metrics": {
                    "final_full_score": float(t["final_full_score"]),
                    "n_pairs": int(t["n_pairs"]),
                    **pm["metrics"],
                },
                "meta": {**meta, "selection_mode": "nuclei_pairs"},
            }
            for t in tiles
            for pm in t["pairs_mapped"][:pairs_to_take_per_tile]
        ]

    print(f"[INFO] Selection: level={level}, tiles={len(tiles)}, anchors={len(records)}",
          flush=True)
    _log_stage4b_event(
        tiles_dir, ANCHORS_SELECTED_EVENT, level=str(level),
        scored_tiles=int(len(scored)), selected_tiles=int(len(tiles)),
        anchors=int(len(records)),
    )
    return records

QA_POINT_COLORS = ((0, 0, 255), (0, 255, 255), (255, 255, 0),
                   (255, 0, 255), (0, 255, 0), (255, 128, 0))  # BGR


def _draw_qa_point(image, point, label, color):
    x, y = (int(round(value)) for value in point)
    cv2.circle(image, (x, y), 5, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, 3, cv2.LINE_AA)
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    (width, height), baseline = cv2.getTextSize(label, font, font_scale, 1)
    # Keep the ID readable on both brightfield and fluorescence, including near
    # the tile edges. The point remains in the record's original tile coordinates.
    tx = max(2, min(x + 10, image.shape[1] - width - 3))
    ty = max(height + 2, min(y - 8, image.shape[0] - baseline - 3))
    cv2.putText(image, label, (tx, ty), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)


def save_qa_point_overlays(base_dir, final_records):
    """Draw every accepted anchor per tile, without changing detection or JSON.

    One PNG per tile/role, named by token. Labels reuse the saved nucleus_id
    (zero-based), so the same number identifies the same pair on both sides and
    in the centroid/patch records. Density anchors carry the same ID convention.
    """
    print("\n[INFO] Drawing accepted QA points by tile...", flush=True)
    base_dir = Path(base_dir)
    anchors_dir = base_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    by_tile = {}
    for rec in final_records:
        by_tile.setdefault(rec["tile"], []).append(rec)

    written = []
    for tile, records in by_tile.items():
        images = {}
        for role in ("moving", "fixed"):
            path = base_dir / raw_tile_name(tile, role)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
            if image is None:
                print(f"[WARN] Missing or unreadable {role} ({SIDES[role]['token']}) "
                      f"raw tile: {path}", flush=True)
                break
            images[role] = image
        if len(images) != 2:
            continue

        for index, rec in enumerate(records):
            label = str(rec.get("nucleus_id", rec.get("anchor_id", index)))
            color = QA_POINT_COLORS[index % len(QA_POINT_COLORS)]
            for role, image in images.items():
                _draw_qa_point(image, rec["original"][SIDES[role]["token"]], label, color)

        # Save only after all records have been drawn; a later pair must not
        # reload the raw image and overwrite earlier points on this same tile.
        for role, image in images.items():
            path = anchors_dir / f"{tile}_{SIDES[role]['token']}_point.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write Stage-4b QA overlay: {path}")
            written.append(path)
    print(f"[DONE] QA overlays saved: {len(written)} images for "
          f"{len(written) // 2}/{len(by_tile)} tiles.\n", flush=True)
    return written


def save_qa_overlays_and_json(base_dir, final_records, write_qa=True):
    """Write the anchors JSON (and, with QA on, the point overlays); returns the overlay
    paths written, [] when QA is off."""
    point_files = []
    if write_qa:
        point_files = save_qa_point_overlays(base_dir, final_records)
    else:
        print("\n[INFO] Skipping optional Stage-4b QA images.", flush=True)

    json_path = os.path.join(base_dir, "standout_nuclei.json")
    with open(json_path, "w") as f:
        json.dump(final_records, f, indent=2)
    print(f"[INFO] Saved standout nuclei info -> {json_path}", flush=True)
    return point_files


def write_stage4b_qa_manifest(base_dir, prefixes, level, write_qa, write_density_qa, point_files,
                              final_records=()):
    """Record which QC images this run drew, so a resume can tell what is missing
    (`qc_artifacts.stage4b_qc_missing`). A group that was on is recorded even when the
    level draws nothing for it, so it is not asked for again on every resume; a group
    drawn short of its inputs (point overlays for a tile whose raw images are gone) is
    recorded as incomplete, with the reason, and asked for again."""
    import qc_artifacts

    base = Path(base_dir)
    groups, incomplete = {}, {}
    if write_qa:
        groups["stage4b_overlays"] = (
            [f"{p}_aligned.jpg" for p in prefixes] + [f"{p}_standout.jpg" for p in prefixes]
            + [str(Path(path).resolve().relative_to(base.resolve())) for path in point_files]
        )
        tiles_with_anchors = {rec["tile"] for rec in final_records if isinstance(rec, dict) and "tile" in rec}
        drawn = len(point_files) // 2
        if tiles_with_anchors and drawn < len(tiles_with_anchors):
            incomplete["stage4b_overlays"] = (
                f"point overlays drawn for {drawn} of {len(tiles_with_anchors)} tiles with "
                f"anchors; the others have no raw tile image")
    if write_density_qa:
        groups["stage4b_density"] = (
            [f"{DENSITY_QC_DIR}/{p}_density.jpg" for p in prefixes] if level == "density" else []
        )
    if groups:
        qc_artifacts.write_stage4b_manifest(base, groups, incomplete)
        print(f"[INFO] QC manifest: {qc_artifacts.STAGE4B_MANIFEST} covers "
              + ", ".join(f"{g} ({len(files)} files"
                          f"{', INCOMPLETE: ' + incomplete[g] if g in incomplete else ''})"
                          for g, files in groups.items()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find standout DAPI--H&E nucleus anchors")
    parser.add_argument("tiles_dir")
    parser.add_argument("--anchor-mode", choices=list(stage4b_modes.LEVELS), default=None,
                        help="the level this process runs; default: the first rung of the "
                             "configured stage4b.anchor_mode (auto starts at nuclei_pairs)")
    parser.add_argument("--all-masked", action="store_true",
                        help="score every masked tile pair instead of the stage4b.n_tiles "
                             "spatial subset; what the launcher passes after a top-up")
    parser.add_argument("--all-tiles", action="store_true",
                        help="process every Stage-3 tile pair; bypass the CLI subset cap")
    parser.add_argument("--write-centroids", action="store_true",
                        help="also write Stage-4c global coordinates, without patch images")
    parser.add_argument(
        "--skip-qa-images",
        action="store_true",
        help="force the aligned/standout/anchor visualizations off even when "
             "qc_images.stage4b_overlays is on; JSON is unchanged",
    )
    parser.add_argument("--qa-only", action="store_true",
                        help="re-draw the QC images (stage4b_overlays, stage4b_density) for "
                             "the anchors already in standout_nuclei.json, scoring the same "
                             "level and scope (--anchor-mode, --all-masked) the run shipped "
                             "with; the JSON is not rewritten")
    args = parser.parse_args()
    base_dir = args.tiles_dir
    if args.all_tiles:
        Path(base_dir, COMPLETION_FILE).unlink(missing_ok=True)
    # `*_aligned.jpg`, `*_standout.jpg` and `anchors/` are for a human to look at;
    # nothing downstream reads them. They used to be switched by the launcher passing
    # `--skip-qa-images` for every dapi run, which made them unreachable in that mode.
    # They now follow `qc_images.stage4b_overlays` like every other QC group, so
    # `--qc-images stage4b_overlays` turns them on for one run. The flag stays as an
    # explicit off for a direct invocation, and still wins.
    from my_utils import qc_images_enabled
    write_qa = qc_images_enabled("stage4b_overlays") and not args.skip_qa_images
    print(f"[Stage4b] QA overlays: {'on' if write_qa else 'off'} "
          f"(qc_images.stage4b_overlays)", flush=True)
    # Per-tile density-field pictures are the density level's own QC group; they
    # cost a render per tile and mean nothing on the pairs level.
    write_density_qa = qc_images_enabled("stage4b_density") and not args.skip_qa_images
    print(f"[Stage4b] density QC images: {'on' if write_density_qa else 'off'} "
          f"(qc_images.stage4b_density; drawn on the density level only, into "
          f"tiles/{DENSITY_QC_DIR}/)", flush=True)
    t_global_start = time.perf_counter()

    ctx = load_runtime_context(base_dir, all_tiles=args.all_tiles,
                               all_masked=args.all_masked, level=args.anchor_mode)
    case_id = ctx["case_id"]
    dapi_mask_scale = ctx["dapi_mask_scale"]
    he_mask_scale = ctx["he_mask_scale"]
    stage4b_params = ctx["stage4b_params"]
    dapi_files = ctx["dapi_files"]

    # Phase 1: estimate a robust median scale
    med_scale = run_phase1_median_scale(
        dapi_files=dapi_files,
        base_dir=base_dir,
        case_id=case_id,
        stage4b_params=stage4b_params,
    )

    # Phase 2 setup: narrow the scale search around the median
    p2 = stage4b_params["phase2"]
    median_frac = float(p2["median_scale_frac"])
    new_scale_range = make_scale_range_around_median(
        med_scale,
        frac=median_frac,
        hard_min=0.5,
        hard_max=3.0,
    )
    print(
        f"[INFO] Phase-2: use scale_range around median: {new_scale_range} "
        f"(median={med_scale:.4f}, frac=±{median_frac*100:.1f}%)",
        flush=True,
    )

    # Collect valid tiles for Phase 2/3
    phase23_files = collect_phase23_files(dapi_files, base_dir)

    # Phase 2/3: coarse alignment + refinement + standout nuclei pairing
    good_tile_count, fallback_candidates = run_phase23_alignment_and_pairing(
        phase23_files=phase23_files,
        base_dir=base_dir,
        case_id=case_id,
        dapi_mask_scale=dapi_mask_scale,
        he_mask_scale=he_mask_scale,
        stage4b_params=stage4b_params,
        new_scale_range=new_scale_range,
        write_qa=write_qa,
        write_density_qa=write_density_qa,
    )

    prefixes = [prefix_of(p) for p in phase23_files]
    if args.qa_only:
        # The anchors on disk are the run's result; only the pictures are redrawn.
        json_path = Path(base_dir) / "standout_nuclei.json"
        if not json_path.is_file():
            raise SystemExit(f"--qa-only needs {json_path}; run Stage 4b first")
        final_records = json.loads(json_path.read_text())
        point_files = save_qa_point_overlays(base_dir, final_records) if write_qa else []
        write_stage4b_qa_manifest(base_dir, prefixes, stage4b_params["level"], write_qa,
                                  write_density_qa, point_files, final_records)
        print("[INFO] --qa-only: QC images redrawn; standout_nuclei.json unchanged", flush=True)
    else:
        # Final selection
        final_records = select_final_records(
            fallback_candidates=fallback_candidates,
            case_id=case_id,
            dapi_mask_scale=dapi_mask_scale,
            he_mask_scale=he_mask_scale,
            stage4b_params=stage4b_params,
            tiles_dir=base_dir,
        )

        # Save QA overlays and output JSON
        point_files = save_qa_overlays_and_json(base_dir, final_records, write_qa=write_qa)
        write_stage4b_qa_manifest(base_dir, prefixes, stage4b_params["level"], write_qa,
                                  write_density_qa, point_files, final_records)
        if args.write_centroids:
            import subprocess
            subprocess.run([sys.executable, str(Path(__file__).with_name("4c_get_nuclei_patches.py")),
                            str(base_dir), "--centroids-only"], check=True)
        if args.all_tiles:
            write_all_tiles_completion(base_dir, prefixes, write_qa)

    t_global_end = time.perf_counter()
    print("\n==============================", flush=True)
    print(f"[TOTAL TIME] {t_global_end - t_global_start:.2f} sec", flush=True)
    print("==============================", flush=True)
    print("[DONE]", flush=True)
