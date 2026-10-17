# ============================================================
# cli_stage2_orb.py - Stage 2 (headless): automatic feature alignment
#
# Replaces the interactive manual-alignment GUI. Runs feature-based
# registration moving -> H&E on the same low-level images the GUI
# loads, and writes initial_alignment.json in the exact schema
# the GUI's "Save" produces (keys H_mat, H_mat_level_0), plus an
# overlay preview.
#
# Matching is performed on the level image saved by Stage 1. In GUI runs that
# image has already been rotated/flipped into the user-confirmed orientation.
# DAPI matching uses the intensity-preserving `1_dapi_level_image.png`; the
# Glasbey `1_dapi_lut.png` artifact is reserved for visualization/overlays.
# The fitted homography therefore maps oriented pixels -> H&E pixels. Before
# saving, it is composed with the Stage-1 original -> oriented affine so H_mat
# keeps the original-image coordinate convention required by later stages.
# Headless Stage 1 stores identity, so the same path also covers CLI runs.
#
# Detectors are tried as a cascade (parameters.json ['stage2']['methods'],
# default ORB then SIFT):
#
#       run ORB  ->  reliable?  -- yes -->  use ORB
#                       |
#                       no
#                       v
#       run SIFT ->  reliable?  -- yes -->  use SIFT
#                       |
#                       no
#                       v
#                     fail (exit 2, fall back to the GUI)
#
# ORB is fast but its binary intensity-order descriptors are fragile
# across modalities; SIFT's gradient descriptors usually survive the
# cross-modal DAPI<->H&E case, so it is worth the extra seconds only
# when ORB has already failed.
#
# "Reliable" = enough RANSAC inliers, a healthy inlier ratio, a small
# median reprojection error, and a non-mirrored/non-degenerate linear
# part. Inlier COUNTS are not comparable between detectors (SIFT
# normally returns far more matches), which is why the ratio and the
# reprojection error carry the decision and why each method may set its
# own thresholds via a per-method override block.
#
# Usage:
#   python cli_stage2_orb.py --engine <dir> --run-dir <dir> --pair {dapi-he,he-he}
# ============================================================
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tifffile

import run_roles
from alignment_geometry import compose_with_orientation, lift_to_level0
from run_artifacts import INITIAL_ALIGNMENT, INITIAL_OVERLAY
from cli_common import load_json, save_json, log_stage_event, banner
import overlays
from my_utils import percentile_stretch_to_u8


def _roles(info, mode_or_roles):
    """The run's roles, from a `RunRoles` or from a 4.10 `--mode` token."""
    if isinstance(mode_or_roles, run_roles.RunRoles):
        return mode_or_roles
    pair = run_roles.coerce_pair(mode_or_roles)
    try:
        return run_roles.read(info)
    except Exception:
        # A partial `images_info` (the unit tests build these) still names its sides
        # by token; resolve what is there against the pair the mode meant.
        moving = run_roles.image_role(
            pair, run_roles.MOVING,
            info.get(f"{pair.key_prefix(run_roles.MOVING)}_path", ""),
            info.get(f"{pair.key_prefix(run_roles.MOVING)}_level", 0),
            gui_affine=info.get(f"{pair.key_prefix(run_roles.MOVING)}_gui_affine"),
        )
        fixed = run_roles.image_role(
            pair, run_roles.FIXED,
            info.get(f"{pair.key_prefix(run_roles.FIXED)}_path", ""),
            info.get(f"{pair.key_prefix(run_roles.FIXED)}_level", 0),
        )
        return run_roles.RunRoles(pair=pair, moving=moving, fixed=fixed)

STAGE2_DEFAULTS = {
    "random_seed": 43,
    "methods": ["orb", "sift"],
    "n_features": 8000,
    "ratio": 0.75,
    "ransac_thresh": 5.0,
    "min_inliers": 15,
    "min_inlier_ratio": 0.15,
    "max_reproj_px": 3.0,
    # Every correct fit measured in this tree sits at 0.963-1.000; the collapsed ones
    # are orders of magnitude below. 0.8 is far from both.
    "min_axis_ratio": 0.8,
}


def _prep(img_bgr, invert=False):
    """Bring one image to the 8-bit representation ORB and SIFT can match on.

    A 16-bit fluorescence image is converted through `log1p` and a percentile window
    rather than by scaling between its min and max. Two reasons, both measured on
    `geo_lung_l7` (PLAN_4.8.md sections 3-6):

    * min-max hands the whole 8-bit range to the single brightest pixel, so one hot
      spot pushes the tissue into the bottom tenth of the range -- l7's p99 landed at
      30.8/255 and Stage 2 fitted 5 inliers on garbage;
    * even a percentile window does not fix it, because the histogram is skewed, not
      merely offset: l7's median is 3 against a p99 of 1321. Fluorescence intensity is
      multiplicative -- a nucleus is some *factor* brighter than background -- while
      ORB compares intensity orderings and SIFT compares gradients, both of which read
      local differences. The logarithm is what turns the quantity the microscope
      measures into the quantity the descriptor compares, and unlike histogram
      equalisation it is the same fixed map for every slide.

    CLAHE follows only for images that arrive as 8-bit (the H&E). After the log the
    dynamic range is already spread globally, and equalising again inside 8x8 tiles
    amplifies noise in the tiles that hold no tissue: it cost l7 40 -> 31 inliers, l3
    110 -> 27 and l6 322 -> 209.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if gray.dtype != np.uint8:
        gray = percentile_stretch_to_u8(np.log1p(gray.astype(np.float32)))
        return cv2.bitwise_not(gray) if invert else gray
    if invert:
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detector(method, n_features):
    """Return (detector, matcher norm) for 'orb' or 'sift'."""
    if method == "orb":
        return cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    if method == "sift":
        return cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    raise ValueError(f"unknown stage-2 method: {method!r} (expected 'orb' or 'sift')")


def ome_physical_pixel_size(path):
    """Return an OME image's (x, y) pixel spacing in micrometers, if known."""
    if not path:
        return None

    try:
        with tifffile.TiffFile(path) as image:
            metadata = image.ome_metadata
        if not metadata:
            return None
        pixels = ET.fromstring(metadata).find(".//{*}Pixels")
        if pixels is None:
            return None

        factors = {
            "pm": 1e-6,
            "nm": 1e-3,
            "µm": 1.0,
            "μm": 1.0,
            "um": 1.0,
            "mm": 1e3,
            "cm": 1e4,
            "m": 1e6,
        }
        values = []
        for axis in ("X", "Y"):
            size = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
            factor = factors.get(unit)
            if factor is None or not np.isfinite(size) or size <= 0:
                return None
            values.append(size * factor)
        return tuple(values)
    except (KeyError, OSError, ValueError, TypeError, ET.ParseError):
        return None


def physical_matching_scale(info, mode):
    """Return moving-level px / fixed-level px in physical image coordinates."""
    roles = _roles(info, mode)
    moving_size = ome_physical_pixel_size(roles.moving.path)
    fixed_size = ome_physical_pixel_size(roles.fixed.path)
    if moving_size is None or fixed_size is None:
        return None

    moving_downsample = float(2 ** int(roles.moving.level))
    fixed_downsample = float(2 ** int(roles.fixed.level))
    scale_x = moving_size[0] * moving_downsample / (fixed_size[0] * fixed_downsample)
    scale_y = moving_size[1] * moving_downsample / (fixed_size[1] * fixed_downsample)
    scale = float(np.sqrt(scale_x * scale_y))
    return scale if np.isfinite(scale) and scale > 0 else None


def dimension_matching_scale(src_path, dst_path):
    """Estimate a fallback scale from image spans when OME spacing is absent."""
    src = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    dst = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)
    if src is None or dst is None:
        return None

    scale = max(dst.shape[:2]) / float(max(src.shape[:2]))
    if not np.isfinite(scale) or scale <= 0 or np.isclose(scale, 1.0, rtol=0.05):
        return None
    return scale


BRIGHTFIELD_PIXEL_SIZES_UM = ((0.25, "40x"), (0.50, "20x"))
DAPI_REFERENCE_PIXEL_SIZE_UM = 0.2125


def _append_matching_scale(scales, sources, value, source):
    """Append one positive scale unless the same hypothesis is already present."""
    if value is None:
        return
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return
    if any(np.isclose(value, existing, rtol=1e-6, atol=0.0) for existing in scales):
        return
    scales.append(value)
    sources[value] = source


def magnification_matching_scales(info, mode, src_path, dst_path):
    """Scale hypotheses to try when either side lacks physical pixel size.

    A real moving-image header remains authoritative. If the moving DAPI header is
    absent, use Xenium's 0.2125 um/px as a *matching-only* hypothesis; the Stage-2
    homography and the other image's real header still determine the recovered MPP
    recorded by later stages. If the H&E header is absent, try ordinary 40x and 20x
    brightfield pitches. For H&E-to-H&E, preserve each known header and enumerate
    40x/20x pitches only for missing sides (moving candidates outermost), deduplicating
    their relative scales. DAPI-to-DAPI starts with direct-grid matching, followed
    by a valid image-span fallback; H&E-to-DAPI retains direct-grid matching only.
    The image-span ratio is the last backstop for the brightfield ladders.

    Dividing the two images' long sides, which used to be the only fallback, assumes the
    tissue fills both frames to the same extent. That is a coincidence rather than a
    fact, and it is why it is now tried after the two it cannot justify itself against.
    It is kept because it is the only hypothesis that can reach a non-standard pitch:
    the mouse-brain sections are a scan whose pixel is near neither 0.25 nor 0.50, and
    40x/20x alone would fail them.

    Returns `(scales, sources)`: the scales in `physical_matching_scale`'s convention --
    moving-level pixels per fixed-level pixel, so a caller can treat them the same way --
    and a map from each to the hypothesis it came from, for the run record.
    """
    roles = _roles(info, mode)
    if roles.moving.modality == "dapi" and roles.fixed.modality == "dapi":
        # Missing MPP does not justify a microscopy-specific pitch prior. Try the
        # selected grids first, then span as a matching hypothesis, not calibration.
        scales, sources = [1.0], {1.0: "selected_levels"}
        _append_matching_scale(
            scales, sources, dimension_matching_scale(src_path, dst_path),
            "image_dimensions",
        )
        return scales, sources
    if roles.fixed.modality != "he":
        # Fluorescence is not a brightfield scanner. A crop's image spans also
        # say nothing about its pixel spacing: match the selected grids directly.
        return [None], {None: "selected_levels"}
    moving_size = ome_physical_pixel_size(roles.moving.path)
    fixed_size = ome_physical_pixel_size(roles.fixed.path)
    if roles.moving.modality == "he":
        # Symmetric missing-MPP hypotheses for H&E only. Do not overwrite metadata
        # or apply the DAPI/Xenium prior to a brightfield image.
        def pitches(size):
            if size is not None:
                return [(float(np.sqrt(size[0] * size[1])), "ome")]
            return BRIGHTFIELD_PIXEL_SIZES_UM

        scales, sources = [], {}
        level_ratio = float(2 ** int(roles.moving.level)) / float(
            2 ** int(roles.fixed.level)
        )
        for moving_pitch, moving_label in pitches(moving_size):
            for fixed_pitch, fixed_label in pitches(fixed_size):
                scale = moving_pitch / fixed_pitch * level_ratio
                source = f"he_moving_{moving_label}_fixed_{fixed_label}"
                _append_matching_scale(scales, sources, scale, source)
        _append_matching_scale(
            scales, sources, dimension_matching_scale(src_path, dst_path),
            "image_dimensions",
        )
        return (scales or [None]), sources

    scales, sources = [], {}
    if moving_size is None:
        if roles.moving.modality != "dapi":
            return [None], {None: "selected_levels"}
        moving_mpp = DAPI_REFERENCE_PIXEL_SIZE_UM
        moving_source = "assumed_dapi_reference"
        print(
            "[Stage2-CLI] moving DAPI MPP unavailable; using 0.2125 um/px "
            "as a matching hypothesis",
            flush=True,
        )
    else:
        moving_mpp = float(np.sqrt(moving_size[0] * moving_size[1]))
        moving_source = "moving_ome"

    moving_mpp *= float(2 ** int(roles.moving.level))
    fixed_downsample = float(2 ** int(roles.fixed.level))
    if fixed_size is not None:
        fixed_mpp = float(np.sqrt(fixed_size[0] * fixed_size[1]))
        scale = moving_mpp / (fixed_mpp * fixed_downsample)
        source = (
            "fixed_ome_plus_dapi_reference"
            if moving_source == "assumed_dapi_reference"
            else "ome_metadata"
        )
        _append_matching_scale(scales, sources, scale, source)
        print(
            f"[Stage2-CLI] fixed OME + moving hypothesis -> matching scale "
            f"{scale:.4f}",
            flush=True,
        )
    else:
        for pitch, label in BRIGHTFIELD_PIXEL_SIZES_UM:
            scale = moving_mpp / (pitch * fixed_downsample)
            source = f"magnification_{label}"
            if moving_source == "assumed_dapi_reference":
                source += "_plus_dapi_reference"
            _append_matching_scale(scales, sources, scale, source)
            print(f"[Stage2-CLI] hypothesis {label}: H&E {pitch} um/px "
                  f"-> matching scale {scale:.4f}", flush=True)
    span = dimension_matching_scale(src_path, dst_path)
    before = len(scales)
    _append_matching_scale(scales, sources, span, "image_dimensions")
    if len(scales) != before:
        print(f"[Stage2-CLI] hypothesis image-span: matching scale {span:.4f}",
              flush=True)
    # `_matching_pixel_grids` treats a scale within 5% of 1.0 as "no rescale", so the
    # first hypothesis doubles as the direct comparison whenever it lands there.
    return (scales or [None]), sources


def matching_scale_hypotheses(info, mode, src_path, dst_path):
    """Return every ordered scale Stage 2 should try and its provenance.

    A complete pair starts with the exact OME-derived scale, but feature extraction
    is sensitive to the resampling grid: a correct scale can land one correspondence
    below the quality gate. Therefore image span remains a fallback even when both
    headers are present. Missing-header cases use the symmetric hypotheses above.
    """
    physical_scale = physical_matching_scale(info, mode)
    if physical_scale is None:
        scales, sources = magnification_matching_scales(
            info, mode, src_path, dst_path
        )
        return physical_scale, scales, sources

    scales, sources = [], {}
    _append_matching_scale(scales, sources, physical_scale, "ome_metadata")
    span = dimension_matching_scale(src_path, dst_path)
    _append_matching_scale(scales, sources, span, "image_dimensions")
    return physical_scale, scales, sources


def _matching_pixel_grids(src, dst, physical_scale):
    """Downsample the finer physical grid; return images and level->match affines."""
    scale = 1.0 if physical_scale is None else float(physical_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("physical_scale must be positive and finite")

    src_to_match = np.eye(3, dtype=np.float64)
    dst_to_match = np.eye(3, dtype=np.float64)
    if np.isclose(scale, 1.0, rtol=0.05):
        return src, dst, src_to_match, dst_to_match

    if scale < 1.0:
        height, width = src.shape[:2]
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        src = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_AREA)
        src_to_match[0, 0] = new_width / width
        src_to_match[1, 1] = new_height / height
    else:
        height, width = dst.shape[:2]
        new_width = max(1, round(width / scale))
        new_height = max(1, round(height / scale))
        dst = cv2.resize(dst, (new_width, new_height), interpolation=cv2.INTER_AREA)
        dst_to_match[0, 0] = new_width / width
        dst_to_match[1, 1] = new_height / height

    return src, dst, src_to_match, dst_to_match


FLIPS = (None, "v", "h")


def _flip_to_original(flip, shape):
    """3x3 mapping original moving-image px -> flipped px, for the given flip."""
    height, width = shape[:2]
    if flip == "v":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    if flip == "h":
        return np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.eye(3, dtype=np.float64)


def feature_homography(method, src_path, dst_path, invert_src,
                       n_features=8000, ratio=0.75, ransac_thresh=5.0,
                       random_seed=43, physical_scale=None, flip=None):
    """Fit src-image px -> dst-image px with `method`.

    `flip` reflects the moving image before matching -- "v" about the horizontal
    axis, "h" about the vertical one. ORB and SIFT are rotation-invariant and are
    not mirror-invariant: both estimate a dominant orientation and encode the
    gradient pattern relative to it, and a reflection reverses that pattern, so
    the same nucleus produces descriptors that do not resemble each other.
    Measured on `cosmx_100_percent`, whose DAPI and H&E are mirror images: 26
    matches survive the ratio test unflipped against 199-208 flipped, and the
    fits collapse (axis ratio 0.066-0.62) against clean similarities (0.998+).
    All four rotations behave alike within each handedness, which is why only the
    reflection is searched here and the rotation is left to the homography.

    The returned `H` always maps the *original* moving-level image, with the flip
    composed in, so nothing downstream needs to know a flip happened.

    Returns a diagnostics dict: H (Stage-1 image px), H_scaled (matching px),
    n_matches, n_inliers, inlier_ratio, reproj_px (median over inliers),
    det and axis_ratio (shape of the linear part), flip.
    """
    cv2.setRNGSeed(int(random_seed))
    # Preserve 16-bit DAPI pixels until _prep performs one monotonic conversion
    # to the 8-bit representation required by ORB/SIFT.
    src = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    dst = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)
    if src is None or dst is None:
        raise FileNotFoundError(f"could not read {src_path} or {dst_path}")

    original_to_flipped = _flip_to_original(flip, src.shape)
    if flip == "v":
        src = np.ascontiguousarray(np.flipud(src))
    elif flip == "h":
        src = np.ascontiguousarray(np.fliplr(src))

    # Independently selected pyramid levels need not have comparable physical
    # pixel sizes. Match the finer image on the coarser physical grid, but keep
    # the complete selected-level coordinate systems for the saved transform.
    src, dst, src_to_match, dst_to_match = _matching_pixel_grids(
        src, dst, physical_scale
    )
    g_src = _prep(src, invert=invert_src)
    g_dst = _prep(dst, invert=False)

    det_obj, norm = _detector(method, n_features)
    k1, d1 = det_obj.detectAndCompute(g_src, None)
    k2, d2 = det_obj.detectAndCompute(g_dst, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise RuntimeError(f"too few {method.upper()} features detected")

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"only {len(good)} good {method.upper()} matches (need >=4)")

    pts_src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_s, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    if H_s is None:
        raise RuntimeError("findHomography returned None")
    H_s = H_s.astype(np.float64)
    inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), bool)
    n_inl = int(inl.sum())

    # src_to_match works in flipped pixels, so the flip is composed on its right to
    # bring the transform back onto the original moving-level image.
    H_full = np.linalg.inv(dst_to_match) @ H_s @ src_to_match @ original_to_flipped
    if abs(H_full[2, 2]) > 1e-12:
        H_full = H_full / H_full[2, 2]

    # Median residual of the inliers, in the actual Stage-1 H&E pixels.
    if n_inl:
        original_src = cv2.perspectiveTransform(
            pts_src[inl],
            np.linalg.inv(src_to_match @ original_to_flipped),
        )
        original_dst = cv2.perspectiveTransform(
            pts_dst[inl], np.linalg.inv(dst_to_match)
        )
        proj = cv2.perspectiveTransform(original_src, H_full).reshape(-1, 2)
        reproj = float(np.median(np.linalg.norm(
            proj - original_dst.reshape(-1, 2), axis=1
        )))
    else:
        reproj = float("inf")

    return {
        "method": method,
        "H": H_full,
        "H_scaled": H_s,
        "n_matches": len(good),
        "n_inliers": n_inl,
        "inlier_ratio": n_inl / float(len(good)),
        "reproj_px": reproj,
        "det": float(np.linalg.det(H_full[:2, :2])),
        # Singular values separate the two things `det` conflates: a mirror flips its
        # sign while leaving them alike, a collapse drives the smaller one to zero.
        "axis_ratio": float(_axis_ratio(H_full[:2, :2])),
        "physical_scale": physical_scale,
        "flip": flip,
        "src_matching_scale": (float(src_to_match[0, 0]), float(src_to_match[1, 1])),
        "dst_matching_scale": (float(dst_to_match[0, 0]), float(dst_to_match[1, 1])),
    }


def _axis_ratio(linear):
    """Smaller singular value over larger: 1 is a similarity, 0 is a collapsed axis."""
    try:
        values = np.linalg.svd(np.asarray(linear, dtype=np.float64), compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    if values.size < 2 or not np.isfinite(values).all() or values[0] <= 0:
        return 0.0
    return float(values[1] / values[0])


def check_fit(fit, cfg):
    """Return a list of reasons the fit is untrustworthy (empty list == reliable)."""
    reasons = []
    if fit["n_inliers"] < cfg["min_inliers"]:
        reasons.append(f"{fit['n_inliers']} inliers < {cfg['min_inliers']}")
    if fit["inlier_ratio"] < cfg["min_inlier_ratio"]:
        reasons.append(f"inlier ratio {fit['inlier_ratio']:.2f} < {cfg['min_inlier_ratio']}")
    if not (fit["reproj_px"] <= cfg["max_reproj_px"]):
        reasons.append(f"median reprojection {fit['reproj_px']:.2f} px "
                       f"> {cfg['max_reproj_px']} px")
    # `det <= 0` used to reject here, which conflated a collapsed fit with a mirrored
    # one. A mirror is a fact about the section -- CosMx's DAPI and H&E are mirror
    # images, and a hand alignment of that pair has det -0.27 -- so rejecting every
    # negative determinant made the pipeline unable to align a whole class of data
    # unless a person set the flip in Stage 1 first, which only the GUI offers.
    #
    # What actually distinguishes them is the shape of the linear part. Across every
    # accepted fit in this tree the two singular values agree to within 4%
    # (axis_ratio 0.963-1.000), mirrored ones included; a collapsed fit drives the
    # smaller one toward zero -- the rejected CosMx candidate at det -0.00171 is that.
    axis_ratio = fit.get("axis_ratio")
    if axis_ratio is None:
        axis_ratio = _axis_ratio(np.asarray(fit["H"])[:2, :2])
    if not np.isfinite(fit["det"]) or fit["det"] == 0 or axis_ratio < cfg["min_axis_ratio"]:
        reasons.append(
            f"degenerate linear part (det={fit['det']:.3g}, "
            f"axis ratio {axis_ratio:.3g} < {cfg['min_axis_ratio']})")
    return reasons


def orientation_affine(info, mode):
    """Return Stage 1's original-level -> oriented-level 3x3 affine."""
    roles = _roles(info, mode)
    key = f"{roles.pair.key_prefix(run_roles.MOVING)}_gui_affine"
    affine = roles.moving.gui_affine
    if affine is None:
        affine = info.get(key, np.eye(3))
    A = np.asarray(affine, dtype=np.float64)
    if A.shape == (2, 3):
        A = np.vstack([A, [0.0, 0.0, 1.0]])
    if A.shape != (3, 3):
        raise ValueError(f"{key} must be 2x3 or 3x3, got {A.shape}")
    if not np.isfinite(A).all() or abs(np.linalg.det(A)) < 1e-12:
        raise ValueError(f"{key} must be a finite, invertible affine")
    return A


def save_overlay(src_path, dst_path, H, out_path, invert_src, modality="dapi"):
    """Draw the moving image under the fitted transform, over the fixed one.

    The rendering is the moving modality's (see `overlays.py`): a DAPI image is
    windowed on percentiles and added into green, an H&E image is alpha-blended as
    the picture it is. `invert_src` is the *detector's* polarity match and is not
    applied here: it would turn a DAPI image's empty background white and paint out
    the fixed image under the whole warped rectangle.

    The moving image is read as Stage 1 wrote it. For a 16-bit DAPI that matters:
    `cv2.imread(..., IMREAD_COLOR)` drops the low byte, so a slide peaking at 8,297
    of 65,535 arrived at 32 of 255 and was invisible.
    """
    bg = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)
    fg = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if bg is None or fg is None:
        return
    if bg.dtype != np.uint8:
        bg = percentile_stretch_to_u8(bg)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    if modality == "dapi":
        if fg.ndim == 3 and fg.shape[2] >= 3:
            fg = cv2.cvtColor(fg[..., :3], cv2.COLOR_BGR2GRAY)
        elif fg.ndim == 3:
            fg = fg[..., 0]
    elif fg.ndim == 3 and fg.shape[2] >= 3:
        fg = cv2.cvtColor(fg[..., :3], cv2.COLOR_BGR2RGB)   # the renderer expects RGB
    cv2.imwrite(str(out_path), overlays.overlay(bg, fg, H, modality))


def stage2_config(engine_dir: Path, mode: str, method: str):
    """Shared stage-2 params from parameters.json, overlaid with the per-method block."""
    block = run_roles.coerce_pair(mode).params_block
    params = load_json(engine_dir.parent / "parameters.json").get(block, {}).get("stage2", {})
    cfg = dict(STAGE2_DEFAULTS)
    cfg.update({k: v for k, v in params.items() if k not in ("orb", "sift")})
    cfg.update(params.get(method, {}))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    run_roles.add_pair_arguments(ap)
    ap.add_argument("--methods", default=None,
                    help="comma-separated detector cascade (default: parameters.json "
                         "['stage2']['methods'], i.e. orb,sift)")
    ap.add_argument("--no-manual-fallback", action="store_true",
                    help="the caller has manual alignment switched off, so report a "
                         "failure as terminal instead of pointing at the GUI")
    ap.add_argument("--min-inliers", type=int, default=None,
                    help="override the minimum RANSAC inlier count for every method")
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    info = load_json(run_dir / "images_info.json")
    pair = run_roles.pair_from_args(args, run_dir=run_dir)
    roles = _roles(info, pair.name)
    original_to_oriented = orientation_affine(info, roles)

    # The Stage-1 level images are named by token. The moving one is the intensity
    # image prepared the way the matcher prepares it, not the Glasbey LUT: the LUT is
    # a categorical palette that renders continuous intensity as noise, and it exists
    # only when `qc_images.stage1_extras` is on.
    src_path = run_roles.artifact_path(run_dir, run_roles.MOVING, "1_{token}_level_image.png")
    overlay_src_path = src_path
    moving_level = int(roles.moving.level)
    # A fluorescence image is bright on dark; a brightfield one is the reverse. The
    # detector matches polarity by inverting the moving image when the two differ.
    invert_src = roles.moving.modality != roles.fixed.modality
    dst_path = run_roles.artifact_path(run_dir, run_roles.FIXED, "1_{token}_level_image.png")
    he_level = int(roles.fixed.level)
    print(f"[Stage2-CLI] pair {roles.pair.name}: {src_path.name} -> {dst_path.name}",
          flush=True)
    if not src_path.exists():
        message = f"Missing Stage-1 registration image: {src_path}. Rerun Stage 1."
        if pair.name == "dapi-he":
            message += " DAPI matching no longer uses the visualization LUT."
        raise SystemExit(message)
    physical_scale, matching_scales, scale_sources = matching_scale_hypotheses(
        info, roles, src_path, dst_path
    )

    if args.methods:
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    else:
        methods = stage2_config(engine_dir, pair.name, "")["methods"]

    banner(f"STAGE 2 (headless {'/'.join(m.upper() for m in methods)}, pair={pair.name})")
    if physical_scale is not None:
        print(
            f"[Stage2-CLI] moving/H&E physical pixel scale: {physical_scale:.4f} "
            "(matching on the coarser pixel grid)",
            flush=True,
        )
    else:
        print(
            "[Stage2-CLI] physical pixel metadata unavailable; "
            "using the fallback scale ladder",
            flush=True,
        )
    log_stage_event(run_dir, "stage2_events", "cli_stage2_orb_start",
                    pair=pair.name, methods=methods,
                    physical_pixel_scale=physical_scale,
                    matching_scales=matching_scales,
                    matching_scale_sources={str(key): value
                                            for key, value in scale_sources.items()})

    # ---- cascade: take the first method whose fit passes the quality gate.
    #      A low-inlier / high-residual fit means the detector could not find a
    #      trustworthy correspondence (common for cross-modal DAPI<->H&E), so we
    #      fall through to the next one rather than write a garbage alignment. ----
    fit = None
    attempts = []
    # Module A is the scale ladder above; module B reflects the moving image and runs
    # module A again. The reflection is outermost because it is the more expensive
    # assumption to be wrong about -- a wrong scale still matches some features, a wrong
    # handedness matches almost none -- and because it is the rarer case, so the datasets
    # that do not need it never pay for it.
    for flip in FLIPS:
        if flip is not None:
            print(f"[Stage2-CLI] every scale hypothesis failed unflipped; "
                  f"retrying with the moving image flipped "
                  f"{'vertically' if flip == 'v' else 'horizontally'}", flush=True)
        for scale_index, matching_scale in enumerate(matching_scales):
            if scale_index:
                print(
                    f"[Stage2-CLI] retrying at scale {matching_scale:.4f} "
                    f"({scale_sources.get(matching_scale, 'unknown hypothesis')}) "
                    "because the previous scale hypothesis was rejected",
                    flush=True,
                )
            for method in methods:
                cfg = stage2_config(engine_dir, pair.name, method)
                if args.min_inliers is not None:
                    cfg["min_inliers"] = args.min_inliers
                try:
                    cand = feature_homography(
                        method, src_path, dst_path, invert_src=invert_src,
                        n_features=int(cfg["n_features"]), ratio=float(cfg["ratio"]),
                        ransac_thresh=float(cfg["ransac_thresh"]),
                        random_seed=int(cfg["random_seed"]),
                        physical_scale=matching_scale, flip=flip)
                except (RuntimeError, ValueError, cv2.error) as exc:
                    print(f"[Stage2-CLI] {method.upper()}: FAILED ({exc})", flush=True)
                    attempts.append({
                        "method": method,
                        "matching_scale": matching_scale,
                        "flip": flip,
                        "reasons": [str(exc)],
                    })
                    log_stage_event(run_dir, "stage2_events", "cli_stage2_method_rejected",
                                    method=method, reasons=[str(exc)])
                    continue

                reasons = check_fit(cand, cfg)
                label = method.upper() + ("" if flip is None else f" flip-{flip}")
                print(f"[Stage2-CLI] {label}: {cand['n_inliers']}/{cand['n_matches']} inliers "
                      f"(ratio {cand['inlier_ratio']:.2f}), median reprojection "
                      f"{cand['reproj_px']:.2f} px -> {'OK' if not reasons else 'REJECTED'}", flush=True)
                attempts.append({"method": method, "n_matches": cand["n_matches"],
                                 "n_inliers": cand["n_inliers"],
                                 "inlier_ratio": round(cand["inlier_ratio"], 4),
                                 "reproj_px": round(cand["reproj_px"], 3),
                                 "matching_scale": matching_scale,
                                 "flip": flip,
                                 "reasons": reasons})
                if not reasons:
                    fit = cand
                    break
                for r in reasons:
                    print(f"[Stage2-CLI]   - {r}", flush=True)
                log_stage_event(run_dir, "stage2_events", "cli_stage2_method_rejected",
                                method=method, n_inliers=cand["n_inliers"],
                                inlier_ratio=round(cand["inlier_ratio"], 4),
                                reproj_px=round(cand["reproj_px"], 3), reasons=reasons)
            if fit is not None:
                break
        if fit is not None:
            break

    if fit is None:
        log_stage_event(run_dir, "stage2_events", "cli_stage2_orb_failed",
                        attempts=attempts,
                        manual_fallback=not args.no_manual_fallback)
        detail = "\n".join(f"    {a['method'].upper()}: " + "; ".join(a["reasons"])
                           for a in attempts)
        moving_flag = f"--pair {pair.name} --moving"
        moving_image = roles.moving.path
        # Point at the fail-safe launcher, not at run_pharaoh_gui.sh: this RUN_DIR
        # already exists and only needs its initial alignment set, after which the
        # same run continues from Stage 3.
        remedy = (
            "  Manual alignment is switched off for this run (--stage2-nomanual), so "
            "this pair has no initial alignment.\n\n"
            if args.no_manual_fallback else
            "  Set the initial alignment by hand, then let this run finish from Stage 3:\n"
            f"      ./run_pharaoh_failsafe.sh --run-dir {run_dir} \\\n"
            f"          {moving_flag} {moving_image} --fixed {roles.fixed.path}\n"
            "  From a batch, every queued dataset is done in one pass instead:\n"
            "      ./run_pharaoh_failsafe.sh <batch output root>\n\n"
        )
        sys.stderr.write(
            f"\n[Stage2-CLI] ERROR: automatic alignment failed for every detector "
            f"({', '.join(m.upper() for m in methods)}).\n{detail}\n"
            f"  The automatic registration is not reliable for this image pair "
            f"(often the case for cross-modal DAPI<->H&E data).\n"
            f"{remedy}")
        raise SystemExit(2)

    # fit["H"] maps the already-oriented Stage-1 PNG to H&E. Later stages use
    # coordinates from the original moving image, so restore that convention:
    # original -> Stage-1 orientation -> H&E.
    H_oriented_to_he = fit["H"]
    H_mat = compose_with_orientation(H_oriented_to_he, original_to_oriented)
    print(f"[Stage2-CLI] using {fit['method'].upper()} homography: "
          f"{fit['n_inliers']} inliers", flush=True)

    H_mat_level_0 = lift_to_level0(H_mat, moving_level, he_level)

    data = {
        "active_mode": "original",
        "mask_pose": None,
        "original_pose": None,
        "auto_alignment": {
            "method": fit["method"].upper(),
            "random_seed": int(stage2_config(engine_dir, pair.name, fit["method"])["random_seed"]),
            "n_inliers": int(fit["n_inliers"]),
            "n_matches": int(fit["n_matches"]),
            "inlier_ratio": round(fit["inlier_ratio"], 4),
            "reproj_px": round(fit["reproj_px"], 3),
            "attempts": attempts,
            "matching_image": src_path.name,
            "visualization_image": overlay_src_path.name,
            "physical_pixel_scale": physical_scale,
            "applied_matching_scale": fit.get("physical_scale"),
            "applied_flip": fit.get("flip"),
            "matching_scale_source": scale_sources.get(
                fit.get("physical_scale"),
                "ome_metadata" if physical_scale is not None else "none"),
            "matching_src_scale": list(fit.get("src_matching_scale", (1.0, 1.0))),
            "matching_dst_scale": list(fit.get("dst_matching_scale", (1.0, 1.0))),
            "matched_coordinate_space": ("stage1_oriented_level_image_to_he_level_image"
                                         if pair.name == "dapi-he" else "stage1_oriented_level_image_to_fixed_level_image"),
            "saved_coordinate_space": ("original_moving_level_image_to_he_level_image"
                                       if pair.name == "dapi-he" else "original_moving_level_image_to_fixed_level_image"),
            "orientation_affine_composed": original_to_oriented.tolist(),
        },
        "H_mat": H_mat.tolist(),
        "H_mat_level_0": H_mat_level_0.tolist(),
    }
    save_json(run_dir / INITIAL_ALIGNMENT, data)
    print(f"[Stage2-CLI] wrote {INITIAL_ALIGNMENT}", flush=True)

    # The overlay source is the oriented PNG saved by Stage 1, so preview it
    # with the direct fit (not the original-coordinate H_mat).
    # Automatic and interactive entry points share the same initial-overlay name.
    save_overlay(overlay_src_path, dst_path, H_oriented_to_he,
                 run_dir / INITIAL_OVERLAY, invert_src,
                 modality=roles.moving.modality)

    log_stage_event(run_dir, "stage2_events", "cli_stage2_orb_done",
                    method=fit["method"].upper(), inliers=int(fit["n_inliers"]))


if __name__ == "__main__":
    main()
