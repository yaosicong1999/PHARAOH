# ============================================================
# overlays.py - the moving image drawn over the fixed one, by modality
#
# Two renderings existed, one per engine. A DAPI moving image is intensity: it
# is windowed on percentiles into a single channel and *added* into green over
# the untouched fixed image, so the fixed image survives wherever the moving one
# has no signal (the Glasbey LUT used to give that for free by being black
# outside the nuclei; blended at 0.7/0.8 it was unreadable). An H&E moving image
# is a picture: the old H&E-onto-H&E engine alpha-blended the warped RGB at 0.5
# over the fixed H&E, which is what lets a person see both stains at once.
#
# Which rendering applies is decided by what the moving image *is*, never by
# which stage is drawing or which engine it lives in. Stage 2's preview and
# Stage 6's final overlay both come through here.
#
# For a DAPI moving image the numbers are exactly the ones Stage 2 and Stage 6
# used, so a `dapi-he` overlay is byte-identical to before.
# ============================================================
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from my_utils import percentile_stretch_to_u8  # noqa: E402

#: How much of the warped moving image is added (DAPI) or mixed (H&E) into the fixed.
DAPI_TINT_WEIGHT = 0.8
HE_BLEND_ALPHA = 0.5


def check_modality(modality: str) -> str:
    modality = str(modality).strip().lower()
    if modality not in ("dapi", "he"):
        raise ValueError(f"modality must be 'dapi' or 'he', got {modality!r}")
    return modality


def moving_overlay_base(image, modality: str) -> np.ndarray:
    """The moving image as it should be warped for an overlay, as BGR uint8.

    DAPI: a percentile window of the single channel into green, background at
    zero. H&E: the RGB itself. A 16-bit H&E (rare, but `read_image` can hand one
    back) is scaled down to 8 bits.
    """
    modality = check_modality(modality)
    image = np.asarray(image)
    if modality == "dapi":
        gray = image[..., 0] if image.ndim == 3 else image
        if gray.dtype != np.uint8:
            gray = percentile_stretch_to_u8(gray)
        bgr = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        bgr[..., 1] = gray                       # green: the moving image
        return bgr
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb = image[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb.astype(np.float32) / 256.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


def blend(fixed_bgr, warped_bgr, modality: str) -> np.ndarray:
    """Combine the warped moving image with the fixed one, by the moving modality.

    DAPI is added on top of the whole fixed image; H&E is mixed half and half.
    """
    modality = check_modality(modality)
    if modality == "dapi":
        return cv2.addWeighted(fixed_bgr, 1.0, warped_bgr, DAPI_TINT_WEIGHT, 0)
    return cv2.addWeighted(fixed_bgr, 1.0 - HE_BLEND_ALPHA, warped_bgr, HE_BLEND_ALPHA, 0)


def warp(base_bgr, homography, shape_hw) -> np.ndarray:
    """Warp an overlay base into the fixed image's frame."""
    return cv2.warpPerspective(
        base_bgr, np.asarray(homography, dtype=np.float64),
        (int(shape_hw[1]), int(shape_hw[0])),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )


def overlay(fixed_bgr, moving_image, homography, modality: str) -> np.ndarray:
    """The moving image, prepared for its modality, warped by `homography` onto the
    fixed image and blended the way that modality reads best."""
    base = moving_overlay_base(moving_image, modality)
    return blend(fixed_bgr, warp(base, homography, fixed_bgr.shape[:2]), modality)
