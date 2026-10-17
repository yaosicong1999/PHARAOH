"""Stage-2 matrix conventions shared by automatic and manual alignment."""
import math

import numpy as np


def compose_with_orientation(H_oriented_to_he, original_to_oriented):
    """Convert an oriented-image fit to the original moving-image convention."""
    A = np.asarray(original_to_oriented, dtype=np.float64)
    if A.shape == (2, 3):
        A = np.vstack([A, [0, 0, 1]])
    H = np.asarray(H_oriented_to_he, dtype=np.float64) @ A
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def lift_to_level0(H, moving_level, fixed_level):
    """Lift original moving -> fixed level-image coordinates to level zero."""
    sd, sh = float(2 ** moving_level), float(2 ** fixed_level)
    S_d = np.diag([sd, sd, 1.0])
    S_h = np.diag([sh, sh, 1.0])
    return S_h @ np.asarray(H, dtype=np.float64) @ np.linalg.inv(S_d)


ASSUMED_COARSE_MPP_20X = 0.50
MIN_PLAUSIBLE_MPP_80X = 0.125


def _scalar_mpp(value, label):
    values = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    scalar = float(np.sqrt(np.prod(values)) if values.size > 1 else values[0]) \
        if values.size else None
    if scalar is None:
        raise ValueError(f"{label} image has no usable MPP for Stage-2 scale inference")
    return scalar


def estimate_level0_pixel_scale_from_homography(
    homography_level0,
    moving_shape,
    fixed_shape,
    *,
    grid_size=9,
    min_samples=9,
    max_scale_spread=1.5,
    min_axis_ratio=0.5,
):
    """Measure fixed pixels per moving pixel from a level-0 homography.

    One moving pixel spans ``sqrt(abs(det(J)))`` fixed pixels locally, where ``J``
    is the homography Jacobian. A robust median over the image overlap avoids
    treating a single projective location as the scanner scale.

    The estimate is intentionally conservative: a transform whose scale varies by
    more than ``max_scale_spread`` across the overlap, or whose local axes collapse
    below ``min_axis_ratio``, is registration/deformation rather than trustworthy
    physical-spacing evidence.
    """
    H = np.asarray(homography_level0, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("Stage-2 H_mat_level_0 must be a finite 3x3 matrix")
    if abs(float(H[2, 2])) <= 1e-12:
        raise ValueError("Stage-2 H_mat_level_0 has a zero projective denominator")
    H = H / H[2, 2]

    moving_h, moving_w = (int(v) for v in moving_shape[:2])
    fixed_h, fixed_w = (int(v) for v in fixed_shape[:2])
    if min(moving_h, moving_w, fixed_h, fixed_w) <= 1:
        raise ValueError("image shapes must contain positive level-0 height and width")
    # Exclude the exact border, where a mild projective term is least representative.
    xs = np.linspace(0.05 * (moving_w - 1), 0.95 * (moving_w - 1), int(grid_size))
    ys = np.linspace(0.05 * (moving_h - 1), 0.95 * (moving_h - 1), int(grid_size))
    scales = []
    axis_ratios = []
    determinant_signs = []
    for y in ys:
        for x in xs:
            numerator_u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
            numerator_v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
            denominator = H[2, 0] * x + H[2, 1] * y + H[2, 2]
            if not np.isfinite(denominator) or abs(float(denominator)) <= 1e-12:
                continue
            u = numerator_u / denominator
            v = numerator_v / denominator
            if not (0.0 <= u < fixed_w and 0.0 <= v < fixed_h):
                continue
            jacobian = np.array([
                [(H[0, 0] - u * H[2, 0]) / denominator,
                 (H[0, 1] - u * H[2, 1]) / denominator],
                [(H[1, 0] - v * H[2, 0]) / denominator,
                 (H[1, 1] - v * H[2, 1]) / denominator],
            ], dtype=np.float64)
            determinant = float(np.linalg.det(jacobian))
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if (not np.isfinite(determinant) or abs(determinant) <= 1e-18
                    or not np.all(np.isfinite(singular_values))
                    or singular_values[0] <= 0):
                continue
            scales.append(math.sqrt(abs(determinant)))
            axis_ratios.append(float(singular_values[-1] / singular_values[0]))
            determinant_signs.append(1 if determinant > 0 else -1)

    if len(scales) < int(min_samples):
        raise ValueError(
            "Stage-2 homography has too little valid image overlap to estimate MPP "
            f"({len(scales)} samples; need {int(min_samples)})"
        )
    if len(set(determinant_signs)) != 1:
        raise ValueError("Stage-2 homography changes orientation inside the image overlap")

    scales = np.asarray(scales, dtype=np.float64)
    axis_ratios = np.asarray(axis_ratios, dtype=np.float64)
    scale_p05, scale_median, scale_p95 = np.percentile(scales, [5, 50, 95])
    spread = float(scale_p95 / scale_p05) if scale_p05 > 0 else math.inf
    axis_ratio_p05 = float(np.percentile(axis_ratios, 5))
    if not np.isfinite(spread) or spread > float(max_scale_spread):
        raise ValueError(
            "Stage-2 homography scale varies too much to estimate MPP "
            f"(p95/p05={spread:.3f}, limit={float(max_scale_spread):.3f})"
        )
    if axis_ratio_p05 < float(min_axis_ratio):
        raise ValueError(
            "Stage-2 homography is too anisotropic to estimate MPP "
            f"(axis-ratio p05={axis_ratio_p05:.3f}, limit={float(min_axis_ratio):.3f})"
        )

    return {
        "fixed_pixels_per_moving_pixel": float(scale_median),
        "scale_p05": float(scale_p05),
        "scale_p95": float(scale_p95),
        "scale_p95_over_p05": spread,
        "axis_ratio_p05": axis_ratio_p05,
        "sample_count": int(scales.size),
        "determinant_sign": int(determinant_signs[0]),
        "method": "median_level0_homography_jacobian_area_scale",
    }


def estimate_mpp_pair_from_homography(
    homography_level0,
    moving_shape,
    fixed_shape,
    *,
    moving_base_mpp=None,
    fixed_base_mpp=None,
    assumed_coarse_mpp=ASSUMED_COARSE_MPP_20X,
    **scale_options,
):
    """Resolve one or two missing MPP values from Stage 2's relative pixel scale.

    A real header is the absolute physical anchor whenever either side has one. If
    both headers are absent, the coarser side (the side with the larger inferred
    MPP) is conservatively assumed to be 20x, 0.50 um/px, and the other side is
    derived from the homography. The assumption creates an explicit absolute anchor;
    the homography alone can only determine the ratio between the two MPP values.
    """
    moving_mpp = None if moving_base_mpp is None else _scalar_mpp(
        moving_base_mpp, "moving")
    fixed_mpp = None if fixed_base_mpp is None else _scalar_mpp(
        fixed_base_mpp, "fixed")
    if moving_mpp is not None and fixed_mpp is not None:
        raise ValueError("both images already have usable MPP")

    estimate = estimate_level0_pixel_scale_from_homography(
        homography_level0, moving_shape, fixed_shape, **scale_options)
    scale = float(estimate["fixed_pixels_per_moving_pixel"])

    if fixed_mpp is not None:
        moving_mpp = float(fixed_mpp * scale)
        anchor_role = "fixed"
        derived_role = "moving"
        anchor_kind = "known_mpp"
    elif moving_mpp is not None:
        fixed_mpp = float(moving_mpp / scale)
        anchor_role = "moving"
        derived_role = "fixed"
        anchor_kind = "known_mpp"
    else:
        coarse_mpp = _scalar_mpp(assumed_coarse_mpp, "assumed coarse")
        if scale >= 1.0:
            # moving_mpp = fixed_mpp * scale, so moving is the coarser side.
            moving_mpp = coarse_mpp
            fixed_mpp = float(coarse_mpp / scale)
            anchor_role, derived_role = "moving", "fixed"
        else:
            fixed_mpp = coarse_mpp
            moving_mpp = float(coarse_mpp * scale)
            anchor_role, derived_role = "fixed", "moving"
        finer_mpp = min(moving_mpp, fixed_mpp)
        if finer_mpp < MIN_PLAUSIBLE_MPP_80X * (1.0 - 1e-6):
            raise ValueError(
                "Stage-2 relative scale plus the 20x Case-4 anchor would imply "
                f"MPP {finer_mpp:.6g} um/px, finer than the supported 80x limit "
                f"{MIN_PLAUSIBLE_MPP_80X:g} um/px"
            )
        anchor_kind = "assumed_20x_coarse_side"
        estimate["assumed_coarse_base_mpp"] = coarse_mpp

    estimate.update({
        "moving_base_mpp": float(moving_mpp),
        "fixed_base_mpp": float(fixed_mpp),
        "anchor_role": anchor_role,
        "derived_role": derived_role,
        "anchor_kind": anchor_kind,
    })
    return estimate


def estimate_moving_mpp_from_homography(
    homography_level0,
    moving_shape,
    fixed_shape,
    fixed_base_mpp,
    **scale_options,
):
    """Backward-compatible Case-2 wrapper: infer moving MPP from fixed MPP."""
    return estimate_mpp_pair_from_homography(
        homography_level0,
        moving_shape,
        fixed_shape,
        fixed_base_mpp=fixed_base_mpp,
        **scale_options,
    )


def estimate_fixed_mpp_from_homography(
    homography_level0,
    moving_shape,
    fixed_shape,
    moving_base_mpp,
    **scale_options,
):
    """Case 3: infer fixed MPP from a known moving MPP."""
    return estimate_mpp_pair_from_homography(
        homography_level0,
        moving_shape,
        fixed_shape,
        moving_base_mpp=moving_base_mpp,
        **scale_options,
    )
