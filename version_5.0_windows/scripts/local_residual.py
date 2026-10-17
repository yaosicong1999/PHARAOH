"""Compact-support local correction on top of Stage 2's global homography.

Density anchors in serial sections describe local conserved architecture.  They are
useful near the tile that produced them, but a small spatial cluster must never be
allowed to extrapolate a new projective transform across the whole slide.  This
module therefore keeps Stage 2 as the global map and interpolates *residual vectors*
with a finite-radius Wendland C2 kernel.  Outside every control point's support the
correction is exactly zero.

All stored coordinates are level-0 pixels in the fixed image.  Forward mapping is

    fixed = H_stage2(moving) + local_residual(H_stage2(moving))

and inverse mapping uses a few fixed-point iterations before applying ``H^-1``.
"""
from __future__ import annotations

import numpy as np


TRANSFORM_TYPE = "local_residual"


def project_homography(matrix, points):
    """Apply a 3x3 homography to an ``(N, 2)`` point array."""
    matrix = np.asarray(matrix, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    if np.any(~np.isfinite(denominator)) or np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("homography maps a local-residual coordinate to infinity")
    result = projected[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("homography produced a non-finite local-residual coordinate")
    return result


def _parts(transform):
    """Validate and return ``(base, controls, vectors, radius, weight, iterations)``."""
    if not isinstance(transform, dict):
        raise ValueError("local residual transform must be a JSON object")
    block = transform.get("local_residual", transform)
    base = transform.get("base_homography_3x3", transform.get("homography_3x3"))
    if base is None:
        base = block.get("base_homography_3x3")
    base = np.asarray(base, dtype=np.float64)
    controls = np.asarray(
        block.get("control_points_fixed_level0", []), dtype=np.float64
    ).reshape(-1, 2)
    vectors = np.asarray(
        block.get("residual_vectors_fixed_level0", []), dtype=np.float64
    ).reshape(-1, 2)
    if base.shape != (3, 3):
        raise ValueError(f"base_homography_3x3 must be 3x3, got {base.shape}")
    if vectors.shape != controls.shape:
        raise ValueError(
            f"local residual vectors must match controls, got {vectors.shape} and {controls.shape}"
        )
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(controls)) or not np.all(np.isfinite(vectors)):
        raise ValueError("local residual transform contains NaN or infinite values")
    radius = float(block.get("support_radius_level0_px", 0.0))
    baseline_weight = float(block.get("baseline_weight", 0.0))
    inverse_iterations = int(block.get("inverse_iterations", 3))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("local residual support radius must be positive")
    if not np.isfinite(baseline_weight) or baseline_weight <= 0:
        raise ValueError("local residual baseline weight must be positive")
    if inverse_iterations < 1:
        raise ValueError("local residual inverse_iterations must be >= 1")
    return base, controls, vectors, radius, baseline_weight, inverse_iterations


def displacement(points, controls, vectors, radius, baseline_weight, batch_size=200_000):
    """Evaluate the compact residual field at fixed-space level-0 ``points``.

    The positive ``baseline_weight`` is an always-present zero-displacement vote.  It
    damps isolated controls, and the compact kernel makes the result exactly zero at
    and beyond ``radius`` from every control.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    controls = np.asarray(controls, dtype=np.float64).reshape(-1, 2)
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    radius = float(radius)
    baseline_weight = float(baseline_weight)
    if len(controls) == 0:
        return np.zeros_like(points)
    if controls.shape != vectors.shape:
        raise ValueError("local residual controls and vectors must have the same shape")

    result = np.empty_like(points)
    radius2 = radius * radius
    for start in range(0, len(points), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), len(points))
        p = points[start:stop]
        total = np.zeros((len(p), 2), dtype=np.float64)
        weight_sum = np.full(len(p), baseline_weight, dtype=np.float64)
        for control, vector in zip(controls, vectors):
            delta = p - control
            distance2 = np.einsum("ij,ij->i", delta, delta)
            inside = distance2 < radius2
            if not np.any(inside):
                continue
            ratio = np.sqrt(distance2[inside]) / radius
            one_minus = 1.0 - ratio
            weight = one_minus ** 4 * (4.0 * ratio + 1.0)  # Wendland C2
            total[inside] += weight[:, None] * vector
            weight_sum[inside] += weight
        result[start:stop] = total / weight_sum[:, None]
    return result


def forward_points(points_moving_level0, transform):
    """Map moving level-0 points through the Stage-2 base plus local residual."""
    base, controls, vectors, radius, baseline_weight, _ = _parts(transform)
    baseline_fixed = project_homography(base, points_moving_level0)
    return baseline_fixed + displacement(
        baseline_fixed, controls, vectors, radius, baseline_weight
    )


def inverse_points(points_fixed_level0, transform):
    """Map fixed level-0 points back to moving level 0 for image remapping."""
    base, controls, vectors, radius, baseline_weight, iterations = _parts(transform)
    target = np.asarray(points_fixed_level0, dtype=np.float64).reshape(-1, 2)
    baseline_fixed = target.copy()
    for _ in range(iterations):
        correction = displacement(
            baseline_fixed, controls, vectors, radius, baseline_weight
        )
        baseline_fixed = target - correction
    return project_homography(np.linalg.inv(base), baseline_fixed)


def display_remap(transform, out_shape_hw, fixed_display_scale, moving_display_scale):
    """Build OpenCV ``remap`` arrays for a fixed-image display canvas."""
    height, width = map(int, out_shape_hw)
    gx, gy = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    fixed_level0 = np.column_stack([gx.ravel(), gy.ravel()]) * float(fixed_display_scale)
    moving_level0 = inverse_points(fixed_level0, transform)
    moving_display = moving_level0 / float(moving_display_scale)
    return (
        moving_display[:, 0].reshape(height, width).astype(np.float32),
        moving_display[:, 1].reshape(height, width).astype(np.float32),
    )
