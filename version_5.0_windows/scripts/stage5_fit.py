"""Stage 5's fit, shared by the CLI and anything else that fits the final transform:
the pair ordering, grid-balanced sample, stable model selection, and RANSAC settings.

No GUI here. `balanced_grid_sample_pairs` is the sampler the 4.10 Stage-5 gallery ran
when `balance_points_bool` was on; 4.11's CLI had only a warning where it should have
been (`he-he` ships with it on), and the RANSAC threshold, iteration cap and
confidence were literals in the CLI rather than parameters.
"""
import hashlib

import cv2
import numpy as np

import local_residual


DENSITY_LOCAL_DEFAULTS = {
    "enabled": True,
    # Radius is measured from the median Stage-3 tile width in level-0 pixels.
    # Six tile widths bridge nearby rich-tissue samples while remaining finite.
    "support_radius_tile_factor": 6.0,
    # An always-present zero-residual vote.  This damps isolated tiles and makes
    # every correction fade continuously back to Stage 2.
    "baseline_weight": 0.1,
    "inverse_iterations": 3,
    "min_tiles": 3,
    # Per-tile safety gates, also relative to the Stage-3 tile width.
    "max_residual_tile_factor": 0.75,
    "max_anchor_spread_tile_factor": 0.25,
    # Do not publish a local field unless it improves its own anchor evidence.
    "promotion_max_median_ratio": 0.95,
    "promotion_max_p90_ratio": 1.0,
}

STABLE_MODEL_DEFAULTS = {
    # Opt in per modality pair.  Version 4.14 enables this for DAPI -> H&E only.
    "enabled": False,
    # A homography is accepted only when grouped validation gives a material
    # improvement over the lower-variance affine model.
    "candidate_modes": ["affine", "homography"],
    "group_permutations": 32,
    "cv_folds": 5,
    "cv_group_permutations": 8,
    "homography_min_cv_improvement_fraction": 0.05,
    # Candidate choice balances tile consensus, spatial support, and stability of
    # the transform over the complete anchor extent (including its edges).
    "stability_grid_size": 7,
    "spatial_grid": [6, 6],
    "support_weight": 2.0,
    "spatial_weight": 1.0,
    "stability_weight": 1.0,
}

DEFAULTS = {
    "transform_mode": "homography",
    "random_seed": 43,
    "balance_points_bool": False,
    "balance_grid": [6, 6],
    "balance_max_per_cell": 4,
    # level-0 pixels of the fixed image at the reference pixel size; the CLI scales it
    # to the image at hand (scripts/physical_scale.py)
    "ransac_reproj_threshold_px": 8.0,
    "max_iters": 10000,
    "confidence": 0.995,
    "density_local_fit": DENSITY_LOCAL_DEFAULTS,
    "stable_model_selection": STABLE_MODEL_DEFAULTS,
}
VALID_TRANSFORMS = ("homography", "affine")


def load_stage5_params(block):
    """The effective Stage-5 settings from a pair's `stage5` block."""
    params = dict(DEFAULTS)
    block = dict(block or {})
    density_local = dict(DENSITY_LOCAL_DEFAULTS)
    density_local.update(block.pop("density_local_fit", {}) or {})
    stable_model = dict(STABLE_MODEL_DEFAULTS)
    stable_model.update(block.pop("stable_model_selection", {}) or {})
    params.update(block)
    params["density_local_fit"] = density_local
    params["stable_model_selection"] = stable_model
    params["transform_mode"] = str(params["transform_mode"]).lower().strip()
    params["random_seed"] = int(params["random_seed"])
    params["balance_points_bool"] = bool(params["balance_points_bool"])
    grid = params["balance_grid"]
    params["balance_grid"] = (max(1, int(grid[0])), max(1, int(grid[1])))
    params["balance_max_per_cell"] = max(1, int(params["balance_max_per_cell"]))
    params["ransac_reproj_threshold_px"] = float(params["ransac_reproj_threshold_px"])
    params["max_iters"] = int(params["max_iters"])
    params["confidence"] = float(params["confidence"])
    if params["ransac_reproj_threshold_px"] <= 0:
        raise ValueError("stage5.ransac_reproj_threshold_px must be positive")
    if not 0 < params["confidence"] < 1:
        raise ValueError("stage5.confidence must be between 0 and 1")
    density_local["enabled"] = bool(density_local["enabled"])
    density_local["support_radius_tile_factor"] = float(
        density_local["support_radius_tile_factor"]
    )
    density_local["baseline_weight"] = float(density_local["baseline_weight"])
    density_local["inverse_iterations"] = int(density_local["inverse_iterations"])
    density_local["min_tiles"] = int(density_local["min_tiles"])
    density_local["max_residual_tile_factor"] = float(
        density_local["max_residual_tile_factor"]
    )
    density_local["max_anchor_spread_tile_factor"] = float(
        density_local["max_anchor_spread_tile_factor"]
    )
    density_local["promotion_max_median_ratio"] = float(
        density_local["promotion_max_median_ratio"]
    )
    density_local["promotion_max_p90_ratio"] = float(
        density_local["promotion_max_p90_ratio"]
    )
    positive = (
        "support_radius_tile_factor", "baseline_weight", "inverse_iterations",
        "min_tiles", "max_residual_tile_factor", "max_anchor_spread_tile_factor",
        "promotion_max_median_ratio", "promotion_max_p90_ratio",
    )
    if any(density_local[key] <= 0 for key in positive):
        raise ValueError("stage5.density_local_fit numeric settings must be positive")
    stable_model["enabled"] = bool(stable_model["enabled"])
    stable_model["candidate_modes"] = [
        str(value).strip().lower() for value in stable_model["candidate_modes"]
    ]
    if not stable_model["candidate_modes"] or any(
        value not in VALID_TRANSFORMS for value in stable_model["candidate_modes"]
    ):
        raise ValueError(
            "stage5.stable_model_selection.candidate_modes must contain affine "
            "and/or homography"
        )
    stable_model["group_permutations"] = max(
        1, int(stable_model["group_permutations"])
    )
    stable_model["cv_folds"] = max(2, int(stable_model["cv_folds"]))
    stable_model["cv_group_permutations"] = max(
        1, int(stable_model["cv_group_permutations"])
    )
    stable_model["stability_grid_size"] = max(
        3, int(stable_model["stability_grid_size"])
    )
    grid = stable_model["spatial_grid"]
    stable_model["spatial_grid"] = (
        max(1, int(grid[0])), max(1, int(grid[1]))
    )
    for key in (
        "homography_min_cv_improvement_fraction", "support_weight",
        "spatial_weight", "stability_weight",
    ):
        stable_model[key] = float(stable_model[key])
        if stable_model[key] < 0:
            raise ValueError(f"stage5.stable_model_selection.{key} must be nonnegative")
    return params


def records_are_density(records):
    """Whether Stage 4b shipped architecture anchors rather than nucleus pairs."""
    modes = {str(record.get("mode", "")).strip().lower() for record in records}
    return bool(modes) and modes <= {"density", "nuclear_architecture_anchor"}


def infer_tile_size_level0(tile_info, tile_level):
    """Median Stage-3 tile width in level-0 pixels, or ``None`` when unavailable."""
    widths = []
    for value in (tile_info or {}).values():
        try:
            width = max(float(value["w"]), float(value["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(width) and width > 0:
            widths.append(width)
    if not widths:
        return None
    return float(np.median(widths) * (2 ** int(tile_level)))


def _error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.9)),
        "max_px": float(np.max(values)),
    }


def build_density_local_transform(records, base_homography, tile_size_level0_px, params):
    """Build a finite-support, one-control-per-tile density correction.

    Each tile contributes one median residual vector, regardless of whether Stage 4b
    emitted one, two, or three architecture anchors.  Badly inconsistent or extreme
    tile residuals are omitted.  The returned ``promoted`` flag says whether the local
    field improved the full anchor set enough to publish; a caller should retain the
    Stage-2 homography when it is false.
    """
    cfg = dict(DENSITY_LOCAL_DEFAULTS)
    cfg.update(params or {})
    base = np.asarray(base_homography, dtype=np.float64)
    tile_size = float(tile_size_level0_px)
    if base.shape != (3, 3):
        raise ValueError("density local fit requires a 3x3 Stage-2 homography")
    if not np.isfinite(tile_size) or tile_size <= 0:
        raise ValueError("density local fit requires a positive level-0 tile size")

    groups = {}
    for record in records:
        groups.setdefault(str(record.get("tile_id", "")), []).append(record)
    full_src = np.asarray([record["src"] for record in records], dtype=np.float64)
    full_dst = np.asarray([record["dst"] for record in records], dtype=np.float64)
    accepted = []
    rejected = []
    max_residual = float(cfg["max_residual_tile_factor"]) * tile_size
    max_spread = float(cfg["max_anchor_spread_tile_factor"]) * tile_size
    all_src = []
    all_dst = []

    for tile_id in sorted(groups):
        rows = groups[tile_id]
        src = np.asarray([row["src"] for row in rows], dtype=np.float64)
        dst = np.asarray([row["dst"] for row in rows], dtype=np.float64)
        baseline = local_residual.project_homography(base, src)
        vectors = dst - baseline
        control = np.median(baseline, axis=0)
        vector = np.median(vectors, axis=0)
        spread = float(np.max(np.linalg.norm(vectors - vector, axis=1)))
        magnitude = float(np.linalg.norm(vector))
        reasons = []
        if magnitude > max_residual:
            reasons.append(f"residual {magnitude:.3f}px > {max_residual:.3f}px")
        if spread > max_spread:
            reasons.append(f"anchor spread {spread:.3f}px > {max_spread:.3f}px")
        item = {
            "tile_id": tile_id,
            "anchor_count": int(len(rows)),
            "control_point_fixed_level0": control.tolist(),
            "residual_vector_fixed_level0": vector.tolist(),
            "residual_magnitude_px": magnitude,
            "anchor_spread_px": spread,
        }
        if reasons:
            item["rejection_reasons"] = reasons
            rejected.append(item)
        else:
            accepted.append(item)
            # Promotion is evaluated on the anchors whose tile passed the safety
            # gates.  A deliberately rejected tile must not veto an otherwise safe
            # local field merely because its untrusted points remain far away.
            all_src.append(src)
            all_dst.append(dst)

    if all_src:
        src = np.concatenate(all_src, axis=0)
        dst = np.concatenate(all_dst, axis=0)
    else:
        src = np.empty((0, 2), dtype=np.float64)
        dst = np.empty((0, 2), dtype=np.float64)
    baseline_pred = local_residual.project_homography(base, src)
    baseline_error = np.linalg.norm(baseline_pred - dst, axis=1)
    support_radius = float(cfg["support_radius_tile_factor"]) * tile_size
    model = {
        "interpolator": "normalized_wendland_c2",
        "coordinate_space": "fixed_level0_after_stage2",
        "support_radius_level0_px": support_radius,
        "support_radius_tile_factor": float(cfg["support_radius_tile_factor"]),
        "baseline_weight": float(cfg["baseline_weight"]),
        "inverse_iterations": int(cfg["inverse_iterations"]),
        "control_points_fixed_level0": [
            item["control_point_fixed_level0"] for item in accepted
        ],
        "residual_vectors_fixed_level0": [
            item["residual_vector_fixed_level0"] for item in accepted
        ],
        "tile_ids": [item["tile_id"] for item in accepted],
        "anchor_counts": [item["anchor_count"] for item in accepted],
    }
    full_transform = {
        "base_homography_3x3": base.tolist(),
        "local_residual": model,
    }
    local_pred = local_residual.forward_points(src, full_transform)
    local_error = np.linalg.norm(local_pred - dst, axis=1)
    if len(src):
        before = _error_summary(baseline_error)
        after = _error_summary(local_error)
    else:
        before = after = {"median_px": None, "p90_px": None, "max_px": None}
    gates = {
        "minimum_unique_tiles": len(accepted) >= int(cfg["min_tiles"]),
        "improves_median": bool(len(src)) and after["median_px"] <= (
            before["median_px"] * float(cfg["promotion_max_median_ratio"])
        ),
        "bounded_p90": bool(len(src)) and after["p90_px"] <= (
            before["p90_px"] * float(cfg["promotion_max_p90_ratio"])
        ),
    }
    full_baseline_pred = local_residual.project_homography(base, full_src)
    full_local_pred = local_residual.forward_points(full_src, full_transform)
    return {
        "promoted": bool(cfg.get("enabled", True)) and all(gates.values()),
        "model": model,
        "accepted_tiles": accepted,
        "rejected_tiles": rejected,
        "num_unique_tiles_before_gates": int(len(groups)),
        "num_unique_tiles_used": int(len(accepted)),
        "tile_size_level0_px": tile_size,
        "baseline_anchor_error": before,
        "local_anchor_error": after,
        "quality_gates": gates,
        "settings": cfg,
        "baseline_errors": np.linalg.norm(full_baseline_pred - full_dst, axis=1),
        "errors": np.linalg.norm(full_local_pred - full_dst, axis=1),
    }


def order_records(records):
    """The centroid records in a fixed order, (tile_id, nucleus_id), and a note saying so.

    `cv2.findHomography(RANSAC)` draws its minimal samples by input index, so the same
    pairs in another order land on a marginally different consensus (TODO item 8:
    0.05 px between two batches of the same run, and Stage 8 flipping on two of
    thirteen datasets). Stage 4b's tile order is not part of the answer; this is.
    """
    def key(record):
        return (str(record.get("tile_id", "")), int(record.get("nucleus_id", 0) or 0))
    return sorted(records, key=key), "sorted by (tile_id, nucleus_id)"


def balanced_grid_sample_pairs(src, dst, grid_shape=(6, 6), max_per_cell=4, prefer="spread",
                               score=None, use_dst_for_binning=False, random_seed=43):
    """Indices of a spatially balanced subset: at most `max_per_cell` pairs per cell of
    a `grid_shape` grid over the moving points' bounding box, chosen to be spread out
    within the cell (`prefer="spread"`), at random, or by `score`. The 4.10 sampler."""
    src = np.asarray(src)
    dst = np.asarray(dst)
    pts = dst if use_dst_for_binning else src
    n = len(pts)
    if n == 0:
        return np.array([], dtype=int)
    gx, gy = max(int(grid_shape[0]), 1), max(int(grid_shape[1]), 1)
    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xr, yr = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    xi = np.clip(np.floor((x - xmin) / xr * gx).astype(int), 0, gx - 1)
    yi = np.clip(np.floor((y - ymin) / yr * gy).astype(int), 0, gy - 1)
    cell_to_indices = {}
    for i in range(n):
        cell_to_indices.setdefault((int(xi[i]), int(yi[i])), []).append(i)
    keep = []
    rng = np.random.default_rng(random_seed)
    for _, inds in cell_to_indices.items():
        inds = np.array(inds, dtype=int)
        if len(inds) <= max_per_cell:
            keep.extend(inds.tolist())
            continue
        if prefer == "random":
            picked = rng.choice(inds, size=max_per_cell, replace=False)
        elif prefer == "score":
            if score is None:
                raise ValueError("prefer='score' requires score")
            order = np.argsort(-np.asarray(score)[inds])
            picked = inds[order[:max_per_cell]]
        else:
            cell_pts = pts[inds]
            picked_local = [0]
            remaining = list(range(1, len(cell_pts)))
            while len(picked_local) < max_per_cell and remaining:
                chosen = cell_pts[picked_local]
                cand = cell_pts[remaining]
                d2 = ((cand[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
                best_j = int(np.argmax(d2.min(axis=1)))
                picked_local.append(remaining[best_j])
                remaining.pop(best_j)
            picked = inds[np.array(picked_local, dtype=int)]
        keep.extend(picked.tolist())
    return np.array(sorted(set(keep)), dtype=int)


def fit_transform(src, dst, transform_type, threshold, max_iters, confidence, random_seed):
    """RANSAC fit of `transform_type`; returns (3x3 matrix, inlier mask, method label)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cv2.setRNGSeed(int(random_seed))
    if transform_type == "affine":
        A, inliers = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
            maxIters=int(max_iters), confidence=float(confidence), refineIters=10,
        )
        if A is None:
            raise RuntimeError("affine RANSAC failed (estimateAffine2D returned None)")
        return np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64), inliers, "cv2.estimateAffine2D(RANSAC) -> 3x3"
    if transform_type != "homography":
        raise ValueError(f"invalid transform_mode: {transform_type}")
    H, inliers = cv2.findHomography(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
        maxIters=int(max_iters), confidence=float(confidence),
    )
    if H is None:
        raise RuntimeError("homography RANSAC failed (findHomography returned None)")
    return np.asarray(H, dtype=np.float64), inliers, "cv2.findHomography(RANSAC)"


def _project_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denominator = projected[:, 2]
    if np.any(~np.isfinite(denominator) | (np.abs(denominator) < 1e-12)):
        raise RuntimeError("candidate transform has an invalid projective denominator")
    result = projected[:, :2] / denominator[:, None]
    if not np.isfinite(result).all():
        raise RuntimeError("candidate transform produced non-finite coordinates")
    return result


def _canonical_tile_groups(src, dst, tile_ids):
    """Return tile groups whose order does not depend on the input row order."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    raw_ids = np.asarray(tile_ids, dtype=object)
    if len(raw_ids) != len(src):
        raise ValueError("tile_ids length must match the Stage-5 anchor arrays")
    normalized = np.asarray([
        str(value).strip()
        if value is not None and str(value).strip()
        else f"__row_{index:08d}"
        for index, value in enumerate(raw_ids)
    ], dtype=object)
    groups = []
    for tile_id in sorted(set(normalized.tolist())):
        indices = np.flatnonzero(normalized == tile_id)
        # A tile normally contributes two nuclei.  Sort those rows geometrically so
        # even an externally shuffled centroid JSON produces the same RANSAC stream.
        values = np.column_stack((src[indices], dst[indices]))
        order = np.lexsort(tuple(values[:, column] for column in reversed(range(4))))
        groups.append((tile_id, indices[order]))
    return normalized, groups


def _evaluation_grid(src, size):
    """A fixed grid over the anchor extent, including slightly expanded edges."""
    src = np.asarray(src, dtype=np.float64)
    lower = np.quantile(src, 0.01, axis=0)
    upper = np.quantile(src, 0.99, axis=0)
    span = np.maximum(upper - lower, 1.0)
    lower = lower - 0.05 * span
    upper = upper + 0.05 * span
    return np.asarray([
        (x, y)
        for y in np.linspace(lower[1], upper[1], int(size))
        for x in np.linspace(lower[0], upper[0], int(size))
    ], dtype=np.float64)


def _spatial_support(tile_centers, supported, grid_shape):
    tile_centers = np.asarray(tile_centers, dtype=np.float64)
    supported = np.asarray(supported, dtype=bool)
    gx, gy = int(grid_shape[0]), int(grid_shape[1])
    lower = tile_centers.min(axis=0)
    upper = tile_centers.max(axis=0)
    span = np.maximum(upper - lower, 1e-9)
    bins = np.floor((tile_centers - lower) / span * (gx, gy)).astype(int)
    bins[:, 0] = np.clip(bins[:, 0], 0, gx - 1)
    bins[:, 1] = np.clip(bins[:, 1], 0, gy - 1)
    occupied = len({tuple(value) for value in bins[supported]})

    all_hull_area = 0.0
    supported_hull_area = 0.0
    if len(tile_centers) >= 3:
        all_hull_area = float(cv2.contourArea(cv2.convexHull(
            tile_centers.astype(np.float32)
        )))
    if np.count_nonzero(supported) >= 3:
        supported_hull_area = float(cv2.contourArea(cv2.convexHull(
            tile_centers[supported].astype(np.float32)
        )))
    hull_fraction = (
        supported_hull_area / all_hull_area if all_hull_area > 0 else 0.0
    )
    return int(occupied), float(hull_fraction)


def _fit_group_candidate(src, dst, groups, mode, threshold, max_iters, confidence,
                         random_seed, permutation_index):
    rng_seed = int(random_seed) + 1009 * (int(permutation_index) + 1)
    rng = np.random.default_rng(rng_seed)
    group_order = rng.permutation(len(groups))
    indices = np.concatenate([groups[index][1] for index in group_order])
    matrix, _, method = fit_transform(
        np.asarray(src)[indices], np.asarray(dst)[indices], mode, threshold,
        max_iters, confidence, rng_seed,
    )
    return matrix, method


def _fit_stable_family(src, dst, tile_ids, mode, threshold, max_iters, confidence,
                       random_seed, settings, num_permutations):
    """Fit one model family repeatedly and select its spatially stable medoid."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    normalized_ids, groups = _canonical_tile_groups(src, dst, tile_ids)
    if len(groups) < (3 if mode == "affine" else 4):
        raise RuntimeError(f"too few unique tiles for stable {mode} fitting")

    tile_centers = np.asarray([
        np.median(src[indices], axis=0) for _, indices in groups
    ], dtype=np.float64)
    grid = _evaluation_grid(src, settings["stability_grid_size"])
    candidates = []
    for permutation_index in range(int(num_permutations)):
        try:
            matrix, method = _fit_group_candidate(
                src, dst, groups, mode, threshold, max_iters, confidence,
                random_seed, permutation_index,
            )
            residual = np.linalg.norm(_project_points(matrix, src) - dst, axis=1)
            row_inliers = residual <= float(threshold)
            supported_tiles = np.asarray([
                float(np.median(residual[indices])) <= float(threshold)
                for _, indices in groups
            ], dtype=bool)
            occupied, hull_fraction = _spatial_support(
                tile_centers, supported_tiles, settings["spatial_grid"]
            )
            inlier_errors = residual[row_inliers]
            candidates.append({
                "mode": mode,
                "permutation_index": int(permutation_index),
                "matrix": matrix,
                "method": method,
                "predictions": _project_points(matrix, grid),
                "num_row_inliers": int(np.count_nonzero(row_inliers)),
                "num_supported_tiles": int(np.count_nonzero(supported_tiles)),
                "occupied_spatial_cells": occupied,
                "supported_tile_hull_fraction": hull_fraction,
                "inlier_median_reproj_error_px": (
                    float(np.median(inlier_errors)) if len(inlier_errors) else float("inf")
                ),
                "inlier_p95_reproj_error_px": (
                    float(np.quantile(inlier_errors, 0.95))
                    if len(inlier_errors) else float("inf")
                ),
            })
        except (RuntimeError, cv2.error, np.linalg.LinAlgError):
            continue
    if not candidates:
        raise RuntimeError(f"all stable {mode} RANSAC candidates failed")

    predictions = np.asarray([item["predictions"] for item in candidates])
    pairwise = np.linalg.norm(
        predictions[:, None, :, :] - predictions[None, :, :, :], axis=-1
    )
    pairwise_edge_p95 = np.quantile(pairwise, 0.95, axis=2)
    instability = np.median(pairwise_edge_p95, axis=1)
    max_tiles = max(item["num_supported_tiles"] for item in candidates)
    max_cells = max(item["occupied_spatial_cells"] for item in candidates)
    max_hull = max(item["supported_tile_hull_fraction"] for item in candidates)

    for index, item in enumerate(candidates):
        support_penalty = 1.0 - item["num_supported_tiles"] / max(max_tiles, 1)
        cell_penalty = 1.0 - item["occupied_spatial_cells"] / max(max_cells, 1)
        hull_penalty = 1.0 - (
            item["supported_tile_hull_fraction"] / max(max_hull, 1e-12)
        )
        spatial_penalty = 0.5 * (cell_penalty + hull_penalty)
        stability_penalty = float(instability[index]) / float(threshold)
        item["edge_instability_p95_px"] = float(instability[index])
        item["support_penalty"] = float(support_penalty)
        item["spatial_penalty"] = float(spatial_penalty)
        item["stability_penalty"] = float(stability_penalty)
        item["selection_score"] = float(
            settings["support_weight"] * support_penalty
            + settings["spatial_weight"] * spatial_penalty
            + settings["stability_weight"] * stability_penalty
        )

    selected = min(candidates, key=lambda item: (
        item["selection_score"],
        -item["num_supported_tiles"],
        -item["occupied_spatial_cells"],
        -item["supported_tile_hull_fraction"],
        item["inlier_median_reproj_error_px"],
        item["permutation_index"],
    ))
    matrix = np.asarray(selected["matrix"], dtype=np.float64)
    residual = np.linalg.norm(_project_points(matrix, src) - dst, axis=1)
    inliers = (residual <= float(threshold)).astype(np.uint8)[:, None]

    serializable_candidates = []
    for item in candidates:
        serializable_candidates.append({
            key: value for key, value in item.items()
            if key not in {"matrix", "predictions", "method"}
        })
    diagnostics = {
        "mode": mode,
        "num_unique_tiles": int(len(groups)),
        "num_candidates_requested": int(num_permutations),
        "num_candidates_succeeded": int(len(candidates)),
        "selected_permutation_index": int(selected["permutation_index"]),
        "selected_score": float(selected["selection_score"]),
        "selected_num_row_inliers": int(selected["num_row_inliers"]),
        "selected_num_supported_tiles": int(selected["num_supported_tiles"]),
        "selected_occupied_spatial_cells": int(selected["occupied_spatial_cells"]),
        "selected_supported_tile_hull_fraction": float(
            selected["supported_tile_hull_fraction"]
        ),
        "selected_edge_instability_p95_px": float(
            selected["edge_instability_p95_px"]
        ),
        "candidates": serializable_candidates,
    }
    return matrix, inliers, selected["method"], diagnostics


def _stable_grouped_cross_validation(src, dst, tile_ids, modes, threshold, max_iters,
                                     confidence, random_seed, settings):
    normalized_ids, groups = _canonical_tile_groups(src, dst, tile_ids)
    unique_ids = [tile_id for tile_id, _ in groups]
    n_folds = min(int(settings["cv_folds"]), len(unique_ids))
    fold_by_tile = {
        tile_id: int(hashlib.sha256(tile_id.encode("utf-8")).hexdigest()[:8], 16)
        % n_folds
        for tile_id in unique_ids
    }
    folds = np.asarray([fold_by_tile[tile_id] for tile_id in normalized_ids], dtype=int)
    results = {}
    for mode in modes:
        row_errors = []
        tile_errors = []
        succeeded = 0
        for fold in range(n_folds):
            held_out = folds == fold
            training = ~held_out
            if not np.any(held_out):
                continue
            try:
                matrix, _, _, _ = _fit_stable_family(
                    np.asarray(src)[training], np.asarray(dst)[training],
                    normalized_ids[training], mode, threshold, max_iters, confidence,
                    random_seed + 7919 * (fold + 1), settings,
                    settings["cv_group_permutations"],
                )
            except RuntimeError:
                continue
            errors = np.linalg.norm(
                _project_points(matrix, np.asarray(src)[held_out])
                - np.asarray(dst)[held_out], axis=1
            )
            held_ids = normalized_ids[held_out]
            row_errors.extend(errors.tolist())
            tile_errors.extend([
                float(np.median(errors[held_ids == tile_id]))
                for tile_id in sorted(set(held_ids.tolist()))
            ])
            succeeded += 1
        if not tile_errors:
            results[mode] = {"folds_succeeded": 0, "valid": False}
            continue
        row_values = np.asarray(row_errors, dtype=np.float64)
        tile_values = np.asarray(tile_errors, dtype=np.float64)
        results[mode] = {
            "folds_succeeded": int(succeeded),
            "valid": bool(succeeded == n_folds),
            "row_median_px": float(np.median(row_values)),
            "row_p90_px": float(np.quantile(row_values, 0.90)),
            "row_p95_px": float(np.quantile(row_values, 0.95)),
            "tile_median_px": float(np.median(tile_values)),
            "tile_p90_px": float(np.quantile(tile_values, 0.90)),
            "tile_p95_px": float(np.quantile(tile_values, 0.95)),
            "tile_within_threshold_fraction": float(np.mean(tile_values <= threshold)),
        }
    return results


def fit_stable_transform(src, dst, tile_ids, threshold, max_iters, confidence,
                         random_seed, settings):
    """Select an order-stable affine or homography from tile-group RANSAC starts.

    The 100% run, cell polygons, FOV labels, and any downstream evaluation are never
    read here.  Model-family choice uses deterministic tile-group cross-validation;
    candidate choice uses only anchor consensus, spatial coverage, and agreement of
    candidate predictions at the edges of the observed anchor domain.
    """
    cfg = dict(STABLE_MODEL_DEFAULTS)
    cfg.update(settings or {})
    modes = list(dict.fromkeys(cfg["candidate_modes"]))
    cross_validation = _stable_grouped_cross_validation(
        src, dst, tile_ids, modes, threshold, max_iters, confidence, random_seed, cfg
    )

    full_fits = {}
    for mode in modes:
        try:
            full_fits[mode] = _fit_stable_family(
                src, dst, tile_ids, mode, threshold, max_iters, confidence,
                random_seed, cfg, cfg["group_permutations"],
            )
        except RuntimeError:
            continue
    if not full_fits:
        raise RuntimeError("stable Stage-5 selection could not fit any candidate model")

    selected_mode = modes[0] if modes[0] in full_fits else next(iter(full_fits))
    selection_reason = f"only valid candidate family: {selected_mode}"
    if "affine" in full_fits and "homography" in full_fits:
        affine_cv = cross_validation.get("affine", {})
        homography_cv = cross_validation.get("homography", {})
        margin = float(cfg["homography_min_cv_improvement_fraction"])
        if affine_cv.get("valid") and homography_cv.get("valid"):
            median_better = homography_cv["tile_median_px"] <= (
                affine_cv["tile_median_px"] * (1.0 - margin)
            )
            p90_better = homography_cv["tile_p90_px"] <= (
                affine_cv["tile_p90_px"] * (1.0 - margin)
            )
            if median_better and p90_better:
                selected_mode = "homography"
                selection_reason = (
                    "homography improved both grouped-CV tile median and P90 by at "
                    f"least {margin:.1%}"
                )
            else:
                selected_mode = "affine"
                selection_reason = (
                    "affine simplicity gate retained: homography did not improve both "
                    f"grouped-CV tile median and P90 by {margin:.1%}"
                )
        else:
            selected_mode = "affine"
            selection_reason = "affine retained because grouped homography CV was incomplete"

    matrix, inliers, base_method, family_diagnostics = full_fits[selected_mode]
    family_summaries = {
        mode: fit[3] for mode, fit in full_fits.items()
    }
    diagnostics = {
        "enabled": True,
        "selected_mode": selected_mode,
        "selection_reason": selection_reason,
        "settings": {
            **cfg,
            "spatial_grid": list(cfg["spatial_grid"]),
        },
        "grouped_cross_validation": cross_validation,
        "full_data_fits": family_summaries,
        "selected_family": family_diagnostics,
    }
    method = f"stable tile-group multi-start {base_method}"
    return matrix, inliers, method, selected_mode, diagnostics
