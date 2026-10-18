#!/usr/bin/env python3
"""Stage 8: refine the DAPI-to-H&E homography from dense high-confidence pairs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
_PROCESS_STARTED = time.perf_counter()
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from cli_common import finalize_pipeline_timing, log_stage_event, save_json
import physical_scale
DEFAULTS = {
    "ransac_reproj_threshold_px": 4.0,
    "max_iters": 20000,
    "confidence": 0.999,
    "holdout_fraction": 0.20,
    "random_seed": 43,
    "min_pairs": 100,
    "min_inliers": 100,
    "min_inlier_ratio": 0.30,
    "coverage_grid_size": 8,
    "min_occupied_grid_cells": 12,
    "min_axis_span_level0_px": 5000.0,
    "max_holdout_median_error_ratio": 1.05,
    "max_holdout_p95_error_ratio": 1.05,
    "max_holdout_median_degradation_px": 1.0,
    "max_transform_shift_p95_px": 10.0,
    "stability_refits": 10,
    "stability_drop_fraction": 0.10,
    "stability_seed": 1000,
    "stability_max_shift_p95_um": 1.0,
    "promote_to_canonical": True,
    "refresh_stage6_overlay": True,
}


VALID_TRANSFORM_MODES = ("homography", "affine")
#: transform_mode value that fits the model Stage 5 selected for this run
FOLLOW_STAGE5 = "stage5"


def load_params(path: Path, pair: str = "dapi-he"):
    data = json.loads(path.read_text())
    # Match Stage 7's pair-key precedence, including Stage-5 model inheritance.
    # Legacy parameter files still work, but cannot override a present pair block.
    # Another mixed pair reads its own block and inherits dapi-he's when it has none.
    dapi = data.get("dapi-he", data.get("dapi", {}))
    if pair != "dapi-he" and isinstance(data.get(pair, {}).get("stage8_refinement"), dict):
        dapi = data[pair]
    configured = dapi.get("stage8_refinement", {})
    params = dict(DEFAULTS)
    params.setdefault("transform_mode",
                      dapi.get("stage5", {}).get("transform_mode", "homography"))
    params.update(configured)
    mode = str(params["transform_mode"]).lower().strip()
    if mode not in VALID_TRANSFORM_MODES + (FOLLOW_STAGE5,):
        raise ValueError(
            f"stage8_refinement.transform_mode={params['transform_mode']!r} is not "
            f"supported; expected one of {', '.join(VALID_TRANSFORM_MODES + (FOLLOW_STAGE5,))}. "
            f"Stage 8 replaces Stage 5's transform, so it fits the same models Stage 5 does -- "
            f"tps and local_tps are Stage-5-only and cannot be refined from pairs."
        )
    params["transform_mode"] = mode
    return params


def resolve_transform_mode(params, initial_transform):
    """With transform_mode "stage5", fit the model Stage 5 selected for this run.

    Stage 5 chooses between an affine model and a homography by tile-grouped
    cross-validation. Refining an affine result with a homography adds perspective terms
    that the dense pairs constrain poorly: the refit then changes with which pairs Stage 7
    happened to keep. A Stage-5 local residual field refines its base homography.
    """
    params = dict(params)
    if params["transform_mode"] == FOLLOW_STAGE5:
        selected = str(initial_transform.get("transform_type", "homography")).lower()
        params["transform_mode"] = "affine" if selected == "affine" else "homography"
    return params


def project_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), np.asarray(matrix, dtype=np.float64)
    ).reshape(-1, 2)


def reprojection_errors(points_src, points_dst, matrix):
    return np.linalg.norm(project_points(points_src, matrix) - points_dst, axis=1)


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"median": None, "mean": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def deterministic_holdout_mask(pairs: pd.DataFrame, fraction: float):
    fraction = float(np.clip(fraction, 0.05, 0.45))
    dapi_id = pairs["dapi_nucleus_id"].to_numpy(np.int64)
    he_id = pairs["he_nucleus_id"].to_numpy(np.int64)
    stable_hash = (dapi_id * 73856093) ^ (he_id * 19349663)
    mask = (np.abs(stable_hash) % 10000) < int(round(fraction * 10000))
    if int(mask.sum()) < 4 or int((~mask).sum()) < 4:
        mask = np.zeros(len(pairs), dtype=bool)
        mask[::max(2, int(round(1.0 / fraction)))] = True
    return mask


def spatial_coverage(points, grid_size):
    points = np.asarray(points, dtype=np.float64)
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    span = hi - lo
    safe_span = np.maximum(span, 1.0)
    bins = np.clip(((points - lo) / safe_span * int(grid_size)).astype(int), 0, int(grid_size) - 1)
    occupied = len({(int(x), int(y)) for x, y in bins})
    return {
        "grid_size": int(grid_size),
        "occupied_grid_cells": int(occupied),
        "x_span_level0_px": float(span[0]),
        "y_span_level0_px": float(span[1]),
    }


def _mode(params):
    """The model to fit. `load_params` has already validated and normalised it."""
    return str(params.get("transform_mode", "homography")).lower().strip()


def _fit_homography(src, dst, params):
    """Fit the model `transform_mode` names, returned as a 3x3 either way.

    Stage 8 replaces Stage 5's transform, so fitting a different model than Stage 5 did
    would silently change the model on whichever datasets happen to be promoted -- half
    of them, in practice. `resolve_transform_mode` sets the mode to the model Stage 5
    selected for the run.
    """
    cv2.setRNGSeed(int(params["random_seed"]))
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if _mode(params) == "affine":
        affine, inliers = cv2.estimateAffine2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(params["ransac_reproj_threshold_px"]),
            maxIters=int(params["max_iters"]),
            confidence=float(params["confidence"]),
        )
        if affine is None or not np.isfinite(affine).all():
            return None, None
        matrix = np.vstack([affine.astype(np.float64), [0.0, 0.0, 1.0]])
        return matrix, inliers.ravel().astype(bool) if inliers is not None else None

    matrix, inliers = cv2.findHomography(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(params["ransac_reproj_threshold_px"]),
        maxIters=int(params["max_iters"]),
        confidence=float(params["confidence"]),
    )
    if matrix is None or not np.isfinite(matrix).all():
        return None, None
    matrix = matrix.astype(np.float64)
    if abs(float(matrix[2, 2])) > 1e-12:
        matrix /= float(matrix[2, 2])
    return matrix, inliers.ravel().astype(bool) if inliers is not None else None


def subsampling_stability(src, dst, final_matrix, params, fixed_mpp):
    """Refit on random subsets of the pairs and measure how far each refit moves them.

    Which nuclei Stage 7 pairs depends on the Stage-5 transform, so a refined transform that
    changes when a few pairs are left out reflects RANSAC's choice of inliers rather than
    the tissue. Each of `stability_refits` refits leaves out `stability_drop_fraction` of
    the pairs; the 95th-percentile shift of the pairs between that refit and the all-pair
    fit must stay within `stability_max_shift_p95_um` in the fixed image for every refit.
    """
    bound_px = float(params["stability_max_shift_p95_um"]) / float(fixed_mpp)
    drop = float(params["stability_drop_fraction"])
    shifts = []
    for k in range(int(params["stability_refits"])):
        keep = np.random.default_rng(int(params["stability_seed"]) + k).random(len(src)) >= drop
        matrix, _ = _fit_homography(src[keep], dst[keep], params)
        if matrix is None:
            return {"refits": k + 1, "failed_refit": k, "bound_um": float(params["stability_max_shift_p95_um"]),
                    "bound_px": bound_px, "passed": False}
        shift = np.linalg.norm(project_points(src, matrix) - project_points(src, final_matrix), axis=1)
        shifts.append(float(np.percentile(shift, 95)))
    worst = max(shifts) if shifts else 0.0
    return {"refits": len(shifts), "drop_fraction": drop, "fixed_mpp": float(fixed_mpp),
            "max_shift_p95_px": worst, "max_shift_p95_um": worst * float(fixed_mpp),
            "bound_um": float(params["stability_max_shift_p95_um"]), "bound_px": bound_px,
            "passed": bool(worst <= bound_px)}


def refine_from_pairs(pairs, initial_matrix, params, stage5_anchors=None, tokens=("dapi", "he"),
                      fixed_mpp=None):
    """Refine the moving->fixed transform from Stage 7's pairs; columns are named by
    the run's tokens (`dapi`/`he` for dapi-he, `he`/`dapi` for he-dapi). `fixed_mpp`
    (fixed-image um per level-0 pixel) is needed for the subsampling-stability gate,
    which is skipped without it."""
    m, f = tokens
    required = {
        f"{m}_nucleus_id", f"{f}_nucleus_id", f"{m}_x_level0", f"{m}_y_level0",
        f"{f}_x_level0", f"{f}_y_level0",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"high-confidence pair table is missing columns: {sorted(missing)}")

    pairs = pairs.copy()
    finite = np.isfinite(
        pairs[[f"{m}_x_level0", f"{m}_y_level0", f"{f}_x_level0", f"{f}_y_level0"]]
        .to_numpy(np.float64)
    ).all(axis=1)
    pairs = pairs.loc[finite].drop_duplicates(
        subset=[f"{m}_nucleus_id", f"{f}_nucleus_id"]
    ).reset_index(drop=True)
    src = pairs[[f"{m}_x_level0", f"{m}_y_level0"]].to_numpy(np.float64)
    dst = pairs[[f"{f}_x_level0", f"{f}_y_level0"]].to_numpy(np.float64)
    coverage = spatial_coverage(src, int(params["coverage_grid_size"])) if len(src) else {}

    gates = {
        "minimum_pairs": len(src) >= int(params["min_pairs"]),
        "minimum_spatial_grid_coverage": bool(coverage) and (
            coverage["occupied_grid_cells"] >= int(params["min_occupied_grid_cells"])
        ),
        "minimum_axis_span": bool(coverage) and (
            min(coverage["x_span_level0_px"], coverage["y_span_level0_px"])
            >= float(params["min_axis_span_level0_px"])
        ),
    }
    if len(src) < 8:
        return {
            "accepted": False, "matrix": None, "pairs": pairs, "gates": gates,
            "coverage": coverage, "reason": "fewer than 8 usable pairs",
        }

    holdout = deterministic_holdout_mask(pairs, float(params["holdout_fraction"]))
    train_matrix, train_inliers = _fit_homography(src[~holdout], dst[~holdout], params)
    final_matrix, final_inliers = _fit_homography(src, dst, params)
    if train_matrix is None or final_matrix is None:
        gates["homography_fit_succeeded"] = False
        return {
            "accepted": False, "matrix": None, "pairs": pairs, "gates": gates,
            "coverage": coverage, "reason": "RANSAC homography fit failed",
        }
    gates["homography_fit_succeeded"] = True

    initial_holdout = reprojection_errors(src[holdout], dst[holdout], initial_matrix)
    refined_holdout = reprojection_errors(src[holdout], dst[holdout], train_matrix)
    initial_all = reprojection_errors(src, dst, initial_matrix)
    refined_all = reprojection_errors(src, dst, final_matrix)
    refined_inlier_errors = (
        refined_all[final_inliers] if final_inliers is not None else np.asarray([], dtype=np.float64)
    )
    transform_shift = np.linalg.norm(
        project_points(src, final_matrix) - project_points(src, initial_matrix), axis=1
    )
    inlier_count = int(final_inliers.sum()) if final_inliers is not None else 0
    inlier_ratio = float(inlier_count / max(len(src), 1))

    initial_holdout_metrics = metric_summary(initial_holdout)
    refined_holdout_metrics = metric_summary(refined_holdout)
    gates.update({
        "minimum_inliers": inlier_count >= int(params["min_inliers"]),
        "minimum_inlier_ratio": inlier_ratio >= float(params["min_inlier_ratio"]),
        # `all_pair_median_improved` was removed in 4.6. It required a strict improvement
        # on every pair, three quarters of which the candidate was fitted to, so it was an
        # in-sample test with no tolerance: a candidate that improved the held-out median
        # was still rejected when the all-pair median moved by a thousandth of a pixel the
        # wrong way. It also got harder to pass the better Stage 5 became, which is
        # backwards. The two held-out ratio gates below already forbid getting worse, and
        # they are measured on data the candidate never saw.
        "holdout_median_degradation_bounded": (
            refined_holdout_metrics["median"]
            <= initial_holdout_metrics["median"] * float(params["max_holdout_median_error_ratio"])
        ),
        "holdout_p95_degradation_bounded": (
            refined_holdout_metrics["p95"]
            <= initial_holdout_metrics["p95"] * float(params["max_holdout_p95_error_ratio"])
        ),
        "bounded_transform_shift": (
            float(np.percentile(transform_shift, 95))
            <= float(params["max_transform_shift_p95_px"])
        ),
        # An absolute companion to the ratio bound above. A ratio alone is permissive
        # when the initial error is already large and punishing when it is small, so a
        # candidate must also not push the held-out median more than a fixed number of
        # pixels the wrong way.
        "holdout_median_degradation_px_bounded": (
            refined_holdout_metrics["median"] - initial_holdout_metrics["median"]
            <= float(params["max_holdout_median_degradation_px"])
        ),
    })
    stability = None
    if int(params.get("stability_refits", 0)) > 0 and fixed_mpp is not None:
        stability = subsampling_stability(src, dst, final_matrix, params, fixed_mpp)
        gates["stable_under_pair_subsampling"] = stability["passed"]

    # Stage-5 anchor residuals are recorded for diagnosis but do not gate promotion.
    #
    # 4.4 gated on them, and that comparison is biased by construction: the anchors are
    # the points Stage 5 fitted its own homography to, so the initial transform is
    # in-sample there while the candidate, fitted to the Stage-7 dense pairs, is not. On
    # human_crc_p1 the candidate improved on all 2,054 dense pairs, on the 432 held-out
    # pairs and on external 10x keypoints (1.50 -> 1.08 um), yet was rejected for a
    # 1.18 px degradation on Stage 5's own fitting points. Promotion is now decided on
    # data neither transform was fitted to.
    #
    # The held-out pairs are not perfectly independent of Stage 5 either -- Stage 7 found
    # them as reciprocal nearest neighbours under the Stage-5 transform, so their
    # selection is conditioned on it. That residual bias also favours the initial
    # transform, which keeps the gate conservative in the safe direction.
    anchor_metrics = None
    if stage5_anchors is not None and len(stage5_anchors[0]):
        anchor_src, anchor_dst = stage5_anchors
        initial_anchor = reprojection_errors(anchor_src, anchor_dst, initial_matrix)
        refined_anchor = reprojection_errors(anchor_src, anchor_dst, final_matrix)
        anchor_metrics = {
            "initial": metric_summary(initial_anchor),
            "refined": metric_summary(refined_anchor),
            "median_degradation_px": float(np.median(refined_anchor) - np.median(initial_anchor)),
            "gating": False,
            "note": "in-sample for the Stage-5 transform; diagnostic only since 4.5",
        }

    pairs["stage8_holdout"] = holdout
    pairs["stage8_ransac_inlier"] = final_inliers if final_inliers is not None else False
    pairs["stage8_initial_residual_px"] = initial_all
    pairs["stage8_refined_residual_px"] = refined_all
    accepted = bool(all(gates.values()))
    return {
        "accepted": accepted,
        "matrix": final_matrix,
        "train_matrix": train_matrix,
        "pairs": pairs,
        "gates": gates,
        "coverage": coverage,
        "num_pairs": int(len(src)),
        "num_train_pairs": int((~holdout).sum()),
        "num_holdout_pairs": int(holdout.sum()),
        "num_inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "initial_all_errors": metric_summary(initial_all),
        "refined_all_errors": metric_summary(refined_all),
        "refined_inlier_errors": metric_summary(refined_inlier_errors),
        "initial_holdout_errors": initial_holdout_metrics,
        "refined_holdout_errors": refined_holdout_metrics,
        "transform_shift": metric_summary(transform_shift),
        "subsampling_stability": stability,
        "stage5_anchor_validation": anchor_metrics,
        "reason": "all quality gates passed" if accepted else "one or more quality gates failed",
    }


def load_stage5_anchors(run_dir, tokens=("dapi", "he")):
    m, f = tokens
    path = run_dir / "nuclei_patches" / "nuclei_centroids_global.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = data.get("data", data) if isinstance(data, dict) else data
    if not rows:
        return None
    src = np.asarray([row[f"{m}_centroid_global"] for row in rows], dtype=np.float64)
    dst = np.asarray([row[f"{f}_centroid_global"] for row in rows], dtype=np.float64)
    return src, dst


def _stage7_physical_scale(run_dir: Path):
    """The per-modality factors Stage 7 recorded for this run, or None."""
    try:
        summary = json.loads((run_dir / "dense_nucleus_pairs" / "summary.json").read_text())
        scale = summary["physical_scale"]
        if not isinstance(scale.get("factor"), dict):
            return None
        return {key: scale[key] for key in (
            "mpp", "reference_mpp", "factor", "roles", "mpp_by_role",
            "mpp_source", "mpp_source_by_role", "mpp_estimate_by_role",
            "mpp_estimate_error", "tolerance",
        )
                if key in scale}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Refine DAPI--H&E alignment from Stage-7 high-confidence pairs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parameters", default=str(PROJECT_ROOT / "parameters.json"))
    parser.add_argument("--no-refresh-overlay", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    import run_roles
    roles = run_roles.read(run_dir)
    if roles.moving.modality == roles.fixed.modality:
        raise SystemExit(f"Stage 8 refines Stage 7's cross-modality nucleus pairs and does "
                         f"not apply to a same-modality pair; this run is {roles.pair.name}")
    m, f = roles.moving.token, roles.fixed.token
    stem = roles.pair.homography_stem
    pair_path = run_dir / "dense_nucleus_pairs" / "high_confidence_pairs.csv.gz"
    canonical_path = run_dir / f"{stem}.json"
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)

    configured = load_params(Path(args.parameters), roles.pair.name)
    # Stage 8's pixel gates follow the same physical rescaling as Stage 7's, and use
    # the factors Stage 7 recorded so the two stages never disagree about the image;
    # a Stage-7 summary without them (an older run) is measured again from the images.
    scale = _stage7_physical_scale(run_dir)
    if scale is None:
        scale = physical_scale.run_factors(
            json.loads((run_dir / "images_info.json").read_text()), run_dir=run_dir,
        )
    gate_spec = physical_scale.stage8_gates(roles.moving.modality, roles.fixed.modality)
    params, scaled_gates = physical_scale.scale_gates(configured, scale["factor"], gate_spec)
    scale = dict(scale, gates=scaled_gates)
    print(f"[Stage8] physical scale: {physical_scale.describe(scale)}"
          + ("" if scaled_gates else "; gates unchanged"), flush=True)
    for key, change in scaled_gates.items():
        print(f"[Stage8]   {key}: {change['before']:g} -> {change['after']:g} px", flush=True)
    started = _PROCESS_STARTED
    log_stage_event(run_dir, "stage8_events", "cli_stage8_start", num_pairs_file=str(pair_path))
    print("\n========== STAGE 8: HIGH-CONFIDENCE ALIGNMENT REFINEMENT ==========", flush=True)
    print(f"[Stage8] run_dir={run_dir}  pair={roles.pair.name}  moving={m}  fixed={f}", flush=True)
    print(f"[Stage8] parameters={json.dumps(configured, sort_keys=True)}", flush=True)

    previous_canonical_transform = json.loads(canonical_path.read_text())
    backup_path = run_dir / f"{stem}_stage5.json"
    # An explicit Stage-8 rerun must use the preserved Stage-5 result as its
    # baseline. Otherwise it would refine an already-refined transform and the
    # quality decision would no longer be reproducible.
    if int(previous_canonical_transform.get("stage", 0)) == 8 and backup_path.exists():
        initial_transform_path = backup_path
        initial_transform = json.loads(backup_path.read_text())
    else:
        initial_transform_path = canonical_path
        initial_transform = previous_canonical_transform
    matrix_key = "homography_3x3" if "homography_3x3" in initial_transform else "initial_homography_3x3"
    if matrix_key not in initial_transform:
        raise ValueError("canonical transform has no homography_3x3")
    initial_matrix = np.asarray(initial_transform[matrix_key], dtype=np.float64)
    params = resolve_transform_mode(params, initial_transform)
    print(f"[Stage8] model: {_mode(params)} (transform_mode={configured['transform_mode']}, "
          f"Stage 5 selected {initial_transform.get('transform_type', 'n/a')})", flush=True)
    # um per fixed level-0 pixel for the stability gate: measured, else estimated, else the
    # modality's reference pixel size (the same fallback the pixel gates use)
    fixed_mpp = (scale.get("mpp_by_role") or {}).get("fixed") \
        or (scale.get("reference_mpp") or {}).get(roles.fixed.modality)
    pairs = pd.read_csv(pair_path)
    anchors = load_stage5_anchors(run_dir, (m, f))
    result = refine_from_pairs(pairs, initial_matrix, params, stage5_anchors=anchors, tokens=(m, f),
                               fixed_mpp=fixed_mpp)

    scored_path = run_dir / "dense_nucleus_pairs" / "high_confidence_pairs_stage8_scored.csv.gz"
    result["pairs"].to_csv(scored_path, index=False, compression="gzip")
    candidate_path = run_dir / f"{stem}_stage8.json"

    candidate_transform = None
    promoted = bool(result["accepted"] and params["promote_to_canonical"] and result["matrix"] is not None)
    if result["matrix"] is not None:
        candidate_transform = {
            "from": f"{m}_level0",
            "to": f"{f}_level0",
            "method": (f"Stage8 cv2.{'estimateAffine2D' if _mode(params) == 'affine' else 'findHomography'}"
                       f"(RANSAC) from Stage7 high-confidence pairs"),
            "transform_type": _mode(params),
            "stage": 8,
            "sampling": {
                "type": "stage7_high_confidence_pairs",
                "source": str(pair_path),
                "num_points": int(result["num_pairs"]),
                "num_train_points": int(result["num_train_pairs"]),
                "num_holdout_points": int(result["num_holdout_pairs"]),
            },
            "ransacReprojThreshold": float(params["ransac_reproj_threshold_px"]),
            "maxIters": int(params["max_iters"]),
            "confidence": float(params["confidence"]),
            "num_points": int(result["num_pairs"]),
            "num_inliers": int(result["num_inliers"]),
            "inlier_median_reproj_error_px": result["refined_inlier_errors"]["median"],
            "inlier_mean_reproj_error_px": result["refined_inlier_errors"]["mean"],
            "quality_gates": result["gates"],
            "promoted_to_canonical": promoted,
            "homography_3x3": result["matrix"].tolist(),
        }
        save_json(candidate_path, candidate_transform)

    if promoted:
        if int(previous_canonical_transform.get("stage", 0)) != 8:
            shutil.copy2(canonical_path, backup_path)
        save_json(canonical_path, candidate_transform)
        np.savetxt(
            run_dir / f"{stem}.csv",
            result["matrix"], delimiter=",", fmt="%.10f",
        )

    overlay_refreshed = False
    overlay_refresh_error = None
    display_level = int(roles.moving.level)
    alternating_gif_path = run_dir / f"6_alternating_L{display_level}.gif"
    refresh_requested = bool(params["refresh_stage6_overlay"]) and not args.no_refresh_overlay
    if promoted and refresh_requested:
        command = [
            sys.executable, str(PROJECT_ROOT / "cli_stage6.py"),
            "--engine", str(ENGINE_DIR),
            "--run-dir", str(run_dir), "--pair", roles.pair.name,
        ]
        proc = subprocess.run(command)
        overlay_refreshed = proc.returncode == 0
        if not overlay_refreshed:
            overlay_refresh_error = f"cli_stage6 exited with {proc.returncode}"

    elapsed = time.perf_counter() - started
    public_result = {key: value for key, value in result.items() if key not in {"pairs", "matrix", "train_matrix"}}
    summary = {
        "stage": 8,
        "pair": roles.pair.name,
        "moving_token": m,
        "fixed_token": f,
        "accepted": bool(result["accepted"]),
        "promoted_to_canonical": promoted,
        "reason": result["reason"],
        "initial_transform_path": str(initial_transform_path),
        "previous_canonical_stage": previous_canonical_transform.get("stage", 5),
        "candidate_transform_path": str(candidate_path) if candidate_transform is not None else None,
        "canonical_transform_path": str(canonical_path),
        "canonical_stage_after": (
            8 if promoted else previous_canonical_transform.get("stage", 5)
        ),
        "stage5_backup_path": str(backup_path) if backup_path.exists() else None,
        "scored_pairs_path": str(scored_path),
        "overlay_refresh_requested": refresh_requested,
        "overlay_refreshed": overlay_refreshed,
        "overlay_refresh_error": overlay_refresh_error,
        "alternating_gif_refreshed": bool(overlay_refreshed and alternating_gif_path.exists()),
        "alternating_gif_path": str(alternating_gif_path) if alternating_gif_path.exists() else None,
        "initial_homography_3x3": initial_matrix.tolist(),
        "candidate_homography_3x3": result["matrix"].tolist() if result["matrix"] is not None else None,
        "metrics": public_result,
        "elapsed_seconds": elapsed,
        "parameters": configured,
        "physical_scale": scale,
        "effective_gates": {key: params[key] for key in gate_spec if key in params},
    }
    summary_path = run_dir / "stage8_alignment_refinement.json"
    save_json(summary_path, summary)
    log_stage_event(
        run_dir, "stage8_events", "cli_stage8_done",
        elapsed_seconds=elapsed, accepted=bool(result["accepted"]), promoted=promoted,
    )
    timing = finalize_pipeline_timing(
        run_dir,
        stage8_elapsed_seconds=elapsed,
        source="stage8",
        completed_through_stage=8,
    )
    print(
        f"[Stage8] accepted={result['accepted']} promoted={promoted} "
        f"pairs={len(result['pairs']):,} elapsed={elapsed:.2f}s",
        flush=True,
    )
    if result.get("initial_holdout_errors"):
        print(
            f"[Stage8] holdout median: initial={result['initial_holdout_errors']['median']:.3f}px "
            f"refined={result['refined_holdout_errors']['median']:.3f}px; "
            f"inliers={result['num_inliers']}/{result['num_pairs']}",
            flush=True,
        )
    print(f"[Stage8] total pipeline time={timing.get('total_elapsed_seconds')}s", flush=True)
    print(f"[Stage8] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
