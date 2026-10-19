# ============================================================
# cli_stage5.py - Stage 5 (headless): final alignment from nuclei pairs
#
# The engine's alignment math lives in a nested closure inside the Tk
# gallery (calculate_alignment_transform), so it cannot be imported.
# This is a faithful headless transcription of that compute for the
# configured transform modes (homography / affine), reading the same
# nuclei_patches/nuclei_centroids_global.json and writing the same
# <moving>_to_he_homography_level0.json (+ .csv).
#
# Usage:
#   python cli_stage5.py --engine <dir> --run-dir <dir> --pair {dapi-he,he-he}
# ============================================================
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import run_roles
from cli_common import load_json, save_json, log_stage_event, banner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    run_roles.add_pair_arguments(ap)
    ap.add_argument("--apply-manual-centroids", action="store_true",
                    help="include Stage-5 GUI keypoint corrections in this fit")
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    pair = run_roles.pair_from_args(args, run_dir=run_dir)
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    roles = run_roles.read(run_dir)
    moving, fixed = roles.moving.token, roles.fixed.token
    src_key = f"{moving}_centroid_global"
    dst_key = f"{fixed}_centroid_global"

    banner(f"STAGE 5 (headless alignment, pair={pair.name})")
    log_stage_event(run_dir, "stage5_events", "cli_stage5_start", pair=pair.name)

    # ---- params: read directly from the integrated scripts/parameters.json ----
    block = roles.params_block
    from stage5_fit import (balanced_grid_sample_pairs, build_density_local_transform,
                            fit_stable_transform, fit_transform,
                            infer_tile_size_level0, load_stage5_params, order_records,
                            records_are_density)
    import physical_scale
    params = load_stage5_params(
        load_json(engine_dir.parent / "parameters.json").get(block, {}).get("stage5", {}))
    transform_type = params["transform_mode"]
    balance_points_bool = params["balance_points_bool"]
    random_seed = params["random_seed"]
    stable_model_cfg = params["stable_model_selection"]

    if transform_type not in ("homography", "affine"):
        raise ValueError(
            f"invalid transform_mode: {transform_type}; "
            f"set parameters.json['stage5']['transform_mode'] to 'homography' or 'affine'."
        )

    # ---- load nuclei centroid pairs ----
    info_path = run_dir / "nuclei_patches" / "nuclei_centroids_global.json"
    data = load_json(info_path)
    if isinstance(data, dict) and "data" in data:
        nuclei_info = data["data"]
        nuclei_meta = data.get("meta", {})
    else:
        nuclei_info = data
        nuclei_meta = {}

    if args.apply_manual_centroids:
        from manual_keypoints import load_manual_centroid_map, replace_global_with_manual_if_any
        manual_map = load_manual_centroid_map(info_path.parent)
        nuclei_info = replace_global_with_manual_if_any(nuclei_info, manual_map, nuclei_meta, roles)

    n = len(nuclei_info)
    min_needed = 3 if transform_type == "affine" else 4
    if n < min_needed:
        raise RuntimeError(f"need >= {min_needed} nucleus pairs, got {n}")

    # The fit must not depend on the order Stage 4b happened to list the tiles in.
    nuclei_info, order_note = order_records(nuclei_info)
    src_all = np.array([x[src_key] for x in nuclei_info], dtype=np.float64)
    dst_all = np.array([x[dst_key] for x in nuclei_info], dtype=np.float64)

    # Density anchors in serial sections are local architecture observations. They
    # must not replace Stage 2's whole-slide homography with a model supported by a
    # few spatially clustered tiles. Instead, aggregate one residual per tile and
    # publish a compact-support field that is exactly Stage 2 outside its coverage.
    density_cfg = params["density_local_fit"]
    if records_are_density(nuclei_info) and density_cfg["enabled"]:
        from run_artifacts import initial_alignment_path
        import physical_scale

        initial = load_json(initial_alignment_path(run_dir))
        base_homography = np.asarray(initial["H_mat_level_0"], dtype=np.float64)
        tile_info_path = run_dir / "tiles" / f"{moving}_tile_info.json"
        tile_info = load_json(tile_info_path) if tile_info_path.is_file() else {}
        tile_level = int(nuclei_meta.get(f"{moving}_tile_level", 0))
        tile_size = infer_tile_size_level0(tile_info, tile_level)

        scale = physical_scale.run_factors(
            load_json(run_dir / "images_info.json"),
            {"dapi": float(params.get("reference_mpp_dapi", physical_scale.REFERENCE_MPP["dapi"])),
             "he": float(params.get("reference_mpp_he", physical_scale.REFERENCE_MPP["he"]))},
            run_dir=run_dir)
        fixed_factor = float(scale["roles"]["fixed"])
        threshold = float(params["ransac_reproj_threshold_px"] * fixed_factor)
        fit_records = [
            {"tile_id": row.get("tile_id", ""), "src": row[src_key], "dst": row[dst_key]}
            for row in nuclei_info
        ]

        if tile_size is None:
            promoted = False
            diagnostics = {
                "promoted": False,
                "reason": f"missing or invalid {tile_info_path.name}",
                "quality_gates": {"tile_geometry_available": False},
                "settings": density_cfg,
            }
            baseline_h = np.column_stack([src_all, np.ones(len(src_all))]) @ base_homography.T
            baseline_pred = baseline_h[:, :2] / baseline_h[:, 2:3]
            errors = np.linalg.norm(baseline_pred - dst_all, axis=1)
            local_model = None
        else:
            result = build_density_local_transform(
                fit_records, base_homography, tile_size, density_cfg
            )
            promoted = bool(result["promoted"])
            errors = np.asarray(
                result["errors"] if promoted else result["baseline_errors"], dtype=np.float64
            )
            local_model = result.pop("model")
            result.pop("errors")
            result.pop("baseline_errors")
            diagnostics = result

        within = errors <= threshold
        within_count = int(np.count_nonzero(within))
        within_errors = errors[within]
        med_err = float(np.median(within_errors)) if len(within_errors) else float("nan")
        mean_err = float(np.mean(within_errors)) if len(within_errors) else float("nan")
        transform_type_out = "local_residual" if promoted else "homography"
        method = (
            "Stage2 homography + compact-support density residual"
            if promoted else "Stage2 homography (density local fit declined)"
        )
        out = {
            "stage": 5,
            "from": f"{moving}_level0",
            "to": f"{fixed}_level0",
            "method": method,
            "transform_type": transform_type_out,
            "anchor_mode": "density",
            # The canonical matrix and CSV intentionally remain the safe Stage-2
            # global map. Local-aware consumers use the JSON residual block.
            "homography_3x3": base_homography.tolist(),
            "base_homography_3x3": base_homography.tolist(),
            "sampling": {
                "type": "tile_aggregated_local",
                "order": order_note,
                "balance_points_bool": False,
                "num_points_before_sampling": int(len(src_all)),
                "num_points_after_sampling": int(len(src_all)),
                "selected_indices_from_original": list(range(len(src_all))),
            },
            "ransacReprojThreshold": threshold,
            "ransac_reproj_threshold_px_configured": float(
                params["ransac_reproj_threshold_px"]
            ),
            "physical_scale": {
                key: scale[key]
                for key in ("mpp", "reference_mpp", "factor", "roles", "mpp_by_role",
                            "mpp_source", "mpp_source_by_role", "mpp_estimate_by_role",
                            "mpp_estimate_error")
                if key in scale
            },
            "num_points": int(len(src_all)),
            # Compatibility name: for a local field this is the number of anchor
            # observations within the configured error threshold, not RANSAC votes.
            "num_inliers": within_count,
            "inlier_median_reproj_error_px": med_err,
            "inlier_mean_reproj_error_px": mean_err,
            "density_local_fit": diagnostics,
        }
        if promoted:
            out["local_residual"] = local_model

        out_path = run_dir / f"{roles.homography_stem}.json"
        save_json(out_path, out)
        np.savetxt(
            run_dir / f"{roles.homography_stem}.csv",
            base_homography,
            delimiter=",",
            fmt="%.10f",
        )
        save_json(
            run_dir / "nuclei_centroids_global_used_for_stage5.json",
            {"meta": nuclei_meta, "manual_override_applied": bool(args.apply_manual_centroids),
             "data": nuclei_info},
        )
        print(
            f"[Stage5-CLI] density local fit: {'promoted' if promoted else 'declined'}; "
            f"tiles={diagnostics.get('num_unique_tiles_used', 0)}, "
            f"anchors within {threshold:g}px={within_count}/{len(src_all)} -> {out_path.name}",
            flush=True,
        )
        log_stage_event(
            run_dir, "stage5_events", "cli_stage5_done",
            num_inliers=within_count, num_points=int(len(src_all)),
            transform_type=transform_type_out, anchor_mode="density",
        )
        return

    if balance_points_bool:
        keep_idx = balanced_grid_sample_pairs(
            src_all, dst_all, grid_shape=params["balance_grid"],
            max_per_cell=params["balance_max_per_cell"], prefer="spread",
            random_seed=random_seed,
        )
        if len(keep_idx) < min_needed:
            keep_idx = np.arange(len(src_all), dtype=int)
        sampling_type = "grid_balanced"
        print(f"[Stage5-CLI] grid-balanced sample: {len(keep_idx)} of {len(src_all)} pairs "
              f"({params['balance_grid'][0]}x{params['balance_grid'][1]} grid, at most "
              f"{params['balance_max_per_cell']} per cell)", flush=True)
    else:
        keep_idx = np.arange(len(src_all), dtype=int)
        sampling_type = "all_points"
    src = src_all[keep_idx]
    dst = dst_all[keep_idx]
    tile_ids_all = np.asarray([
        str(record.get("tile_id", "")) for record in nuclei_info
    ], dtype=object)
    tile_ids = tile_ids_all[keep_idx]

    # The reprojection threshold counts fixed-image level-0 pixels at the reference
    # pixel size; scale it to this image the way Stages 7 and 8 scale theirs.
    scale = physical_scale.run_factors(
        load_json(run_dir / "images_info.json"),
        {"dapi": float(params.get("reference_mpp_dapi", physical_scale.REFERENCE_MPP["dapi"])),
         "he": float(params.get("reference_mpp_he", physical_scale.REFERENCE_MPP["he"]))},
        run_dir=run_dir)
    # By role, not by modality: a same-modality pair has two images and one modality.
    fixed_factor = float(scale["roles"]["fixed"])
    ransac_thr = params["ransac_reproj_threshold_px"] * fixed_factor
    maxIters, confidence = params["max_iters"], params["confidence"]
    print(f"[Stage5-CLI] physical scale: {physical_scale.describe(scale)}; RANSAC threshold "
          f"{params['ransac_reproj_threshold_px']:g} -> {ransac_thr:g} px", flush=True)

    model_selection = None
    fitted_transform_type = transform_type
    if stable_model_cfg["enabled"]:
        H, inliers, method_str, fitted_transform_type, model_selection = (
            fit_stable_transform(
                src, dst, tile_ids, ransac_thr, maxIters, confidence,
                random_seed, stable_model_cfg,
            )
        )
        print(
            f"[Stage5-CLI] stable model selection: {fitted_transform_type}; "
            f"{model_selection['selection_reason']}", flush=True,
        )
    else:
        H, inliers, method_str = fit_transform(
            src, dst, transform_type, ransac_thr, maxIters, confidence, random_seed)

    # reprojection error
    src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float64)], axis=1)
    proj = (H.astype(np.float64) @ src_h.T).T
    proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
    err = np.linalg.norm(proj_xy - dst, axis=1)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if inliers is not None:
        mask = inliers.ravel().astype(bool)
        err_in = err[mask]
        med_err = float(np.median(err_in)) if len(err_in) else float("nan")
        mean_err = float(np.mean(err_in)) if len(err_in) else float("nan")
    else:
        med_err = float(np.median(err))
        mean_err = float(np.mean(err))

    out = {
        "from": f"{moving}_level0",
        "to": f"{fixed}_level0",
        "method": method_str,
        "transform_type": fitted_transform_type,
        "sampling": {
            "type": sampling_type,
            "order": order_note,
            "balance_points_bool": bool(balance_points_bool),
            "balance_grid": list(params["balance_grid"]),
            "balance_max_per_cell": int(params["balance_max_per_cell"]),
            "num_points_before_sampling": int(len(src_all)),
            "num_points_after_sampling": int(len(src)),
            "selected_indices_from_original": keep_idx.tolist(),
        },
        "ransacReprojThreshold": float(ransac_thr),
        "ransac_reproj_threshold_px_configured": float(params["ransac_reproj_threshold_px"]),
        "physical_scale": {
            key: scale[key]
            for key in ("mpp", "reference_mpp", "factor", "roles", "mpp_by_role",
                        "mpp_source", "mpp_source_by_role", "mpp_estimate_by_role",
                        "mpp_estimate_error")
            if key in scale
        },
        "maxIters": int(maxIters),
        "confidence": float(confidence),
        "random_seed": int(random_seed),
        "num_points": int(len(src)),
        "num_inliers": int(inlier_count),
        "inlier_median_reproj_error_px": med_err,
        "inlier_mean_reproj_error_px": mean_err,
        "homography_3x3": H.tolist(),
    }
    if model_selection is not None:
        out["stable_model_selection"] = model_selection

    out_path = run_dir / f"{roles.homography_stem}.json"
    save_json(out_path, out)
    np.savetxt(run_dir / f"{roles.homography_stem}.csv",
               np.array(H, dtype=np.float64), delimiter=",", fmt="%.10f")
    save_json(run_dir / "nuclei_centroids_global_used_for_stage5.json",
              {"meta": nuclei_meta, "manual_override_applied": bool(args.apply_manual_centroids), "data": nuclei_info})

    print(f"[Stage5-CLI] {method_str}: {inlier_count}/{len(src)} inliers, "
          f"median reproj err {med_err:.3f}px -> {out_path.name}", flush=True)
    log_stage_event(run_dir, "stage5_events", "cli_stage5_done",
                    num_inliers=int(inlier_count), num_points=int(len(src)),
                    transform_type=fitted_transform_type,
                    stable_model_selection=bool(model_selection is not None))


if __name__ == "__main__":
    main()
