"""Physical field of view is independent of centroid quantization and QC output."""
import math

from alignment_geometry import (ASSUMED_COARSE_MPP_20X,
                                estimate_mpp_pair_from_homography)
from role_parameters import role_parameter

from my_utils import inspect_image_pyramid, scalar_mpp, choose_level_by_target_mpp


GEOMETRY_DEFAULTS = {
    "patch_size_um": None,             # Full side length, not a radius.
    "patch_target_mpp": None,          # None retains tile level minus one.
    "moving_patch_level_override": None,
    "fixed_patch_level_override": None,
    "fixed_patch_margin": 1.1,
}


def _positive(value, key):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"stage4c.{key} must be finite and > 0")
    return number


def resolve_geometry(roles, tile_levels, params, fixed_scale_level0,
                     level0_homography=None):
    """Preserve legacy moving size/levels unless physical settings are supplied.

    fixed_scale_level0 maps a moving level-0 pixel to fixed level-0 pixels.
    Therefore its patch-grid ratio is s0 * 2**(moving_level - fixed_level).
    """
    size_um = params.get("patch_size_um")
    target = params.get("patch_target_mpp")
    if size_um is not None:
        size_um = _positive(size_um, "patch_size_um")
    if target is not None:
        target = _positive(target, "patch_target_mpp")
    margin = _positive(params.get("fixed_patch_margin", 1.1), "fixed_patch_margin")
    context = {}
    mpps = {}
    pyramids = {}
    base_mpps = {}
    mpp_sources = {}
    needs_physical_metadata = target is not None or size_um is not None
    for role in ("moving", "fixed"):
        override = params.get(f"{role}_patch_level_override")
        pyramid = None
        mpp0 = None
        if needs_physical_metadata or override is not None:
            pyramid = inspect_image_pyramid(roles.role(role).path)
            mpp0 = scalar_mpp(pyramid.get("base_mpp"))
        pyramids[role] = pyramid
        base_mpps[role] = mpp0
        mpp_sources[role] = "ome_header" if mpp0 is not None else None

    mpp_estimates = {}
    mpp_estimate_error = None
    missing_roles = [role for role, value in base_mpps.items() if value is None]
    if needs_physical_metadata and missing_roles and level0_homography is not None:
        try:
            estimate = estimate_mpp_pair_from_homography(
                level0_homography,
                pyramids["moving"]["level_shapes"][0],
                pyramids["fixed"]["level_shapes"][0],
                moving_base_mpp=base_mpps["moving"],
                fixed_base_mpp=base_mpps["fixed"],
            )
        except ValueError as exc:
            mpp_estimate_error = str(exc)
            if len(missing_roles) == 2:
                # Absolute scale is unknowable without a header. If even the relative
                # Stage-2 scale is unsafe, keep Stage 4 usable with an explicit equal-
                # 20x last resort instead of pretending the failed homography was sound.
                for role in missing_roles:
                    base_mpps[role] = ASSUMED_COARSE_MPP_20X
                    mpp_sources[role] = "assumed_20x_without_trustworthy_stage2_scale"
                print(
                    "[WARN] neither image has OME MPP and the Stage-2 relative scale "
                    f"was rejected ({exc}); assuming both level-0 images are 20x "
                    f"({ASSUMED_COARSE_MPP_20X:g} um/px)",
                    flush=True,
                )
            elif missing_roles == ["fixed"]:
                # The moving physical patch remains well-defined; the existing
                # homography grid ratio can still size the fixed crop.
                print(
                    "[WARN] fixed image has no OME MPP and its Stage-2 estimate was "
                    f"rejected ({exc}); using the homography grid ratio for its crop",
                    flush=True,
                )
            else:
                # A physical moving patch cannot be constructed without this value.
                raise
        else:
            for role in missing_roles:
                base_mpps[role] = float(estimate[f"{role}_base_mpp"])
                mpp_estimates[role] = estimate
            if len(missing_roles) == 2:
                anchor = estimate["anchor_role"]
                derived = estimate["derived_role"]
                mpp_sources[anchor] = "assumed_20x"
                mpp_sources[derived] = "stage2_homography_from_assumed_20x"
            elif missing_roles == ["moving"]:
                mpp_sources["moving"] = "stage2_homography_from_fixed_ome"
            else:
                mpp_sources["fixed"] = "stage2_homography_from_moving_ome"
            print(
                "[INFO] resolved missing level-0 MPP from Stage-2 H_mat_level_0: "
                f"moving={base_mpps['moving']:.6g} um/px "
                f"({mpp_sources['moving']}), fixed={base_mpps['fixed']:.6g} um/px "
                f"({mpp_sources['fixed']}); {estimate['sample_count']} overlap samples, "
                f"scale p05/p50/p95={estimate['scale_p05']:.4f}/"
                f"{estimate['fixed_pixels_per_moving_pixel']:.4f}/"
                f"{estimate['scale_p95']:.4f}",
                flush=True,
            )
    elif needs_physical_metadata and len(missing_roles) == 2:
        # A normal Stage-4 run always has H_mat_level_0. This branch keeps direct
        # callers deterministic and makes the absolute-scale assumption explicit.
        for role in missing_roles:
            base_mpps[role] = ASSUMED_COARSE_MPP_20X
            mpp_sources[role] = "assumed_20x_without_stage2_scale"
        print(
            "[WARN] neither image has OME MPP or a Stage-2 homography; assuming both "
            f"level-0 images are 20x ({ASSUMED_COARSE_MPP_20X:g} um/px)",
            flush=True,
        )

    for role, legacy in (("moving", "dapi"), ("fixed", "he")):
        tile_level = int(tile_levels[role])
        level = max(0, tile_level - 1)
        override = params.get(f"{role}_patch_level_override")
        pyramid = pyramids[role]
        mpp0 = base_mpps[role]
        if override is not None:
            level = int(override)
            if level < 0 or float(override) != level:
                raise ValueError(f"stage4c.{role}_patch_level_override must be a nonnegative integer")
            n_levels = len(pyramid.get("level_shapes") or [])
            if n_levels and level >= n_levels:
                raise ValueError(f"{role} patch level {level} outside the {n_levels}-level pyramid")
        elif target is not None:
            if mpp0 is None:
                print(f"[WARN] {role} has no usable MPP; retaining legacy patch level {level}", flush=True)
            else:
                level = choose_level_by_target_mpp(mpp0, len(pyramid["level_shapes"]), target)
        context[f"{legacy}_tile_level"] = tile_level
        context[f"{legacy}_patch_level"] = level
        mpps[role] = mpp0 * 2 ** level if mpp0 is not None else None

    moving_len = int(role_parameter(params, "moving_patch_len", "dapi_patch_len", 200))
    if size_um is not None:
        if mpps["moving"] is None:
            raise ValueError("patch_size_um requires usable moving-image MPP; no physical spacing will be invented")
        moving_len = int(math.ceil(size_um / mpps["moving"]))
    if moving_len < 2:
        raise ValueError("moving patch must be at least 2 pixels wide")
    if size_um is not None and mpps["fixed"] is not None:
        fixed_len = int(math.ceil(size_um * margin / mpps["fixed"]))
    else:
        grid_ratio = float(fixed_scale_level0) * 2 ** (
            context["dapi_patch_level"] - context["he_patch_level"]
        )
        fixed_len = int(math.ceil(moving_len * grid_ratio * margin))
    if fixed_len % 2:
        fixed_len += 1
    if fixed_len < 2:
        raise ValueError("fixed patch must be at least 2 pixels wide")
    context.update(dapi_patch_len=moving_len, he_patch_len=fixed_len,
                   patch_size_um=size_um, patch_target_mpp=target,
                   moving_patch_mpp=mpps["moving"], fixed_patch_mpp=mpps["fixed"],
                   moving_patch_mpp_source=mpp_sources["moving"],
                   fixed_patch_mpp_source=mpp_sources["fixed"],
                   moving_mpp_estimate=mpp_estimates.get("moving"),
                   fixed_mpp_estimate=mpp_estimates.get("fixed"),
                   mpp_estimate_by_role=mpp_estimates,
                   mpp_estimate_error=mpp_estimate_error,
                   fixed_patch_margin=margin)
    return context


def local_centroid(global_xy, level, side_length):
    """Local coordinates in a fixed-size, edge-padded crop, without clamping anchors."""
    xy = [float(v) / 2 ** level for v in global_xy]
    local = [v - math.floor(v - side_length // 2) for v in xy]
    return [int(v) if v.is_integer() else v for v in local]
