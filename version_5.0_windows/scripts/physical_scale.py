"""Level-0-pixel gates, tuned at a reference pixel size, rescaled to the image at hand.

Stages 7 and 8 gate on level-0 pixel counts: nucleus areas, candidate and pair
distances, a RANSAC threshold, a transform shift. Those constants were tuned on the
10x public Xenium datasets, whose H&E is 0.2738 um/px and whose DAPI (morphology
focus) is 0.2125 um/px, so each is really a physical size in disguise. On an image
scanned finer -- the GEO `GSE311609` H&E is 0.08625 um/px, 3.17x finer -- every length
gate is 3.17x too tight and every area gate 10x, which is how B15 lost three quarters
of its H&E nuclei to a 5000 px^2 ceiling that there means 37 um^2, smaller than a
nucleus (TODO.md item 10).

`run_factors` reads each image's OME `PhysicalSizeX` and returns, per modality, the
factor reference / actual; `scale_gates` multiplies each gate by that factor to the
gate's power (1 for a length, 2 for an area). A caller that supplies the run directory
can recover either missing side from Stage 2's moving->fixed level-0 homography and
the other side's header. If both headers are missing, the coarser side is explicitly
assumed to be 20x (0.50 um/px), and the other is derived from the homography. An image
within 1% of its reference gets a factor of exactly 1.0. `he_min_area_level0_px` is a
detection parameter inside Stage 4a's cache snapshot: Stage 4a scales it the same way
before it detects (`DETECTION_GATES`), so the snapshot and Stage 7's expectation
agree. The cross-tile dedup radius is applied per side, each in its own frame (Stage
7's `dedup_radii`).
"""

#: the detection parameter Stage 4a scales before it caches, identically to Stage 7
DETECTION_GATES = {"he_min_area_level0_px": ("he", 2)}


def reference_mpp_from_block(block):
    """Per-modality references from a `stage7_dense_pairs` block, defaults otherwise."""
    return {
        "dapi": float(block.get("reference_mpp_dapi", REFERENCE_MPP["dapi"])),
        "he": float(block.get("reference_mpp_he", REFERENCE_MPP["he"])),
    }
import math
import json
from pathlib import Path

#: um/px the level-0 constants were tuned at, per modality
REFERENCE_MPP = {"dapi": 0.2125, "he": 0.2738}
#: |factor - 1| at or below this is treated as exactly 1
TOLERANCE = 0.01

def stage7_gates(fixed_modality="he"):
    """Stage-7 gates: key -> (modality whose pixels it counts, power).

    The area gates are named by modality and count that modality's pixels whichever
    role it plays; the distances are measured in the fixed image's frame, where the
    moving nuclei are projected and matched.
    """
    return {
        "dapi_min_area_level0_px": ("dapi", 2),
        "dapi_max_area_level0_px": ("dapi", 2),
        "he_min_area_level0_px": ("he", 2),
        "he_max_area_level0_px": ("he", 2),
        "candidate_distance_level0_px": (fixed_modality, 1),
        "max_pair_distance_level0_px": (fixed_modality, 1),
        "high_confidence_distance_level0_px": (fixed_modality, 1),
    }


def stage8_gates(moving_modality="dapi", fixed_modality="he"):
    """Stage-8 gates: reprojection, shift and degradation live in the fixed frame; the
    axis span is measured on the moving side."""
    return {
        "ransac_reproj_threshold_px": (fixed_modality, 1),
        "max_holdout_median_degradation_px": (fixed_modality, 1),
        "max_transform_shift_p95_px": (fixed_modality, 1),
        "min_axis_span_level0_px": (moving_modality, 1),
    }


#: the dapi-he instances, which every earlier run used
STAGE7_GATES = stage7_gates("he")
STAGE8_GATES = stage8_gates("dapi", "he")


def image_mpp(path):
    """Level-0 um/px (x axis) of an OME-TIFF, or None when it has none or cannot be read."""
    try:
        import tifffile
        from my_utils import ome_physical_pixel_size

        with tifffile.TiffFile(str(path)) as tif:
            size = ome_physical_pixel_size(tif)
    except Exception:
        return None
    if not size:
        return None
    value = float(size[0])
    return value if math.isfinite(value) and value > 0 else None


def scale_factor(reference_mpp, actual_mpp, tolerance=TOLERANCE):
    """reference / actual, or exactly 1.0 when unknown or within `tolerance` of 1."""
    if actual_mpp is None or not math.isfinite(actual_mpp) or actual_mpp <= 0:
        return 1.0
    factor = float(reference_mpp) / float(actual_mpp)
    return 1.0 if abs(factor - 1.0) <= float(tolerance) else factor


def _estimate_missing_mpps(roles, mpp_by_role, run_dir):
    """Resolve missing MPP values from Stage 2, including the explicit Case-4 prior.

    Returns ``(values, sources, estimates, error)``. With one real header, that side
    is the absolute anchor. With neither, the homography supplies only the ratio and
    the coarser side is assumed to be 20x. If the Case-4 relative scale is rejected,
    both sides use equal 20x as a visible last resort so physical-patch stages can
    still run.
    """
    missing_roles = [role for role, value in mpp_by_role.items() if value is None]
    if run_dir is None or not missing_roles:
        return {}, {}, {}, None
    try:
        from alignment_geometry import (ASSUMED_COARSE_MPP_20X,
                                        estimate_mpp_pair_from_homography)
        from my_utils import inspect_image_pyramid
        from run_artifacts import initial_alignment_path

        alignment = json.loads(initial_alignment_path(Path(run_dir)).read_text())
        homography = alignment.get("H_mat_level_0")
        if homography is None:
            raise ValueError("initial alignment has no H_mat_level_0")
        moving_pyramid = inspect_image_pyramid(roles.moving.path)
        fixed_pyramid = inspect_image_pyramid(roles.fixed.path)
        estimate = estimate_mpp_pair_from_homography(
            homography,
            moving_pyramid["level_shapes"][0],
            fixed_pyramid["level_shapes"][0],
            moving_base_mpp=mpp_by_role["moving"],
            fixed_base_mpp=mpp_by_role["fixed"],
        )
        values = {
            role: float(estimate[f"{role}_base_mpp"])
            for role in missing_roles
        }
        if len(missing_roles) == 2:
            sources = {
                estimate["anchor_role"]: "assumed_20x",
                estimate["derived_role"]: "stage2_homography_from_assumed_20x",
            }
        elif missing_roles == ["moving"]:
            sources = {"moving": "stage2_homography_from_fixed_ome"}
        else:
            sources = {"fixed": "stage2_homography_from_moving_ome"}
        estimates = {role: estimate for role in missing_roles}
        return values, sources, estimates, None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        if len(missing_roles) == 2:
            # Import here too in case opening the alignment failed before the import
            # above completed in a future refactor.
            from alignment_geometry import ASSUMED_COARSE_MPP_20X

            values = {role: ASSUMED_COARSE_MPP_20X for role in missing_roles}
            sources = {
                role: "assumed_20x_without_trustworthy_stage2_scale"
                for role in missing_roles
            }
            return values, sources, {}, str(exc)
        return {}, {}, {}, str(exc)


def run_factors(info, references=None, tolerance=TOLERANCE, run_dir=None):
    """Per-modality scale factors for a run, from its `images_info.json` record.

    Returns {"mpp": {modality: um/px or None}, "reference_mpp": {...},
    "factor": {modality: float}, "roles": {"moving": float, "fixed": float},
    "mpp_by_role": {...}, "mpp_source_by_role": {...}}. The modality-keyed factors
    serve the gates, which are named by modality (a mixed pair maps each modality to
    one side); a same-modality pair has one modality and two images, so anything that
    means *a side* -- Stage 5's threshold in the fixed frame -- must read `roles`,
    never `factor`.
    """
    import run_roles

    references = dict(REFERENCE_MPP, **(references or {}))
    roles = run_roles.read(info)
    mpp_by_role = {
        "moving": image_mpp(roles.moving.path),
        "fixed": image_mpp(roles.fixed.path),
    }
    sources = {
        role: "ome_header" if value is not None else None
        for role, value in mpp_by_role.items()
    }
    recovered, recovered_sources, estimates, estimate_error = _estimate_missing_mpps(
        roles, mpp_by_role, run_dir)
    for role, value in recovered.items():
        mpp_by_role[role] = float(value)
        sources[role] = recovered_sources[role]
    for role in sources:
        if sources[role] is None:
            sources[role] = "missing"

    out = {"mpp": {}, "reference_mpp": {}, "factor": {}, "roles": {},
           "mpp_by_role": dict(mpp_by_role), "mpp_source_by_role": sources,
           "mpp_estimate_by_role": estimates, "tolerance": float(tolerance)}
    if estimate_error is not None:
        out["mpp_estimate_error"] = estimate_error
    for role, side in (("moving", roles.moving), ("fixed", roles.fixed)):
        modality = side.modality
        actual = mpp_by_role[role]
        factor = scale_factor(references[modality], actual, tolerance)
        out["roles"][role] = factor
        if modality not in out["factor"]:
            out["mpp"][modality] = actual
            out.setdefault("mpp_source", {})[modality] = sources[role]
            out["reference_mpp"][modality] = float(references[modality])
            out["factor"][modality] = factor
    return out


def scale_gates(params, factors, spec):
    """A copy of `params` with each gate in `spec` scaled, and a report of what moved."""
    scaled = dict(params)
    report = {}
    for key, (modality, power) in spec.items():
        if key not in params or modality not in factors:
            continue
        factor = float(factors[modality]) ** int(power)
        before = float(params[key])
        after = before * factor
        scaled[key] = after
        if factor != 1.0:
            report[key] = {"modality": modality, "power": int(power),
                           "before": before, "after": after}
    return scaled, report


def describe(scale):
    """One log line: the factors and where they came from, per side when the two
    sides share a modality."""
    parts = []
    roles = scale.get("roles", {})
    def spacing_text(actual, source):
        if actual is None:
            return "no OME pixel size"
        if source in ("stage2_homography_from_fixed_ome",
                      "stage2_homography_from_moving_ome"):
            suffix = " (Stage-2 estimate)"
        elif source == "assumed_20x":
            suffix = " (assumed 20x)"
        elif source == "stage2_homography_from_assumed_20x":
            suffix = " (Stage-2 relative scale from assumed 20x)"
        elif source in ("assumed_20x_without_trustworthy_stage2_scale",
                        "assumed_20x_without_stage2_scale"):
            suffix = " (assumed 20x; no trustworthy Stage-2 scale)"
        else:
            suffix = ""
        return f"{actual:.5f} um/px{suffix}"

    if roles and len(scale["factor"]) < 2:
        modality = next(iter(scale["factor"]))
        for role, factor in roles.items():
            actual = scale.get("mpp_by_role", {}).get(role)
            source = scale.get("mpp_source_by_role", {}).get(role)
            parts.append(
                f"{role} ({modality}): {spacing_text(actual, source)}"
                f" (reference {scale['reference_mpp'][modality]:.4f}) -> x{factor:.4f}")
        return "; ".join(parts)
    for modality, factor in scale["factor"].items():
        actual = scale["mpp"].get(modality)
        source = scale.get("mpp_source", {}).get(modality)
        parts.append(
            f"{modality}: {spacing_text(actual, source)}"
            f" (reference {scale['reference_mpp'][modality]:.4f}) -> x{factor:.4f}")
    return "; ".join(parts)
