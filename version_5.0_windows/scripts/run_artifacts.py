"""Shared output names; legacy names are read-only resume compatibility."""
from pathlib import Path

INITIAL_ALIGNMENT = "initial_alignment.json"
INITIAL_OVERLAY = "2_initial_overlay_original.png"
LEGACY_INITIAL_ALIGNMENT = "manual_initial_alignment.json"


def initial_alignment_path(run_dir):
    """Find a saved initial matrix, preferring the canonical name if both exist.

    Writers must use INITIAL_ALIGNMENT directly, never this legacy-aware resolver.
    With neither file present, return the canonical path for completion checks and
    useful missing-file errors. Old runs are read without renaming their artifacts.
    """
    root = Path(run_dir)
    current = root / INITIAL_ALIGNMENT
    legacy = root / LEGACY_INITIAL_ALIGNMENT
    return legacy if not current.exists() and legacy.is_file() else current


# ---------------------------------------------------------------- products by stage
#: What each stage writes into a run directory, as glob patterns relative to it. A
#: stage rerun under --force clears its own products first (`clear_stage_products`), so
#: nothing an earlier run left -- a tile the new sampling no longer has, a Stage-6
#: base-image cache from another level -- can be read as if this run had made it.
#: Directories are cleared whole. `{stem}` is the pair's homography stem.
STAGE_PRODUCTS = {
    "1": ("1_*.png", "images_info.json"),
    "2": ("2_*.png", INITIAL_ALIGNMENT, LEGACY_INITIAL_ALIGNMENT,
          "fail_safe_manual_alignment_summary.json"),
    "3": ("tiles", "3_*.png", "sampled_points.json"),
    "4a": ("tiles/*_mask.png", "tiles/nuclei_mask_info.json", "tiles/spatially_selected_tiles.json",
           "tiles/stage7_detection_cache.*", "tiles/stage4b_all_tiles.json"),
    "4b": ("tiles/standout_nuclei.json", "tiles/*_aligned.jpg", "tiles/*_standout.jpg",
           "tiles/anchors", "tiles/density_qc", "tiles/stage4b_qa_manifest.json"),
    "4c": ("nuclei_patches",),
    "5": ("{stem}.json", "{stem}.csv", "{stem}_stage5.json", "{stem}_stage8.json",
          "nuclei_centroids_global_used_for_stage5.json"),
    "6": ("6_*",),
    "7": ("dense_nucleus_pairs", "7_*.png"),
    "8": ("stage8_alignment_refinement.json", "{stem}_stage8.json"),
}


STAGE_ORDER = ("1", "2", "3", "4a", "4b", "4c", "5", "6", "7", "8")


def stages_from(stage):
    """`stage` and every stage after it, in pipeline order."""
    stage = str(stage)
    return STAGE_ORDER[STAGE_ORDER.index(stage):]


def clear_from_stage(run_dir, stage, homography_stem=None):
    """Remove the products of `stage` and of every stage after it.

    What a forced run, a Stage-2 relevel or a Stage-4b climb owes the run directory the
    moment it decides to recompute from `stage`: if it then stops early (--stop-after,
    a failure, Ctrl-C), nothing from the earlier run is left for a resume to mistake for
    this one's. Clearing lazily, stage by stage as each reruns, is not enough for that.
    """
    removed = []
    for later in stages_from(stage):
        removed.extend(clear_stage_products(run_dir, later, homography_stem))
    return removed


def stage_products(run_dir, stage, homography_stem=None):
    """The files and directories `stage` has written into `run_dir`, present now."""
    root = Path(run_dir)
    found = []
    for pattern in STAGE_PRODUCTS[str(stage)]:
        if "{stem}" in pattern:
            if not homography_stem:
                continue
            pattern = pattern.replace("{stem}", homography_stem)
        found.extend(sorted(root.glob(pattern)))
    return found


def clear_stage_products(run_dir, stage, homography_stem=None):
    """Remove `stage`'s products before it runs again; returns what was removed."""
    import shutil

    removed = []
    for path in stage_products(run_dir, stage, homography_stem):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed


def stage3_complete(run_dir):
    """Both role metadata files are required by every later stage."""
    import run_roles
    try:
        return all(run_roles.artifact_path(run_dir, role, "tiles/{token}_tile_info.json").is_file()
                   for role in run_roles.ROLES)
    except (OSError, ValueError, KeyError, TypeError):
        return False
