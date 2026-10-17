# ============================================================
# 4a_generate_nuclei_masks.py - Stage 4a: Nuclei mask generation
#
# Masks every Stage-3 tile on both sides of the pair, in parallel, and writes
# nuclei_mask_info.json plus the Stage-7 detection cache.
#
# Each side is masked by what it *is*: `nucleus_masks.masker_for(modality)` hands
# back the Otsu masker for a DAPI side and the hematoxylin masker for an H&E side,
# and the side's token from `run_roles` names its files. This stage used to be
# written as "the DAPI side" then "the H&E side", which bolted each masker to a
# slot and is why `he-dapi` and `dapi-dapi` could not run; `engine_he/` was a
# second, older copy of the stage that ran the hematoxylin masker twice.
#
# Usage: python 4a_generate_nuclei_masks.py <TILES_DIR> [--selected-masks-only]
#                                                       [--add-tiles N]
# ============================================================
import warnings
import os
from pathlib import Path
import sys
from glob import glob
import numpy as np
import cv2
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dense_nucleus_detection import (
    CACHE_META_NAME,
    CACHE_NPZ_NAME,
    CACHE_SCHEMA_VERSION,
    DETECTION_DEFAULTS,
    detection_params_snapshot,
)
ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION_ROOT = os.path.dirname(ENGINE_ROOT)
if VERSION_ROOT not in sys.path:
    sys.path.insert(0, VERSION_ROOT)
import run_roles
import nucleus_masks
from nucleus_masks import (
    mask_filename,
    masker_for,
    side_params,
    stroma_recut_settings,
    tile_filename,
    tile_ids,
)
from spatial_tile_selection import (
    create_selection_manifest,
    select_spatially_uniform_tile_ids,
)
from stage4_all_tiles import COMPLETION_FILE, shared_stage3_ids
warnings.filterwarnings('ignore')


# -----------------------------
# Utilities
# -----------------------------
def load_stage4a_params_from_parameters_json(
    out_folder: str,
    default_dapi_offset=0,
    default_dapi_min_area_factor=10e-5,
    default_dapi_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
    *, pair=None,
):
    """
    Read ../../parameters.json and return stage4a params.
    Falls back to the provided defaults if the file or keys are missing.
    """
    from my_utils import cli_params_block

    params = {
        "dapi_thr_offset": int(default_dapi_offset),
        "dapi_mask_min_area_factor": float(default_dapi_min_area_factor),
        "dapi_mask_upscale_factor": int(default_dapi_upscale),
        "he_mask_n_smooth": int(default_he_n_smooth),
        "he_mask_intensity_threshold": float(default_he_thr),
        "he_mask_upscale_factor": int(default_he_upscale),
        "he_mask_source": "macenko",
        "he_pigment_correction": False,
        "tile_screen_min_dapi_coverage": None,
        "tile_screen_max_pigment": None,
        "he_mask_stroma_recut": True,
        "he_mask_stroma_trigger": 1.5,
        "he_mask_stroma_k": 3.0,
        "he_mask_stroma_floor": 0.25,
    }

    data = cli_params_block(pair)

    stage4a = data.get("stage4a", {}) if isinstance(data, dict) else {}

    try:
        params["dapi_thr_offset"] = int(
            float(stage4a.get("dapi_thr_offset", params["dapi_thr_offset"]))
        )
    except Exception:
        pass

    try:
        params["dapi_mask_min_area_factor"] = float(
            stage4a.get("dapi_mask_min_area_factor", params["dapi_mask_min_area_factor"])
        )
    except Exception:
        pass

    try:
        params["dapi_mask_upscale_factor"] = int(
            stage4a.get("dapi_mask_upscale_factor", params["dapi_mask_upscale_factor"])
        )
    except Exception:
        pass

    try:
        params["he_mask_n_smooth"] = int(
            stage4a.get("he_mask_n_smooth", params["he_mask_n_smooth"])
        )
    except Exception:
        pass

    try:
        params["he_mask_intensity_threshold"] = float(
            stage4a.get("he_mask_intensity_threshold", params["he_mask_intensity_threshold"])
        )
    except Exception:
        pass

    source = str(stage4a.get("he_mask_source", params["he_mask_source"])).strip()
    if source not in ("macenko", "dense_detector"):
        raise ValueError(
            f"stage4a.he_mask_source must be 'macenko' or 'dense_detector', got {source!r}"
        )
    params["he_mask_source"] = source
    params["he_pigment_correction"] = bool(
        stage4a.get("he_pigment_correction", params["he_pigment_correction"])
    )
    for key in ("tile_screen_min_dapi_coverage", "tile_screen_max_pigment"):
        value = stage4a.get(key, params[key])
        params[key] = None if value is None else float(value)
    params["he_mask_stroma_recut"] = bool(
        stage4a.get("he_mask_stroma_recut", params["he_mask_stroma_recut"])
    )
    for key in ("he_mask_stroma_trigger", "he_mask_stroma_k", "he_mask_stroma_floor"):
        params[key] = float(stage4a.get(key, params[key]))

    try:
        params["he_mask_upscale_factor"] = int(
            stage4a.get("he_mask_upscale_factor", params["he_mask_upscale_factor"])
        )
    except Exception:
        pass

    # The raw block travels with the flat dict so a `<token>_` override on one side
    # of a same-modality pair can be found by `nucleus_masks.side_params`.
    params["_raw_block"] = dict(stage4a) if isinstance(stage4a, dict) else {}
    return params

def load_stage4a_effective_params(
    out_folder: str,
    default_dapi_offset=0,
    default_dapi_min_area_factor=10e-5,
    default_dapi_upscale=2,
    default_he_n_smooth=2,
    default_he_thr=0.6,
    default_he_upscale=2,
    *, pair=None,
):
    """
    Resolve Stage 4a parameters with this priority:
      1) parameters.json -> stage4a
      2) pilot_output_parameters.json overrides:
           - dapi_thr_offset
           - he_mask_intensity_threshold
      3) hard-coded defaults
    """
    params = load_stage4a_params_from_parameters_json(
        out_folder,
        default_dapi_offset=default_dapi_offset,
        default_dapi_min_area_factor=default_dapi_min_area_factor,
        default_dapi_upscale=default_dapi_upscale,
        default_he_n_smooth=default_he_n_smooth,
        default_he_thr=default_he_thr,
        default_he_upscale=default_he_upscale,
        pair=pair,
    )

    pilot_path = os.path.join(out_folder, "../pilot_tiles", "pilot_output_parameters.json")
    if not os.path.exists(pilot_path):
        print(f"[WARN] {pilot_path}: not exists.", flush=True)
        return params
    try:
        with open(pilot_path, "r") as f:
            data = json.load(f)
            print(f"[WARN] loading pilot parameters.", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to read {pilot_path}: {e}. Use parameters.json/defaults.", flush=True)
        return params

    src = data.get("mask_preview", data) if isinstance(data, dict) else {}

    dapi_off = src.get("dapi_thr_offset", params["dapi_thr_offset"])
    he_thr = src.get("he_intensity_threshold", params["he_mask_intensity_threshold"])

    try:
        dapi_off = int(float(dapi_off))
    except Exception:
        dapi_off = params["dapi_thr_offset"]
    dapi_off = max(-100, min(100, dapi_off))

    try:
        he_thr = float(he_thr)
    except Exception:
        he_thr = params["he_mask_intensity_threshold"]
    he_thr = max(0.0, min(1.0, he_thr))

    params["dapi_thr_offset"] = dapi_off
    params["he_mask_intensity_threshold"] = he_thr
    from pilot_mask_parameters import apply_token_overrides
    apply_token_overrides(params, src)
    return params

def run_parallel(file_list, worker_fn, stage_name, max_workers=8, print_fail=False):
    """
    Run a single-file worker function over a list of files in parallel and
    print progress in the format expected by the gallery launcher.
    """
    results = [None] * len(file_list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker_fn, path): index
            for index, path in enumerate(file_list)
        }
        total = len(future_to_index)
        done = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            msg = future.result()
            results[index] = msg
            done += 1
            print(f"[PROGRESS] {stage_name} {done}/{total}", flush=True)
    if print_fail:
        for msg in results:
            text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
            if text.startswith("Failed:"):
                print(f"[{stage_name} FAIL] {text}", flush=True)
    return results


def _result_message(result):
    return result.get("message", "") if isinstance(result, dict) else str(result)


def load_dense_detection_params(pair=None):
    """Load the effective Stage-7 detector parameters for cache production."""
    from my_utils import cli_params_block

    configured = cli_params_block(pair).get("stage7_dense_pairs", {})
    params = dict(DETECTION_DEFAULTS)
    if isinstance(configured, dict):
        for key in params:
            if key in configured:
                params[key] = configured[key]
    return params


def dense_detection_params_for(roles, run_dir):
    """Stage 7's detector settings for this run, or None when Stage 7 has no reader.

    Stages 7-8 pair nuclei across modalities and never run for a same-modality pair,
    so caching dense detections for `he-he` or `dapi-dapi` was work with no consumer --
    and it read every Stage-3 tile beyond the mask subset for nothing. For a mixed pair
    the H&E minimum area is scaled to the image's pixel size exactly as Stage 7 scales
    it (`physical_scale.DETECTION_GATES`), so the cache snapshot Stage 7 checks matches.
    """
    if roles.pair.same_modality:
        return None
    import physical_scale
    from my_utils import cli_params_block

    params = load_dense_detection_params(roles.params_block)
    block = cli_params_block(roles.params_block).get("stage7_dense_pairs", {})
    try:
        info = json.load(open(os.path.join(run_dir, "images_info.json")))
    except (OSError, ValueError):
        return params
    scale = physical_scale.run_factors(
        info, physical_scale.reference_mpp_from_block(block), run_dir=run_dir,
    )
    scaled, report = physical_scale.scale_gates(params, scale["factor"], physical_scale.DETECTION_GATES)
    for key, change in report.items():
        print(f"[INFO] dense detection {key}: {change['before']:g} -> {change['after']:g} "
              f"level-0 px (physical scale, as Stage 7 applies it)", flush=True)
    return scaled


def load_stage4b_tile_count(default=60, available_tiles=None, *, pair=None):
    """How many Stage-3 tiles Stage 4 works on; the rule lives in `stage4_tile_budget`
    so Stage 4b scores exactly what Stage 4a masks."""
    from stage4_tile_budget import stage4b_tile_count

    return stage4b_tile_count(available_tiles=available_tiles, pair=pair, default=default)


# -----------------------------
# Sides
# -----------------------------
class Side:
    """One side of the pair as Stage 4a sees it: what it is and what it is called."""

    def __init__(self, image, role):
        self.role = role
        self.modality = image.modality
        self.token = image.token

    def tile_file(self, out_folder, tile_id):
        return os.path.join(out_folder, tile_filename(tile_id, self.token, self.modality))

    def mask_file(self, out_folder, tile_id):
        return os.path.join(out_folder, mask_filename(tile_id, self.token))

    def tile_ids(self, out_folder):
        return tile_ids(out_folder, self.token, self.modality)


def resolve_sides(out_folder):
    """The pair's two sides, DAPI sides first.

    The hematoxylin masker's stroma recut reads the same tile's DAPI mask, so every
    DAPI side has to be masked before any H&E side starts. For `dapi-he` that is the
    order the stage always had.
    """
    run_dir = os.path.abspath(os.path.join(out_folder, os.pardir))
    roles = run_roles.read(run_dir)
    sides = [Side(roles.moving, run_roles.MOVING), Side(roles.fixed, run_roles.FIXED)]
    sides.sort(key=lambda side: 0 if side.modality == "dapi" else 1)
    return roles, sides


def partner_dapi_token(sides, side):
    """The other side's token when the other side is DAPI, else None."""
    for other in sides:
        if other is not side and other.modality == "dapi":
            return other.token
    return None


def stage3_tile_levels(out_folder, sides):
    """The pyramid level each side's tiles were cut at, keyed by token."""
    run_dir = os.path.abspath(os.path.join(out_folder, os.pardir))
    sampled_path = os.path.join(run_dir, "sampled_points.json")
    info_path = os.path.join(run_dir, "images_info.json")
    with open(sampled_path, "r") as handle:
        sampled = json.load(handle)
    with open(info_path, "r") as handle:
        info = json.load(handle)
    # Stage 3 records the levels it read under the 4.10 names; the fallback rule below
    # is older still and assumes both sides step down by the same number of levels.
    legacy = {run_roles.MOVING: "dapi_tile_level", run_roles.FIXED: "he_tile_level"}
    levels = {}
    for side in sides:
        recorded = sampled.get(f"{side.token}_tile_level", sampled.get(legacy[side.role]))
        if recorded is not None:
            levels[side.token] = max(0, int(recorded))
            continue
        if side.role == run_roles.MOVING:
            levels[side.token] = 1
        else:
            moving_level = levels.get(
                next(s.token for s in sides if s.role == run_roles.MOVING), 1)
            levels[side.token] = max(
                1, int(info.get("HE_level", 1))
                - (int(info.get("DAPI_level", moving_level)) - moving_level),
            )
    return levels


def write_dense_detection_cache(out_folder, results_by_side, dense_params, levels):
    """Write compact local centroid/area arrays for Stage 7, keyed by side token.

    For `dapi-he` the keys are `dapi_*` and `he_*`, which is what Stage 7 reads.
    """
    valid = []
    for side, results in results_by_side:
        for result in results:
            if not isinstance(result, dict) or result.get("dense_components") is None:
                continue
            valid.append((side.token, result["tile_id"], result["dense_components"]))

    tokens = [side.token for side, _ in results_by_side]
    tile_keys = sorted({tile_id for _, tile_id, _ in valid})
    successful_tiles = {
        token: sorted(tile_id for t, tile_id, _ in valid if t == token)
        for token in tokens
    }
    tile_index = {tile_id: index for index, tile_id in enumerate(tile_keys)}
    arrays = {}
    for token in tokens:
        batches = []
        for t, tile_id, components in valid:
            if t != token or tile_id not in tile_index:
                continue
            components = np.asarray(components, dtype=np.float64).reshape(-1, 3)
            if len(components):
                batches.append(
                    np.column_stack([
                        np.full(len(components), tile_index[tile_id], dtype=np.float64),
                        components,
                    ])
                )
        arrays[token] = (
            np.concatenate(batches)
            if batches
            else np.empty((0, 4), dtype=np.float64)
        )

    cache_path = os.path.join(out_folder, CACHE_NPZ_NAME)
    np.savez(
        cache_path,
        tile_keys=np.asarray(tile_keys, dtype="U32"),
        **{f"{token}_components": arrays[token] for token in tokens},
    )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "producer": "stage4a_exact_stage7_detectors",
        "cache_file": CACHE_NPZ_NAME,
        "coordinate_system": "native Stage-3 tile pixels; rows are tile_index,x,y,area",
        "detection_parameters": detection_params_snapshot(dense_params),
    }
    for token in tokens:
        metadata[f"{token}_tile_level"] = int(levels[token])
    metadata["tile_count"] = len(tile_keys)
    for token in tokens:
        metadata[f"{token}_successful_tile_ids"] = successful_tiles[token]
    for token in tokens:
        metadata[f"{token}_detection_count"] = int(len(arrays[token]))
    meta_path = os.path.join(out_folder, CACHE_META_NAME)
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        "[INFO] Saved Stage-7 detection cache: "
        + ", ".join(f"{token}={len(arrays[token]):,}" for token in tokens)
        + f" -> {cache_path}",
        flush=True,
    )
    return metadata


def screen_tiles_for_pairing(out_folder, eligible_ids, min_dapi_coverage,
                             max_pigment, requested, dapi_side, he_side):
    """Narrow Stage 4b's candidate tiles to ones both modalities can actually pair on.

    Stage 4b pairs isolated DAPI nuclei against isolated H&E nuclei, and its tile score
    is a Dice between the two masks. Two conditions make a tile useless for that before
    any of it runs, and both can be read off the Stage-3 tiles without building a mask:

    * **too few DAPI nuclei.** `geo_lung_l7` is alveolar parenchyma; a fifth of its
      tiles hold under 25 nuclei and score near zero however well they are aligned
      (PLAN_4.8.md sections 8-9).
    * **too much pigment.** On `geo_lung_l8`, tiles heavy with anthracotic pigment
      produce an H&E mask that is up to 87% soot, and the agreement it does show with
      the DAPI is carried by that soot rather than by nuclei (sections 10, 18).

    The two conditions are about a DAPI side and an H&E side, so the screen only
    applies to a mixed pair; a same-modality pair keeps every eligible tile.

    Never returns fewer than `requested` tiles: if the screen is too strict for a
    slide, the rejected tiles are added back best-first, so a run can only gain
    candidates by this, never lose them.
    """
    if min_dapi_coverage is None and max_pigment is None:
        return list(eligible_ids), None
    if dapi_side is None or he_side is None:
        return list(eligible_ids), None

    from my_utils import find_pigment

    scored, report = {}, {"screened": 0, "kept": 0, "backfilled": 0}
    for tile_id in eligible_ids:
        dapi_path = dapi_side.tile_file(out_folder, tile_id)
        he_path = he_side.tile_file(out_folder, tile_id)
        if not (os.path.exists(dapi_path) and os.path.exists(he_path)):
            continue
        image = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        cut, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dapi = 100.0 * float((image > cut).mean())
        he_rgb = cv2.imread(he_path, cv2.IMREAD_COLOR)
        pigment = (100.0 * float(find_pigment(he_rgb[..., ::-1]).mean())
                   if he_rgb is not None else 0.0)
        scored[tile_id] = (dapi, pigment)

    report["screened"] = len(scored)
    if not scored:
        return list(eligible_ids), None

    def passes(tile_id):
        dapi, pigment = scored[tile_id]
        return ((min_dapi_coverage is None or dapi >= float(min_dapi_coverage))
                and (max_pigment is None or pigment <= float(max_pigment)))

    kept = [tile_id for tile_id in scored if passes(tile_id)]
    report["kept"] = len(kept)
    if len(kept) < int(requested):
        # Rank the rejects by how far each is inside the screen and take the best.
        rest = sorted(
            (tile_id for tile_id in scored if tile_id not in set(kept)),
            key=lambda t: (-scored[t][0] / max(float(min_dapi_coverage or 1.0), 1e-6)
                           + scored[t][1] / max(float(max_pigment or 1.0), 1e-6)),
        )
        needed = int(requested) - len(kept)
        kept = kept + rest[:needed]
        report["backfilled"] = min(needed, len(rest))
    return kept, report


def prepare_stage4_selection(out_folder, sides, stage4a_params, stage4b_tile_count,
                             shared_ids, *, persist=True):
    """The same pre-mask screen and spatial selection for CLI and GUI preview."""
    dapi_side = next((s for s in sides if s.modality == "dapi"), None)
    he_side = next((s for s in sides if s.modality == "he"), None)
    candidate_ids, screen_report = screen_tiles_for_pairing(
        out_folder,
        sorted(shared_ids),
        stage4a_params["tile_screen_min_dapi_coverage"],
        stage4a_params["tile_screen_max_pigment"],
        stage4b_tile_count, dapi_side=dapi_side, he_side=he_side,
    )
    if screen_report is not None:
        print(
            f"[INFO] Tile screen: {screen_report['kept']} of "
            f"{screen_report['screened']} tiles pass "
            f"(DAPI >= {stage4a_params['tile_screen_min_dapi_coverage']}%, "
            f"pigment <= {stage4a_params['tile_screen_max_pigment']}%)"
            + (f"; backfilled {screen_report['backfilled']} to reach "
               f"{stage4b_tile_count}" if screen_report["backfilled"] else ""),
            flush=True,
        )
    return create_selection_manifest(
        out_folder, stage4b_tile_count, eligible_ids=candidate_ids,
        source="stage4a_for_stage4b", persist=persist,
    )


def preview_stage4_selection(out_folder):
    """Select from existing raw tiles, without masks/cache/output writes."""
    roles, sides = resolve_sides(out_folder)
    params = load_stage4a_effective_params(out_folder, pair=roles.params_block)
    shared_ids = set.intersection(*(set(side.tile_ids(out_folder)) for side in sides))
    count = load_stage4b_tile_count(available_tiles=len(shared_ids), pair=roles.params_block)
    _, manifest = prepare_stage4_selection(
        out_folder, sides, params, count, shared_ids, persist=False,
    )
    return manifest


def side_worker(side, sides, stage4a_params, dense_params, mask_tile_ids):
    """The masker for one side with its settings bound, ready for `run_parallel`."""
    settings = side_params(stage4a_params, stage4a_params.get("_raw_block", {}),
                           side.modality, side.token)
    kwargs = {"token": side.token, "dense_params": dense_params,
              "mask_tile_ids": mask_tile_ids}
    if side.modality == "he":
        kwargs["stroma_recut"] = stroma_recut_settings(settings)
        kwargs["partner_dapi_token"] = partner_dapi_token(sides, side)
        for key in ("mask_stroma_recut", "mask_stroma_trigger",
                    "mask_stroma_k", "mask_stroma_floor"):
            settings.pop(key, None)
    kwargs.update(settings)
    return partial(masker_for(side.modality), **kwargs)


def main():
    t0 = time.perf_counter()
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python 4a_generate_nuclei_masks.py <output_folder> "
            "[--all-tiles | --selected-masks-only (the default)] [--add-tiles N|all]"
        )
    out_folder = sys.argv[1]
    # Persisting masks for the spatial Stage-4b subset only is the default for every
    # pair: Stage 4b scores that subset and nothing downstream reads the rest.
    # `--selected-masks-only` is that default spelled out; `--all-tiles` (the GUI's
    # full-slide detection) is the one way to persist every tile's mask.
    all_tiles = "--all-tiles" in sys.argv[2:]
    if all_tiles and "--selected-masks-only" in sys.argv[2:]:
        raise ValueError("--all-tiles cannot be combined with --selected-masks-only")
    selected_masks_only = not all_tiles
    # `--add-tiles N` tops up an existing run instead of starting one: Stage 4b came
    # up short of usable tiles, and Stage 3 extracted three to five times more than
    # Stage 4a masked. It adds N of the unmasked ones after applying the same quality
    # screen, using the already masked tiles as farthest-point anchors so the new tiles
    # fill spatial gaps. It leaves the cache, manifest and existing masks alone, so a
    # run that already succeeded is never re-selected (PLAN_4.8.md section 19).
    # `--add-tiles all` masks every tile still unmasked: the launcher's last rung,
    # density over the whole Stage-3 pool, wants nothing left out.
    add_tiles, add_all_tiles = 0, False
    for index, value in enumerate(sys.argv[2:]):
        if value == "--add-tiles" and index + 1 < len(sys.argv[2:]):
            requested = str(sys.argv[2:][index + 1]).strip().lower()
            if requested == "all":
                add_tiles, add_all_tiles = -1, True
            else:
                add_tiles = int(requested)
    if all_tiles and add_tiles:
        raise ValueError("--all-tiles cannot be combined with --add-tiles")
    print(f"[INFO] Using output folder: {out_folder}", flush=True)

    roles, sides = resolve_sides(out_folder)
    print(f"[INFO] Pair {roles.pair.name}: "
          + ", ".join(f"{s.role}={s.token} ({s.modality} masker)" for s in sides),
          flush=True)
    dapi_side = next((s for s in sides if s.modality == "dapi"), None)
    he_side = next((s for s in sides if s.modality == "he"), None)

    if not add_tiles:
        for stale_name in (CACHE_NPZ_NAME, CACHE_META_NAME):
            stale_path = os.path.join(out_folder, stale_name)
            if os.path.exists(stale_path):
                os.unlink(stale_path)

    stage4a_params = load_stage4a_effective_params(
        out_folder,
        default_dapi_offset=0,
        default_dapi_min_area_factor=10e-5,
        default_dapi_upscale=2,
        default_he_n_smooth=2,
        default_he_thr=0.6,
        default_he_upscale=2,
        pair=roles.params_block,
    )
    dense_params = dense_detection_params_for(
        roles, os.path.abspath(os.path.join(out_folder, os.pardir)))
    levels = stage3_tile_levels(out_folder, sides)
    dense_by_side = {}
    for side in sides:
        if dense_params is None:
            dense_by_side[side.token] = None
            continue
        side_dense = dict(dense_params)
        side_dense["_work_pixel_area_level0"] = float(2 ** levels[side.token]) ** 2
        dense_by_side[side.token] = side_dense
    cache_note = ("dense Stage-7 detections are cached for every tile" if dense_params is not None
                  else f"no Stage-7 cache: {roles.pair.name} is a same-modality pair and runs no "
                       "Stage 7, so only the masked tiles are read")

    print(
        "[INFO] Effective stage4a params: "
        f"dapi_thr_offset={stage4a_params['dapi_thr_offset']}, "
        f"dapi_mask_min_area_factor={stage4a_params['dapi_mask_min_area_factor']}, "
        f"dapi_mask_upscale_factor={stage4a_params['dapi_mask_upscale_factor']}, "
        f"he_mask_n_smooth={stage4a_params['he_mask_n_smooth']}, "
        f"he_mask_intensity_threshold={stage4a_params['he_mask_intensity_threshold']}, "
        f"he_mask_upscale_factor={stage4a_params['he_mask_upscale_factor']}, "
        f"he_mask_source={stage4a_params['he_mask_source']}, "
        f"he_pigment_correction={stage4a_params['he_pigment_correction']}",
        flush=True,
    )

    ids_by_side = {side.token: set(side.tile_ids(out_folder)) for side in sides}
    shared_ids = set.intersection(*ids_by_side.values())
    if all_tiles:
        expected = set(shared_stage3_ids(out_folder, roles))
        missing = sorted(expected - shared_ids)
        if missing:
            raise ValueError(f"Stage 3 raw tile pairs are missing: {missing[:5]}")
        if not expected:
            raise ValueError("No Stage-3 tile pairs to process")
        shared_ids = expected
        ids_by_side = {side.token: expected for side in sides}
        Path(out_folder, COMPLETION_FILE).unlink(missing_ok=True)
        Path(out_folder, "nuclei_mask_info.json").unlink(missing_ok=True)
    stage4b_tile_count = load_stage4b_tile_count(available_tiles=len(shared_ids), pair=roles.params_block)

    if add_tiles:
        fixed_side = next(s for s in sides if s.role == run_roles.FIXED)
        already = {
            tile_id for tile_id in shared_ids
            if os.path.exists(fixed_side.mask_file(out_folder, tile_id))
        }
        remaining = sorted(shared_ids - already)
        if not remaining:
            print("[INFO] Tile top-up: no unmasked tiles remain", flush=True)
            return
        if add_all_tiles:
            chosen = set(remaining)
        else:
            # Preserve the quality screen, but make the actual top-up spatially
            # complementary to every tile already carrying a mask. The screen always
            # returns at least the requested number by backfilling its best rejects.
            candidates, screen_report = screen_tiles_for_pairing(
                out_folder, remaining,
                stage4a_params["tile_screen_min_dapi_coverage"],
                stage4a_params["tile_screen_max_pigment"],
                requested=int(add_tiles), dapi_side=dapi_side, he_side=he_side,
            )
            moving_side = next(s for s in sides if s.role == run_roles.MOVING)
            tile_info_path = Path(out_folder) / f"{moving_side.token}_tile_info.json"
            tile_info = json.loads(tile_info_path.read_text())
            chosen = set(select_spatially_uniform_tile_ids(
                tile_info,
                min(int(add_tiles), len(candidates)),
                eligible_ids=candidates,
                anchor_ids=already,
            ))
            if screen_report is not None:
                print(
                    f"[INFO] Tile top-up screen: {screen_report['kept']} of "
                    f"{screen_report['screened']} remaining tiles pass"
                    + (f"; backfilled {screen_report['backfilled']}"
                       if screen_report["backfilled"] else ""),
                    flush=True,
                )
        print(f"[INFO] Tile top-up: masking {len(chosen)} of {len(remaining)} "
              f"unmasked tiles ({len(already)} already masked"
              f"{'; --add-tiles all' if add_all_tiles else '; spatial gap filling'})",
              flush=True)
        for side in sides:
            run_parallel(
                file_list=[side.tile_file(out_folder, t) for t in sorted(chosen)],
                worker_fn=side_worker(side, sides, stage4a_params, None, chosen),
                stage_name=f"{side.token} top-up", max_workers=8,
            )
        # The manifest is deliberately not rewritten: Stage 4b rebuilds it from the
        # masks that now exist, which is exactly the enlarged set.
        print(f"[INFO] Tile top-up complete: "
              f"{len(already) + len(chosen)} tiles now carry masks", flush=True)
        return

    if all_tiles:
        selection_path, selection_manifest = create_selection_manifest(
            out_folder, len(shared_ids), eligible_ids=shared_ids, source="stage4a_all_tiles",
        )
    else:
        selection_path, selection_manifest = prepare_stage4_selection(
            out_folder, sides, stage4a_params, stage4b_tile_count, shared_ids,
        )
    selected_tile_ids = set(selection_manifest["selected_tile_ids"])
    mask_tile_ids = selected_tile_ids if selected_masks_only else None
    if selected_masks_only:
        print(
            f"[INFO] Stage 4a will persist masks for {len(selected_tile_ids)}/"
            f"{len(shared_ids)} spatially selected tiles (the default; --all-tiles "
            f"persists every tile's); {cache_note}",
            flush=True,
        )
    else:
        print(
            f"[INFO] Stage 4a will persist masks for all "
            f"{len(shared_ids)} tiles for gallery use; {cache_note}",
            flush=True,
        )
    print(f"[INFO] Tile-selection manifest: {selection_path}", flush=True)
    # A forced rerun may contain masks from an older all-tile policy. Remove
    # only masks outside the new deterministic Stage-4b subset.
    if selected_masks_only:
        for side in sides:
            suffix = f"_{side.token}{nucleus_masks.MASK_SUFFIX}"
            for path in glob(os.path.join(out_folder, f"*{suffix}")):
                tile_id = os.path.basename(path)[: -len(suffix)]
                if tile_id not in selected_tile_ids:
                    os.unlink(path)

    # ---------------- each side, DAPI sides first ----------------
    results_by_side = []
    timings = []
    for side in sides:
        tile_ids_to_read = sorted(ids_by_side[side.token])
        if dense_params is None and mask_tile_ids is not None:
            # Nothing to cache for the rest: read only what gets a mask.
            tile_ids_to_read = [t for t in tile_ids_to_read if t in mask_tile_ids]
        images = [side.tile_file(out_folder, t) for t in tile_ids_to_read]
        t_start = time.perf_counter()
        print("[ETA_START]", flush=True)
        print(f"[INFO] Starting {side.token} processing ({side.modality} masker): "
              f"{len(images)} tiles", flush=True)
        results = run_parallel(
            file_list=images,
            worker_fn=side_worker(side, sides, stage4a_params,
                                  dense_by_side[side.token], mask_tile_ids),
            stage_name=side.token,
            max_workers=8,
            print_fail=True,
        )
        n_fail = sum(_result_message(msg).startswith("Failed:") for msg in results)
        if all_tiles and n_fail:
            raise RuntimeError(f"{side.token} masking failed for {n_fail} tiles; full-GUI masking is incomplete")
        print(f"[INFO] {side.token} done. failed={n_fail}/{len(images)}", flush=True)
        timings.append((side.token, time.perf_counter() - t_start))
        results_by_side.append((side, results))

    if dense_params is not None:
        cache_metadata = write_dense_detection_cache(
            out_folder, results_by_side, dense_params, levels,
        )
    else:
        cache_metadata = None
        print(f"[INFO] No Stage-7 detection cache written ({roles.pair.name}: same-modality "
              "pair, Stages 7-8 do not run)", flush=True)

    # ---------------- Summary ----------------
    t1 = time.perf_counter()
    print("\n================ Timing Summary ================")
    for token, seconds in timings:
        print(f"{token:<8} stage time : {seconds:.2f} s")
    print(f"Total time      : {t1 - t0:.2f} s")
    print("================================================\n")
    print("[DONE] Nuclei masking finished", flush=True)

    mask_info = {
        "mask_scale": {
            side.token: int(side_params(
                stage4a_params, stage4a_params.get("_raw_block", {}),
                side.modality, side.token)["mask_upscale_factor"])
            for side in sides
        },
        "effective_params": {
            "dapi_thr_offset": stage4a_params["dapi_thr_offset"],
            "dapi_mask_min_area_factor": stage4a_params["dapi_mask_min_area_factor"],
            "dapi_mask_upscale_factor": stage4a_params["dapi_mask_upscale_factor"],
            "he_mask_n_smooth": stage4a_params["he_mask_n_smooth"],
            "he_mask_intensity_threshold": stage4a_params["he_mask_intensity_threshold"],
            "he_mask_upscale_factor": stage4a_params["he_mask_upscale_factor"],
        },
        "stage7_detection_cache": cache_metadata,
        "stage4b_mask_selection": {
            "manifest": os.path.basename(selection_path),
            "selected_tile_count": len(selected_tile_ids),
            "available_tile_count": len(shared_ids),
            "persisted_mask_policy": (
                "stage4b_spatial_subset" if selected_masks_only else "all_tiles"
            ),
            "persisted_mask_tile_count": (
                len(selected_tile_ids)
                if selected_masks_only
                else len(shared_ids)
            ),
        },
        "sides": {
            side.role: {"token": side.token, "modality": side.modality}
            for side in sides
        },
    }
    json_path = os.path.join(out_folder, "nuclei_mask_info.json")
    with open(json_path, "w") as f:
        json.dump(mask_info, f, indent=2)
    print(f"[INFO] Saved nuclei mask info -> {json_path}", flush=True)

if __name__ == "__main__":
    main()
