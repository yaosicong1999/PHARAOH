# ============================================================
# nucleus_masks.py - the nucleus masker for one side of a tile pair, by modality
#
# Stage 4a used to be written as "the DAPI side" and "the H&E side": the moving
# slot always ran the Otsu masker on `*_dapi_u8.png`, the fixed slot always ran the
# hematoxylin masker on `*_he.png`, and `engine_he/` was a second, older copy of
# the same stage that ran the hematoxylin masker twice. That wiring by *slot* is the
# whole reason `he-dapi` and `dapi-dapi` could not run: both maskers existed, they
# were just bolted to a position.
#
# Here a masker is chosen by the side's *modality*, and the side's *token* (the word
# `run_roles` spells it with -- `dapi`, `he`, `hefg`, `hebg`, ...) decides the file
# names. Every pair runs the same code:
#
#     dapi-he    moving `dapi` -> mask_dapi     fixed `he`   -> mask_he
#     he-he      moving `hefg` -> mask_he       fixed `hebg` -> mask_he
#     dapi-dapi  moving `dapifg` -> mask_dapi   fixed `dapibg` -> mask_dapi
#
# For `dapi-he` the tokens are the words 4.10 used, so the files it writes are
# byte-identical to before -- checked against a real slide's Stage-4a output when
# this module replaced the slot-wired code.
#
# One cross-side dependency survives and is named explicitly: the hematoxylin
# masker's stroma recut vetoes a degenerate result against the *DAPI* nuclear
# fraction of the same tile. It is given the partner's token when the partner is
# DAPI and nothing otherwise, so an H&E-onto-H&E run simply skips the veto.
# ============================================================
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
# The dense detectors still live beside the engine's stage scripts. There is one
# engine now, so this is where they are; moving them up here is a later tidy-up.
_ENGINE = _SCRIPTS / "engine_dapi"
if str(_ENGINE) not in sys.path:
    sys.path.append(str(_ENGINE))

from my_utils import (  # noqa: E402
    fill_holes_binary,
    inpaint_pigment,
    remove_small_components,
    segment_super_dark_nuclei_full,
    upsample_tile,
)

MODALITIES = ("dapi", "he")

#: The Stage-3 tile file each modality is masked from. DAPI tiles are the uint8
#: rendering of the 16-bit intensities Stage 3 also keeps; H&E tiles are RGB.
TILE_SUFFIX = {"dapi": "_u8.png", "he": ".png"}
MASK_SUFFIX = "_mask.png"

#: Which `stage4a` keys each masker reads, without their modality prefix. A key is
#: looked up as `<token>_<key>` first, so one side of a same-modality pair can be
#: tuned on its own, then as `<modality>_<key>`.
PARAM_KEYS = {
    "dapi": ("thr_offset", "mask_min_area_factor", "mask_upscale_factor"),
    "he": ("mask_n_smooth", "mask_intensity_threshold", "mask_upscale_factor",
           "mask_source", "pigment_correction", "mask_stroma_recut",
           "mask_stroma_trigger", "mask_stroma_k", "mask_stroma_floor"),
}


def check_modality(modality: str) -> str:
    modality = str(modality).strip().lower()
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    return modality


# -----------------------------
# File names
# -----------------------------
def tile_filename(tile_id: str, token: str, modality: str) -> str:
    return f"{tile_id}_{token}{TILE_SUFFIX[check_modality(modality)]}"


def mask_filename(tile_id: str, token: str) -> str:
    return f"{tile_id}_{token}{MASK_SUFFIX}"


def tile_id_of(path, token: str, modality: str) -> str:
    """The tile id a Stage-3 tile file belongs to, or a ValueError if it is not one."""
    name = os.path.basename(str(path))
    suffix = f"_{token}{TILE_SUFFIX[check_modality(modality)]}"
    if not name.endswith(suffix):
        raise ValueError(f"{name} is not a {token} tile ({suffix})")
    return name[: -len(suffix)]


def tile_ids(tiles_dir, token: str, modality: str):
    """Every tile id that has this side's Stage-3 tile on disk."""
    suffix = f"_{token}{TILE_SUFFIX[check_modality(modality)]}"
    return sorted(
        name[: -len(suffix)]
        for name in os.listdir(str(tiles_dir))
        # FAT/exFAT volumes can expose macOS AppleDouble sidecars such as
        # ``._tile_0001_dapi_u8.png``.  They copy the real suffix but are metadata,
        # not images, and must never enter the Stage-4 tile pool.
        if not name.startswith("._") and name.endswith(suffix)
    )


def mask_coverage(mask_path) -> "float | None":
    """A mask's nuclear fraction, or None when it is not on disk.

    Masks store nuclei as the *dark* pixels and background as 255, because Stage 4b's
    `read_mask` reads them as `img < 128`.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else float((mask < 128).mean())


# -----------------------------
# Parameters
# -----------------------------
def side_params(effective: dict, raw_block: dict, modality: str, token: str) -> dict:
    """This side's masker settings, keyed without their prefix.

    `effective` is the validated flat dict Stage 4a builds (`dapi_thr_offset`,
    `he_mask_source`, ...); `raw_block` is the `stage4a` block itself, consulted for a
    `<token>_` override so one side of `he-he` can differ from the other.
    """
    modality = check_modality(modality)
    out = {}
    for key in PARAM_KEYS[modality]:
        value = effective.get(f"{modality}_{key}")
        if isinstance(raw_block, dict) and f"{token}_{key}" in raw_block:
            value = raw_block[f"{token}_{key}"]
        if f"{token}_{key}" in effective.get("_token_overrides", {}):
            value = effective["_token_overrides"][f"{token}_{key}"]
        out[key] = value
    return out


def stroma_recut_settings(side: dict) -> "dict | None":
    """The recut's three numbers, or None when it is switched off."""
    if not side.get("mask_stroma_recut", True):
        return None
    return {"trigger": float(side["mask_stroma_trigger"]),
            "k": float(side["mask_stroma_k"]),
            "floor": float(side["mask_stroma_floor"])}


# -----------------------------
# Hematoxylin helpers
# -----------------------------
def _dense_detector_background(rgb_tile, dense_params, upscale, engine_dir):
    """The dense detector's nuclei, inverted and rescaled to the mask convention."""
    from dense_nucleus_detection import he_nucleus_labels

    if dense_params is None:
        raise ValueError("mask_source='dense_detector' needs the dense parameters")
    labels, detector_upsample = he_nucleus_labels(rgb_tile, dense_params, engine_dir)
    height, width = rgb_tile.shape[:2]
    target = (int(width * upscale), int(height * upscale))
    if labels is None:
        return np.ones((target[1], target[0]), dtype=bool)
    nuclei = (labels > 0).astype(np.uint8)
    if int(detector_upsample) != int(upscale):
        # The two paths have their own upsample factors; the file Stage 4b opens has
        # to keep the Macenko path's geometry whichever detector filled it.
        nuclei = cv2.resize(nuclei, target, interpolation=cv2.INTER_NEAREST_EXACT)
    return nuclei == 0


HEMATOXYLIN_REFERENCE = np.array([[0.5626, 0.2159],
                                  [0.7201, 0.8012],
                                  [0.4062, 0.5581]])


def hematoxylin_concentration(rgb, Io=240):
    """Hematoxylin in optical-density units, on the *fixed* reference stain vectors.

    Deliberately not `normalizeStaining`'s per-tile estimate: the same colour has to
    mean the same number in every tile for the levels below to be comparable at all.
    """
    od = -np.log((rgb.reshape(-1, 3).astype(np.float32) + 1) / Io)
    return np.linalg.lstsq(HEMATOXYLIN_REFERENCE, od.T, rcond=None)[0][0].reshape(
        rgb.shape[:2]
    )


def _histogram_mode(values, bins=128):
    hist, edges = np.histogram(values, bins=bins)
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _otsu_level(values, bins=256):
    """Where the tissue's hematoxylin splits into its two populations."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist)
    w2 = w1[-1] - w1
    m1 = np.cumsum(hist * centres)
    m2 = m1[-1] - m1
    with np.errstate(invalid="ignore", divide="ignore"):
        between = w1 * w2 * (
            (m1 / np.maximum(w1, 1e-9)) - (m2 / np.maximum(w2, 1e-9))
        ) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centres[int(np.argmax(between))])


def tissue_background_hematoxylin(values):
    """The non-nuclear population's level: the mode of the *lower* of the two.

    The mode of the tissue as a whole assumes most of the tile is not nucleus. On
    `human_crc_p2`, 42% nuclear by area, that is false: on its densest tiles the plain
    mode lands at 0.892, 1.014 and 1.192 against a slide whose nuclei sit at 1.031 and
    whose background sits at 0.423 -- it has found the nuclei. Splitting first returns
    0.378, 0.213 and 0.587 on those same tiles.
    """
    lower = values[values <= _otsu_level(values)]
    return _histogram_mode(lower if lower.size >= 50 else values)


def recut_hematoxylin_avid_stroma(rgb_tile, background, dapi_coverage,
                                  trigger, k, floor):
    """Re-cut a mask whose threshold fell *below* the tissue it had to sit above.

    `segment_super_dark_nuclei_full` cuts at a level that is near-constant in optical
    density across slides (0.44-0.90 measured over thirteen datasets), while the
    non-nuclear tissue it must clear ranges from 0.091 on `mouse_brain` to 0.551 on
    `artery_P4H`. Arterial wall takes up hematoxylin at 0.748 against 1.223 in a
    nucleus there, so the cut lands underneath the wall and the whole wall is masked:
    the H&E mask covers 39% of the tile where the DAPI puts nuclei at 4%, and Stage 4b
    fails for want of tiles whose Dice ceiling is even reachable.

    The wall and the nuclei are 2.16x apart, which is wider than `human_crc_p2`
    manages, so the tissue is separable and only the cut is misplaced. Returns a
    replacement `background`, or None to keep what was already computed.
    """
    rgb = cv2.resize(np.asarray(rgb_tile)[..., :3],
                     (background.shape[1], background.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    concentration = hematoxylin_concentration(rgb)
    tissue = concentration > 0.05
    nuclei = ~background.astype(bool)
    if tissue.sum() < 500 or nuclei.sum() < 200:
        return None

    values = concentration[tissue]
    level = tissue_background_hematoxylin(values)
    if level <= 1e-6:
        return None
    # Where the shipped cut actually landed, relative to the tissue it has to clear.
    # Below 1 it is under the tissue itself. Measured over thirteen datasets: arterial
    # 0.62-0.94, every other slide 1.79-10.30.
    cut = float(np.percentile(concentration[nuclei], 5))
    if cut / level >= trigger:
        return None

    below = values[values <= level]
    if below.size < 50:
        return None
    spread = float(np.median(np.abs(below - level))) * 1.4826
    if spread <= 0:
        return None

    from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

    recut = concentration > level + k * spread
    recut = remove_small_objects(recut, min_size=30)
    recut = remove_small_holes(recut, area_threshold=60)
    recut = closing(recut, disk(1))          # the cleanup `threshold_super_dark` does

    # Last guard. The background estimate can still be wrong, and when it is the recut
    # lands above everything and empties the mask. The DAPI says how much nucleus is
    # really in this tile, so a recut far below it has not found a cleaner nuclear mask,
    # it has lost the nuclei. The DAPI is used only to veto a degenerate result, never
    # to choose which pixels the H&E mask keeps: matching the two masks by area would
    # inflate Stage 4b's Dice without improving the alignment it certifies.
    if dapi_coverage is not None and recut.mean() < floor * dapi_coverage:
        print(f"[Stage4a] recut left {100 * recut.mean():.1f}% against "
              f"{100 * dapi_coverage:.1f}% DAPI nuclei; keeping the original mask",
              flush=True)
        return None
    return ~recut


# -----------------------------
# The two maskers
# -----------------------------
def _skip_or_cache(image_file, tile_id, dense_components, dense_error):
    return {
        "message": (
            f"Cached dense detections: {os.path.basename(image_file)}"
            if dense_error is None
            else f"Processed without dense cache: {os.path.basename(image_file)} ({dense_error})"
        ),
        "tile_id": tile_id,
        "dense_components": dense_components,
    }


def mask_dapi(
    image_file,
    *,
    token="dapi",
    thr_offset=0,
    mask_min_area_factor=10e-5,
    mask_upscale_factor=2,
    dense_params=None,
    mask_tile_ids=None,
    connectivity=8,
    invert=True,
):
    """Otsu-plus-offset nucleus mask of one DAPI tile.

    Input:  `<tile>_<token>_u8.png`  (uint8 grayscale)
    Output: `<tile>_<token>_mask.png` (nuclei dark, background 255)
    """
    from dense_nucleus_detection import segment_dapi_centroids

    try:
        if not os.path.exists(image_file):
            return f"Skipped (DAPI not found): {image_file}"
        tile_id = tile_id_of(image_file, token, "dapi")
        folder_name = os.path.dirname(image_file)

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Failed: {image_file}, Error: cv2.imread returned None"
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        dense_error = None
        try:
            dense_components = (
                segment_dapi_centroids(img, dense_params)
                if dense_params is not None
                else None
            )
        except Exception as exc:
            dense_components = None
            dense_error = str(exc)
        if mask_tile_ids is not None and tile_id not in mask_tile_ids:
            return _skip_or_cache(image_file, tile_id, dense_components, dense_error)
        upscale = int(mask_upscale_factor)
        img = upsample_tile(img, upscale)

        # 1) Otsu + optional offset
        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(int(otsu_thr) + int(thr_offset), 0, 255))
        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        fg_ratio0 = float((mask > 0).mean())
        # 2) fill holes
        mask = fill_holes_binary(mask)
        fg_ratio1 = float((mask > 0).mean())
        # 3) remove tiny connected components
        min_area = float(mask_min_area_factor) * img.shape[0] ** 2
        min_area = min_area * (upscale ** 2)  # Adjust area threshold after upsampling.
        mask, cc_info = remove_small_components(
            mask, min_area=int(min_area), connectivity=int(connectivity)
        )
        fg_ratio2 = float((mask > 0).mean())
        if invert:
            mask = 255 - mask

        mask_save_path = os.path.join(folder_name, mask_filename(tile_id, token))
        cv2.imwrite(mask_save_path, mask)

        kept = total = None
        if isinstance(cc_info, dict):
            kept = cc_info.get("cc_kept", None)
            total = cc_info.get("cc_total", None)
        extra = f" fg {fg_ratio0:.3f}->{fg_ratio1:.3f}->{fg_ratio2:.3f}"
        if kept is not None and total is not None:
            extra += f" cc kept {kept}/{total}"
        return {
            "message": (f"Processed: {os.path.basename(image_file)} otsu={int(otsu_thr)} "
                        f"used={thr}{extra} saved: {mask_save_path}"),
            "tile_id": tile_id,
            "dense_components": dense_components,
        }
    except Exception as e:
        return f"Failed: {image_file}, Error: {e}"


def mask_he(
    image_file,
    *,
    token="he",
    mask_upscale_factor=2,
    mask_n_smooth=2,
    mask_intensity_threshold=0.6,
    dense_params=None,
    mask_tile_ids=None,
    mask_source="macenko",
    pigment_correction=False,
    stroma_recut=None,
    partner_dapi_token=None,
):
    """Hematoxylin nucleus mask of one H&E tile: `<tile>_<token>_mask.png`.

    `mask_source` selects which of the two detections this function already runs is
    written out:

    * "macenko" -- `segment_super_dark_nuclei_full`: per-tile Macenko normalisation
      then a fixed `mask_intensity_threshold`. Anthracotic pigment corrupts the
      per-tile stain estimate, and on `geo_lung_l8` this leaves the mask nearly empty
      on tiles whose nuclei are plainly visible (PLAN_4.8.md section 10).
    * "dense_detector" -- the labels behind `segment_he_centroids`, which is computed
      on the same tile either way for Stage 7: fixed stain vector, Otsu cut.

    `partner_dapi_token` names the other side of the pair when that side is DAPI; the
    stroma recut then vetoes a degenerate cut against that tile's DAPI nuclear
    fraction. When the partner is not DAPI there is nothing to veto against.

    Nuclei are the *dark* pixels and background is 255, because Stage 4b's
    `read_mask` reads the file as `img < 128`.
    """
    from dense_nucleus_detection import segment_he_centroids

    engine_dir = str(_ENGINE)
    try:
        rgb_tile = np.array(Image.open(image_file))
        tile_id = tile_id_of(image_file, token, "he")
        folder_name = os.path.dirname(image_file)
        dense_error = None
        try:
            dense_components = (
                segment_he_centroids(rgb_tile, dense_params, engine_dir)
                if dense_params is not None
                else None
            )
        except Exception as exc:
            dense_components = None
            dense_error = str(exc)
        if mask_tile_ids is not None and tile_id not in mask_tile_ids:
            return _skip_or_cache(image_file, tile_id, dense_components, dense_error)
        upscale = int(mask_upscale_factor)
        pigment_fraction = 0.0
        if pigment_correction and mask_source != "dense_detector":
            # Only the Macenko path needs this: the dense detector deconvolves onto a
            # fixed stain vector, which pigment cannot move.
            rgb_tile, pigment_fraction = inpaint_pigment(rgb_tile)
        if mask_source == "dense_detector":
            background = _dense_detector_background(
                rgb_tile, dense_params, upscale, engine_dir
            )
        else:
            # `mask_dark` is a misnomer inherited from the GUI: the function returns
            # `H_gray > threshold`, which is the *background*, and the write below
            # stores it as 255.
            _, background = segment_super_dark_nuclei_full(
                rgb_tile,
                upsample_scale=upscale,
                n_smooth=int(mask_n_smooth),
                intensity_threshold=float(mask_intensity_threshold),
            )
        if stroma_recut and mask_source != "dense_detector":
            partner_coverage = None
            if partner_dapi_token:
                partner_coverage = mask_coverage(
                    os.path.join(folder_name, mask_filename(tile_id, partner_dapi_token))
                )
            replacement = recut_hematoxylin_avid_stroma(
                rgb_tile, background, partner_coverage,
                stroma_recut["trigger"], stroma_recut["k"], stroma_recut["floor"],
            )
            if replacement is not None:
                background = replacement
        mask_save_path = os.path.join(folder_name, mask_filename(tile_id, token))
        cv2.imwrite(mask_save_path, background.astype(np.uint8) * 255)
        pigment_note = (f" pigment={100 * pigment_fraction:.2f}%"
                        if pigment_correction else "")
        return {
            "message": f"Processed: {os.path.basename(image_file)}{pigment_note}",
            "tile_id": tile_id,
            "dense_components": dense_components,
        }
    except Exception as e:
        return f"Failed: {os.path.basename(image_file)}, Error: {e}"


MASKERS = {"dapi": mask_dapi, "he": mask_he}


def masker_for(modality: str):
    """The masker a side runs, decided by what the side *is*, never by its slot."""
    return MASKERS[check_modality(modality)]
