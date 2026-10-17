"""Estimate Stage-3's peak memory from what it is about to read.

A blanket `--stage3-memory-per-job-gib` cannot be right for every slide. Set low it
lets the gate admit a dataset the RAM cannot take; set high it can push the admission
threshold above the machine's free memory, and with nothing running there is then no
job left to finish and release any -- the queue head waits forever.

Stage 3's peak is dominated by one thing: it holds a whole pyramid level of both
slides at once. That size is known before the stage starts, from metadata alone, so
the estimate can be derived per dataset instead of guessed once for all of them.

Fitted against the nine-dataset manifest, with each run's peak RSS sampled once a
second by `--monitor`:

    peak_gib ~= 2.95 + 0.68 * read_gib        (max residual 0.53 GiB over 9 datasets)

`read_gib` is the DAPI tile level (uint16, one channel) plus the H&E tile level
(uint8, three channels). The intercept is the part that does not scale: interpreter,
OpenCV/tifffile, the tissue and density masks, and the CVT sampling arrays.
"""

from __future__ import annotations


# peak_gib = (BASELINE_GIB + PER_READ_GIB * read_gib) * SAFETY_MARGIN
BASELINE_GIB = 3.0
PER_READ_GIB = 0.7
SAFETY_MARGIN = 1.3      # covers all nine measured peaks; x1.0 covers only four

# Returned when the pyramids cannot be inspected at all, so the caller still has a
# number rather than falling back to a constant that may deadlock the gate.
FALLBACK_GIB = 8.0

DEFAULT_TARGET_TILE_MPP = 0.425
ASSUMED_BASE_MPP_40X = 0.25


def _level_bytes(shape, bytes_per_pixel):
    height, width = int(shape[0]), int(shape[1])
    return float(height) * float(width) * float(bytes_per_pixel)


def _tile_level(pyramid, target_mpp, *, assume_finer_when_unknown):
    """The level Stage 3 will read, by the same rule Stage 3 uses.

    `choose_level_by_target_mpp` is imported rather than reimplemented so the estimate
    cannot drift from the selection it is estimating.
    """
    from my_utils import choose_level_by_target_mpp

    shapes = pyramid.get("level_shapes") or []
    if not shapes:
        return None
    n_levels = len(shapes)
    base_mpp = pyramid.get("base_mpp")
    if base_mpp is not None:
        level = choose_level_by_target_mpp(base_mpp, n_levels, target_mpp)
        if level is not None:
            return level
    # No physical spacing. Stage 3 derives it from the Stage-2 homography, which is not
    # available here, so assume 40x and then take one level finer: over-estimating the
    # read is the safe direction, and on mouse_brain -- a 20x slide -- one finer is
    # exactly the level Stage 3 goes on to choose.
    level = choose_level_by_target_mpp(ASSUMED_BASE_MPP_40X, n_levels, target_mpp)
    if level is None:
        return 0
    return max(0, level - 1) if assume_finer_when_unknown else level


def estimate_stage3_read_gib(dapi_path, he_path, target_mpp=DEFAULT_TARGET_TILE_MPP):
    """GiB of image data Stage 3 will hold, or None if neither slide can be inspected."""
    from my_utils import inspect_image_pyramid

    total = 0.0
    seen = False
    for path, bytes_per_pixel in ((dapi_path, 2), (he_path, 3)):
        if not path:
            continue
        try:
            pyramid = inspect_image_pyramid(str(path))
            shapes = pyramid.get("level_shapes") or []
            level = _tile_level(pyramid, target_mpp, assume_finer_when_unknown=True)
            if level is None or level >= len(shapes):
                continue
            total += _level_bytes(shapes[level], bytes_per_pixel)
            seen = True
        except Exception:
            continue
    return (total / 2 ** 30) if seen else None


def estimate_stage3_peak_gib(dapi_path, he_path, target_mpp=DEFAULT_TARGET_TILE_MPP):
    """Estimated Stage-3 peak RSS in GiB for this pair of slides.

    Never returns None: a dataset whose pyramids cannot be read still needs a number
    for the admission policy, and refusing to produce one is what strands the queue.
    """
    read_gib = estimate_stage3_read_gib(dapi_path, he_path, target_mpp)
    if read_gib is None:
        return FALLBACK_GIB
    return round((BASELINE_GIB + PER_READ_GIB * read_gib) * SAFETY_MARGIN, 2)
