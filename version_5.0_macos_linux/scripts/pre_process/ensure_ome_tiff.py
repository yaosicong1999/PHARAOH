"""Bring a pipeline input to OME-TIFF before a run starts.

Stage 1 reads pyramid levels and `PhysicalSizeX/Y` through `tifffile`, so a plain
BTF/TIFF/NDPI/SVS/JPEG/CZI has to be converted first. Doing it here rather than inside the
pipeline keeps three properties:

* the conversion runs once per *file*, not once per run, so a re-run or a second
  dataset sharing the same H&E reuses the converted file instead of redoing it;
* nothing rewrites the caller's manifest -- the converted path is substituted in
  memory for this run only, and the JSON on disk keeps naming the original image;
* the cost stays out of the pipeline timings, because every caller finishes this
  step before starting its clock.

An existing `.ome.tif` beside the source is always reused, never overwritten -- it
may be a file someone converted deliberately, with settings this module does not
know about. When it is older than its source that is reported, not silently fixed.
"""

import time
from pathlib import Path

from image_conversion import (
    CZI_SUFFIXES,
    SUPPORTED_SUFFIXES,
    convert_image_to_ome_tif,
    czi_scene_indices,
    output_path,
)

OME_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff")


def is_ome_tiff(path) -> bool:
    """Whether `path` already names an OME-TIFF, by extension."""
    return str(path).lower().endswith(OME_TIFF_SUFFIXES)


def conversion_target(path) -> Path:
    """Where `ensure_ome_tiff` would write the converted copy of `path`."""
    return output_path(Path(path).expanduser().resolve())


def ensure_ome_tiff(path, *, log=print, cache=None, dry_run=False, label=None):
    """Return `(ome_tiff_path, record)` for one pipeline input.

    `record` is None when the input was already an OME-TIFF and nothing happened.
    Otherwise it describes what was done -- `action` is one of "converted",
    "reused", or "planned" (dry runs) -- and carries the wall time so a caller can
    report it separately from the pipeline it precedes.

    `cache` is an optional dict shared across inputs: a batch where several datasets
    name the same fixed H&E converts it once and reuses the result for the rest.
    """
    source = Path(path).expanduser().resolve()
    if is_ome_tiff(source):
        return source, None
    if cache is not None and source in cache:
        return cache[source], None

    prefix = f"[pre_process] {label}: " if label else "[pre_process] "
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"{source} is not an OME-TIFF and cannot be converted to one "
            f"automatically; supported inputs are: {supported}"
        )
    if not source.is_file():
        raise ValueError(f"image not found: {source}")
    if source.suffix.lower() in CZI_SUFFIXES:
        _split_czi_scenes(source, prefix=prefix, log=log, dry_run=dry_run)

    target = output_path(source)
    record = {
        "source": str(source),
        "output": str(target),
        "seconds": 0.0,
    }

    if target.exists():
        record["action"] = "reused"
        log(f"{prefix}reusing {target.name} beside {source.name}")
        if target.stat().st_mtime < source.stat().st_mtime:
            # Never overwrite: the existing file may have been produced deliberately,
            # with a tile size, compression or --mpp this module cannot infer.
            record["stale"] = True
            log(f"{prefix}WARNING: {target.name} is older than {source.name}; "
                f"delete it to reconvert")
    elif dry_run:
        record["action"] = "planned"
        log(f"{prefix}would convert {source.name} -> {target.name}")
    else:
        record["action"] = "converted"
        log(f"{prefix}converting {source.name} -> {target.name} "
            f"(written beside the source; the manifest is not modified)")
        started = time.perf_counter()
        try:
            produced = convert_image_to_ome_tif(source)
        except Exception as exc:
            # A corrupt source, an unreadable series, or a directory that cannot be
            # written all surface here. Name the file and the manual route rather than
            # letting a tifffile traceback stand in for the diagnosis.
            raise ValueError(
                f"could not convert {source} to {target.name}: "
                f"{type(exc).__name__}: {exc}. Convert it manually with "
                f"scripts/pre_process/convert_image_to_ome_tif.py and point the run "
                f"at the result"
            ) from exc
        record["seconds"] = round(time.perf_counter() - started, 6)
        target = Path(produced)
        record["output"] = str(target)
        log(f"{prefix}converted in {record['seconds']:.1f}s")

    if cache is not None:
        cache[source] = target
    return target, record


def _split_czi_scenes(source, *, prefix, log, dry_run):
    """Give every scene of a multi-scene CZI its own OME-TIFF, then ask which to use.

    The scenes are separate tissue sections, so no run can use the slide as one
    image; each scene is converted (or reused) beside the source, and the caller
    is told to name the wanted `NAME_sceneN.ome.tif` in place of the `.czi`.
    A single-scene file falls through to the ordinary conversion.
    """
    scenes = czi_scene_indices(source)
    if len(scenes) <= 1:
        return
    outputs = [output_path(source, scene=scene) for scene in scenes]
    log(f"{prefix}{source.name} holds {len(scenes)} scenes; each becomes its own OME-TIFF")
    for scene, target in zip(scenes, outputs):
        if target.exists():
            log(f"{prefix}reusing {target.name}")
        elif dry_run:
            log(f"{prefix}would convert scene {scene} -> {target.name}")
        else:
            log(f"{prefix}converting scene {scene} -> {target.name}")
            convert_image_to_ome_tif(source, scene=scene)
    names = ", ".join(target.name for target in outputs)
    raise ValueError(
        f"{source} holds {len(scenes)} scenes, one per tissue section, so it cannot "
        f"be used as one image; {'they would be' if dry_run else 'each has been'} "
        f"converted beside it ({names}). Name the scene this run should use in place "
        f"of the .czi"
    )
