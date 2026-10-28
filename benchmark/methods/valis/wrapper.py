#!/usr/bin/env python3
"""VALIS DAPI -> H&E registration wrapper.

Calls the authors' `valis` package (https://github.com/MathOnco/valis) through its
documented entry points -- `registration.Valis(...)`, `.register()`, `.register_micro()` --
and leaves its reader choice, detector, matcher and non-rigid registrar as shipped.

Two input modes
---------------
* native (main benchmark): VALIS reads the full pyramids itself and picks its own level
  from its own caps; the condition sets those caps with `--caps-scale N` at the shipped
  1024:512:2048 ratio (N=1 is VALIS as shipped). How the DAPI reaches VALIS is the one
  thing that separates the two benchmark methods (`--dapi-reader`):
    - tifffile (method valis_tifffile): tifffile decodes the Xenium DAPI -- its own channel
      only, as every other method sees it -- and it is re-encoded losslessly for VALIS.
    - bioformats (method valis_bioformats): VALIS's own BioFormatsSlideReader opens the
      original file. It decodes JPEG 2000 itself, but hands VALIS every channel of the
      multi-file OME; VALIS registers on DAPI yet builds its tissue mask from all of them.
  The H&E is opened with VALIS's own reader and compared pixel for pixel with tifffile; if
  it does not read correctly (tiny last levels declaring an oversized tile) it is
  re-encoded the same way. libvips cannot be given the Xenium DAPI directly: it either
  refuses its JPEG 2000 (`tiff2vips: tile size out of range`) or decodes it as zeros.
  A re-encode is a deflate OME-TIFF with every level and the bit depth kept, so the
  non-rigid and micro stages can still re-read finer levels. Re-encodes are cached per source file
  (`--input-cache`) and re-made once per series (`--cache-series`, normally by
  prepare_valis_inputs.py before the series' VALIS batches), so a series' conditions share
  one conversion; every run is charged the probe and its series' conversion time, so the
  charge does not depend on which condition happened to run first.
* render (level-matched ablation, `--pinned-input render`, the default with pinned levels):
  one pinned level of each image is rendered to an 8-bit single-page TIFF (fixed 0.1-99.9
  percentile stretch) and the caps are the rendered size, so VALIS registers exactly those
  levels. The render is charged.
* truncated (level-matched ablation, `--pinned-input truncated`): each image is re-encoded
  losslessly, at its bit depth, as a pyramid whose level 0 is its pinned level (the coarser
  levels follow), so VALIS reads nothing finer than the pinned level. VALIS holds both
  images at one size and never upsamples past level 0, so L is the longer side of the
  smaller pinned level. The stage's finest cap is L (rigid: max_image_dim_px;
  nonrigid: max_non_rigid_registration_dim_px; micro: micro_dim) and the others keep the
  shipped ratios, N = L / shipped finest cap (1024, 2048, 4096), floored at VALIS's
  shipped values and kept in order (`matched_caps`):
      image     = clip(1024 N, 1024, L)
      processed = clip( 512 N,  512, image)
      non-rigid = clip(2048 N, image, L)
      micro     = L
  The re-encode is charged.

The DAPI is not inverted, so VALIS processes it with its own fluorescence path
(ChannelGetter) and the H&E with its brightfield path (OD). Render mode writes the DAPI
single-channel, so it exists for valis_tifffile only.

Output
------
Rigid stage only: a DAPI->H&E matrix. Otherwise the native VALIS transform -- matrices,
the full native displacement fields and every shape argument -- exported without
resampling; evaluation replays VALIS's own `warp_xy_from_to` arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shutil
import sys
import time
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import contract  # noqa: E402
from bench.pyramid import level_sizes, read_level, reencode_pyramid  # noqa: E402

STAGES = ("rigid", "nonrigid", "micro")
PERCENTILE_LOW, PERCENTILE_HIGH = 0.1, 99.9
SHIPPED_CAPS = {"max_image_dim_px": 1024, "max_processed_image_dim_px": 512,
                "max_non_rigid_registration_dim_px": 2048}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dapi", required=True, type=pathlib.Path)
    p.add_argument("--he", required=True, type=pathlib.Path)
    p.add_argument("--outdir", required=True, type=pathlib.Path)
    p.add_argument("--dapi-mpp", required=True, type=float)
    p.add_argument("--he-mpp", required=True, type=float)
    p.add_argument("--stage", choices=STAGES, required=True,
                   help="rigid only; rigid + non-rigid; or both plus register_micro.")
    p.add_argument("--caps-scale", type=float,
                   help="Native mode: VALIS's three caps at N x the shipped values.")
    p.add_argument("--dapi-level", type=int, help="Pinned mode: pinned 0-based DAPI level.")
    p.add_argument("--he-level", type=int, help="Pinned mode: pinned 0-based H&E level.")
    p.add_argument("--pinned-input", choices=("render", "truncated"), default="render",
                   help="Pinned mode: 8-bit single-page renders of the pinned levels with every "
                        "cap at their size (render), or lossless pyramids truncated at the pinned "
                        "levels with the stage's caps from matched_caps (truncated).")
    p.add_argument("--dapi-reader", choices=("tifffile", "bioformats"), required=True,
                   help="tifffile: the DAPI decoded by tifffile (its own channel) and handed "
                        "to VALIS as a cached lossless re-encode. bioformats: the original "
                        "file opened by VALIS's own BioFormatsSlideReader (reader_dict), "
                        "every channel included; no re-encode.")
    p.add_argument("--input-cache", type=pathlib.Path,
                   help="Native mode: directory of re-encoded pyramids shared across runs.")
    p.add_argument("--cache-series",
                   help="Native mode with --input-cache: the series (e.g. iter2) whose own "
                        "conversion is charged; a series without one re-encodes once.")
    p.add_argument("--device", default="auto",
                   help="auto lets VALIS take a visible CUDA GPU; cpu hides it. VALIS has no "
                        "MPS path, so mps resolves to cpu.")
    p.add_argument("--keep-temporary", action="store_true")
    return p


# --------------------------------------------------------------------------------------
# Input adaptation
# --------------------------------------------------------------------------------------

def source_key(path: pathlib.Path) -> str:
    stat = path.resolve().stat()
    text = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def valis_can_read(path: pathlib.Path):
    """(True, None) only if VALIS's own reader returns the same pixels tifffile decodes.

    Opening is not enough: libvips reads a Xenium JPEG 2000 DAPI either not at all
    ("tile size out of range") or, for some files, as an all-zero image without raising.
    So one level (the smallest with a side >= 256 px) is read both ways and must match
    exactly; anything else is re-encoded.
    """
    import numpy as np
    from valis import slide_io
    try:
        sizes = level_sizes(path)
        level = max(i for i, (w, h) in enumerate(sizes) if max(w, h) >= 256)
        reader = slide_io.get_slide_reader(str(path))(str(path))
        if len(reader.metadata.slide_dimensions) != len(sizes):
            return False, "VALIS sees a different number of pyramid levels"
        seen = np.asarray(reader.slide2image(level=level))
        truth = read_level(path, level)
        if truth.ndim == 3 and truth.shape[0] in (1, 3, 4) and truth.shape[0] < truth.shape[-1]:
            truth = np.moveaxis(truth, 0, -1)
        if seen.size != truth.size or not np.array_equal(seen.reshape(truth.shape), truth):
            return False, f"VALIS's reader returns different pixels at level {level} (max {seen.max()} vs {truth.max()})"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"


def cache_conversions(record: pathlib.Path) -> dict:
    """{series: {"conversion_sec": ..., "converted_at": ...}} of a cache record. A record from
    before conversions were kept per series holds the one made before iter1, so it is iter1's."""
    if not record.exists():
        return {}
    stored = json.loads(record.read_text())
    if "conversions" in stored:
        return stored["conversions"]
    return {"iter1": {"conversion_sec": stored["conversion_sec"]}} if "conversion_sec" in stored else {}


def native_input(path: pathlib.Path, name: str, scratch: pathlib.Path, cache, probe: bool = True,
                 series: str | None = None) -> dict:
    """The file itself if VALIS reads it correctly; otherwise a full-pyramid re-encode.

    With a cache, the re-encode is made once per series (`series`, e.g. iter2): the first use
    in a series -- normally prepare_valis_inputs.py before the series' VALIS batches --
    decodes and re-encodes the source again, overwriting the cached file, and records that
    conversion time for the series. Every run of the series is then charged the probe it ran
    plus the series' conversion time, whichever run made it (`conversion_sec_from_cache`), so
    no condition is cheaper for having run after another and each series measures its own.
    probe=False re-encodes without asking VALIS's reader first (the tifffile DAPI).
    """
    probe_start = time.perf_counter()
    readable, error = valis_can_read(path) if probe else (False, "decoded by tifffile by design")
    probe_sec = time.perf_counter() - probe_start
    if readable:
        sizes = [list(s) for s in level_sizes(path)]
        return {"path": str(path.resolve()), "level": 0, "direct": True, "cached": False,
                "probe_sec": probe_sec, "conversion_sec": 0.0, "conversion_sec_from_cache": False,
                "width": int(sizes[0][0]), "height": int(sizes[0][1]),
                "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
                "available_levels": sizes}
    print(f"[valis] {path.name}: {error if not probe else 'not read correctly by VALIS reader (' + error + ')'}; "
          "using a lossless re-encode", flush=True)
    if cache is not None and series is None:
        raise SystemExit("a shared input cache needs the series its conversion time belongs to")
    folder = scratch if cache is None else cache / source_key(path)
    destination, record = folder / name, folder / (name + ".json")
    conversions = cache_conversions(record) if cache is not None else {}
    cached = cache is not None and destination.exists() and series in conversions
    if cached:
        sizes = json.loads(record.read_text())["sizes"]
        conversion_sec = float(conversions[series]["conversion_sec"])
    else:
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / (name + ".partial.ome.tiff")
        start = time.perf_counter()
        sizes = reencode_pyramid(path, partial)
        conversion_sec = time.perf_counter() - start
        partial.replace(destination)
        if cache is not None:
            conversions[series] = {"conversion_sec": conversion_sec,
                                   "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            temporary = folder / (name + ".json.tmp")
            temporary.write_text(json.dumps({"source": str(path.resolve()), "sizes": sizes,
                                             "conversions": conversions}) + "\n")
            temporary.replace(record)
    return {"path": str(destination), "level": 0, "direct": False, "cached": cached,
            "reader_error": error, "probe_sec": probe_sec, "conversion_sec": conversion_sec,
            "conversion_sec_from_cache": cached, "conversion_series": series,
            "width": int(sizes[0][0]), "height": int(sizes[0][1]),
            "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
            "available_levels": sizes}


def truncated_input(path: pathlib.Path, level: int, destination: pathlib.Path) -> dict:
    """A lossless pyramid whose level 0 is the pinned level, re-encoded for this run."""
    sizes = level_sizes(path)
    if not 0 <= level < len(sizes):
        raise SystemExit(f"{path}: level {level} out of range 0..{len(sizes) - 1}")
    start = time.perf_counter()
    truncated = reencode_pyramid(path, destination, start_level=level)
    conversion_sec = time.perf_counter() - start
    if list(truncated[0]) != list(sizes[level]):
        raise SystemExit(f"{path}: truncated level 0 is {truncated[0]}, expected {sizes[level]}")
    return {"path": str(destination), "level": level, "direct": False, "cached": False,
            "probe_sec": 0.0, "conversion_sec": conversion_sec, "conversion_sec_from_cache": False,
            "width": int(sizes[level][0]), "height": int(sizes[level][1]),
            "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
            "available_levels": [list(s) for s in truncated], "truncated_from_level": level}


MATCHED_FINEST = {"rigid": 1024, "nonrigid": 2048, "micro": 4096}   # shipped finest cap per stage


def matched_caps(side: int, stage: str) -> tuple[dict, float]:
    """Truncated mode: the stage's finest cap at `side`, the rest at the shipped ratios,
    floored at the shipped values and ordered processed <= image <= non-rigid <= micro."""
    n = side / MATCHED_FINEST[stage]
    image = min(side, max(SHIPPED_CAPS["max_image_dim_px"], SHIPPED_CAPS["max_image_dim_px"] * n))
    processed = min(image, max(SHIPPED_CAPS["max_processed_image_dim_px"],
                               SHIPPED_CAPS["max_processed_image_dim_px"] * n))
    non_rigid = min(side, max(image, SHIPPED_CAPS["max_non_rigid_registration_dim_px"] * n))
    return {"max_image_dim_px": int(round(image)), "max_processed_image_dim_px": int(round(processed)),
            "max_non_rigid_registration_dim_px": int(round(non_rigid)), "micro_dim": int(side)}, n


def render_input(path: pathlib.Path, level: int, destination: pathlib.Path, *, rgb: bool) -> dict:
    """One pinned level as an 8-bit single-page TIFF. RGB is VALIS's brightfield path,
    single-channel its fluorescence path."""
    import numpy as np
    import tifffile

    sizes = level_sizes(path)
    if not 0 <= level < len(sizes):
        raise SystemExit(f"{path}: level {level} out of range 0..{len(sizes) - 1}")
    array = read_level(path, level)
    if array.ndim == 2:
        array = array[:, :, None]
    original_dtype = str(array.dtype)
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)]
        low = float(np.percentile(finite, PERCENTILE_LOW))
        high = float(np.percentile(finite, PERCENTILE_HIGH))
        if high <= low:
            low, high = float(finite.min()), float(max(finite.max(), finite.min() + 1))
        array = (np.clip((array.astype(np.float32) - low) / (high - low), 0, 1) * 255).round().astype(np.uint8)
    if rgb:
        array = np.repeat(array, 3, axis=2) if array.shape[2] == 1 else array[:, :, :3]
        tifffile.imwrite(str(destination), array, photometric="rgb")
    else:
        tifffile.imwrite(str(destination), array[:, :, 0], photometric="minisblack")
    return {"path": str(destination), "level": level, "cached": False,
            "width": int(sizes[level][0]), "height": int(sizes[level][1]),
            "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
            "original_dtype": original_dtype, "available_levels": [list(s) for s in sizes]}


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------

@contextmanager
def audit_reader_calls(audit: dict, explicit_classes=()):
    """Record which file and level VALIS's readers open, without replacing them: the
    classes VALIS picks itself and any handed to it through reader_dict."""
    from valis import slide_io
    original_factory = slide_io.get_slide_reader
    restorations, patched = [], set()

    def observe(original, method):
        def call(reader, *args, **kwargs):
            # BioFormatsSlideReader.slide2vips fetches its tiles with slide2image on a thread
            # pool. Those inner calls are recorded as nested and not added to the totals.
            nested = getattr(reader, "_bench_in_slide2vips", 0) > 0
            if method == "slide2vips":
                reader._bench_in_slide2vips = getattr(reader, "_bench_in_slide2vips", 0) + 1
            start = time.perf_counter()
            try:
                result = original(reader, *args, **kwargs)
            finally:
                seconds = time.perf_counter() - start
                if method == "slide2vips":
                    reader._bench_in_slide2vips -= 1
            level = kwargs.get("level", args[0] if args else None)
            size = ([int(result.width), int(result.height)] if hasattr(result, "width")
                    else [int(result.shape[1]), int(result.shape[0])])
            audit["reads"].append({"stage": audit["stage"], "path": str(reader.src_f),
                                   "level_0based": None if level is None else int(level),
                                   "reader": type(reader).__name__, "api": method,
                                   "returned_size_wh": size, "seconds": seconds, "nested": nested})
            return result
        return call

    def patch(cls):
        if cls is not None and cls not in patched:
            patched.add(cls)
            for method in ("slide2vips", "slide2image"):
                if hasattr(cls, method):
                    restorations.append((cls, method, getattr(cls, method), method in cls.__dict__))
                    setattr(cls, method, observe(getattr(cls, method), method))

    def factory(*args, **kwargs):
        cls = original_factory(*args, **kwargs)
        patch(cls)
        return cls

    for cls in explicit_classes:
        patch(cls)
    slide_io.get_slide_reader = factory
    try:
        yield
    finally:
        slide_io.get_slide_reader = original_factory
        for cls, name, original, owned in reversed(restorations):
            setattr(cls, name, original) if owned else delattr(cls, name)


VALIS_STAGES = ("convert_imgs", "process_imgs", "rigid_register", "micro_rigid_register",
                "non_rigid_register", "measure_error", "cleanup")


@contextmanager
def time_valis_stages(cls, sink: dict):
    """Time the stages register() calls by patching them on the CLASS for the duration.

    Not on the instance: Valis.register() ends by pickling the registrar, and a function
    stored in the instance cannot be pickled -- register() then catches that error, kills
    the JVM and returns None, which breaks a later register_micro() that reads through
    Bio-Formats. Class attributes are not part of an instance's pickle.
    """
    originals = {}

    def timed(name, method):
        def call(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return method(self, *args, **kwargs)
            finally:
                sink[name] = sink.get(name, 0.0) + time.perf_counter() - start
        return call

    for name in VALIS_STAGES:
        if name in cls.__dict__:
            originals[name] = cls.__dict__[name]
            setattr(cls, name, timed(name, originals[name]))
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(cls, name, method)


def run_valis(registration, src_dir, inputs, results_dir, stage, caps, reader_dict=None):
    """Run VALIS with the roles stated, not inferred: moving = DAPI, reference = H&E.

    VALIS orders its two images twice, and both orders change the result:
    * Valis.__init__ sorts the image list by FULL PATH (valtils.sort_nicely). That order is
      slide_dict's, and preprocessing.collect_img_stats -- the joint intensity normalisation
      of the processed images -- treats the first image differently from the rest (masked
      pixels for the first, whole images for the others), so the normalisation target
      depends on which image comes first.
    * With imgs_ordered=True the rigid stage re-reads the processed images and orders the
      stack by SLIDE NAME (serial_rigid.get_image_files -> valtils.sort_nicely).
    Neither may follow what the input files are called or where they live. Both images are
    therefore handed over from this run's scratch directory under role names:
      <scratch>/moving_dapi/<file>   a symlink to the DAPI's folder (a multi-file OME needs
                                     its companion files beside it); slide name moving_dapi
      <scratch>/reference_he<ext>    a symlink to the H&E; slide name reference_he
    so both orders are [moving, reference] -- DAPI first, as in every earlier benchmark and
    CosMx run -- and the reference is found by its own file name. Both orders are checked
    with VALIS's own sort before anything runs.
    """
    import numpy as np
    from valis import valtils
    dapi_source, he_source = (pathlib.Path(p).resolve() for p in inputs)
    moving_dir = pathlib.Path(src_dir) / "moving_dapi"
    os.symlink(dapi_source.parent, moving_dir)
    dapi_path = str(moving_dir / dapi_source.name)
    # Only the image extension: VALIS derives the slide name by stripping it, so any other
    # dotted part of the source name (e.g. "he_image.repaired_level2_tile0.ome.tif") would
    # survive into the name and the reference would no longer be found.
    lower = he_source.name.lower()
    suffix = next((ext for ext in (".ome.tiff", ".ome.tif") if lower.endswith(ext)), he_source.suffix)
    he_path = str(pathlib.Path(src_dir) / ("reference_he" + suffix))
    os.symlink(he_source, he_path)
    names = {dapi_path: "moving_dapi", he_path: "reference_he"}
    path_order, name_order = list(names), list(names.values())
    valtils.sort_nicely(path_order)
    valtils.sort_nicely(name_order)
    if (path_order != [dapi_path, he_path] or name_order != ["moving_dapi", "reference_he"]
            or valtils.get_name(he_path) != "reference_he"):
        raise SystemExit(f"VALIS would not order the images as [moving, reference]: "
                         f"paths {path_order}, names {name_order}")
    if reader_dict:
        reader_dict = {dapi_path: info for info in reader_dict.values()}
    # Explicit img_list: never let VALIS discover inputs by scanning a directory, where an
    # external macOS volume's AppleDouble ._*.tiff files look like extra slides.
    kwargs = dict(src_dir=str(src_dir), dst_dir=str(results_dir),
                  name="valis", img_list=names, imgs_ordered=True,
                  reference_img_f=he_path, align_to_reference=True, do_rigid=True,
                  check_for_reflections=False,
                  **{k: caps[k] for k in SHIPPED_CAPS})
    if stage == "rigid":
        kwargs["non_rigid_registrar_cls"] = None
    start = time.perf_counter()
    registrar = registration.Valis(**kwargs)
    if stage == "rigid" and not hasattr(registrar, "non_rigid_reg_kwargs"):
        # Valis.cleanup() writes non_rigid_reg_kwargs unconditionally, but the attribute
        # exists only with a non-rigid registrar; without this the documented way of
        # switching that stage off dies in teardown after a successful rigid registration.
        registrar.non_rigid_reg_kwargs = {}
    timings = {"valis_build_sec": time.perf_counter() - start, "valis_stage_sec": {}}

    def shapes():
        return {name: {attr: (np.asarray(getattr(slide, attr)).tolist()
                              if getattr(slide, attr, None) is not None else None)
                       for attr in ("processed_img_shape_rc", "reg_img_shape_rc", "aligned_slide_shape_rc")}
                for name, slide in registrar.slide_dict.items()}

    audit = {"stage": "register", "reads": []}
    stage_shapes = {"roles": {"moving": {"name": "moving_dapi", "file": str(dapi_source),
                                         "handed_over_as": dapi_path},
                              "reference": {"name": "reference_he", "file": str(he_source),
                                            "handed_over_as": he_path}},
                    "image_list_order": [names[p] for p in path_order], "stack_order": name_order}
    explicit = [info[0] for info in (reader_dict or {}).values()]
    with audit_reader_calls(audit, explicit), time_valis_stages(type(registrar), timings["valis_stage_sec"]):
        start = time.perf_counter()
        rigid_result, _, error_df = registrar.register(reader_dict=reader_dict)
        timings["valis_register_sec"] = time.perf_counter() - start
        # Valis.register() catches its own exceptions, prints them, kills the JVM and returns
        # (None, None, None). Recorded, so a silent internal failure is visible in the record.
        stage_shapes["register_returned_none"] = rigid_result is None
        stage_shapes["after_main"] = shapes()
        stage_shapes["after_main_field_rc"] = getattr(registrar, "_full_displacement_shape_rc", None)
        if stage == "micro":
            audit["stage"] = "register_micro"
            start = time.perf_counter()
            registrar.register_micro(max_non_rigid_registration_dim_px=caps["micro_dim"])
            timings["valis_register_micro_sec"] = time.perf_counter() - start
            stage_shapes["after_micro"] = shapes()
            stage_shapes["after_micro_field_rc"] = getattr(registrar, "_full_displacement_shape_rc", None)

    # Reader time per modality and stage: wall time inside the readers' slide2vips/
    # slide2image calls. Bio-Formats decodes there; a pyvips image may be lazy and decode
    # later inside the stage that uses it, so for VipsSlideReader this is a lower bound.
    roles = {str(dapi_source): "dapi", str(he_source): "he"}
    reader_sec = {}
    for read in audit["reads"]:
        modality = roles.get(str(pathlib.Path(read["path"]).resolve()), "other")
        read["modality"] = modality
        if read["nested"]:
            continue
        per = reader_sec.setdefault(modality, {"reader": read["reader"], "total": 0.0})
        per[read["stage"]] = per.get(read["stage"], 0.0) + read["seconds"]
        per["total"] += read["seconds"]
    timings["reader_sec"] = reader_sec
    stage_shapes["reader_calls"] = audit["reads"]
    stage_shapes = json.loads(json.dumps(stage_shapes, default=lambda v: np.asarray(v).tolist()))
    return registrar, error_df, timings, stage_shapes


def slide_by_file(registrar, path: str):
    """The Slide VALIS built from `path`: by source file, else by the name VALIS derives
    from the filename (valtils.get_name strips every suffix)."""
    def stem(name):
        name = pathlib.Path(name)
        while name.suffix:
            name = name.with_suffix("")
        return name.name
    for key, slide in registrar.slide_dict.items():
        source = getattr(slide, "src_f", "")
        if source and pathlib.Path(source).resolve() == pathlib.Path(path).resolve():
            return slide
        if stem(key) == stem(path):
            return slide
    raise SystemExit(f"VALIS has no slide for {path}; have {sorted(registrar.slide_dict)}")


def remove_scratch(path: pathlib.Path) -> None:
    """rmtree that tolerates an AppleDouble sidecar vanishing together with its TIFF on an
    external macOS volume; every other error still propagates."""
    def onerror(_function, _name, exc_info):
        if not isinstance(exc_info[1], FileNotFoundError):
            raise exc_info[1]
    shutil.rmtree(path, onerror=onerror)


def verify(dapi_slide, he_slide) -> None:
    """VALIS reports its own failures as warnings and returns anyway; check what is used."""
    import numpy as np
    for label, slide in (("DAPI", dapi_slide), ("H&E", he_slide)):
        matrix = getattr(slide, "M", None)
        if matrix is None or not np.isfinite(np.asarray(matrix, dtype=float)).all():
            raise SystemExit(f"VALIS produced no usable transform for the {label} slide")


def rigid_level0(dapi_slide, he_slide, frames) -> tuple:
    """The rigid stage as one DAPI-level0 -> H&E-level0 homography, recovered exactly
    from VALIS's own point API (it is a composition of linear maps). Refuses a fit that
    does not replay the API to 1e-6 px."""
    import numpy as np
    d, h = frames["dapi"], frames["he"]
    xs, ys = np.meshgrid(np.linspace(0, d["width"] - 1, 3), np.linspace(0, d["height"] - 1, 3))
    src = np.column_stack([xs.ravel(), ys.ravel()])
    dst = np.asarray(dapi_slide.warp_xy_from_to(src, he_slide, non_rigid=False), dtype=float)

    def normalise(points):
        centre = points.mean(axis=0)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(points - centre, axis=1))
        return (points - centre) * scale, np.array([[scale, 0, -scale * centre[0]],
                                                    [0, scale, -scale * centre[1]], [0, 0, 1.]])
    s, ns = normalise(src)
    t, nt = normalise(dst)
    n = len(s)
    A, b = np.zeros((2 * n, 8)), np.zeros(2 * n)
    A[0::2, 0:3] = np.column_stack([s, np.ones(n)])
    A[0::2, 6], A[0::2, 7], b[0::2] = -s[:, 0] * t[:, 0], -s[:, 1] * t[:, 0], t[:, 0]
    A[1::2, 3:6] = np.column_stack([s, np.ones(n)])
    A[1::2, 6], A[1::2, 7], b[1::2] = -s[:, 0] * t[:, 1], -s[:, 1] * t[:, 1], t[:, 1]
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = np.linalg.inv(nt) @ np.append(solution, 1.0).reshape(3, 3) @ ns
    M /= M[2, 2]
    predicted = np.column_stack([src, np.ones(n)]) @ M.T
    residual = float(np.max(np.linalg.norm(predicted[:, :2] / predicted[:, 2:] - dst, axis=1)))
    if not np.isfinite(M).all() or residual > 1e-6:
        raise SystemExit("rigid matrix does not replay VALIS's point API to 1e-6 px")
    to_frame = np.diag([*(1 / np.asarray(contract.scale_xy(d))), 1.0])
    to_level0 = np.diag([*contract.scale_xy(h), 1.0])
    return to_level0 @ M @ to_frame, residual


def export_native(outdir: pathlib.Path, dapi_slide, he_slide, version: str) -> dict:
    """Matrices, full native displacement fields and shape arguments, unresampled."""
    import numpy as np
    import pyvips
    from valis import warp_tools
    native = outdir / "valis_native"
    native.mkdir(exist_ok=True)
    slides = {}
    for name, slide in (("dapi", dapi_slide), ("he", he_slide)):
        info = {}
        for attr in ("M", "processed_img_shape_rc", "reg_img_shape_rc",
                     "aligned_slide_shape_rc", "slide_dimensions_wh"):
            array = np.asarray(getattr(slide, attr))
            if not np.isfinite(array).all():
                raise SystemExit(f"nonfinite VALIS {name}.{attr}")
            info[attr] = array.tolist()
        for attr in ("fwd_dxdy", "bk_dxdy"):
            field = getattr(slide, attr)
            info[attr] = None
            if field is None:
                continue
            if isinstance(field, pyvips.Image):
                if field.bands != 2:
                    raise SystemExit(f"invalid VALIS {attr} layout")
                for row in range(0, field.height, 256):
                    block = field.extract_area(0, row, field.width, min(256, field.height - row))
                    if not np.isfinite(warp_tools.vips2numpy(block)).all():
                        raise SystemExit(f"nonfinite VALIS {attr}")
                relative = f"valis_native/{name}_{attr}.tiff"
                field.tiffsave(str(outdir / relative), tile=True, bigtiff=True, compression="deflate")
                info[attr] = {"path": relative, "representation": "pyvips",
                              "shape": [2, field.height, field.width]}
                continue
            field = np.asarray(field)
            if field.ndim != 3 or field.shape[0] != 2:
                raise SystemExit(f"invalid VALIS {attr} layout")
            for row in range(0, field.shape[1], 256):
                if not np.isfinite(field[:, row:row + 256]).all():
                    raise SystemExit(f"nonfinite VALIS {attr}")
            relative = f"valis_native/{name}_{attr}.npy"
            np.save(outdir / relative, field, allow_pickle=False)
            info[attr] = {"path": relative, "representation": "numpy",
                          "shape": list(field.shape), "dtype": str(field.dtype)}
        slides[name] = info
    if slides["dapi"]["fwd_dxdy"] is None:
        raise SystemExit("non-rigid result has no forward DAPI displacement")
    payload = {"valis_version": version, "slides": slides}
    (native / "transform.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    wrapper_start = time.perf_counter()
    pinned = args.dapi_level is not None and args.he_level is not None
    if pinned == (args.caps_scale is not None):
        raise SystemExit("pass exactly one of --caps-scale (native) or --dapi-level/--he-level (render)")
    if pinned and args.dapi_reader != "tifffile":
        raise SystemExit("render mode writes the DAPI single-channel: it is valis_tifffile only")
    for path in (args.dapi, args.he):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # VALIS's detectors pick their device with torch.cuda.is_available() and take no
    # argument, so CPU is forced by hiding the GPU before torch is imported.
    if args.device in ("cpu", "mps"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import_start = time.perf_counter()
    import numpy as np
    import torch
    from valis import registration
    version = importlib.metadata.version("valis-wsi")
    import_sec = time.perf_counter() - import_start
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scratch = args.outdir / "_valis_input"
    if scratch.exists():
        remove_scratch(scratch)
    scratch.mkdir(parents=True)
    render_start = time.perf_counter()
    matched_scale = side = None
    if pinned and args.pinned_input == "truncated":
        mode = "truncated"
        inputs = {"dapi": truncated_input(args.dapi, args.dapi_level, scratch / "0_dapi.ome.tiff"),
                  "he": truncated_input(args.he, args.he_level, scratch / "1_he.ome.tiff")}
        # VALIS holds both images at one size and never upsamples past a pyramid's level 0,
        # so the smaller pinned level bounds every stage: L is its longer side.
        side = min(max(r["width"], r["height"]) for r in inputs.values())
        caps, matched_scale = matched_caps(side, args.stage)
    elif pinned:
        mode = "render"
        inputs = {"dapi": render_input(args.dapi, args.dapi_level, scratch / "0_dapi.tiff", rgb=False),
                  "he": render_input(args.he, args.he_level, scratch / "1_he.tiff", rgb=True)}
        side = max(max(r["width"], r["height"]) for r in inputs.values())
        caps = {key: side for key in SHIPPED_CAPS}
        caps["micro_dim"] = side
    else:
        mode = "native"
        if args.input_cache:
            args.input_cache.mkdir(parents=True, exist_ok=True)
        if args.dapi_reader == "bioformats":
            sizes = [list(s) for s in level_sizes(args.dapi)]
            dapi_input = {"path": str(args.dapi.resolve()), "level": 0, "direct": True,
                          "reader": "BioFormatsSlideReader", "cached": False,
                          "probe_sec": 0.0, "conversion_sec": 0.0, "conversion_sec_from_cache": False,
                          "width": int(sizes[0][0]), "height": int(sizes[0][1]),
                          "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
                          "available_levels": sizes}
        else:
            dapi_input = native_input(args.dapi, "0_dapi.ome.tiff", scratch, args.input_cache, probe=False,
                                      series=args.cache_series)
        inputs = {"dapi": dapi_input,
                  "he": native_input(args.he, "1_he.ome.tiff", scratch, args.input_cache,
                                     series=args.cache_series)}
        caps = {key: int(round(value * args.caps_scale)) for key, value in SHIPPED_CAPS.items()}
        # register_micro ships at twice the main non-rigid cap; N=1 reproduces it.
        caps["micro_dim"] = 2 * caps["max_non_rigid_registration_dim_px"]
    render_sec = time.perf_counter() - render_start
    micro_headroom = args.stage == "micro" and caps["micro_dim"] > caps["max_non_rigid_registration_dim_px"]
    print(f"[valis] {version} mode={mode} stage={args.stage} device={device} caps={caps}", flush=True)

    reader_dict = None
    if mode == "native" and args.dapi_reader == "bioformats":
        from valis import slide_io
        reader_dict = {inputs["dapi"]["path"]: [slide_io.BioFormatsSlideReader, {}]}
    results_dir = args.outdir / "_valis_results"
    try:
        registrar, error_df, stage_timings, stage_shapes = run_valis(
            registration, scratch, [inputs["dapi"]["path"], inputs["he"]["path"]],
            results_dir, args.stage, caps, reader_dict)
        if len(registrar.slide_dict) != 2:
            raise SystemExit(f"VALIS expected two slides; got {sorted(registrar.slide_dict)}")
        dapi_slide = slide_by_file(registrar, inputs["dapi"]["path"])
        he_slide = slide_by_file(registrar, inputs["he"]["path"])
        verify(dapi_slide, he_slide)
        micro_effective = None
        before, after = stage_shapes.get("after_main_field_rc"), stage_shapes.get("after_micro_field_rc")
        if args.stage == "micro" and before is not None and after is not None:
            micro_effective = before != after  # did register_micro actually refine the field?

        frames = {name: contract.input_frame(path, r["level"], r["width"], r["height"],
                                             r["level0_width"], r["level0_height"], mpp,
                                             package_input=r["path"])
                  for name, r, path, mpp in (("dapi", inputs["dapi"], args.dapi, args.dapi_mpp),
                                             ("he", inputs["he"], args.he, args.he_mpp))}
        rigid_M, rigid_residual = rigid_level0(dapi_slide, he_slide, frames)
        export_start = time.perf_counter()
        if args.stage == "rigid":
            contract.write_matrix(args.outdir, "VALIS", frames, rigid_M,
                                  native={"valis_version": version, "stage": "rigid"})
        else:
            payload = export_native(args.outdir, dapi_slide, he_slide, version)
            contract.write(args.outdir, "VALIS", frames,
                           {"kind": "valis_native", "path": "valis_native/transform.json",
                            "stored_direction": "dapi_to_he", "requires_inverse": False,
                            "point_api": "valis.warp_tools.warp_xy_from_to"},
                           native={"valis_version": version, "stage": args.stage,
                                   "slides": payload["slides"]})
        export_sec = time.perf_counter() - export_start
        errors = json.loads(error_df.to_json(orient="records")) if error_df is not None else None
    finally:
        registration.kill_jvm()

    if not args.keep_temporary:
        shutil.rmtree(results_dir, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
    # Charged: preparing inputs VALIS can read (probe + re-encode, or the pinned-level
    # render), everything VALIS does, and emitting a transform another program can apply.
    # A re-encode reused from the cache is charged its recorded conversion time.
    if mode in ("native", "truncated"):
        preparation = {"input_probe_sec": sum(r["probe_sec"] for r in inputs.values()),
                       "input_conversion_sec": sum(r["conversion_sec"] for r in inputs.values())}
    else:
        preparation = {"input_render_sec": render_sec}
    # valis_stage_sec and reader_sec break valis_register_sec down; they are not added again.
    breakdown = {key: stage_timings.pop(key) for key in ("valis_stage_sec", "reader_sec")}
    charged = {**preparation, **stage_timings, "export_sec": export_sec}
    timings = {"import_sec": import_sec, "input_wall_sec": render_sec, **charged,
               "method_sec": sum(v for v in charged.values() if v), **breakdown,
               "conversion_sec_from_cache": any(r.get("conversion_sec_from_cache") for r in inputs.values()),
               "not_charged": ["import_sec"],
               "wrapper_total_sec": time.perf_counter() - wrapper_start}
    (args.outdir / "run_metadata.json").write_text(json.dumps({
        "method": f"valis_{args.dapi_reader}", "package": "valis-wsi", "package_version": version,
        "device": device, "device_requested": args.device,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "dapi": str(args.dapi.resolve()), "he": str(args.he.resolve()),
        "mode": mode, "dapi_reader": args.dapi_reader, "stage": args.stage,
        "caps_scale": args.caps_scale, "caps": caps,
        "pinned_levels": [args.dapi_level, args.he_level] if pinned else None,
        "pinned_input": args.pinned_input if pinned else None,
        "matched_side_px": side, "matched_scale": matched_scale,
        "micro_headroom": micro_headroom if args.stage == "micro" else None,
        "micro_effective": micro_effective,
        "inputs": inputs, "stage_shapes": stage_shapes,
        "rigid_level0": rigid_M.tolist(), "rigid_replay_residual_px": rigid_residual,
        "valis_error_df": errors, "timings": timings,
    }, indent=2))
    print(f"[valis] OK -> {args.outdir} | method {timings['method_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
