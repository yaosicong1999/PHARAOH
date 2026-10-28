#!/usr/bin/env python3
"""RegWSI (DeeperHistReg) DAPI -> H&E registration wrapper.

Calls the authors' DeeperHistReg package (https://github.com/MWod/DeeperHistReg), where
RegWSI -- the ACROBAT 2023 winner -- is released, with its shipped presets and its own
registration code. The algorithm is not modified.

Input adaptation
----------------
* DeeperHistReg's loaders count TIFF pages, so the SubIFD pyramid of an OME-TIFF is
  invisible to them. One pyramid level is therefore rendered to a single-page TIFF.
* Its preprocessing assumes 8-bit brightfield input. The 16-bit fluorescence DAPI is
  mapped by a fixed 0.1-99.9 percentile stretch, identical for every dataset, and then
  inverted so that the framework's own `1 - gray` leaves both images in one polarity.
* The level is chosen with the shared tier rule (bench.pyramid.select_level), or pinned
  explicitly for the level-matched ablation. No further resampling is applied.

Output
------
DeeperHistReg's dense displacement field is a *pull* field on the padded common (H&E)
canvas: H&E -> DAPI. It is stored untouched (`displacement_field.mha`) with its
post-processing parameters; evaluation solves it for DAPI -> H&E points numerically and
checks every root. Coordinate chain:

    frame  = level0 / level0_per_frame_px
    common = (frame * resample_ratio + pad) / initial_resample_ratio
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import contract  # noqa: E402
from bench.pyramid import TIER_MAX_SIDE, level_sizes, read_level, select_level  # noqa: E402

PRESETS = ("initial", "initial_nonrigid", "initial_nonrigid_high_resolution")
PERCENTILE_LOW, PERCENTILE_HIGH = 0.1, 99.9   # fixed; never tuned per sample
MPS_BUDGET_MARKER = "MPS_BUDGET_EXCEEDED"
EXIT_MPS_BUDGET = 87


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dapi", required=True, type=pathlib.Path)
    p.add_argument("--he", required=True, type=pathlib.Path)
    p.add_argument("--outdir", required=True, type=pathlib.Path)
    p.add_argument("--dapi-mpp", required=True, type=float, help="DAPI level-0 um/px")
    p.add_argument("--he-mpp", required=True, type=float, help="H&E level-0 um/px")
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--resolution", choices=tuple(TIER_MAX_SIDE),
                   help="Shared tier: largest pyramid level at or under this longest side.")
    p.add_argument("--dapi-level", type=int, help="Pinned 0-based DAPI level (ablation).")
    p.add_argument("--he-level", type=int, help="Pinned 0-based H&E level (ablation).")
    p.add_argument("--loader-resample-ratio", type=float, default=1.0,
                   help="loading_params.source/target_resample_ratio. The shipped presets use "
                        "0.2; the benchmark uses 1.0 so a tier is the size registered at.")
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda:0")
    p.add_argument("--memory-budget-gib", type=float, default=0.0,
                   help="Stop when MPS allocation crosses this. Unified-memory allocations "
                        "never reach RSS, so this has to be watched from inside. 0 disables.")
    p.add_argument("--keep-temporary", action="store_true")
    return p


# --------------------------------------------------------------------------------------
# Input adaptation
# --------------------------------------------------------------------------------------

def render_level(path, level: int, destination, *, invert: bool) -> dict:
    """One level as an 8-bit, 3-band, single-page TIFF."""
    import numpy as np
    import tifffile

    sizes = level_sizes(path)
    if not 0 <= level < len(sizes):
        raise SystemExit(f"{path}: level {level} out of range 0..{len(sizes) - 1}")
    array = read_level(path, level)
    if array.ndim == 2:
        array = array[:, :, None]
    original_dtype = str(array.dtype)
    stretch = None
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)]
        low = float(np.percentile(finite, PERCENTILE_LOW))
        high = float(np.percentile(finite, PERCENTILE_HIGH))
        if high <= low:
            low, high = float(finite.min()), float(max(finite.max(), finite.min() + 1))
        array = (np.clip((array.astype(np.float32) - low) / (high - low), 0, 1) * 255).round().astype(np.uint8)
        stretch = {"low": low, "high": high, "percentiles": [PERCENTILE_LOW, PERCENTILE_HIGH]}
    if invert:
        array = 255 - array
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]
    tifffile.imwrite(str(destination), array, photometric="rgb")
    return {"path": str(destination), "level": level,
            "width": int(sizes[level][0]), "height": int(sizes[level][1]),
            "level0_width": int(sizes[0][0]), "level0_height": int(sizes[0][1]),
            "original_dtype": original_dtype, "intensity_stretch": stretch,
            "inverted": invert, "available_levels": [list(s) for s in sizes]}


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------

class MpsPeakSampler(threading.Thread):
    """Polled peak of torch.mps.driver_allocated_memory; MPS has no peak of its own, and
    Metal buffers on unified memory barely move the process's RSS."""

    def __init__(self, limit_bytes: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.limit_bytes, self.interval, self.peak_bytes = limit_bytes, interval, 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        import torch
        while not self._stop_event.is_set():
            try:
                used = torch.mps.driver_allocated_memory()
            except Exception:
                return
            self.peak_bytes = max(self.peak_bytes, used)
            if self.limit_bytes and used > self.limit_bytes:
                # os._exit: the main thread is inside a torch op that checks no flag. The
                # runner classifies the stop from this log line.
                print(f"[regwsi] {MPS_BUDGET_MARKER}: {used / 2**30:.1f} GiB on mps against "
                      f"a budget of {self.limit_bytes / 2**30:.1f} GiB", flush=True)
                os._exit(EXIT_MPS_BUDGET)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)
        return self.peak_bytes / 2**20 if self.peak_bytes else None


def resolve_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_device(params: dict, device: str) -> dict:
    """The shipped presets hard-code cuda:0 in several nested places."""
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "device":
                    node[key] = device
                elif key == "cuda":
                    node[key] = device.startswith("cuda")
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(params)
    return params


def run_pipeline(deeperhistreg, params, source_path, target_path, save_path, device) -> dict:
    """Drive the stages directly: run_registration() swallows every exception into a
    print, and leaves the preprocessed tensors on the CPU when the transforms are not."""
    pipeline = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(params)
    pipeline.source_path, pipeline.target_path, pipeline.save_path = source_path, target_path, save_path
    timings = {}

    def timed(key, call):
        start = time.perf_counter()
        call()
        timings[key] = time.perf_counter() - start

    timed("load_sec", pipeline.load_images)
    timed("preprocessing_sec", pipeline.run_prepreprocessing)
    if device != "cpu":
        pipeline.pre_source = pipeline.pre_source.to(device)
        pipeline.pre_target = pipeline.pre_target.to(device)
        pipeline.current_displacement_field = pipeline.current_displacement_field.to(device)
    timed("initial_registration_sec", pipeline.run_initial_registration)
    timed("nonrigid_registration_sec", pipeline.run_nonrigid_registration)
    pipeline.total_registration_time = sum(timings.values())
    timed("save_sec", pipeline.save_final)
    return timings


def main() -> int:
    args = build_parser().parse_args()
    wrapper_start = time.perf_counter()
    pinned = args.dapi_level is not None and args.he_level is not None
    if pinned == (args.resolution is not None):
        raise SystemExit("pass exactly one of --resolution or (--dapi-level and --he-level)")
    for path in (args.dapi, args.he):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    import_start = time.perf_counter()
    import numpy as np
    import SimpleITK as sitk
    import torch
    import deeperhistreg
    import_sec = time.perf_counter() - import_start

    device = resolve_device(args.device)
    params = set_device(getattr(deeperhistreg.configs, f"default_{args.preset}")(), device)
    params["loading_params"]["source_resample_ratio"] = args.loader_resample_ratio
    params["loading_params"]["target_resample_ratio"] = args.loader_resample_ratio
    params.update(save_final_images=False, save_final_displacement_field=True, case_name="regwsi")

    render_start = time.perf_counter()
    rendered = args.outdir / "_rendered"
    rendered.mkdir(exist_ok=True)
    if pinned:
        dapi_level, he_level = args.dapi_level, args.he_level
    else:
        dapi_level = select_level(level_sizes(args.dapi), args.resolution)
        he_level = select_level(level_sizes(args.he), args.resolution)
    renders = {"dapi": render_level(args.dapi, dapi_level, rendered / "dapi.tiff", invert=True),
               "he": render_level(args.he, he_level, rendered / "he.tiff", invert=False)}
    render_sec = time.perf_counter() - render_start
    for name, r in renders.items():
        print(f"[regwsi] {name} level {r['level']}: {r['width']}x{r['height']} ({r['original_dtype']})",
              flush=True)
    print(f"[regwsi] preset={args.preset} device={device} loader_ratio={args.loader_resample_ratio} "
          f"initial_resolution={params['preprocessing_params']['initial_resolution']}", flush=True)

    temporary = args.outdir / "_temporary"
    temporary.mkdir(exist_ok=True)
    mps = MpsPeakSampler(int(args.memory_budget_gib * 2**30)) if device == "mps" else None
    if mps:
        mps.start()
    stage_timings = run_pipeline(deeperhistreg, params, renders["dapi"]["path"],
                                 renders["he"]["path"], str(temporary), device)
    if device.startswith("cuda"):
        peak_gpu_mib = torch.cuda.max_memory_allocated() / 2**20
    else:
        peak_gpu_mib = mps.stop() if mps else None

    results = temporary / "regwsi" / "Results_Final"
    if not (results / "displacement_field.mha").exists():
        raise SystemExit(f"DeeperHistReg produced no displacement field in {results}")
    for name in ("displacement_field.mha", "postprocessing_params.json", "registration_time.json"):
        if (results / name).exists():
            shutil.copy(results / name, args.outdir / name)
    post = json.loads((args.outdir / "postprocessing_params.json").read_text())
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(args.outdir / "displacement_field.mha")))
    if field.ndim != 3 or field.shape[0] != 2:
        raise SystemExit("invalid RegWSI displacement-field layout")
    for row in range(0, field.shape[1], 256):
        if not np.isfinite(field[:, row:row + 256]).all():
            raise SystemExit("nonfinite RegWSI displacement field")

    frames = {name: contract.input_frame(path, r["level"], r["width"], r["height"],
                                         r["level0_width"], r["level0_height"], mpp)
              for name, r, path, mpp in (("dapi", renders["dapi"], args.dapi, args.dapi_mpp),
                                         ("he", renders["he"], args.he, args.he_mpp))}
    contract.write(args.outdir, "RegWSI", frames,
                   {"kind": "regwsi_pull_field", "path": "displacement_field.mha",
                    "stored_direction": "he_to_dapi", "requires_inverse": True,
                    "postprocessing_path": "postprocessing_params.json",
                    "field_frame": "padded common canvas", "interpolation": "bilinear"},
                   native={"field_shape": list(field.shape), "postprocessing_params": post})

    if not args.keep_temporary:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(rendered, ignore_errors=True)
    # The render is charged: it is what makes the input readable to DeeperHistReg, just as
    # VoltRon's and PHARAOH's own image reading is part of their time.
    timings = {"import_sec": import_sec, "render_sec": render_sec, **stage_timings,
               "method_sec": render_sec + sum(stage_timings.values()),
               "not_charged": ["import_sec"],
               "wrapper_total_sec": time.perf_counter() - wrapper_start}
    (args.outdir / "run_metadata.json").write_text(json.dumps({
        "method": "regwsi", "package": "deeperhistreg",
        "package_version": getattr(deeperhistreg, "__version__", "1.0.1 (pypi)"),
        "torch_version": torch.__version__, "device": device, "peak_gpu_mib": peak_gpu_mib,
        "dapi": str(args.dapi.resolve()), "he": str(args.he.resolve()),
        "preset": args.preset, "resolution": args.resolution,
        "pinned_levels": [args.dapi_level, args.he_level] if pinned else None,
        "loader_resample_ratio": args.loader_resample_ratio,
        "renders": renders, "registration_parameters": params,
        "postprocessing_params": post, "timings": timings,
    }, indent=2))
    print(f"[regwsi] OK -> {args.outdir} | method {timings['method_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
