# ============================================================
# cli_stage1.py - Stage 1 (headless): read images + build masks
#
# Replaces the interactive Stage-1 image selection with a non-interactive run
# that uses the shared lightweight image helpers, an H&E threshold of 240,
# a dtype-aware DAPI LUT threshold, and an identity orientation (case 0), then
# writes the exact same artifacts.
#
# Usage:
#   python cli_stage1.py --engine <engine_dir> --run-dir <run_dir>
#       --pair {dapi-he,he-he,dapi-dapi,he-dapi} --fixed <path> --moving <path>
#       [--he-threshold 240] [--moving-threshold N]
# ============================================================
import argparse
import json
import sys
import time
from pathlib import Path

from cli_common import save_json, log_stage_event, banner

HE_DEFAULT_THRESHOLD = 240      # matches he_val = tk.IntVar(value=240)
DAPI_LUT_DEFAULT_THRESHOLD = None  # resolved from dtype + pixels after image read
HERE = Path(__file__).resolve().parent
_PROCESS_STARTED = time.perf_counter()

# Stage 1 used to import OpenCV, NumPy, and Pillow before it could parse the
# command line or write its start event.  On macOS the first uncached import of
# the GUI-enabled OpenCV build can take many seconds.  Keep module startup
# lightweight and load only the compute dependencies after Stage 1 is visible.
cv2 = None
np = None
LUT = None


def _load_compute_dependencies():
    global cv2, np, LUT
    if cv2 is None:
        import cv2 as cv2_module
        import numpy as numpy_module

        cv2 = cv2_module
        np = numpy_module
        LUT = np.fromfile(
            HERE / "glasbey_inverted.lut", dtype=np.uint8
        ).reshape(256, 3)
    return cv2, np


def _write_rgb_png(path, image):
    """Save an RGB/RGBA/grayscale array through OpenCV without importing PIL."""
    if image.ndim == 3 and image.shape[-1] == 3:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[-1] == 4:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        encoded = image
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"could not write Stage 1 image: {path}")


def _channel_extras(selection):
    if not selection:
        return {}
    return {
        "channel_index": int(selection["index"]),
        "channel_name": str(selection.get("name") or ""),
        "channel_source": str(selection.get("source") or ""),
    }


def _record_roles(info, pair_name, moving_path, moving_level, moving_threshold,
                  fixed_path, fixed_level, fixed_threshold, *,
                  moving_channel=None, fixed_channel=None):
    """Add the `roles` block and the `pair` key to `images_info.json`, in place.

    Stage 1 is where a role binds to a modality, so it is the one place that has to
    state the binding; every later stage reads it back through `run_roles` rather than
    inferring the pair from which flat key happens to be present. 4.10 forced that
    inference: the headless Stage 1 never wrote a `mode` key, only the GUI did.

    Historical flat keys remain in place; optional channel metadata is additive.
    """
    import run_roles

    pair = run_roles.pair_from_name(pair_name)
    moving = run_roles.image_role(
        pair, run_roles.MOVING, moving_path, moving_level,
        threshold=moving_threshold, orientation_case=0,
        gui_affine=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        **_channel_extras(moving_channel),
    )
    fixed = run_roles.image_role(
        pair, run_roles.FIXED, fixed_path, fixed_level, threshold=fixed_threshold,
        **_channel_extras(fixed_channel),
    )
    info[run_roles.ROLES_KEY] = run_roles.roles_block(pair, moving, fixed)
    # The flat keys, spelled by token. For `dapi-he` these are the keys the caller
    # already wrote (`DAPI_path`, `HE_level`, ...) with the same values; for `he-he`
    # they are `HEFG_*` / `HEBG_*`, and nothing writes `HE0_*` any more.
    for key, value in run_roles.flat_keys(pair, moving, fixed).items():
        info.setdefault(key, value)
    return info


def _read_he_image(path, level, threshold):
    from my_utils import read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel

    image, level = read_image(path, channel="he", level=level)
    processed = enhance_hematoxylin_channel(extract_hematoxylin_channel(image))
    _, mask = cv2.threshold(processed, int(threshold), 255, cv2.THRESH_BINARY)
    return image, level, mask.astype(np.uint8)


def _read_dapi_image(path, level, threshold, channel_index=None):
    """Shared fluorescence reader for either role; never quantize detector input."""
    from my_utils import (
        read_image,
        dapi_to_lut_rgb,
        resolve_dapi_channel,
        resolve_dapi_lut_threshold,
    )

    try:
        selection = resolve_dapi_channel(path, requested=channel_index)
    except ValueError as exc:
        raise SystemExit(f"[cli_stage1] invalid DAPI channel for {path}: {exc}") from exc
    image, level = read_image(
        path, keep_16bit=True, force_rgb=False,
        channel=int(selection["index"]), level=level,
    )
    gray = image[..., 0] if image.ndim == 3 else image
    try:
        effective_threshold = resolve_dapi_lut_threshold(gray, threshold)
    except ValueError as exc:
        requested = "automatic" if threshold is None else str(threshold)
        raise SystemExit(
            f"[cli_stage1] invalid DAPI LUT threshold {requested} for {path}: {exc}"
        ) from exc
    return (
        gray,
        level,
        dapi_to_lut_rgb(gray, LUT, threshold=effective_threshold),
        effective_threshold,
        selection,
    )


def _write_dapi_images(run_dir, token, gray, lut):
    from my_utils import percentile_stretch_to_u8, qc_images_enabled

    _write_rgb_png(run_dir / f"1_{token}_level_image.png", gray)
    _write_rgb_png(run_dir / f"1_{token}_preview.png", percentile_stretch_to_u8(gray))
    if qc_images_enabled("stage1_extras"):
        cv2.imwrite(str(run_dir / f"1_{token}_lut.png"), lut)


def run_dapi_mode(run_dir, he_path, dapi_path, he_thr, dapi_thr, *,
                  he_level=None, moving_level=None, moving_channel=None):
    _load_compute_dependencies()
    from my_utils import (
        select_paired_image_levels,
        qc_images_enabled,
    )

    moving_level, he_level = select_paired_image_levels(
        dapi_path,
        he_path,
        requested_moving_level=moving_level,
        requested_fixed_level=he_level,
    )

    print("[Stage1-CLI] reading H&E ...", flush=True)
    he_orig, he_level, he_mask = _read_he_image(he_path, he_level, he_thr)

    print("[Stage1-CLI] reading DAPI ...", flush=True)
    dapi_gray, dapi_level, dapi_lut, dapi_thr, dapi_channel = _read_dapi_image(
        dapi_path, moving_level, dapi_thr, moving_channel
    )

    # ---- save artifacts (identical filenames to the GUI) ----
    _write_rgb_png(run_dir / "1_he_level_image.png", he_orig)
    # Raw intensities are the Stage-2 input; preview/LUT are visualization only.
    _write_dapi_images(run_dir, "dapi", dapi_gray, dapi_lut)
    # The level images above are Stage-2 inputs and are always written. These two are
    # for looking at.
    if qc_images_enabled("stage1_extras"):
        cv2.imwrite(str(run_dir / "1_he_threshold_mask.png"), he_mask)

    info = {
        "RUN_ID": run_dir.name.replace("runs_", "", 1),
        "HE_path": str(he_path),
        "HE_level": int(he_level),
        "DAPI_path": str(dapi_path),
        "DAPI_level": int(dapi_level),
        "DAPI_gui_affine": np.eye(3, dtype=np.float32).tolist(),
        "DAPI_orientation_case": 0,
        "HE_threshold": int(he_thr),
        "DAPI_LUT_threshold": int(dapi_thr),
    }
    _record_roles(info, "dapi-he",
                  moving_path=dapi_path, moving_level=dapi_level,
                  moving_threshold=dapi_thr, moving_channel=dapi_channel,
                  fixed_path=he_path, fixed_level=he_level,
                  fixed_threshold=he_thr)
    save_json(run_dir / "images_info.json", info)
    return info


def run_fixed_dapi_pair(run_dir, fixed_path, moving_path, fixed_thr, moving_thr, *,
                        fixed_level=None, moving_level=None, fixed_channel=None,
                        moving_channel=None, pair_name="dapi-dapi"):
    """Bind either moving modality to fixed DAPI, using the existing side readers."""
    _load_compute_dependencies()
    import run_roles
    from my_utils import select_paired_image_levels, qc_images_enabled

    pair = run_roles.PAIRS[pair_name]
    moving_level, fixed_level = select_paired_image_levels(
        moving_path, fixed_path,
        requested_moving_level=moving_level, requested_fixed_level=fixed_level,
    )
    levels = {}
    effective_thresholds = {
        run_roles.FIXED: fixed_thr,
        run_roles.MOVING: moving_thr,
    }
    requested_channels = {
        run_roles.FIXED: fixed_channel,
        run_roles.MOVING: moving_channel,
    }
    channel_selections = {}
    for role, path, level, threshold in (
        (run_roles.FIXED, fixed_path, fixed_level, fixed_thr),
        (run_roles.MOVING, moving_path, moving_level, moving_thr),
    ):
        token = pair.token(role)
        modality = pair.modality(role)
        print(f"[Stage1-CLI] reading {modality} ({token}, {role}) ...", flush=True)
        if modality == "dapi":
            (gray, levels[role], lut, effective_thresholds[role],
             channel_selections[role]) = _read_dapi_image(
                path, level, threshold, requested_channels[role]
            )
            _write_dapi_images(run_dir, token, gray, lut)
        else:
            image, levels[role], mask = _read_he_image(path, level, threshold)
            _write_rgb_png(run_dir / pair.stage1_image(role), image)
            if qc_images_enabled("stage1_extras"):
                cv2.imwrite(str(run_dir / f"1_{token}_threshold_mask.png"), mask)
    info = {"RUN_ID": run_dir.name.replace("runs_", "", 1)}
    _record_roles(info, pair.name,
                  moving_path=moving_path, moving_level=levels[run_roles.MOVING],
                  moving_threshold=effective_thresholds[run_roles.MOVING],
                  moving_channel=channel_selections.get(run_roles.MOVING),
                  fixed_path=fixed_path, fixed_level=levels[run_roles.FIXED],
                  fixed_threshold=effective_thresholds[run_roles.FIXED],
                  fixed_channel=channel_selections.get(run_roles.FIXED))
    save_json(run_dir / "images_info.json", info)
    return info


def run_he_mode(run_dir, he_path, moving_path, he_thr, moving_thr, *,
                he_level=None, moving_level=None):
    _load_compute_dependencies()
    from my_utils import (
        select_paired_image_levels,
        qc_images_enabled,
    )

    moving_level, he_level = select_paired_image_levels(
        moving_path,
        he_path,
        requested_moving_level=moving_level,
        requested_fixed_level=he_level,
    )

    import run_roles

    # Both sides are H&E, so both are spelled with fg/bg: `hefg` moves, `hebg` is fixed.
    pair = run_roles.PAIRS["he-he"]
    moving_tok, fixed_tok = pair.token(run_roles.MOVING), pair.token(run_roles.FIXED)

    print(f"[Stage1-CLI] reading H&E ({fixed_tok}, fixed) ...", flush=True)
    he_orig, he_level, he_mask = _read_he_image(he_path, he_level, he_thr)
    print(f"[Stage1-CLI] reading H&E ({moving_tok}, moving) ...", flush=True)
    moving_orig, moving_level_used, moving_mask = _read_he_image(moving_path, moving_level, moving_thr)

    _write_rgb_png(run_dir / pair.stage1_image(run_roles.FIXED), he_orig)
    _write_rgb_png(run_dir / pair.stage1_image(run_roles.MOVING), moving_orig)
    if qc_images_enabled("stage1_extras"):
        cv2.imwrite(str(run_dir / f"1_{fixed_tok}_threshold_mask.png"), he_mask)
        cv2.imwrite(str(run_dir / f"1_{moving_tok}_threshold_mask.png"), moving_mask)

    info = {"RUN_ID": run_dir.name.replace("runs_", "", 1)}
    _record_roles(info, "he-he",
                  moving_path=moving_path, moving_level=moving_level_used,
                  moving_threshold=moving_thr,
                  fixed_path=he_path, fixed_level=he_level,
                  fixed_threshold=he_thr)
    save_json(run_dir / "images_info.json", info)
    return info


def write_extras_only(run_dir):
    """Draw the `stage1_extras` images for a run Stage 1 has already read.

    A late `--qc-images stage1_extras` used to rerun Stage 1 outright, which re-chose
    the pyramid levels, reset both thresholds to their defaults and wrote orientation
    case 0 over whatever the fail-safe or the GUI had saved -- and every later stage,
    already on disk, was left standing on metadata that no longer matched it. This
    reads each image at the level and threshold `images_info.json` records and writes
    the LUT (DAPI) or threshold mask (H&E) per role, nothing else.
    """
    import run_roles
    import qc_artifacts

    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        raise SystemExit(f"--extras-only needs an existing {info_path}; run Stage 1 first")
    info = json.loads(info_path.read_text())
    before = info_path.read_bytes()
    roles = run_roles.read(info)
    defaults = {"dapi": DAPI_LUT_DEFAULT_THRESHOLD, "he": HE_DEFAULT_THRESHOLD}
    written = []
    for side in (roles.fixed, roles.moving):
        threshold = side.threshold if side.threshold is not None else defaults[side.modality]
        target = qc_artifacts.stage1_extra(run_dir, side)
        print(f"[Stage1-CLI] extras: reading {side.modality} ({side.token}, {side.role}) at "
              f"level {side.level}, threshold {threshold} ...", flush=True)
        if side.modality == "dapi":
            _, _, lut, _, _ = _read_dapi_image(
                side.path, int(side.level), threshold, side.channel_index
            )
            cv2.imwrite(str(target), lut)
        else:
            _, _, mask = _read_he_image(side.path, int(side.level), threshold)
            cv2.imwrite(str(target), mask)
        written.append(target.name)
    assert info_path.read_bytes() == before, "extras-only must not touch images_info.json"
    log_stage_event(run_dir, "stage1_events", "cli_stage1_extras_done", files=written)
    print(f"[Stage1-CLI] extras written: {', '.join(written)}; images_info.json untouched",
          flush=True)


LOSSY_SUFFIXES = (".jpg", ".jpeg")


def reject_lossy_input(role: str, path: str) -> None:
    """Refuse a lossy input at the point the pipeline first reads pixels.

    A manifest may name a JPEG -- `run_pharaoh_parallel.py` converts one to an OME-TIFF
    beside it before any dataset starts, and that is the supported route in. What must
    never happen is a stage reading the JPEG itself, because the conversion changes the
    container without recovering anything: the 8x8 block boundaries stay in the pixels,
    and Stage 2's ORB compares intensity orderings, which those boundaries perturb.

    Placed here rather than in the runner so it also holds for a direct `run_pharaoh.py`
    invocation, which does no conversion of its own.
    """
    source = Path(path)
    if source.suffix.lower() not in LOSSY_SUFFIXES:
        return
    converted = source.with_name(source.stem + ".ome.tif")
    hint = (f"convert it first, or name {converted.name} which already sits beside it"
            if converted.exists() else
            "run it through run_pharaoh_parallel.py, which converts inputs before the "
            "batch starts, or convert it with scripts/pre_process/convert_image_to_ome_tif.py")
    raise SystemExit(
        f"[cli_stage1] refusing to read {role} as {source.suffix}: {source}\n"
        f"    Stage 1 reads OME-TIFF only -- {hint}."
    )


def main():
    import run_roles
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    run_roles.add_pair_arguments(ap)
    ap.add_argument("--fixed", "--he", dest="he", required=True, help="fixed image path (DAPI or H&E)")
    ap.add_argument("--moving", required=True, help="moving image path (DAPI or H&E)")
    ap.add_argument("--fixed-threshold", "--he-threshold", dest="he_threshold", type=int, default=None)
    ap.add_argument("--moving-threshold", type=int, default=None)
    ap.add_argument("--fixed-level", "--he-level", dest="he_level", type=int, default=None,
                    help="explicit fixed-image pyramid level (0 is full resolution)")
    ap.add_argument("--moving-level", type=int, default=None,
                    help="explicit moving-image pyramid level (0 is full resolution)")
    ap.add_argument("--fixed-channel", "--fixed_channel", type=int, default=None,
                    help="zero-based fixed-image DAPI channel; valid only when the "
                         "fixed side of --pair is dapi")
    ap.add_argument("--moving-channel", "--moving_channel", type=int, default=None,
                    help="zero-based moving-image DAPI channel; valid only when the "
                         "moving side of --pair is dapi")
    ap.add_argument("--extras-only", action="store_true",
                    help="draw only the stage1_extras QC images (LUT / threshold mask) from "
                         "the settings images_info.json already records -- its levels, "
                         "thresholds and orientation -- and rewrite nothing else; for a run "
                         "that asks for QC after Stage 1 has run")
    args = ap.parse_args()
    pair = run_roles.pair_from_args(args)
    if args.he_level is not None and args.he_level < 0:
        ap.error("--he-level must be non-negative")
    if args.moving_level is not None and args.moving_level < 0:
        ap.error("--moving-level must be non-negative")
    if args.fixed_channel is not None and args.fixed_channel < 0:
        ap.error("--fixed-channel must be non-negative")
    if args.moving_channel is not None and args.moving_channel < 0:
        ap.error("--moving-channel must be non-negative")
    if args.fixed_channel is not None and pair.fixed != "dapi":
        ap.error(f"--fixed-channel is only valid for a fixed DAPI image; "
                 f"--pair {pair.name} has fixed {pair.fixed.upper()}")
    if args.moving_channel is not None and pair.moving != "dapi":
        ap.error(f"--moving-channel is only valid for a moving DAPI image; "
                 f"--pair {pair.name} has moving {pair.moving.upper()}")
    if args.he_threshold is not None and args.he_threshold < 0:
        ap.error("--fixed-threshold must be non-negative")
    if args.moving_threshold is not None and args.moving_threshold < 0:
        ap.error("--moving-threshold must be non-negative")

    reject_lossy_input("--he", args.he)
    reject_lossy_input("--moving", args.moving)

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load the mode-specific lightweight my_utils without importing the old
    # interactive Stage-1 modules.
    sys.path.insert(0, str(engine_dir))

    banner(f"STAGE 1 (headless, pair={pair.name}) -> {run_dir}")
    log_stage_event(
        run_dir,
        "stage1_events",
        "cli_stage1_start",
        pair=pair.name,
        requested_he_level=args.he_level,
        requested_moving_level=args.moving_level,
        requested_fixed_channel=args.fixed_channel,
        requested_moving_channel=args.moving_channel,
        process_startup_elapsed_seconds=time.perf_counter() - _PROCESS_STARTED,
    )

    dependency_started = time.perf_counter()
    _load_compute_dependencies()
    log_stage_event(
        run_dir,
        "stage1_events",
        "cli_stage1_dependencies_ready",
        elapsed_seconds=time.perf_counter() - dependency_started,
    )
    if args.extras_only:
        write_extras_only(run_dir)
        return

    defaults = {"dapi": DAPI_LUT_DEFAULT_THRESHOLD, "he": HE_DEFAULT_THRESHOLD}
    moving_thr = args.moving_threshold if args.moving_threshold is not None else defaults[pair.moving]
    fixed_thr = args.he_threshold if args.he_threshold is not None else defaults[pair.fixed]
    if pair.name == "dapi-he":
        info = run_dapi_mode(run_dir, args.he, args.moving,
                             fixed_thr, moving_thr,
                             he_level=args.he_level, moving_level=args.moving_level,
                             moving_channel=args.moving_channel)
    elif pair.fixed == "dapi":
        info = run_fixed_dapi_pair(run_dir, args.he, args.moving, fixed_thr, moving_thr,
                                   fixed_level=args.he_level, moving_level=args.moving_level,
                                   fixed_channel=args.fixed_channel,
                                   moving_channel=args.moving_channel,
                                   pair_name=pair.name)
    elif pair.name == "he-he":
        info = run_he_mode(run_dir, args.he, args.moving,
                           fixed_thr, moving_thr,
                           he_level=args.he_level, moving_level=args.moving_level)
    else:
        ap.error(f"Stage 1 does not yet support {pair.name}")

    import run_roles

    roles = run_roles.read(info)
    log_stage_event(
        run_dir, "stage1_events", "cli_stage1_done",
        he_level=int(roles.fixed.level), moving_level=int(roles.moving.level),
        fixed_channel=roles.fixed.channel_index,
        moving_channel=roles.moving.channel_index,
    )
    print("[Stage1-CLI] wrote images_info.json", flush=True)


if __name__ == "__main__":
    main()
