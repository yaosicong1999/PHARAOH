#!/usr/bin/env python3
"""Stage 3 without a window: run `stage3_core.Stage3Sampler` on a run directory.

The Tk window in `engine_dapi/3_get_tiles.py` runs the same sampler with panels and a
progress dialog around it; this entry point runs it as it is, so the CLI process
never imports a GUI toolkit. (Until 4.11 it instantiated the window class without
`Tk.__init__` and stubbed its widgets, which still loaded tkinter and ImageTk.)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import run_roles
from cli_common import load_module_from_path, log_stage_event, banner


def _orientation_case(run_dir: Path) -> int:
    """Return the orientation of the *computational* tile coordinate system.

    Both automatic Stage 2 and the manual GUI save ``H_mat`` in original moving-image
    coordinates.  The GUI composes its double-precision display transform with the
    original-to-display Stage-1 affine before saving.  Stage 1's orientation is
    therefore presentation metadata only once Stage 2 has saved the matrix.

    Keeping it in the Stage-3 rectifier made two equivalent matrices take different
    numerical paths: the H&E tile was rotated during ``warpPerspective`` and Stage 4b
    rotated the moving mask to match, so nearest-neighbour resize/downsampling could
    change the Dice search merely because the user chose a display orientation.  Tiles
    now use the same original-coordinate convention as the ORB/SIFT path: case 0.

    ``run_dir`` remains in the signature because tests and callers used this helper
    before the coordinate convention was made explicit.
    """
    del run_dir
    return 0


def make_headless_app(engine_dir, run_dir):
    """The Stage-3 worker for the CLI and the pilot: `stage3_core.Stage3Sampler`.

    Returns (module, sampler). The module is `stage3_core`, which imports no GUI
    toolkit; the sampler has `sample()` / `extract()` and, for callers written
    against the window, `on_sampling_clicked()` / `on_extract_clicked()`.
    """
    engine_dir, run_dir = Path(engine_dir), Path(run_dir)
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    core = load_module_from_path("stage3_core", engine_dir / "stage3_core.py")
    return core, core.Stage3Sampler(run_dir)


def redraw_debug_images(run_dir, roles, sample):
    """Draw the `stage3_debug` images again for a run whose tiles already exist.

    The images are by-products of the sampling, so `sample` (the seeded sampling on
    the same Stage-1 images) is repeated with the saved `sampled_points.json` guarded:
    it is put back as it was, and a run whose re-sampling differs fails rather than
    ship pictures of a different sampling -- and takes the pictures it just drew back
    with it, so the next resume does not find them and call the job done. The
    sampling writes the moving side's levels and the points; the extraction step later
    adds the fixed side's levels to the same file, so the check is that everything the
    sampling wrote matches what was saved.
    """
    import qc_artifacts

    sampled_path = Path(run_dir) / "sampled_points.json"
    if not sampled_path.is_file():
        raise SystemExit(f"--qc-only needs {sampled_path}; run Stage 3 first")
    saved = sampled_path.read_bytes()
    reproduced = False
    try:
        sample()
        resampled = json.loads(sampled_path.read_text())
        recorded = json.loads(saved)
        reproduced = all(recorded.get(key) == value for key, value in resampled.items())
    finally:
        sampled_path.write_bytes(saved)
        if not reproduced:
            for path in qc_artifacts.stage3_debug_expected(run_dir, roles):
                path.unlink(missing_ok=True)
    if not reproduced:
        raise SystemExit("[Stage3-CLI] --qc-only: re-sampling did not reproduce the saved "
                         "sampled_points.json (restored) and the debug images drawn were "
                         "removed; the tiles were not touched. Re-run Stage 3 with --force "
                         "to resample.")
    log_stage_event(run_dir, "stage3_events", "cli_stage3_qc_done")
    print("[Stage3-CLI] debug images re-drawn; tiles and sampled_points.json unchanged",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--qc-only", action="store_true",
                    help="re-draw the stage3_debug images by repeating the saved sampling; "
                         "the tiles and sampled_points.json are left exactly as they are, "
                         "and a sampling that does not reproduce the saved points is an error")
    args = ap.parse_args()
    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(engine_dir)
    banner(f"STAGE 3 ({'QC images only' if args.qc_only else 'headless'}) -> {run_dir}")
    m3, obj = make_headless_app(engine_dir, run_dir)

    print(f"[Stage3-CLI] stage3 params = {obj.stage3}", flush=True)
    display_case_id = obj.display_case_id
    print(f"[Stage3-CLI] tile orientation case = {obj.case_id} "
          f"(Stage-1 display case = {display_case_id})", flush=True)
    log_stage_event(run_dir, "stage3_events", "cli_stage3_start",
                    orientation_case=obj.case_id,
                    display_orientation_case=display_case_id)

    if args.qc_only:
        redraw_debug_images(run_dir, run_roles.read(run_dir), obj.sample)
        return

    # ---- sampling (writes sampled_points.json, sets obj.current_tiles) ----
    obj.sample()
    n_points = int(len(obj.current_points_xy)) if obj.current_points_xy is not None else -1
    print(f"[Stage3-CLI] sampling done: {n_points} tile centroids", flush=True)

    # ---- extraction (writes tiles/ + *_tile_info.json) ----
    print("[Stage3-CLI] Extracting tiles...", flush=True)
    obj.extract(m3.NullProgress())

    tiles_dir = run_dir / "tiles"
    roles = run_roles.read(run_dir)
    ok = all((tiles_dir / f"{side.token}_tile_info.json").exists()
             for side in (roles.moving, roles.fixed))
    if not ok:
        raise RuntimeError(f"stage3 did not produce tile info json in {tiles_dir}")
    print(f"[Stage3-CLI] extraction done -> {tiles_dir}", flush=True)
    log_stage_event(run_dir, "stage3_events", "cli_stage3_done", n_points=n_points)


if __name__ == "__main__":
    main()
