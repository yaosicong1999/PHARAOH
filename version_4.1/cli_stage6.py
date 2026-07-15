# ============================================================
# cli_stage6.py - Stage 6 (headless): build final-alignment outputs
#
# The engine's Stage 6 is a viewer, but its "Load alignment" handler
# (FinalAlignmentApp.load_alignment_and_keypoints) is what actually
# *writes* the output files: the final overlay, the manual-initial
# overlay, the alternating GIF, and the keypoint / base cache panels.
#
# We reuse that exact method via reflection: build the app object with
# __new__ (no Tk window), populate the plain attributes __init__ sets,
# stub the two methods that touch Tk widgets, call the engine's real
# base-image builders, then call load_alignment_and_keypoints.
#
# Outputs (both modes use the "6_" prefix, L = display level):
#   {p}_overlay_final_L{L}.png     final alignment overlay   (key output)
#   {p}_overlay_manual_L{L}.png    ORB initial-alignment overlay
#   {p}_alternating_L{L}.gif       final vs. initial toggle
#   {p}_cache_{mv}_base/kp, {p}_cache_he_base/kp
#
# Usage:
#   python cli_stage6.py --engine <dir> --run-dir <dir> --mode {dapi,he}
# ============================================================
import argparse
import os
import sys
from pathlib import Path

from cli_common import load_module_from_path, load_json, log_stage_event, banner


def _noop(*args, **kwargs):
    return None


def build_cache(run_dir, prefix, moving, level):
    return {
        f"{moving}_base": run_dir / f"{prefix}_cache_{moving}_base_L{level}.png",
        "he_base":       run_dir / f"{prefix}_cache_he_base_L{level}.png",
        f"{moving}_kp":  run_dir / f"{prefix}_cache_{moving}_kp_L{level}.png",
        "he_kp":         run_dir / f"{prefix}_cache_he_kp_L{level}.png",
        "overlay":       run_dir / f"{prefix}_overlay_final_L{level}.png",
        "manual":        run_dir / f"{prefix}_overlay_manual_L{level}.png",
        "alternating":   run_dir / f"{prefix}_alternating_L{level}.gif",
        "cells":         run_dir / f"{prefix}_cells_centroids_L{level}.png",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["dapi", "he"])
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    if args.mode == "dapi":
        prefix, moving = "6", "dapi"
        level_field, case_field = "DAPI_level", "DAPI_orientation_case"
    else:
        prefix, moving = "6", "he0"
        level_field, case_field = "HE0_level", "HE0_orientation_case"

    banner(f"STAGE 6 (headless final-alignment outputs, mode={args.mode})")
    log_stage_event(run_dir, "stage6_events", "cli_stage6_start", mode=args.mode)

    m6 = load_module_from_path("engine_stage6", engine_dir / "6_final_alignment.py")
    # Never pop up an error dialog on a non-widget object; surface the real error.
    if hasattr(m6, "messagebox"):
        m6.messagebox.showerror = _noop
        m6.messagebox.showinfo = _noop
        m6.messagebox.showwarning = _noop

    App = m6.FinalAlignmentApp
    info = load_json(run_dir / "images_info.json")
    level = int(info[level_field])

    obj = App.__new__(App)
    obj.run_dir = run_dir
    obj.info = info
    obj.case_id = int(info.get(case_field, 0))
    obj.display_level = level
    obj.display_scale = float(2 ** level)
    obj.cache = build_cache(run_dir, prefix, moving, level)

    # state attributes __init__ would set (superset covering both engines;
    # extras are harmless, and load_alignment_and_keypoints overwrites most)
    obj.alignment_loaded = False
    obj.H3 = None
    obj.warped_dapi = None
    obj.warped_he0 = None
    obj.he_dapi_overlay = None
    obj.he_he0_overlay = None
    obj.dapi_pts0 = None
    obj.he0_pts0 = None
    obj.he_pts0 = None
    obj.selected_indices = None
    obj.other_indices = None
    obj.cells_df = None
    obj.cells_pts_lvl2 = None
    obj.tps_inv = None
    obj._panel3_show_overlay = True

    # stub the only widget-touching methods reached by the compute path
    obj._make_tkimg = _noop
    obj.refresh_images_after_alignment = _noop

    # build base display images (engine's own methods, no widgets)
    if args.mode == "dapi":
        obj.dapi_rgb = App._load_or_build_dapi_base(obj)
        obj.he_rgb, obj.he16 = App._load_or_build_he_base(obj)
    else:
        obj.he0_rgb, obj.he0_img = App._load_or_build_he0_base(obj)
        obj.he_rgb, obj.he16 = App._load_or_build_he_base(obj)

    # the real writer: overlays, alternating gif, keypoint panels
    App.load_alignment_and_keypoints(obj)

    overlay = obj.cache["overlay"]
    if not overlay.exists():
        raise RuntimeError(f"stage6 did not produce {overlay}")
    print(f"[Stage6-CLI] wrote {overlay.name}, {obj.cache['manual'].name}, "
          f"{obj.cache['alternating'].name}", flush=True)
    log_stage_event(run_dir, "stage6_events", "cli_stage6_done",
                    overlay=str(overlay), display_level=level)


if __name__ == "__main__":
    main()
