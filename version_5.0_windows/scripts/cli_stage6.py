# ============================================================
# cli_stage6.py - Stage 6 (headless): build final-alignment outputs
#
# CLI and the Tk viewer use the same GUI-free AlignmentOutputs class.
# This entry never imports the viewer, tkinter, ImageTk, or cell-table readers,
# including when optional QC PNG/GIF outputs are enabled. --minimal keeps only
# the final overlay and avoids building the unused moving LUT display panel.
#
# Outputs (both modes use the "6_" prefix, L = display level):
#   {p}_overlay_final_L{L}.png     final alignment overlay   (key output)
#   {p}_overlay_manual_L{L}.png    ORB initial-alignment overlay
#   {p}_alternating_L{L}.gif       final vs. initial toggle
#   {p}_cache_{mv}_base/kp, {p}_cache_he_base/kp
#
# Everything but the final overlay is quality-control output, gated by the
# `stage6_overlays` group: off by default in parameters.json, and switchable per
# run with run_pharaoh.py's --qc-images / --no-qc-images (which reach this process
# through PHARAOH_QC_IMAGES). --minimal forces the same reduction locally.
#
# Usage:
#   python cli_stage6.py --engine <dir> --run-dir <dir> --pair {dapi-he,he-he} [--minimal]
# ============================================================
import argparse
import os
import sys
from pathlib import Path

from cli_common import log_stage_event, banner


def main():
    import run_roles
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    run_roles.add_pair_arguments(ap)
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="write only the final overlay (skip GUI cache panels, manual overlay, and GIF)",
    )
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    pair = run_roles.pair_from_args(args, run_dir=run_dir)
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    roles = run_roles.read(run_dir)
    banner(f"STAGE 6 (headless final-alignment outputs, pair={roles.pair.name})")
    log_stage_event(run_dir, "stage6_events", "cli_stage6_start", pair=pair.name)

    from stage6_outputs import AlignmentOutputs
    # `qc_images.stage6_overlays` off means the same thing --minimal already means: write
    # the final overlay and skip the Stage-2 comparison, the alternating GIF and the
    # cached keypoint panels. On one CRC slide those are ~97 MB of a 120 MB stage.
    try:
        from my_utils import qc_images_enabled
        overlays_wanted = qc_images_enabled("stage6_overlays")
    except Exception:
        overlays_wanted = True
    obj = AlignmentOutputs(
        run_dir,
        minimal_outputs=bool(args.minimal) or not overlays_wanted,
        lut_path=engine_dir / "glasbey_inverted.lut",
    )
    obj.prepare_base_images()
    obj.load_alignment_and_keypoints()

    overlay = obj.cache["overlay"]
    if not overlay.exists():
        raise RuntimeError(f"stage6 did not produce {overlay}")
    alternating = obj.cache["alternating"]
    # `minimal_outputs` is the effective setting: --minimal or qc_images.stage6_overlays
    # turned off. Asserting on args.minimal alone demanded a GIF that was never meant to
    # be written.
    minimal = bool(getattr(obj, "minimal_outputs", args.minimal))
    if not minimal and not alternating.exists():
        raise RuntimeError(f"stage6 did not produce {alternating}")
    if minimal:
        print(f"[Stage6-CLI] wrote {overlay.name} (minimal outputs)", flush=True)
    else:
        print(f"[Stage6-CLI] wrote {overlay.name}, {obj.cache['manual'].name}, "
              f"{obj.cache['alternating'].name}", flush=True)
    log_stage_event(
        run_dir, "stage6_events", "cli_stage6_done",
        overlay=str(overlay),
        # Log the effective setting, not args.minimal: with stage6_overlays off and
        # no --minimal, the old form recorded a GIF path for a GIF never written.
        alternating_gif=str(alternating) if not minimal else None,
        display_level=obj.display_level,
        minimal=minimal,
    )


if __name__ == "__main__":
    main()
