#!/usr/bin/env python3
# ============================================================
# run_pharaoh.py - Headless end-to-end pipeline (Stages 1-5)
#
# Fully automatic version of the version_4.0 / version_4.0_he GUI
# pipelines: no manual selection, no manual alignment. Stage 2 uses
# automatic ORB registration instead of the interactive overlay.
#
#   DAPI -> H&E mode:  python run_pharaoh.py --dapi <dapi> --he <he>
#   HE0  -> H&E mode:  python run_pharaoh.py --he0  <he0>  --he <he>
#
# Stages:
#   1  read images + build masks         (cli_stage1.py)
#   2  automatic ORB initial alignment   (cli_stage2_orb.py)
#   3  tile sampling + extraction        (cli_stage3.py)
#   4  nuclei masks/standout/patches     engine 4a -> 4b -> 4c
#   5  final alignment from nuclei pairs (cli_stage5.py)
#
# Each stage is skipped if its output already exists (use --force to
# recompute). Outputs land in <run-dir> (default: ./runs_<timestamp>).
# ============================================================
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd, cwd=None):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(f"[run_pharaoh] step failed (exit {proc.returncode}): {printable}")


def step(name, artifact: Path, force: bool, fn):
    if artifact.exists() and not force:
        print(f"\n### {name}: SKIP (found {artifact.name})", flush=True)
        return
    print(f"\n########## {name} ##########", flush=True)
    fn()
    if not artifact.exists():
        raise SystemExit(f"[run_pharaoh] {name} finished but {artifact} is missing")
    print(f"### {name}: OK ({artifact})", flush=True)


def step_glob(name, directory: Path, pattern: str, force: bool, fn):
    """Like step() but the output filename is level-dependent, matched by glob."""
    existing = sorted(directory.glob(pattern))
    if existing and not force:
        print(f"\n### {name}: SKIP (found {existing[0].name})", flush=True)
        return
    print(f"\n########## {name} ##########", flush=True)
    fn()
    produced = sorted(directory.glob(pattern))
    if not produced:
        raise SystemExit(f"[run_pharaoh] {name} finished but no {pattern} in {directory}")
    print(f"### {name}: OK ({produced[0]})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Headless PHAROAH pipeline (ORB auto-alignment).")
    ap.add_argument("--dapi", help="DAPI image path (DAPI -> H&E mode)")
    ap.add_argument("--hefg", "--he0", dest="hefg",
                    help="H&E-FG image path (H&E-FG -> H&E mode)")
    ap.add_argument("--he", required=True, help="fixed H&E image path")
    ap.add_argument("--run-dir", default=None,
                    help="output dir (default ./runs_<timestamp>)")
    ap.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    ap.add_argument("--stop-after", type=str, default=None,
                    choices=["1", "2", "3", "4", "5", "6"], help="stop after this stage")
    args = ap.parse_args()

    if bool(args.dapi) == bool(args.hefg):
        ap.error("provide exactly one of --dapi or --hefg")

    if args.dapi:
        mode, moving, engine = "dapi", args.dapi, (HERE / "engine_dapi").resolve()
        stage1_file = "1_read_dapi_he.py"
    else:
        mode, moving, engine = "he", args.hefg, (HERE / "engine_he").resolve()
        stage1_file = "1_read_he0_he.py"

    he = str(Path(args.he).resolve())
    moving = str(Path(moving).resolve())
    for p in (he, moving):
        if not Path(p).exists():
            ap.error(f"image not found: {p}")
    if not engine.exists():
        ap.error(f"engine dir not found: {engine}")

    # ---- validate the integrated parameters file (read directly by every
    #      stage; no per-engine copy is generated) ----
    top_params = (HERE / "parameters.json").resolve()
    if not top_params.exists():
        ap.error(f"integrated parameters file not found: {top_params}")
    block_key = "dapi" if mode == "dapi" else "he0fg"
    all_params = json.loads(top_params.read_text())
    if block_key not in all_params:
        ap.error(f"{top_params} has no '{block_key}' block")
    print(f"[run_pharaoh] parameters: {top_params} ['{block_key}']")

    # Run dirs live under version_4.1/ (not inside the engine folders).
    # The stage scripts read parameters.json from their own engine dir (stage 3
    # via its module dir; 4a/4b/4c/5 from the engine dir too), so the run
    # location is fully independent of where parameters are found.
    run_dir = Path(args.run_dir).resolve() if args.run_dir else \
        HERE / datetime.now().strftime("runs_%Y%m%d%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = run_dir / "tiles"

    print(f"[run_pharaoh] mode={mode}  engine={engine.name}  run_dir={run_dir}")
    print(f"[run_pharaoh] fixed  H&E : {he}")
    print(f"[run_pharaoh] moving img : {moving}")

    stop = args.stop_after

    step("Stage 1 (read + masks)", run_dir / "images_info.json", args.force,
         lambda: run([PY, HERE / "cli_stage1.py", "--engine", engine, "--run-dir", run_dir,
                      "--mode", mode, "--he", he, "--moving", moving,
                      "--stage1-file", stage1_file]))
    if stop == "1":
        return

    step("Stage 2 (ORB alignment)", run_dir / "manual_initial_alignment.json", args.force,
         lambda: run([PY, HERE / "cli_stage2_orb.py", "--engine", engine,
                      "--run-dir", run_dir, "--mode", mode]))
    if stop == "2":
        return

    step("Stage 3 (tiles)", tiles_dir / "he_tile_info.json", args.force,
         lambda: run([PY, HERE / "cli_stage3.py", "--engine", engine, "--run-dir", run_dir]))
    if stop == "3":
        return

    # ---- Stage 4: engine helper scripts, run from the engine dir with
    #      the tiles dir as argv[1] (exactly how the Stage-4 GUI drives them) ----
    step("Stage 4a (nuclei masks)", tiles_dir / "nuclei_mask_info.json", args.force,
         lambda: run([PY, "4a_generate_nuclei_masks.py", tiles_dir], cwd=engine))
    step("Stage 4b (standout nuclei)", tiles_dir / "standout_nuclei.json", args.force,
         lambda: run([PY, "4b_find_standout_nuclei.py", tiles_dir], cwd=engine))
    step("Stage 4c (nuclei patches)", run_dir / "nuclei_patches" / "nuclei_centroids_global.json",
         args.force,
         lambda: run([PY, "4c_get_nuclei_patches.py", tiles_dir], cwd=engine))
    if stop == "4":
        return

    moving_prefix = "dapi" if mode == "dapi" else "he0"
    step("Stage 5 (final alignment)", run_dir / f"{moving_prefix}_to_he_homography_level0.json",
         args.force,
         lambda: run([PY, HERE / "cli_stage5.py", "--engine", engine,
                      "--run-dir", run_dir, "--mode", mode]))
    if stop == "5":
        return

    # ---- Stage 6: build final-alignment overlays / cache panels ----
    overlay_prefix = "6"  # both dapi and he0 engines emit 6_* stage-6 outputs
    step_glob("Stage 6 (alignment outputs)", run_dir, f"{overlay_prefix}_overlay_final_L*.png",
              args.force,
              lambda: run([PY, HERE / "cli_stage6.py", "--engine", engine,
                           "--run-dir", run_dir, "--mode", mode]))

    print(f"\n[run_pharaoh] DONE. Final homography: "
          f"{run_dir / (moving_prefix + '_to_he_homography_level0.json')}", flush=True)


if __name__ == "__main__":
    main()
