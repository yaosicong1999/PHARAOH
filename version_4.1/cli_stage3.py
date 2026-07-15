# ============================================================
# cli_stage3.py - Stage 3 (headless): tile sampling + extraction
#
# The engine's Stage 3 compute lives in Tk-app methods
# (Stage3SamplingApp.on_sampling_clicked / .on_extract_clicked). Those
# methods do the real work but also poke a few Tk widgets at the end.
# We build the app object with __new__ (never calling Tk.__init__ -> no
# window / no display needed), populate the plain attributes __init__
# would set, stub the handful of widget touchpoints (a superset that
# covers both the dapi and he0 engines, which structure sampling
# slightly differently), replace the progress-dialog runner with a
# synchronous no-op, and call the real handlers. This reuses each
# engine's exact logic verbatim.
#
# Outputs: <run_dir>/tiles/{dapi|he0,he}_tile_info.json (+ tile PNGs),
#          <run_dir>/sampled_points.json, overlays.
#
# Usage:
#   python cli_stage3.py --engine <engine_dir> --run-dir <run_dir>
# ============================================================
import argparse
import os
import sys
import types
from pathlib import Path

from cli_common import load_module_from_path, log_stage_event, banner


class _NullQueue:
    """Stand-in for the Tk progress queue: swallow every progress message."""
    def put(self, *args, **kwargs):
        pass


class _DummyWidget:
    """Stand-in for Tk buttons/panels: accept any config/configure call."""
    def config(self, *args, **kwargs):
        pass
    configure = config


def _noop(*args, **kwargs):
    return None


def _sync_run_with_progress(self, title, worker_fn):
    print(f"[Stage3-CLI] {title}", flush=True)
    worker_fn(_NullQueue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    engine_dir = Path(args.engine).resolve()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))

    banner(f"STAGE 3 (headless) -> {run_dir}")

    m3 = load_module_from_path("engine_stage3", engine_dir / "3_get_tiles.py")
    # never pop up a Tk error dialog from a windowless run; surface real errors
    if hasattr(m3, "messagebox"):
        for _n in ("showerror", "showinfo", "showwarning"):
            setattr(m3.messagebox, _n, _noop)

    App = m3.Stage3SamplingApp

    obj = App.__new__(App)                       # no Tk.__init__, no window
    obj.run_dir = run_dir
    obj.stage3 = m3.load_stage3_params(Path(m3.__file__))
    obj.case_id = 0                              # CLI stage-1 uses identity orientation
    obj.sampling_counter = 0
    obj.has_sampling_outputs = False
    obj.current_tiles = None
    obj.current_points_xy = None

    # widget touchpoints reached by on_sampling_clicked (superset: dapi + he0)
    obj.panel_mid = None
    obj.panel_right = None
    obj.btn_pilot = _DummyWidget()
    obj.btn_extract = _DummyWidget()
    obj._update_sampling_gui = _noop            # dapi engine
    obj._set_panel_image = _noop                # he0 engine
    obj._run_with_progress = types.MethodType(_sync_run_with_progress, obj)

    print(f"[Stage3-CLI] stage3 params = {obj.stage3}", flush=True)
    log_stage_event(run_dir, "stage3_events", "cli_stage3_start")

    # ---- sampling (writes sampled_points.json, sets obj.current_tiles) ----
    App.on_sampling_clicked(obj)
    obj.has_sampling_outputs = True
    n_points = int(len(obj.current_points_xy)) if obj.current_points_xy is not None else -1
    print(f"[Stage3-CLI] sampling done: {n_points} tile centroids", flush=True)

    # ---- extraction (writes tiles/ + *_tile_info.json) ----
    App.on_extract_clicked(obj)

    tiles_dir = run_dir / "tiles"
    moving_ok = (tiles_dir / "dapi_tile_info.json").exists() or (tiles_dir / "he0_tile_info.json").exists()
    ok = moving_ok and (tiles_dir / "he_tile_info.json").exists()
    if not ok:
        raise RuntimeError(f"stage3 did not produce tile info json in {tiles_dir}")
    print(f"[Stage3-CLI] extraction done -> {tiles_dir}", flush=True)
    log_stage_event(run_dir, "stage3_events", "cli_stage3_done", n_points=n_points)


if __name__ == "__main__":
    main()
