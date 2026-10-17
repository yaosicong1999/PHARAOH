# ============================================================
# gui_stage1_orient.py - Fail-safe Stage 1: orientation only
#
# The ordinary Stage-1 GUI (scripts/gui_read_images.py) starts with two file
# dialogs, because a person opening it has not chosen their images yet. A run
# that lands in manual_alignment_queue.json already has both images: they were
# read, levelled and thresholded by the headless pipeline, and only automatic
# Stage-2 registration failed. Asking for those two paths again is an invitation
# to hand-pick the wrong file into an existing RUN_DIR.
#
# So this variant is the same window with the selectors removed: both images are
# loaded from the command line, and the user is asked for the one thing the
# headless pipeline cannot decide -- the moving image's orientation -- plus the
# thresholds that set how much of each image is visible in Stage 2.
#
# The DAPI LUT threshold defaults to a value derived from the image's own
# intensity distribution (see dapi_lut_window.py) rather than the GUI's fixed
# 300 counts, which is wrong for exactly the skewed, high-background images that
# tend to fail automatic alignment in the first place.
#
# Usage:
#   python gui_stage1_orient.py --run-dir RUN_DIR --pair {dapi-he,he-he,dapi-dapi,he-dapi}
#       --fixed FIXED_PATH --moving MOVING_PATH [--name DATASET_NAME]
#
# Exit codes: 0 = saved, 3 = window closed without saving.
# ============================================================
import argparse
import sys
import tkinter as tk
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cv2  # noqa: E402  (after sys.path setup, like the other stage scripts)

from cli_common import log_stage_event  # noqa: E402
from gui_read_images import LUT, UnifiedStage1  # noqa: E402
import run_roles
from my_utils import (  # noqa: E402
    dapi_to_lut_rgb, percentile_stretch_to_u8, select_paired_image_levels,
)

from dapi_lut_window import auto_lut_window, clip_to_window  # noqa: E402

EXIT_SAVED = 0
EXIT_CANCELLED = 3


class FailSafeStage1(UnifiedStage1):
    """Stage 1 with both images supplied and only the orientation left to set."""

    def __init__(self, run_dir: Path, mode: str, he_path: str, moving_path: str,
                 name=None):
        self.preset_he_path = str(he_path)
        self.preset_moving_path = str(moving_path)
        self.dataset_name = name or run_dir.name
        self.saved = False
        self._pending_path = None
        self._lut_window = None
        self._paired_levels = None
        self._saved_roles = (
            run_roles.read(run_dir) if (run_dir / "images_info.json").exists() else None
        )
        super().__init__(run_dir, pair=run_roles.coerce_pair(mode))
        self.title(f"FAIL-SAFE STEP 1: orientation - {self.dataset_name}")
        # Load after the window is on screen: reading two pyramid levels takes
        # seconds, and a blank unresponsive window looks like a crash.
        self.after(50, self._autoload)

    # -----------------------------
    # UI: selectors out, a header in
    # -----------------------------
    def _build_ui(self):
        super()._build_ui()
        self.fixed_selectors.grid_remove()
        self.moving_selectors.grid_remove()

        self.header = tk.Label(self, justify="left", anchor="w")
        self.header.grid(row=0, column=0, columnspan=2, sticky="we", padx=8, pady=(8, 4))
        self._set_header("Loading images ...")

    def _set_header(self, status):
        moving_label = "DAPI" if self.forced_modality == "dapi" else "H&E-FG"
        self.header.config(
            text=(
                f"{self.dataset_name}\n"
                f"{moving_label} (moving):  {Path(self.preset_moving_path).name}\n"
                f"{self.fixed_modality.upper()} (fixed):  {Path(self.preset_he_path).name}\n"
                f"{status}"
            )
        )

    def _autoload(self):
        # Both paths are known before anything is read, so the joint level choice is
        # known too. Seeding the moving level keeps the H&E load from picking a level
        # on its own and then re-reading both images once the pair is settled.
        moving_level, he_level = self._paired_stage1_levels()
        self.moving_level = moving_level
        self._pending_path = self.preset_he_path
        if self.fixed_modality == "dapi":
            self.select_fixed_dapi()
        else:
            self.select_he()
        self._pending_path = self.preset_moving_path
        if self.forced_modality == "dapi":
            self.on_select_dapi()
        else:
            self.on_select_moving_he()
        self._pending_path = None
        self._set_header(
            f"auto levels: fixed {self.he_level} · moving {self.moving_level}   |   "
            "Rotate/flip the moving image until it matches the fixed image, then "
            "Confirm & Save to open manual alignment."
        )
        self._fit_window()

    def _paired_stage1_levels(self):
        """The jointly chosen Stage-1 levels, from the two paths given to us.

        The base class can only answer this once both images have been picked in the
        window; here they are known from the start, so the answer is available before
        the first read -- and it is the same answer the headless Stage 1 uses.
        """
        if self._paired_levels is None:
            self._paired_levels = select_paired_image_levels(
                self.preset_moving_path, self.preset_he_path
            )
        return self._paired_levels

    def _ask_image_path(self, title):
        return self._pending_path

    def _resolve_dapi_channel(self, path, role):
        # Preserve CLI channel overrides when handing an existing run to the GUI.
        if self._saved_roles is not None:
            side = getattr(self._saved_roles, role)
            if (Path(side.path).resolve() == Path(path).resolve()
                    and side.modality == "dapi" and side.channel_index is not None):
                return {
                    "index": side.channel_index, "name": side.extras.get("channel_name", ""),
                    "source": side.extras.get("channel_source", ""),
                }
        return super()._resolve_dapi_channel(path, role)

    # -----------------------------
    # DAPI display window
    # -----------------------------
    def _compute_lut_window(self):
        self._lut_window = auto_lut_window(self.dapi_img_view)
        return self._lut_window

    def _dapi_slider_spec(self):
        window = self._compute_lut_window()
        return (
            f"DAPI LUT Threshold (auto: {window.cutoff})",
            0,
            window.slider_max,
            window.slider_step,
            window.cutoff,
        )

    def _dapi_lut_image(self, threshold):
        window = self._lut_window or self._compute_lut_window()
        return dapi_to_lut_rgb(
            clip_to_window(self.dapi_img_view, window.high), LUT, threshold=int(threshold)
        )

    def _rebuild_moving_slider(self, label, lo, hi, step, default):
        super()._rebuild_moving_slider(label, lo, hi, step, default)
        if self.moving_modality != "dapi":
            return
        tk.Button(self.mv_slider_frame, text="Auto", command=self._reset_lut_threshold)\
            .pack(anchor="e", pady=(2, 0))

    def _reset_lut_threshold(self):
        window = self._compute_lut_window()
        self.mv_thr.set(window.cutoff)
        self.update_moving(window.cutoff)

    # -----------------------------
    # Save
    # -----------------------------
    def _on_saved(self, info):
        self.saved = True
        if self.moving_modality == "dapi":
            # Written by the headless Stage 1 as well: 1_dapi_level_image.png keeps
            # the raw intensities Stage 2 registers on and looks black in a viewer.
            cv2.imwrite(
                str(self.run_dir / f"1_{run_roles.read(info).moving.token}_preview.png"),
                percentile_stretch_to_u8(self.dapi_img_view),
            )
        event = {
            "pair": info["pair"],
            "dataset": self.dataset_name,
            "he_level": run_roles.read(info).fixed.level,
            "orientation_case": run_roles.read(info).moving.orientation_case,
        }
        if self._lut_window is not None:
            event["auto_lut_window"] = self._lut_window.as_event()
            event["lut_threshold_used"] = int(self.mv_thr.get())
        log_stage_event(
            self.run_dir, "stage1_events", "fail_safe_stage1_saved", **event
        )

    def _saved_message(self):
        return (
            f"Stage 1 saved (pair={run_roles.pair_for(self.moving_modality, self.fixed_modality).name}) to:\n{self.run_dir}\n\n"
            f"Stage 2 manual alignment opens next."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    run_roles.add_pair_arguments(parser)
    parser.add_argument("--fixed", "--he", dest="he", required=True, help="fixed image path")
    parser.add_argument("--moving", required=True, help="DAPI or H&E-FG image path")
    parser.add_argument("--name", default=None, help="dataset name, for the window title")
    args = parser.parse_args(argv)

    app = FailSafeStage1(
        Path(args.run_dir).resolve(),
        run_roles.pair_from_args(args).name,
        args.he,
        args.moving,
        name=args.name,
    )
    app.mainloop()
    # Reached only when the window is closed without confirming; a save exits
    # from inside the callback.
    return EXIT_SAVED if app.saved else EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main())
