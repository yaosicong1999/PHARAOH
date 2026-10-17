# ============================================================
# gui_pipeline.py - Unified pipeline launcher (DAPI->H&E and H&E-FG->H&E)
#
# Same structure as the per-modality launchers, but:
#   - Stage 1 runs the UNIFIED loader (gui_read_images.py) where the user
#     chooses DAPI or H&E independently for the moving and fixed roles.
#   - Stage 2 opens the interactive alignment window, as in 4.1. Its optional
#     in-window unchecked ORB/SIFT preview does not replace the user's Save action.
#   - Stages 3-6 are dispatched to engine_dapi/ (the one engine) according to the
#     mode recorded in images_info.json.
#   - A successful DAPI Stage 6 automatically launches the Stage-7 dense-pair
#     export, which then launches Stage-8 alignment refinement.
# Usage: python gui_pipeline.py
# ============================================================
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QLineEdit
)

from cli_common import log_stage_event
from run_artifacts import initial_alignment_path, stage3_complete
import run_roles
from patch_qc import patch_gallery_ready

PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_ROOT = VERSION_ROOT / "outputs"
STAGE1_SCRIPT = "gui_read_images.py"          # unified, top-level
# Stages 3-6 live in the per-mode engines. DAPI Stages 7-8 also live in
# engine_dapi and are reached through stable top-level compatibility CLIs.
STAGE_SCRIPTS = {
    "2": "2_manual_alignment.py",
    "3": "3_get_tiles.py",
    "4": "4_tile_gallery.py",
    "5": "5_nucleus_patch_gallery.py",
    "6": "6_final_alignment.py",
}


UNSELECTED_PAIR_TEXT = "Pair: select Moving (foreground) and Fixed (background) in Stage 1"
STAGE1_BUTTON_TEXT = "Run Stage 1: Select Moving / Fixed Images"
STAGE2_BUTTON_TEXT = "Run Stage 2: Manual Alignment"


def pair_description(run_dir):
    if run_dir is None or not (Path(run_dir) / "images_info.json").exists():
        return UNSELECTED_PAIR_TEXT
    try:
        pair = run_roles.read(run_dir).pair
    except (OSError, ValueError, KeyError, TypeError):
        return UNSELECTED_PAIR_TEXT
    labels = {"dapi": "DAPI", "he": "H&E"}
    return (f"Pair: {pair.name} | Moving (foreground): {labels[pair.moving]} "
            f"→ Fixed (background): {labels[pair.fixed]}")


def now_run_dir_name():
    return datetime.now().strftime("runs_%Y%m%d%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_pair(run_dir: Path):
    """The persisted pair controls dispatch and completion, never a GUI mode."""
    try:
        return run_roles.read(run_dir).pair.name
    except (OSError, ValueError, KeyError, TypeError):
        return None


def engine_dir_for(mode: str) -> Path:
    # One engine serves every pair since 4.11; `mode` only ever named which one.
    del mode
    return PROJECT_ROOT / "engine_dapi"


# ---- stage-completion checks (mode-agnostic) ----
def stage1_done(run_dir: Path) -> bool:
    return (run_dir / "images_info.json").exists()

def stage2_done(run_dir: Path) -> bool:
    return initial_alignment_path(run_dir).exists()

def stage3_done(run_dir: Path) -> bool:
    return stage3_complete(run_dir)

def stage4_done(run_dir: Path) -> bool:
    return (run_dir / "nuclei_patches" / "nuclei_centroids_global.json").exists()

def alignment_done(run_dir: Path) -> bool:
    try:
        import run_roles
        return (run_dir / f"{run_roles.read(run_dir).homography_stem}.json").exists()
    except (OSError, ValueError, KeyError, TypeError):
        return False

def stage6_done(run_dir: Path) -> bool:
    return any(run_dir.glob("6_overlay_final_L*.png"))

def stage6_output_signature(run_dir: Path):
    return {
        str(path.resolve()): path.stat().st_mtime_ns
        for path in run_dir.glob("6_overlay_final_L*.png")
    }

def stage6_output_updated(run_dir: Path, previous_signature) -> bool:
    current = stage6_output_signature(run_dir)
    return any(
        path not in previous_signature or previous_signature[path] != mtime
        for path, mtime in current.items()
    )

def dense_pairs_done(run_dir: Path) -> bool:
    return (run_dir / "dense_nucleus_pairs" / "summary.json").exists()

def stage8_done(run_dir: Path) -> bool:
    return (run_dir / "stage8_alignment_refinement.json").exists()

def should_auto_run_stage7(stage_id, exit_code, run_dir, mode) -> bool:
    return (
        stage_id == "6"
        and int(exit_code) == 0
        and bool(run_dir)
        and mode in ("dapi-he", "dapi")
        and stage6_done(run_dir)
    )


def status_text(done: bool) -> str:
    return "FINISHED ✅" if done else "NOT FINISHED ❌"

def status_style(done: bool) -> str:
    return "color: #15803d; font-weight: 600;" if done else "color: #b45309; font-weight: 600;"

def ready_text() -> str:
    return "READY ▶"

def ready_style() -> str:
    return "color: #1d4ed8; font-weight: 600;"


class PipelineWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PHARAOH 4.11 · Moving → Fixed · Stages 1–8")
        self.resize(940, 590)

        self.proc = None
        self.active_stage = None
        self.pre_stage6_outputs = {}

        # ---- RUN_DIR selector ----
        self.run_dir_edit = QLineEdit("")
        self.run_dir_edit.setPlaceholderText("Select or create a RUN_DIR (e.g. .../runs_202601201317)")
        self.btn_choose_dir = QPushButton("Choose RUN_DIR…")
        self.btn_new_dir = QPushButton("New RUN_DIR")
        self.btn_choose_dir.clicked.connect(self.on_choose_dir)
        self.btn_new_dir.clicked.connect(self.on_new_dir)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("RUN_DIR:"))
        top_row.addWidget(self.run_dir_edit, stretch=1)
        top_row.addWidget(self.btn_choose_dir)
        top_row.addWidget(self.btn_new_dir)

        self.mode_label = QLabel(UNSELECTED_PAIR_TEXT)
        self.mode_label.setStyleSheet("font-weight: 600; color: #6d28d9;")
        self.mode_label.setWordWrap(True)

        # ---- stage buttons + statuses ----
        self.btn_stage1 = QPushButton(STAGE1_BUTTON_TEXT)
        self.btn_stage2 = QPushButton(STAGE2_BUTTON_TEXT)
        self.btn_stage3 = QPushButton("Run Stage 3: Tile Extraction")
        self.btn_stage4 = QPushButton("Run Stage 4: Detect Nuclei (Patches Optional)")
        self.btn_stage5 = QPushButton("Run Stage 5: Keypoint QC (Optional)")
        self.btn_stage6 = QPushButton("Run Stage 6: Calculate Matrix + Display → Auto Stages 7–8")
        self.btn_stage7 = QPushButton("Run/Re-run Stage 7: Export Dense DAPI–H&E Nucleus Pairs")
        self.btn_stage8 = QPushButton("Run/Re-run Stage 8: Refine Alignment from Dense Pairs")
        self.btn_stage1.clicked.connect(lambda: self.run_stage("1"))
        self.btn_stage2.clicked.connect(lambda: self.run_stage("2"))
        self.btn_stage3.clicked.connect(lambda: self.run_stage("3"))
        self.btn_stage4.clicked.connect(lambda: self.run_stage("4"))
        self.btn_stage5.clicked.connect(lambda: self.run_stage("5"))
        self.btn_stage6.clicked.connect(lambda: self.run_stage("6"))
        self.btn_stage7.clicked.connect(lambda: self.run_stage("7"))
        self.btn_stage8.clicked.connect(lambda: self.run_stage("8"))

        self.lbl_stage1_status = QLabel(status_text(False))
        self.lbl_stage2_status = QLabel(status_text(False))
        self.lbl_stage3_status = QLabel(status_text(False))
        self.lbl_stage4_status = QLabel(status_text(False))
        self.lbl_stage5_status = QLabel(status_text(False))
        self.lbl_stage6_status = QLabel(status_text(False))
        self.lbl_stage7_status = QLabel(status_text(False))
        self.lbl_stage8_status = QLabel(status_text(False))
        for lbl in (self.lbl_stage1_status, self.lbl_stage2_status, self.lbl_stage3_status,
                    self.lbl_stage4_status, self.lbl_stage5_status, self.lbl_stage6_status,
                    self.lbl_stage7_status, self.lbl_stage8_status):
            lbl.setStyleSheet(status_style(False))

        def header(text):
            h = QLabel(text)
            h.setStyleSheet("font-size: 14px; font-weight: 700;")
            return h

        box = QGroupBox()
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)
        grid.addWidget(header("Stage 1: select images"), 0, 0, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 2: initial alignment"), 0, 1, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 3: extract tiles"), 3, 0, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 4: detect nuclei; optional patches"), 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 5: optional keypoint QC"), 6, 0, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 6: view final alignment"), 6, 1, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 7: dense nucleus pairs"), 9, 0, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 8: refine alignment"), 9, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.btn_stage1, 1, 0)
        grid.addWidget(self.btn_stage2, 1, 1)
        grid.addWidget(self.btn_stage3, 4, 0)
        grid.addWidget(self.btn_stage4, 4, 1)
        grid.addWidget(self.btn_stage5, 7, 0)
        grid.addWidget(self.btn_stage6, 7, 1)
        grid.addWidget(self.btn_stage7, 10, 0)
        grid.addWidget(self.btn_stage8, 10, 1)

        def status_row(lbl):
            row = QHBoxLayout()
            row.addWidget(QLabel("Status:"))
            row.addWidget(lbl)
            row.addStretch(1)
            w = QWidget(); w.setLayout(row)
            return w

        grid.addWidget(status_row(self.lbl_stage1_status), 2, 0)
        grid.addWidget(status_row(self.lbl_stage2_status), 2, 1)
        grid.addWidget(status_row(self.lbl_stage3_status), 5, 0)
        grid.addWidget(status_row(self.lbl_stage4_status), 5, 1)
        grid.addWidget(status_row(self.lbl_stage5_status), 8, 0)
        grid.addWidget(status_row(self.lbl_stage6_status), 8, 1)
        grid.addWidget(status_row(self.lbl_stage7_status), 11, 0)
        grid.addWidget(status_row(self.lbl_stage8_status), 11, 1)
        box.setLayout(grid)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.mode_label)
        layout.addWidget(box)
        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_status()

    # ---------------------
    # RUN_DIR controls
    # ---------------------
    def on_choose_dir(self):
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        d = QFileDialog.getExistingDirectory(
            self,
            "Choose RUN_DIR",
            str(DEFAULT_OUTPUT_ROOT),
        )
        if d:
            self.run_dir_edit.setText(d)
            self.refresh_status()

    def on_new_dir(self):
        run_dir = DEFAULT_OUTPUT_ROOT / now_run_dir_name()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir_edit.setText(str(run_dir))
        self.refresh_status()

    def get_run_dir(self):
        txt = self.run_dir_edit.text().strip()
        return Path(txt).resolve() if txt else None

    # ---------------------
    # Status refresh
    # ---------------------
    def refresh_status(self):
        run_dir = self.get_run_dir()
        idle = (self.proc is None)

        mode = read_pair(run_dir) if run_dir else None
        self.mode_label.setText(pair_description(run_dir))

        done1 = bool(run_dir) and stage1_done(run_dir)
        done2 = bool(run_dir) and stage2_done(run_dir)
        done3 = bool(run_dir) and stage3_done(run_dir)
        done4 = bool(run_dir) and stage4_done(run_dir)
        align_ok = bool(run_dir) and alignment_done(run_dir)
        done6 = bool(run_dir) and stage6_done(run_dir)
        done7 = bool(run_dir) and dense_pairs_done(run_dir)
        done8 = bool(run_dir) and stage8_done(run_dir)

        for lbl, done in ((self.lbl_stage1_status, done1), (self.lbl_stage2_status, done2),
                          (self.lbl_stage3_status, done3), (self.lbl_stage4_status, done4)):
            lbl.setText(status_text(done))
            lbl.setStyleSheet(status_style(done))

        can_stage5 = bool(run_dir) and done4 and patch_gallery_ready(run_dir) and idle
        can_stage6 = bool(run_dir) and done4 and idle
        can_stage7 = bool(run_dir) and mode == "dapi-he" and align_ok and done6 and idle
        can_stage8 = bool(run_dir) and mode == "dapi-he" and done7 and idle
        self.lbl_stage5_status.setText(ready_text() if can_stage5 else
                                      "OPTIONAL — GENERATE PATCHES IN STAGE 4" if done4 else "LOCKED 🔒")
        self.lbl_stage5_status.setStyleSheet(ready_style() if can_stage5 else status_style(False))
        if done6 and idle:
            self.lbl_stage6_status.setText(status_text(True))
            self.lbl_stage6_status.setStyleSheet(status_style(True))
        elif can_stage6:
            self.lbl_stage6_status.setText(ready_text())
            self.lbl_stage6_status.setStyleSheet(ready_style())
        elif bool(run_dir) and done4 and idle and not align_ok:
            self.lbl_stage6_status.setText("MISSING FINAL ALIGNMENT 🔒")
            self.lbl_stage6_status.setStyleSheet(status_style(False))
        else:
            self.lbl_stage6_status.setText("LOCKED 🔒")
            self.lbl_stage6_status.setStyleSheet(status_style(False))

        if done7:
            self.lbl_stage7_status.setText(status_text(True))
            self.lbl_stage7_status.setStyleSheet(status_style(True))
        elif mode == "he-he":
            self.lbl_stage7_status.setText("DAPI–H&E ONLY")
            self.lbl_stage7_status.setStyleSheet(status_style(False))
        elif can_stage7:
            self.lbl_stage7_status.setText(ready_text())
            self.lbl_stage7_status.setStyleSheet(ready_style())
        elif mode == "dapi-he" and idle and not done6:
            self.lbl_stage7_status.setText("WAITING FOR STAGE 6 🔒")
            self.lbl_stage7_status.setStyleSheet(status_style(False))
        else:
            self.lbl_stage7_status.setText("LOCKED 🔒")
            self.lbl_stage7_status.setStyleSheet(status_style(False))

        if done8:
            try:
                stage8_summary = json.loads(
                    (run_dir / "stage8_alignment_refinement.json").read_text()
                )
                promoted = bool(stage8_summary.get("promoted_to_canonical"))
                retained_stage = stage8_summary.get("canonical_stage_after", 5)
            except Exception:
                promoted = False
                retained_stage = 5
            self.lbl_stage8_status.setText(
                "REFINED + PROMOTED ✅"
                if promoted else f"CHECKED; STAGE {retained_stage} KEPT ✅"
            )
            self.lbl_stage8_status.setStyleSheet(status_style(True))
        elif mode == "he-he":
            self.lbl_stage8_status.setText("DAPI–H&E ONLY")
            self.lbl_stage8_status.setStyleSheet(status_style(False))
        elif can_stage8:
            self.lbl_stage8_status.setText(ready_text())
            self.lbl_stage8_status.setStyleSheet(ready_style())
        else:
            self.lbl_stage8_status.setText("WAITING FOR STAGE 7 🔒")
            self.lbl_stage8_status.setStyleSheet(status_style(False))

        self.btn_stage1.setEnabled(bool(run_dir) and idle and (not done1))
        self.btn_stage2.setEnabled(bool(run_dir) and done1 and idle and (not done2))
        self.btn_stage3.setEnabled(bool(run_dir) and done2 and idle and (not done3))
        self.btn_stage4.setEnabled(bool(run_dir) and done3 and idle)
        self.btn_stage5.setEnabled(can_stage5)
        self.btn_stage6.setEnabled(can_stage6)
        self.btn_stage7.setEnabled(can_stage7)
        self.btn_stage8.setEnabled(can_stage8)

    # ---------------------
    # Run stages
    # ---------------------
    def run_stage(self, stage_id):
        run_dir = self.get_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "No RUN_DIR", "Please choose or create a RUN_DIR first.")
            return
        ensure_dir(run_dir)
        if self.proc is not None:
            QMessageBox.information(self, "Busy", "A stage is already running.")
            return

        if stage_id == "1":
            script_path = PROJECT_ROOT / STAGE1_SCRIPT
            work_dir = PROJECT_ROOT
            process_args = [str(script_path), str(run_dir)]
        else:
            mode = read_pair(run_dir)
            if mode is None:
                QMessageBox.warning(self, "No pair", "Run Stage 1 first to select moving and fixed images.")
                return
            engine = engine_dir_for(mode)
            if stage_id == "6":
                script_path = PROJECT_ROOT / "gui_stage6.py"
                work_dir = PROJECT_ROOT
                process_args = [str(script_path), str(run_dir)]
            elif stage_id == "7":
                if mode != "dapi-he":
                    QMessageBox.warning(self, "DAPI only", "Stage 7 is only available for DAPI → H&E runs.")
                    return
                script_path = PROJECT_ROOT / "cli_stage7.py"
                work_dir = PROJECT_ROOT
                process_args = [str(script_path), "--run-dir", str(run_dir)]
            elif stage_id == "8":
                if mode != "dapi-he":
                    QMessageBox.warning(self, "DAPI only", "Stage 8 is only available for DAPI → H&E runs.")
                    return
                script_path = PROJECT_ROOT / "cli_stage8.py"
                work_dir = PROJECT_ROOT
                process_args = [str(script_path), "--run-dir", str(run_dir)]
            else:
                script_path = engine / STAGE_SCRIPTS[stage_id]
                work_dir = engine
                process_args = [str(script_path), str(run_dir)]

        if not script_path.exists():
            QMessageBox.critical(self, "Script not found", f"Cannot find script:\n{script_path}")
            return

        if stage_id == "1":
            log_stage_event(
                run_dir,
                "pipeline_events",
                "pipeline_start",
                source="gui_pipeline",
            )

        self.active_stage = stage_id
        self.pre_stage2_exists = stage2_done(run_dir)
        if stage_id == "6":
            self.pre_stage6_outputs = stage6_output_signature(run_dir)

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(process_args)
        self.proc.setWorkingDirectory(str(work_dir))
        self.proc.readyReadStandardOutput.connect(self._drain_stdout)
        self.proc.readyReadStandardError.connect(self._drain_stderr)
        self.proc.finished.connect(self._on_finished)

        for b in (self.btn_stage1, self.btn_stage2, self.btn_stage3,
                  self.btn_stage4, self.btn_stage5, self.btn_stage6, self.btn_stage7,
                  self.btn_stage8):
            b.setEnabled(False)
        self.proc.start()

    def _drain_stdout(self):
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data.strip():
            print(data, end="")

    def _drain_stderr(self):
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        if data.strip():
            print(data, end="", file=sys.stderr)

    def _on_finished(self, exitCode, exitStatus):
        stage = self.active_stage
        self.active_stage = None
        self.proc = None
        self.refresh_status()
        run_dir = self.get_run_dir()

        if stage == "6":
            mode = read_pair(run_dir) if run_dir else None
            if exitCode != 0:
                QMessageBox.critical(self, "Stage Failed", f"Stage 6 failed (exit code = {exitCode}).")
                return
            if (
                not run_dir
                or not stage6_done(run_dir)
                or not stage6_output_updated(run_dir, self.pre_stage6_outputs)
            ):
                QMessageBox.warning(
                    self,
                    "Stage 6 Not Saved",
                    "Stage 6 ended without creating or updating 6_overlay_final_L*.png.",
                )
                return
            if should_auto_run_stage7(stage, exitCode, run_dir, mode):
                print("[GUI] Stage 6 complete; automatically starting Stage 7.", flush=True)
                self.lbl_stage7_status.setText("STARTING AUTOMATICALLY ▶")
                self.lbl_stage7_status.setStyleSheet(ready_style())
                self.run_stage("7")
            else:
                QMessageBox.information(
                    self,
                    "Stage 6 Saved",
                    "Stage 6 output detected. Stage 7 is DAPI–H&E only.",
                )
            return

        if stage == "7":
            ok = exitCode == 0 and bool(run_dir) and dense_pairs_done(run_dir)
            if not ok:
                QMessageBox.warning(
                    self, "Stage 7 Not Saved",
                    "Stage 7 ended without producing dense_nucleus_pairs/summary.json.",
                )
                return
            print("[GUI] Stage 7 complete; automatically starting Stage 8.", flush=True)
            self.lbl_stage8_status.setText("STARTING AUTOMATICALLY ▶")
            self.lbl_stage8_status.setStyleSheet(ready_style())
            self.run_stage("8")
            return

        if stage == "8":
            ok = exitCode == 0 and bool(run_dir) and stage8_done(run_dir)
            if not ok:
                QMessageBox.warning(
                    self, "Stage 8 Not Saved",
                    "Stage 8 ended without producing stage8_alignment_refinement.json.",
                )
                return
            try:
                summary = json.loads(
                    (run_dir / "stage8_alignment_refinement.json").read_text()
                )
            except Exception:
                summary = {}
            if summary.get("promoted_to_canonical"):
                QMessageBox.information(
                    self, "Stage 8 Promoted",
                    "The dense-pair refinement passed all quality gates and is now the final alignment.",
                )
            else:
                kept_stage = summary.get("canonical_stage_after", 5)
                QMessageBox.information(
                    self, "Stage 8 Checked",
                    f"The candidate did not pass every quality gate, so the existing Stage-{kept_stage} alignment was kept.",
                )
            return

        checks = {
            "1": (stage1_done, "images_info.json"),
            "2": (stage2_done, "initial_alignment.json"),
            "3": (stage3_done, "tiles/*_tile_info.json"),
            "4": (stage4_done, "nuclei_patches/nuclei_centroids_global.json"),
        }
        if stage in checks:
            fn, what = checks[stage]
            ok = exitCode == 0 and bool(run_dir) and fn(run_dir)
            if ok:
                QMessageBox.information(self, f"Stage {stage} Saved", f"Stage {stage} outputs detected ({what}).")
            else:
                QMessageBox.warning(self, f"Stage {stage} Not Saved",
                                    f"Stage {stage} ended without producing {what}.")
            return
        if exitCode != 0:
            QMessageBox.critical(self, "Stage Failed", f"Stage {stage} failed (exit code = {exitCode}).")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PipelineWindow()
    w.show()
    sys.exit(app.exec_())
