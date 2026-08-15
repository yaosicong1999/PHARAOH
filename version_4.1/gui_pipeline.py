# ============================================================
# gui_pipeline.py - Unified pipeline launcher (DAPI->H&E and H&E-FG->H&E)
#
# Same structure as the per-modality launchers, but:
#   - Stage 1 runs the UNIFIED loader (gui_read_images.py) where the user
#     chooses the moving modality (DAPI or H&E-FG) alongside the fixed H&E.
#   - Stages 2-6 are dispatched to engine_dapi/ or engine_he/ according
#     to the mode recorded in images_info.json.
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

PROJECT_ROOT = Path(__file__).resolve().parent
STAGE1_SCRIPT = "gui_read_images.py"          # unified, top-level
# stages 2-6 live in the per-mode engine folders (same filenames)
STAGE_SCRIPTS = {
    "2": "2_manual_alignment.py",
    "3": "3_get_tiles.py",
    "4": "4_tile_gallery.py",
    "5": "5_nucleus_patch_gallery.py",
    "6": "6_final_alignment.py",
}


def now_run_dir_name():
    return datetime.now().strftime("runs_%Y%m%d%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_mode(run_dir: Path):
    """Return 'dapi' or 'he0' from images_info.json, or None if not available."""
    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        return None
    try:
        info = json.load(open(info_path))
    except Exception:
        return None
    if info.get("mode") in ("dapi", "he0"):
        return info["mode"]
    if "DAPI_path" in info:
        return "dapi"
    if "HE0_path" in info:
        return "he0"
    return None


def engine_dir_for(mode: str) -> Path:
    return PROJECT_ROOT / ("engine_he" if mode == "he0" else "engine_dapi")


# ---- stage-completion checks (mode-agnostic) ----
def stage1_done(run_dir: Path) -> bool:
    return (run_dir / "images_info.json").exists()

def stage2_done(run_dir: Path) -> bool:
    return (run_dir / "manual_initial_alignment.json").exists() or \
           (run_dir / "clicked_blob_initial_alignment.json").exists()

def stage3_done(run_dir: Path) -> bool:
    tiles = run_dir / "tiles"
    if not tiles.is_dir():
        return False
    moving = (tiles / "dapi_tile_info.json").exists() or (tiles / "he0_tile_info.json").exists()
    return moving and (tiles / "he_tile_info.json").exists()

def stage4_done(run_dir: Path) -> bool:
    return (run_dir / "nuclei_patches" / "nuclei_centroids_global.json").exists()

def alignment_done(run_dir: Path) -> bool:
    return (run_dir / "dapi_to_he_homography_level0.json").exists() or \
           (run_dir / "he0_to_he_homography_level0.json").exists()


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
        self.setWindowTitle("Unified Pipeline (DAPI / H&E-FG → H&E) *** Stage 1-6")
        self.resize(880, 440)

        self.proc = None
        self.active_stage = None

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

        self.mode_label = QLabel("Mode: (choose DAPI or H&E-FG in Stage 1)")
        self.mode_label.setStyleSheet("font-weight: 600; color: #6d28d9;")

        # ---- stage buttons + statuses ----
        self.btn_stage1 = QPushButton("Run Stage 1: Image Selection (DAPI / H&&E-FG + H&&E)")
        self.btn_stage2 = QPushButton("Run Stage 2: Manual Alignment (+ auto ORB)")
        self.btn_stage3 = QPushButton("Run Stage 3: Tile Extraction")
        self.btn_stage4 = QPushButton("Run Stage 4: Nuclei Patch Extraction")
        self.btn_stage5 = QPushButton("Run Stage 5: Nuclei Gallery + Final Alignment Calc")
        self.btn_stage6 = QPushButton("Run Stage 6: Final Alignment Display")
        self.btn_stage1.clicked.connect(lambda: self.run_stage("1"))
        self.btn_stage2.clicked.connect(lambda: self.run_stage("2"))
        self.btn_stage3.clicked.connect(lambda: self.run_stage("3"))
        self.btn_stage4.clicked.connect(lambda: self.run_stage("4"))
        self.btn_stage5.clicked.connect(lambda: self.run_stage("5"))
        self.btn_stage6.clicked.connect(lambda: self.run_stage("6"))

        self.lbl_stage1_status = QLabel(status_text(False))
        self.lbl_stage2_status = QLabel(status_text(False))
        self.lbl_stage3_status = QLabel(status_text(False))
        self.lbl_stage4_status = QLabel(status_text(False))
        self.lbl_stage5_status = QLabel(status_text(False))
        self.lbl_stage6_status = QLabel(status_text(False))
        for lbl in (self.lbl_stage1_status, self.lbl_stage2_status, self.lbl_stage3_status,
                    self.lbl_stage4_status, self.lbl_stage5_status, self.lbl_stage6_status):
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
        grid.addWidget(header("Stage 4: extract nuclei patches"), 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 5: nuclei gallery + final alignment"), 6, 0, alignment=Qt.AlignLeft)
        grid.addWidget(header("Stage 6: view final alignment"), 6, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.btn_stage1, 1, 0)
        grid.addWidget(self.btn_stage2, 1, 1)
        grid.addWidget(self.btn_stage3, 4, 0)
        grid.addWidget(self.btn_stage4, 4, 1)
        grid.addWidget(self.btn_stage5, 7, 0)
        grid.addWidget(self.btn_stage6, 7, 1)

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
        d = QFileDialog.getExistingDirectory(self, "Choose RUN_DIR", str(PROJECT_ROOT))
        if d:
            self.run_dir_edit.setText(d)
            self.refresh_status()

    def on_new_dir(self):
        run_dir = PROJECT_ROOT / now_run_dir_name()
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

        mode = read_mode(run_dir) if run_dir else None
        if mode:
            self.mode_label.setText(f"Mode: {'H&E-FG → H&E' if mode == 'he0' else 'DAPI → H&E'}  "
                                    f"(engine_{'he' if mode == 'he0' else 'dapi'})")
        else:
            self.mode_label.setText("Mode: (choose DAPI or H&E-FG in Stage 1)")

        done1 = bool(run_dir) and stage1_done(run_dir)
        done2 = bool(run_dir) and stage2_done(run_dir)
        done3 = bool(run_dir) and stage3_done(run_dir)
        done4 = bool(run_dir) and stage4_done(run_dir)
        align_ok = bool(run_dir) and alignment_done(run_dir)

        for lbl, done in ((self.lbl_stage1_status, done1), (self.lbl_stage2_status, done2),
                          (self.lbl_stage3_status, done3), (self.lbl_stage4_status, done4)):
            lbl.setText(status_text(done))
            lbl.setStyleSheet(status_style(done))

        can_stage5 = bool(run_dir) and done4 and idle
        can_stage6 = bool(run_dir) and done4 and align_ok and idle
        self.lbl_stage5_status.setText(ready_text() if can_stage5 else "LOCKED 🔒")
        self.lbl_stage5_status.setStyleSheet(ready_style() if can_stage5 else status_style(False))
        if can_stage6:
            self.lbl_stage6_status.setText(ready_text())
            self.lbl_stage6_status.setStyleSheet(ready_style())
        elif bool(run_dir) and done4 and idle and not align_ok:
            self.lbl_stage6_status.setText("MISSING FINAL ALIGNMENT 🔒")
            self.lbl_stage6_status.setStyleSheet(status_style(False))
        else:
            self.lbl_stage6_status.setText("LOCKED 🔒")
            self.lbl_stage6_status.setStyleSheet(status_style(False))

        self.btn_stage1.setEnabled(bool(run_dir) and idle and (not done1))
        self.btn_stage2.setEnabled(bool(run_dir) and done1 and idle and (not done2))
        self.btn_stage3.setEnabled(bool(run_dir) and done2 and idle and (not done3))
        self.btn_stage4.setEnabled(bool(run_dir) and done3 and idle)
        self.btn_stage5.setEnabled(can_stage5)
        self.btn_stage6.setEnabled(can_stage6)

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
        else:
            mode = read_mode(run_dir)
            if mode is None:
                QMessageBox.warning(self, "No mode", "Run Stage 1 first (choose DAPI or H&E-FG).")
                return
            engine = engine_dir_for(mode)
            script_path = engine / STAGE_SCRIPTS[stage_id]
            work_dir = engine

        if not script_path.exists():
            QMessageBox.critical(self, "Script not found", f"Cannot find script:\n{script_path}")
            return

        self.active_stage = stage_id
        self.pre_stage2_exists = stage2_done(run_dir)

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments([str(script_path), str(run_dir)])
        self.proc.setWorkingDirectory(str(work_dir))
        self.proc.readyReadStandardOutput.connect(self._drain_stdout)
        self.proc.readyReadStandardError.connect(self._drain_stderr)
        self.proc.finished.connect(self._on_finished)

        for b in (self.btn_stage1, self.btn_stage2, self.btn_stage3,
                  self.btn_stage4, self.btn_stage5, self.btn_stage6):
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

        checks = {
            "1": (stage1_done, "images_info.json"),
            "2": (stage2_done, "alignment json"),
            "3": (stage3_done, "tiles/*_tile_info.json"),
            "4": (stage4_done, "nuclei_patches/nuclei_centroids_global.json"),
        }
        if stage in checks:
            fn, what = checks[stage]
            ok = bool(run_dir) and fn(run_dir)
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
