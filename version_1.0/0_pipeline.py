import os
import sys
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QLineEdit
)

# =========================
# CONFIG: change to your filenames
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
STEP1_SCRIPT = "1_read_dapi_he.py"
STEP2A_SCRIPT = "2a_blob_matching.py"
STEP2B_SCRIPT = "2b_manual_alignment.py"
STEP3_SCRIPT = "3_get_tiles.py"
STEP4_SCRIPT = "4_tile_gallery.py"
STEP5_SCRIPT = "8_nucleus_patch_gallery.py"
STEP6_SCRIPT = "9_final_alignment.py"
# =========================
# Helpers
# =========================
def now_run_dir_name():
    return datetime.now().strftime("runs_%Y%m%d%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def step1_done(run_dir: Path) -> bool:
    return (run_dir / "images_info.json").exists()

def step2_done(run_dir: Path) -> bool:
    return (run_dir / "clicked_blob_initial_alignment.json").exists() or (run_dir / "manual_initial_alignment.json").exists()

def step3_done(run_dir: Path) -> bool:
    tiles_dir = run_dir / "tiles"
    if not tiles_dir.is_dir():
        return False
    dapi_info = tiles_dir / "dapi_tile_info.json"
    he_info   = tiles_dir / "he_tile_info.json"
    return dapi_info.exists() and he_info.exists()

def step4_done(run_dir: Path) -> bool:
    nuclei_dir = run_dir / "nuclei_patches"
    if not nuclei_dir.is_dir():
        return False
    nuclei_info = nuclei_dir / "nuclei_centroids_global.json"
    return nuclei_info.exists()

def alignment_done(run_dir: Path) -> bool:
    return (run_dir / "dapi_to_he_affine_level0.json").exists()

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
        self.setWindowTitle("Overall Pipeline *** Step 1-6")
        self.resize(860, 420)

        self.proc = None  # QProcess
        self.active_step = None

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

        # ---- Step 1 UI ----
        self.btn_step1 = QPushButton("Run Step 1: Image Selection")
        self.lbl_step1_status = QLabel(status_text(False))
        self.lbl_step1_status.setStyleSheet(status_style(False))
        self.btn_step1.clicked.connect(lambda: self.run_step("1"))

        # ---- Step 2 UI ----
        self.btn_step2a = QPushButton("Run Step 2A: Blob Matching")
        self.btn_step2b = QPushButton("Run Step 2B: Manual Alignment")
        self.lbl_step2_status = QLabel(status_text(False))
        self.lbl_step2_status.setStyleSheet(status_style(False))
        self.btn_step2a.clicked.connect(lambda: self.run_step("2a"))
        self.btn_step2b.clicked.connect(lambda: self.run_step("2b"))
        step2_btn_box = QWidget()
        step2_btn_layout = QVBoxLayout(step2_btn_box)
        step2_btn_layout.setContentsMargins(0, 0, 0, 0)
        step2_btn_layout.setSpacing(8)
        step2_btn_layout.addWidget(self.btn_step2a)
        step2_btn_layout.addWidget(self.btn_step2b)
        box = QGroupBox("Pipeline")
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)

        # ---- Step 3 UI ----
        self.btn_step3 = QPushButton("Run Step 3: Tile Extraction")
        self.lbl_step3_status = QLabel(status_text(False))
        self.lbl_step3_status.setStyleSheet(status_style(False))
        self.btn_step3.clicked.connect(lambda: self.run_step("3"))
        self.btn_step4 = QPushButton("Run Step 4: Nuclei Patch Extraction")
        self.lbl_step4_status = QLabel(status_text(False))
        self.lbl_step4_status.setStyleSheet(status_style(False))
        self.btn_step4.clicked.connect(lambda: self.run_step("4"))
        self.btn_step5 = QPushButton("Run Step 5: Nuclei Patch Gallery + Final Alignment Calculation")
        self.lbl_step5_status = QLabel(status_text(False))
        self.lbl_step5_status.setStyleSheet(status_style(False))
        self.btn_step5.clicked.connect(lambda: self.run_step("5"))
        self.btn_step6 = QPushButton("Run Step 6: Final Alignment Display")
        self.lbl_step6_status = QLabel(status_text(False))
        self.lbl_step6_status.setStyleSheet(status_style(False))
        self.btn_step6.clicked.connect(lambda: self.run_step("6"))

        # Header row
        hdr1 = QLabel("Step 1: Select Images")
        hdr1.setStyleSheet("font-size: 14px; font-weight: 700;")
        hdr2 = QLabel("Step 2: Get Initial Alignment")
        hdr2.setStyleSheet("font-size: 14px; font-weight: 700;")
        hdr3 = QLabel("Step 3: Extract Tiles")
        hdr3.setStyleSheet("font-size: 14px; font-weight: 700;")
        hdr4 = QLabel("Step 4: Extract Nuclei Patches")
        hdr4.setStyleSheet("font-size: 14px; font-weight: 700;")
        hdr5 = QLabel("Step 5: View Nuclei Patches and Get Final Alignment")
        hdr5.setStyleSheet("font-size: 14px; font-weight: 700;")
        hdr6 = QLabel("Step 6: View Final Alignment")
        hdr6.setStyleSheet("font-size: 14px; font-weight: 700;")

        grid.addWidget(hdr1, 0, 0, alignment=Qt.AlignLeft)
        grid.addWidget(hdr2, 0, 1, alignment=Qt.AlignLeft)
        grid.addWidget(hdr3, 3, 0, alignment=Qt.AlignLeft)
        grid.addWidget(hdr4, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(hdr5, 6, 0, alignment=Qt.AlignLeft)
        grid.addWidget(hdr6, 6, 1, alignment=Qt.AlignLeft)

        # Buttons row
        grid.addWidget(self.btn_step1, 1, 0)
        grid.addWidget(step2_btn_box, 1, 1)
        grid.addWidget(self.btn_step3, 4, 0)
        grid.addWidget(self.btn_step4, 4, 1)
        grid.addWidget(self.btn_step5, 7, 0)
        grid.addWidget(self.btn_step6, 7, 1)

        # Status row
        s1 = QHBoxLayout()
        s1.addWidget(QLabel("Status:"))
        s1.addWidget(self.lbl_step1_status)
        s1.addStretch(1)
        s2 = QHBoxLayout()
        s2.addWidget(QLabel("Status:"))
        s2.addWidget(self.lbl_step2_status)
        s2.addStretch(1)
        s3 = QHBoxLayout()
        s3.addWidget(QLabel("Status:"))
        s3.addWidget(self.lbl_step3_status)
        s3.addStretch(1)
        s4 = QHBoxLayout()
        s4.addWidget(QLabel("Status:"))
        s4.addWidget(self.lbl_step4_status)
        s4.addStretch(1)
        s5 = QHBoxLayout()
        s5.addWidget(QLabel("Status:"))
        s5.addWidget(self.lbl_step5_status)
        s5.addStretch(1)
        s6 = QHBoxLayout()
        s6.addWidget(QLabel("Status:"))
        s6.addWidget(self.lbl_step6_status)
        s6.addStretch(1)

        w_s1 = QWidget(); w_s1.setLayout(s1)
        w_s2 = QWidget(); w_s2.setLayout(s2)
        w_s3 = QWidget(); w_s3.setLayout(s3)
        w_s4 = QWidget(); w_s4.setLayout(s4)
        w_s5 = QWidget(); w_s5.setLayout(s5)
        w_s6 = QWidget(); w_s6.setLayout(s6)

        grid.addWidget(w_s1, 2, 0)
        grid.addWidget(w_s2, 2, 1)
        grid.addWidget(w_s3, 5, 0)
        grid.addWidget(w_s4, 5, 1)
        grid.addWidget(w_s5, 8, 0)
        grid.addWidget(w_s6, 8, 1)

        box.setLayout(grid)

        # ---- Main layout ----
        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(box)
        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_status()

    # ---------------------
    # RUN_DIR controls
    # ---------------------
    def on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose RUN_DIR", os.getcwd())
        if d:
            self.run_dir_edit.setText(d)
            self.refresh_status()

    def on_new_dir(self):
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = PROJECT_ROOT / f"runs_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir_edit.setText(str(run_dir))
        self.refresh_status()

    def get_run_dir(self) -> Path:
        txt = self.run_dir_edit.text().strip()
        if not txt:
            return None
        return Path(txt).resolve()

    # ---------------------
    # Status refresh
    # ---------------------
    def refresh_status(self):
        run_dir = self.get_run_dir()

        done1 = bool(run_dir) and step1_done(run_dir)
        done2 = bool(run_dir) and step2_done(run_dir)
        done3 = bool(run_dir) and step3_done(run_dir)
        done4 = bool(run_dir) and step4_done(run_dir)

        # ---- statuses ----
        idle = (self.proc is None)
        self.lbl_step1_status.setText(status_text(done1))
        self.lbl_step1_status.setStyleSheet(status_style(done1))
        self.lbl_step2_status.setText(status_text(done2))
        self.lbl_step2_status.setStyleSheet(status_style(done2))
        self.lbl_step3_status.setText(status_text(done3))
        self.lbl_step3_status.setStyleSheet(status_style(done3))
        self.lbl_step4_status.setText(status_text(done4))
        self.lbl_step4_status.setStyleSheet(status_style(done4))
        can_view_step5 = bool(run_dir) and done4 and idle
        align_ok = bool(run_dir) and alignment_done(run_dir)
        can_view_step6 = bool(run_dir) and done4 and align_ok and idle
        self.lbl_step5_status.setText(ready_text() if can_view_step5 else "LOCKED 🔒")
        self.lbl_step5_status.setStyleSheet(ready_style() if can_view_step5 else status_style(False))
        if can_view_step6:
            self.lbl_step6_status.setText(ready_text())
            self.lbl_step6_status.setStyleSheet(ready_style())
        else:
            if bool(run_dir) and done4 and idle and (not align_ok):
                self.lbl_step6_status.setText("MISSING FINAL ALIGNMENT 🔒")
                self.lbl_step6_status.setStyleSheet(status_style(False))
            else:
                self.lbl_step6_status.setText("LOCKED 🔒")
                self.lbl_step6_status.setStyleSheet(status_style(False))

        # Step1：需要 run_dir + idle；done 后禁用
        self.btn_step1.setEnabled(bool(run_dir) and idle and (not done1))
        # Step2：必须 Step1 done + idle；done 后禁用
        can_run_step2 = bool(run_dir) and done1 and idle and (not done2)
        self.btn_step2a.setEnabled(can_run_step2)
        self.btn_step2b.setEnabled(can_run_step2)
        # Step3：必须 Step2 done + idle；done 后禁用
        self.btn_step3.setEnabled(bool(run_dir) and done2 and idle and (not done3))
        # Step4：必须 Step3 done + idle；done 后禁用
        self.btn_step4.setEnabled(bool(run_dir) and done3 and idle and (not done4))
        # Step5：必须 Step4 done + idle
        self.btn_step5.setEnabled(can_view_step5)
        # Step6：必须 Step4 done + idle
        self.btn_step6.setEnabled(can_view_step6)

    # ---------------------
    # Run steps
    # ---------------------
    def run_step(self, step_id):
        run_dir = self.get_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "No RUN_DIR", "Please choose or create a RUN_DIR first.")
            return

        ensure_dir(run_dir)

        if self.proc is not None:
            QMessageBox.information(self, "Busy", "A step is already running.")
            return

        if step_id == "1":
            script = STEP1_SCRIPT
            args = [str(run_dir)]
        elif step_id == "2a":
            script = STEP2A_SCRIPT
            args = [str(run_dir)]
        elif step_id == "2b":
            script = STEP2B_SCRIPT
            args = [str(run_dir)]
        elif step_id == "3":
            script = STEP3_SCRIPT
            args = [str(run_dir)]
        elif step_id == "4":
            script = STEP4_SCRIPT
            args = [str(run_dir)]
        elif step_id == "5":
            script = STEP5_SCRIPT
            args = [str(run_dir)]
        elif step_id == "6":
            script = STEP6_SCRIPT
            args = [str(run_dir)]
        else:
            raise ValueError(step_id)

        script_path = Path(__file__).resolve().parent / script
        if not script_path.exists():
            QMessageBox.critical(self, "Script not found", f"Cannot find script:\n{script_path}")
            return

        self.active_step = step_id
        self.pre_step1_exists = step1_done(run_dir)
        self.pre_step2_exists = step2_done(run_dir)

        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments([str(script_path)] + args)
        self.proc.setWorkingDirectory(str(script_path.parent))

        self.proc.readyReadStandardOutput.connect(self._drain_stdout)
        self.proc.readyReadStandardError.connect(self._drain_stderr)
        self.proc.finished.connect(self._on_finished)

        # disable all while running
        self.btn_step1.setEnabled(False)
        self.btn_step2a.setEnabled(False)
        self.btn_step2b.setEnabled(False)
        self.btn_step3.setEnabled(False)
        self.btn_step4.setEnabled(False)
        self.btn_step5.setEnabled(False)
        self.btn_step6.setEnabled(False)

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
        step = self.active_step
        self.active_step = None
        self.proc = None
        self.refresh_status()
        run_dir = self.get_run_dir()

        if step == "1":
            ok = bool(run_dir) and step1_done(run_dir)
            if ok:
                QMessageBox.information(self, "Step 1 Saved", "Step 1 outputs detected (images_info.json found).")
            else:
                QMessageBox.warning(self, "Step 1 Not Saved",
                                    "Step 1 window closed without saving.\n"
                                    "No images_info.json found, so Step 1 is NOT finished.")
            return
        if step in ("2a", "2b"):
            ok_now = bool(run_dir) and step2_done(run_dir)
            if ok_now and not self.pre_step2_exists:
                QMessageBox.information(self, "Step 2 Saved", "Step 2 outputs detected (alignment json found).")
            elif ok_now and self.pre_step2_exists:
                QMessageBox.information(self, "Step 2 Already Done", "Alignment json already existed before this run.")
            else:
                QMessageBox.warning(self, "Step 2 Not Saved",
                                    "Step 2 window closed without saving.\n"
                                    "No alignment json found, so Step 2 is NOT finished.")
            return
        if step == "3":
            ok = bool(run_dir) and step3_done(run_dir)
            if ok:
                QMessageBox.information(self, "Step 3 Saved", "Step 3 outputs detected (tiles/ has files).")
            else:
                QMessageBox.warning(self, "Step 3 Not Saved",
                                    "Step 3 window/process ended without producing tiles.\n"
                                    "tiles/ is empty (or missing), so Step 3 is NOT finished.")
            return
        if step == "4":
            ok = bool(run_dir) and step4_done(run_dir)
            if ok:
                QMessageBox.information(self, "Step 4 Saved", "Step 4 outputs detected (nuclei/ has nuclei_info.json).")
            else:
                QMessageBox.warning(self, "Step 4 Not Saved",
                                    "Step 4 ended without producing nuclei outputs.\n"
                                    "nuclei/ is empty (or missing), so Step 4 is NOT finished.")
            return
        # fallback (shouldn't happen)
        if exitCode != 0:
            print(f"[INFO] step={step} exitCode={exitCode} exitStatus={exitStatus}", flush=True)
            QMessageBox.critical(self, "Step Failed", f"Step {step} failed (exit code = {exitCode}).")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = PipelineWindow()
    w.show()
    sys.exit(app.exec_())