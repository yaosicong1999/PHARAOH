# ============================================================
# 3_get_tiles.py - Stage 3: Tile extraction (the Tk window)
#
# The computation -- tissue mask, CVT sampling of tile centroids, cropping the moving
# tiles and rectifying the fixed ones into <RUN_DIR>/tiles/ -- lives in
# `stage3_core.py` (`Stage3Sampler`), which imports no GUI toolkit. This file is the
# window around it: three image panels, three buttons and a progress dialog. The
# headless entry point is `cli_stage3.py`, which runs the same `Stage3Sampler`.
# Usage: python 3_get_tiles.py <RUN_DIR>
# ============================================================
import sys
import queue
import subprocess
import threading
from pathlib import Path

_ENGINE_DIR = str(Path(__file__).resolve().parent)
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import ImageTk

from stage3_core import *          # noqa: F401,F403 -- the computation, and every name tests reach through this module
from stage3_core import Stage3Sampler, log_event, apply_orientation_case, cv2_to_pil, fit_to_tile, load_image_any
import run_roles


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, title="Working..."):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)

        self._t0 = time.perf_counter()
        self._running = True
        self._stage_i = 0
        self._stage_total = 0
        self._stage_name = "Starting..."
        self._suffix = ""

        self.var = tk.DoubleVar(value=0.0)

        # 顶部状态行
        self.lbl = tk.Label(self, text="", anchor="w")
        self.lbl.pack(side="top", fill="x", padx=12, pady=(12, 6))

        self.pb = ttk.Progressbar(
            self, orient="horizontal", mode="determinate",
            maximum=100.0, variable=self.var, length=420
        )
        self.pb.pack(side="top", fill="x", padx=12, pady=(0, 10))

        self.txt = tk.Text(self, height=12, width=70)
        self.txt.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt.configure(state="disabled")

        self.btn_close = tk.Button(self, text="Close", state="disabled", command=self.destroy)
        self.btn_close.pack(side="bottom", pady=(0, 12))

        self.transient(parent)
        self.grab_set()

        self._tick_elapsed()

    def start_elapsed(self):
        def tick():
            if not self._running:
                return
            elapsed = time.perf_counter() - self._t0
            self.lbl.config(
                text=f"{self.lbl.cget('text').split(' | ')[0]} | Elapsed: {elapsed:.1f}s"
            )
            self.after(200, tick)

        tick()

    def stop_elapsed(self):
        self._running = False
        try:
            self.lbl.config(text=self._format_header())
        except tk.TclError:
            pass

    def _format_header(self):
        elapsed = time.perf_counter() - self._t0
        suffix = self._suffix or ""
        if self._stage_total > 0 and self._stage_i > 0:
            return f"STAGE {self._stage_i}/{self._stage_total}  {self._stage_name}   Elapsed: {elapsed:,.1f}s{suffix}"
        else:
            return f"{self._stage_name}   Elapsed: {elapsed:,.1f}s{suffix}"

    def _tick_elapsed(self):
        if not self._running:
            return
        try:
            self.lbl.config(text=self._format_header())
            self.after(200, self._tick_elapsed)
        except tk.TclError:
            return

    def set_stage(self, i: int, total: int, name: str):
        self._stage_i = int(i)
        self._stage_total = int(total)
        self._stage_name = str(name)
        self.lbl.config(text=self._format_header())

    def set_status(self, s: str):
        self._stage_name = str(s)
        self.lbl.config(text=self._format_header())

    def set_progress(self, p: float):
        self.var.set(float(p))

    def log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s.rstrip() + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def enable_close(self):
        self.btn_close.config(state="normal")

    def mark_done(self):
        self._suffix = " (Done)"
        self.lbl.config(text=self._format_header())

    def mark_failed(self):
        self._suffix = " (Failed)"
        self.lbl.config(text=self._format_header())


class Stage3SamplingApp(tk.Tk, Stage3Sampler):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.initialize_sampling_state(run_dir)
        self.title("Stage 3 — Sample tiles")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tile_size = (420, 420)
        print(f"[PARAM] stage3 = {self.stage3}", flush=True)
        print(f"[INFO] tile orientation_case={self.case_id}; "
              f"display orientation_case={self.display_case_id}", flush=True)

        top = tk.Frame(self)
        top.pack(side="top", fill="x", padx=12, pady=(10, 6))

        tk.Label(
            top,
            text=f"RUN_DIR: {self.run_dir}",
            anchor="w",
            font=("Helvetica", 12, "bold"),
        ).pack(side="left", fill="x", expand=True)

        mid = tk.Frame(self)
        mid.pack(side="top", fill="x", padx=12, pady=(6, 6))

        moving_label = "DAPI Image (LUT)" if self.roles.moving.modality == "dapi" else "H&E Image"
        self.panel_left  = self._make_image_panel(mid, f"Moving (foreground): {moving_label}")
        self.panel_mid   = self._make_image_panel(mid, "Available Mask for Tile Centroids")
        self.panel_right = self._make_image_panel(mid, "Sampled Tile Centroids")

        self.panel_left.grid(row=0, column=0, padx=8, sticky="n")
        self.panel_mid.grid(row=0, column=1, padx=8, sticky="n")
        self.panel_right.grid(row=0, column=2, padx=8, sticky="n")

        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)

        bottom = tk.Frame(self)
        bottom.pack(side="top", fill="x", padx=12, pady=(2, 12))

        self.btn_sampling = tk.Button(
            bottom, text="Sample tile centroids",
            command=self.on_sampling_clicked
        )
        self.btn_sampling.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_pilot = tk.Button(
            bottom, text="Tile pilot examination",
            command=self.on_pilot_clicked,
            state="disabled"
        )
        self.btn_pilot.pack(side="left", fill="x", expand=True, padx=(6, 6))

        self.btn_extract = tk.Button(
            bottom, text="Extract current tiles",
            command=self.on_extract_clicked,
            state="disabled"
        )
        self.btn_extract.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self._load_initial_left()
        self._set_placeholder(self.panel_mid, "Not Available Now")
        self._set_placeholder(self.panel_right, "Not Available Now")

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

        log_event(self.run_dir, "system_ready_initial_start")

    # --------------------------
    # window / panel setup
    # --------------------------
    def _make_image_panel(self, parent, title):
        frame = tk.Frame(parent)
        tk.Label(frame, text=title, font=("Helvetica", 12, "bold")).pack(side="top", pady=(0, 6))
        lbl = tk.Label(frame)
        lbl.pack(side="top")
        frame._img_label = lbl
        return frame

    def _set_panel_image(self, panel_frame, cv_img, apply_orientation=True):
        if apply_orientation:
            cv_img = apply_orientation_case(
                cv_img, getattr(self, "display_case_id", self.case_id)
            )
        pil = cv2_to_pil(cv_img)
        tile = fit_to_tile(pil, size=self.tile_size)
        tk_img = ImageTk.PhotoImage(tile)
        panel_frame._img_label.configure(image=tk_img)
        panel_frame._img_label.image = tk_img

    def _set_placeholder(self, panel_frame, text="placeholder"):
        w, h = self.tile_size  # (W,H)
        img = np.full((h, w, 3), 240, np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (80, 80, 80)

        # --- compute centered position ---
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2 + th

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        self._set_panel_image(panel_frame, img, apply_orientation=False)

    def _load_initial_left(self):
        moving = self.roles.moving
        filename = (f"1_{moving.token}_lut.png" if moving.modality == "dapi"
                    else self.roles.pair.stage1_image(run_roles.MOVING))
        img_dir = self.run_dir / filename
        img = load_image_any(img_dir)
        if img is None:
            self._set_placeholder(self.panel_left, "Missing moving preview")
            messagebox.showwarning("Missing", f"Can't find {img_dir.name} in RUN_DIR.")
            return

        self._set_panel_image(self.panel_left, img, apply_orientation=False)

    # --------------------------
    # button callbacks
    # --------------------------
    def on_sampling_clicked(self):
        log_event(self.run_dir, "user_click_sample_centroids")
        try:
            result = self._run_sampling_core()
            self._update_sampling_gui(
                avail_path=result["avail_path"],
                overlay_path=result["overlay_path"],
                avail_fallback=result["avail_flipped"],
                overlay_fallback=result["overlay_bgr"],
            )

            self.btn_pilot.config(state="normal")
            self.btn_extract.config(state="normal")
            self.has_sampling_outputs = True
            print("[DONE] sampling finished.", flush=True)

            log_event(
                self.run_dir,
                "system_ready_sample_centroids",
                n_points=int(len(result["points_xy"])),
                dapi_level=int(result["dapi_level"]),
                dapi_tile_level=int(result["dapi_tile_level"]),
                random_seed=int(result["random_seed"]),
                tile_size_sampling=float(result["tile_size_sampling"]),
                sampled_points_json=str(result["sampled_points_json"]),
                overlay_path=str(result["overlay_path"]),
            )

        except Exception as e:
            messagebox.showerror("Sampling failed", str(e))
            raise

    def _update_sampling_gui(self, avail_path, overlay_path, avail_fallback, overlay_fallback):
        # QC files may be stale after QC is disabled or the sampling parameters change.
        # Display the exact arrays from this sampling, irrespective of disk previews.
        self._set_panel_image(self.panel_mid, avail_fallback)
        self._set_panel_image(self.panel_right, overlay_fallback)

    def on_pilot_clicked(self):
        """
        Launch 3b.py for tile pilot examination.
        Pass RUN_DIR as argv[1].
        """
        try:
            script_dir = Path(__file__).resolve().parent
            # The pilot tools are GUI-only and live outside the engine directories so
            # nothing on the CLI path can reach them.
            pilot_script = (script_dir.parent / "gui_pilot" / script_dir.name
                            / "3b_tile_pilot.py")

            if not pilot_script.exists():
                messagebox.showerror("Missing", f"Cannot find {pilot_script}")
                return

            # Use same python executable (important for env)
            py = sys.executable

            cmd = [py, str(pilot_script), str(self.run_dir)]
            print("[INFO] launching:", " ".join(cmd), flush=True)

            # Non-blocking
            subprocess.Popen(cmd, cwd=str(script_dir))

        except Exception as e:
            messagebox.showerror("Failed to launch 3b.py", str(e))

    def on_extract_clicked(self):
        log_event(self.run_dir, "user_click_save_tiles")
        if not self.has_sampling_outputs:
            messagebox.showwarning("Not ready", "Please run sampling first.")
            return
        self._run_with_progress("Extracting tiles...", self.extract_tiles)

    def on_close(self):
        log_event(self.run_dir, "user_click_exit")
        log_event(self.run_dir, "system_ready_exit", exit_mode="window_close")
        self.destroy()

    # --------------------------
    # progress / background task
    # --------------------------
    def _run_with_progress(self, title, worker_fn):
        dlg = ProgressDialog(self, title=title)
        q = queue.Queue()

        def pump():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "stage":
                        i, total, name = payload
                        dlg.set_stage(i, total, name)
                        dlg.log(f"[STAGE] {i}/{total} {name}")
                    elif kind == "progress":
                        dlg.set_progress(payload)
                    elif kind == "log":
                        dlg.log(payload)
                    elif kind == "done":
                        dlg.mark_done()
                        dlg.stop_elapsed()
                        dlg.set_progress(100)
                        dlg.enable_close()
                        return
                    elif kind == "error":
                        dlg.log("[ERROR] " + str(payload))
                        dlg.mark_failed()
                        dlg.stop_elapsed()
                        dlg.enable_close()
                        return
            except queue.Empty:
                pass
            self.after(100, pump)

        def bg():
            try:
                worker_fn(q)
                q.put(("done", None))
            except Exception as e:
                q.put(("error", str(e)))

        threading.Thread(target=bg, daemon=True).start()
        pump()

def main():
    if len(sys.argv) < 2:
        print("Usage: python 3_get_tiles.py <RUN_DIR>")
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] RUN_DIR not found: {run_dir}")
        sys.exit(2)

    app = Stage3SamplingApp(run_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
