from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np
import cv2
import os
import subprocess
import threading
import json
import sys
import time
import traceback

def apply_orientation_to_tile(img, case_id):
    """
    img: np.ndarray (H,W) or (H,W,3)
    case_id: int in [0..7]
    """
    if case_id == 0:
        return img
    if case_id == 1:      # rot90 CW
        return np.rot90(img, k=3)
    if case_id == 2:      # rot180
        return np.rot90(img, k=2)
    if case_id == 3:      # rot90 CCW
        return np.rot90(img, k=1)
    if case_id == 4:      # flip vertical
        return np.flipud(img)
    if case_id == 5:      # flip horizontal
        return np.fliplr(img)
    if case_id == 6:      # rot90 CW + flip H (transpose)
        if img.ndim == 3:
            return np.transpose(np.rot90(img, k=3), (1, 0, 2))
        else:
            return np.transpose(np.rot90(img, k=3))
    if case_id == 7:      # rot90 CW + flip V
        return np.flipud(np.rot90(img, k=3))

    raise ValueError(f"Unknown orientation case: {case_id}")



def show_tile_gallery_in_memory(
        dapi_tiles,
        he_tiles,
        output_folder,
        case_id,
        display_size=(256, 256),
        bg_color=(255, 255, 255)
):
    assert len(dapi_tiles) == len(he_tiles)

    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )
    root.title("STEP 4: DAPI & H&E Tile Gallery")
    idx = [0]
    # ---------- Labels ----------
    tk.Label(root, text="DAPI", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="DAPI nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)
    tk.Label(root, text="Standout nuclei", font=("Helvetica", 15)).grid(row=0, column=4)

    dapi_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)
    standout_nuclei_label = tk.Label(root)

    dapi_label.grid(row=1, column=0, padx=8, pady=8)
    dapi_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)
    standout_nuclei_label.grid(row=1, column=4, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # ---------- Utils ----------
    # ----------------------------
    # File-based step completion
    # ----------------------------
    tiles_dir = Path(output_folder)                # .../RUN_DIR/tiles
    run_dir = tiles_dir.parent                     # .../RUN_DIR

    # all tile bases (tile_000, tile_001, ...)
    tile_bases = []
    for d in dapi_tiles:
        base = d["filename"].replace("_dapi.png", "")
        tile_bases.append(base)

    def _all_exist(relpaths):
        # relpaths: list[str] relative to tiles_dir
        for rp in relpaths:
            if not (tiles_dir / rp).exists():
                return False
        return True

    def step5_done_by_files():
        return (tiles_dir / "nuclei_mask_info.json").exists()

    def step6_done_by_files():
        return (tiles_dir / "standout_nuclei.json").exists()
    def step7_done_by_files():
        return (run_dir / "nuclei_patches" / "nuclei_centroids_global.json").exists()

    def refresh_button_states(running=False):
        """
        running=True: disable all buttons (when a subprocess is running)
        running=False: enable/disable according to file existence
        """
        if running:
            btn_run5.config(state="disabled")
            btn_run6.config(state="disabled")
            btn_run7.config(state="disabled")
            return

        s5 = step5_done_by_files()
        s6 = step6_done_by_files()
        s7 = step7_done_by_files()

        # allow re-run step5 anytime
        btn_run5.config(state="normal")
        # step6 only if step5 done
        btn_run6.config(state=("normal" if s5 else "disabled"))
        # step7 only if step6 done
        btn_run7.config(state=("normal" if s6 else "disabled"))

        btn_prev.config(state="normal")
        btn_next.config(state="normal")
        btn_reload.config(state="normal")
    def pad_to_fixed_size(img_pil):
        img_pil = ImageOps.contain(img_pil, display_size)
        canvas = Image.new("RGB", display_size, bg_color)
        x = (display_size[0] - img_pil.width) // 2
        y = (display_size[1] - img_pil.height) // 2
        canvas.paste(img_pil, (x, y))
        return canvas

    def load_optional(path, is_mask=False, case_id=None):
        if os.path.exists(path):
            if is_mask:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read mask: {path}")
                if case_id is not None:
                    img = apply_orientation_to_tile(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                if case_id is not None:
                    img = apply_orientation_to_tile(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.full((display_size[1], display_size[0], 3), 240, np.uint8)
            text = "NOT AVAILABLE NOW"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = (display_size[0] - text_w) // 2
            y = (display_size[1] + text_h) // 2
            cv2.putText(
                img,
                text,
                (x, y),
                font,
                font_scale,
                (120, 120, 120),
                thickness,
                cv2.LINE_AA
            )
        return pad_to_fixed_size(Image.fromarray(img))

    def launch_script_with_progress(root, script_name, title):
        start_time = time.time()
        refresh_button_states(running=True)
        progress_win = tk.Toplevel(root)
        progress_win.title(title)
        progress_win.geometry("420x160")
        progress_win.resizable(False, False)
        tk.Label(
            progress_win,
            text=f"{title}…",
            font=("Arial", 12)
        ).pack(pady=(15, 5))
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_win,
            variable=progress_var,
            maximum=100,
            length=360
        )
        progress_bar.pack(pady=10)
        status_label = tk.Label(progress_win, text="Starting…")
        status_label.pack(pady=(5, 10))
        threading.Thread(
            target=run_process,
            args=(script_name, progress_var, status_label, progress_win, start_time),
            daemon=True
        ).start()
        messagebox.showinfo(
            "Launched",
            f"{title} has been launched."
        )

    def run_process(script_name, progress_var, status_label, progress_win, start_time):
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    script_name,
                    output_folder
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in proc.stdout:
                line = line.strip()
                print("[Running]", line)
                if line.startswith("[INFO]"):
                    status_label.config(
                        text=line.replace("[INFO]", "").strip()
                    )
                elif line.startswith("[ETA_START]"):
                    start_time = time.time()
                    progress_var.set(0)
                    status_label.config(text="Starting main loop...")
                elif line.startswith("[PROGRESS]"):
                    _, stage, frac = line.split()
                    cur, total = map(int, frac.split("/"))

                    now = time.time()
                    elapsed = now - start_time

                    if cur > 0:
                        avg_time = elapsed / cur
                        remaining = avg_time * (total - cur)
                    else:
                        remaining = 0

                    def fmt(sec):
                        m, s = divmod(int(sec), 60)
                        h, m = divmod(m, 60)
                        if h > 0:
                            return f"{h:d}h {m:02d}m {s:02d}s"
                        else:
                            return f"{m:02d}m {s:02d}s"

                    pct = (cur / total) * 100
                    progress_var.set(pct)

                    status_label.config(
                        text=(
                            f"{stage}: {cur}/{total}  |  "
                            f"Elapsed: {fmt(elapsed)}  |  "
                            f"ETA: {fmt(remaining)}"
                        )
                    )
                elif line.startswith("[DONE]"):
                    progress_var.set(100)
                    status_label.config(text="Finished 🎉")
                    messagebox.showinfo(
                        "Done",
                        f"{script_name} finished successfully!\nLaunching next step..."
                    )
                    root.after(0, on_reload)
                    progress_win.destroy()
                    root.after(0, lambda: refresh_button_states(running=False))
                    if script_name == "7_get_nuclei_patches.py":
                        messagebox.showinfo(
                            "Step 7 Finished",
                            "Nucleus patch extracted.\nYou can open the nuclei patch gallery manually (Step 8)."
                        )
                    break

            proc.wait()
            root.after(0, lambda: refresh_button_states(running=False))
        except Exception as e:
            root.after(0, lambda: refresh_button_states(running=False))
            messagebox.showerror(
                "Error",
                f"Failed to run {script_name}:\n{e}"
            )
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    # ---------- Core ----------
    def update_images():
        i = idx[0]
        dapi = dapi_tiles[i]
        he = he_tiles[i]

        # ----------------------------
        # resolve base & filenames
        # ----------------------------
        base = dapi["filename"].replace("_dapi.png", "")
        dapi_img_name = dapi["filename"]
        he_img_name = he["filename"]
        # ----------------------------
        # DAPI image (from disk)
        # ----------------------------
        dapi_pil = load_optional(
            os.path.join(output_folder, dapi_img_name),
            is_mask=False,
            case_id=case_id
        )
        # ----------------------------
        # DAPI mask
        # ----------------------------
        dapi_mask_pil = load_optional(
            os.path.join(output_folder, f"{base}_dapi_mask.png"),
            is_mask=True,
            case_id=case_id
        )
        # ----------------------------
        # HE image (from disk)
        # ----------------------------
        he_pil = load_optional(
            os.path.join(output_folder, he_img_name),
            is_mask=False
        )
        # ----------------------------
        # HE mask
        # ----------------------------
        he_mask_pil = load_optional(
            os.path.join(output_folder, f"{base}_he_mask.png"),
            is_mask=True
        )
        # ----------------------------
        # Standout Nuclei
        # ----------------------------
        standout_pil = load_optional(
            os.path.join(output_folder, f"{base}_standout.jpg"),
            is_mask=False
        )
        # ----------------------------
        # Tk images
        # ----------------------------
        imgs = [
            ImageTk.PhotoImage(dapi_pil),
            ImageTk.PhotoImage(dapi_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
            ImageTk.PhotoImage(standout_pil),
        ]

        labels = [
            dapi_label,
            dapi_mask_label,
            he_label,
            he_mask_label,
            standout_nuclei_label,
        ]

        for lbl, im in zip(labels, imgs):
            lbl.config(image=im)
            lbl.image = im

        info_label.config(
            text=f"{i + 1}/{len(dapi_tiles)} | {dapi['type'].capitalize()}"
        )

    def next_tile():
        idx[0] = (idx[0] + 1) % len(dapi_tiles)
        update_images()

    def prev_tile():
        idx[0] = (idx[0] - 1) % len(dapi_tiles)
        update_images()

    def on_reload():
        update_images()
        refresh_button_states(running=False)

    # ---------- Buttons ----------
    btn_prev = ttk.Button(root, text="⟨ Previous", command=prev_tile)
    btn_next = ttk.Button(root, text="Next ⟩", command=next_tile)
    btn_reload = ttk.Button(
        root,
        text="⟳ Refresh",
        style="Gallery.TButton",
        command=on_reload
    )
    btn_run5 = ttk.Button(
        root,
        text="▶  Run Nuclei Masking",
        style="Gallery.TButton",
        command=lambda: launch_script_with_progress(
            root,
            "5_generate_nuclei_masks.py",
            "Nuclei Masking"
        )
    )
    btn_run6 = ttk.Button(
        root,
        text="▶  Run Standout Nuclei Detection",
        style="Gallery.TButton",
        state="disabled",
        command=lambda: launch_script_with_progress(
            root,
            "6_find_standout_nuclei.py",
            "Standout Nuclei Detection"
        )
    )
    btn_run7 = ttk.Button(
        root,
        text="▶  Run Nucleus Patch Cropping",
        style="Gallery.TButton",
        state="disabled",
        command=lambda: launch_script_with_progress(
            root,
            "7_get_nuclei_patches.py",
            "Get Nucleus Patches"
        )
    )

    btn_prev.grid(row=3, column=0, pady=(0, 6))
    btn_next.grid(row=3, column=2, pady=(0, 6))
    btn_reload.grid(row=3, column=4, pady=(0, 6))
    btn_run5.grid(row=4, column=0, pady=(10, 16))
    btn_run6.grid(row=4, column=2, pady=(10, 16))
    btn_run7.grid(row=4, column=4, pady=(10, 16))
    refresh_button_states(running=False)
    update_images()
    root.mainloop()

def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 4_tile_gallery.py <tiles_folder>")
    output_folder = sys.argv[1]

    output_folder = output_folder + "/tiles/"
    with open(os.path.join(output_folder, "dapi_tile_info.json"), "r") as f:
        dapi_info = json.load(f)
    with open(os.path.join(output_folder, "he_tile_info.json"), "r") as f:
        he_info = json.load(f)
    with open(os.path.join(output_folder, "../images_info.json"), "r") as f:
        case_id = json.load(f)['DAPI_orientation_case']
    keys = sorted(dapi_info.keys())

    dapi_tiles = [dapi_info[k] for k in keys]
    he_tiles = [he_info[k] for k in keys]

    show_tile_gallery_in_memory(
        dapi_tiles=dapi_tiles,
        he_tiles=he_tiles,
        output_folder=output_folder,
        case_id=case_id
    )

if __name__ == "__main__":
    main()