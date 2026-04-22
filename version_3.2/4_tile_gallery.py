from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np
import cv2
import subprocess
import threading
import json
import sys
import time
import traceback
from datetime import datetime
print("[Stage4] imports done", flush=True)

# =========================
# Logging
# =========================
def log_event(run_dir, event_name, stage="stage4", **extra):
    out_json = Path(run_dir) / "pipeline_times.json"
    now_str = datetime.now().isoformat(timespec="seconds")

    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    stage_key = f"{stage}_events"
    if stage_key not in data or not isinstance(data[stage_key], list):
        data[stage_key] = []

    rec = {
        "event": event_name,
        "time": now_str,
    }
    rec.update(extra)
    data[stage_key].append(rec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[LOG] {event_name} -> {out_json}", flush=True)

# =========================
# Image helpers
# =========================
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
    if case_id == 6:      # transpose
        if img.ndim == 3:
            return np.transpose(np.rot90(img, k=3), (1, 0, 2))
        return np.transpose(np.rot90(img, k=3))
    if case_id == 7:      # rot90 CW + flip V
        return np.flipud(np.rot90(img, k=3))

    raise ValueError(f"Unknown orientation case: {case_id}")

def pad_to_fixed_size(img_pil, display_size, bg_color):
    img_pil = ImageOps.contain(img_pil, display_size)
    canvas = Image.new("RGB", display_size, bg_color)
    x = (display_size[0] - img_pil.width) // 2
    y = (display_size[1] - img_pil.height) // 2
    canvas.paste(img_pil, (x, y))
    return canvas

def load_optional(path, display_size, bg_color, is_mask=False, case_id=None):
    path = Path(path)

    if path.exists():
        if is_mask:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read mask: {path}")
            if case_id is not None:
                img = apply_orientation_to_tile(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(str(path))
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
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
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
            cv2.LINE_AA,
        )

    return pad_to_fixed_size(Image.fromarray(img), display_size, bg_color)

# =========================
# Stage-completion checks
# =========================
def stage4a_done_by_files(tiles_dir: Path) -> bool:
    return (tiles_dir / "nuclei_mask_info.json").exists()


def stage4b_done_by_files(tiles_dir: Path) -> bool:
    return (tiles_dir / "standout_nuclei.json").exists()


def stage4c_done_by_files(run_dir: Path) -> bool:
    return (run_dir / "nuclei_patches" / "nuclei_centroids_global.json").exists()

# =========================
# Subprocess helpers
# =========================
def make_progress_window(root, title):
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

    return progress_win, progress_var, status_label

def format_elapsed(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def run_process(
    root,
    run_dir,
    output_folder,
    script_name,
    progress_var,
    status_label,
    progress_win,
    start_time,
    system_ready_event,
    on_reload,
    refresh_button_states,
):
    try:
        proc = subprocess.Popen(
            [sys.executable, script_name, str(output_folder)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.strip()
            print("[Running]", line, flush=True)

            if line.startswith("[INFO]"):
                status_label.config(text=line.replace("[INFO]", "").strip())

            elif line.startswith("[ETA_START]"):
                start_time = time.time()
                progress_var.set(0)
                status_label.config(text="Starting main loop...")

            elif line.startswith("[PROGRESS]"):
                _, stage, frac = line.split()
                cur, total = map(int, frac.split("/"))

                now = time.time()
                elapsed = now - start_time
                remaining = (elapsed / cur) * (total - cur) if cur > 0 else 0

                pct = (cur / total) * 100
                progress_var.set(pct)
                status_label.config(
                    text=(
                        f"{stage}: {cur}/{total}  |  "
                        f"Elapsed: {format_elapsed(elapsed)}  |  "
                        f"ETA: {format_elapsed(remaining)}"
                    )
                )

            elif line.startswith("[DONE]"):
                progress_var.set(100)
                status_label.config(text="Finished 🎉")

                log_event(
                    run_dir,
                    system_ready_event,
                    script_name=script_name,
                    elapsed_sec=round(time.time() - start_time, 3),
                )

                messagebox.showinfo(
                    "Done",
                    f"{script_name} finished successfully."
                )

                root.after(0, on_reload)
                progress_win.destroy()
                root.after(0, lambda: refresh_button_states(running=False))

                if script_name == "4c_get_nuclei_patches.py":
                    messagebox.showinfo(
                        "Stage 4c Finished",
                        "Nucleus patch extraction finished.\nYou can open the nuclei patch gallery manually in stage 5."
                    )
                break

        proc.wait()
        root.after(0, lambda: refresh_button_states(running=False))

    except Exception as e:
        root.after(0, lambda: refresh_button_states(running=False))
        messagebox.showerror("Error", f"Failed to run {script_name}:\n{e}")
        traceback.print_exc()
        messagebox.showerror("Error", str(e))

def launch_script_with_progress(
    root,
    run_dir,
    output_folder,
    script_name,
    title,
    user_click_event,
    system_ready_event,
    on_reload,
    refresh_button_states,
):
    log_event(run_dir, user_click_event)

    start_time = time.time()
    refresh_button_states(running=True)

    progress_win, progress_var, status_label = make_progress_window(root, title)

    threading.Thread(
        target=run_process,
        args=(
            root,
            run_dir,
            output_folder,
            script_name,
            progress_var,
            status_label,
            progress_win,
            start_time,
            system_ready_event,
            on_reload,
            refresh_button_states,
        ),
        daemon=True,
    ).start()

    messagebox.showinfo("Launched", f"{title} has been launched.")

# =========================
# Main gallery
# =========================
def show_tile_gallery_in_memory(
    dapi_tiles,
    he_tiles,
    output_folder,
    run_dir,
    case_id,
    display_size=(256, 256),
    bg_color=(255, 255, 255),
):
    assert len(dapi_tiles) == len(he_tiles)

    tiles_dir = Path(output_folder).resolve()
    run_dir = Path(run_dir).resolve()

    root = tk.Tk()
    root.title("STAGE 4: DAPI & H&E Tile Gallery")

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )

    idx = [0]

    def on_close():
        log_event(run_dir, "user_click_exit")
        log_event(run_dir, "system_ready_exit", exit_mode="window_close")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

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

    def refresh_button_states(running=False):
        if running:
            btn_run4a.config(state="disabled")
            btn_run4b.config(state="disabled")
            btn_run4c.config(state="disabled")
            btn_prev.config(state="disabled")
            btn_next.config(state="disabled")
            btn_reload.config(state="disabled")
            return

        done_4a = stage4a_done_by_files(tiles_dir)
        done_4b = stage4b_done_by_files(tiles_dir)
        done_4c = stage4c_done_by_files(run_dir)

        btn_run4a.config(state="normal")
        btn_run4b.config(state=("normal" if done_4a else "disabled"))
        btn_run4c.config(state=("normal" if done_4b else "disabled"))

        btn_prev.config(state="normal")
        btn_next.config(state="normal")
        btn_reload.config(state="normal")

        print(
            f"[INFO] stage states | 4a={done_4a} 4b={done_4b} 4c={done_4c}",
            flush=True
        )

    def update_images():
        i = idx[0]
        dapi = dapi_tiles[i]
        he = he_tiles[i]

        base = dapi["filename"].replace("_dapi.png", "")
        dapi_img_name = dapi["filename"].replace("_dapi.png", "_dapi_u8.png")
        he_img_name = he["filename"]

        dapi_pil = load_optional(
            tiles_dir / dapi_img_name,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=False,
            case_id=case_id,
        )
        dapi_mask_pil = load_optional(
            tiles_dir / f"{base}_dapi_mask.png",
            display_size=display_size,
            bg_color=bg_color,
            is_mask=True,
            case_id=case_id,
        )
        he_pil = load_optional(
            tiles_dir / he_img_name,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=False,
            case_id=None,
        )
        he_mask_pil = load_optional(
            tiles_dir / f"{base}_he_mask.png",
            display_size=display_size,
            bg_color=bg_color,
            is_mask=True,
            case_id=None,
        )
        standout_pil = load_optional(
            tiles_dir / f"{base}_standout.jpg",
            display_size=display_size,
            bg_color=bg_color,
            is_mask=False,
            case_id=None,
        )

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

        info_label.config(text=f"{i + 1}/{len(dapi_tiles)} | {dapi['type'].capitalize()}")

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

    btn_run4a = ttk.Button(
        root,
        text="▶  Run Nuclei Masking",
        style="Gallery.TButton",
        command=lambda: launch_script_with_progress(
            root=root,
            run_dir=run_dir,
            output_folder=tiles_dir,
            script_name="4a_generate_nuclei_masks.py",
            title="Nuclei Masking",
            user_click_event="user_click_generate_nuclei_masks",
            system_ready_event="system_ready_generate_nuclei_masks",
            on_reload=on_reload,
            refresh_button_states=refresh_button_states,
        )
    )

    btn_run4b = ttk.Button(
        root,
        text="▶  Run Standout Nuclei Detection",
        style="Gallery.TButton",
        state="disabled",
        command=lambda: launch_script_with_progress(
            root=root,
            run_dir=run_dir,
            output_folder=tiles_dir,
            script_name="4b_find_standout_nuclei.py",
            title="Standout Nuclei Detection",
            user_click_event="user_click_find_standout_nuclei",
            system_ready_event="system_ready_find_standout_nuclei",
            on_reload=on_reload,
            refresh_button_states=refresh_button_states,
        )
    )

    btn_run4c = ttk.Button(
        root,
        text="▶  Run Nucleus Patch Cropping",
        style="Gallery.TButton",
        state="disabled",
        command=lambda: launch_script_with_progress(
            root=root,
            run_dir=run_dir,
            output_folder=tiles_dir,
            script_name="4c_get_nuclei_patches.py",
            title="Get Nucleus Patches",
            user_click_event="user_click_generate_nuclei_patches",
            system_ready_event="system_ready_generate_nuclei_patches",
            on_reload=on_reload,
            refresh_button_states=refresh_button_states,
        )
    )

    btn_prev.grid(row=3, column=0, pady=(0, 6))
    btn_next.grid(row=3, column=2, pady=(0, 6))
    btn_reload.grid(row=3, column=4, pady=(0, 6))
    btn_run4a.grid(row=4, column=0, pady=(10, 16))
    btn_run4b.grid(row=4, column=2, pady=(10, 16))
    btn_run4c.grid(row=4, column=4, pady=(10, 16))

    refresh_button_states(running=False)
    update_images()
    log_event(run_dir, "system_ready_initial_start")
    root.mainloop()


# =========================
# Entry point
# =========================
def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 4.py <RUN_DIR>")

    run_dir = Path(sys.argv[1]).resolve()
    tiles_dir = run_dir / "tiles"

    with open(tiles_dir / "dapi_tile_info.json", "r") as f:
        dapi_info = json.load(f)
    with open(tiles_dir / "he_tile_info.json", "r") as f:
        he_info = json.load(f)
    with open(run_dir / "images_info.json", "r") as f:
        case_id = json.load(f)["DAPI_orientation_case"]

    keys = sorted(dapi_info.keys())
    dapi_tiles = [dapi_info[k] for k in keys]
    he_tiles = [he_info[k] for k in keys]

    show_tile_gallery_in_memory(
        dapi_tiles=dapi_tiles,
        he_tiles=he_tiles,
        output_folder=tiles_dir,
        run_dir=run_dir,
        case_id=case_id,
    )


if __name__ == "__main__":
    main()