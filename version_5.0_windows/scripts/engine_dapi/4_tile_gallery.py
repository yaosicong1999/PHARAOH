# ============================================================
# 4_tile_gallery.py - Stage 4: Tile gallery & nuclei orchestration
#
# Tk gallery over extracted tiles; runs 4a (nuclei masks) ->
# 4b (standout nuclei) -> 4c (nucleus patches) as subprocesses.
# Usage: python 4_tile_gallery.py <RUN_DIR>
# ============================================================
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
import os
import queue
from datetime import datetime

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR.parent))
import nucleus_masks
import run_roles
from stage4_all_tiles import all_tiles_detection_ready
from patch_qc import patch_images_ready

print("[Stage4] imports done", flush=True)

# -----------------------------
# Logging
# -----------------------------
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

# -----------------------------
# Image helpers
# -----------------------------
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

# -----------------------------
# Stage-completion checks
# -----------------------------
def load_gallery_tiles(tiles_dir, roles):
    """The full GUI reviews every Stage-3 pair, independently of a CLI subset."""
    tiles_dir = Path(tiles_dir)
    moving = json.loads((tiles_dir / f"{roles.moving.token}_tile_info.json").read_text())
    fixed = json.loads((tiles_dir / f"{roles.fixed.token}_tile_info.json").read_text())
    return [
        {"tile_id": key, "moving": moving[key], "fixed": fixed[key]}
        for key in sorted(set(moving).intersection(fixed))
    ]


def gallery_image_paths(tiles_dir, tile, roles):
    """Resolve by tile ID and role token, not by stripping an old DAPI suffix."""
    tiles_dir = Path(tiles_dir)
    key = tile["tile_id"]

    def raw_path(role):
        side = roles.role(role)
        meta = tile[role]
        # Legacy metadata can name a uint16 DAPI tile; display its uint8 rendering.
        canonical = tiles_dir / nucleus_masks.tile_filename(key, side.token, side.modality)
        preferred = meta.get(f"filename_{side.token}_u8") if side.modality == "dapi" else None
        if preferred:
            return tiles_dir / preferred
        if canonical.is_file():
            return canonical
        return tiles_dir / meta["filename"]

    return (
        raw_path(run_roles.MOVING),
        tiles_dir / nucleus_masks.mask_filename(key, roles.moving.token),
        raw_path(run_roles.FIXED),
        tiles_dir / nucleus_masks.mask_filename(key, roles.fixed.token),
        tiles_dir / f"{key}_standout.jpg",
    )


def gallery_orientation_case(tile, roles):
    # New Stage-3 tiles use canonical coordinates, independently of Stage-1 display
    # rotation. Honor the tile metadata, with the same legacy fallback as Stage 4b.
    return int(tile["fixed"].get("meta", {}).get("case_id", roles.moving.orientation_case))


def stage4a_done_by_files(tiles_dir: Path) -> bool:
    tiles_dir = Path(tiles_dir)
    if not (tiles_dir / "nuclei_mask_info.json").is_file():
        return False
    roles = run_roles.read(tiles_dir.parent)
    tiles = load_gallery_tiles(tiles_dir, roles)
    return bool(tiles) and all(
        (tiles_dir / nucleus_masks.mask_filename(tile["tile_id"], side.token)).is_file()
        for tile in tiles for side in (roles.moving, roles.fixed)
    )


def stage4b_done_by_files(tiles_dir: Path) -> bool:
    return all_tiles_detection_ready(tiles_dir)


def stage4c_done_by_files(run_dir: Path) -> bool:
    return patch_images_ready(run_dir)

# -----------------------------
# Subprocess helpers
# -----------------------------
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

def stage_command(run_dir, output_folder, script_name):
    """Same engine; full GUI processes all tiles and forces standout previews."""
    roles = run_roles.read(run_dir)
    command = [sys.executable, str(ENGINE_DIR / script_name), str(output_folder)]
    env = os.environ.copy()
    env["PHARAOH_PAIR"] = roles.pair.name
    if script_name in ("4a_generate_nuclei_masks.py", "4b_find_standout_nuclei.py"):
        command.append("--all-tiles")
    if script_name == "4b_find_standout_nuclei.py":
        # Subprocess-local: do not change parameters.json, the parent environment,
        # CLI --no-qc-images, or any later stage's QC settings.
        env["PHARAOH_QC_IMAGES"] = "stage4b_overlays"
        command.append("--write-centroids")
    if script_name == "4c_get_nuclei_patches.py" and (
        Path(run_dir) / "nuclei_patches/nuclei_centroids_global.json"
    ).exists():
        command.append("--patches-only")
    return command, env


def stage_output_ready(run_dir, output_folder, script_name):
    checks = {
        "4a_generate_nuclei_masks.py": lambda: stage4a_done_by_files(Path(output_folder)),
        "4b_find_standout_nuclei.py": lambda: stage4b_done_by_files(Path(output_folder)),
        "4c_get_nuclei_patches.py": lambda: stage4c_done_by_files(Path(run_dir)),
    }
    return checks[script_name]()


def run_process(run_dir, output_folder, script_name, events):
    """Worker: drain stdout and wait for exit; never call Tk from this thread."""
    try:
        command, env = stage_command(run_dir, output_folder, script_name)
        proc = subprocess.Popen(
            command,
            cwd=str(ENGINE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                print("[Running]", line, flush=True)
                events.put(("line", line))

        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"{script_name} exited with code {returncode}; see the pipeline log.")
        if not stage_output_ready(run_dir, output_folder, script_name):
            raise RuntimeError(f"{script_name} exited without its required output file.")
        events.put(("finished", None))

    except Exception as e:
        traceback.print_exc()
        events.put(("error", str(e)))

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

    start_time = [time.time()]
    refresh_button_states(running=True)

    progress_win, progress_var, status_label = make_progress_window(root, title)
    events = queue.Queue()

    def poll_events():
        # All widget updates, dialogs and gallery reloads run on Tk's main thread.
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind in ("finished", "error"):
                    progress_win.destroy()
                    try:
                        on_reload()
                    except Exception as exc:
                        kind, payload = "error", f"Gallery refresh failed: {exc}"
                    finally:
                        refresh_button_states(running=False)
                    if kind == "error":
                        messagebox.showerror("Error", f"Failed to run {script_name}:\n{payload}")
                    else:
                        log_event(
                            run_dir, system_ready_event, script_name=script_name,
                            elapsed_sec=round(time.time() - start_time[0], 3),
                        )
                        # Images have already reloaded, before any modal Done dialog.
                        messagebox.showinfo("Done", f"{title} finished successfully.")
                    return
                line = payload
                if line.startswith("[INFO]"):
                    status_label.config(text=line.removeprefix("[INFO]").strip())
                elif line.startswith("[ETA_START]"):
                    start_time[0] = time.time()
                    progress_var.set(0)
                    status_label.config(text="Starting main loop...")
                elif line.startswith("[PROGRESS]"):
                    _, stage, frac = line.split()
                    cur, total = map(int, frac.split("/"))
                    elapsed = time.time() - start_time[0]
                    remaining = elapsed / cur * (total - cur) if cur > 0 else 0
                    progress_var.set(100 * cur / max(total, 1))
                    status_label.config(text=(
                        f"{stage}: {cur}/{total} | Elapsed: {format_elapsed(elapsed)}"
                        f" | ETA: {format_elapsed(remaining)}"
                    ))
                # [DONE] may describe an intermediate output, before JSON is saved.
                # Only the process exit + required artifact check means completion.
        except queue.Empty:
            root.after(100, poll_events)

    threading.Thread(
        target=run_process,
        args=(run_dir, output_folder, script_name, events),
        daemon=True,
    ).start()
    root.after(100, poll_events)

# -----------------------------
# Main gallery
# -----------------------------
def show_tile_gallery_in_memory(
    output_folder,
    run_dir,
    roles,
    display_size=(256, 256),
    bg_color=(255, 255, 255),
):
    tiles_dir = Path(output_folder).resolve()
    run_dir = Path(run_dir).resolve()
    tiles = load_gallery_tiles(tiles_dir, roles)

    root = tk.Tk()
    root.title(f"STAGE 4: {roles.pair.name} — All Tile Gallery")

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )

    idx = [0]
    running_stage = [False]

    def on_close():
        if running_stage[0]:
            messagebox.showinfo("Stage running", "Wait for the current stage to finish before closing.")
            return
        log_event(run_dir, "user_click_exit")
        log_event(run_dir, "system_ready_exit", exit_mode="window_close")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # ---------- Labels ----------
    names = {"dapi": "DAPI", "he": "H&E"}
    moving_name, fixed_name = names[roles.moving.modality], names[roles.fixed.modality]
    tk.Label(root, text=f"Moving: {moving_name}", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text=f"{moving_name} nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text=f"Fixed: {fixed_name}", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text=f"{fixed_name} nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)
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
        running_stage[0] = running
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

        btn_prev.config(state="normal" if tiles else "disabled")
        btn_next.config(state="normal" if tiles else "disabled")
        btn_reload.config(state="normal")

        print(
            f"[INFO] stage states | 4a={done_4a} 4b={done_4b} 4c={done_4c}",
            flush=True
        )

    def update_images():
        labels = [dapi_label, dapi_mask_label, he_label, he_mask_label, standout_nuclei_label]
        if not tiles:
            for label in labels:
                label.config(image="", text="No paired tiles available", width=30, height=12)
                label.image = None
            info_label.config(text="No paired tiles found — complete Stage 3 tile extraction first.")
            return
        i = idx[0]
        tile = tiles[i]
        dapi_path, dapi_mask_path, he_path, he_mask_path, standout_path = gallery_image_paths(
            tiles_dir, tile, roles
        )
        case_id = gallery_orientation_case(tile, roles)

        dapi_pil = load_optional(
            dapi_path,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=False,
            case_id=case_id,
        )
        dapi_mask_pil = load_optional(
            dapi_mask_path,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=True,
            case_id=case_id,
        )
        he_pil = load_optional(
            he_path,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=False,
            case_id=None,
        )
        he_mask_pil = load_optional(
            he_mask_path,
            display_size=display_size,
            bg_color=bg_color,
            is_mask=True,
            case_id=None,
        )
        standout_pil = load_optional(
            standout_path,
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

        for lbl, im in zip(labels, imgs):
            lbl.config(image=im, text="", width=0, height=0)
            lbl.image = im

        info_label.config(text=f"All Stage 3 tiles: {i + 1}/{len(tiles)} | {tile['tile_id']}")

    def next_tile():
        idx[0] = (idx[0] + 1) % len(tiles)
        update_images()

    def prev_tile():
        idx[0] = (idx[0] - 1) % len(tiles)
        update_images()

    def on_reload():
        previous_id = tiles[idx[0]]["tile_id"] if tiles else None
        tiles[:] = load_gallery_tiles(tiles_dir, roles)
        idx[0] = next((i for i, tile in enumerate(tiles) if tile["tile_id"] == previous_id), 0)
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
        text="▶  Generate Nuclei Patches (Optional QC)",
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


# -----------------------------
# Entry point
# -----------------------------
def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 4_tile_gallery.py <RUN_DIR>")

    run_dir = Path(sys.argv[1]).resolve()
    tiles_dir = run_dir / "tiles"

    show_tile_gallery_in_memory(
        output_folder=tiles_dir,
        run_dir=run_dir,
        roles=run_roles.read(run_dir),
    )


if __name__ == "__main__":
    main()
