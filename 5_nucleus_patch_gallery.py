from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import os
import json
import sys
from pathlib import Path
import time
from datetime import datetime
import shutil

# ==========================================================
# Logging / parameter loading
# ==========================================================
def log_event(run_dir, event_name, stage="stage5", **extra):
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

def load_stage5_params(
    run_dir,
    default_mode="tps",
    default_balance_points_bool=False,
    default_dapi_mask_upscale_factor=1,
    default_he_mask_upscale_factor=1,
):
    path = Path(run_dir) / "../parameters.json"

    params = {
        "transform_mode": str(default_mode),
        "balance_points_bool": bool(default_balance_points_bool),
        "dapi_mask_upscale_factor": int(default_dapi_mask_upscale_factor),
        "he_mask_upscale_factor": int(default_he_mask_upscale_factor),
    }

    if not os.path.exists(path):
        print(f"[INFO] parameters from: {path} do not exist", flush=True)
        return params

    try:
        with open(path, "r") as f:
            data = json.load(f)
            print(f"[INFO] loaded parameters from: {path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Use defaults.", flush=True)
        return params

    if not isinstance(data, dict):
        return params

    stage5 = data.get("stage5", {})
    stage4c = data.get("stage4c", {})

    mode = str(stage5.get("transform_mode", params["transform_mode"])).strip().lower()
    if mode in {"affine", "homography", "tps", "local_tps"}:
        params["transform_mode"] = mode
    else:
        print(
            f"[WARN] Invalid stage5.transform_mode={mode}. "
            f"Use default {params['transform_mode']}.",
            flush=True
        )

    params["balance_points_bool"] = bool(
        stage5.get("balance_points_bool", params["balance_points_bool"])
    )

    try:
        params["dapi_mask_upscale_factor"] = int(
            stage4c.get("dapi_mask_upscale_factor", params["dapi_mask_upscale_factor"])
        )
    except Exception:
        pass

    try:
        params["he_mask_upscale_factor"] = int(
            stage4c.get("he_mask_upscale_factor", params["he_mask_upscale_factor"])
        )
    except Exception:
        pass

    return params

def load_manual_centroid_map(output_folder):
    manual_path = Path(output_folder) / "manual_centroids.json"
    manual_map = {}

    if not manual_path.exists():
        return manual_map

    with open(manual_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            key = (
                rec.get("tile_id"),
                rec.get("nucleus_id"),
                rec.get("image_kind"),
            )
            xy = rec.get("manual_centroid_xy", None)
            if xy is None or len(xy) != 2:
                continue

            manual_map[key] = [float(xy[0]), float(xy[1])]

    return manual_map

def replace_global_with_manual_if_any(nuclei_info, manual_map, meta):
    dapi_patch_level = int(meta.get("dapi_patch_level", 0))
    he_patch_level = int(meta.get("he_patch_level", 0))

    dapi_scale_to_level0 = float(2 ** dapi_patch_level)
    he_scale_to_level0 = float(2 ** he_patch_level)

    updated = []

    for rec in nuclei_info:
        rec = dict(rec)  # copy
        tile_id = rec.get("tile_id")
        nucleus_id = rec.get("nucleus_id")

        # ---------- DAPI ----------
        dapi_key = (tile_id, nucleus_id, "dapi")
        if dapi_key in manual_map:
            old_local = np.asarray(rec["dapi_centroid_local"], dtype=np.float64)
            old_global0 = np.asarray(rec["dapi_centroid_global"], dtype=np.float64)
            new_local = np.asarray(manual_map[dapi_key], dtype=np.float64)

            old_global_patch = old_global0 / dapi_scale_to_level0
            patch_origin = old_global_patch - old_local
            new_global_patch = patch_origin + new_local
            new_global0 = new_global_patch * dapi_scale_to_level0

            rec["dapi_centroid_local"] = [float(new_local[0]), float(new_local[1])]
            rec["dapi_centroid_global"] = [float(new_global0[0]), float(new_global0[1])]
            rec["dapi_manual_override"] = True

        # ---------- HE ----------
        he_key = (tile_id, nucleus_id, "he")
        if he_key in manual_map and rec.get("he_centroid_local") is not None and rec.get(
            "he_centroid_global") is not None:
            old_local = np.asarray(rec["he_centroid_local"], dtype=np.float64)
            old_global0 = np.asarray(rec["he_centroid_global"], dtype=np.float64)
            new_local = np.asarray(manual_map[he_key], dtype=np.float64)

            old_global_patch = old_global0 / he_scale_to_level0
            patch_origin = old_global_patch - old_local
            new_global_patch = patch_origin + new_local
            new_global0 = new_global_patch * he_scale_to_level0

            rec["he_centroid_local"] = [float(new_local[0]), float(new_local[1])]
            rec["he_centroid_global"] = [float(new_global0[0]), float(new_global0[1])]
            rec["he_manual_override"] = True

        updated.append(rec)
    return updated


# ==========================================================
# display helpers
# ==========================================================
def apply_orientation_to_patch(img, case_id):
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

def invert_orientation_point(x, y, w, h, case_id):
    """
    Map point from oriented/display image coords back to original image coords.
    x, y: coords on displayed (already oriented) image
    w, h: original image width/height BEFORE orientation
    case_id: same DAPI_orientation_case
    """
    x = float(x)
    y = float(y)

    if case_id == 0:   # identity
        return x, y
    elif case_id == 1: # rot90 CW
        # new(x',y') = old(h-1-y, x)
        # invert => old(x,y) = (y', h-1-x')
        return y, h - 1 - x
    elif case_id == 2: # rot180
        return w - 1 - x, h - 1 - y
    elif case_id == 3: # rot90 CCW
        # new(x',y') = old(y, w-1-x)
        # invert => old(x,y) = (w-1-y', x')
        return w - 1 - y, x
    elif case_id == 4: # flip vertical
        return x, h - 1 - y
    elif case_id == 5: # flip horizontal
        return w - 1 - x, y
    elif case_id == 6: # transpose
        # new(x',y') = old(y, x)
        return y, x
    elif case_id == 7: # rot90 CW + flip V
        # from your implementation: np.flipud(np.rot90(img, k=3))
        # equivalent inverse:
        return w - 1 - y, h - 1 - x
    else:
        raise ValueError(f"Unknown orientation case: {case_id}")

def ensure_vanilla_backup(path):
    path = Path(path)
    vanilla_path = path.with_name(f"{path.stem}_vanilla{path.suffix}")
    if path.exists() and (not vanilla_path.exists()):
        path.rename(vanilla_path)
    return vanilla_path

def restore_from_vanilla(path):
    path = Path(path)
    vanilla_path = path.with_name(f"{path.stem}_vanilla{path.suffix}")

    if not vanilla_path.exists():
        return False

    shutil.copy2(vanilla_path, path)
    return True

def redraw_point_on_image(base_path, x, y, color=(0, 255, 0), radius=4, is_mask=False):
    base_path = Path(base_path)
    vanilla_path = ensure_vanilla_backup(base_path)

    if not vanilla_path.exists():
        return

    if is_mask:
        img = cv2.imread(str(vanilla_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imread(str(vanilla_path), cv2.IMREAD_COLOR)
        if img is None:
            return

    h, w = img.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    cv2.circle(img, (x, y), int(radius), color, -1)
    cv2.imwrite(str(base_path), img)

def get_scaled_dot_radius(img_np, ref_patch=200, ref_radius=4):
    h, w = img_np.shape[:2]
    patch_size = max(h, w)
    r = int(round(ref_radius * patch_size / ref_patch))
    return max(2, r)


# ==========================================================
# Gallery UI
# ==========================================================
def show_nucleus_patch_in_memory(
        tile_id,
        nucleus_id,
        output_folder,
        run_dir,
        case_id,
        stage5_params,
        display_size=(256, 256),
        bg_color=(255, 255, 255)
):
    assert len(tile_id) == len(nucleus_id)
    run_dir = Path(run_dir)

    # ---------- Window setup ----------
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )
    root.title("STAGE 5: DAPI & H&E Patch Gallery")

    idx = [0]
    show_auto_centroids = [True]
    dapi_mode = ["raw"]

    def on_close():
        log_event(run_dir, "user_click_exit")
        log_event(run_dir, "system_ready_exit", exit_mode="window_close")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # ---------- Static widgets ----------
    tk.Label(root, text="DAPI", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="DAPI nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)

    dapi_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    dapi_label.grid(row=1, column=0, padx=8, pady=8)
    dapi_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # ---------- File and image helpers ----------
    manual_path = Path(output_folder) / "manual_centroids.json"
    dropped_dir = Path(output_folder) / "dropped_patch_pair"
    dropped_dir.mkdir(exist_ok=True)

    def resolve_path(tile_name, nucleus_name, kind, show_overlay):
        if kind == "dapi_img":
            if dapi_mode[0] == "raw":
                plain_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_raw.png"
                overlay_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_raw_overlay.png"
            else:
                plain_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_patch.png"
                overlay_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_patch_overlay.png"

            first = overlay_name if show_overlay else plain_name
            second = plain_name if show_overlay else overlay_name

            p1 = os.path.join(output_folder, first)
            if os.path.exists(p1):
                return p1
            return os.path.join(output_folder, second)

        overlay_map = {
            "dapi_mask": f"{tile_name}_nucleus_{nucleus_name}_dapi_mask_overlay.png",
            "he_img": f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png",
            "he_mask": f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png",
        }
        plain_map = {
            "dapi_mask": f"{tile_name}_nucleus_{nucleus_name}_dapi_mask.png",
            "he_img": f"{tile_name}_nucleus_{nucleus_name}_he_patch.png",
            "he_mask": f"{tile_name}_nucleus_{nucleus_name}_he_mask.png",
        }

        first = overlay_map[kind] if show_overlay else plain_map[kind]
        second = plain_map[kind] if show_overlay else overlay_map[kind]

        p1 = os.path.join(output_folder, first)
        if os.path.exists(p1):
            return p1
        return os.path.join(output_folder, second)

    def get_pair_png_files(tile_name, nucleus_name):
        prefix = f"{tile_name}_nucleus_{nucleus_name}_"
        return sorted([
            p for p in Path(output_folder).glob(f"{prefix}*.png")
            if p.is_file()
        ])

    def move_pair_png_files(tile_name, nucleus_name):
        moved = []
        for src in get_pair_png_files(tile_name, nucleus_name):
            dst = dropped_dir / src.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved.append(dst.name)
        return moved

    def pop_manual_entries(tile_name, nucleus_name):
        removed = []
        kept = []

        if manual_path.exists():
            with open(manual_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        kept.append(line)
                        continue

                    same_pair = (
                            rec.get("tile_id") == tile_name and
                            rec.get("nucleus_id") == nucleus_name
                    )

                    if same_pair:
                        removed.append(rec)
                    else:
                        kept.append(json.dumps(rec))

            with open(manual_path, "w") as f:
                for line in kept:
                    f.write(line + "\n")

        return removed

    def pop_global_entries(tile_name, nucleus_name):
        info_path = Path(output_folder) / "nuclei_centroids_global.json"
        if not info_path.exists():
            return None, []

        with open(info_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "data" in data:
            meta = data.get("meta", {})
            records = data["data"]
        else:
            meta = {}
            records = data

        removed = []
        kept = []

        for rec in records:
            same_pair = (
                    rec.get("tile_id") == tile_name and
                    rec.get("nucleus_id") == nucleus_name
            )
            if same_pair:
                removed.append(rec)
            else:
                kept.append(rec)

        out = {"meta": meta, "data": kept}
        with open(info_path, "w") as f:
            json.dump(out, f, indent=2)

        return meta, removed

    def save_dropped_pair_metadata(tile_name, nucleus_name, manual_removed, global_meta, global_removed, moved_pngs):
        out_path = dropped_dir / f"{tile_name}_nucleus_{nucleus_name}_dropped.json"
        out = {
            "tile_id": tile_name,
            "nucleus_id": nucleus_name,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "moved_png_files": moved_pngs,
            "manual_centroids": manual_removed,
            "nuclei_centroids_global": {
                "meta": global_meta if global_meta is not None else {},
                "data": global_removed,
            },
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return out_path

    def load_raw_numpy(path, is_mask=False, case_id=None, apply_case=False):
        if not os.path.exists(path):
            return None

        is_overlay = path.endswith("_overlay.png")

        if is_mask and not is_overlay:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read mask: {path}")
            if apply_case and case_id is not None:
                img = apply_orientation_to_patch(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if apply_case and case_id is not None:
                img = apply_orientation_to_patch(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_optional(path, is_mask=False, case_id=None):
        if os.path.exists(path):
            is_overlay = path.endswith("_overlay.png")

            if is_mask and not is_overlay:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read mask: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
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
            cv2.putText(img, text, (x, y), font, font_scale, (120, 120, 120), thickness)

        return pad_to_fixed_size(Image.fromarray(img))

    def pad_to_fixed_size(img_pil):
        img_pil = ImageOps.contain(img_pil, display_size)
        canvas = Image.new("RGB", display_size, bg_color)
        x = (display_size[0] - img_pil.width) // 2
        y = (display_size[1] - img_pil.height) // 2
        canvas.paste(img_pil, (x, y))
        return canvas

    # ---------- Manual centroid picking helpers ---------
    def upsert_manual_centroid(tile_name, nucleus_name, image_kind, x, y, extra=None):
        rec = {
            "tile_id": tile_name,
            "nucleus_id": nucleus_name,
            "image_kind": image_kind,
            "manual_centroid_xy": [int(x), int(y)],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            rec.update(extra)

        records = []
        if manual_path.exists():
            with open(manual_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        old = json.loads(line)
                    except Exception:
                        continue

                    same_key = (
                            old.get("tile_id") == tile_name and
                            old.get("nucleus_id") == nucleus_name and
                            old.get("image_kind") == image_kind
                    )
                    if not same_key:
                        records.append(old)

        records.append(rec)

        with open(manual_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def delete_manual_centroid(tile_name, nucleus_name, image_kind):
        if not manual_path.exists():
            return False

        kept = []
        removed = False

        with open(manual_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    kept.append(line)
                    continue

                same_pair = (
                        rec.get("tile_id") == tile_name and
                        rec.get("nucleus_id") == nucleus_name and
                        rec.get("image_kind") == image_kind
                )

                if same_pair:
                    removed = True
                    continue

                kept.append(json.dumps(rec))

        with open(manual_path, "w") as f:
            for line in kept:
                f.write(line + "\n")

        return removed

    def open_zoom_picker(img_np_rgb, title, tile_name, nucleus_name, image_kind, pixel_zoom=3):
        if img_np_rgb is None:
            messagebox.showwarning("No image", "Image not available.")
            return

        top = tk.Toplevel(root)
        top.withdraw()
        top.title(title)
        top.configure(bg="black")
        top.transient(root)

        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()

        H, W = img_np_rgb.shape[:2]
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)

        fit_scale = min(max_w / W, max_h / H)
        scale = min(float(pixel_zoom), float(fit_scale))
        scale = max(scale, 1.0)

        scale_int = int(np.floor(scale))
        scale = max(scale_int, 1)

        disp_w = int(W * scale)
        disp_h = int(H * scale)

        pil = Image.fromarray(img_np_rgb)
        if scale != 1.0:
            pil_disp = pil.resize((disp_w, disp_h), resample=Image.NEAREST)
        else:
            pil_disp = pil

        tk_img = ImageTk.PhotoImage(pil_disp)

        canvas = tk.Canvas(top, width=disp_w, height=disp_h, highlightthickness=0, bg="black", cursor="crosshair")
        canvas.pack(padx=10, pady=10)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        canvas._img_ref = tk_img
        top._scale = scale
        top._orig_W = W
        top._orig_H = H

        info = tk.Label(top, text="", fg="white", bg="black", font=("Helvetica", 14))
        info.pack(pady=(0, 10))

        picked = {"x": None, "y": None, "mark": None}

        def to_orig_xy(cx, cy):
            x = int(cx // scale)
            y = int(cy // scale)
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            return float(x), float(y)

        def on_move(ev):
            x, y = to_orig_xy(ev.x, ev.y)
            info.config(text=f"{image_kind} | tile={tile_name} nucleus={nucleus_name} | x={int(x)}, y={int(y)}")

        def on_click(ev):
            x, y = to_orig_xy(ev.x, ev.y)
            picked["x"], picked["y"] = x, y

            if picked["mark"] is not None:
                canvas.delete(picked["mark"])
            r = 10
            tag = "pick"
            canvas.delete(tag)
            canvas.create_line(ev.x - r, ev.y, ev.x + r, ev.y, fill="green", width=5, tags=tag)
            canvas.create_line(ev.x, ev.y - r, ev.x, ev.y + r, fill="green", width=5, tags=tag)
            picked["mark"] = tag

        def save_and_close(event=None):
            if picked["x"] is None:
                messagebox.showwarning("No point", "Click a point first.")
                return

            x_pick = int(round(picked["x"]))
            y_pick = int(round(picked["y"]))

            if image_kind == "dapi":
                x_pick_real, y_pick_real = invert_orientation_point(
                    x_pick, y_pick, W, H, case_id
                )
                x_pick_real = int(round(np.clip(x_pick_real, 0, W - 1)))
                y_pick_real = int(round(np.clip(y_pick_real, 0, H - 1)))
            else:
                x_pick_real, y_pick_real = x_pick, y_pick

            if image_kind == "dapi":
                coord_msg = f"x = {x_pick_real}, y = {y_pick_real} (saved in real patch coords after inverse orientation)"
            else:
                coord_msg = f"x = {x_pick_real}, y = {y_pick_real} (saved in real patch coords)"

            confirm = messagebox.askyesno(
                "Confirm manual centroid",
                f"Confirm this point for {image_kind}?\n\n{coord_msg}"
            )

            if not confirm:
                picked["x"], picked["y"] = None, None
                picked["mark"] = None
                canvas.delete("pick")
                info.config(
                    text=f"{image_kind} | tile={tile_name} nucleus={nucleus_name} | (click to pick)"
                )
                return

            upsert_manual_centroid(
                tile_name,
                nucleus_name,
                image_kind,
                x_pick_real,
                y_pick_real,
                extra={"scale_used": float(scale)}
            )

            update_manual_point_files(
                tile_name=tile_name,
                nucleus_name=nucleus_name,
                image_kind=image_kind,
                x_display=x_pick,
                y_display=y_pick,
                img_np_rgb=img_np_rgb,
            )

            top.destroy()
            update_images()

        def reset_pick(event=None):
            picked["x"], picked["y"] = None, None
            canvas.delete("pick")
            info.config(text=f"{image_kind} | tile={tile_name} nucleus={nucleus_name} | (click to pick)")

        def cancel(event=None):
            top.destroy()

        def delete_current_manual(event=None):
            confirm = messagebox.askyesno(
                "Delete manual centroid",
                f"Delete current manual centroid for:\n\n"
                f"{image_kind} | tile={tile_name} | nucleus={nucleus_name}?\n\n"
                f"This will restore the related image(s) from vanilla."
            )

            if not confirm:
                return

            delete_manual_centroid(tile_name, nucleus_name, image_kind)
            restore_manual_point_files(tile_name, nucleus_name, image_kind)

            top.destroy()
            update_images()

        def reopen(new_zoom):
            top.destroy()
            open_zoom_picker(img_np_rgb, title, tile_name, nucleus_name, image_kind, pixel_zoom=new_zoom)

        top.bind("1", lambda e: reopen(1))
        top.bind("2", lambda e: reopen(2))
        top.bind("3", lambda e: reopen(3))
        top.bind("4", lambda e: reopen(4))

        canvas.bind("<Motion>", on_move)
        canvas.bind("<Button-1>", on_click)
        top.bind("<Return>", save_and_close)
        top.bind("s", save_and_close)
        top.bind("r", reset_pick)
        top.bind("<Escape>", cancel)
        top.bind("<Delete>", delete_current_manual)
        top.bind("<BackSpace>", delete_current_manual)

        top.update_idletasks()

        win_w = top.winfo_reqwidth()
        win_h = top.winfo_reqheight()

        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()

        x0 = (screen_w - win_w) // 2
        y0 = (screen_h - win_h) // 2

        top.geometry(f"{win_w}x{win_h}+{x0}+{y0}")
        top.deiconify()
        top.lift()
        top.focus_force()

        hint = tk.Label(
            top,
            text="Click to pick centroid • Enter/S to save • Delete to remove manual point • R to reset • Esc to close",
            fg="white", bg="black", font=("Helvetica", 12)
        )
        hint.pack(pady=(0, 10))

    def update_manual_point_files(tile_name, nucleus_name, image_kind, x_display, y_display, img_np_rgb):
        r = get_scaled_dot_radius(img_np_rgb, ref_patch=200, ref_radius=4)

        dapi_up = int(stage5_params.get("dapi_mask_upscale_factor", 1))
        he_up = int(stage5_params.get("he_mask_upscale_factor", 1))

        if image_kind == "dapi":
            # use NON-upsampled DAPI image as the reference geometry
            ref_path = Path(output_folder) / f"{tile_name}_nucleus_{nucleus_name}_dapi_raw.png"
            if not ref_path.exists():
                ref_path = Path(output_folder) / f"{tile_name}_nucleus_{nucleus_name}_dapi_patch.png"

            ref_img = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
            if ref_img is None:
                return

            ref_h, ref_w = ref_img.shape[:2]

            # x,y were clicked on oriented/displayed DAPI image
            # first bring them back to original non-oriented DAPI coordinates
            x_base, y_base = invert_orientation_point(x_display, y_display, ref_w, ref_h, case_id)
            x_base = int(round(np.clip(x_base, 0, ref_w - 1)))
            y_base = int(round(np.clip(y_base, 0, ref_h - 1)))

            targets = [
                # non-mask files: use base coords directly
                (f"{tile_name}_nucleus_{nucleus_name}_dapi_raw.png", False, 1, x_base, y_base, r),
                (f"{tile_name}_nucleus_{nucleus_name}_dapi_raw_overlay.png", False, 1, x_base, y_base, r),
                (f"{tile_name}_nucleus_{nucleus_name}_dapi_patch.png", False, 1, x_base, y_base, r),
                (f"{tile_name}_nucleus_{nucleus_name}_dapi_patch_overlay.png", False, 1, x_base, y_base, r),

                # mask files: upscale AFTER inverse-orientation
                (f"{tile_name}_nucleus_{nucleus_name}_dapi_mask.png", True,
                 dapi_up, int(round(x_base * dapi_up)), int(round(y_base * dapi_up)), max(2, int(round(r * dapi_up)))),

                (f"{tile_name}_nucleus_{nucleus_name}_dapi_mask_overlay.png", False,
                 dapi_up, int(round(x_base * dapi_up)), int(round(y_base * dapi_up)), max(2, int(round(r * dapi_up)))),
            ]

        elif image_kind == "he":
            x_base = int(round(x_display))
            y_base = int(round(y_display))

            targets = [
                (f"{tile_name}_nucleus_{nucleus_name}_he_patch.png", False, 1, x_base, y_base, r),
                (f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png", False, 1, x_base, y_base, r),
                (f"{tile_name}_nucleus_{nucleus_name}_he_mask.png", True,
                 he_up, int(round(x_base * he_up)), int(round(y_base * he_up)), max(2, int(round(r * he_up)))),
                (f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png", False,
                 he_up, int(round(x_base * he_up)), int(round(y_base * he_up)), max(2, int(round(r * he_up)))),
            ]
        else:
            return

        for fname, is_mask, _, x_draw, y_draw, r_draw in targets:
            fpath = Path(output_folder) / fname
            if fpath.exists():
                redraw_point_on_image(
                    fpath,
                    x=x_draw,
                    y=y_draw,
                    color=(0, 255, 0),
                    radius=r_draw,
                    is_mask=is_mask
                )

    def restore_manual_point_files(tile_name, nucleus_name, image_kind):
        if image_kind == "dapi":
            targets = [
                f"{tile_name}_nucleus_{nucleus_name}_dapi_raw.png",
                f"{tile_name}_nucleus_{nucleus_name}_dapi_raw_overlay.png",
                f"{tile_name}_nucleus_{nucleus_name}_dapi_patch.png",
                f"{tile_name}_nucleus_{nucleus_name}_dapi_patch_overlay.png",
                f"{tile_name}_nucleus_{nucleus_name}_dapi_mask.png",
                f"{tile_name}_nucleus_{nucleus_name}_dapi_mask_overlay.png",
            ]
        elif image_kind == "he":
            targets = [
                f"{tile_name}_nucleus_{nucleus_name}_he_patch.png",
                f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png",
                f"{tile_name}_nucleus_{nucleus_name}_he_mask.png",
                f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png",
            ]
        else:
            return

        for fname in targets:
            restore_from_vanilla(Path(output_folder) / fname)

    # ---------- transformation helpers ---------
    def _tps_kernel(r2: np.ndarray) -> np.ndarray:
        out = np.zeros_like(r2, dtype=np.float64)
        mask = r2 > 0
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    def fit_tps(src: np.ndarray, dst: np.ndarray, reg: float = 1e-3):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)

        n = src.shape[0]
        if n < 3:
            raise ValueError("TPS needs at least 3 points.")
        if src.shape != dst.shape or src.shape[1] != 2:
            raise ValueError("src and dst must both be (N,2).")

        diff = src[:, None, :] - src[None, :, :]
        r2 = np.sum(diff * diff, axis=2)
        K = _tps_kernel(r2)

        P = np.concatenate([np.ones((n, 1)), src], axis=1)

        L = np.zeros((n + 3, n + 3), dtype=np.float64)
        L[:n, :n] = K + reg * np.eye(n)
        L[:n, n:] = P
        L[n:, :n] = P.T

        Y = np.zeros((n + 3, 2), dtype=np.float64)
        Y[:n, :] = dst

        params = np.linalg.solve(L, Y)
        w = params[:n, :]
        a = params[n:, :]

        return {"ctrl": src, "w": w, "a": a, "reg": float(reg)}

    def apply_tps(xy: np.ndarray, model: dict) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)

        ctrl = model["ctrl"]
        w = model["w"]
        a = model["a"]

        diff = xy[:, None, :] - ctrl[None, :, :]
        r2 = np.sum(diff * diff, axis=2)
        K = _tps_kernel(r2)

        P = np.concatenate([np.ones((xy.shape[0], 1)), xy], axis=1)

        return K @ w + P @ a

    def balanced_grid_sample_pairs(
        src,
        dst,
        grid_shape=(6, 6),
        max_per_cell=4,
        prefer="spread",
        score=None,
        use_dst_for_binning=False,
    ):
        src = np.asarray(src)
        dst = np.asarray(dst)
        pts = dst if use_dst_for_binning else src

        n = len(pts)
        if n == 0:
            return np.array([], dtype=int)

        gx, gy = int(grid_shape[0]), int(grid_shape[1])
        gx = max(gx, 1)
        gy = max(gy, 1)

        x = pts[:, 0]
        y = pts[:, 1]

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())

        xr = max(xmax - xmin, 1e-6)
        yr = max(ymax - ymin, 1e-6)

        xi = np.floor((x - xmin) / xr * gx).astype(int)
        yi = np.floor((y - ymin) / yr * gy).astype(int)

        xi = np.clip(xi, 0, gx - 1)
        yi = np.clip(yi, 0, gy - 1)

        cell_to_indices = {}
        for i in range(n):
            key = (xi[i], yi[i])
            cell_to_indices.setdefault(key, []).append(i)

        keep = []

        for key, inds in cell_to_indices.items():
            inds = np.array(inds, dtype=int)

            if len(inds) <= max_per_cell:
                keep.extend(inds.tolist())
                continue

            if prefer == "random":
                picked = np.random.choice(inds, size=max_per_cell, replace=False)

            elif prefer == "score":
                if score is None:
                    raise ValueError("prefer='score' requires score")
                cell_scores = np.asarray(score)[inds]
                order = np.argsort(-cell_scores)
                picked = inds[order[:max_per_cell]]

            else:
                cell_pts = pts[inds]
                picked_local = [0]
                remaining = list(range(1, len(cell_pts)))

                while len(picked_local) < max_per_cell and remaining:
                    chosen = cell_pts[picked_local]
                    cand = cell_pts[remaining]

                    d2 = ((cand[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
                    min_d2 = d2.min(axis=1)
                    best_j = int(np.argmax(min_d2))
                    picked_local.append(remaining[best_j])
                    remaining.pop(best_j)

                picked = inds[np.array(picked_local, dtype=int)]

            keep.extend(picked.tolist())

        keep = np.array(sorted(set(keep)), dtype=int)
        return keep

    def print_sampling_summary(src, keep_idx, name="balanced sample"):
        src = np.asarray(src)
        kept = src[keep_idx]
        print(f"[INFO] {name}: kept {len(keep_idx)}/{len(src)} pairs", flush=True)
        if len(kept) > 0:
            print(
                f"[INFO] {name}: kept x=[{kept[:,0].min():.1f}, {kept[:,0].max():.1f}] "
                f"y=[{kept[:,1].min():.1f}, {kept[:,1].max():.1f}]",
                flush=True
            )

    def calculate_alignment_transform(transform_type=None):
        if transform_type is None:
            transform_type = stage5_params["transform_mode"]
        balance_points_bool = stage5_params["balance_points_bool"]

        calc_t0 = time.time()
        log_event(run_dir, "user_click_calculate_transformation", transform_type=transform_type)
        try:
            info_path = Path(output_folder) / "nuclei_centroids_global.json"
            if not info_path.exists():
                messagebox.showerror("Missing file", f"Cannot find:\n{info_path}")
                return

            with open(info_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    nuclei_info = data["data"]
                    nuclei_meta = data.get("meta", {})
                else:
                    nuclei_info = data
                    nuclei_meta = {}

            manual_map = load_manual_centroid_map(output_folder)
            nuclei_info = replace_global_with_manual_if_any(nuclei_info, manual_map, nuclei_meta)

            transform_type_local = transform_type.lower().strip()
            if transform_type_local not in ("homography", "affine", "tps", "local_tps"):
                messagebox.showerror(
                    "Invalid transform_type",
                    "transform_type must be 'homography', 'affine', 'tps', or 'local_tps'"
                )
                return

            if len(nuclei_info) < 3 and transform_type_local == "affine":
                messagebox.showerror("Not enough points", "Need at least 3 nucleus pairs to estimate affine.")
                return
            if len(nuclei_info) < 4 and transform_type_local in ("homography", "tps", "local_tps"):
                messagebox.showerror("Not enough points",
                                     "Need at least 4 nucleus pairs to estimate homography / TPS init.")
                return

            src_all = np.array([x["dapi_centroid_global"] for x in nuclei_info], dtype=np.float32)
            dst_all = np.array([x["he_centroid_global"] for x in nuclei_info], dtype=np.float32)

            if balance_points_bool:
                keep_idx = balanced_grid_sample_pairs(
                    src_all,
                    dst_all,
                    grid_shape=(6, 6),
                    max_per_cell=4,
                    prefer="spread",
                    score=None,
                    use_dst_for_binning=False
                )
            else:
                keep_idx = np.arange(len(src_all), dtype=int)

            if transform_type_local == "affine" and len(keep_idx) < 3:
                keep_idx = np.arange(len(src_all), dtype=int)
            if transform_type_local in ("homography", "tps", "local_tps") and len(keep_idx) < 4:
                keep_idx = np.arange(len(src_all), dtype=int)

            sample_name = "grid-balanced sample" if balance_points_bool else "all-point sample"
            print_sampling_summary(src_all, keep_idx, name=sample_name)

            src = src_all[keep_idx]
            dst = dst_all[keep_idx]

            ransac_thr = 8.0
            maxIters = 10000
            confidence = 0.995

            H = None
            inliers = None
            tps_model = None
            tps_model_fwd = None
            tps_model_inv = None
            src_in = None
            dst_in = None

            if transform_type_local == "affine":
                A, inliers = cv2.estimateAffine2D(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence,
                    refineIters=10
                )
                if A is not None:
                    H = np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64)
                method_str = "grid-balanced cv2.estimateAffine2D(RANSAC) -> 3x3"

                if H is None:
                    messagebox.showerror(
                        "Affine RANSAC failed",
                        "Affine estimation returned None. Try adjusting threshold or check point order / outliers."
                    )
                    return

                src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
                proj = (H @ src_h.T).T
                proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
                err = np.linalg.norm(proj_xy - dst, axis=1)

            elif transform_type_local == "homography":
                H, inliers = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence
                )
                method_str = "grid-balanced cv2.findHomography(RANSAC)"

                if H is None:
                    messagebox.showerror(
                        "Homography RANSAC failed",
                        "Homography estimation returned None. Try adjusting threshold or check point order / outliers."
                    )
                    return

                src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
                proj = (H.astype(np.float64) @ src_h.T).T
                proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
                err = np.linalg.norm(proj_xy - dst, axis=1)

            else:
                H, inliers = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence
                )

                if transform_type_local == "tps":
                    method_str = "grid-balanced cv2.findHomography(RANSAC) + TPS (forward+inverse)"
                else:
                    method_str = "grid-balanced cv2.findHomography(RANSAC) + TPS control pairs (LOCAL_TPS-ready)"

                if H is None:
                    messagebox.showerror(
                        "Homography RANSAC failed",
                        "Homography estimation returned None. Cannot initialize TPS."
                    )
                    return

                mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(src), dtype=bool)
                src_in = src[mask].astype(np.float64)
                dst_in = dst[mask].astype(np.float64)

                if len(src_in) < 3:
                    messagebox.showerror(
                        "Not enough inliers",
                        f"Need at least 3 inliers for TPS, got {len(src_in)}."
                    )
                    return

                tps_reg = 1e-2
                tps_model_fwd = fit_tps(src_in, dst_in, reg=tps_reg)
                tps_model_inv = fit_tps(dst_in, src_in, reg=tps_reg)
                tps_model = tps_model_fwd

                proj_xy = apply_tps(src.astype(np.float64), tps_model_fwd)
                err = np.linalg.norm(proj_xy - dst.astype(np.float64), axis=1)

            inlier_count = int(inliers.sum()) if inliers is not None else 0
            total_balanced = len(src)
            total_raw = len(src_all)

            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                err_in = err[mask]
                med_err = float(np.median(err_in)) if len(err_in) else float("nan")
                mean_err = float(np.mean(err_in)) if len(err_in) else float("nan")
            else:
                med_err = float(np.median(err))
                mean_err = float(np.mean(err))

            out = {
                "from": "dapi_level0",
                "to": "he_level0",
                "method": method_str,
                "transform_type": transform_type_local,
                "sampling": {
                    "type": ("grid_balanced" if balance_points_bool else "all_points"),
                    "balance_points_bool": bool(balance_points_bool),
                    "grid_shape": [6, 6] if balance_points_bool else None,
                    "max_per_cell": 4 if balance_points_bool else None,
                    "num_points_before_sampling": int(total_raw),
                    "num_points_after_sampling": int(total_balanced),
                    "selected_indices_from_original": keep_idx.tolist(),
                },
                "ransacReprojThreshold": float(ransac_thr),
                "maxIters": int(maxIters),
                "confidence": float(confidence),
                "num_points": int(total_balanced),
                "num_inliers": int(inlier_count),
                "inlier_median_reproj_error_px": med_err,
                "inlier_mean_reproj_error_px": mean_err,
            }

            if transform_type_local in ("affine", "homography"):
                out["homography_3x3"] = H.tolist()
            else:
                out["initial_homography_3x3"] = H.tolist()
                out["tps_regularization"] = float(tps_reg)

                out["forward_tps"] = {
                    "from": "dapi_level0",
                    "to": "he_level0",
                    "control_points_src": tps_model_fwd["ctrl"].tolist(),
                    "control_points_dst": dst_in.tolist(),
                    "weights": tps_model_fwd["w"].tolist(),
                    "affine_2x3": tps_model_fwd["a"].T.tolist(),
                }

                out["inverse_tps"] = {
                    "from": "he_level0",
                    "to": "dapi_level0",
                    "control_points_src": tps_model_inv["ctrl"].tolist(),
                    "control_points_dst": src_in.tolist(),
                    "weights": tps_model_inv["w"].tolist(),
                    "affine_2x3": tps_model_inv["a"].T.tolist(),
                }

                if transform_type_local == "local_tps":
                    out["local_tps_ready"] = True
                    out["note"] = (
                        "This file stores forward/inverse TPS control pairs and global TPS parameters. "
                        "Downstream code may ignore stored weights and refit local TPS on neighborhoods at runtime."
                    )

            out_path = run_dir / "dapi_to_he_homography_level0.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            # also save 3x3 matrix as csv
            csv_path = run_dir / "dapi_to_he_homography_level0.csv"
            if "homography_3x3" in out:
                H_csv = np.array(out["homography_3x3"], dtype=np.float64)
                print("[INFO] Saving final homography_3x3 to csv", flush=True)
            elif "initial_homography_3x3" in out:
                H_csv = np.array(out["initial_homography_3x3"], dtype=np.float64)
                print("[WARN] Saving initial_homography_3x3 to csv", flush=True)
            else:
                H_csv = None
                print("[WARN] No 3x3 homography available for csv export", flush=True)

            if H_csv is not None:
                np.savetxt(csv_path, H_csv, delimiter=",", fmt="%.10f")

            used_points_path = run_dir / "nuclei_centroids_global_used_for_stage5.json"
            with open(used_points_path, "w") as f:
                json.dump({
                    "meta": nuclei_meta,
                    "manual_override_applied": True,
                    "data": nuclei_info
                }, f, indent=2)

            elapsed_sec = round(time.time() - calc_t0, 3)
            log_event(
                run_dir,
                "system_ready_calculate_transformation",
                transform_type=transform_type_local,
                output_file=str(out_path),
                num_points_before_sampling=int(total_raw),
                num_points_after_sampling=int(total_balanced),
                num_inliers=int(inlier_count),
                median_inlier_reproj_error_px=med_err,
                mean_inlier_reproj_error_px=mean_err,
                elapsed_sec=elapsed_sec,
            )

            if transform_type_local in ("affine", "homography"):
                msg = (
                    f"Saved:\n{out_path}\n{csv_path}\n\n"
                    f"Balance sample: {balance_points_bool}\n"
                    f"Transform mode: {transform_type_local}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Inliers: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"H (3x3):\n{H}"
                )
            elif transform_type_local == "tps":
                csv_note = "CSV contains the initial homography only; full TPS transform is stored in JSON."
                msg = (
                    f"Saved:\n{out_path}\n{csv_path}\n\n"
                    f"{csv_note}\n\n"
                    f"Balance sample: {balance_points_bool}\n"
                    f"Transform mode: {transform_type_local}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Homography inliers used for TPS: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"Initial H (3x3):\n{H}\n\n"
                    f"TPS control points: {len(tps_model['ctrl'])}"
                )
            else:
                csv_note = "CSV contains the initial homography only; full TPS transform is stored in JSON."
                msg = (
                    f"Saved:\n{out_path}\n{csv_path}\n\n"
                    f"{csv_note}\n\n"
                    f"Balance sample: {balance_points_bool}\n"
                    f"Transform mode: {transform_type_local}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Homography inliers used for LOCAL_TPS-ready control pairs: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"Initial H (3x3):\n{H}\n\n"
                    f"Control points saved: {len(tps_model['ctrl'])}"
                )

            messagebox.showinfo("Transform estimated", msg)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------- Gallery update callbacks ----------
    def update_images():
        i = idx[0]
        tile_name = tile_id[i]
        nucleus_name = nucleus_id[i]

        dapi_img_path = resolve_path(tile_name, nucleus_name, "dapi_img", show_auto_centroids[0])
        dapi_mask_path = resolve_path(tile_name, nucleus_name, "dapi_mask", show_auto_centroids[0])
        he_img_path = resolve_path(tile_name, nucleus_name, "he_img", show_auto_centroids[0])
        he_mask_path = resolve_path(tile_name, nucleus_name, "he_mask", show_auto_centroids[0])

        dapi_pil = load_optional(dapi_img_path, is_mask=False, case_id=case_id)
        dapi_mask_pil = load_optional(dapi_mask_path, is_mask=True, case_id=case_id)
        he_pil = load_optional(he_img_path, is_mask=False)
        he_mask_pil = load_optional(he_mask_path, is_mask=True)

        raw_dapi = load_raw_numpy(dapi_img_path, is_mask=False, case_id=case_id, apply_case=True)
        raw_dapi_mask = load_raw_numpy(dapi_mask_path, is_mask=True, case_id=case_id, apply_case=True)
        raw_he = load_raw_numpy(he_img_path, is_mask=False, apply_case=False)
        raw_he_mask = load_raw_numpy(he_mask_path, is_mask=True, apply_case=False)

        imgs = [
            ImageTk.PhotoImage(dapi_pil),
            ImageTk.PhotoImage(dapi_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]

        labels = [dapi_label, dapi_mask_label, he_label, he_mask_label]
        for lbl, im in zip(labels, imgs):
            lbl.config(image=im)
            lbl.image = im

        def bind_click(label_widget, raw_img, kind):
            label_widget.bind(
                "<Button-1>",
                lambda ev: open_zoom_picker(
                    raw_img,
                    title=f"{kind.upper()} | {tile_name} nucleus {nucleus_name}",
                    tile_name=tile_name,
                    nucleus_name=nucleus_name,
                    image_kind=kind,
                    pixel_zoom=10
                )
            )

        # only allow manual centroid picking on DAPI and H&E image panels
        bind_click(dapi_label, raw_dapi, "dapi")
        bind_click(he_label, raw_he, "he")

        # disable clicking on mask panels
        dapi_mask_label.unbind("<Button-1>")
        he_mask_label.unbind("<Button-1>")

        info_label.config(
            text=f"{i + 1}/{len(tile_id)} | {tile_name.split('_', 1)[0].capitalize()} | click only DAPI / H&E to adjust centroid"
        )

    def toggle_centroids():
        show_auto_centroids[0] = not show_auto_centroids[0]
        btn_toggle.config(text=("🙈 Hide auto centroids" if show_auto_centroids[0] else "👁️ Unhide auto centroids"))
        update_images()

    def toggle_dapi_mode():
        dapi_mode[0] = "raw" if dapi_mode[0] == "luted" else "luted"
        btn_dapi_mode.config(text=("🧬 DAPI: RAW" if dapi_mode[0] == "raw" else "🎨 DAPI: LUT"))
        update_images()

    def drop_current_patch_pair():
        i = idx[0]
        tile_name = tile_id[i]
        nucleus_name_cur = nucleus_id[i]

        confirm = messagebox.askyesno(
            "Drop patch pair",
            f"Drop this patch pair?\n\n"
            f"tile={tile_name}\n"
            f"nucleus={nucleus_name_cur}\n\n"
            f"This will:\n"
            f"1. move all related .png into dropped_patch_pair/\n"
            f"2. remove matching entries from manual_centroids.json\n"
            f"3. remove matching entries from nuclei_centroids_global.json"
        )
        if not confirm:
            return

        moved_pngs = move_pair_png_files(tile_name, nucleus_name_cur)
        manual_removed = pop_manual_entries(tile_name, nucleus_name_cur)
        global_meta, global_removed = pop_global_entries(tile_name, nucleus_name_cur)

        dropped_json_path = save_dropped_pair_metadata(
            tile_name=tile_name,
            nucleus_name=nucleus_name_cur,
            manual_removed=manual_removed,
            global_meta=global_meta,
            global_removed=global_removed,
            moved_pngs=moved_pngs,
        )

        # also remove from in-memory tile_id / nucleus_id
        del tile_id[i]
        del nucleus_id[i]

        if len(tile_id) == 0:
            messagebox.showinfo(
                "All patch pairs dropped",
                f"Dropped last patch pair.\n\nSaved metadata to:\n{dropped_json_path}"
            )
            root.destroy()
            return

        if idx[0] >= len(tile_id):
            idx[0] = len(tile_id) - 1

        update_images()

        messagebox.showinfo(
            "Patch pair dropped",
            f"Dropped patch pair:\n"
            f"{tile_name} nucleus {nucleus_name_cur}\n\n"
            f"Moved {len(moved_pngs)} png files.\n"
            f"Saved dropped metadata to:\n{dropped_json_path}"
        )

    def next_patch():
        idx[0] = (idx[0] + 1) % len(tile_id)
        update_images()

    def prev_patch():
        idx[0] = (idx[0] - 1) % len(tile_id)
        update_images()

    # ---------- Buttons and layout ----------
    btn_prev = ttk.Button(root, text="⟨ Previous", command=prev_patch)
    btn_next = ttk.Button(root, text="Next ⟩", command=next_patch)

    mid_btn_frame = ttk.Frame(root)
    mid_btn_frame.grid(row=3, column=1, columnspan=2, pady=(8, 8))

    btn_calc = ttk.Button(
        mid_btn_frame,
        text="🧮 Calculate alignment transform",
        style="Gallery.TButton",
        command=calculate_alignment_transform
    )

    btn_toggle = ttk.Button(
        mid_btn_frame,
        text="🙈 Hide auto centroids",
        style="Gallery.TButton",
        command=toggle_centroids
    )

    btn_dapi_mode = ttk.Button(
        mid_btn_frame,
        text="🎨 DAPI: LUT",
        style="Gallery.TButton",
        command=toggle_dapi_mode
    )

    btn_drop = ttk.Button(
        mid_btn_frame,
        text="🗑 Drop patch pair",
        style="Gallery.TButton",
        command=drop_current_patch_pair
    )

    btn_calc.grid(row=0, column=0, padx=6)
    btn_toggle.grid(row=0, column=1, padx=6)
    btn_dapi_mode.grid(row=0, column=2, padx=6)
    btn_drop.grid(row=0, column=3, padx=6)

    btn_prev.grid(row=3, column=0, sticky="w", padx=(20, 10), pady=(8, 8))
    btn_next.grid(row=3, column=3, sticky="e", padx=(10, 20), pady=(8, 8))

    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)

    # ---------- Start UI ----------
    update_images()
    log_event(run_dir, "system_ready_initial_start")
    root.mainloop()


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 5_nucleus_gallery.py <patches_folder>")

    run_dir = Path(sys.argv[1]).resolve()
    nuclei_dir = run_dir / "nuclei_patches"

    stage5_params = load_stage5_params(run_dir)
    print(
        "[INFO] Effective stage5 params: "
        f"transform_mode={stage5_params['transform_mode']}, "
        f"balance_points_bool={stage5_params['balance_points_bool']}, "
        f"dapi_mask_upscale_factor={stage5_params['dapi_mask_upscale_factor']}, "
        f"he_mask_upscale_factor={stage5_params['he_mask_upscale_factor']}",
        flush=True,
    )

    info_path = nuclei_dir / "nuclei_centroids_global.json"
    with open(info_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            nuclei_info = data["data"]
        else:
            nuclei_info = data
    with open(os.path.join(nuclei_dir, "../images_info.json"), "r") as f:
        case_id = json.load(f)['DAPI_orientation_case']


    tile_id = [x['tile_id'] for x in nuclei_info]
    nucleus_id = [x['nucleus_id'] for x in nuclei_info]

    show_nucleus_patch_in_memory(
        tile_id=tile_id,
        nucleus_id=nucleus_id,
        output_folder=nuclei_dir,
        run_dir=run_dir,
        case_id=case_id,
        stage5_params=stage5_params,
    )


if __name__ == "__main__":
    main()