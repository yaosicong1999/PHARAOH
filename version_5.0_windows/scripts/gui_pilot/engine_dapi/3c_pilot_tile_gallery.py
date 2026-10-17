"""Role-aware pilot gallery using the production Stage-4a masking workers."""
from pathlib import Path
import copy
import json
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from cli_common import load_module_from_path
from pilot_mask_parameters import apply_token_overrides
from nucleus_masks import side_params
import run_roles


def preview_masks(run_dir, pilot_dir, tile_id, preview=None):
    """Same parameters, level calibration, order and worker as Stage 4a."""
    run_dir, pilot_dir = Path(run_dir), Path(pilot_dir)
    engine = SCRIPTS / run_roles.read(run_dir).engine
    sys.path.insert(0, str(engine))
    stage = load_module_from_path("pilot_shared_stage4a", engine / "4a_generate_nuclei_masks.py")
    tiles = pilot_dir / "tiles"
    roles, sides = stage.resolve_sides(tiles)
    params = stage.load_stage4a_effective_params(run_dir / "tiles", pair=roles.params_block)
    if preview is not None:
        apply_token_overrides(params, preview)
    dense = stage.load_dense_detection_params(roles.params_block)
    levels = stage.stage3_tile_levels(tiles, sides)
    paths = {}
    for side in sides:  # DAPI first, so H&E recut sees its partner's mask.
        calibrated = dict(dense, _work_pixel_area_level0=float(2 ** levels[side.token]) ** 2)
        result = stage.side_worker(side, sides, copy.deepcopy(params), calibrated, {tile_id})(
            side.tile_file(tiles, tile_id))
        if not isinstance(result, dict) or not str(result.get("message", "")).startswith("Processed:"):
            raise RuntimeError(str(result))
        paths[side.role] = (Path(side.tile_file(tiles, tile_id)), Path(side.mask_file(tiles, tile_id)))
    return paths


def show_pilot_gallery(run_dir, pilot_dir):
    run_dir, pilot_dir = Path(run_dir), Path(pilot_dir)
    roles = run_roles.read(run_dir)
    engine = SCRIPTS / roles.engine
    sys.path.insert(0, str(engine))
    stage = load_module_from_path("pilot_gallery_stage4a", engine / "4a_generate_nuclei_masks.py")
    _, sides = stage.resolve_sides(pilot_dir / "tiles")
    tile_ids = sorted(set.intersection(*(set(s.tile_ids(pilot_dir / "tiles")) for s in sides)))
    if not tile_ids:
        raise ValueError("No paired pilot tiles. Run the pilot extractor first.")
    effective = stage.load_stage4a_effective_params(run_dir / "tiles", pair=roles.params_block)
    root = tk.Tk()
    root.title(f"Stage 3 pilot — {roles.pair.name}")
    panels, variables = {}, {}
    names = {"dapi": "DAPI", "he": "H&E"}
    for col, image in enumerate((roles.moving, roles.fixed)):
        frame = ttk.LabelFrame(root, text=f"{image.role.capitalize()} ({names[image.modality]})")
        frame.grid(row=0, column=col, padx=8, pady=8)
        panels[image.role] = [ttk.Label(frame), ttk.Label(frame)]
        for j, panel in enumerate(panels[image.role]):
            panel.grid(row=0, column=j)
        settings = side_params(effective, effective.get("_raw_block", {}), image.modality, image.token)
        key = "thr_offset" if image.modality == "dapi" else "mask_intensity_threshold"
        variable = tk.StringVar(value=str(settings[key]))
        variables[image.token] = (key, variable)
        label = "Otsu offset" if image.modality == "dapi" else "Hematoxylin threshold"
        ttk.Label(frame, text=label).grid(row=1, column=0)
        ttk.Spinbox(frame, textvariable=variable, width=10,
                    from_=-100 if image.modality == "dapi" else 0,
                    to=100 if image.modality == "dapi" else 1,
                    increment=5 if image.modality == "dapi" else 0.01).grid(row=1, column=1)
        if image.modality == "he":
            ttk.Label(frame, text=f"Mask source: {settings['mask_source']} (Stage 4a)").grid(row=2, columnspan=2)
    index = [0]
    status = ttk.Label(root)
    status.grid(row=1, column=0, columnspan=2)

    def values():
        preview = {"sides": {token: {key: float(var.get())}
                              for token, (key, var) in variables.items()}}
        apply_token_overrides({}, preview)  # Validate before either preview or save.
        return preview

    def refresh():
        try:
            paths = preview_masks(run_dir, pilot_dir, tile_ids[index[0]], values())
            for role, pair_paths in paths.items():
                for panel, path in zip(panels[role], pair_paths):
                    with Image.open(path) as raw:
                        display = ImageOps.pad(raw.convert("RGB"), (320, 320), color="white")
                    photo = ImageTk.PhotoImage(display)
                    panel.configure(image=photo)
                    panel.image = photo
            status.configure(text=f"{index[0] + 1}/{len(tile_ids)} · {tile_ids[index[0]]} · Preview only; Save applies to later masking")
        except Exception as exc:
            messagebox.showerror("Pilot preview failed", str(exc))

    def navigate(delta):
        index[0] = (index[0] + delta) % len(tile_ids)
        refresh()

    def save():
        try:
            preview = values()
            path = pilot_dir / "pilot_output_parameters.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            # Canonical side overrides supersede legacy global pilot thresholds.
            data["mask_preview"] = preview
            path.write_text(json.dumps(data, indent=2))
            status.configure(text="Saved. Rerun masking to apply to existing outputs.")
        except Exception as exc:
            messagebox.showerror("Cannot save pilot parameters", str(exc))

    controls = ttk.Frame(root)
    controls.grid(row=2, column=0, columnspan=2, pady=8)
    for col, (label, callback) in enumerate((
            ("Previous", lambda: navigate(-1)), ("Next", lambda: navigate(1)),
            ("Preview", refresh), ("Save masking parameters", save))):
        ttk.Button(controls, text=label, command=callback).grid(row=0, column=col, padx=5)
    refresh()
    root.mainloop()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python 3c_pilot_tile_gallery.py <RUN_DIR> <PILOT_DIR>")
    show_pilot_gallery(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
