"""Pilot extraction: select existing Stage-3 points and reuse the CLI extractor."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from cli_stage3 import make_headless_app
from run_artifacts import initial_alignment_path, INITIAL_ALIGNMENT
import run_roles


def farthest_point_subset(points_xy, k, start="centroid", seed=43):
    pts = np.asarray(points_xy, dtype=np.float64)
    n = len(pts)
    if not n or k < 1:
        raise ValueError("Pilot requires sampled points and pilot_k >= 1")
    if k >= n:
        return np.arange(n, dtype=int)
    first = (int(np.random.default_rng(seed).integers(n)) if start == "random"
             else int(np.argmin(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1))))
    selected = [first]
    distance = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(1, k):
        distance[selected] = -1
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(pts - pts[index], axis=1))
    return np.asarray(selected, dtype=int)


def extract_pilot(run_dir):
    run_dir = Path(run_dir).resolve()
    roles = run_roles.read(run_dir)
    sampled = json.loads((run_dir / "sampled_points.json").read_text())
    points = np.asarray(sampled["points_xy"], dtype=np.int32)
    pilot_dir = run_dir / "pilot_tiles"
    pilot_dir.mkdir(exist_ok=True)
    # Isolated context: never rewrite the full run's sample list or tile metadata.
    shutil.copy2(run_dir / "images_info.json", pilot_dir / "images_info.json")
    shutil.copy2(initial_alignment_path(run_dir), pilot_dir / INITIAL_ALIGNMENT)
    m3, app = make_headless_app(SCRIPTS / roles.engine, pilot_dir)
    indices = farthest_point_subset(points, int(app.stage3.get("pilot_k", 10)))
    sampling_level = int(sampled.get(f"{roles.moving.token}_level", sampled.get("dapi_level", roles.moving.level)))
    tile_level = int(sampled.get(f"{roles.moving.token}_tile_level", sampled.get("dapi_tile_level", 1)))
    tile_size = float(sampled["tile_size_sampling"])
    subset = dict(sampled, points_xy=points[indices].tolist())
    (pilot_dir / "sampled_points.json").write_text(json.dumps(subset, indent=2))
    app.current_points_xy = points[indices]
    app.current_tiles = m3.centroids_to_tiles(points[indices], tile_size)
    app.level_selection_tiles = m3.centroids_to_tiles(points, tile_size)
    saved_fixed_level = sampled.get(f"{roles.fixed.token}_tile_level", sampled.get("he_tile_level"))
    if saved_fixed_level is not None:
        app.extraction_fixed_level = int(saved_fixed_level)
    app.current_dapi_tile_level = tile_level
    app.current_dapi_level = sampling_level
    app.current_tile_size = tile_size
    app.current_tile_size_at_extract = tile_size * (2.0 ** (sampling_level - tile_level))
    app.target_tile_mpp = float(app.stage3.get("target_tile_mpp", 0.425))
    from my_utils import inspect_image_pyramid
    app.dapi_base_mpp = m3.scalar_mpp(inspect_image_pyramid(roles.moving.path).get("base_mpp"))
    app.dapi_base_mpp_assumed = app.dapi_base_mpp is None
    app.has_sampling_outputs = True
    app.on_extract_clicked()
    for image in (roles.moving, roles.fixed):
        if not (pilot_dir / "tiles" / f"{image.token}_tile_info.json").exists():
            raise RuntimeError(f"Pilot extraction did not produce {image.token} tiles")
    (pilot_dir / "pilot_selection.json").write_text(json.dumps({
        "pair": roles.pair.name, "run_dir": str(run_dir),
        "selected_indices": indices.tolist(),
        "selected_points_xy": points[indices].tolist(),
        "tile_size_sampling": tile_size,
        "moving_level": sampling_level, "moving_tile_level": tile_level,
    }, indent=2))
    return pilot_dir


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python 3b_tile_pilot.py <RUN_DIR> [--no-gallery]")
    run_dir = Path(sys.argv[1]).resolve()
    pilot_dir = extract_pilot(run_dir)
    if "--no-gallery" not in sys.argv[2:]:
        subprocess.run([sys.executable, str(Path(__file__).with_name("3c_pilot_tile_gallery.py")),
                        str(run_dir), str(pilot_dir)], check=True)


if __name__ == "__main__":
    main()
