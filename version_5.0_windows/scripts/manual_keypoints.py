"""Manual keypoint corrections shared by Stage-5 QC and the Stage-6 matrix step."""
from pathlib import Path
import json
import numpy as np

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

def replace_global_with_manual_if_any(nuclei_info, manual_map, meta, roles=None):
    """Apply patch-local edits on each role's grid, preserving the patch origin.

    The old dapi/he metadata slots are still accepted; new records and manual
    edits use the role token (e.g. hefg/hebg), never the modality alone.
    """
    sides = ((roles.moving.token, "dapi"), (roles.fixed.token, "he")) if roles else (
        ("dapi", "dapi"), ("he", "he"))
    updated = []
    for original in nuclei_info:
        rec = dict(original)
        for token, legacy_slot in sides:
            key = (rec.get("tile_id"), rec.get("nucleus_id"), token)
            if key not in manual_map:
                continue
            local_key, global_key = f"{token}_centroid_local", f"{token}_centroid_global"
            if rec.get(local_key) is None or rec.get(global_key) is None:
                continue
            level = int(meta.get(f"{token}_patch_level", meta.get(f"{legacy_slot}_patch_level", 0)))
            scale = float(2 ** level)
            old_local = np.asarray(rec[local_key], dtype=np.float64)
            old_global0 = np.asarray(rec[global_key], dtype=np.float64)
            new_local = np.asarray(manual_map[key], dtype=np.float64)
            # Keep operation order identical to the historical dapi-he path.
            patch_origin = old_global0 / scale - old_local
            new_global0 = (patch_origin + new_local) * scale
            rec[local_key] = new_local.tolist()
            rec[global_key] = new_global0.tolist()
            rec[f"{token}_manual_override"] = True
        updated.append(rec)
    return updated
