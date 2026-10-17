"""Optional Stage-4c gallery images and their resume completion marker."""
import hashlib
import json
from pathlib import Path

from my_utils import qc_images_enabled
import run_roles

MANIFEST = "patch_images_manifest.json"


def patch_images_requested():
    # Overlays depend on the underlying patches; either group enables extraction.
    return qc_images_enabled("stage4c_patches") or qc_images_enabled("stage4c_overlays")


def centroid_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_images_ready(run_dir, overlays=None):
    folder = Path(run_dir) / "nuclei_patches"
    try:
        record = json.loads((folder / MANIFEST).read_text())
        if record["centroids_sha256"] != centroid_digest(folder / "nuclei_centroids_global.json"):
            return False
        if (qc_images_enabled("stage4c_overlays") if overlays is None else overlays) and not record["overlays"]:
            return False
        return all((folder / name).is_file() for name in record["files"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def patch_gallery_ready(run_dir):
    """QC can be reopened after manual edits/drops; require the remaining pairs' images."""
    folder = Path(run_dir) / "nuclei_patches"
    try:
        roles = run_roles.read(run_dir)
        table = json.loads((folder / "nuclei_centroids_global.json").read_text())
        if not table["data"]:
            return False
        if (folder / MANIFEST).exists():
            manifest = json.loads((folder / MANIFEST).read_text())
            source = manifest.get("standout_sha256")
            if source and source != centroid_digest(Path(run_dir) / "tiles/standout_nuclei.json"):
                return False
        for row in table["data"]:
            for side in (roles.moving, roles.fixed):
                prefix = f"{row['tile_id']}_nucleus_{row['nucleus_id']}_{side.token}"
                suffixes = ("patch", "raw", "mask") if side.modality == "dapi" else ("patch", "mask")
                if not all((folder / f"{prefix}_{s}.png").is_file() for s in suffixes):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
