#!/usr/bin/env python3
"""Load and apply PHARAOH registration transforms.

The canonical ``*_homography_level0.json`` may contain either a single global
matrix or a global matrix plus a compact-support local residual field.  Legacy
post-processing code accepted only the sibling CSV and therefore silently lost
the local field.  This module gives post-processing scripts one common adapter:

* CSV/TXT -> a 3x3 moving-level0 to fixed-level0 matrix;
* JSON with ``transform_type=homography`` or ``affine`` -> the stored matrix;
* JSON with ``transform_type=local_residual`` -> the complete global + local map.

Forward coordinates always run from moving level-0 pixels to fixed level-0
pixels.  Image resampling uses ``inverse_points`` because each output pixel must
look up its source coordinate.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import local_residual  # noqa: E402


MATRIX_TYPES = frozenset({"homography", "affine"})
SUPPORTED_TYPES = MATRIX_TYPES | {local_residual.TRANSFORM_TYPE}
CANONICAL_RUN_TRANSFORMS = (
    "dapi_to_he_homography_level0.json",
    "dapi_to_he_homography_level0.csv",
)


def _validated_matrix(value, *, name="matrix"):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape 3x3 (or affine 2x3); got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"{name} is singular")
    return matrix


def _load_text_matrix(path):
    errors = []
    for delimiter in (",", None):
        try:
            return _validated_matrix(
                np.loadtxt(path, delimiter=delimiter, dtype=np.float64),
                name=str(path),
            )
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(
        "could not parse transform as a comma- or whitespace-delimited matrix: "
        f"{path}; {errors[-1]}"
    )


def _json_matrix(payload):
    for key in ("homography_3x3", "base_homography_3x3", "initial_homography_3x3"):
        if key in payload:
            return _validated_matrix(payload[key], name=key)
    if "affine_2x3" in payload:
        return _validated_matrix(payload["affine_2x3"], name="affine_2x3")
    raise ValueError("transform JSON contains no homography_3x3/base_homography_3x3 matrix")


@dataclass(frozen=True)
class RegistrationTransform:
    """One moving-level0 -> fixed-level0 PHARAOH transformation."""

    source_path: Path
    transform_type: str
    matrix: np.ndarray
    payload: dict | None = None

    @property
    def nonlinear(self):
        return self.transform_type == local_residual.TRANSFORM_TYPE

    def forward_points(self, points):
        """Map ``(..., 2)`` moving level-0 coordinates into fixed level 0."""
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1:] != (2,):
            raise ValueError(f"points must have final dimension 2; got {array.shape}")
        shape = array.shape
        flat = array.reshape(-1, 2)
        if self.nonlinear:
            mapped = local_residual.forward_points(flat, self.payload)
        else:
            mapped = local_residual.project_homography(self.matrix, flat)
        return mapped.reshape(shape)

    def inverse_points(self, points):
        """Map ``(..., 2)`` fixed level-0 coordinates back into moving level 0."""
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1:] != (2,):
            raise ValueError(f"points must have final dimension 2; got {array.shape}")
        shape = array.shape
        flat = array.reshape(-1, 2)
        if not len(flat):
            return flat.reshape(shape)
        if self.nonlinear:
            mapped = local_residual.inverse_points(
                flat, self._local_payload_for_fixed_points(flat)
            )
        else:
            mapped = local_residual.project_homography(np.linalg.inv(self.matrix), flat)
        return mapped.reshape(shape)

    def _local_payload_for_fixed_points(self, points):
        """Drop controls whose compact support cannot reach this query batch.

        Full-slide image warping evaluates one output tile at a time.  Restricting
        the residual model to nearby controls preserves the exact compact-support
        result while avoiding all-control work for every level-0 pixel.  The query
        bounds are expanded by both the kernel radius and the largest saved vector
        so the fixed-point inverse iterations cannot enter an omitted support.
        """
        block = self.payload["local_residual"]
        controls = np.asarray(
            block.get("control_points_fixed_level0", []), dtype=np.float64
        ).reshape(-1, 2)
        vectors = np.asarray(
            block.get("residual_vectors_fixed_level0", []), dtype=np.float64
        ).reshape(-1, 2)
        if not len(controls):
            return self.payload
        radius = float(block["support_radius_level0_px"])
        max_vector = float(np.linalg.norm(vectors, axis=1).max()) if len(vectors) else 0.0
        low = np.min(points, axis=0) - radius - max_vector
        high = np.max(points, axis=0) + radius + max_vector
        keep = np.all((controls >= low) & (controls <= high), axis=1)
        if np.all(keep):
            return self.payload
        local_block = dict(block)
        local_block["control_points_fixed_level0"] = controls[keep].tolist()
        local_block["residual_vectors_fixed_level0"] = vectors[keep].tolist()
        if "tile_ids" in local_block:
            tile_ids = list(local_block["tile_ids"])
            local_block["tile_ids"] = [value for value, use in zip(tile_ids, keep) if use]
        if "anchor_counts" in local_block:
            counts = list(local_block["anchor_counts"])
            local_block["anchor_counts"] = [value for value, use in zip(counts, keep) if use]
        payload = dict(self.payload)
        payload["local_residual"] = local_block
        return payload

    def transform_xy(self, x, y):
        """Map equal-shaped x/y arrays and return transformed x/y arrays."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"x and y must have identical shapes; got {x.shape} and {y.shape}")
        mapped = self.forward_points(np.stack([x, y], axis=-1))
        return mapped[..., 0], mapped[..., 1]


def load_transform(path):
    """Load a matrix CSV/TXT or a canonical PHARAOH transform JSON."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"transform not found: {source}")

    suffix = source.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return RegistrationTransform(source, "homography", _load_text_matrix(source))
    if suffix != ".json":
        raise ValueError("transform must be a .json, .csv, or .txt file")

    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("transform JSON must contain an object")
    transform_type = str(payload.get("transform_type", "")).strip().lower()
    if not transform_type:
        transform_type = (
            local_residual.TRANSFORM_TYPE
            if isinstance(payload.get("local_residual"), dict)
            else "homography"
        )
    if transform_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"unsupported transform_type={transform_type!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_TYPES))
        )
    matrix = _json_matrix(payload)
    if transform_type == local_residual.TRANSFORM_TYPE:
        # Validate the complete residual model now rather than halfway through a
        # multi-million-row coordinate table or a pyramidal image write.
        local_residual.forward_points(np.empty((0, 2), dtype=np.float64), payload)
    return RegistrationTransform(source, transform_type, matrix, payload)


def resolve_run_transform(run_dir):
    """Return the final level-0 transform exported by a PHARAOH run.

    The canonical JSON is preferred because a dense/local-residual run stores its
    non-linear field only there.  Matrix-only/nuclei-pair runs remain compatible
    with the CSV fallback.
    """
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"PHARAOH run directory not found: {directory}")
    for name in CANONICAL_RUN_TRANSFORMS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"PHARAOH run has no final level-0 transform: {directory}; expected "
        + " or ".join(CANONICAL_RUN_TRANSFORMS)
    )


def read_run_image_paths(run_dir):
    """Return the moving DAPI and fixed H&E paths recorded by a PHARAOH run."""
    directory = Path(run_dir).expanduser().resolve()
    info_path = directory / "images_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"PHARAOH images_info.json not found: {info_path}")
    payload = json.loads(info_path.read_text())
    roles = payload.get("roles") if isinstance(payload.get("roles"), dict) else {}
    moving = roles.get("moving") if isinstance(roles.get("moving"), dict) else {}
    fixed = roles.get("fixed") if isinstance(roles.get("fixed"), dict) else {}
    dapi = moving.get("path") or payload.get("DAPI_path")
    he = fixed.get("path") or payload.get("HE_path")
    if not dapi or not he:
        raise ValueError(f"PHARAOH run does not record both DAPI and H&E paths: {info_path}")
    return Path(dapi).expanduser(), Path(he).expanduser()


def load_matrix(path):
    """Load a matrix-only transform and reject JSONs containing a local field."""
    transform = load_transform(path)
    if transform.nonlinear:
        raise ValueError(
            f"{transform.source_path} contains a local residual field and cannot be "
            "reduced to one matrix without discarding the final refinement"
        )
    return transform.matrix


def transform_xy(x, y, transform):
    """Compatibility helper returning mapped x/y arrays.

    ``transform`` may be a :class:`RegistrationTransform` or an in-memory 3x3
    matrix, which keeps existing callers and small unit tests source-compatible.
    """
    if isinstance(transform, RegistrationTransform):
        return transform.transform_xy(x, y)
    matrix = _validated_matrix(transform)
    temporary = RegistrationTransform(Path("<in-memory>"), "homography", matrix)
    return temporary.transform_xy(x, y)
