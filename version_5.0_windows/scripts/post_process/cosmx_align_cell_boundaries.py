#!/usr/bin/env python3
"""Map CosMx cell-boundary vertices from the stitched DAPI canvas to H&E level 0.

CosMx polygon exports use global stage-like coordinates rather than stitched-image
pixels.  This tool first applies the fixed-canvas convention used by the CosMx
stitcher::

    dapi_x = x_global_px - gx_min
    dapi_y = gy_max - y_global_px

It then applies the complete PHARAOH run transform.  Both matrix/nuclei-pair runs
and canonical JSONs carrying a dense local-residual field are supported.  The raw
columns and row order are preserved; transformed coordinates are appended as
``x_he_level0_px`` and ``y_he_level0_px`` by default.  Output is always Parquet.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pharaoh_transform import load_transform, resolve_run_transform, transform_xy
from xenium_align_cell_boundaries import (
    _TableWriter,
    _input_row_count,
    _iter_input_chunks,
    table_format,
)


DEFAULT_CHUNK_SIZE = 300_000
OUTPUT_COORDINATE_DTYPE = np.float32


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(
            f"boundary table is missing {column!r}; available columns: "
            + ", ".join(map(str, frame.columns))
        )
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"column {column!r} contains NaN or infinite values")
    return values


def canvas_origin_from_metadata(path: Path) -> tuple[int, int]:
    columns = [
        "fov", "CenterX_local_px", "CenterY_local_px",
        "CenterX_global_px", "CenterY_global_px",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["x0"] = frame["CenterX_global_px"] - frame["CenterX_local_px"]
    frame["y0"] = frame["CenterY_global_px"] + frame["CenterY_local_px"]
    anchors = frame.groupby("fov", sort=False)[["x0", "y0"]].median().round()
    if anchors.empty:
        raise ValueError(f"CosMx metadata contains no FOV anchors: {path}")
    return int(anchors["x0"].min()), int(anchors["y0"].max())


def infer_canvas_origin(
    path: Path,
    kind: str,
    chunk_size: int,
    *,
    fov_column: str,
    x_column: str,
    y_column: str,
    local_x_column: str,
    local_y_column: str,
) -> tuple[int, int]:
    """Infer one robust anchor per FOV directly from the polygon export."""
    per_fov: dict[int, list[tuple[float, float]]] = {}
    for frame in _iter_input_chunks(path, kind, chunk_size):
        required = (fov_column, x_column, y_column, local_x_column, local_y_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError("boundary table is missing: " + ", ".join(missing))
        fovs = pd.to_numeric(frame[fov_column], errors="raise")
        working = pd.DataFrame({
            "fov": fovs,
            "x0": _numeric(frame, x_column) - _numeric(frame, local_x_column),
            "y0": _numeric(frame, y_column) + _numeric(frame, local_y_column),
        })
        for fov, row in working.groupby("fov", sort=False)[["x0", "y0"]].median().iterrows():
            per_fov.setdefault(int(fov), []).append((float(row.x0), float(row.y0)))
    if not per_fov:
        raise ValueError("boundary table contains no FOV anchors")
    anchors = np.asarray(
        [np.median(np.asarray(values), axis=0) for values in per_fov.values()],
        dtype=np.float64,
    )
    return int(round(float(anchors[:, 0].min()))), int(round(float(anchors[:, 1].max())))


def _mapper_xy(mapper, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mapper, "transform_xy"):
        return mapper.transform_xy(x, y)
    return transform_xy(x, y, mapper)


def align_cosmx_cell_boundaries(
    input_path,
    output_path,
    transform_path=None,
    *,
    transform=None,
    transform_label=None,
    metadata_path=None,
    gx_min=None,
    gy_max=None,
    fov_column="fov",
    x_column="x_global_px",
    y_column="y_global_px",
    local_x_column="x_local_px",
    local_y_column="y_local_px",
    output_x_column="x_he_level0_px",
    output_y_column="y_he_level0_px",
    chunk_size=DEFAULT_CHUNK_SIZE,
    show_progress=True,
):
    """Align a CosMx polygon table while preserving all source columns and rows."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if (transform_path is None) == (transform is None):
        raise ValueError("provide exactly one of transform_path or transform")
    if not source.is_file():
        raise FileNotFoundError(f"boundary table not found: {source}")
    if source == destination:
        raise ValueError("input_path and output_path must be different files")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    input_kind = table_format(source)
    output_kind = table_format(destination)
    if output_kind != "parquet":
        raise ValueError("output_path must end in .parquet")

    if (gx_min is None) != (gy_max is None):
        raise ValueError("--gx-min and --gy-max must be supplied together")
    if gx_min is None:
        if metadata_path is not None:
            metadata = Path(metadata_path).expanduser().resolve()
            if not metadata.is_file():
                raise FileNotFoundError(f"CosMx metadata not found: {metadata}")
            gx_min, gy_max = canvas_origin_from_metadata(metadata)
        else:
            gx_min, gy_max = infer_canvas_origin(
                source, input_kind, chunk_size,
                fov_column=fov_column, x_column=x_column, y_column=y_column,
                local_x_column=local_x_column, local_y_column=local_y_column,
            )
    gx_min, gy_max = float(gx_min), float(gy_max)

    resolved = None if transform_path is None else Path(transform_path).expanduser().resolve()
    mapper = transform if transform is not None else load_transform(resolved)
    label = transform_label or str(resolved if resolved is not None else type(mapper).__name__)
    transform_type = getattr(mapper, "transform_type", type(mapper).__name__)
    print(f"[align] input: {source} ({input_kind})", flush=True)
    print(f"[align] output: {destination} (parquet)", flush=True)
    print(f"[align] transform: {label} ({transform_type})", flush=True)
    print(f"[align] CosMx canvas origin: gx_min={gx_min:g}, gy_max={gy_max:g}", flush=True)
    print("[align] direction: CosMx global px -> DAPI canvas level-0 px -> H&E level-0 px", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    writer = None
    total_rows = chunks = 0
    progress = tqdm(
        total=_input_row_count(source, input_kind), desc="Align CosMx boundaries",
        unit="vertices", unit_scale=True, dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        writer = _TableWriter(temporary, output_kind)
        for chunks, frame in enumerate(_iter_input_chunks(source, input_kind, chunk_size), 1):
            x = _numeric(frame, x_column) - gx_min
            y = gy_max - _numeric(frame, y_column)
            mapped_x, mapped_y = _mapper_xy(mapper, x, y)
            mapped_x = np.asarray(mapped_x, dtype=np.float64)
            mapped_y = np.asarray(mapped_y, dtype=np.float64)
            if mapped_x.shape != x.shape or mapped_y.shape != y.shape:
                raise ValueError("transform returned coordinate arrays with the wrong shape")
            if not np.isfinite(mapped_x).all() or not np.isfinite(mapped_y).all():
                bad = int(np.count_nonzero(~(np.isfinite(mapped_x) & np.isfinite(mapped_y))))
                raise ValueError(f"transform produced {bad:,} invalid vertices in chunk {chunks}")
            result = frame.copy()
            result[output_x_column] = mapped_x.astype(OUTPUT_COORDINATE_DTYPE)
            result[output_y_column] = mapped_y.astype(OUTPUT_COORDINATE_DTYPE)
            writer.write(result)
            total_rows += len(result)
            progress.update(len(result))
        if chunks == 0:
            raise ValueError("boundary table contains no rows")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        progress.close()
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    print(f"[align] DONE: {total_rows:,} vertices -> {destination}", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", "--input-path", required=True)
    parser.add_argument("--output_path", "--output-path", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--transform", "--matrix")
    group.add_argument("--run_dir", "--run-dir")
    parser.add_argument("--metadata", default=None, help="CosMx metadata CSV used for canvas origin")
    parser.add_argument("--gx_min", "--gx-min", type=float, default=None)
    parser.add_argument("--gy_max", "--gy-max", type=float, default=None)
    parser.add_argument("--fov_column", "--fov-column", default="fov")
    parser.add_argument("--x_column", "--x-column", default="x_global_px")
    parser.add_argument("--y_column", "--y-column", default="y_global_px")
    parser.add_argument("--local_x_column", "--local-x-column", default="x_local_px")
    parser.add_argument("--local_y_column", "--local-y-column", default="y_local_px")
    parser.add_argument("--output_x_column", "--output-x-column", default="x_he_level0_px")
    parser.add_argument("--output_y_column", "--output-y-column", default="y_he_level0_px")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args(argv)
    try:
        selected = resolve_run_transform(args.run_dir) if args.run_dir else args.transform
        align_cosmx_cell_boundaries(
            args.input_path, args.output_path, selected,
            metadata_path=args.metadata, gx_min=args.gx_min, gy_max=args.gy_max,
            fov_column=args.fov_column, x_column=args.x_column, y_column=args.y_column,
            local_x_column=args.local_x_column, local_y_column=args.local_y_column,
            output_x_column=args.output_x_column, output_y_column=args.output_y_column,
            chunk_size=args.chunk_size,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
