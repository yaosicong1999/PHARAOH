#!/usr/bin/env python3
"""Transform Xenium transcript coordinates from micrometers to H&E level 0.

Input may be the run's own ``transcripts.zarr.zip`` (or an unzipped ``.zarr``), or a
``.parquet``/``.csv``/``.csv.gz`` table carrying ``x_location`` and ``y_location``.
A table's other columns are preserved; a Zarr store is read from its level-0 grid --
the deeper levels are subsampled views of the same transcripts and reading one would
silently drop most of them -- and yields the columns the Xenium ``transcripts.parquet``
export uses, so both inputs give the same output.

**Output is always Parquet.** These tables run to tens of millions of rows, and one
format downstream is worth more than the choice.

The PHARAOH transform may be read from the canonical JSON, including a local
residual field, or from a legacy headerless 3x3 CSV/TXT matrix. It consumes DAPI
level-0 pixels, whereas Xenium transcript coordinates are in micrometers, so the
coordinates are divided by ``dapi_mpp`` (default: 0.2125 um/px) first.

For 10x's own ``*_he_imagealignment.csv`` use ``xenium_align_10x_transcripts.py`` instead:
that matrix runs the other way and is not in the same units.
"""

import argparse
import gzip
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pharaoh_transform import load_matrix as load_homography
from pharaoh_transform import load_transform, resolve_run_transform, transform_xy


DEFAULT_CHUNK_SIZE = 300_000
DEFAULT_DAPI_MPP = 0.2125
OUTPUT_COORDINATE_DTYPE = np.float32


def table_format(path):
    """Return ``zarr``, ``parquet``, ``csv``, or ``csv.gz`` based on the full suffix."""
    name = Path(path).name.lower()
    if name.endswith(".zarr.zip") or name.endswith(".zarr"):
        return "zarr"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".csv"):
        return "csv"
    raise ValueError(
        f"unsupported table extension for {path}; expected .zarr.zip, .zarr, "
        ".parquet, .csv, or .csv.gz"
    )


def transform_chunk(
    frame,
    matrix,
    x_column="x_location",
    y_column="y_location",
    dapi_mpp=DEFAULT_DAPI_MPP,
):
    """Convert micron coordinates to DAPI pixels, then apply the homography."""
    missing = [column for column in (x_column, y_column) if column not in frame.columns]
    if missing:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"transcript table is missing required column(s): {', '.join(missing)}; "
            f"available columns: {available}"
        )

    try:
        x = pd.to_numeric(frame[x_column], errors="raise").to_numpy(dtype=np.float64)
        y = pd.to_numeric(frame[y_column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"columns {x_column!r} and {y_column!r} must contain numeric coordinates"
        ) from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            f"columns {x_column!r} and {y_column!r} contain NaN or infinite values"
        )

    dapi_mpp = float(dapi_mpp)
    if not np.isfinite(dapi_mpp) or dapi_mpp <= 0:
        raise ValueError("dapi_mpp must be a positive finite number")
    x = x / dapi_mpp
    y = y / dapi_mpp

    if hasattr(matrix, "transform_xy"):
        transformed_x, transformed_y = matrix.transform_xy(x, y)
    else:
        transformed_x, transformed_y = transform_xy(x, y, matrix)
    result = frame.copy()
    # Homography math stays float64; persisted coordinates use float32. Around
    # 100k-pixel WSI coordinates this retains sub-pixel precision while halving
    # coordinate-column storage relative to float64.
    result[x_column] = transformed_x.astype(OUTPUT_COORDINATE_DTYPE)
    result[y_column] = transformed_y.astype(OUTPUT_COORDINATE_DTYPE)
    return result


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Parquet input/output requires pyarrow in the active Python environment. "
            "Install pyarrow or use CSV/CSV.GZ."
        ) from exc
    return pa, pq


def _require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "Reading transcripts.zarr.zip requires zarr in the active Python "
            "environment. Install zarr, or export the table to Parquet first."
        ) from exc
    return zarr


def _zarr_root(path):
    """Open the Xenium transcript store and check it is the one we think it is."""
    zarr = _require_zarr()
    root = zarr.open(str(path), mode="r")
    if "grids" not in root:
        raise ValueError(f"{path} has no `grids` group; is this a transcripts store?")
    units = str(root.attrs.get("spatial_units", "")).lower()
    if units and units != "micron":
        raise ValueError(
            f"{path} reports spatial_units={units!r}; this reader assumes microns"
        )
    return root


def _iter_zarr_chunks(path, chunk_size):
    """Yield the level-0 grid tiles as frames, batched to roughly `chunk_size` rows.

    Level 0 is the full set; the deeper levels are progressively subsampled views of
    the same transcripts and would silently under-count if read instead. Column names
    match the Xenium `transcripts.parquet` export so a store and an exported table
    give the same output.
    """
    root = _zarr_root(path)
    genes = list(root.attrs.get("gene_names", []))
    tiles = sorted(root["grids"]["0"])
    pending, pending_rows = [], 0
    for name in tiles:
        tile = root["grids"]["0"][name]
        location = np.asarray(tile["location"])
        if location.size == 0:
            continue
        gene_index = np.asarray(tile["gene_identity"]).reshape(-1)
        frame = pd.DataFrame({
            "x_location": location[:, 0].astype(np.float64),
            "y_location": location[:, 1].astype(np.float64),
            "z_location": location[:, 2].astype(np.float32),
            "feature_name": ([genes[i] if 0 <= i < len(genes) else "" for i in gene_index]
                             if genes else gene_index),
            "qv": np.asarray(tile["quality_score"]).reshape(-1).astype(np.float32),
            "valid": np.asarray(tile["valid"]).reshape(-1).astype(np.uint8),
            "codeword_index": np.asarray(tile["codeword_identity"])[:, 0].astype(np.uint16),
        })
        pending.append(frame)
        pending_rows += len(frame)
        if pending_rows >= chunk_size:
            yield pd.concat(pending, ignore_index=True)
            pending, pending_rows = [], 0
    if pending:
        yield pd.concat(pending, ignore_index=True)


def _iter_input_chunks(path, kind, chunk_size):
    if kind == "zarr":
        yield from _iter_zarr_chunks(path, chunk_size)
        return

    if kind == "parquet":
        _, pq = _require_pyarrow()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return

    yield from pd.read_csv(path, chunksize=chunk_size)


def _input_row_count(path, kind):
    """Return an exact Parquet row count and a line-based CSV row count."""
    if kind == "zarr":
        root = _zarr_root(path)
        total = root.attrs.get("number_rnas")
        if total:
            return int(total)
        return sum(int(root["grids"]["0"][name]["location"].shape[0])
                   for name in root["grids"]["0"])
    if kind == "parquet":
        _, pq = _require_pyarrow()
        return int(pq.ParquetFile(path).metadata.num_rows)
    opener = gzip.open if kind == "csv.gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


class _TableWriter:
    """Incrementally write one of the supported output formats."""

    def __init__(self, path, kind):
        self.path = Path(path)
        self.kind = kind
        self.rows = 0
        self.columns = None
        self._handle = None
        self._parquet_writer = None
        self._pa = None
        if kind == "csv":
            self._handle = open(self.path, "w", encoding="utf-8", newline="")
        elif kind == "csv.gz":
            self._handle = gzip.open(self.path, "wt", encoding="utf-8", newline="")
        else:
            self._pa, pq = _require_pyarrow()
            self._pq = pq

    def write(self, frame):
        columns = list(frame.columns)
        if self.columns is None:
            self.columns = columns
        elif columns != self.columns:
            raise ValueError("input columns changed between chunks")

        if self.kind in {"csv", "csv.gz"}:
            frame.to_csv(self._handle, index=False, header=(self.rows == 0))
        else:
            table = self._pa.Table.from_pandas(frame, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = self._pq.ParquetWriter(
                    self.path,
                    table.schema,
                    compression="brotli",
                    compression_level=9,
                )
            elif table.schema != self._parquet_writer.schema:
                try:
                    table = table.cast(self._parquet_writer.schema)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "column types changed between chunks and cannot be written "
                        "to one Parquet file"
                    ) from exc
            self._parquet_writer.write_table(table)
        self.rows += len(frame)

    def close(self):
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        if self._handle is not None:
            self._handle.close()


def align_transcripts(
    input_path,
    output_path,
    matrix_path=None,
    *,
    transform=None,
    transform_label=None,
    x_column="x_location",
    y_column="y_location",
    dapi_mpp=DEFAULT_DAPI_MPP,
    chunk_size=DEFAULT_CHUNK_SIZE,
    show_progress=True,
):
    """Align a transcript table and return the resolved output path."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if (matrix_path is None) == (transform is None):
        raise ValueError("provide exactly one of matrix_path or transform")
    resolved_matrix_path = (
        None if matrix_path is None else Path(matrix_path).expanduser().resolve()
    )
    input_kind = table_format(source)
    output_kind = table_format(destination)
    if output_kind != "parquet":
        raise ValueError(
            f"output must be .parquet, got {destination.name}; these tables run to "
            "tens of millions of rows and one downstream format is worth more than "
            "the choice"
        )

    if not source.exists():
        raise FileNotFoundError(f"input not found: {source}")
    if input_kind != "zarr" and not source.is_file():
        raise FileNotFoundError(f"input table not found: {source}")
    if source == destination:
        raise ValueError("input_path and output_path must be different files")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not x_column or not y_column or x_column == y_column:
        raise ValueError("x_column and y_column must be distinct, non-empty names")
    dapi_mpp = float(dapi_mpp)
    if not np.isfinite(dapi_mpp) or dapi_mpp <= 0:
        raise ValueError("dapi_mpp must be a positive finite number")

    matrix = transform if transform is not None else load_transform(resolved_matrix_path)
    label = transform_label or str(
        resolved_matrix_path if resolved_matrix_path is not None else type(matrix).__name__
    )
    print(f"[align] input:  {source} ({input_kind})", flush=True)
    print(f"[align] output: {destination} (parquet)", flush=True)
    transform_type = getattr(matrix, "transform_type", type(matrix).__name__)
    print(f"[align] transform: {label} ({transform_type})", flush=True)
    print(
        f"[align] input units: um; DAPI MPP={dapi_mpp:.8g} um/px",
        flush=True,
    )
    print(
        "[align] direction: Xenium um -> DAPI level-0 px -> H&E level-0 px",
        flush=True,
    )
    print(f"[align] coordinate columns: {x_column}, {y_column}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if output_kind == "csv.gz" else destination.suffix
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=suffix, dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    writer = None
    total_rows = 0
    chunk_count = 0
    progress = tqdm(
        total=_input_row_count(source, input_kind),
        desc="Align transcripts",
        unit="rows",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        writer = _TableWriter(temporary, output_kind)
        for chunk_count, frame in enumerate(
            _iter_input_chunks(source, input_kind, chunk_size), start=1
        ):
            try:
                aligned = transform_chunk(
                    frame,
                    matrix,
                    x_column,
                    y_column,
                    dapi_mpp=dapi_mpp,
                )
            except ValueError as exc:
                raise ValueError(f"input chunk {chunk_count}: {exc}") from exc
            writer.write(aligned)
            total_rows += len(aligned)
            progress.update(len(aligned))
        if chunk_count == 0:
            raise ValueError("input table contains no readable chunks")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        progress.close()
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()

    print(
        f"[align] DONE: {total_rows:,} rows -> {destination} "
        f"({destination.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert Xenium micron coordinates to DAPI level-0 pixels, then "
            "apply a DAPI-level0 to H&E-level0 PHARAOH transform."
        )
    )
    parser.add_argument("--input_path", "--input-path", required=True)
    parser.add_argument("--output_path", "--output-path", required=True)
    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument(
        "--matrix", "--transform",
        help="PHARAOH transform JSON, or a headerless 3x3 matrix in CSV/TXT format",
    )
    transform_group.add_argument(
        "--run_dir", "--run-dir",
        help=(
            "completed PHARAOH run; automatically selects the final level-0 JSON "
            "or matrix-only CSV fallback"
        ),
    )
    parser.add_argument("--x_column", "--x-column", default="x_location")
    parser.add_argument("--y_column", "--y-column", default="y_location")
    parser.add_argument(
        "--dapi_mpp",
        "--dapi-mpp",
        type=float,
        default=DEFAULT_DAPI_MPP,
        help=(
            "DAPI level-0 micrometers per pixel used to convert Xenium micron "
            f"coordinates (default: {DEFAULT_DAPI_MPP}; use 1 for pixel input)"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"rows per processing chunk (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    args = parser.parse_args(argv)
    try:
        transform_path = (
            resolve_run_transform(args.run_dir) if args.run_dir else args.matrix
        )
        align_transcripts(
            args.input_path,
            args.output_path,
            transform_path,
            x_column=args.x_column,
            y_column=args.y_column,
            dapi_mpp=args.dapi_mpp,
            chunk_size=args.chunk_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
