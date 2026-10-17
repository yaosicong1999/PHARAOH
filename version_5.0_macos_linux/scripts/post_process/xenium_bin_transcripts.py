#!/usr/bin/env python3
"""Bin H&E-level0 transcript coordinates into a bin-by-gene Parquet table.

The input coordinates are expected in raw H&E level-0 pixels. A bin can be
defined either in pixels at the target MPP or directly in micrometers:

    bin_um = bin_size_target_px * target_mpp
    raw_bin_px = bin_um / raw_mpp

The output locations contain both raw H&E and target-MPP pixel coordinates.
"""

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm


DEFAULT_CHUNK_SIZE = 300_000
DEFAULT_PARQUET_ROW_CHUNK_SIZE = 2_000
DEFAULT_KEY_REDUCE_GROUP = 16
LOCATION_DTYPE = np.float32
LOCATION_DECIMALS = 3
BAD_PREFIXES = frozenset(
    {
        "BLANK",
        "NegControlCodeword",
        "NegControlProbe",
        "UnassignedCodeword",
        "DeprecatedCodeword",
    }
)
_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def table_format(path):
    name = Path(path).name.lower()
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".csv"):
        return "csv"
    raise ValueError(
        f"unsupported input extension for {path}; expected .parquet, .csv, or .csv.gz"
    )


def resolve_bin_geometry(
    raw_mpp,
    target_mpp,
    *,
    bin_size_target_px=None,
    bin_size_um=None,
):
    """Return normalized physical, raw-pixel, and target-pixel bin sizes."""
    raw_mpp = _positive(raw_mpp, "raw_mpp")
    target_mpp = _positive(target_mpp, "target_mpp")
    supplied = int(bin_size_target_px is not None) + int(bin_size_um is not None)
    if supplied != 1:
        raise ValueError(
            "provide exactly one of bin_size_target_px or bin_size_um"
        )
    if bin_size_target_px is not None:
        target_px = _positive(bin_size_target_px, "bin_size_target_px")
        physical_um = target_px * target_mpp
        definition = "target_pixels"
    else:
        physical_um = _positive(bin_size_um, "bin_size_um")
        target_px = physical_um / target_mpp
        definition = "micrometers"
    return {
        "definition": definition,
        "raw_mpp": raw_mpp,
        "target_mpp": target_mpp,
        "bin_size_um": physical_um,
        "bin_size_raw_px": physical_um / raw_mpp,
        "bin_size_target_px": target_px,
        "raw_to_target_scale": raw_mpp / target_mpp,
    }


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Transcript binning requires pyarrow for disk-backed aggregation. "
            "Install pyarrow in the active Python environment."
        ) from exc
    return pq


def _available_columns(path, kind):
    if kind == "parquet":
        pq = _require_pyarrow()
        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0).columns)


def _iter_transcript_chunks(path, kind, columns, chunk_size):
    if kind == "parquet":
        pq = _require_pyarrow()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
        return
    yield from pd.read_csv(path, usecols=columns, chunksize=chunk_size)


def _input_row_count(path, kind):
    """Return an exact Parquet row count and a line-based CSV row count."""
    if kind == "parquet":
        pq = _require_pyarrow()
        return int(pq.ParquetFile(path).metadata.num_rows)
    opener = gzip.open if kind == "csv.gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _gene_mask(frame):
    """Prefer Xenium's is_gene flag, then also exclude known controls."""
    if "is_gene" in frame.columns:
        values = frame["is_gene"]
        if pd.api.types.is_bool_dtype(values.dtype):
            keep = values.fillna(False).to_numpy(dtype=bool)
        else:
            keep = (
                values.astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
                .to_numpy()
            )
    else:
        keep = np.ones(len(frame), dtype=bool)
    feature = frame["feature_name"].map(
        lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
    )
    prefixes = feature.str.split("_", n=1).str[0]
    return keep & ~prefixes.isin(BAD_PREFIXES).to_numpy(), feature


def aggregate_chunk(frame, raw_bin_px, x_column, y_column):
    """Filter one transcript chunk and aggregate count by bin and gene."""
    try:
        x = pd.to_numeric(frame[x_column], errors="raise").to_numpy(dtype=np.float64)
        y = pd.to_numeric(frame[y_column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"columns {x_column!r}/{y_column!r} must contain numeric coordinates"
        ) from exc
    finite = np.isfinite(x) & np.isfinite(y)
    gene_keep, feature = _gene_mask(frame)
    keep = finite & gene_keep
    if not np.any(keep):
        return pd.DataFrame(
            {
                "bin_x": pd.Series(dtype=np.int32),
                "bin_y": pd.Series(dtype=np.int32),
                "feature_name": pd.Series(dtype=str),
                "count": pd.Series(dtype=np.int32),
            }
        ), int(np.count_nonzero(keep))

    bin_x64 = np.floor(x[keep] / raw_bin_px).astype(np.int64)
    bin_y64 = np.floor(y[keep] / raw_bin_px).astype(np.int64)
    if (
        np.any(bin_x64 < _INT32_MIN)
        or np.any(bin_x64 > _INT32_MAX)
        or np.any(bin_y64 < _INT32_MIN)
        or np.any(bin_y64 > _INT32_MAX)
    ):
        raise ValueError("computed bin index exceeds signed int32 range")

    local = pd.DataFrame(
        {
            "bin_x": bin_x64.astype(np.int32),
            "bin_y": bin_y64.astype(np.int32),
            "feature_name": feature.loc[keep].to_numpy(),
        }
    )
    aggregated = (
        local.groupby(["bin_x", "bin_y", "feature_name"], sort=False)
        .size()
        .reset_index(name="count")
    )
    aggregated["count"] = aggregated["count"].astype(np.int32)
    return aggregated, int(np.count_nonzero(keep))


def _pack_bin_keys(bin_x, bin_y):
    """Pack two signed int32 bin indices into sortable uint64 keys."""
    x = np.asarray(bin_x, dtype=np.int64)
    y = np.asarray(bin_y, dtype=np.int64)
    x_offset = (x - _INT32_MIN).astype(np.uint64)
    y_offset = (y - _INT32_MIN).astype(np.uint64)
    return (x_offset << np.uint64(32)) | y_offset


def _unpack_bin_keys(keys):
    keys = np.asarray(keys, dtype=np.uint64)
    x = (keys >> np.uint64(32)).astype(np.int64) + _INT32_MIN
    y = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64) + _INT32_MIN
    return x.astype(np.int32), y.astype(np.int32)


def _reduce_key_files(key_files, work_dir, group_size=DEFAULT_KEY_REDUCE_GROUP):
    """Disk-backed staged union of per-chunk unique bin keys."""
    current = list(key_files)
    stage = 0
    while len(current) > 1:
        next_files = []
        for group_index, start in enumerate(range(0, len(current), group_size)):
            group = current[start : start + group_size]
            arrays = [np.load(path, mmap_mode="r") for path in group]
            merged = np.unique(np.concatenate(arrays))
            out = work_dir / f"keys_stage{stage + 1}_{group_index:05d}.npy"
            np.save(out, merged)
            next_files.append(out)
        current = next_files
        stage += 1
    return current[0]


def _atomic_replace(staged, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)


def _write_counts_parquet(
    counts,
    bin_ids,
    gene_names,
    output_path,
    *,
    row_chunk_size=DEFAULT_PARQUET_ROW_CHUNK_SIZE,
    show_progress=True,
):
    """Write CSR rows as a dense, wide Parquet table indexed by bin_id."""
    if counts.shape != (len(bin_ids), len(gene_names)):
        raise ValueError(
            "count matrix dimensions do not match bin IDs and gene names: "
            f"{counts.shape} versus ({len(bin_ids)}, {len(gene_names)})"
        )
    if not isinstance(row_chunk_size, int) or row_chunk_size <= 0:
        raise ValueError("parquet_row_chunk_size must be a positive integer")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Writing cnts.parquet requires pyarrow") from exc

    max_count = int(counts.data.max()) if counts.nnz else 0
    if max_count <= np.iinfo(np.uint8).max:
        count_dtype = np.dtype(np.uint8)
    elif max_count <= np.iinfo(np.uint16).max:
        count_dtype = np.dtype(np.uint16)
    else:
        count_dtype = np.dtype(np.uint32)

    writer = None
    progress = tqdm(
        total=counts.shape[0],
        desc="Write cnts.parquet",
        unit="bins",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        starts = range(0, counts.shape[0], row_chunk_size)
        if counts.shape[0] == 0:
            starts = [0]
        for start in starts:
            end = min(start + row_chunk_size, counts.shape[0])
            block = counts[start:end].toarray().astype(count_dtype, copy=False)
            frame = pd.DataFrame(
                block,
                columns=gene_names,
                index=pd.Index(bin_ids[start:end], name="bin_id"),
            )
            table = pa.Table.from_pandas(frame, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="brotli",
                    compression_level=9,
                )
            writer.write_table(table)
            progress.update(end - start)
    finally:
        progress.close()
        if writer is not None:
            writer.close()
    return count_dtype.name


def _validate_count_location_alignment(
    count_path, locs_path, expected_bin_ids, batch_size=100_000
):
    """Re-read both final formats and prove their bin IDs align row-for-row."""
    locs_offset = 0
    for frame in pd.read_csv(
        locs_path, sep="\t", usecols=["bin_id"], chunksize=batch_size
    ):
        observed = frame["bin_id"].astype(str).to_numpy()
        expected = np.asarray(
            expected_bin_ids[locs_offset : locs_offset + len(observed)], dtype=str
        )
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                f"written locs.tsv bin_id order differs at row {locs_offset}"
            )
        locs_offset += len(observed)
    if locs_offset != len(expected_bin_ids):
        raise RuntimeError("written locs.tsv has the wrong number of rows")

    pq = _require_pyarrow()
    parquet = pq.ParquetFile(count_path)
    if parquet.metadata.num_rows != len(expected_bin_ids):
        raise RuntimeError(
            "cnts.parquet/locs.tsv row mismatch: "
            f"{parquet.metadata.num_rows} versus {len(expected_bin_ids)}"
        )
    offset = 0
    for batch in parquet.iter_batches(columns=["bin_id"], batch_size=batch_size):
        observed = np.asarray(batch.column(0).to_pylist(), dtype=object)
        expected = np.asarray(expected_bin_ids[offset : offset + len(observed)], dtype=object)
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                f"cnts.parquet and locs.tsv bin_id order differs at row {offset}"
            )
        offset += len(observed)
    if offset != len(expected_bin_ids):
        raise RuntimeError("cnts.parquet bin_id validation ended at the wrong row")


def bin_transcripts(
    input_path,
    output_dir,
    raw_mpp,
    target_mpp,
    *,
    bin_size_target_px=None,
    bin_size_um=None,
    x_column="x_location",
    y_column="y_location",
    min_transcripts_per_bin=1,
    chunk_size=DEFAULT_CHUNK_SIZE,
    parquet_row_chunk_size=DEFAULT_PARQUET_ROW_CHUNK_SIZE,
    show_progress=True,
):
    """Bin an aligned transcript table and return its output paths."""
    source = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    kind = table_format(source)
    geometry = resolve_bin_geometry(
        raw_mpp,
        target_mpp,
        bin_size_target_px=bin_size_target_px,
        bin_size_um=bin_size_um,
    )
    if not source.is_file():
        raise FileNotFoundError(f"input transcript table not found: {source}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(parquet_row_chunk_size, int) or parquet_row_chunk_size <= 0:
        raise ValueError("parquet_row_chunk_size must be a positive integer")
    if not isinstance(min_transcripts_per_bin, int) or min_transcripts_per_bin < 1:
        raise ValueError("min_transcripts_per_bin must be an integer >= 1")
    if not x_column or not y_column or x_column == y_column:
        raise ValueError("x_column and y_column must be distinct, non-empty names")
    if "feature_name" in {x_column, y_column}:
        raise ValueError("coordinate columns cannot also be feature_name")

    _require_pyarrow()
    available = _available_columns(source, kind)
    required = [x_column, y_column, "feature_name"]
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(
            f"transcript table is missing required column(s): {', '.join(missing)}; "
            f"available columns: {', '.join(available)}"
        )
    read_columns = required + (["is_gene"] if "is_gene" in available else [])

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "counts": output_dir / "cnts.parquet",
        "locations": output_dir / "locs.tsv",
        "locations_full": output_dir / "locs_full_table.tsv",
        "genes": output_dir / "genes.tsv",
        "metadata": output_dir / "binning.json",
    }

    print(f"[bin] input: {source} ({kind})", flush=True)
    print(
        f"[bin] raw MPP={geometry['raw_mpp']:.8g}, "
        f"target MPP={geometry['target_mpp']:.8g}",
        flush=True,
    )
    print(
        f"[bin] bin edge={geometry['bin_size_um']:.8g} um = "
        f"{geometry['bin_size_raw_px']:.8g} raw px = "
        f"{geometry['bin_size_target_px']:.8g} target px",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="transcript_binning_", dir=output_dir) as tmp:
        work_dir = Path(tmp)
        agg_files = []
        key_files = []
        genes = set()
        total_input_rows = 0
        total_gene_rows = 0
        total_agg_rows = 0

        input_progress = tqdm(
            total=_input_row_count(source, kind),
            desc="Aggregate transcripts",
            unit="rows",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        try:
            for chunk_index, frame in enumerate(
                _iter_transcript_chunks(source, kind, read_columns, chunk_size)
            ):
                aggregated, kept = aggregate_chunk(
                    frame,
                    geometry["bin_size_raw_px"],
                    x_column,
                    y_column,
                )
                total_input_rows += len(frame)
                total_gene_rows += kept
                total_agg_rows += len(aggregated)
                genes.update(aggregated["feature_name"].astype(str).unique())

                agg_path = work_dir / f"agg_{chunk_index:05d}.parquet"
                aggregated.to_parquet(agg_path, index=False, compression="brotli")
                agg_files.append(agg_path)
                keys = np.unique(
                    _pack_bin_keys(aggregated["bin_x"], aggregated["bin_y"])
                )
                key_path = work_dir / f"keys_{chunk_index:05d}.npy"
                np.save(key_path, keys)
                key_files.append(key_path)
                input_progress.set_postfix(
                    genes=len(genes), local_entries=f"{total_agg_rows:,}"
                )
                input_progress.update(len(frame))
        finally:
            input_progress.close()

        if not agg_files:
            raise ValueError("input transcript table contains no readable chunks")
        if total_gene_rows == 0 or not genes:
            raise ValueError("no gene transcripts remain after filtering")

        final_key_file = _reduce_key_files(key_files, work_dir)
        bin_keys = np.load(final_key_file, mmap_mode="r")
        gene_names = sorted(genes)
        gene_to_col = {gene: index for index, gene in enumerate(gene_names)}
        print(
            f"[bin] indexing {len(bin_keys):,} occupied bins and "
            f"{len(gene_names):,} genes ...",
            flush=True,
        )

        rows_path = work_dir / "coo_rows.dat"
        cols_path = work_dir / "coo_cols.dat"
        vals_path = work_dir / "coo_values.dat"
        rows = np.memmap(rows_path, mode="w+", dtype=np.int64, shape=(total_agg_rows,))
        cols = np.memmap(cols_path, mode="w+", dtype=np.int32, shape=(total_agg_rows,))
        values = np.memmap(vals_path, mode="w+", dtype=np.int32, shape=(total_agg_rows,))
        offset = 0
        index_progress = tqdm(
            total=total_agg_rows,
            desc="Build sparse matrix",
            unit="entries",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        try:
            for agg_path in agg_files:
                aggregated = pd.read_parquet(agg_path)
                end = offset + len(aggregated)
                keys = _pack_bin_keys(aggregated["bin_x"], aggregated["bin_y"])
                mapped_rows = np.searchsorted(bin_keys, keys)
                if np.any(mapped_rows >= len(bin_keys)) or not np.array_equal(
                    np.asarray(bin_keys[mapped_rows]), keys
                ):
                    raise RuntimeError("internal bin-index mapping failed")
                mapped_cols = aggregated["feature_name"].map(gene_to_col)
                if mapped_cols.isna().any():
                    raise RuntimeError("internal gene-index mapping failed")
                rows[offset:end] = mapped_rows
                cols[offset:end] = mapped_cols.to_numpy(dtype=np.int32)
                values[offset:end] = aggregated["count"].to_numpy(dtype=np.int32)
                index_progress.update(len(aggregated))
                offset = end
        finally:
            index_progress.close()
        rows.flush()
        cols.flush()
        values.flush()

        counts = coo_matrix(
            (values, (rows, cols)),
            shape=(len(bin_keys), len(gene_names)),
            dtype=np.int32,
        ).tocsr()
        counts.sum_duplicates()
        totals = np.asarray(counts.sum(axis=1)).ravel()
        keep_bins = totals >= min_transcripts_per_bin
        if not np.all(keep_bins):
            counts = counts[keep_bins]
            selected_keys = np.asarray(bin_keys)[keep_bins]
        else:
            selected_keys = np.asarray(bin_keys)

        bin_x, bin_y = _unpack_bin_keys(selected_keys)
        raw_bin_px = geometry["bin_size_raw_px"]
        target_bin_px = geometry["bin_size_target_px"]
        locations = pd.DataFrame(
            {
                "bin_x": bin_x,
                "bin_y": bin_y,
                "x_raw": ((bin_x.astype(np.float64) + 0.5) * raw_bin_px).astype(
                    LOCATION_DTYPE
                ),
                "y_raw": ((bin_y.astype(np.float64) + 0.5) * raw_bin_px).astype(
                    LOCATION_DTYPE
                ),
                "x_target": (
                    (bin_x.astype(np.float64) + 0.5) * target_bin_px
                ).astype(LOCATION_DTYPE),
                "y_target": (
                    (bin_y.astype(np.float64) + 0.5) * target_bin_px
                ).astype(LOCATION_DTYPE),
            }
        )
        locations["x"] = locations["x_target"]
        locations["y"] = locations["y_target"]
        locations.index = (
            locations["bin_x"].astype(str) + "_" + locations["bin_y"].astype(str)
        )
        locations.index.name = "bin_id"
        bin_ids = locations.index.astype(str).to_numpy()
        locs = locations.reset_index()[["bin_id", "x", "y"]]
        locations_full = locations.reset_index()
        if not np.array_equal(locs["bin_id"].to_numpy(), bin_ids):
            raise RuntimeError("internal locs.tsv bin_id order changed unexpectedly")
        if counts.shape[0] != len(locs):
            raise RuntimeError(
                f"count/location row mismatch before writing: {counts.shape[0]} "
                f"versus {len(locs)}"
            )
        genes_frame = pd.DataFrame({"gene_name": gene_names})
        genes_frame.index.name = "gene_index"

        metadata = {
            "input_path": str(source),
            "input_format": kind,
            "input_coordinate_space": "he_level0_pixels",
            "x_column": x_column,
            "y_column": y_column,
            **geometry,
            "min_transcripts_per_bin": min_transcripts_per_bin,
            "input_rows": total_input_rows,
            "gene_transcript_rows": total_gene_rows,
            "filtered_rows": total_input_rows - total_gene_rows,
            "n_bins": int(counts.shape[0]),
            "n_genes": int(counts.shape[1]),
            "nonzero_bin_gene_entries": int(counts.nnz),
            "count_location_rows_aligned": True,
            "coordinate_dtype": np.dtype(LOCATION_DTYPE).name,
            "location_tsv_decimal_places": LOCATION_DECIMALS,
            "outputs": {name: str(path) for name, path in final_paths.items()},
            "location_columns": {
                "locs.tsv": ["bin_id", "x", "y"],
                "x_raw_y_raw": "raw H&E level-0 pixels",
                "x_target_y_target": "pixels at target_mpp",
                "x_y": "aliases of x_target/y_target",
            },
        }

        staged_counts = work_dir / final_paths["counts"].name
        staged_locations = work_dir / final_paths["locations"].name
        staged_locations_full = work_dir / final_paths["locations_full"].name
        staged_genes = work_dir / final_paths["genes"].name
        staged_metadata = work_dir / final_paths["metadata"].name
        count_dtype = _write_counts_parquet(
            counts,
            bin_ids,
            gene_names,
            staged_counts,
            row_chunk_size=parquet_row_chunk_size,
            show_progress=show_progress,
        )
        metadata["count_dtype"] = count_dtype
        locs.to_csv(
            staged_locations,
            sep="\t",
            index=False,
            float_format=f"%.{LOCATION_DECIMALS}f",
        )
        locations_full.to_csv(
            staged_locations_full,
            sep="\t",
            index=False,
            float_format=f"%.{LOCATION_DECIMALS}f",
        )
        genes_frame.to_csv(staged_genes, sep="\t")
        _validate_count_location_alignment(staged_counts, staged_locations, bin_ids)
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, staged in (
            ("counts", staged_counts),
            ("locations", staged_locations),
            ("locations_full", staged_locations_full),
            ("genes", staged_genes),
            ("metadata", staged_metadata),
        ):
            _atomic_replace(staged, final_paths[name])

    print(
        f"[bin] DONE: matrix={counts.shape[0]:,} bins x "
        f"{counts.shape[1]:,} genes, nnz={counts.nnz:,}",
        flush=True,
    )
    for name, path in final_paths.items():
        print(f"[bin] {name}: {path}", flush=True)
    return final_paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Bin H&E-level0 transcript coordinates at a physical or "
            "target-pixel bin size."
        )
    )
    parser.add_argument("--input_path", "--input-path", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--raw_mpp", "--raw-mpp", required=True, type=float)
    parser.add_argument("--target_mpp", "--target-mpp", required=True, type=float)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument(
        "--bin_size_target_px", "--bin-size-target-px", type=float
    )
    size.add_argument("--bin_size_um", "--bin-size-um", type=float)
    parser.add_argument("--x_column", "--x-column", default="x_location")
    parser.add_argument("--y_column", "--y-column", default="y_location")
    parser.add_argument(
        "--min_transcripts_per_bin",
        "--min-transcripts-per-bin",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--parquet_row_chunk_size",
        "--parquet-row-chunk-size",
        type=int,
        default=DEFAULT_PARQUET_ROW_CHUNK_SIZE,
        help=(
            "bins densified per cnts.parquet write batch "
            f"(default: {DEFAULT_PARQUET_ROW_CHUNK_SIZE:,})"
        ),
    )
    args = parser.parse_args(argv)
    try:
        bin_transcripts(
            args.input_path,
            args.output_dir,
            args.raw_mpp,
            args.target_mpp,
            bin_size_target_px=args.bin_size_target_px,
            bin_size_um=args.bin_size_um,
            x_column=args.x_column,
            y_column=args.y_column,
            min_transcripts_per_bin=args.min_transcripts_per_bin,
            chunk_size=args.chunk_size,
            parquet_row_chunk_size=args.parquet_row_chunk_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
