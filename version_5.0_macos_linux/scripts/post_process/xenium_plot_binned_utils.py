#!/usr/bin/env python3
"""Shared readers and plotting helpers for Xenium binned Parquet outputs."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CLIP_Q = (0.0, 99.0)
DEFAULT_DPI = 150
DEFAULT_BATCH_SIZE = 2_000
DEFAULT_MAX_GRID_CELLS = 50_000_000
DEFAULT_OVERLAY_ALPHA = 0.65


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Reading cnts.parquet requires pyarrow in the active Python environment"
        ) from exc
    return pq


def resolve_inputs(input_path, locs_path=None):
    """Validate a count Parquet and its compact sibling location table."""
    counts = Path(input_path).expanduser().resolve()
    if counts.suffix.lower() != ".parquet":
        raise ValueError(f"--input must end in .parquet: {counts}")
    if not counts.is_file():
        raise FileNotFoundError(f"count table not found: {counts}")

    locations = (
        Path(locs_path).expanduser().resolve()
        if locs_path is not None
        else counts.parent / "locs.tsv"
    )
    if not locations.is_file():
        raise FileNotFoundError(
            f"location table not found: {locations}; place locs.tsv beside the "
            "count table or provide --locs"
        )
    return counts, locations


def parquet_gene_names(count_path):
    """Return count columns in their physical Parquet order."""
    pq = _require_pyarrow()
    parquet = pq.ParquetFile(count_path)
    names = list(parquet.schema_arrow.names)
    if "bin_id" not in names:
        raise ValueError(
            "count Parquet has no bin_id field; expected cnts.parquet from "
            "xenium_bin_transcripts.py"
        )
    genes = [name for name in names if name != "bin_id"]
    if not genes:
        raise ValueError("count Parquet contains no gene columns")
    return genes


def read_locations(locs_path):
    """Read only the compact bin ID and target-pixel coordinates."""
    header = pd.read_csv(locs_path, sep="\t", nrows=0)
    required = ["bin_id", "x", "y"]
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError(
            f"location table is missing column(s): {', '.join(missing)}; "
            "expected bin_id, x, y"
        )
    locs = pd.read_csv(
        locs_path,
        sep="\t",
        usecols=required,
        dtype={"bin_id": "string", "x": np.float32, "y": np.float32},
    )
    if locs.empty:
        raise ValueError("location table contains no bins")
    if locs["bin_id"].isna().any() or not locs["bin_id"].is_unique:
        raise ValueError("locs.tsv bin_id values must be present and unique")
    if not np.isfinite(locs[["x", "y"]].to_numpy()).all():
        raise ValueError("locs.tsv x/y coordinates must be finite numbers")
    return locs


def _validate_batch_ids(batch, expected_ids, offset):
    index = batch.schema.get_field_index("bin_id")
    if index < 0:
        raise ValueError("count Parquet batch has no bin_id field")
    observed = np.asarray(batch.column(index).to_pylist(), dtype=object)
    expected = expected_ids[offset : offset + len(observed)]
    if not np.array_equal(observed, expected):
        raise ValueError(
            "cnts.parquet and locs.tsv bin_id order differs at row "
            f"{offset}; regenerate the binning outputs together"
        )


def _validate_row_counts(parquet, locs):
    count_rows = int(parquet.metadata.num_rows)
    if count_rows != len(locs):
        raise ValueError(
            "cnts.parquet and locs.tsv row counts differ: "
            f"{count_rows:,} versus {len(locs):,}"
        )


def read_gene_counts(
    count_path,
    locs,
    genes,
    *,
    batch_size=DEFAULT_BATCH_SIZE,
    show_progress=True,
):
    """Read selected genes in batches and validate every bin ID in order."""
    pq = _require_pyarrow()
    parquet = pq.ParquetFile(count_path)
    _validate_row_counts(parquet, locs)
    expected_ids = locs["bin_id"].astype(str).to_numpy(dtype=object)
    parts = {gene: [] for gene in genes}
    offset = 0
    progress = tqdm(
        total=len(locs),
        desc="Read selected genes",
        unit="bins",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["bin_id", *genes]
        ):
            _validate_batch_ids(batch, expected_ids, offset)
            for gene in genes:
                column = batch.column(batch.schema.get_field_index(gene))
                if column.null_count:
                    raise ValueError(f"gene column {gene!r} contains null counts")
                parts[gene].append(column.to_numpy(zero_copy_only=False))
            offset += batch.num_rows
            progress.update(batch.num_rows)
    finally:
        progress.close()
    if offset != len(locs):
        raise ValueError("count Parquet ended before all location rows were matched")
    return {
        gene: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
        for gene, chunks in parts.items()
    }


def read_umi_counts(
    count_path,
    locs,
    *,
    batch_size=DEFAULT_BATCH_SIZE,
    show_progress=True,
):
    """Compute exact per-bin UMI totals in bounded Parquet row batches."""
    pq = _require_pyarrow()
    parquet = pq.ParquetFile(count_path)
    _validate_row_counts(parquet, locs)
    names = list(parquet.schema_arrow.names)
    gene_names = [name for name in names if name != "bin_id"]
    if not gene_names:
        raise ValueError("count Parquet contains no gene columns")
    expected_ids = locs["bin_id"].astype(str).to_numpy(dtype=object)
    parts = []
    offset = 0
    progress = tqdm(
        total=len(locs),
        desc="Sum UMI counts",
        unit="bins",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=names):
            _validate_batch_ids(batch, expected_ids, offset)
            totals = np.zeros(batch.num_rows, dtype=np.uint64)
            for gene in gene_names:
                column = batch.column(batch.schema.get_field_index(gene))
                if column.null_count:
                    raise ValueError(f"gene column {gene!r} contains null counts")
                values = column.to_numpy(zero_copy_only=False)
                np.add(totals, values, out=totals, casting="unsafe")
            parts.append(totals)
            offset += batch.num_rows
            progress.update(batch.num_rows)
    finally:
        progress.close()
    if offset != len(locs):
        raise ValueError("count Parquet ended before all location rows were matched")
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)


def validate_clip_q(clip_q):
    if len(clip_q) != 2:
        raise ValueError("clip_q must contain exactly two percentiles")
    qmin, qmax = (float(value) for value in clip_q)
    if not (0 <= qmin < qmax <= 100):
        raise ValueError("clip_q must satisfy 0 <= MIN < MAX <= 100")
    return qmin, qmax


def _bin_indices(bin_ids):
    x_indices = np.empty(len(bin_ids), dtype=np.int64)
    y_indices = np.empty(len(bin_ids), dtype=np.int64)
    pattern = re.compile(r"^(-?\d+)_(-?\d+)$")
    for index, bin_id in enumerate(bin_ids):
        match = pattern.match(str(bin_id))
        if match is None:
            return None
        x_indices[index] = int(match.group(1))
        y_indices[index] = int(match.group(2))
    return x_indices, y_indices


def _figure_size_from_spans(x_span, y_span, longest_inches=10.0):
    x_span = max(float(x_span), 1.0)
    y_span = max(float(y_span), 1.0)
    if x_span >= y_span:
        return longest_inches, max(2.0, longest_inches * y_span / x_span)
    return max(2.0, longest_inches * x_span / y_span), longest_inches


def _figure_size(locs, longest_inches=10.0):
    return _figure_size_from_spans(
        locs["x"].max() - locs["x"].min(),
        locs["y"].max() - locs["y"].min(),
        longest_inches,
    )


def read_he_preview(he_img, max_dimension):
    """Read a downsampled RGB preview while retaining full-image dimensions."""
    from PIL import Image

    path = Path(he_img).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"H&E image not found: {path}")
    if not isinstance(max_dimension, int) or max_dimension <= 0:
        raise ValueError("H&E preview maximum dimension must be positive")

    # Whole-slide exports commonly exceed Pillow's generic decompression-bomb
    # threshold. This is a local, explicitly supplied image and is immediately
    # reduced to the small QC output resolution.
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            full_width, full_height = image.size
            if full_width <= 0 or full_height <= 0:
                raise ValueError(f"H&E image has invalid dimensions: {image.size}")
            image.draft("RGB", (max_dimension, max_dimension))
            image.thumbnail(
                (max_dimension, max_dimension),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            preview = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read H&E image {path}: {exc}") from exc
    return preview, (int(full_width), int(full_height)), path


def _metadata_bin_size(count_path):
    metadata_path = Path(count_path).parent / "binning.json"
    if not metadata_path.is_file():
        return None
    try:
        value = float(json.loads(metadata_path.read_text())["bin_size_target_px"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _infer_bin_size(locs, bin_x, bin_y, count_path):
    bin_size = _metadata_bin_size(count_path)
    if bin_size is not None:
        return bin_size
    candidates = []
    for indices, coordinates in (
        (bin_x, locs["x"].to_numpy(dtype=np.float64)),
        (bin_y, locs["y"].to_numpy(dtype=np.float64)),
    ):
        order = np.argsort(indices, kind="stable")
        delta_index = np.diff(indices[order])
        delta_coord = np.diff(coordinates[order])
        valid = delta_index != 0
        if np.any(valid):
            ratios = np.abs(delta_coord[valid] / delta_index[valid])
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if len(ratios):
                candidates.append(float(np.median(ratios)))
    return float(np.median(candidates)) if candidates else None


def _plot_as_grid(
    ax,
    locs,
    values,
    cmap,
    norm,
    max_grid_cells,
    count_path,
    alpha,
):
    indices = _bin_indices(locs["bin_id"].astype(str).to_numpy())
    if indices is None:
        return False
    bin_x, bin_y = indices
    x_min, x_max = int(bin_x.min()), int(bin_x.max())
    y_min, y_max = int(bin_y.min()), int(bin_y.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    cells = width * height
    if cells <= 0 or cells > max_grid_cells:
        return False
    bin_size = _infer_bin_size(locs, bin_x, bin_y, count_path)
    if bin_size is None:
        return False
    grid = np.full((height, width), np.nan, dtype=np.float32)
    grid[bin_y - y_min, bin_x - x_min] = values.astype(np.float32, copy=False)
    masked = np.ma.masked_invalid(grid)
    x_center_min = float(locs["x"].min())
    x_center_max = float(locs["x"].max())
    y_center_min = float(locs["y"].min())
    y_center_max = float(locs["y"].max())
    ax.imshow(
        masked,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
        extent=(
            x_center_min - bin_size / 2,
            x_center_max + bin_size / 2,
            y_center_max + bin_size / 2,
            y_center_min - bin_size / 2,
        ),
        alpha=alpha,
    )
    return True


def _plot_as_scatter(
    ax, locs, values, cmap, norm, point_size, count_path, alpha
):
    if point_size is None:
        bin_size = _metadata_bin_size(count_path)
        x_span = max(float(locs["x"].max() - locs["x"].min()), 1.0)
        y_span = max(float(locs["y"].max() - locs["y"].min()), 1.0)
        span = max(x_span, y_span)
        if bin_size is not None:
            side_points = max(0.1, 10.0 * 72.0 * bin_size / span)
            point_size = side_points**2
        else:
            point_size = max(0.05, min(20.0, 200_000.0 / len(locs)))
    ax.scatter(
        locs["x"],
        locs["y"],
        c=values,
        s=point_size,
        marker="s",
        cmap=cmap,
        norm=norm,
        linewidths=0,
        rasterized=True,
        alpha=alpha,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()


def safe_gene_filename(gene):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(gene)).strip("._")
    return (cleaned or "gene")[:120]


def plot_spatial_values(
    locs,
    values,
    output_path,
    *,
    title,
    colorbar_label,
    count_path,
    cmap_name="turbo",
    clip_q=DEFAULT_CLIP_Q,
    log1p_transform=False,
    dpi=DEFAULT_DPI,
    point_size=None,
    max_grid_cells=DEFAULT_MAX_GRID_CELLS,
    he_img=None,
    overlay_alpha=DEFAULT_OVERLAY_ALPHA,
):
    """Render binned values as a compact white-background spatial QC plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if len(values) != len(locs):
        raise ValueError(
            f"value/location row mismatch: {len(values):,} versus {len(locs):,}"
        )
    if not np.issubdtype(np.asarray(values).dtype, np.number):
        raise ValueError("plot values must be numeric")
    if not np.isfinite(overlay_alpha) or not 0 < overlay_alpha <= 1:
        raise ValueError("overlay_alpha must satisfy 0 < alpha <= 1")
    qmin, qmax = validate_clip_q(clip_q)
    numeric = np.asarray(values)
    low, high = np.percentile(numeric, [qmin, qmax])
    plotted = np.clip(numeric, low, high).astype(np.float32, copy=False)
    if log1p_transform:
        plotted = np.log1p(plotted)
        colorbar_label = f"log1p({colorbar_label})"
    vmin = float(np.min(plotted))
    vmax = float(np.max(plotted))
    if vmax <= vmin:
        vmax = vmin + 1.0

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(cmap_name).copy()
    he_preview = None
    he_size = None
    if he_img is not None:
        he_preview, he_size, _ = read_he_preview(
            he_img, max(256, int(round(10 * dpi)))
        )
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    else:
        cmap.set_bad("white")
    norm = Normalize(vmin=vmin, vmax=vmax)
    figsize = (
        _figure_size_from_spans(*he_size)
        if he_size is not None
        else _figure_size(locs)
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if he_preview is not None:
        full_width, full_height = he_size
        ax.imshow(
            he_preview,
            origin="upper",
            interpolation="nearest",
            extent=(0, full_width, full_height, 0),
            aspect="equal",
        )
    used_grid = _plot_as_grid(
        ax,
        locs,
        plotted,
        cmap,
        norm,
        int(max_grid_cells),
        count_path,
        overlay_alpha if he_preview is not None else 1.0,
    )
    if not used_grid:
        _plot_as_scatter(
            ax,
            locs,
            plotted,
            cmap,
            norm,
            point_size,
            count_path,
            overlay_alpha if he_preview is not None else 1.0,
        )
    if he_size is not None:
        full_width, full_height = he_size
        ax.set_xlim(0, full_width)
        ax.set_ylim(full_height, 0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(
        f"{colorbar_label} (clipped {qmin:g}–{qmax:g} percentile)"
    )
    fig.tight_layout()
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "facecolor": "white"}
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": 85, "optimize": True}
    fig.savefig(output, **save_kwargs)
    plt.close(fig)
    return output, "grid" if used_grid else "scatter"
