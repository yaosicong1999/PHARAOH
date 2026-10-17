#!/usr/bin/env python3
"""Plot one or more genes from a Xenium cnts.parquet spatial bin table."""

import argparse
import difflib
from pathlib import Path

from xenium_plot_binned_utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_Q,
    DEFAULT_DPI,
    DEFAULT_MAX_GRID_CELLS,
    DEFAULT_OVERLAY_ALPHA,
    parquet_gene_names,
    plot_spatial_values,
    read_gene_counts,
    read_locations,
    resolve_inputs,
    safe_gene_filename,
)


def select_genes(available, *, genenames=None, first_k=None):
    if (genenames is None) == (first_k is None):
        raise ValueError("provide exactly one of --genename or --first_k")
    if first_k is not None:
        if first_k <= 0:
            raise ValueError("--first_k must be a positive integer")
        return available[:first_k]

    selected = list(dict.fromkeys(genenames))
    missing = [gene for gene in selected if gene not in available]
    if missing:
        details = []
        for gene in missing:
            matches = difflib.get_close_matches(gene, available, n=3)
            suggestion = f" (closest: {', '.join(matches)})" if matches else ""
            details.append(f"{gene!r}{suggestion}")
        raise ValueError("gene(s) not found in cnts.parquet: " + ", ".join(details))
    return selected


def plot_genes(
    input_path,
    *,
    genenames=None,
    first_k=None,
    locs_path=None,
    output_dir=None,
    cmap="turbo",
    clip_q=DEFAULT_CLIP_Q,
    log1p_transform=False,
    dpi=DEFAULT_DPI,
    point_size=None,
    batch_size=DEFAULT_BATCH_SIZE,
    max_grid_cells=DEFAULT_MAX_GRID_CELLS,
    he_img=None,
    overlay_alpha=DEFAULT_OVERLAY_ALPHA,
    show_progress=True,
):
    counts_path, locations_path = resolve_inputs(input_path, locs_path)
    available = parquet_gene_names(counts_path)
    selected = select_genes(
        available, genenames=genenames, first_k=first_k
    )
    if not selected:
        raise ValueError("no genes were selected")
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else counts_path.parent / "plots"
    )
    locs = read_locations(locations_path)
    values_by_gene = read_gene_counts(
        counts_path,
        locs,
        selected,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    print(
        f"[plot-gene] {len(locs):,} aligned bins; plotting "
        f"{len(selected):,}/{len(available):,} genes",
        flush=True,
    )
    outputs = []
    used_names = set()
    for gene in selected:
        stem = safe_gene_filename(gene)
        if stem in used_names:
            suffix = 2
            while f"{stem}_{suffix}" in used_names:
                suffix += 1
            stem = f"{stem}_{suffix}"
        used_names.add(stem)
        overlay_suffix = "_overlay" if he_img is not None else ""
        output = output_dir / f"xenium_binned_gene_{stem}{overlay_suffix}.jpg"
        output, mode = plot_spatial_values(
            locs,
            values_by_gene[gene],
            output,
            title=f"Xenium binned expression: {gene}",
            colorbar_label=f"{gene} count",
            count_path=counts_path,
            cmap_name=cmap,
            clip_q=clip_q,
            log1p_transform=log1p_transform,
            dpi=dpi,
            point_size=point_size,
            max_grid_cells=max_grid_cells,
            he_img=he_img,
            overlay_alpha=overlay_alpha,
        )
        print(f"[plot-gene] saved ({mode}): {output}", flush=True)
        outputs.append(output)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected genes from cnts.parquet using its row-aligned locs.tsv."
        )
    )
    parser.add_argument("--input", required=True, help="bin-by-gene cnts.parquet")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--genename",
        "--gene-name",
        nargs="+",
        help="one or more exact Parquet gene column names",
    )
    selector.add_argument(
        "--first_k",
        "--first-k",
        type=int,
        help="plot the first K gene columns in Parquet order",
    )
    parser.add_argument(
        "--locs",
        help="location TSV (default: locs.tsv beside --input)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        help="plot directory (default: plots beside --input)",
    )
    parser.add_argument(
        "--he_img",
        "--he-img",
        help=(
            "optional H&E image at the same target-MPP pixel scale as locs.tsv"
        ),
    )
    parser.add_argument(
        "--overlay_alpha",
        "--overlay-alpha",
        type=float,
        default=DEFAULT_OVERLAY_ALPHA,
        help=f"bin overlay opacity (default: {DEFAULT_OVERLAY_ALPHA})",
    )
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument(
        "--clip_q",
        "--clip-q",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=DEFAULT_CLIP_Q,
    )
    parser.add_argument("--log1p", action="store_true")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--point_size",
        "--point-size",
        type=float,
        help="scatter marker area; only used if the regular grid is too large",
    )
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--max_grid_cells",
        "--max-grid-cells",
        type=int,
        default=DEFAULT_MAX_GRID_CELLS,
        help="largest dense plotting grid before switching to scatter",
    )
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_grid_cells <= 0:
        parser.error("--max_grid_cells must be positive")
    if args.point_size is not None and args.point_size <= 0:
        parser.error("--point_size must be positive")
    if not 0 < args.overlay_alpha <= 1:
        parser.error("--overlay_alpha must satisfy 0 < alpha <= 1")
    try:
        plot_genes(
            args.input,
            genenames=args.genename,
            first_k=args.first_k,
            locs_path=args.locs,
            output_dir=args.output_dir,
            cmap=args.cmap,
            clip_q=args.clip_q,
            log1p_transform=args.log1p,
            dpi=args.dpi,
            point_size=args.point_size,
            batch_size=args.batch_size,
            max_grid_cells=args.max_grid_cells,
            he_img=args.he_img,
            overlay_alpha=args.overlay_alpha,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
