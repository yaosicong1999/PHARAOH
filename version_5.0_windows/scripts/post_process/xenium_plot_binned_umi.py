#!/usr/bin/env python3
"""Plot per-bin total UMI from a Xenium cnts.parquet spatial bin table."""

import argparse
from pathlib import Path

from xenium_plot_binned_utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_Q,
    DEFAULT_DPI,
    DEFAULT_MAX_GRID_CELLS,
    DEFAULT_OVERLAY_ALPHA,
    plot_spatial_values,
    read_locations,
    read_umi_counts,
    resolve_inputs,
)


def plot_umi(
    input_path,
    *,
    locs_path=None,
    output_img=None,
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
    output = (
        Path(output_img).expanduser().resolve()
        if output_img is not None
        else counts_path.parent / "plots" / (
            "xenium_binned_umi_overlay.jpg"
            if he_img is not None
            else "xenium_binned_umi.jpg"
        )
    )
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        raise ValueError("--output_img must end in .jpg, .jpeg, .png, .tif, or .tiff")
    locs = read_locations(locations_path)
    umi = read_umi_counts(
        counts_path,
        locs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    output, mode = plot_spatial_values(
        locs,
        umi,
        output,
        title="Xenium binned total UMI",
        colorbar_label="UMI count",
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
    print(
        f"[plot-umi] {len(locs):,} aligned bins; saved ({mode}): {output}",
        flush=True,
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-bin total UMI from cnts.parquet using its row-aligned locs.tsv."
        )
    )
    parser.add_argument("--input", required=True, help="bin-by-gene cnts.parquet")
    parser.add_argument(
        "--locs",
        help="location TSV (default: locs.tsv beside --input)",
    )
    parser.add_argument(
        "--output_img",
        "--output-img",
        help=(
            "output image (default: plots/xenium_binned_umi.jpg, or "
            "xenium_binned_umi_overlay.jpg with --he_img)"
        ),
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
        plot_umi(
            args.input,
            locs_path=args.locs,
            output_img=args.output_img,
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
