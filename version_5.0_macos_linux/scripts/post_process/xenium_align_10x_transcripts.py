#!/usr/bin/env python3
"""Transform Xenium transcripts to H&E level 0 using 10x's own alignment matrix.

The companion to `xenium_align_transcripts.py`, for the `*_he_imagealignment.csv`
10x ships beside an H&E image. Same inputs, same output, same units on the way out --
the difference is entirely in how that matrix has to be read.

**10x's matrix runs the other way, and it is in pixels.** It maps *H&E level-0 pixels
to DAPI level-0 pixels*, so this tool inverts it. Neither fact is stated in the file,
which is three unlabelled rows, so both were established by measurement on
`human_liver_cancer` (DAPI 42618x20528 px at 0.2125 um/px, H&E 15404x33795 px):

* the linear part is a 90-degree rotation scaled by 1.28797, and 0.2125 * 1.28797 =
  0.2737 um/px, which is the H&E resolution 10x's own scans carry -- consistent with
  H&E pixels going *in* and DAPI pixels coming *out*, and with nothing else;
* applied forward to the DAPI corners it lands 171% outside the H&E canvas; inverted
  and applied to the DAPI corners it covers it;
* inverted, it agrees with PHARAOH's independently fitted
  `dapi_to_he_homography_level0.json` to a median corner distance of **7.4 px** --
  2.0 um -- on a 33,795 px canvas.

Read it in the wrong units instead and the corners land ~90,000 px away, so a mistake
here is not subtle. `--assume-direction` is provided for a dataset where 10x has
shipped the other convention; the default is the one measured above.

    python xenium_align_10x_transcripts.py \\
        --input_path  .../transcripts.zarr.zip \\
        --output_path .../transcripts_he_10x.parquet \\
        --matrix      .../Xenium_..._he_imagealignment.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xenium_align_transcripts import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DAPI_MPP,
    align_transcripts,
)
from pharaoh_transform import load_matrix  # noqa: E402

DIRECTIONS = ("he_to_dapi", "dapi_to_he")


def dapi_to_he_matrix(matrix_path, direction="he_to_dapi"):
    """10x's matrix as a DAPI-level0 to H&E-level0 homography.

    `direction` names what the file itself expresses. The default, `he_to_dapi`, is
    what 10x ships and what this module's docstring measures; it is inverted here.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    matrix = load_matrix(matrix_path)
    return np.linalg.inv(matrix) if direction == "he_to_dapi" else matrix


def write_matrix(matrix, path):
    """Write a 3x3 as the headerless CSV `align_transcripts` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert Xenium micron coordinates to H&E level-0 pixels using 10x's "
            "own *_he_imagealignment.csv, which is inverted first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_path", "--input-path", required=True,
                        help="transcripts.zarr.zip, .zarr, .parquet, .csv or .csv.gz")
    parser.add_argument("--output_path", "--output-path", required=True,
                        help="output .parquet")
    parser.add_argument("--matrix", required=True,
                        help="10x's *_he_imagealignment.csv")
    parser.add_argument("--assume-direction", "--assume_direction",
                        choices=DIRECTIONS, default="he_to_dapi",
                        help="what the supplied matrix expresses (default: "
                             "he_to_dapi, which is what 10x ships)")
    parser.add_argument("--x_column", "--x-column", default="x_location")
    parser.add_argument("--y_column", "--y-column", default="y_location")
    parser.add_argument("--dapi_mpp", "--dapi-mpp", type=float,
                        default=DEFAULT_DAPI_MPP,
                        help=f"DAPI level-0 um/px (default: {DEFAULT_DAPI_MPP})")
    parser.add_argument("--chunk_size", "--chunk-size", type=int,
                        default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--keep-matrix", "--keep_matrix", type=Path, default=None,
                        help="also save the inverted matrix here, as a headerless CSV "
                             "usable with xenium_align_transcripts.py")
    args = parser.parse_args(argv)

    matrix = dapi_to_he_matrix(args.matrix, args.assume_direction)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    print(f"[10x] matrix:    {args.matrix}", flush=True)
    print(f"[10x] as given:  {args.assume_direction.replace('_to_', ' level-0 px -> ')}"
          " level-0 px", flush=True)
    if args.assume_direction == "he_to_dapi":
        print("[10x] inverted so it reads DAPI level-0 px -> H&E level-0 px", flush=True)
    print(f"[10x] linear scale {scale:.5f}: H&E is {args.dapi_mpp / scale:.5f} um/px "
          f"if that reading is right", flush=True)

    destination = Path(args.output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    inverted = write_matrix(
        matrix,
        args.keep_matrix if args.keep_matrix
        else destination.with_name(destination.stem + "_matrix_dapi_to_he.csv"),
    )
    print(f"[10x] wrote the DAPI->H&E form to {inverted}", flush=True)

    align_transcripts(
        args.input_path,
        args.output_path,
        inverted,
        x_column=args.x_column,
        y_column=args.y_column,
        dapi_mpp=args.dapi_mpp,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
