#!/usr/bin/env python3
# ============================================================
# convert_btf_to_ome_tiff.py
#
# Convert a BigTIFF / TIFF (e.g. a 10x Visium `tissue_image.btf`) into a
# tiled, pyramidal OME-TIFF that PHARAOH reads natively through the
# `.ome.tif` pyramid branch in my_utils.read_image().
#
# Why: `.btf` is not matched by PHARAOH's `.ome.tif` / `.tif` extension
# checks, so it falls back to PIL and fails on gigapixel images. A
# pyramidal OME-TIFF lets every stage pick a downsampled level instead of
# decoding the full-resolution base each time.
#
# Usage:
#   python convert_btf_to_ome_tiff.py IN.btf OUT.ome.tif
#   python convert_btf_to_ome_tiff.py IN.btf OUT.ome.tif \
#       --levels 5 --tile 512 --compression zlib --downsample 2
#
#   --levels       number of pyramid sub-resolutions BELOW the base
#                  (default 5 -> base + 5 levels = /1 /2 /4 /8 /16 /32)
#   --tile         tile size in px (default 512)
#   --compression  zlib | lzw | jpeg | none   (default zlib, lossless)
#   --downsample   per-level factor (default 2)
#   --series       source series index to read (default 0)
# ============================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                       # pragma: no cover
    _HAVE_CV2 = False


# Keep every cv2 / numpy op below OpenCV's INT_MAX element limit (2**31) and
# bound peak memory, by processing in horizontal strips of source rows.
_MAX_STRIP_ELEMS = 1_200_000_000


def _downsample_strip(strip, nw, out_rows):
    """Area-downsample one source strip to (out_rows, nw[, C])."""
    if _HAVE_CV2:
        return cv2.resize(strip, (nw, out_rows), interpolation=cv2.INTER_AREA)
    # numpy block-mean fallback (strip is already small enough)
    fy = strip.shape[0] // out_rows
    fx = strip.shape[1] // nw
    crop = strip[:out_rows * fy, :nw * fx]
    if crop.ndim == 3:
        crop = crop.reshape(out_rows, fy, nw, fx, crop.shape[2])
        return crop.mean(axis=(1, 3)).astype(strip.dtype)
    crop = crop.reshape(out_rows, fy, nw, fx)
    return crop.mean(axis=(1, 3)).astype(strip.dtype)


def _downsample(arr, factor):
    """Downsample a (possibly huge) H,W[,C] array by `factor`, strip by strip.

    `arr` may be a numpy array or a zarr array (lazy tiled reads); only one
    strip is materialised at a time so this scales to >INT_MAX-element images.
    """
    if factor == 1:
        return np.asarray(arr)
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    nh, nw = max(1, h // factor), max(1, w // factor)
    out_shape = (nh, nw, c) if arr.ndim == 3 else (nh, nw)
    out = np.empty(out_shape, dtype=arr.dtype)

    # source rows per strip, rounded down to a multiple of `factor`
    rows_budget = max(factor, int(_MAX_STRIP_ELEMS // (w * c)))
    src_rows = max(factor, (rows_budget // factor) * factor)
    out_rows_per = max(1, src_rows // factor)

    for oy0 in range(0, nh, out_rows_per):
        oy1 = min(nh, oy0 + out_rows_per)
        sy0, sy1 = oy0 * factor, oy1 * factor
        strip = np.asarray(arr[sy0:sy1, :nw * factor])   # materialise one strip
        out[oy0:oy1] = _downsample_strip(strip, nw, oy1 - oy0)
    return out


def _to_yxc(arr):
    """Return an H,W (grayscale) or H,W,C (RGB-ish) array + photometric str."""
    if arr.ndim == 2:
        return arr, "minisblack"
    if arr.ndim == 3:
        # move a small channel axis to the last position (handle C,Y,X)
        if arr.shape[0] in (2, 3, 4) and arr.shape[-1] not in (2, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        c = arr.shape[-1]
        if c == 1:
            return arr[..., 0], "minisblack"
        if c in (3, 4):
            return arr, "rgb"
        # unusual channel count: keep as-is, treat as separate samples
        return arr, "minisblack"
    raise ValueError(f"unsupported array shape {arr.shape}")


def main():
    ap = argparse.ArgumentParser(description="Convert a BigTIFF/TIFF to a pyramidal OME-TIFF.")
    ap.add_argument("src", help="input .btf / .tif")
    ap.add_argument("dst", help="output .ome.tif")
    ap.add_argument("--levels", type=int, default=5,
                    help="pyramid sub-resolutions below the base (default 5)")
    ap.add_argument("--tile", type=int, default=512, help="tile size px (default 512)")
    ap.add_argument("--compression", default="zlib",
                    choices=["zlib", "lzw", "jpeg", "none"],
                    help="tile compression (default zlib, lossless)")
    ap.add_argument("--downsample", type=int, default=2,
                    help="per-level downsample factor (default 2)")
    ap.add_argument("--series", type=int, default=0, help="source series index (default 0)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        ap.error(f"input not found: {src}")
    if not dst.name.endswith((".ome.tif", ".ome.tiff")):
        ap.error("output name must end in .ome.tif (so PHARAOH uses the OME pyramid reader)")
    if args.levels < 0:
        ap.error("--levels must be >= 0")

    print(f"[convert] reading base (series={args.series}) from {src.name} ...", flush=True)
    with tifffile.TiffFile(str(src)) as tif:
        base = tif.series[args.series].asarray()
    base, photometric = _to_yxc(base)
    print(f"[convert] base shape={base.shape} dtype={base.dtype} photometric={photometric}",
          flush=True)

    compression = None if args.compression == "none" else args.compression
    tile = (args.tile, args.tile)

    # Build the pyramid levels in memory (base + N sub-resolutions).
    levels = [base]
    for i in range(args.levels):
        nxt = _downsample(levels[-1], args.downsample)
        levels.append(nxt)
        h, w = nxt.shape[:2]
        print(f"[convert]   level {i + 1}: {w}x{h}", flush=True)
        if min(h, w) <= 1:
            print(f"[convert]   stopping: level {i + 1} reached minimum size", flush=True)
            break

    n_sub = len(levels) - 1
    write_kw = dict(photometric=photometric, tile=tile, compression=compression)

    print(f"[convert] writing pyramidal OME-TIFF -> {dst}  "
          f"(base + {n_sub} levels, tile={args.tile}, compression={args.compression})",
          flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(dst), bigtiff=True, ome=True) as tw:
        tw.write(levels[0], subifds=n_sub, metadata={"axes": "YXC" if base.ndim == 3 else "YX"},
                 **write_kw)
        for lvl in levels[1:]:
            tw.write(lvl, subfiletype=1, **write_kw)

    print(f"[convert] DONE. {dst}  ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
