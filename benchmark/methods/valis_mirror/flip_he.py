#!/usr/bin/env python3
"""Write the left-right mirror image of a pyramidal OME-TIFF, for wrapper.py --mirror double --he-flipped.

    python methods/valis_mirror/flip_he.py he_image.ome.tif he_image_flipx.ome.tif

Every pyramid level is flipped left-right (x_k -> W_k - 1 - x_k at level k, which is x -> W - 1 - x at level 0
with the pixel-centre convention), so the copy has the same levels, shapes, data type and pixel size as the
source and wrapper.py accepts it as the source's flip. Lossless (deflate), 1024-px tiles, one strip of 1024 rows
read at a time, so memory stays near one strip of level 0. The source may be RGB (an H&E) or single-channel.
"""
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import zarr

TILE = 1024


def pixel_size(t: tifffile.TiffFile) -> dict:
    """OME pixel size of the first image, as tifffile's OME metadata keys; empty if there is none."""
    if not t.ome_metadata:
        return {}
    pixels = ET.fromstring(t.ome_metadata).find("./{*}Image/{*}Pixels")
    out = {}
    for axis in ("X", "Y"):
        if pixels is not None and f"PhysicalSize{axis}" in pixels.attrib:
            out[f"PhysicalSize{axis}"] = float(pixels.attrib[f"PhysicalSize{axis}"])
            out[f"PhysicalSize{axis}Unit"] = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
    return out


def flipped_tiles(z, shape):
    """TILE x TILE tiles of the flipped level, row by row; reads one strip of TILE rows at a time."""
    h, w = shape[:2]
    for y in range(0, h, TILE):
        strip = np.asarray(z[y:y + TILE])[:, ::-1]
        for x in range(0, w, TILE):
            tile = np.zeros((TILE, TILE) + strip.shape[2:], strip.dtype)
            part = strip[:, x:x + TILE]
            tile[:part.shape[0], :part.shape[1]] = part
            yield tile


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="pyramidal OME-TIFF (YX or YXS)")
    p.add_argument("destination", type=Path, help="the flipped copy (refused if it exists)")
    args = p.parse_args()
    if args.destination.exists():
        raise SystemExit(f"{args.destination} exists")
    start = time.time()
    partial = args.destination.with_name(args.destination.name + ".partial")
    with tifffile.TiffFile(args.source) as t:
        levels = t.series[0].levels
        shape0 = levels[0].shape
        if len(shape0) not in (2, 3) or (len(shape0) == 3 and shape0[2] not in (3, 4)):
            raise SystemExit(f"{args.source}: expected YX or YXS (RGB) levels, got shape {shape0}")
        rgb = len(shape0) == 3
        with tifffile.TiffWriter(partial, bigtiff=True, ome=True) as out:
            for k, level in enumerate(levels):
                z = zarr.open(t.aszarr(level=k), mode="r")
                kw = dict(shape=level.shape, dtype=level.dtype, tile=(TILE, TILE), compression="zlib",
                          photometric="rgb" if rgb else "minisblack", maxworkers=8)
                if k == 0:
                    kw.update(subifds=len(levels) - 1, metadata={"axes": "YXS" if rgb else "YX", **pixel_size(t)})
                else:
                    kw.update(subfiletype=1, metadata=None)
                out.write(flipped_tiles(z, level.shape), **kw)
                print(f"  level {k}: {level.shape[1]} x {level.shape[0]} ({time.time() - start:.0f} s)", flush=True)
    with tifffile.TiffFile(args.source) as a, tifffile.TiffFile(partial) as b:
        la, lb = a.series[0].levels, b.series[0].levels
        if [l.shape for l in la] != [l.shape for l in lb] or not np.array_equal(lb[-1].asarray(), la[-1].asarray()[:, ::-1]):
            raise SystemExit(f"{partial} does not check out as the flip of {args.source}; left in place")
    partial.replace(args.destination)
    print(f"[flip_he] {args.destination} ({time.time() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
