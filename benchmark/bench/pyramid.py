"""Pyramid levels, the shared resolution tiers, and lossless pyramid re-encoding.

A tier is a hard cap on an image's LONGEST side, the same dimension VALIS's own caps
bound, so "2k" names the same working size for every method. The selected level is the
largest one at or under the cap; levels halve, so it lands in (0.5, 1.0] of the cap, and
the actual size is always recorded rather than inferred from the label.
"""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

TIER_MAX_SIDE = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384}


# --------------------------------------------------------------------------------------
# Level enumeration
# --------------------------------------------------------------------------------------

def _wh(shape, axes: str) -> tuple[int, int]:
    if axes and "X" in axes and "Y" in axes and len(axes) == len(shape):
        return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
    if len(shape) >= 3 and shape[-1] in (3, 4):
        return int(shape[-2]), int(shape[-3])
    return int(shape[-1]), int(shape[-2])


def level_sizes(path, series: int = 0) -> list[tuple[int, int]]:
    """(width, height) of every level, level 0 first.

    Series levels when tifffile reports them, otherwise the first page's SubIFDs. The
    second case is what a Xenium multi-file morphology_focus_0000.ome.tif looks like to
    tifffile: one series level, with the reduced levels stored as SubIFDs.
    """
    import tifffile
    with tifffile.TiffFile(str(path)) as handle:
        selected = handle.series[series]
        levels = getattr(selected, "levels", None)
        if levels and len(levels) > 1:
            return [_wh(level.shape, getattr(level, "axes", "")) for level in levels]
        page = handle.pages[0]
        return [_wh(page.shape, page.axes)] + [_wh(sub.shape, sub.axes) for sub in (page.pages or [])]


def read_level(path, level: int):
    """One level as an array. tifffile, because Xenium DAPI tiles are JPEG 2000, which
    libtiff/pyvips decode to zeros without raising and OpenSlide refuses outright."""
    import tifffile
    with tifffile.TiffFile(str(path)) as handle:
        levels = getattr(handle.series[0], "levels", None)
        if levels and len(levels) > 1:
            return levels[level].asarray()
        page = handle.pages[0]
        return page.asarray() if level == 0 else page.pages[level - 1].asarray()


def select_level(sizes, tier: str) -> int:
    """Largest level whose longest side is at or under the tier; the smallest otherwise."""
    cap = TIER_MAX_SIDE[tier]
    under = [i for i, (w, h) in enumerate(sizes) if max(w, h) <= cap]
    if under:
        return max(under, key=lambda i: max(sizes[i]))
    return len(sizes) - 1


# --------------------------------------------------------------------------------------
# Lossless re-encoding (VALIS input adaptation)
# --------------------------------------------------------------------------------------

class _TileCache:
    """Per-level LRU of decoded tiles; never holds a whole slide."""

    def __init__(self, max_bytes):
        self.max_bytes, self.entries, self.bytes = max_bytes, OrderedDict(), 0

    def read(self, page, handle, index):
        if index in self.entries:
            self.entries.move_to_end(index)
            return self.entries[index]
        handle.seek(page.dataoffsets[index])
        data, _, _ = page.decode(handle.read(page.databytecounts[index]), index,
                                 jpegtables=page.jpegtables)
        if data is not None:
            data = data.reshape(data.shape[-3:])
        size = 0 if data is None else data.nbytes
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes -= 0 if old is None else old.nbytes
            self.entries[index] = data
            self.bytes += size
        return data


def _region(page, handle, x0, y0, x1, y1, cache):
    import numpy as np
    if page.planarconfig == 2:
        raise ValueError("planar-separate samples need an explicit adapter")
    channels = int(page.samplesperpixel)
    width = page.imagewidth
    tw = page.tilewidth if page.is_tiled else width
    th = page.tilelength if page.is_tiled else page.rowsperstrip
    nx = math.ceil(width / tw)
    out = np.zeros((y1 - y0, x1 - x0, channels), dtype=page.dtype)
    for ty in range(y0 // th, (y1 - 1) // th + 1):
        for tx in range(x0 // tw, (x1 - 1) // tw + 1):
            data = cache.read(page, handle, ty * nx + tx)
            if data is None:  # sparse tile
                continue
            left, top = max(x0, tx * tw), max(y0, ty * th)
            right, bottom = min(x1, (tx + 1) * tw), min(y1, (ty + 1) * th)
            out[top - y0:bottom - y0, left - x0:right - x0] = data[
                top - ty * th:bottom - ty * th, left - tx * tw:right - tx * tw, :channels]
    return out


def reencode_pyramid(source, destination, tile: int = 256, *, cache_bytes=768 * 2**20,
                     compression_level: int = 6, maxworkers: int = 4,
                     start_level: int = 0) -> list[list[int]]:
    """Copy every level, at its bit depth, into a deflate OME-TIFF with SubIFDs.

    Tile/strip-wise, so memory stays bounded on a 70k px slide. Pixels are not mapped.
    `start_level` > 0 writes a truncated pyramid: that level becomes level 0 and only the
    coarser levels follow (the level-matched VALIS ablation), with the OME physical pixel
    size scaled to it.
    """
    import numpy as np
    import tifffile
    if tile < 16 or tile % 16:
        raise ValueError("tile size must be a positive multiple of 16")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(str(source)) as original, \
            tifffile.TiffWriter(str(destination), bigtiff=True, ome=True) as writer:
        physical = {}
        if original.ome_metadata:
            pixels = ET.fromstring(original.ome_metadata).find(".//{*}Image/{*}Pixels")
            if pixels is not None:
                physical = {k: v for k, v in pixels.attrib.items()
                            if k in ("PhysicalSizeX", "PhysicalSizeY",
                                     "PhysicalSizeXUnit", "PhysicalSizeYUnit")}
        levels = list(original.series[0].levels)
        if len(levels) == 1 and original.pages[0].pages:
            levels = [original.pages[0], *original.pages[0].pages]
        if not 0 <= start_level < len(levels):
            raise ValueError(f"start_level {start_level} out of range 0..{len(levels) - 1}")
        if start_level:
            first = lambda level: level if isinstance(level, tifffile.TiffPage) else level.pages[0]
            w0, wk = int(first(levels[0]).imagewidth), int(first(levels[start_level]).imagewidth)
            h0, hk = int(first(levels[0]).imagelength), int(first(levels[start_level]).imagelength)
            for key, ratio in (("PhysicalSizeX", w0 / wk), ("PhysicalSizeY", h0 / hk)):
                if key in physical:
                    physical[key] = repr(float(physical[key]) * ratio)
            levels = levels[start_level:]
        sizes = []
        for index, level in enumerate(levels):
            pages = ([level] if isinstance(level, tifffile.TiffPage) else list(level.pages))
            axes = getattr(level, "axes", "")
            if any(a in axes and level.shape[axes.index(a)] != 1 for a in ("Z", "T")):
                raise ValueError("only a single 2-D plane is supported")
            if len(pages) != 1:
                raise ValueError("multiple channel pages need an explicit channel policy")
            page = pages[0]
            w, h, c = int(page.imagewidth), int(page.imagelength), int(page.samplesperpixel)
            if c not in (1, 3, 4):
                raise ValueError("unsupported sample layout")
            out_c = 3 if c == 4 else c
            sizes.append([w, h])
            cache = _TileCache(cache_bytes)

            def tiles(page=page, w=w, h=h, out_c=out_c, cache=cache):
                for y in range(0, h, tile):
                    for x in range(0, w, tile):
                        x1, y1 = min(x + tile, w), min(y + tile, h)
                        data = _region(page, original.filehandle, x, y, x1, y1, cache)[..., :out_c]
                        padded = np.zeros((tile, tile, out_c), dtype=page.dtype)
                        padded[:y1 - y, :x1 - x] = data
                        yield padded[..., 0] if out_c == 1 else padded

            kwargs = dict(shape=(h, w) if out_c == 1 else (h, w, out_c), dtype=page.dtype,
                          tile=(tile, tile), compression="deflate",
                          compressionargs={"level": compression_level}, maxworkers=maxworkers,
                          photometric="minisblack" if out_c == 1 else "rgb")
            if index == 0:
                kwargs.update(subifds=len(levels) - 1, metadata=physical)
            else:
                kwargs["subfiletype"] = 1
            started = time.monotonic()
            writer.write(tiles(), **kwargs)
            print(f"[pyramid] {Path(source).name} level {index} {w}x{h} "
                  f"in {time.monotonic() - started:.0f}s", flush=True)
    return sizes
