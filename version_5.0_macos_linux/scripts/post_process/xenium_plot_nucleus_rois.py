#!/usr/bin/env python3
"""Plot spatially separated DAPI/H&E ROIs with raw Xenium nucleus boundaries.

Input vertices are in image-origin micrometers. The transform direction is DAPI
level-0 px -> H&E level-0 px and may be a canonical PHARAOH JSON (including a
local field) or a legacy 3x3 CSV/TXT matrix.
Each row shows the SAME square DAPI physical field, with H&E resampled into
that field's orientation. This is visualization, not another registration fit.
TIFF images are read locally at level 0, never decoded as whole-slide arrays.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin

from pharaoh_transform import load_matrix as load_homography
from pharaoh_transform import load_transform


GREEN = (0, 210, 75)
CYAN = (0, 185, 255)
MAX_READ_PIXELS = 16_000_000
MAX_FIGURE_PIXELS = 64_000_000


def positive_pair(value):
    """Normalize scalar or (x, y) level-0 MPP without losing anisotropy."""
    values = np.asarray(value, dtype=float).ravel()
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2 or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("DAPI MPP must be one or two positive finite values (um/pixel)")
    return values


def ome_mpp(metadata):
    if not metadata:
        return None
    factors = {"µm": 1., "μm": 1., "um": 1., "nm": .001,
               "pm": .000001, "mm": 1000., "cm": 10000., "m": 1e6}
    try:
        pixels = ET.fromstring(metadata).find("./{*}Image/{*}Pixels")
        return positive_pair([
            float(pixels.attrib[f"PhysicalSize{axis}"])
            * factors[pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")]
            for axis in ("X", "Y")])
    except (ET.ParseError, AttributeError, KeyError, TypeError, ValueError):
        return None


def local_ome_level0(tif, path, channel):
    """Resolve a plane stored in this member of a multi-file OME image.

    Inspect XML BEFORE accessing tif.series: constructing the series tries to
    open every sibling channel and assemble an unsupported multi-file pyramid.
    We need only one local level-0 IFD and its original OME physical spacing.
    """
    if not tif.ome_metadata:
        return None
    root = ET.fromstring(tif.ome_metadata)
    pixels = root.find("./{*}Image/{*}Pixels")
    if pixels is None:
        return None
    path = Path(path).resolve()

    def is_local(data):
        uuid = data.find("./{*}UUID")
        if uuid is None:
            return True
        # A file may have been renamed after export (e.g. CRC_P1.ome.tif ->
        # he-raw.ome.tif). Its UUID, unlike FileName, still identifies it.
        if uuid.text and uuid.text == root.attrib.get("UUID"):
            return True
        filename = uuid.attrib.get("FileName")
        if filename:
            return (path.parent / filename).resolve() == path
        return uuid.text == root.attrib.get("UUID")

    records = pixels.findall("./{*}TiffData")
    if not any(not is_local(record) for record in records):
        return None
    size_c = int(pixels.attrib["SizeC"])
    if not 0 <= channel < size_c:
        raise ValueError(f"OME channel {channel} out of range (0..{size_c - 1})")
    channels = pixels.findall("./{*}Channel")
    spp = int(channels[0].attrib.get("SamplesPerPixel", 1)) if channels else 1
    if any(int(c.attrib.get("SamplesPerPixel", 1)) != spp for c in channels):
        raise ValueError("multi-file OME channels with different SamplesPerPixel are not supported")
    sizes = {axis: int(pixels.attrib[f"Size{axis}"]) for axis in "ZCT"}
    sizes["C"] //= spp
    stride, strides = 1, {}
    for axis in pixels.attrib["DimensionOrder"][2:]:
        strides[axis] = stride
        stride *= sizes[axis]
    target = (channel // spp) * strides["C"]  # first Z and T plane
    matches = []
    for record in records:
        if not is_local(record):
            continue
        attributes = record.attrib
        first = sum(int(attributes.get(f"First{axis}", 0)) * strides[axis] for axis in "ZCT")
        ifd = int(attributes.get("IFD", 0))
        count = int(attributes.get("PlaneCount", attributes.get(
            "NumPlanes", 1 if "IFD" in attributes else len(tif.pages))))
        if first <= target < first + count:
            matches.append(ifd + target - first)
    if len(matches) != 1:
        raise ValueError(f"OME channel {channel}, Z=0, T=0 does not map to one local plane in "
                         f"{path.name}; supply the file containing the requested channel")
    if not 0 <= matches[0] < len(tif.pages):
        raise ValueError(f"OME references local IFD {matches[0]}, which is missing from {path.name}")
    page = tif.pages[matches[0]]
    if (page.imagewidth, page.imagelength) != (int(pixels.attrib["SizeX"]), int(pixels.attrib["SizeY"])):
        raise ValueError("local OME page dimensions do not match level 0; refusing a reduced/cropped plane")
    print(f"[Image] multi-file OME: {path.name}, local level-0 IFD {matches[0]}, "
          f"channel {channel}, Z=0, T=0 (no sibling files or pyramid assembly)", flush=True)
    return page, channel % spp


def project_matrix_points(xy, matrix):
    """Apply a panel/crop homography; registration uses pharaoh_transform."""
    xy = np.asarray(xy, dtype=np.float64)
    projected = xy @ matrix[:2, :2].T + matrix[:2, 2]
    denominator = xy @ matrix[2, :2] + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return projected / denominator[..., None]


class ImageReader:
    """One open level-0 image; TIFF chunks are decoded only for requested ROIs."""

    def __init__(self, path, modality, channel=0):
        self.path, self.modality, self.channel = Path(path), modality, channel
        self.tif = self.store = self.pil = None
        self.mpp = None
        try:
            if self.path.suffix.lower() in {".tif", ".tiff"}:
                import tifffile
                import zarr
                self.tif = tifffile.TiffFile(self.path)
                self.mpp = ome_mpp(self.tif.ome_metadata)
                local = local_ome_level0(self.tif, self.path, channel)
                if local is None:
                    series = self.tif.series[0].levels[0]
                else:
                    series, self.channel = local
                self.axes = series.axes
                if "X" not in self.axes or "Y" not in self.axes:
                    raise ValueError(f"cannot identify TIFF X/Y axes: {self.axes}")
                self.height = series.shape[self.axes.index("Y")]
                self.width = series.shape[self.axes.index("X")]
                self.store = series.aszarr()
                self.array = zarr.open(self.store, mode="r")
                if not hasattr(self.array, "shape"):
                    self.array = self.array["0"]
                colour = "S" if "S" in self.axes else "C" if "C" in self.axes else None
                self.colour_axis = colour
                channels = series.shape[self.axes.index(colour)] if colour else 1
                if modality == "dapi" and not 0 <= self.channel < channels:
                    raise ValueError(f"DAPI channel {channel} out of range (0..{channels - 1})")
                if modality == "he" and channels not in (1, 3, 4):
                    raise ValueError("H&E must be grayscale, RGB or RGBA")
            else:
                self.pil = Image.open(self.path)
                self.width, self.height = self.pil.size
                if modality == "dapi" and not 0 <= channel < len(self.pil.getbands()):
                    raise ValueError("DAPI channel out of range")
        except Exception:
            self.close()
            raise

    def read(self, x0, y0, x1, y1):
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("invalid or out-of-bounds ROI")
        if (x1 - x0) * (y1 - y0) > MAX_READ_PIXELS:
            raise ValueError("level-0 ROI exceeds 16 megapixels; reduce --tile-size-um or check matrix")
        if self.pil is not None:
            patch = self.pil.crop((x0, y0, x1, y1))
            array = np.asarray(patch.convert("RGB") if self.modality == "he" else patch)
            if self.modality == "dapi" and array.ndim == 3:
                array = array[..., self.channel]
        else:
            indices, kept = [], []
            for axis in self.axes:
                if axis in "XY":
                    indices.append(slice(x0, x1) if axis == "X" else slice(y0, y1))
                    kept.append(axis)
                elif axis == self.colour_axis and self.modality == "he":
                    indices.append(slice(0, 3))
                    kept.append(axis)
                else:
                    indices.append(self.channel if axis == self.colour_axis else 0)
            array = np.asarray(self.array[tuple(indices)])
            wanted = ["Y", "X"] + ([self.colour_axis] if self.colour_axis in kept else [])
            array = array.transpose([kept.index(axis) for axis in wanted])
        if self.modality == "he":
            if array.ndim == 2:
                array = array[..., None]
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=2)
        return np.ascontiguousarray(array)

    def close(self):
        try:
            if self.store is not None:
                self.store.close()
        finally:
            if self.tif is not None:
                self.tif.close()
            if self.pil is not None:
                self.pil.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BoundaryIndex:
    """Compact vertex array and per-polygon boxes; query without rescanning a table."""

    def __init__(self, path, mpp):
        path = Path(path)
        required = {"cell_id", "vertex_x", "vertex_y"}
        wanted = required | {"label_id"}
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            available = pq.ParquetFile(path).schema_arrow.names
            frame = pd.read_parquet(path, columns=[c for c in available if c in wanted])
        elif path.name.lower().endswith((".csv", ".csv.gz")):
            frame = pd.read_csv(path, usecols=lambda c: c in wanted)
        else:
            raise ValueError("raw nucleus boundaries must be .csv, .csv.gz or .parquet (not Zarr)")
        if not required.issubset(frame.columns):
            raise ValueError(f"missing boundary columns: {sorted(required - set(frame.columns))}")
        if frame.empty or frame.isna().any().any():
            raise ValueError("boundary table is empty or contains missing values")
        xy = frame[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(xy)):
            raise ValueError("boundary coordinates contain NaN or infinity")
        # Stable grouping preserves vertex order, even if nuclei are interleaved.
        # label_id prevents separate nuclei belonging to one cell from merging.
        keys = [c for c in ("cell_id", "label_id") if c in frame.columns]
        codes = frame.groupby(keys, sort=False).ngroup().to_numpy()
        group_ids = frame[keys].drop_duplicates().to_numpy()
        del frame
        if np.any(np.diff(codes) < 0):
            order = np.argsort(codes, kind="stable")
            xy, codes = xy[order], codes[order]
        # Fixed-length exports can pad a polygon with copies of its closing
        # vertex. They are not new nuclei or separate short rings. Never remove
        # a matching coordinate across a cell/label boundary.
        duplicate = np.r_[False, np.all(xy[1:] == xy[:-1], axis=1) & (codes[1:] == codes[:-1])]
        duplicate_count = int(duplicate.sum())
        if duplicate_count:
            xy, codes = xy[~duplicate], codes[~duplicate]
        group_starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        group_ends = np.r_[group_starts[1:], len(xy)]
        del codes
        starts, ends, ring_groups = [], [], []
        for group, (start, stop) in enumerate(zip(group_starts, group_ends)):
            # Older exports may omit label_id. Repeated closing vertices still
            # separate multiple rings, rather than drawing a bridge between them.
            while start < stop:
                end = stop
                if "label_id" not in keys:
                    closing = np.flatnonzero(np.all(xy[start + 1:stop] == xy[start], axis=1))
                    if len(closing):
                        end = start + 2 + closing[0]
                        if stop - end < 3:
                            end = stop  # final retraced spur, not another polygon
                # An explicit label identifies ONE whole polygon. Revisiting its
                # first vertex before the last row is not a polygon separator.
                # Retain even short intervals until AFTER reduceat, so skipped
                # vertices cannot contaminate the previous valid polygon's bbox
                # or centroid. The intervals must partition the entire array.
                starts.append(start)
                ends.append(end)
                ring_groups.append(group)
                start = end
        self.xy = xy / positive_pair(mpp)
        self.starts, self.ends = np.asarray(starts), np.asarray(ends)
        self.low = np.minimum.reduceat(self.xy, self.starts)
        self.high = np.maximum.reduceat(self.xy, self.starts)
        counts = self.ends - self.starts
        closed = np.all(self.xy[self.starts] == self.xy[self.ends - 1], axis=1)
        sums = np.add.reduceat(self.xy, self.starts) - closed[:, None] * self.xy[self.starts]
        self.centers = sums / np.maximum(1, counts - closed)[:, None]
        short = counts - closed < 3
        x, y = self.xy.T
        cross = np.r_[x[:-1] * y[1:] - y[:-1] * x[1:], 0.]
        cross[self.ends - 1] = x[self.ends - 1] * y[self.starts] - y[self.ends - 1] * x[self.starts]
        twice_area = np.abs(np.add.reduceat(cross, self.starts))
        # Account for cancellation in shoelace sums at large WSI coordinates.
        magnitude = np.maximum(np.abs(self.low), np.abs(self.high)).max(axis=1)
        tolerance = 32 * np.finfo(float).eps * np.maximum(1., magnitude ** 2) * counts
        zero_area = (~np.all(self.high > self.low, axis=1) | (twice_area <= tolerance)) & ~short
        valid = ~(short | zero_area)
        examples = []
        for index in np.flatnonzero(~valid)[:5]:
            identity = dict(zip(keys, map(str, group_ids[ring_groups[index]])))
            examples.append({**identity, "reason": "too_few_vertices" if short[index] else "zero_area"})
        self.qc = {"removed_consecutive_duplicate_vertices": duplicate_count,
                   "skipped_too_few_vertices": int(short.sum()),
                   "skipped_zero_area": int(zero_area.sum()),
                   "kept_polygons": int(valid.sum()), "skipped_examples": examples}
        self.has_repairs = bool(duplicate_count or np.any(~valid))
        if self.has_repairs:
            print(f"[Boundary QC] removed {duplicate_count:,} duplicate/padding vertices; "
                  f"skipped {int((~valid).sum()):,} degenerate polygons "
                  f"(too few vertices: {short.sum():,}, zero area: {zero_area.sum():,}); "
                  f"kept {valid.sum():,} polygons", flush=True)
            if examples:
                print(f"[Boundary QC] skipped examples: {json.dumps(examples)}", flush=True)
        self.starts, self.ends = self.starts[valid], self.ends[valid]
        self.low, self.high, self.centers = self.low[valid], self.high[valid], self.centers[valid]
        if not len(self.starts):
            raise ValueError("no valid nucleus polygons in boundary table")

    def near(self, low, high):
        keep = np.all(self.high >= low, axis=1) & np.all(self.low <= high, axis=1)
        return [self.xy[start:end] for start, end in zip(self.starts[keep], self.ends[keep])]


def roi_corners(centers, span):
    return np.asarray(centers)[..., None, :] + np.array([
        [-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * span


def eligible_centers(centers, span, transform, dapi_size, he_size):
    corners = roi_corners(centers, span)
    keep = np.all((corners >= 0) & (corners <= np.asarray(dapi_size) - 1), axis=(1, 2))
    mapped = transform.forward_points(corners)
    if transform.nonlinear:
        continuous = np.ones(len(corners), dtype=bool)
    else:
        w = corners @ transform.matrix[2, :2] + transform.matrix[2, 2]
        # A projective horizon may cross the ROI even if all four corners are finite.
        continuous = (w.min(axis=1) > 0) | (w.max(axis=1) < 0)
    keep &= continuous & np.all(np.isfinite(mapped), axis=(1, 2))
    keep &= np.all((mapped >= 0) & (mapped <= np.asarray(he_size) - 1), axis=(1, 2))
    return keep


def farthest_points(points, n, seed=42):
    """Seeded random first center, then greedy max-min Euclidean DAPI distance."""
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if n < 1 or n > len(points):
        raise ValueError(f"requested {n} ROIs, but only {len(points)} distinct eligible centers; "
                         "reduce --n-points/--tile-size-um or check the matrix direction")
    picks = np.empty(n, dtype=int)
    picks[0] = np.random.default_rng(seed).integers(len(points))
    distance = np.full(len(points), np.inf)
    for index in range(1, n):
        delta = points - points[picks[index - 1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        distance[picks[:index]] = -1
        picks[index] = np.argmax(distance)
    return points[picks]


def panel_to_dapi(center, span, pixels):
    step = np.asarray(span) / pixels
    origin = center - np.asarray(span) / 2 + step / 2
    return np.array([[step[0], 0., origin[0]], [0., step[1], origin[1]], [0., 0., 1.]])


def read_panel(reader, panel_to_source, pixels):
    corners = project_matrix_points(
        np.array([[0, 0], [pixels - 1, 0],
                  [pixels - 1, pixels - 1], [0, pixels - 1]]),
        panel_to_source,
    )
    if not np.all(np.isfinite(corners)):
        raise ValueError("ROI maps to infinity; check matrix")
    low = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - 2)
    high = np.minimum([reader.width, reader.height], np.ceil(corners.max(axis=0)).astype(int) + 3)
    raw = reader.read(*low, *high)
    # Output pixel -> source level-0 -> local crop. One resampling only.
    local = np.array([[1., 0., -low[0]], [0., 1., -low[1]], [0., 0., 1.]]) @ panel_to_source
    if raw.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raw = raw.astype(np.float32)
    return cv2.warpPerspective(raw, local, (pixels, pixels),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT)


def read_transformed_panel(reader, panel_to_moving, transform, pixels):
    """Resample the fixed image into a moving-oriented ROI panel."""
    if not transform.nonlinear:
        panel_to_fixed = transform.matrix @ panel_to_moving
        return read_panel(reader, panel_to_fixed, pixels), panel_to_fixed

    yy, xx = np.indices((pixels, pixels), dtype=np.float64)
    panel_xy = np.stack([xx, yy], axis=-1)
    moving_xy = project_matrix_points(panel_xy, panel_to_moving)
    fixed_xy = transform.forward_points(moving_xy)
    finite = np.all(np.isfinite(fixed_xy), axis=-1)
    if not np.any(finite):
        raise ValueError("ROI has no finite coordinates under the local-field transform")
    low = np.maximum(0, np.floor(fixed_xy[finite].min(axis=0)).astype(int) - 2)
    high = np.minimum(
        [reader.width, reader.height],
        np.ceil(fixed_xy[finite].max(axis=0)).astype(int) + 3,
    )
    if np.any(high <= low):
        raise ValueError("ROI falls outside the fixed image after transformation")
    raw = reader.read(*low, *high)
    if raw.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raw = raw.astype(np.float32)
    map_x = (fixed_xy[..., 0] - low[0]).astype(np.float32)
    map_y = (fixed_xy[..., 1] - low[1]).astype(np.float32)
    map_x[~finite] = -1.0
    map_y[~finite] = -1.0
    panel = cv2.remap(
        np.ascontiguousarray(raw), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return panel, None


def display_rgb(raw, modality):
    if modality == "dapi":
        values = raw.astype(np.float32)
        low, high = np.percentile(values, [1., 99.8])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        gray = np.zeros(values.shape, dtype=np.uint8) if high <= low else (
            np.clip((values - low) / (high - low), 0, 1) ** .8 * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)
    if raw.dtype == np.uint8:
        return raw.copy()
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        return (raw.astype(float) / np.iinfo(raw.dtype).max * 255).astype(np.uint8)
    if not np.all(np.isfinite(raw)) or raw.min() < 0 or raw.max() > 1:
        raise ValueError("floating-point H&E must be in [0, 1]")
    return (raw * 255).astype(np.uint8)


def draw_boundaries(rgb, polygons, world_to_panel, colour, thickness=1):
    for polygon in polygons:
        xy = project_matrix_points(polygon, world_to_panel)
        if not np.all(np.isfinite(xy)) or np.max(np.abs(xy)) > 1e8:
            raise ValueError("boundary maps to an invalid panel coordinate")
        cv2.polylines(rgb, [np.round(xy).astype(np.int32).reshape(-1, 1, 2)],
                      True, colour, thickness, cv2.LINE_AA)


def put_text(rgb, text, xy, scale=.5, colour=(45, 45, 45), thickness=1, max_width=None):
    available = rgb.shape[1] - xy[0] - 24 if max_width is None else max_width
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    scale *= min(1., available / max(1, width))
    cv2.putText(rgb, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def scale_bar(rgb, tile_um):
    side = rgb.shape[0]
    length_um = tile_um / 5
    start, end = (18, side - 20), (18 + int(round(side / 5)), side - 20)
    cv2.line(rgb, start, end, (0, 0, 0), 6)
    cv2.line(rgb, start, end, (255, 255, 255), 2)
    xy = (18, side - 30)
    put_text(rgb, f"{length_um:g} um", xy, .45, (0, 0, 0), 3)
    put_text(rgb, f"{length_um:g} um", xy, .45, (255, 255, 255))


def plot_rois(dapi_path, he_path, matrix_path, boundaries_path, output_path, *,
              n_points=9, tile_size_um=100., dapi_mpp=None, seed=42,
              panel_pixels=512, dapi_channel=0):
    if not np.isfinite(tile_size_um) or tile_size_um <= 0:
        raise ValueError("tile_size_um must be positive and finite")
    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if not isinstance(panel_pixels, (int, np.integer)) or not 256 <= panel_pixels <= 2048:
        raise ValueError("panel_pixels must be an integer from 256 to 2048")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    width, height = 2 * panel_pixels + 72, 124 + n_points * (panel_pixels + 46)
    if width * height > MAX_FIGURE_PIXELS:
        raise ValueError("figure exceeds 64 megapixels; reduce --n-points or --panel-pixels")
    sources = [Path(p).expanduser().resolve() for p in
               (dapi_path, he_path, matrix_path, boundaries_path)]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    dapi_path, he_path, matrix_path, boundaries_path = sources
    output = Path(output_path).expanduser().resolve()
    if output.is_dir() or not output.suffix:
        output = output / "xenium_nucleus_roi_qc.png"
    if output in sources:
        raise ValueError("output must not overwrite an input file")
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a PNG filename or directory")
    transform = load_transform(matrix_path)
    with ImageReader(dapi_path, "dapi", dapi_channel) as dapi, ImageReader(he_path, "he") as he:
        mpp = positive_pair(dapi_mpp) if dapi_mpp is not None else dapi.mpp
        if mpp is None:
            raise ValueError("DAPI OME physical pixel size is missing; supply --dapi-mpp (um/pixel)")
        span = tile_size_um / mpp
        print(f"DAPI MPP (x,y): {mpp.tolist()}; ROI: {tile_size_um:g} um; image reads: level 0", flush=True)
        nuclei = BoundaryIndex(boundaries_path, mpp)
        keep = eligible_centers(nuclei.centers, span, transform,
                                (dapi.width, dapi.height), (he.width, he.height))
        print(f"{len(nuclei.centers):,} nuclei; {keep.sum():,} centers with full DAPI/H&E coverage", flush=True)
        centers = farthest_points(nuclei.centers[keep], n_points, seed)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        put_text(canvas, "Xenium nucleus boundary QC", (24, 31), .8, thickness=2)
        put_text(canvas, f"{tile_size_um:g} um DAPI field | seed {seed} | level 0 reads | matched orientation",
                 (24, 57), .45)
        col_x = (24, 48 + panel_pixels)
        put_text(canvas, "DAPI + nucleus boundaries", (col_x[0], 91), .55, max_width=panel_pixels)
        put_text(canvas, "H&E + mapped boundaries", (col_x[1], 91), .55, max_width=panel_pixels)
        records = []
        for row, center in enumerate(centers):
            to_dapi = panel_to_dapi(center, span, panel_pixels)
            polygons = nuclei.near(center - span / 2, center + span / 2)
            dapi_rgb = display_rgb(read_panel(dapi, to_dapi, panel_pixels), "dapi")
            he_panel, to_he = read_transformed_panel(
                he, to_dapi, transform, panel_pixels
            )
            he_rgb = display_rgb(he_panel, "he")
            thickness = max(1, round(.2 * panel_pixels / tile_size_um))
            moving_to_panel = np.linalg.inv(to_dapi)
            draw_boundaries(dapi_rgb, polygons, moving_to_panel, GREEN, thickness)
            # The fixed image has been pulled into the moving ROI orientation, so
            # mapped boundaries occupy the same panel coordinates as their moving
            # originals even when registration contains a nonlinear field.
            draw_boundaries(he_rgb, polygons, moving_to_panel, CYAN, thickness)
            he_center = transform.forward_points(center)
            y = 124 + row * (panel_pixels + 46)
            for column, (rgb, point) in enumerate(((dapi_rgb, center), (he_rgb, he_center))):
                put_text(canvas, f"ROI {row + 1:02d} | L0 ({point[0]:.1f}, {point[1]:.1f})",
                         (col_x[column], y - 9), .45, max_width=panel_pixels)
                scale_bar(rgb, tile_size_um)
                canvas[y:y + panel_pixels, col_x[column]:col_x[column] + panel_pixels] = rgb
            records.append({"roi": row + 1, "center_dapi_px": center.tolist(),
                            "center_he_px": he_center.tolist(), "nuclei_in_roi_bbox": len(polygons),
                            "panel_to_dapi_px": to_dapi.tolist(),
                            "panel_to_he_px": None if to_he is None else to_he.tolist()})
            print(f"ROI {row + 1}/{n_points}: DAPI ({center[0]:.1f}, {center[1]:.1f}), "
                  f"{len(polygons)} nearby nuclei", flush=True)
        # Keep reproducibility information inside the ONE requested image.
        metadata = {"schema_version": 1, "dapi_path": str(dapi_path), "he_path": str(he_path),
                    "transform_path": str(matrix_path), "boundaries_path": str(boundaries_path),
                    "transform_type": transform.transform_type,
                    "base_matrix_dapi_to_he": transform.matrix.tolist(),
                    "dapi_mpp_xy": mpp.tolist(),
                    "tile_size_um": float(tile_size_um), "seed": int(seed),
                    "panel_pixels": int(panel_pixels), "dapi_channel": int(dapi_channel),
                    "sampling": "seeded greedy farthest-point in DAPI level-0 pixels",
                    "view": "DAPI-native field; H&E resampled to the same DAPI orientation",
                    "nuclei_count": len(nuclei.centers), "eligible_centers": int(keep.sum()),
                    "rois": records}
        if nuclei.has_repairs:
            metadata["nucleus_boundary_qc"] = nuclei.qc
        if n_points > 1:
            delta = (centers[:, None] - centers[None, :]) * mpp
            distances = np.linalg.norm(delta, axis=2)
            np.fill_diagonal(distances, np.inf)
            metadata["min_center_distance_um"] = float(distances.min())
            print(f"Minimum center separation: {distances.min():.1f} um", flush=True)
        info = PngImagePlugin.PngInfo()
        info.add_text("xenium_roi_qc", json.dumps(metadata))
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".png", dir=output.parent)
        os.close(descriptor)
        try:
            Image.fromarray(canvas).save(temporary, pnginfo=info)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(f"Saved {n_points} rows x 2 columns: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dapi", "--dapi-path", "--dapi_path", required=True, type=Path)
    parser.add_argument("--he", "--he-path", "--he_path", required=True, type=Path)
    parser.add_argument("--matrix", "--transform", required=True, type=Path,
                        help="PHARAOH transform JSON, or DAPI-to-H&E matrix CSV/TXT")
    parser.add_argument("--nucleus-boundaries", "--nucleus_boundaries", required=True, type=Path,
                        help="raw Xenium nucleus_boundaries.csv.gz / .csv / .parquet, in um")
    parser.add_argument("--n-points", "--n_points", type=int, default=9)
    parser.add_argument("--tile-size-um", "--tile_size_um", type=float, default=100.,
                        help="full square side in DAPI physical space (default: 100 um)")
    parser.add_argument("--output-path", "--output_path", "--out", required=True, type=Path,
                        help="output PNG filename or directory")
    parser.add_argument("--dapi-mpp", "--dapi_mpp", type=float, nargs="+",
                        help="override OME level-0 MPP: one scalar or two values X Y, in um/pixel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-pixels", type=int, default=512,
                        help="display side, 256..2048; does not change level-0 reading or physical field")
    parser.add_argument("--dapi-channel", type=int, default=0,
                        help="zero-based DAPI channel; first Z/T plane is used")
    args = parser.parse_args(argv)
    try:
        plot_rois(args.dapi, args.he, args.matrix, args.nucleus_boundaries, args.output_path,
                  n_points=args.n_points, tile_size_um=args.tile_size_um, dapi_mpp=args.dapi_mpp,
                  seed=args.seed, panel_pixels=args.panel_pixels, dapi_channel=args.dapi_channel)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
