#!/usr/bin/env python3
"""Warp a moving OME-TIFF into fixed H&E level-0 pixel space.

Example:
    python scripts/post_process/warp_dapi_to_he.py \
        --dapi_img=dapi.ome.tif \
        --he_img=he.ome.tif \
        --matrix=dapi_to_he_homography_level0.json \
        --output_img=dapi_in_he_space.ome.tif

The transform may be PHARAOH's canonical JSON, including a compact-support local
residual field, or a legacy headerless 3x3 CSV/TXT matrix. It consumes moving
level-0 pixels and produces fixed H&E level-0 pixels.

The output grid IS the H&E level-0 grid.  Anything the homography sends outside
it -- negative coordinates included -- is cropped, and every output pixel with
no DAPI source is filled with 0.  The OME header is rewritten for that grid:
SizeX/SizeY and PhysicalSizeX/Y come from the H&E reference, while dtype and
the channel description stay with the DAPI pixels.

``--he_img`` supplies the output extent and pixel spacing.  Use
``--output_size WxH`` when no H&E OME-TIFF is at hand, and ``--output_mpp`` to
override the recorded spacing.
"""

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

from pharaoh_transform import load_matrix as load_homography
from pharaoh_transform import load_transform, read_run_image_paths, resolve_run_transform


# OME PhysicalSize units expressed in micrometers.
_UM_PER_OME_UNIT = {
    "pm": 1e-6,
    "nm": 1e-3,
    "µm": 1.0,
    "μm": 1.0,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}

DEFAULT_TILE_SIZE = 512
DEFAULT_BLOCK_SIZE = 2048
DEFAULT_COMPRESSION = "zlib"
DEFAULT_INTERPOLATION = "linear"

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

# Depths cv2.warpPerspective handles natively; anything else is warped as
# float32 and cast back so the output keeps the DAPI bit depth.
_CV2_WARP_DTYPES = frozenset(
    {np.dtype(name) for name in ("uint8", "uint16", "int16", "float32", "float64")}
)

# Interpolation reaches past the mapped rectangle, so the source read is grown
# by a couple of pixels on every side.
_SOURCE_MARGIN_PX = 2

# Bound one source read and one resize operation, staying below OpenCV's
# INT_MAX element limit.
_MAX_SOURCE_BLOCK_ELEMS = 1_200_000_000
_MAX_RESIZE_ELEMS = 1_200_000_000


def _positive_mpp(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def read_ome_pixel_size_um(tif, series=0):
    """Return level-0 (x, y) spacing in micrometers from OME-XML."""
    metadata = tif.ome_metadata if tif.is_ome else None
    if not metadata:
        return None
    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return None
        values = []
        for axis in ("X", "Y"):
            value = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
            factor = _UM_PER_OME_UNIT.get(unit)
            if factor is None or not math.isfinite(value) or value <= 0:
                return None
            values.append(value * factor)
        return tuple(values)
    except (IndexError, KeyError, TypeError, ValueError, ET.ParseError):
        return None


def read_ome_channel_names(tif, series=0):
    """Return the OME channel names of one series, or an empty list."""
    metadata = tif.ome_metadata if tif.is_ome else None
    if not metadata:
        return []
    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return []
        return [
            channel.attrib.get("Name", "")
            for channel in pixels.findall("./{*}Channel")
        ]
    except (IndexError, TypeError, ValueError, ET.ParseError):
        return []


def _axes_for_shape(shape):
    """Guess the axis labels of a level whose series carries none."""
    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        return "YXS" if shape[-1] in (1, 2, 3, 4) else "CYX"
    # Unknown leading dimensions are treated as singleton time/Z dimensions.
    return "T" * (len(shape) - 2) + "YX"


class Level0Reader:
    """Read level-0 rectangles of an OME-TIFF without materializing the level."""

    def __init__(self, path, channel=None):
        import zarr

        self.path = Path(path)
        self._tif = tifffile.TiffFile(str(self.path))
        if not self._tif.series:
            raise ValueError(f"input contains no image series: {self.path}")
        series = self._tif.series[0]
        level0 = (list(getattr(series, "levels", []) or []) or [series])[0]

        self._store = level0.aszarr()
        root = zarr.open(self._store, mode="r")
        if isinstance(root, zarr.Array):
            self._array = root
        else:
            arrays = list(root.arrays())
            if not arrays:
                raise RuntimeError(f"no array found in TIFF zarr store for {self.path}")
            self._array = arrays[0][1]

        self.axes = str(
            getattr(level0, "axes", "") or _axes_for_shape(self._array.shape)
        ).upper()
        if "Y" not in self.axes or "X" not in self.axes:
            raise ValueError(
                f"cannot identify Y/X dimensions: shape={self._array.shape}, "
                f"axes={self.axes or '?'}"
            )
        if len(self.axes) != len(self._array.shape):
            self.axes = _axes_for_shape(self._array.shape)

        self.height = int(self._array.shape[self.axes.index("Y")])
        self.width = int(self._array.shape[self.axes.index("X")])
        self.dtype = np.dtype(self._array.dtype)
        self.channel_names = read_ome_channel_names(self._tif, 0)
        self._resolve_channels(channel)

    def _resolve_channels(self, channel):
        """Decide whether to keep interleaved samples or one plane."""
        axis = next((letter for letter in "SC" if letter in self.axes), None)
        self.channel_index = None
        self.interleaved = False
        self.channels = 1
        if axis is None:
            if channel not in (None, 0):
                raise ValueError(
                    f"--channel={channel} was requested but {self.path.name} has no "
                    "channel dimension"
                )
            return

        size = int(self._array.shape[self.axes.index(axis)])
        if channel is None and axis == "S" and size in (3, 4):
            # Interleaved RGB(A): keep every sample so colour survives the warp.
            self.interleaved = True
            self.channels = size
            return
        if channel is None and size == 1:
            self.channel_index = 0
            return
        index = 0 if channel is None else int(channel)
        if not 0 <= index < size:
            raise ValueError(
                f"--channel={index} is out of range; {self.path.name} has {size} "
                "channels"
            )
        self.channel_index = index

    @property
    def channel_label(self):
        if self.interleaved:
            return None
        index = self.channel_index or 0
        if index < len(self.channel_names) and self.channel_names[index]:
            return self.channel_names[index]
        return None

    def read_region(self, x0, y0, x1, y1):
        """Return the [y0:y1, x0:x1] rectangle as YX or interleaved YXC."""
        index = []
        kept = []
        for letter, size in zip(self.axes, self._array.shape):
            if letter == "Y":
                index.append(slice(int(y0), int(y1)))
                kept.append("Y")
            elif letter == "X":
                index.append(slice(int(x0), int(x1)))
                kept.append("X")
            elif letter in "SC":
                if self.interleaved:
                    index.append(slice(0, int(size)))
                    kept.append("C")
                else:
                    index.append(int(self.channel_index or 0))
            else:
                index.append(0)
        region = np.asarray(self._array[tuple(index)])
        order = ["Y", "X"] + (["C"] if "C" in kept else [])
        return np.transpose(region, [kept.index(letter) for letter in order])

    def close(self):
        try:
            if self._store is not None and hasattr(self._store, "close"):
                self._store.close()
        finally:
            self._tif.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_he_reference(path):
    """Return the H&E level-0 (width, height) and (x, y) micrometers per pixel."""
    he_path = Path(path).expanduser().resolve()
    if not he_path.is_file():
        raise FileNotFoundError(f"H&E image not found: {he_path}")
    with tifffile.TiffFile(str(he_path)) as tif:
        if not tif.series:
            raise ValueError(f"H&E image contains no image series: {he_path}")
        series = tif.series[0]
        level0 = (list(getattr(series, "levels", []) or []) or [series])[0]
        shape = tuple(int(size) for size in level0.shape)
        axes = str(getattr(level0, "axes", "") or _axes_for_shape(shape)).upper()
        if len(axes) != len(shape) or "Y" not in axes or "X" not in axes:
            axes = _axes_for_shape(shape)
        if "Y" not in axes or "X" not in axes:
            raise ValueError(
                f"cannot identify Y/X dimensions of the H&E level 0: shape={shape}"
            )
        height = shape[axes.index("Y")]
        width = shape[axes.index("X")]
        pixel_size = read_ome_pixel_size_um(tif, 0)
    return width, height, pixel_size


def _translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _perspective_points(points, matrix):
    """Map Nx2 points through a homography, returning None on a horizon hit."""
    points = np.asarray(points, dtype=np.float64)
    denominator = (
        matrix[2, 0] * points[:, 0] + matrix[2, 1] * points[:, 1] + matrix[2, 2]
    )
    if not np.all(np.isfinite(denominator)) or np.any(np.abs(denominator) <= 1e-12):
        return None
    if np.any(denominator > 0) and np.any(denominator < 0):
        # The block straddles the homography's horizon, so its preimage is not
        # a bounded quadrilateral and no finite source window exists.
        return None
    x = (
        matrix[0, 0] * points[:, 0] + matrix[0, 1] * points[:, 1] + matrix[0, 2]
    ) / denominator
    y = (
        matrix[1, 0] * points[:, 0] + matrix[1, 1] * points[:, 1] + matrix[1, 2]
    ) / denominator
    mapped = np.stack([x, y], axis=1)
    if not np.all(np.isfinite(mapped)):
        return None
    return mapped


def _source_window(inverse, x0, y0, x1, y1, source_width, source_height):
    """Return the source rectangle feeding one output block, or None."""
    # A homography maps lines to lines, so the preimage of the output block is a
    # quadrilateral and its corners bound the source read exactly.
    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    mapped = _perspective_points(corners, inverse)
    if mapped is None:
        raise ValueError(
            f"the homography has no bounded inverse over output block "
            f"x=[{x0}, {x1}) y=[{y0}, {y1}); it is not usable for image warping"
        )
    low = np.floor(mapped.min(axis=0)) - _SOURCE_MARGIN_PX
    high = np.ceil(mapped.max(axis=0)) + _SOURCE_MARGIN_PX
    sx0 = int(max(0, min(source_width, low[0])))
    sy0 = int(max(0, min(source_height, low[1])))
    sx1 = int(max(0, min(source_width, high[0])))
    sy1 = int(max(0, min(source_height, high[1])))
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    if (sx1 - sx0) * (sy1 - sy0) > _MAX_SOURCE_BLOCK_ELEMS:
        raise ValueError(
            f"output block x=[{x0}, {x1}) y=[{y0}, {y1}) needs a "
            f"{sx1 - sx0}x{sy1 - sy0} source region; lower --block_size"
        )
    return sx0, sy0, sx1, sy1


def warp_block(reader, transform, inverse, x0, y0, width, height, interpolation):
    """Warp the moving pixels covering one fixed-space output block."""
    if not transform.nonlinear:
        window = _source_window(
            inverse, x0, y0, x0 + width, y0 + height, reader.width, reader.height
        )
        if window is None:
            return None
        sx0, sy0, sx1, sy1 = window
        source = reader.read_region(sx0, sy0, sx1, sy1)
        if source.size == 0:
            return None
        block_matrix = (
            _translation(-x0, -y0)
            @ transform.matrix
            @ _translation(sx0, sy0)
        )
        original_dtype = source.dtype
        if original_dtype not in _CV2_WARP_DTYPES:
            source = source.astype(np.float32)
        warped = cv2.warpPerspective(
            np.ascontiguousarray(source),
            block_matrix,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        # A nonlinear warp must be evaluated inversely at every output pixel.
        # Coordinates remain level-0 throughout; cv2.remap receives coordinates
        # relative to the smallest source window covering this output block.
        yy, xx = np.indices((height, width), dtype=np.float64)
        fixed = np.stack([xx + float(x0), yy + float(y0)], axis=-1)
        moving = transform.inverse_points(fixed)
        finite = np.all(np.isfinite(moving), axis=-1)
        if not np.any(finite):
            return None
        low = np.floor(moving[finite].min(axis=0)).astype(np.int64) - _SOURCE_MARGIN_PX
        high = np.ceil(moving[finite].max(axis=0)).astype(np.int64) + _SOURCE_MARGIN_PX + 1
        sx0 = int(max(0, min(reader.width, low[0])))
        sy0 = int(max(0, min(reader.height, low[1])))
        sx1 = int(max(0, min(reader.width, high[0])))
        sy1 = int(max(0, min(reader.height, high[1])))
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        if (sx1 - sx0) * (sy1 - sy0) > _MAX_SOURCE_BLOCK_ELEMS:
            raise ValueError(
                f"output block x=[{x0}, {x0 + width}) y=[{y0}, {y0 + height}) "
                f"needs a {sx1 - sx0}x{sy1 - sy0} source region; lower --block_size"
            )
        source = reader.read_region(sx0, sy0, sx1, sy1)
        if source.size == 0:
            return None
        original_dtype = source.dtype
        if original_dtype not in _CV2_WARP_DTYPES:
            source = source.astype(np.float32)
        map_x = (moving[..., 0] - sx0).astype(np.float32)
        map_y = (moving[..., 1] - sy0).astype(np.float32)
        map_x[~finite] = -1.0
        map_y[~finite] = -1.0
        warped = cv2.remap(
            np.ascontiguousarray(source),
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    if warped.dtype != original_dtype:
        if np.issubdtype(original_dtype, np.integer):
            limits = np.iinfo(original_dtype)
            warped = np.clip(np.rint(warped), limits.min, limits.max)
        warped = warped.astype(original_dtype, copy=False)
    if warped.ndim == 2 and source.ndim == 3:
        warped = warped[..., None]
    return warped


def _downsample_half(array):
    """Area-downsample an H,W[,C] array by two, column block by column block."""
    height, width = array.shape[:2]
    channels = array.shape[2] if array.ndim == 3 else 1
    next_height = max(1, height // 2)
    next_width = max(1, width // 2)
    shape = (
        (next_height, next_width, channels)
        if array.ndim == 3
        else (next_height, next_width)
    )
    result = np.empty(shape, dtype=array.dtype)

    column_budget = max(2, int(_MAX_RESIZE_ELEMS // (max(1, height) * channels)))
    output_columns = max(1, max(2, (column_budget // 2) * 2) // 2)
    for start in range(0, next_width, output_columns):
        end = min(next_width, start + output_columns)
        block = np.ascontiguousarray(array[: next_height * 2, start * 2 : end * 2])
        resized = cv2.resize(
            block, (end - start, next_height), interpolation=cv2.INTER_AREA
        )
        if array.ndim == 3 and resized.ndim == 2:
            resized = resized[..., None]
        result[:, start:end] = resized
    return result


def _pyramid_shapes(height, width):
    """List the 2x downsampled shapes below one base level."""
    shapes = []
    while height > 1 and width > 1:
        height, width = max(1, height // 2), max(1, width // 2)
        shapes.append((height, width))
        if height == 1 or width == 1:
            break
    return shapes


def _level_write_options(base_options, pixel_size, base_shape, level_shape):
    """Add this level's resolution tags to the shared write options."""
    if pixel_size is None:
        return base_options
    scale_y = base_shape[0] / level_shape[0]
    scale_x = base_shape[1] / level_shape[1]
    return {
        **base_options,
        # Resolution is stored per centimeter, so it is 1e4 / micrometers.
        "resolution": (
            1e4 / (pixel_size[0] * scale_x),
            1e4 / (pixel_size[1] * scale_y),
        ),
        "resolutionunit": "CENTIMETER",
    }


def _iter_output_tiles(
    reader,
    transform,
    inverse,
    output_width,
    output_height,
    tile_size,
    block_size,
    interpolation,
    channels,
    dtype,
    half,
    show_progress,
):
    """Yield the output tiles in file order, halving each strip on the way."""
    strip_shape = (block_size, output_width) + ((channels,) if channels > 1 else ())
    progress = tqdm(
        total=output_height,
        desc="Warp DAPI",
        unit="rows",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for y0 in range(0, output_height, block_size):
            rows = min(block_size, output_height - y0)
            strip = np.zeros((rows,) + strip_shape[1:], dtype=dtype)
            for x0 in range(0, output_width, block_size):
                columns = min(block_size, output_width - x0)
                block = warp_block(
                    reader, transform, inverse, x0, y0, columns, rows, interpolation
                )
                if block is not None:
                    strip[:, x0 : x0 + columns] = block
            if half is not None and rows >= 2:
                # Whole-image halving drops a trailing odd row, and so does this
                # per-strip halving: only the final strip can have odd height.
                half_rows = rows // 2
                half[y0 // 2 : y0 // 2 + half_rows] = _downsample_half(
                    strip[: half_rows * 2]
                )
            for tile_y in range(0, rows, tile_size):
                for tile_x in range(0, output_width, tile_size):
                    tile = strip[
                        tile_y : tile_y + tile_size, tile_x : tile_x + tile_size
                    ]
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        padded = np.zeros(
                            (tile_size, tile_size) + tile.shape[2:], dtype=dtype
                        )
                        padded[: tile.shape[0], : tile.shape[1]] = tile
                        tile = padded
                    yield tile
            progress.update(rows)
    finally:
        progress.close()


def _report_coverage(reader, transform, output_width, output_height):
    """Print the approximate moving-image coverage in the fixed grid."""
    if transform.nonlinear:
        samples = np.linspace(0.0, 1.0, 65)
        boundary = np.concatenate(
            [
                np.column_stack([samples * reader.width, np.zeros_like(samples)]),
                np.column_stack([np.full_like(samples, reader.width), samples * reader.height]),
                np.column_stack([(1.0 - samples) * reader.width, np.full_like(samples, reader.height)]),
                np.column_stack([np.zeros_like(samples), (1.0 - samples) * reader.height]),
            ]
        )
    else:
        boundary = np.array(
            [[0.0, 0.0], [reader.width, 0.0],
             [reader.width, reader.height], [0.0, reader.height]]
        )
    try:
        mapped = transform.forward_points(boundary)
    except ValueError:
        print(
            "[warp] moving extent cannot be mapped as a bounded region; "
            "coverage is not reported",
            flush=True,
        )
        return
    hull = cv2.convexHull(mapped.astype(np.float32)).reshape(-1, 2)
    low = mapped.min(axis=0)
    high = mapped.max(axis=0)
    print(
        f"[warp] DAPI extent in H&E pixels: x=[{low[0]:.1f}, {high[0]:.1f}], "
        f"y=[{low[1]:.1f}, {high[1]:.1f}]",
        flush=True,
    )
    output_rect = np.array(
        [
            [0.0, 0.0],
            [output_width, 0.0],
            [output_width, output_height],
            [0.0, output_height],
        ],
        dtype=np.float32,
    )
    intersection, _ = cv2.intersectConvexConvex(
        hull.astype(np.float32), output_rect
    )
    quad_area = float(cv2.contourArea(hull.astype(np.float32)))
    if quad_area > 0:
        kept = 100.0 * intersection / quad_area
        print(
            f"[warp] {kept:.1f}% of the warped DAPI falls inside the H&E grid; "
            f"the rest is cropped",
            flush=True,
        )
    if intersection > 0:
        filled = 100.0 * intersection / float(output_width * output_height)
        print(
            f"[warp] {filled:.1f}% of the output carries DAPI signal; "
            "the remainder is 0",
            flush=True,
        )
    else:
        print(
            "[warp] WARNING: the warped DAPI does not overlap the H&E grid; "
            "the output would be empty",
            flush=True,
        )


def _parse_output_size(text):
    """Parse a ``WxH`` output size."""
    parts = str(text).lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("output_size must look like WIDTHxHEIGHT, e.g. 40000x30000")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("output_size must contain two integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("output_size must be positive")
    return width, height


def warp_dapi_to_he(
    dapi_img,
    output_img,
    matrix_path,
    *,
    he_img=None,
    output_size=None,
    output_mpp=None,
    channel=None,
    interpolation=DEFAULT_INTERPOLATION,
    tile_size=DEFAULT_TILE_SIZE,
    block_size=DEFAULT_BLOCK_SIZE,
    compression=DEFAULT_COMPRESSION,
    pyramid=True,
    show_progress=True,
):
    """Warp one DAPI OME-TIFF into H&E space and return the output path."""
    source = Path(dapi_img).expanduser().resolve()
    destination = Path(output_img).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"DAPI image not found: {source}")
    if source == destination:
        raise ValueError("dapi_img and output_img must be different files")
    if not destination.name.lower().endswith((".ome.tif", ".ome.tiff")):
        raise ValueError("output_img must end with .ome.tif or .ome.tiff")
    if he_img is None and output_size is None:
        raise ValueError(
            "the output extent is unknown; pass --he_img or --output_size WxH"
        )
    if interpolation not in INTERPOLATIONS:
        raise ValueError(
            f"unsupported interpolation: {interpolation}; expected one of "
            f"{', '.join(sorted(INTERPOLATIONS))}"
        )
    if compression not in {"zlib", "lzw", "none"}:
        raise ValueError(f"unsupported compression: {compression}")
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("tile size must be a positive multiple of 16")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    block_size = max(tile_size, (block_size // tile_size) * tile_size)

    transform = load_transform(matrix_path)
    inverse = None if transform.nonlinear else np.linalg.inv(transform.matrix)
    if transform.nonlinear and block_size > tile_size:
        # Dense inverse maps are evaluated per output block. Keep a field-aware
        # block bounded in memory and close to the TIFF tile granularity.
        print(
            f"[warp] local field: reducing block_size {block_size} -> {tile_size}",
            flush=True,
        )
        block_size = tile_size

    pixel_size = None
    if he_img is not None:
        he_width, he_height, pixel_size = read_he_reference(he_img)
        print(
            f"[warp] H&E reference: {Path(he_img).name} "
            f"level 0 = {he_width}x{he_height}",
            flush=True,
        )
        if pixel_size is None:
            print(
                "[warp] H&E OME header carries no PhysicalSizeX/Y "
                "(pass --output_mpp to record the spacing)",
                flush=True,
            )
    if output_size is not None:
        output_width, output_height = output_size
    else:
        output_width, output_height = he_width, he_height
    if output_mpp is not None:
        value = _positive_mpp(output_mpp, "output_mpp")
        pixel_size = (value, value)

    print(f"[warp] input:  {source}", flush=True)
    print(f"[warp] transform: {transform.source_path}", flush=True)
    print(f"[warp] transform type: {transform.transform_type}", flush=True)
    print("[warp] direction: moving level-0 px -> fixed H&E level-0 px", flush=True)

    with Level0Reader(source, channel=channel) as reader:
        channels = reader.channels
        dtype = reader.dtype
        channel_label = reader.channel_label
        print(
            f"[warp] DAPI level 0: {reader.width}x{reader.height} "
            f"axes={reader.axes} dtype={dtype}",
            flush=True,
        )
        if reader.interleaved:
            print(f"[warp] keeping {channels} interleaved samples", flush=True)
        elif reader.channel_index is not None:
            label = f" ({channel_label})" if channel_label else ""
            print(f"[warp] using channel {reader.channel_index}{label}", flush=True)
        print(
            f"[warp] output: {output_width}x{output_height} "
            f"(H&E grid), interpolation={interpolation}",
            flush=True,
        )
        if pixel_size is not None:
            print(
                f"[warp] output pixel size: x={pixel_size[0]:.6g} um/px, "
                f"y={pixel_size[1]:.6g} um/px",
                flush=True,
            )
        else:
            print(
                "[warp] output carries no PhysicalSize; pass --he_img with OME "
                "spacing or --output_mpp",
                flush=True,
            )
        _report_coverage(reader, transform, output_width, output_height)

        level_shapes = _pyramid_shapes(output_height, output_width) if pyramid else []
        half = None
        if level_shapes:
            half_shape = level_shapes[0] + ((channels,) if channels > 1 else ())
            half = np.zeros(half_shape, dtype=dtype)

        base_shape = (output_height, output_width) + (
            (channels,) if channels > 1 else ()
        )
        photometric = "rgb" if channels > 1 else "minisblack"
        metadata = {"axes": "YXS" if channels > 1 else "YX"}
        if pixel_size is not None:
            metadata.update(
                PhysicalSizeX=pixel_size[0],
                PhysicalSizeXUnit="µm",
                PhysicalSizeY=pixel_size[1],
                PhysicalSizeYUnit="µm",
            )
        if channel_label:
            metadata["Channel"] = {"Name": [channel_label]}
        metadata["Description"] = (
            f"{source.name} warped into the {Path(he_img).name if he_img else 'H&E'} "
            f"level-0 pixel grid with {Path(matrix_path).name}"
        )
        write_options = {
            "photometric": photometric,
            "tile": (tile_size, tile_size),
            "compression": None if compression == "none" else compression,
        }

        print(
            f"[warp] writing pyramidal OME-TIFF -> {destination} "
            f"(base + {len(level_shapes)} levels, tile={tile_size}, "
            f"block={block_size}, compression={compression})",
            flush=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name.split('.')[0]}.",
            suffix="".join(destination.suffixes),
            dir=destination.parent,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with tifffile.TiffWriter(temporary, bigtiff=True, ome=True) as writer:
                writer.write(
                    _iter_output_tiles(
                        reader,
                        transform,
                        inverse,
                        output_width,
                        output_height,
                        tile_size,
                        block_size,
                        INTERPOLATIONS[interpolation],
                        channels,
                        dtype,
                        half,
                        show_progress,
                    ),
                    shape=base_shape,
                    dtype=dtype,
                    subifds=len(level_shapes),
                    metadata=metadata,
                    **_level_write_options(
                        write_options,
                        pixel_size,
                        base_shape[:2],
                        base_shape[:2],
                    ),
                )
                level = half
                for index, shape in enumerate(level_shapes, start=1):
                    if level.shape[:2] != shape:
                        raise RuntimeError(
                            f"level {index} shape {level.shape[:2]} does not match "
                            f"the declared {shape}"
                        )
                    print(f"[warp]   level {index}: {shape[1]}x{shape[0]}", flush=True)
                    writer.write(
                        level,
                        subfiletype=1,
                        **_level_write_options(
                            write_options, pixel_size, base_shape[:2], shape
                        ),
                    )
                    if index < len(level_shapes):
                        level = _downsample_half(level)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    print(
        f"[warp] DONE: {destination} "
        f"({destination.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Warp a DAPI OME-TIFF into the H&E level-0 pixel grid using a "
            "moving-level0 to fixed-H&E-level0 PHARAOH transform."
        )
    )
    parser.add_argument(
        "--dapi_img", "--dapi-img", default=None,
        help="defaults to the moving path recorded by --run-dir",
    )
    parser.add_argument("--output_img", "--output-img", required=True)
    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument(
        "--matrix", "--transform",
        help="PHARAOH transform JSON, or a headerless 3x3 matrix in CSV/TXT format",
    )
    transform_group.add_argument(
        "--run_dir", "--run-dir",
        help=(
            "completed PHARAOH run; automatically uses the canonical final JSON "
            "(including a dense local field) or its matrix-only CSV fallback"
        ),
    )
    parser.add_argument(
        "--he_img",
        "--he-img",
        default=None,
        help="H&E OME-TIFF supplying the output extent and pixel spacing",
    )
    parser.add_argument(
        "--output_size",
        "--output-size",
        default=None,
        help="output WIDTHxHEIGHT in H&E level-0 pixels; replaces the extent "
        "taken from --he_img",
    )
    parser.add_argument(
        "--output_mpp",
        "--output-mpp",
        type=float,
        default=None,
        help="micrometers per pixel recorded in the output header; defaults to "
        "the H&E PhysicalSizeX/Y",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="DAPI channel to warp when the input has a channel dimension "
        "(default: 0, or all samples for interleaved RGB)",
    )
    parser.add_argument(
        "--interpolation",
        default=DEFAULT_INTERPOLATION,
        choices=sorted(INTERPOLATIONS),
        help=f"resampling kernel (default: {DEFAULT_INTERPOLATION})",
    )
    parser.add_argument(
        "--tile_size",
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"output TIFF tile size (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--block_size",
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"output pixels warped per step (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--compression",
        default=DEFAULT_COMPRESSION,
        choices=("zlib", "lzw", "none"),
        help=f"output compression (default: {DEFAULT_COMPRESSION})",
    )
    parser.add_argument(
        "--no_pyramid",
        "--no-pyramid",
        action="store_true",
        help="write only the base level instead of a 2x pyramid",
    )
    args = parser.parse_args(argv)
    try:
        output_size = (
            None if args.output_size is None else _parse_output_size(args.output_size)
        )
        transform_path = (
            resolve_run_transform(args.run_dir) if args.run_dir else args.matrix
        )
        if args.run_dir:
            recorded_dapi, recorded_he = read_run_image_paths(args.run_dir)
        else:
            recorded_dapi = recorded_he = None
        dapi_img = args.dapi_img or recorded_dapi
        he_img = args.he_img or recorded_he
        if not dapi_img:
            raise ValueError("pass --dapi-img when using --transform")
        warp_dapi_to_he(
            dapi_img,
            args.output_img,
            transform_path,
            he_img=he_img,
            output_size=output_size,
            output_mpp=args.output_mpp,
            channel=args.channel,
            interpolation=args.interpolation,
            tile_size=args.tile_size,
            block_size=args.block_size,
            compression=args.compression,
            pyramid=not args.no_pyramid,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        tifffile.TiffFileError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
