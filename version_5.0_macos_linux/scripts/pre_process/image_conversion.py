"""Shared BTF/TIFF/NDPI/SVS/JPEG/CZI to pyramidal OME-TIFF conversion implementation."""

import contextlib
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageOps

try:
    import cv2

    _HAVE_CV2 = True
except Exception:  # pragma: no cover - NumPy fallback remains available
    _HAVE_CV2 = False

try:
    import zarr

    _HAVE_ZARR = True
except Exception:  # pragma: no cover - whole-array reads remain available
    _HAVE_ZARR = False


JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
# NDPI is a TIFF variant, so tifffile reads it with the same code path; it only
# needs its own suffix set because its series layout and defaults differ.
NDPI_SUFFIXES = frozenset({".ndpi"})
# Aperio SVS is also a TIFF: a tiled JPEG baseline, its pyramid as further
# pages, and thumbnail/label/macro overviews beside them.
SVS_SUFFIXES = frozenset({".svs"})
TIFF_SUFFIXES = frozenset({".btf", ".tif", ".tiff"}) | NDPI_SUFFIXES | SVS_SUFFIXES
# Zeiss CZI is not a TIFF: libCZI (via aicspylibczi) assembles the mosaic tiles.
CZI_SUFFIXES = frozenset({".czi"})
SUPPORTED_SUFFIXES = JPEG_SUFFIXES | TIFF_SUFFIXES | CZI_SUFFIXES

# Bound each resize operation and stay below OpenCV's INT_MAX element limit.
_MAX_STRIP_ELEMS = 1_200_000_000

# A slide scan is read in horizontal strips of this many tile rows, so it
# never has to fit in memory at full resolution (see `_StripSource`).
_STRIP_TILE_ROWS = 4

# Overview images a scanner stores next to the baseline pyramid: Hamamatsu's
# NDPI macro and map, Aperio's SVS thumbnail, label and macro.
_OVERVIEW_SERIES_NAMES = frozenset({"Macro", "Map", "Label", "Thumbnail"})

# Aperio writes the scan calibration into the ImageDescription, e.g. "|MPP = 0.2522|".
_SVS_MPP = re.compile(r"\|\s*MPP\s*=\s*([0-9.eE+-]+)")

# libCZI pixel type names (as aicspylibczi reports them, lower-case) and the
# number of samples each carries. Bgr* samples are stored blue-first.
_CZI_SAMPLES_PER_PIXEL = {
    "gray8": 1,
    "gray16": 1,
    "gray32float": 1,
    "bgr24": 3,
    "bgr48": 3,
    "bgr96float": 3,
    "bgra32": 4,
}
_CZI_DTYPES = {
    "gray8": np.uint8,
    "gray16": np.uint16,
    "gray32float": np.float32,
    "bgr24": np.uint8,
    "bgr48": np.uint16,
    "bgr96float": np.float32,
    "bgra32": np.uint8,
}

# TIFF ResolutionUnit codes; 1 (none) carries no physical meaning.
_UM_PER_RESOLUTION_UNIT = {2: 25400.0, 3: 10000.0}

# OME PhysicalSize units expressed in micrometers.
_UM_PER_OME_UNIT = {
    "pm": 1e-6,
    "nm": 1e-3,
    "\u00b5m": 1.0,
    "\u03bcm": 1.0,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}


def image_kind(src):
    """Return the reader family selected by the input extension."""
    suffix = Path(src).suffix.lower()
    if suffix in JPEG_SUFFIXES:
        return "jpeg"
    if suffix in TIFF_SUFFIXES:
        return "tiff"
    if suffix in CZI_SUFFIXES:
        return "czi"
    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise ValueError(f"unsupported input extension; expected one of: {supported}")


def output_path(src, *, scene=None):
    """Replace the input's final extension with ``.ome.tif``.

    Converting one scene of a multi-scene CZI names the scene in the output
    (``slide_scene1.ome.tif``), so the scenes of one slide never overwrite each
    other; ``scene="all"`` or no scene keeps the plain name.
    """
    src = Path(src)
    if scene is None or scene == "all":
        return src.with_suffix(".ome.tif")
    return src.with_name(f"{src.stem}_scene{int(scene)}.ome.tif")


def parse_scene(value):
    """Accept a scene index or ``all`` (the whole mosaic); None passes through."""
    if value is None or isinstance(value, bool):
        return value if value is None else int(value)
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return "all"
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"scene must be an index or 'all', not {value!r}")
    if int(value) < 0:
        raise ValueError("scene must be >= 0")
    return int(value)


def _read_jpeg(src):
    """Read a JPEG as uint8 YX or YXC while honoring EXIF orientation."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        normalized = image if image.mode == "L" else image.convert("RGB")
        normalized.load()
        return np.asarray(normalized).copy()


def _to_yxc(arr):
    """Normalize a TIFF array to grayscale YX or interleaved YXC."""
    if arr.ndim == 2:
        return arr, "minisblack"
    if arr.ndim == 3:
        if arr.shape[0] in (2, 3, 4) and arr.shape[-1] not in (2, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        channels = arr.shape[-1]
        if channels == 1:
            return arr[..., 0], "minisblack"
        if channels in (3, 4):
            return arr, "rgb"
        return arr, "minisblack"
    raise ValueError(f"unsupported TIFF array shape: {arr.shape}")


def _rational(value):
    """Flatten a TIFF RATIONAL tag value to a float."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value
        if not denominator:
            return None
        return float(numerator) / float(denominator)
    return float(value)


def _resolution_pixel_size_um(page, *, default_unit):
    """Read (x, y) micrometers per pixel from the TIFF resolution tags.

    NDPI carries the scanner calibration here: Hamamatsu writes pixels per
    centimeter, so 22743 px/cm becomes 0.4397 um/px.
    """
    x_resolution = _rational(page.tags.valueof("XResolution"))
    y_resolution = _rational(page.tags.valueof("YResolution"))
    if not x_resolution or not y_resolution:
        return None
    unit = page.tags.valueof("ResolutionUnit")
    unit = default_unit if unit is None else int(unit)
    micrometers = _UM_PER_RESOLUTION_UNIT.get(unit)
    if micrometers is None:
        return None
    size = (micrometers / x_resolution, micrometers / y_resolution)
    if not all(np.isfinite(value) and value > 0 for value in size):
        return None
    return size


def _ome_pixel_size_um(tif, series):
    """Read (x, y) micrometers per pixel from OME-XML, when the input is OME."""
    metadata = tif.ome_metadata if tif.is_ome else None
    if not metadata:
        return None
    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return None
        size = []
        for axis in ("X", "Y"):
            value = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "\u00b5m")
            factor = _UM_PER_OME_UNIT.get(unit)
            if factor is None or not np.isfinite(value) or value <= 0:
                return None
            size.append(value * factor)
        return tuple(size)
    except (IndexError, KeyError, TypeError, ValueError, ET.ParseError):
        return None


def _ome_channel_names(tif, series):
    """Return the OME channel names for one image, preserving empty names."""
    metadata = tif.ome_metadata if tif.is_ome else None
    if not metadata:
        return ()
    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return ()
        return tuple(
            channel.attrib.get("Name")
            for channel in pixels.findall("./{*}Channel")
        )
    except (IndexError, TypeError, ValueError, ET.ParseError):
        return ()


def _resolve_tiff_channel(tif, series, count, requested):
    """Resolve an OME-TIFF C plane by zero-based index or channel name."""
    names = _ome_channel_names(tif, series)
    if requested is None:
        if count != 1:
            labels = ", ".join(
                f"{index}:{name or '?'}" for index, name in enumerate(names[:count])
            )
            detail = f" ({labels})" if labels else ""
            raise ValueError(
                f"TIFF series {series} has {count} channels{detail}; pass "
                "--channel INDEX_OR_NAME to select one (for example --channel 0 "
                "or --channel DAPI)"
            )
        index = 0
    else:
        try:
            index = int(requested)
        except (TypeError, ValueError):
            wanted = str(requested).strip().casefold()
            matches = [
                index
                for index, name in enumerate(names[:count])
                if name is not None and name.strip().casefold() == wanted
            ]
            if len(matches) != 1:
                available = ", ".join(
                    f"{index}:{name or '?'}"
                    for index, name in enumerate(names[:count])
                )
                raise ValueError(
                    f"TIFF channel {requested!r} was not found; available channels: "
                    f"{available or f'0..{count - 1}'}"
                )
            index = matches[0]
    if not 0 <= index < count:
        raise ValueError(
            f"channel {requested!r} is out of range; TIFF series {series} has "
            f"channels 0..{count - 1}"
        )
    name = names[index] if index < len(names) else None
    return index, name


def _svs_pixel_size_um(page):
    """Read micrometers per pixel from an Aperio SVS ImageDescription."""
    match = _SVS_MPP.search(page.description or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return (value, value)


def _select_series(tif, series):
    """Resolve the requested series, or pick the slide baseline automatically.

    An NDPI or SVS file stores the slide overviews (macro, map, label,
    thumbnail) as extra series beside the baseline pyramid, and their order is
    not guaranteed, so the default picks the largest non-overview series
    instead of index 0.
    """
    if series is not None:
        series = int(series)
        if series < 0:
            raise ValueError("series must be >= 0")
        if series >= len(tif.series):
            raise ValueError(
                f"series {series} is out of range; input has {len(tif.series)} series"
            )
        return series
    if not (tif.is_ndpi or getattr(tif, "is_svs", False)):
        return 0
    candidates = [
        index
        for index, item in enumerate(tif.series)
        if (item.name or "") not in _OVERVIEW_SERIES_NAMES
    ] or list(range(len(tif.series)))
    return max(
        candidates,
        key=lambda index: int(np.prod(tif.series[index].shape[:2])),
    )


def _squeeze_to_plane(arr, axes):
    """Drop singleton non-YX axes, e.g. the Z axis of a single-plane NDPI scan.

    A shaped ``(1, Y, X)`` TIFF is the 3-D case: without this, `_to_yxc` would
    take its leading 1 for a channel axis and its width for the sample count.
    """
    if arr.ndim <= 2:
        return arr
    axes = (axes or "").upper()
    keep = tuple(
        index
        for index, size in enumerate(arr.shape)
        if size > 1 or (index < len(axes) and axes[index] in "YX")
    )
    squeezed = arr.reshape(tuple(arr.shape[index] for index in keep))
    if squeezed.ndim > 3:
        raise ValueError(
            f"unsupported multi-plane series: shape={arr.shape} axes={axes or '?'}; "
            "pass --series to select a single-plane image"
        )
    return squeezed


def _tiff_pixel_size_um(tif, index):
    """Pixel spacing of one series: OME-XML, then the Aperio MPP, then TIFF tags."""
    keyframe = tif.series[index].keyframe
    size = _ome_pixel_size_um(tif, index)
    if size is None and getattr(tif, "is_svs", False):
        size = _svs_pixel_size_um(keyframe)
    if size is None:
        # NDPI resolution tags are per centimeter; plain TIFF defaults to inch.
        size = _resolution_pixel_size_um(
            keyframe, default_unit=3 if tif.is_ndpi else 2
        )
    return size


class _StripSource:
    """One image plane delivered in strips, so the base level is never held whole.

    A slide scan assembled at full resolution is tens of gigabytes. The writer
    pulls the base level tile by tile from `tiles()`, which reads one strip of
    tile rows at a time through `_read_strip` and, as a side effect,
    area-downsamples each strip into `level1`. Every further level is built
    from `level1` in memory, a quarter of the base. Subclasses set `shape`,
    `dtype`, `photometric`, `pixel_size`, `name`, and `streams`; a source that
    cannot stream (`streams` False) is read whole through `read_all()`.
    """

    streams = True

    def _read_strip(self, row, rows):
        raise NotImplementedError

    def read_all(self):
        raise NotImplementedError

    def describe(self):
        return self.name

    def tiles(self, tile_size, level1=None):
        """Yield the base level as row-major tiles, filling `level1` on the way."""
        height, width = self.shape[:2]
        strip_rows = tile_size * _STRIP_TILE_ROWS
        pad = [(0, 0)] * len(self.shape)
        for row in range(0, height, strip_rows):
            rows = min(strip_rows, height - row)
            strip = self._read_strip(row, rows)
            if strip.shape[:2] != (rows, width):
                raise ValueError(
                    f"{self.name}: got {strip.shape[:2]} for a {rows}x{width} strip"
                )
            print(
                f"[convert]   rows {row}-{row + rows} of {height} read",
                flush=True,
            )
            if level1 is not None:
                # Done before the strip's tiles go out: the consumer stops
                # pulling after the last tile, so nothing after the final
                # yield would run. `row` is a multiple of an even strip
                # height, so the half-resolution rows land at row // 2.
                half = _downsample_half(strip)
                level1[row // 2 : row // 2 + half.shape[0]] = half
            for tile_row in range(0, rows, tile_size):
                for tile_col in range(0, width, tile_size):
                    tile = strip[
                        tile_row : tile_row + tile_size, tile_col : tile_col + tile_size
                    ]
                    if tile.shape[:2] != (tile_size, tile_size):
                        pad[0] = (0, tile_size - tile.shape[0])
                        pad[1] = (0, tile_size - tile.shape[1])
                        tile = np.pad(tile, pad)
                    yield np.ascontiguousarray(tile)


class _TiffSource(_StripSource):
    """One TIFF/BTF/NDPI/SVS series, streamed through tifffile's zarr store.

    A plain YX/YXS series, or one selected C plane from an OME mIF series, is
    read in strips so a slide scan converts within a fraction of its size.
    Other layouts are read whole and normalized through `_squeeze_to_plane`
    and `_to_yxc` when they represent one image plane.
    """

    def __init__(self, tif, name, series, channel=None):
        self.tif = tif
        self.name = name
        self.index = _select_series(tif, series)
        self.selected = tif.series[self.index]
        self.pixel_size = _tiff_pixel_size_um(tif, self.index)
        level0 = self.selected.levels[0]
        axes = (level0.axes or "").upper()
        self._store = None
        self._array = None
        self.channel = None
        self.channel_name = None
        self.channel_names = ()
        self.all_channels = False
        self._read_key = None
        self._source_y_axis = None
        self._source_c_axis = None

        if "C" in axes:
            if axes.count("C") != 1 or len(axes) != len(level0.shape):
                raise ValueError(
                    f"unsupported TIFF channel layout: shape={level0.shape} axes={axes}"
                )
            channel_axis = axes.index("C")
            channel_count = int(level0.shape[channel_axis])
            keep_all = (
                isinstance(channel, str) and channel.strip().casefold() == "all"
            )
            key = []
            output_axes = []
            output_shape = []
            for axis, size in zip(axes, level0.shape):
                if axis == "C":
                    if keep_all:
                        key.append(slice(None))
                        output_axes.append(axis)
                        output_shape.append(int(size))
                    else:
                        self.channel, self.channel_name = _resolve_tiff_channel(
                            tif, self.index, channel_count, channel
                        )
                        key.append(self.channel)
                elif axis in "YXS":
                    key.append(slice(None))
                    output_axes.append(axis)
                    output_shape.append(int(size))
                elif int(size) == 1:
                    key.append(0)
                else:
                    raise ValueError(
                        f"unsupported multi-plane TIFF after selecting C={self.channel}: "
                        f"shape={level0.shape} axes={axes}; non-spatial axis {axis} "
                        f"has {size} planes"
                    )
            output_axes = "".join(output_axes)
            expected_axes = ("CYX",) if keep_all else ("YX", "YXS")
            if output_axes not in expected_axes:
                selection = "all channels" if keep_all else f"C={self.channel}"
                raise ValueError(
                    f"unsupported TIFF channel layout after selecting {selection}: "
                    f"shape={level0.shape} axes={axes}"
                )
            self._read_key = tuple(key)
            self._source_y_axis = axes.index("Y")
            self._source_c_axis = channel_axis
            self.shape = tuple(output_shape)
            self.dtype = np.dtype(level0.dtype)
            self.photometric = "minisblack"
            if output_axes == "YXS":
                samples = self.shape[2]
                self.photometric = "rgb" if samples in (3, 4) else "minisblack"
            self.streams = _HAVE_ZARR
            if keep_all:
                if not self.streams:
                    raise ValueError(
                        "--channel all requires the zarr package for bounded-memory "
                        "OME-TIFF streaming"
                    )
                names = _ome_channel_names(tif, self.index)
                self.channel_names = tuple(
                    names[index] if index < len(names) and names[index] else f"C{index}"
                    for index in range(channel_count)
                )
                self.all_channels = True
            elif not self.streams:
                # A canonical CYX OME-TIFF stores one page per channel. Read
                # only the selected page when zarr is unavailable instead of
                # materializing the complete mIF stack.
                if axes == "CYX" and self.channel < len(self.selected.pages):
                    base = self.selected.pages[self.channel].asarray()
                else:  # uncommon singleton Z/T layouts
                    base = np.asarray(self.selected.asarray())[self._read_key]
                self._array, self.photometric = _to_yxc(base)
                self.shape = self._array.shape
                self.dtype = self._array.dtype
        elif channel is not None:
            raise ValueError(
                f"--channel was given, but TIFF series {self.index} has no C axis "
                f"(shape={level0.shape}, axes={axes or '?'})"
            )
        else:
            self.streams = _HAVE_ZARR and axes in ("YX", "YXS")

        if self.channel is None and not self.all_channels and self.streams:
            samples = level0.shape[2] if len(level0.shape) == 3 else 1
            self.dtype = np.dtype(level0.dtype)
            if samples == 1:
                self.shape = tuple(level0.shape[:2])
                self.photometric = "minisblack"
            else:
                self.shape = tuple(level0.shape)
                self.photometric = "rgb" if samples in (3, 4) else "minisblack"
        elif self.channel is None and not self.all_channels:
            base = _squeeze_to_plane(
                self.selected.asarray(), getattr(self.selected, "axes", "")
            )
            self._array, self.photometric = _to_yxc(base)
            self.shape = self._array.shape
            self.dtype = self._array.dtype

    def describe(self):
        description = f"series {self.index} ({self.selected.name or 'unnamed'})"
        if self.all_channels:
            description += f", all {len(self.channel_names)} channels"
        elif self.channel is not None:
            label = f" ({self.channel_name})" if self.channel_name else ""
            description += f", channel {self.channel}{label}"
        return description

    def read_all(self):
        return self._array

    def _read_strip(self, row, rows):
        if self._store is None:
            self._store = self.selected.aszarr(level=0)
            self._zarr = zarr.open(self._store, mode="r")
        if self._read_key is None:
            strip = np.asarray(self._zarr[row : row + rows])
        else:
            key = list(self._read_key)
            key[self._source_y_axis] = slice(row, row + rows)
            strip = np.asarray(self._zarr[tuple(key)])
        if strip.ndim == 3 and strip.shape[2] == 1:
            strip = strip[..., 0]
        return strip

    def read_channel_strip(self, channel, row, rows):
        """Read one YX strip from a ``--channel all`` CYX source."""
        if not self.all_channels:
            raise ValueError("read_channel_strip requires --channel all")
        if self._store is None:
            self._store = self.selected.aszarr(level=0)
            self._zarr = zarr.open(self._store, mode="r")
        key = list(self._read_key)
        key[self._source_c_axis] = int(channel)
        key[self._source_y_axis] = slice(row, row + rows)
        strip = np.asarray(self._zarr[tuple(key)])
        if strip.ndim != 2:
            raise ValueError(
                f"expected a YX strip for channel {channel}, got shape={strip.shape}"
            )
        return strip

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None


@contextlib.contextmanager
def _open_tiff(src, series, channel=None):
    """Open one TIFF-family series as a `_TiffSource`."""
    with tifffile.TiffFile(src) as tif:
        source = _TiffSource(tif, Path(src).name, series, channel=channel)
        try:
            yield source
        finally:
            source.close()


def _czi_pixel_size_um(meta):
    """Read (x, y) micrometers per pixel from the CZI metadata ``Scaling`` block.

    Zeiss records the distance per pixel in meters, so 2.2e-07 becomes 0.22 um/px.
    """
    if meta is None:
        return None
    size = []
    for axis in ("X", "Y"):
        node = meta.find(f".//Scaling/Items/Distance[@Id='{axis}']/Value")
        if node is None or node.text is None:
            return None
        try:
            value = float(node.text) * 1e6
        except ValueError:
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        size.append(value)
    return tuple(size)


def _czi_plane(dims_shape, *, scene, channel):
    """Choose the single plane to read from a CZI and validate the request.

    ``dims_shape`` is aicspylibczi's ``{axis: (start, stop)}`` map. Every axis
    other than the image axes has to be pinned to one index: ``C`` from
    ``channel`` (required when the file has more than one channel), ``S`` from
    ``scene`` (the whole mosaic when omitted), and any remaining axis only when
    it is a singleton, since a Z-stack or time series has no single image to
    convert.
    """
    plane = {}
    for axis, (start, stop) in dims_shape.items():
        if axis in "XYMA":
            continue
        if axis == "C":
            if channel is None:
                if stop - start != 1:
                    raise ValueError(
                        f"the CZI has {stop - start} channels (C={start}..{stop - 1}); "
                        "pass --channel to select one"
                    )
                channel = start
            elif not start <= int(channel) < stop:
                raise ValueError(
                    f"channel {channel} is out of range; input has C={start}..{stop - 1}"
                )
            plane["C"] = int(channel)
        elif axis == "S":
            if scene is None and stop - start != 1:
                raise ValueError(
                    f"the CZI has {stop - start} scenes (S={start}..{stop - 1}); "
                    "convert them all with convert_image_to_ome_tifs (the "
                    "command-line tool does so when --scene is omitted), or pass "
                    "scene=N for one scene or 'all' for the whole slide"
                )
            if scene is None:
                scene = start
            if scene != "all":
                if not start <= int(scene) < stop:
                    raise ValueError(
                        f"scene {scene} is out of range; input has S={start}..{stop - 1}"
                    )
                plane["S"] = int(scene)
        elif stop - start != 1:
            raise ValueError(
                f"unsupported multi-plane CZI: axis {axis} has {stop - start} planes "
                f"({axis}={start}..{stop - 1}); only single-plane images convert"
            )
        else:
            plane[axis] = start
    if scene not in (None, "all") and "S" not in dims_shape:
        raise ValueError("--scene given, but the CZI has no scene (S) axis")
    if channel is not None and "C" not in dims_shape:
        raise ValueError("--channel given, but the CZI has no channel (C) axis")
    return plane


def _czi_dims_shape(shapes, scene):
    """Reduce aicspylibczi's per-scene ``{axis: (start, stop)}`` list to one map.

    Scenes of different size each get their own entry. Converting one scene
    uses that scene's entry; converting the whole mosaic spans them all, so the
    ranges are merged and the plane checks apply across every scene.
    """
    if scene not in (None, "all") and len(shapes) > 1:
        matching = [
            item
            for item in shapes
            if "S" in item and item["S"][0] <= int(scene) < item["S"][1]
        ]
        if not matching:
            raise ValueError(f"scene {scene} is out of range for this CZI")
        return matching[0]
    merged = {}
    for item in shapes:
        for axis, (start, stop) in item.items():
            low, high = merged.get(axis, (start, stop))
            merged[axis] = (min(low, start), max(high, stop))
    return merged


def _czi_to_yxc(arr, pixel_type):
    """Normalize a libCZI read to grayscale YX or interleaved RGB YXC."""
    samples = _CZI_SAMPLES_PER_PIXEL.get(str(pixel_type).lower())
    if samples is None:
        supported = ", ".join(sorted(_CZI_SAMPLES_PER_PIXEL))
        raise ValueError(
            f"unsupported CZI pixel type {pixel_type!r}; expected one of: {supported}"
        )
    arr = np.asarray(arr)
    expected = 3 if samples > 1 else 2
    # aicspylibczi returns (1, Y, X[, samples]); drop the leading singletons.
    while arr.ndim > expected and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != expected:
        raise ValueError(
            f"unexpected CZI array shape {arr.shape} for pixel type {pixel_type}"
        )
    if samples == 1:
        return arr, "minisblack"
    # libCZI hands the samples back in the stored order, blue first; the
    # alpha of Bgra32 carries nothing for a scan and is dropped.
    return np.ascontiguousarray(arr[..., 2::-1]), "rgb"


class _CziSource(_StripSource):
    """One plane of a CZI, read through libCZI in strips.

    A slide scan is a mosaic of thousands of tiles; libCZI assembles any region
    of it on request, which is what `_read_strip` asks for. A single-image
    (non-mosaic) file is read whole instead.
    """

    def __init__(self, czi, name, *, scene, channel):
        self.czi = czi
        self.name = name
        self.mosaic = czi.is_mosaic()
        dims_shape = _czi_dims_shape(czi.get_dims_shape(), scene)
        self.plane = _czi_plane(dims_shape, scene=scene, channel=channel)
        self.pixel_type = str(czi.pixel_type).lower()
        if self.pixel_type not in _CZI_SAMPLES_PER_PIXEL:
            supported = ", ".join(sorted(_CZI_SAMPLES_PER_PIXEL))
            raise ValueError(
                f"unsupported CZI pixel type {self.pixel_type!r}; "
                f"expected one of: {supported}"
            )
        self.dtype = np.dtype(_CZI_DTYPES[self.pixel_type])
        samples = _CZI_SAMPLES_PER_PIXEL[self.pixel_type]
        self.photometric = "rgb" if samples > 1 else "minisblack"
        # Gaps between tiles or scenes are filled with white for brightfield
        # and black for fluorescence, matching what each kind of scan shows.
        self.background = (1.0,) * 3 if samples > 1 else (0.0,) * 3
        self.scene = scene
        if self.mosaic:
            if scene in (None, "all") or "S" not in dims_shape:
                box = czi.get_mosaic_bounding_box()
            else:
                # libCZI refuses an S constraint on a mosaic; the scene's own
                # bounding box selects it instead.
                box = czi.get_mosaic_scene_bounding_box(int(scene))
                self.plane.pop("S", None)
            self.region = (box.x, box.y, box.w, box.h)
            height, width = box.h, box.w
        else:
            self.region = None
            height = dims_shape["Y"][1] - dims_shape["Y"][0]
            width = dims_shape["X"][1] - dims_shape["X"][0]
        self.shape = (height, width, 3) if samples > 1 else (height, width)
        self.pixel_size = _czi_pixel_size_um(czi.meta)
        self.streams = self.mosaic

    def describe(self):
        if self.scene not in (None, "all"):
            where = f"scene {self.scene}"
        else:
            where = "whole mosaic" if self.mosaic else "single image"
        return f"{where}, plane {self.plane} ({self.pixel_type})"

    def read_all(self):
        """The whole plane as YX / YXC; only for a single-image (non-mosaic) file."""
        arr, _ = self.czi.read_image(**self.plane)
        array, _ = _czi_to_yxc(arr, self.pixel_type)
        return array

    def _read_strip(self, row, rows):
        x, y, width, _ = self.region
        arr = self.czi.read_mosaic(
            region=(x, y + row, width, rows),
            scale_factor=1.0,
            background_color=self.background,
            **self.plane,
        )
        array, _ = _czi_to_yxc(arr, self.pixel_type)
        return array


@contextlib.contextmanager
def _open_czi_file(src):
    """Open a CZI through libCZI; the file handle lives for the block."""
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ValueError(
            "reading .czi requires the aicspylibczi package "
            "(pip install aicspylibczi)"
        ) from exc

    # The reader keeps the handle open for lazy reads; own it so it is closed.
    with open(src, "rb") as handle:
        yield CziFile(handle)


@contextlib.contextmanager
def _open_czi(src, *, scene, channel):
    """Open a CZI as a `_CziSource`."""
    with _open_czi_file(src) as czi:
        yield _CziSource(czi, Path(src).name, scene=scene, channel=channel)


def czi_scene_indices(src):
    """The scene (S) indices a CZI holds; empty when it has no scene axis."""
    indices = set()
    with _open_czi_file(src) as czi:
        for item in czi.get_dims_shape():
            if "S" in item:
                indices.update(range(*item["S"]))
    return sorted(indices)


def _downsample_strip(strip, width, rows):
    """Area-downsample one source strip."""
    if _HAVE_CV2:
        return cv2.resize(strip, (width, rows), interpolation=cv2.INTER_AREA)

    factor_y = strip.shape[0] // rows
    factor_x = strip.shape[1] // width
    crop = strip[: rows * factor_y, : width * factor_x]
    if crop.ndim == 3:
        crop = crop.reshape(
            rows,
            factor_y,
            width,
            factor_x,
            crop.shape[2],
        )
        return crop.mean(axis=(1, 3)).astype(strip.dtype)
    crop = crop.reshape(rows, factor_y, width, factor_x)
    return crop.mean(axis=(1, 3)).astype(strip.dtype)


def _downsample_half(arr):
    """Area-downsample an H,W[,C] array by two, strip by strip."""
    height, width = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    next_height = max(1, height // 2)
    next_width = max(1, width // 2)
    shape = (
        (next_height, next_width, channels)
        if arr.ndim == 3
        else (next_height, next_width)
    )
    result = np.empty(shape, dtype=arr.dtype)

    row_budget = max(2, int(_MAX_STRIP_ELEMS // (width * channels)))
    source_rows = max(2, (row_budget // 2) * 2)
    output_rows = max(1, source_rows // 2)
    for output_start in range(0, next_height, output_rows):
        output_end = min(next_height, output_start + output_rows)
        source_start = output_start * 2
        source_end = output_end * 2
        strip = np.asarray(arr[source_start:source_end, : next_width * 2])
        result[output_start:output_end] = _downsample_strip(
            strip,
            next_width,
            output_end - output_start,
        )
    return result


def _pyramid_shapes(base_shape):
    """The (height, width) of every level `_build_pyramid` would produce."""
    shapes = [tuple(int(value) for value in base_shape[:2])]
    while shapes[-1][0] > 1 and shapes[-1][1] > 1:
        height, width = shapes[-1]
        shapes.append((max(1, height // 2), max(1, width // 2)))
    return shapes


def _build_pyramid(base, *, first_level=0):
    """Halve both dimensions until either resulting dimension reaches one."""
    pyramid = [base]
    level = first_level
    while pyramid[-1].shape[0] > 1 and pyramid[-1].shape[1] > 1:
        next_level = _downsample_half(pyramid[-1])
        pyramid.append(next_level)
        level += 1
        height, width = next_level.shape[:2]
        print(f"[convert]   level {level}: {width}x{height}", flush=True)
        if height == 1 or width == 1:
            print(
                f"[convert]   stopping: level {level} reached 1 in one dimension",
                flush=True,
            )
            break
    return pyramid


def _normalize_mpp(mpp):
    """Accept a scalar or (x, y) micrometers-per-pixel override."""
    if mpp is None:
        return None
    values = np.atleast_1d(np.asarray(mpp, dtype=float)).ravel()
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2 or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("mpp must be one or two positive micrometer values")
    return float(values[0]), float(values[1])


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


def _report_pixel_size(pixel_size, override):
    if pixel_size is None:
        print(
            "[convert] pixel size unknown; output carries no PhysicalSize "
            "(pass --mpp to set it)",
            flush=True,
        )
    else:
        source = "override" if override is not None else "source metadata"
        print(
            f"[convert] pixel size ({source}): "
            f"x={pixel_size[0]:.6g} um/px, y={pixel_size[1]:.6g} um/px",
            flush=True,
        )


def _cyx_tiles(
    read_strip,
    shape,
    tile_size,
    *,
    next_level=None,
    level_label,
):
    """Yield channel-major tiles from a CYX reader and optionally halve it."""
    channels, height, width = (int(value) for value in shape)
    # Two tile rows bound a level-1 source read to about 80 MiB for the user's
    # 20k-pixel-wide uint16 image while still keeping I/O calls reasonably large.
    strip_rows = tile_size * 2
    for channel in range(channels):
        for row in range(0, height, strip_rows):
            rows = min(strip_rows, height - row)
            strip = np.asarray(read_strip(channel, row, rows))
            if strip.shape != (rows, width):
                raise ValueError(
                    f"{level_label} channel {channel}: got strip shape "
                    f"{strip.shape}, expected {(rows, width)}"
                )
            print(
                f"[convert]   {level_label} channel {channel + 1}/{channels}, "
                f"rows {row}-{row + rows} of {height}",
                flush=True,
            )
            if next_level is not None:
                half = _downsample_half(strip)
                next_level[
                    channel, row // 2 : row // 2 + half.shape[0]
                ] = half
            for tile_row in range(0, rows, tile_size):
                for tile_col in range(0, width, tile_size):
                    tile = strip[
                        tile_row : tile_row + tile_size,
                        tile_col : tile_col + tile_size,
                    ]
                    if tile.shape != (tile_size, tile_size):
                        tile = np.pad(
                            tile,
                            (
                                (0, tile_size - tile.shape[0]),
                                (0, tile_size - tile.shape[1]),
                            ),
                        )
                    yield np.ascontiguousarray(tile)


def _temporary_memmap(dst, shape, dtype):
    """Create one output-adjacent scratch array for a CYX pyramid level."""
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{dst.name}.",
        suffix=".pyramid.tmp",
        dir=dst.parent,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape), path


def _release_memmap(array, path):
    """Flush, close, and remove a scratch pyramid level."""
    if array is not None:
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    if path is not None:
        path.unlink(missing_ok=True)


def _write_cyx_pyramid(
    dst,
    source,
    *,
    pixel_size,
    tile_size,
    compression,
):
    """Stream all mIF channels to one CYX pyramidal OME-TIFF.

    The base level is copied channel-by-channel. Level 1 is generated by a
    second source pass; while it is written, level 2 is saved in an
    output-adjacent memmap. Each later level produces the next one, so scratch
    storage peaks at levels 2 + 3 instead of holding the 9.6 GiB base or the
    complete 3.2 GiB pyramid in RAM.
    """
    channels, height, width = (int(value) for value in source.shape)
    spatial_shapes = _pyramid_shapes((height, width))
    subifd_count = len(spatial_shapes) - 1
    output_compression = None if compression == "none" else compression
    write_options = {
        "photometric": "minisblack",
        "tile": (tile_size, tile_size),
        "compression": output_compression,
    }
    metadata = {
        "axes": "CYX",
        "Channel": {"Name": list(source.channel_names)},
    }
    if pixel_size is not None:
        metadata.update(
            PhysicalSizeX=pixel_size[0],
            PhysicalSizeXUnit="\u00b5m",
            PhysicalSizeY=pixel_size[1],
            PhysicalSizeYUnit="\u00b5m",
        )

    print(
        f"[convert] writing {channels}-channel pyramidal OME-TIFF -> {dst} "
        f"(base + {subifd_count} levels, tile={tile_size}, "
        f"compression={compression})",
        flush=True,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    current_memmap = None
    current_path = None
    next_memmap = None
    next_path = None
    base_spatial_shape = spatial_shapes[0]
    try:
        with tifffile.TiffWriter(dst, bigtiff=True, ome=True) as writer:
            writer.write(
                _cyx_tiles(
                    source.read_channel_strip,
                    source.shape,
                    tile_size,
                    level_label="base",
                ),
                shape=source.shape,
                dtype=source.dtype,
                subifds=subifd_count,
                metadata=metadata,
                **_level_write_options(
                    write_options,
                    pixel_size,
                    base_spatial_shape,
                    base_spatial_shape,
                ),
            )

            for level_index, (level_height, level_width) in enumerate(
                spatial_shapes[1:], start=1
            ):
                level_shape = (channels, level_height, level_width)
                if level_index == 1:
                    # Generate level 1 directly from the original channel. The
                    # source is reread once, avoiding a 2.6 GiB level-1 scratch file.
                    def read_level_strip(channel, row, rows):
                        return _downsample_half(
                            source.read_channel_strip(
                                channel, row * 2, rows * 2
                            )
                        )

                else:
                    array = current_memmap

                    def read_level_strip(channel, row, rows, array=array):
                        return np.asarray(array[channel, row : row + rows])

                if level_index + 1 < len(spatial_shapes):
                    next_height, next_width = spatial_shapes[level_index + 1]
                    next_memmap, next_path = _temporary_memmap(
                        dst,
                        (channels, next_height, next_width),
                        source.dtype,
                    )
                else:
                    next_memmap = None
                    next_path = None

                print(
                    f"[convert]   level {level_index}: "
                    f"{level_width}x{level_height}, {channels} channels",
                    flush=True,
                )
                writer.write(
                    _cyx_tiles(
                        read_level_strip,
                        level_shape,
                        tile_size,
                        next_level=next_memmap,
                        level_label=f"level {level_index}",
                    ),
                    shape=level_shape,
                    dtype=source.dtype,
                    subfiletype=1,
                    **_level_write_options(
                        write_options,
                        pixel_size,
                        base_spatial_shape,
                        (level_height, level_width),
                    ),
                )
                if next_memmap is not None:
                    next_memmap.flush()
                _release_memmap(current_memmap, current_path)
                current_memmap, current_path = next_memmap, next_path
                next_memmap = None
                next_path = None
    finally:
        _release_memmap(current_memmap, current_path)
        _release_memmap(next_memmap, next_path)


def _write_pyramid(
    dst,
    base,
    *,
    shape,
    dtype,
    photometric,
    pixel_size,
    tile_size,
    compression,
    level1=None,
    channel_name=None,
):
    """Write `base` and its 2x pyramid to `dst` as a tiled OME-TIFF.

    `base` is either the full-resolution array, from which every level is
    built in memory, or an iterator of row-major tiles (a `_CziSource`), in
    which case `level1` is the array that iterator fills and the remaining
    levels are built from it once the base level has been written.
    """
    shapes = _pyramid_shapes(shape)
    subifd_count = len(shapes) - 1
    output_compression = None if compression == "none" else compression
    write_options = {
        "photometric": photometric,
        "tile": (tile_size, tile_size),
        "compression": output_compression,
    }
    metadata = {"axes": "YXC" if len(shape) == 3 else "YX"}
    if channel_name:
        metadata["Channel"] = {"Name": [channel_name]}
    if pixel_size is not None:
        metadata.update(
            PhysicalSizeX=pixel_size[0],
            PhysicalSizeXUnit="\u00b5m",
            PhysicalSizeY=pixel_size[1],
            PhysicalSizeYUnit="\u00b5m",
        )

    print(
        f"[convert] writing pyramidal OME-TIFF -> {dst} "
        f"(base + {subifd_count} levels, tile={tile_size}, "
        f"compression={compression})",
        flush=True,
    )
    base_shape = shapes[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(dst, bigtiff=True, ome=True) as writer:
        base_options = _level_write_options(
            write_options, pixel_size, base_shape, base_shape
        )
        if isinstance(base, np.ndarray):
            writer.write(
                base, subifds=subifd_count, metadata=metadata, **base_options
            )
            levels = _build_pyramid(base)[1:]
        else:
            writer.write(
                base,
                shape=tuple(shape),
                dtype=dtype,
                subifds=subifd_count,
                metadata=metadata,
                **base_options,
            )
            levels = [] if level1 is None else _build_pyramid(level1, first_level=1)
        for level in levels:
            writer.write(
                level,
                subfiletype=1,
                **_level_write_options(
                    write_options, pixel_size, base_shape, level.shape[:2]
                ),
            )
    if [tuple(level.shape[:2]) for level in levels] != shapes[1:]:
        raise ValueError(
            f"pyramid levels {[level.shape[:2] for level in levels]} "
            f"do not match the planned {shapes[1:]}"
        )


def convert_image_to_ome_tif(
    src,
    *,
    tile_size=512,
    compression="zlib",
    series=None,
    mpp=None,
    scene=None,
    channel=None,
):
    """Convert a supported image to an automatic 2x OME-TIFF pyramid."""
    src = Path(src).expanduser().resolve()
    kind = image_kind(src)
    scene = parse_scene(scene)
    dst = output_path(src, scene=scene)
    override = _normalize_mpp(mpp)

    if not src.is_file():
        raise FileNotFoundError(f"input not found: {src}")
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("tile size must be a positive multiple of 16")
    if compression not in {"zlib", "lzw", "jpeg", "none"}:
        raise ValueError(f"unsupported compression: {compression}")
    if kind != "tiff" and series not in (None, 0):
        raise ValueError("--series applies only to BTF/TIFF/NDPI/SVS input")
    if kind != "czi" and scene is not None:
        raise ValueError("--scene applies only to CZI input")
    if kind not in ("tiff", "czi") and channel is not None:
        raise ValueError("--channel applies only to TIFF/OME-TIFF or CZI input")
    if kind == "czi" and channel is not None:
        try:
            channel = int(channel)
        except (TypeError, ValueError):
            raise ValueError("CZI channel must be a zero-based integer")
        if channel < 0:
            raise ValueError("channel must be >= 0")

    if kind == "jpeg":
        print(f"[convert] reading JPEG from {src.name} ...", flush=True)
        base = _read_jpeg(src)
        photometric = "rgb" if base.ndim == 3 else "minisblack"
        # JPEG DPI describes print size, not microscope calibration, so the
        # spacing only comes from an explicit override here.
        pixel_size = override
        print(
            f"[convert] base shape={base.shape} dtype={base.dtype} "
            f"photometric={photometric}",
            flush=True,
        )
        _report_pixel_size(pixel_size, override)
        _write_pyramid(
            dst,
            base,
            shape=base.shape,
            dtype=base.dtype,
            photometric=photometric,
            pixel_size=pixel_size,
            tile_size=tile_size,
            compression=compression,
        )
    else:
        opener = (
            _open_czi(src, scene=scene, channel=channel)
            if kind == "czi"
            else _open_tiff(src, series, channel=channel)
        )
        with opener as source:
            pixel_size = override if override is not None else source.pixel_size
            print(
                f"[convert] reading {source.describe()} from {source.name} "
                f"({'in strips' if source.streams else 'whole'}) ...",
                flush=True,
            )
            print(
                f"[convert] base shape={source.shape} dtype={source.dtype} "
                f"photometric={source.photometric}",
                flush=True,
            )
            _report_pixel_size(pixel_size, override)
            options = dict(
                shape=source.shape,
                dtype=source.dtype,
                photometric=source.photometric,
                pixel_size=pixel_size,
                tile_size=tile_size,
                compression=compression,
                channel_name=getattr(source, "channel_name", None),
            )
            if getattr(source, "all_channels", False):
                _write_cyx_pyramid(
                    dst,
                    source,
                    pixel_size=pixel_size,
                    tile_size=tile_size,
                    compression=compression,
                )
            elif source.streams:
                shapes = _pyramid_shapes(source.shape)
                level1 = None
                if len(shapes) > 1:
                    level1 = np.empty(
                        shapes[1] + tuple(source.shape[2:]), source.dtype
                    )
                _write_pyramid(
                    dst, source.tiles(tile_size, level1), level1=level1, **options
                )
            else:
                _write_pyramid(dst, source.read_all(), **options)

    print(f"[convert] DONE. {dst} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dst


def convert_image_to_ome_tifs(src, **options):
    """Every OME-TIFF `src` produces: one per scene of a multi-scene CZI, else one.

    A slide scanned as several scenes holds several tissue sections, and each
    is its own image to the pipeline, so with no scene chosen each becomes
    ``NAME_sceneN.ome.tif``. Any other input, a single-scene CZI, or an explicit
    ``scene`` gives the single output of `convert_image_to_ome_tif`.
    """
    if image_kind(src) == "czi" and options.get("scene") is None:
        scenes = czi_scene_indices(src)
        if len(scenes) > 1:
            print(
                f"[convert] {Path(src).name} has {len(scenes)} scenes; "
                "converting each to its own OME-TIFF",
                flush=True,
            )
            return [
                convert_image_to_ome_tif(src, **{**options, "scene": scene})
                for scene in scenes
            ]
    return [convert_image_to_ome_tif(src, **options)]
