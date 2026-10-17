#!/usr/bin/env python3
"""Convert an OME-TIFF to an image at a requested micrometers-per-pixel.

Example:
    python scripts/post_process/convert_to_rescaled_image.py \
        --input_img=slide.ome.tif \
        --output_img=slide_0.5mpp.jpg \
        --output_mpp=0.5

``--input_mpp`` is optional.  When it is omitted, level-0
``PhysicalSizeX`` and ``PhysicalSizeY`` are read from the OME-XML and their
geometric mean (sqrt(x * y)) is used as the input MPP.
"""

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


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
_TIFF_SUFFIXES = (".tif", ".tiff", ".btf")
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _positive_mpp(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def read_ome_pixel_size_um(tif, series=0):
    """Return level-0 (x, y) spacing in micrometers from OME-XML."""
    metadata = tif.ome_metadata
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


def scalar_mpp(pixel_size_xy):
    """Collapse (x, y) pixel spacing using the requested geometric mean."""
    if pixel_size_xy is None:
        return None
    x, y = pixel_size_xy
    return math.sqrt(float(x) * float(y))


def _spatial_shape(level):
    axes = (getattr(level, "axes", "") or "").upper()
    shape = tuple(map(int, level.shape))
    if "Y" in axes and "X" in axes:
        return shape[axes.index("Y")], shape[axes.index("X")]
    if len(shape) >= 2:
        return shape[0], shape[1]
    raise ValueError(f"cannot find Y/X dimensions in shape={shape}, axes={axes or '?'}")


def choose_source_level(levels, input_mpp, output_mpp):
    """Choose the smallest pyramid level that is not coarser than the output."""
    base_height, base_width = _spatial_shape(levels[0])
    candidates = []
    for index, level in enumerate(levels):
        height, width = _spatial_shape(level)
        level_mpp = input_mpp * math.sqrt(
            (base_width / width) * (base_height / height)
        )
        if level_mpp <= output_mpp * (1.0 + 1e-9):
            candidates.append((height * width, index, level_mpp))
    if not candidates:
        return 0, input_mpp
    _, index, level_mpp = min(candidates)
    return index, level_mpp


def _to_yxc(array, axes):
    """Normalize one OME plane to YX or interleaved YXC."""
    array = np.asarray(array)
    axes = (axes or "").upper()

    if axes and len(axes) == array.ndim and "Y" in axes and "X" in axes:
        # A whole-slide output represents one 2-D plane. Non-spatial axes other
        # than channel/sample must therefore be singleton.
        for axis in tuple(axes):
            if axis not in "YXCS":
                position = axes.index(axis)
                if array.shape[position] != 1:
                    raise ValueError(
                        f"unsupported multi-plane OME image: shape={array.shape}, "
                        f"axes={axes}; expected singleton T/Z dimensions"
                    )
                array = np.take(array, 0, axis=position)
                axes = axes[:position] + axes[position + 1 :]

        channel_axes = [axis for axis in "CS" if axis in axes]
        if len(channel_axes) > 1:
            raise ValueError(f"unsupported axes containing both C and S: {axes}")
        order = [axes.index("Y"), axes.index("X")]
        if channel_axes:
            order.append(axes.index(channel_axes[0]))
        array = np.transpose(array, order)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)

    array = np.squeeze(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"unsupported image shape after axis normalization: {array.shape}")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"output supports grayscale, RGB, or RGBA data; got {array.shape[-1]} channels"
        )
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _resize(array, output_width, output_height):
    if array.shape[:2] == (output_height, output_width):
        return array
    resampling = (
        Image.Resampling.LANCZOS
        if output_height <= array.shape[0] and output_width <= array.shape[1]
        else Image.Resampling.BICUBIC
    )
    output_size = (output_width, output_height)

    # Pillow directly and efficiently handles the common uint8 grayscale/RGB(A)
    # cases. Other numeric types are resized one channel at a time as float32,
    # then converted back so TIFF output retains its source bit depth.
    if array.dtype == np.uint8:
        return np.asarray(Image.fromarray(array).resize(output_size, resampling)).copy()

    original_dtype = array.dtype
    planes = [array] if array.ndim == 2 else np.moveaxis(array, -1, 0)
    resized_planes = []
    for plane in planes:
        float_image = Image.fromarray(np.asarray(plane, dtype=np.float32), mode="F")
        resized_planes.append(
            np.asarray(float_image.resize(output_size, resampling), dtype=np.float32)
        )
    resized = (
        resized_planes[0]
        if array.ndim == 2
        else np.stack(resized_planes, axis=-1)
    )
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        resized = np.clip(np.rint(resized), limits.min, limits.max)
    return resized.astype(original_dtype, copy=False)


def _to_uint8(array):
    """Convert data to an 8-bit representation for JPEG/PNG output."""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        scaled = (array.astype(np.float64) - limits.min) * (
            255.0 / (limits.max - limits.min)
        )
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)
        low, high = float(finite.min()), float(finite.max())
        if 0.0 <= low and high <= 1.0:
            scaled = array * 255.0
        else:
            scaled = array
        return np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)
    raise ValueError(f"cannot write dtype {array.dtype} as an 8-bit image")


def _output_kind(path):
    name = path.name.lower()
    if name.endswith(_TIFF_SUFFIXES):
        return "ome-tiff" if name.endswith((".ome.tif", ".ome.tiff")) else "tiff"
    if path.suffix.lower() in _RASTER_SUFFIXES:
        return path.suffix.lower().lstrip(".")
    raise ValueError(
        "unsupported output extension; use .jpg, .jpeg, .png, .tif, .tiff, "
        ".btf, .ome.tif, or .ome.tiff"
    )


def _write_output(path, array, output_mpp, kind):
    if kind in {"tiff", "ome-tiff"}:
        photometric = "rgb" if array.ndim == 3 else "minisblack"
        metadata = None
        if kind == "ome-tiff":
            metadata = {
                "axes": "YXS" if array.ndim == 3 else "YX",
                "PhysicalSizeX": output_mpp,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": output_mpp,
                "PhysicalSizeYUnit": "µm",
            }
        tifffile.imwrite(
            path,
            array,
            bigtiff=array.nbytes > 2**32 - 2**25,
            ome=(kind == "ome-tiff"),
            photometric=photometric,
            # Keep this utility independent of the optional compiled
            # imagecodecs package. JPEG/PNG are compressed by Pillow; TIFF is
            # intentionally uncompressed so it works in the base environment.
            compression=None,
            resolution=(1e4 / output_mpp, 1e4 / output_mpp),
            resolutionunit="CENTIMETER",
            metadata=metadata,
        )
        return

    raster = _to_uint8(array)
    if kind in {"jpg", "jpeg"} and raster.ndim == 3 and raster.shape[-1] == 4:
        raster = raster[..., :3]
    image = Image.fromarray(raster)
    dpi = 25400.0 / output_mpp
    options = {"dpi": (dpi, dpi)}
    if kind in {"jpg", "jpeg"}:
        options.update(quality=95, subsampling=0)
    image.save(path, **options)


def convert_to_rescaled_image(input_img, output_img, output_mpp, input_mpp=None):
    """Convert one OME-TIFF and return the resolved output path."""
    source = Path(input_img).expanduser().resolve()
    destination = Path(output_img).expanduser().resolve()
    output_mpp = _positive_mpp(output_mpp, "output_mpp")
    override_mpp = (
        None if input_mpp is None else _positive_mpp(input_mpp, "input_mpp")
    )
    kind = _output_kind(destination)

    if not source.is_file():
        raise FileNotFoundError(f"input image not found: {source}")
    if source == destination:
        raise ValueError("input_img and output_img must be different files")

    with tifffile.TiffFile(source) as tif:
        if not tif.series:
            raise ValueError(f"input contains no image series: {source}")
        series = tif.series[0]
        levels = list(getattr(series, "levels", None) or [series])
        pixel_size = read_ome_pixel_size_um(tif, series=0)
        detected_mpp = scalar_mpp(pixel_size)
        resolved_input_mpp = override_mpp or detected_mpp
        if resolved_input_mpp is None:
            raise ValueError(
                "input MPP is missing or invalid in the OME PhysicalSizeX/Y "
                "metadata; pass --input_mpp explicitly"
            )

        base_height, base_width = _spatial_shape(levels[0])
        output_width = max(
            1, int(math.floor(base_width * resolved_input_mpp / output_mpp + 0.5))
        )
        output_height = max(
            1, int(math.floor(base_height * resolved_input_mpp / output_mpp + 0.5))
        )
        level_index, level_mpp = choose_source_level(
            levels, resolved_input_mpp, output_mpp
        )
        level = levels[level_index]

        source_label = "--input_mpp" if override_mpp is not None else "OME metadata"
        if pixel_size is not None and override_mpp is None:
            print(
                f"[rescale] input pixel size: x={pixel_size[0]:.6g}, "
                f"y={pixel_size[1]:.6g} um/px; geometric mean="
                f"{resolved_input_mpp:.6g} um/px",
                flush=True,
            )
        else:
            print(
                f"[rescale] input MPP ({source_label}): "
                f"{resolved_input_mpp:.6g} um/px",
                flush=True,
            )
        print(
            f"[rescale] reading pyramid level {level_index} "
            f"(~{level_mpp:.6g} um/px), output={output_width}x{output_height} "
            f"at {output_mpp:.6g} um/px",
            flush=True,
        )
        array = _to_yxc(level.asarray(), getattr(level, "axes", ""))

    resized = _resize(array, output_width, output_height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix="".join(destination.suffixes),
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        _write_output(temporary, resized, output_mpp, kind)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"[rescale] DONE: {destination} "
        f"({destination.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert an OME-TIFF to JPG/TIFF/PNG at a requested MPP."
    )
    parser.add_argument("--input_img", "--input-img", required=True)
    parser.add_argument("--output_img", "--output-img", required=True)
    parser.add_argument("--output_mpp", "--output-mpp", required=True, type=float)
    parser.add_argument(
        "--input_mpp",
        "--input-mpp",
        type=float,
        default=None,
        help=(
            "level-0 micrometers per pixel; when omitted, use "
            "sqrt(PhysicalSizeX * PhysicalSizeY) from OME-XML"
        ),
    )
    args = parser.parse_args(argv)
    try:
        convert_to_rescaled_image(
            args.input_img,
            args.output_img,
            args.output_mpp,
            args.input_mpp,
        )
    except (FileNotFoundError, OSError, ValueError, tifffile.TiffFileError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
