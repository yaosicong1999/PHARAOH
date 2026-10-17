#!/usr/bin/env python3
"""Print scalar raw MPP from an H&E OME-TIFF header."""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import tifffile


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


def get_he_raw_mpp(he_img):
    """Return sqrt(PhysicalSizeX * PhysicalSizeY) in micrometers per pixel."""
    path = Path(he_img).expanduser().resolve()
    name = path.name.lower()
    if not name.endswith((".ome.tif", ".ome.tiff")):
        raise ValueError("he_img must end with .ome.tif or .ome.tiff")
    if not path.is_file():
        raise FileNotFoundError(f"H&E image not found: {path}")

    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome or not tif.ome_metadata:
            raise ValueError("he_img is not a valid OME-TIFF with OME-XML metadata")
        metadata = tif.ome_metadata

    try:
        root = ET.fromstring(metadata)
        pixels = root.find("./{*}Image/{*}Pixels")
        if pixels is None:
            raise ValueError("OME header contains no Image/Pixels element")

        values_um = []
        for axis in ("X", "Y"):
            attribute = f"PhysicalSize{axis}"
            if attribute not in pixels.attrib:
                raise ValueError(f"OME header is missing {attribute}")
            value = float(pixels.attrib[attribute])
            unit = pixels.attrib.get(f"{attribute}Unit", "µm")
            factor = _UM_PER_OME_UNIT.get(unit)
            if factor is None:
                raise ValueError(
                    f"OME header has unsupported {attribute}Unit={unit!r}"
                )
            value_um = value * factor
            if not math.isfinite(value_um) or value_um <= 0:
                raise ValueError(f"OME header has invalid {attribute}={value!r}")
            values_um.append(value_um)
    except ET.ParseError as exc:
        raise ValueError(f"OME-XML metadata cannot be parsed: {exc}") from exc

    return math.sqrt(values_um[0] * values_um[1])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Print sqrt(PhysicalSizeX * PhysicalSizeY) from an H&E OME-TIFF "
            "in micrometers per pixel."
        )
    )
    parser.add_argument("--he_img", "--he-img", required=True)
    args = parser.parse_args(argv)
    try:
        mpp = get_he_raw_mpp(args.he_img)
    except (FileNotFoundError, OSError, ValueError, tifffile.TiffFileError) as exc:
        parser.error(str(exc))

    # Machine-friendly stdout: one value, suitable for --raw_mpp or $(...).
    print(f"{mpp:.12g}")


if __name__ == "__main__":
    main()
