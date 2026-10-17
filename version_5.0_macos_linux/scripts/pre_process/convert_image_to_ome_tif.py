#!/usr/bin/env python3
"""Convert a BTF, TIFF, NDPI, SVS, JPEG, or CZI image to a pyramidal OME-TIFF.

Usage:
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.btf
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.tif
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.ndpi
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.svs
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.jpg
    python scripts/pre_process/convert_image_to_ome_tif.py IMAGE.czi

The output is written beside the input with its final extension replaced by
``.ome.tif``. For example, ``/data/image.jpg`` becomes
``/data/image.ome.tif``.

The physical pixel size is copied from the source when it is recorded there
(NDPI and TIFF resolution tags, the Aperio ``MPP`` of an SVS, OME-XML, or the
CZI ``Scaling`` block), so the
converted file keeps the ``PhysicalSizeX/Y`` that the pipeline reads for
pyramid level selection. Use ``--mpp`` when the source has no calibration,
such as a JPEG.

Every slide format is streamed in strips of tile rows, so a whole-slide scan
never has to fit in memory at full resolution. A CZI is read through libCZI
(the ``aicspylibczi`` package), which assembles the mosaic tiles. A slide with several scenes is split:
each scene becomes ``NAME_sceneN.ome.tif``, unless ``--scene N`` picks one or
``--scene all`` keeps the whole slide as one image. A file with several
channels needs ``--channel``. For an OME-TIFF mIF image, the channel may be
selected by zero-based index or OME channel name, or ``--channel all`` keeps
every channel; CZI channels use an index.
"""

import argparse

from image_conversion import convert_image_to_ome_tifs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BTF, TIFF, NDPI, SVS, JPEG, or CZI image to a pyramidal OME-TIFF."
        )
    )
    parser.add_argument("src", help="input .btf, .tif, .ndpi, .svs, .jpg, or .czi image")
    parser.add_argument(
        "--tile", type=int, default=512, help="tile size px (default 512)"
    )
    parser.add_argument(
        "--compression",
        default="zlib",
        choices=["zlib", "lzw", "jpeg", "none"],
        help="tile compression (default zlib, lossless)",
    )
    parser.add_argument(
        "--series",
        type=int,
        default=None,
        help=(
            "source series for BTF/TIFF/NDPI/SVS input "
            "(default: series 0, or the NDPI/SVS baseline pyramid)"
        ),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "CZI scene (S index) to convert, or 'all' for the whole slide as "
            "one image (default: every scene, each to NAME_sceneN.ome.tif)"
        ),
    )
    parser.add_argument(
        "--channel",
        default=None,
        help=(
            "channel to convert; for OME-TIFF use a zero-based C index, "
            "channel name, or 'all' (for example 0, DAPI, or all), and for "
            "CZI use a C index; required when the input has more than one channel"
        ),
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=(
            "micrometers per pixel at level 0; overrides the source metadata "
            "and is the only way to set it for JPEG input"
        ),
    )
    args = parser.parse_args(argv)

    try:
        convert_image_to_ome_tifs(
            args.src,
            tile_size=args.tile,
            compression=args.compression,
            series=args.series,
            mpp=args.mpp,
            scene=args.scene,
            channel=args.channel,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
