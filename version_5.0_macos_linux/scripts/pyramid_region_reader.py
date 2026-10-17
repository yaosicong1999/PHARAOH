'''Bounded-memory pyramid region reads shared by patch QC and dense scanning.'''
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _axes_for_shape(shape: Tuple[int, ...]) -> str:
    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        return "YXC" if shape[-1] in (1, 2, 3, 4) else "CYX"
    # Unknown leading dimensions are treated as singleton time/Z dimensions.
    return "T" * (len(shape) - 2) + "YX"


class PyramidRegionReader:
    """Read rectangular regions without materializing a whole TIFF pyramid."""

    def __init__(self, path: Path, level: int, modality: str, channel_index=None):
        self.path = Path(path)
        self.level = int(level)
        self.modality = modality
        if modality == "dapi" and channel_index is None:
            from my_utils import resolve_dapi_channel
            channel_index = resolve_dapi_channel(self.path)["index"]
        self.channel_index = None if channel_index is None else int(channel_index)
        if self.channel_index is not None and self.channel_index < 0:
            raise ValueError("channel_index must be non-negative")
        self._tif = None
        self._store = None
        self._array = None
        self._pil = None
        self._pseudo_scale = 1

        try:
            if self.path.suffix.lower() in {".tif", ".tiff"}:
                self._open_tiff()
            else:
                self._open_pil()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _yx_shape(shape: Tuple[int, ...], axes: str) -> Tuple[int, int]:
        axes = axes.replace("S", "C")
        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"cannot identify Y/X dimensions: shape={shape}, axes={axes}")
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

    def _open_tiff(self):
        import tifffile
        import zarr

        self._tif = tifffile.TiffFile(str(self.path))
        series = self._tif.series[0]
        levels = list(getattr(series, "levels", []) or [series])
        if self.level < len(levels):
            selected = levels[self.level]
        elif len(levels) == 1:
            # A plain TIFF has no pyramid.  Preserve the pipeline's pseudo-level
            # convention while reading only the necessary level-0 rectangle.
            selected = levels[0]
            self._pseudo_scale = 2 ** self.level
        else:
            raise ValueError(
                f"{self.path.name} has {len(levels)} pyramid levels; requested {self.level}"
            )

        self._store = selected.aszarr()
        root = zarr.open(self._store, mode="r")
        if isinstance(root, zarr.Array):
            self._array = root
        else:
            arrays = list(root.arrays())
            if not arrays:
                raise RuntimeError(f"no array found in TIFF zarr store for {self.path}")
            self._array = arrays[0][1]

        self.axes = str(getattr(selected, "axes", "") or _axes_for_shape(self._array.shape))
        self.axes = self.axes.replace("S", "C")
        native_h, native_w = self._yx_shape(self._array.shape, self.axes)
        base_axes = str(getattr(levels[0], "axes", "") or _axes_for_shape(levels[0].shape)).replace("S", "C")
        base_h, base_w = self._yx_shape(levels[0].shape, base_axes)
        self.width = max(1, native_w // self._pseudo_scale)
        self.height = max(1, native_h // self._pseudo_scale)
        self.scale_x = float(base_w) / float(self.width)
        self.scale_y = float(base_h) / float(self.height)
        self.level0_width = int(base_w)
        self.level0_height = int(base_h)

    def _open_pil(self):
        self._pil = Image.open(self.path)
        base_w, base_h = self._pil.size
        self._pseudo_scale = 2 ** self.level
        self.width = max(1, base_w // self._pseudo_scale)
        self.height = max(1, base_h // self._pseudo_scale)
        self.scale_x = float(base_w) / float(self.width)
        self.scale_y = float(base_h) / float(self.height)
        self.level0_width = int(base_w)
        self.level0_height = int(base_h)
        self.axes = "YXC" if len(self._pil.getbands()) > 1 else "YX"

    def _read_zarr_native(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        index = []
        kept_axes = []
        for axis, size in zip(self.axes, self._array.shape):
            if axis == "Y":
                index.append(slice(y0, y1)); kept_axes.append("Y")
            elif axis == "X":
                index.append(slice(x0, x1)); kept_axes.append("X")
            elif self.modality == "dapi" and axis in ("C", "I"):
                channel = int(self.channel_index or 0)
                if channel >= int(size):
                    raise ValueError(
                        f"DAPI channel C{channel} is out of range for "
                        f"{self.path.name}; available channels are C0..C{int(size) - 1}"
                    )
                index.append(channel)
            elif axis == "C":
                index.append(slice(0, min(3, int(size)))); kept_axes.append("C")
            else:
                index.append(0)
        a = np.asarray(self._array[tuple(index)])
        wanted = ["Y", "X"] + (["C"] if "C" in kept_axes else [])
        a = np.transpose(a, [kept_axes.index(axis) for axis in wanted])
        return a

    def read_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.width, int(x1)), min(self.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty image region")

        if self._array is not None:
            s = self._pseudo_scale
            nx0, ny0 = x0 * s, y0 * s
            nx1, ny1 = min(self.level0_width, x1 * s), min(self.level0_height, y1 * s)
            a = self._read_zarr_native(nx0, ny0, nx1, ny1)
        else:
            s = self._pseudo_scale
            a = np.asarray(self._pil.crop((x0 * s, y0 * s, x1 * s, y1 * s)))

        target = (x1 - x0, y1 - y0)
        if a.shape[1] != target[0] or a.shape[0] != target[1]:
            a = cv2.resize(a, target, interpolation=cv2.INTER_AREA)
        if self.modality == "dapi" and a.ndim == 3:
            channel = int(self.channel_index or 0)
            if channel >= a.shape[-1]:
                raise ValueError(
                    f"DAPI channel C{channel} is out of range for "
                    f"{self.path.name}; available channels are C0..C{a.shape[-1] - 1}"
                )
            a = a[..., channel]
        if self.modality == "he":
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, axis=2)
            elif a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
            elif a.shape[2] > 3:
                a = a[..., :3]
        return a

    def close(self):
        try:
            if self._store is not None and hasattr(self._store, "close"):
                self._store.close()
        finally:
            if self._tif is not None:
                self._tif.close()
            if self._pil is not None:
                self._pil.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
