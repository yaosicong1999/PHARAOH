"""VALIS's field splines, fitted tile by tile where points fall, and evaluated in numba.

warp_tools maps a point through a native displacement field with RectBivariateSpline
(interpolating bicubic, s=0) fitted to the whole field: 2 x 12028 x 17381 nodes at
micro_4k, several GB read and seconds of fitting per run, whatever the number of points.

Here the field is cut into TILE x TILE node tiles, and only tiles that contain points are
fitted, each on its tile plus a MARGIN-node border (clipped at the field's edge, where the
fit is then the whole field's own). An interpolating cubic spline's dependence on data k
nodes away decays like (2 - sqrt 3)^k = 0.27^k, so a 24-node border leaves the values within
~1e-13 of the whole-field fit: far below the 1e-3 px this benchmark resolves. Tiles are
fitted and evaluated in parallel threads (FITPACK and the kernel release the GIL).

Evaluation follows the CosMx VALIS evaluation (revision/cosmx/valis/scripts): FITPACK's
fpbspl recurrence, boundary clamp and summation order, but a binary knot search instead of
bispeu's linear scan.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

# The mirrored arithmetic was validated against this VALIS version.
SUPPORTED_VALIS_VERSION = "1.2.0"
TILE, MARGIN = 512, 24


@njit(cache=True, inline="always")
def _fpbspl(t, l0, x, h):
    """FITPACK fpbspl for k=3: the four nonzero B-splines at x (0-based l0 = l-1)."""
    hh = np.empty(4)
    h[0] = 1.0
    for j in range(1, 4):
        for i in range(j):
            hh[i] = h[i]
        h[0] = 0.0
        for i in range(1, j + 1):
            li = l0 + i
            lj = li - j
            f = hh[i - 1] / (t[li] - t[lj])
            h[i - 1] = h[i - 1] + f * (t[li] - x)
            h[i] = f * (x - t[lj])


@njit(cache=True, inline="always")
def _span(t, x):
    """FITPACK fpbisp clamp and interval: largest l0 in [3, n-5] with t[l0] <= x."""
    n = t.shape[0]
    if x < t[3]:
        x = t[3]
    if x > t[n - 4]:
        x = t[n - 4]
    l0 = np.searchsorted(t, x, side="right") - 1
    if l0 < 3:
        l0 = 3
    if l0 > n - 5:
        l0 = n - 5
    return l0, x


@njit(cache=True, nogil=True)
def _bispeu(tx, ty, c, xs, ys, out):
    """Pointwise bicubic evaluation; same coefficient order and sums as fpbisp."""
    nky1 = ty.shape[0] - 4
    hx = np.empty(4)
    hy = np.empty(4)
    for p in range(xs.shape[0]):
        lx, x = _span(tx, xs[p])
        ly, y = _span(ty, ys[p])
        _fpbspl(tx, lx, x, hx)
        _fpbspl(ty, ly, y, hy)
        l1 = (lx - 3) * nky1 + (ly - 3)
        s = 0.0
        for i1 in range(4):
            l2 = l1
            for j1 in range(4):
                s = s + c[l2] * hx[i1] * hy[j1]
                l2 += 1
            l1 += nky1
        out[p] = s


def _coefficients(spline):
    """Knots and coefficients of a fitted bicubic RectBivariateSpline, ready for _bispeu."""
    if tuple(spline.degrees) != (3, 3):
        raise ValueError("evaluation is validated for bicubic splines only")
    tx, ty = spline.get_knots()
    c = np.ascontiguousarray(spline.get_coeffs(), dtype=np.float64)
    if c.size != (len(tx) - 4) * (len(ty) - 4):
        raise ValueError("unexpected spline coefficient layout")
    return np.ascontiguousarray(tx, dtype=np.float64), np.ascontiguousarray(ty, dtype=np.float64), c


class TiledFieldSpline:
    """One native displacement field (2 x H x W): xy + field(xy) as warp_tools computes it.
    A tile is fitted the first time a point falls in it and kept for the run, so points can
    arrive in any number of batches (at most ~1.2 x the field in float64 coefficients)."""

    def __init__(self, tools, field):
        self.tools, self.field = tools, field
        self.height, self.width = field.shape[1:]
        self.tiles = {}

    @property
    def tiles_fitted(self):
        return len(self.tiles)

    def _fit(self, tile_row, tile_col):
        h, w = self.height, self.width
        r0, r1 = max(tile_row * TILE - MARGIN, 0), min((tile_row + 1) * TILE + MARGIN, h)
        c0, c1 = max(tile_col * TILE - MARGIN, 0), min((tile_col + 1) * TILE + MARGIN, w)
        window = np.asarray(self.field[:, r0:r1, c0:c1], dtype=np.float64)
        rows, cols = np.arange(r0, r1), np.arange(c0, c1)
        # bbox ends one node past the data, as warp_tools' [0, h, 0, w] does for the field.
        return tuple(_coefficients(self.tools.RectBivariateSpline(rows, cols, window[component],
                                                                  bbox=[r0, r1, c0, c1]))
                     for component in (0, 1))

    def _tile(self, key, n_cols, xy):
        if key not in self.tiles:
            self.tiles[key] = self._fit(key // n_cols, key % n_cols)
        rows = np.ascontiguousarray(xy[:, 1])
        cols = np.ascontiguousarray(xy[:, 0])
        shift = np.empty((2, len(xy)))
        for component, (tx, ty, c) in enumerate(self.tiles[key]):
            _bispeu(tx, ty, c, rows, cols, shift[component])
        return xy + shift.T

    def apply(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        n_rows, n_cols = -(-self.height // TILE), -(-self.width // TILE)
        tile_row = np.clip((xy[:, 1] // TILE).astype(np.int64), 0, n_rows - 1)
        tile_col = np.clip((xy[:, 0] // TILE).astype(np.int64), 0, n_cols - 1)
        key = tile_row * n_cols + tile_col
        order = np.argsort(key, kind="stable")
        keys, starts = np.unique(key[order], return_index=True)
        groups = np.split(order, starts[1:])
        out = np.empty_like(xy)
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            jobs = [(ids, executor.submit(self._tile, int(k), n_cols, xy[ids])) for k, ids in zip(keys, groups)]
            for ids, job in jobs:
                out[ids] = job.result()
        return out
