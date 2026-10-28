"""DAPI -> H&E points from RegWSI's native H&E -> DAPI pull field, by numerical inversion.

The field F(c) = c + bilinear(field)(c) lives on the padded common canvas and is never
resampled or re-exported. Each DAPI query is solved for the canvas point c with F(c) equal
to it: damped Newton steps on the exact bilinear Jacobian from nine starts (an affine
seed, the four nearest mapped coarse nodes and four spatially separated ones). A point is
valid only if a start converges in the field's domain to a well-conditioned root with a
positive Jacobian and all converged starts agree within 0.01 um. This is a numerical
property of the stored field, not registration accuracy, and multi-start agreement detects
observed ambiguity; it is not a proof of global injectivity. No ground truth is used.

The solver is the CosMx RegWSI evaluation's compiled kernel (revision/cosmx/regwsi/scripts):
the whole per-start Newton solve with backtracking and the final checks run in one numba
function, with closed-form 2x2 solves, determinants and singular values, parallel over
starts. The four separated starts are picked from the 32 nearest coarse nodes, and only
points that cannot fill four slots fall back to 128, which selects the same seeds.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

REASONS = ("ok", "no_converged_root", "multiple_roots", "nonpositive_jacobian")
FIRST_POOL, FULL_POOL = 32, 128


def read_mha_memmap(path: Path):
    header = {}
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("missing MHA LOCAL data header")
            key, sep, value = line.decode("ascii").strip().partition(" = ")
            if sep:
                header[key] = value
            if key == "ElementDataFile":
                offset = handle.tell()
                break
    if (header.get("ElementDataFile") != "LOCAL" or header.get("CompressedData") != "False"
            or header.get("BinaryDataByteOrderMSB") != "False"):
        raise ValueError("expected an uncompressed little-endian LOCAL MHA")
    dtype = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8"}.get(header.get("ElementType"))
    dims = tuple(map(int, header["DimSize"].split()))
    if dtype is None or len(dims) != 3 or dims[2] != 2:
        raise ValueError("expected a width x height x 2 MHA")
    shape = (2, dims[1], dims[0])
    if Path(path).stat().st_size != offset + np.prod(shape) * np.dtype(dtype).itemsize:
        raise ValueError("MHA payload size does not match its header")
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)


def pull(field, points):
    """Exact value and Jacobian of F(c) = c + bilinear(field)(c). Outside the grid: NaN."""
    points = np.asarray(points, dtype=np.float64)
    h, w = field.shape[1:]
    inside = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) & (points <= [w - 1, h - 1]).all(axis=1)
    safe = np.clip(np.nan_to_num(points), [0, 0], [w - 1, h - 1])
    x0 = np.minimum(np.floor(safe[:, 0]).astype(int), w - 2)
    y0 = np.minimum(np.floor(safe[:, 1]).astype(int), h - 2)
    u, v = safe[:, 0] - x0, safe[:, 1] - y0
    a = np.asarray(field[:, y0, x0], dtype=np.float64).T
    b = np.asarray(field[:, y0, x0 + 1], dtype=np.float64).T
    c = np.asarray(field[:, y0 + 1, x0], dtype=np.float64).T
    d = np.asarray(field[:, y0 + 1, x0 + 1], dtype=np.float64).T
    value = safe + ((1 - u) * (1 - v))[:, None] * a + (u * (1 - v))[:, None] * b
    value += ((1 - u) * v)[:, None] * c + (u * v)[:, None] * d
    jac = np.empty((len(points), 2, 2))
    jac[:, :, 0] = (1 - v)[:, None] * (b - a) + v[:, None] * (d - c)
    jac[:, :, 1] = (1 - u)[:, None] * (c - a) + u[:, None] * (d - b)
    jac += np.eye(2)
    value[~inside], jac[~inside] = np.nan, np.nan
    return value, jac, inside


@njit(cache=True, inline="always")
def _query(field, x, y, value, jac):
    """Bilinear value and analytic Jacobian at one canvas point; False outside the grid."""
    h, w = field.shape[1], field.shape[2]
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= w - 1 and 0 <= y <= h - 1):
        return False
    x0, y0 = min(int(np.floor(x)), w - 2), min(int(np.floor(y)), h - 2)
    u, v = x - x0, y - y0
    wa, wb = (1 - u) * (1 - v), u * (1 - v)
    wc, wd = (1 - u) * v, u * v
    for axis in range(2):
        a, b = np.float64(field[axis, y0, x0]), np.float64(field[axis, y0, x0 + 1])
        c, d = np.float64(field[axis, y0 + 1, x0]), np.float64(field[axis, y0 + 1, x0 + 1])
        base = x if axis == 0 else y
        value[axis] = ((base + wa * a) + wb * b) + (wc * c + wd * d)
        jac[axis, 0] = ((1 - v) * (b - a) + v * (d - c)) + (1. if axis == 0 else 0.)
        jac[axis, 1] = ((1 - u) * (c - a) + u * (d - b)) + (1. if axis == 1 else 0.)
    return True


@njit(cache=True, parallel=True)
def _solve(field, source, seeds, sx, sy, tolerance_um, max_iterations, roots, valid, det_out, correction):
    """Per start: at most max_iterations Newton steps, each accepting the first of up to 16
    halvings that lowers |F - s|; stop on a non-finite or singular Jacobian or a step of at
    most tolerance_um. Valid: in the grid, |det| > 1e-12, condition < 1e6, correction <=
    tolerance_um and residual <= 1e-5 canvas px. (sx, sy) converts canvas px to H&E um."""
    h, w = field.shape[1], field.shape[2]
    for i in prange(len(seeds)):
        value = np.empty(2)
        jac = np.empty((2, 2))
        trial_value = np.empty(2)
        trial_jac = np.empty((2, 2))
        px = min(max(seeds[i, 0], 0.), w - 1.)
        py = min(max(seeds[i, 1], 0.), h - 1.)
        s0, s1 = source[i, 0], source[i, 1]
        for _ in range(max_iterations):
            if not _query(field, px, py, value, jac):
                break
            r0, r1 = value[0] - s0, value[1] - s1
            if not (np.isfinite(r0) and np.isfinite(r1)):
                break
            det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
            if not (np.isfinite(det) and abs(det) > 1e-12):
                break
            step0 = (jac[1, 1] * r0 - jac[0, 1] * r1) / det
            step1 = (-jac[1, 0] * r0 + jac[0, 0] * r1) / det
            if math.hypot(step0 * sx, step1 * sy) <= tolerance_um:
                break
            old = math.hypot(r0, r1)
            accepted = False
            for power in range(16):
                factor = 0.5 ** power
                qx, qy = px - step0 * factor, py - step1 * factor
                if _query(field, qx, qy, trial_value, trial_jac):
                    norm = math.hypot(trial_value[0] - s0, trial_value[1] - s1)
                    if np.isfinite(norm) and norm < old:
                        px, py, accepted = qx, qy, True
                        break
            if not accepted:
                break
        roots[i, 0], roots[i, 1] = px, py
        valid[i] = False
        det_out[i], correction[i] = np.nan, np.inf
        if not _query(field, px, py, value, jac):
            continue
        r0, r1 = value[0] - s0, value[1] - s1
        residual = math.hypot(r0, r1)
        if not (np.isfinite(jac[0, 0]) and np.isfinite(jac[0, 1]) and np.isfinite(jac[1, 0])
                and np.isfinite(jac[1, 1]) and np.isfinite(residual)):
            continue
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        e, f = (jac[0, 0] + jac[1, 1]) / 2, (jac[0, 0] - jac[1, 1]) / 2
        g, hh = (jac[1, 0] + jac[0, 1]) / 2, (jac[1, 0] - jac[0, 1]) / 2
        q, r = math.hypot(e, hh), math.hypot(f, g)
        condition = (q + r) / max(abs(q - r), 1e-300)
        det_out[i] = det
        if abs(det) > 1e-12:
            c0 = (jac[1, 1] * r0 - jac[0, 1] * r1) / det
            c1 = (-jac[1, 0] * r0 + jac[0, 0] * r1) / det
            correction[i] = math.hypot(c0 * sx, c1 * sy)
        # Negative-orientation roots still count toward ambiguity detection.
        valid[i] = (abs(det) > 1e-12 and condition < 1e6 and correction[i] <= tolerance_um
                    and residual <= 1e-5)


@njit(cache=True, parallel=True)
def _pick(roots, valid, det, correction, sx, sy):
    """Per point, the converged start with the smallest correction, if every converged start
    agrees with it within 0.01 um and its Jacobian is positive (reason codes: REASONS)."""
    n, starts = valid.shape
    chosen = np.full((n, 2), np.nan)
    reason = np.zeros(n, dtype=np.int8)
    for i in prange(n):
        best, count = -1, 0
        for k in range(starts):
            if valid[i, k]:
                count += 1
                if best < 0 or correction[i, k] < correction[i, best]:
                    best = k
        if count == 0:
            reason[i] = 1
            continue
        spread = 0.0
        for k in range(starts):
            if valid[i, k]:
                spread = max(spread, math.hypot((roots[i, k, 0] - roots[i, best, 0]) * sx,
                                                (roots[i, k, 1] - roots[i, best, 1]) * sy))
        if spread > 0.01:
            reason[i] = 2
        elif not det[i, best] > 0:
            reason[i] = 3
        else:
            chosen[i, 0], chosen[i, 1] = roots[i, best, 0], roots[i, best, 1]
    return chosen, reason


@njit(cache=True, parallel=True)
def _select_spread(pool, nodes, separation):
    """Up to four mutually separated nodes per point, taken from its pool in k-nearest order:
    a candidate is kept if it lies at least `separation` from every node already kept."""
    n = pool.shape[0]
    spread = np.empty((n, 4, 2))
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        for column in range(pool.shape[1]):
            if counts[i] == 4:
                break
            x, y = nodes[pool[i, column], 0], nodes[pool[i, column], 1]
            good = True
            for slot in range(counts[i]):
                dx, dy = x - spread[i, slot, 0], y - spread[i, slot, 1]
                if not math.sqrt(dx * dx + dy * dy) >= separation:
                    good = False
                    break
            if good:
                spread[i, counts[i], 0], spread[i, counts[i], 1] = x, y
                counts[i] += 1
    return spread, counts


class RegwsiInverse:
    def __init__(self, field, post: dict, frames: dict, coarse_axis: int = 97):
        self.field, self.post = np.asarray(field), post
        self.dapi_scale = np.asarray(frames["dapi"]["level0_per_frame_px"], dtype=float)
        self.he_scale = np.asarray(frames["he"]["level0_per_frame_px"], dtype=float)
        self.initial = float(post.get("initial_resample_ratio", 1) or 1)
        self.pad_source = np.array([post["pad_1"][1][0], post["pad_1"][0][0]], dtype=float)
        self.pad_target = np.array([post["pad_2"][1][0], post["pad_2"][0][0]], dtype=float)
        # Tolerances are physical: canvas pixels -> H&E micrometres, per axis.
        self.he_um_per_common = (self.initial / post["target_resample_ratio"] * self.he_scale
                                 * float(frames["he"]["mpp_level0"]))
        h, w = self.field.shape[1:]
        yy, xx = np.meshgrid(np.linspace(0, h - 1, min(h, coarse_axis)),
                             np.linspace(0, w - 1, min(w, coarse_axis)), indexing="ij")
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        values, jac, _ = pull(self.field, nodes)
        finite = np.isfinite(values).all(axis=1) & np.isfinite(jac).all(axis=(1, 2))
        if finite.sum() < 4:
            raise ValueError("too few finite field samples")
        self.nodes = nodes[finite]
        self.tree = cKDTree(values[finite])
        design = np.column_stack([values[finite], np.ones(finite.sum())])
        self.affine, _, rank, _ = np.linalg.lstsq(design, self.nodes, rcond=None)
        if rank != 3:
            raise ValueError("degenerate affine initialiser")
        det = np.linalg.det(jac[finite])
        self.diagnostics = {"sampled_nodes": int(len(nodes)),
                            "nonfinite_nodes": int((~finite).sum()),
                            "nonpositive_jacobian_fraction": float(np.mean(det <= 0))}

    def source_common(self, dapi_level0):
        return (np.asarray(dapi_level0) / self.dapi_scale * self.post["source_resample_ratio"]
                + self.pad_source) / self.initial

    def common_he(self, common):
        return (common * self.initial - self.pad_target) / self.post["target_resample_ratio"] * self.he_scale

    def _seeds(self, source):
        affine = np.column_stack([source, np.ones(len(source))]) @ self.affine
        full = min(FULL_POOL, len(self.nodes))
        first = min(FIRST_POOL, full)
        separation = 2 * max(self.field.shape[1:]) / 97
        _, pool = self.tree.query(source, k=first, workers=-1)
        nearest = pool[:, :4]  # the k-nearest order is the same for any k
        spread, counts = _select_spread(pool, self.nodes, float(separation))
        refetch = np.flatnonzero(counts < 4) if first < full else np.empty(0, dtype=int)
        if len(refetch):
            _, pool = self.tree.query(source[refetch], k=full, workers=-1)
            spread[refetch], counts[refetch] = _select_spread(pool, self.nodes, float(separation))
        for slot in range(4):
            ids = np.flatnonzero(counts <= slot)
            spread[ids, slot] = spread[ids, counts[ids] - 1]
        return np.concatenate([affine[:, None, :], self.nodes[nearest], spread], axis=1)

    def apply(self, dapi_level0, tolerance_um=1e-5):
        """(H&E level-0 xy, valid, reason code per point; see REASONS)."""
        source = self.source_common(dapi_level0)
        seeds = self._seeds(source)
        n, starts, _ = seeds.shape
        roots = np.empty((n * starts, 2))
        valid = np.zeros(n * starts, dtype=np.bool_)
        det, correction = np.empty(n * starts), np.empty(n * starts)
        sx, sy = (float(v) for v in self.he_um_per_common)
        _solve(self.field, np.ascontiguousarray(np.repeat(source, starts, axis=0)),
               np.ascontiguousarray(seeds.reshape(-1, 2)), sx, sy, float(tolerance_um), 60,
               roots, valid, det, correction)
        chosen, reason = _pick(roots.reshape(n, starts, 2), valid.reshape(n, starts), det.reshape(n, starts),
                               correction.reshape(n, starts), sx, sy)
        return self.common_he(chosen), reason == 0, reason
