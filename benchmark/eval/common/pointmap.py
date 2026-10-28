"""One entry point for every method: DAPI level-0 points -> H&E level-0 points of a run.

    pm = PointMap(run_dir)
    he_xy, measured, reason = pm.apply(dapi_xy)

A point is measured if it lies in the DAPI image and the method's own map is defined there
(field domain, a checked inverse root). A prediction that lands outside the H&E image is a
successful but bad registration: its coordinate is kept and measured, with reason
"target_outside_he" (as in the CosMx evaluation). Unmeasured points come back as NaN with a
reason; nothing is edge-extrapolated or dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import contract  # noqa: E402

REASONS = ("ok", "target_outside_he", "source_out_of_domain", "map_undefined")


def _within(xy, width, height):
    return (np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1)
            & (xy[:, 0] <= width - 1) & (xy[:, 1] <= height - 1))


class PointMap:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        record_path = self.run_dir / "run_record.json"
        if not record_path.exists() or json.loads(record_path.read_text()).get("outcome") != "success":
            raise ValueError(f"{run_dir}: not a successful run")
        self.contract = contract.read(self.run_dir)
        self.frames = self.contract["frames"]
        artifact = self.contract["artifact"]
        self.kind = artifact["kind"]
        self.dapi_scale = np.asarray(self.frames["dapi"]["level0_per_frame_px"], dtype=float)
        self.he_scale = np.asarray(self.frames["he"]["level0_per_frame_px"], dtype=float)
        self.native_reasons = None
        if self.kind == "matrix":
            self.matrix = np.loadtxt(self.run_dir / artifact["path"], delimiter=",")
            if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
                raise ValueError("invalid stored matrix")
        elif self.kind == "regwsi_pull_field":
            from .regwsi_inverse import REASONS as native, RegwsiInverse, read_mha_memmap
            post = json.loads((self.run_dir / artifact["postprocessing_path"]).read_text())
            self.native = RegwsiInverse(read_mha_memmap(self.run_dir / artifact["path"]), post, self.frames)
            self.native_reasons = native
        elif self.kind == "valis_native":
            from .valis_replay import REASONS as native, ValisReplay
            self.native = ValisReplay(self.run_dir, artifact["path"], self.frames)
            self.native_reasons = native
        else:
            raise ValueError(f"unsupported artifact kind {self.kind!r}")

    def describe(self) -> dict:
        extra = {}
        if self.kind == "regwsi_pull_field":
            extra = {"field_diagnostics": self.native.diagnostics}
        elif self.kind == "valis_native":
            extra = {"interpolation": self.native.interpolation}
        return {"kind": self.kind, "method": self.contract["method"], **extra}

    def apply(self, dapi_level0, chunk: int = 1_000_000):
        """(H&E level-0 xy, measured mask, reason string per point)."""
        xy = np.asarray(dapi_level0, dtype=float).reshape(-1, 2)
        d, h = self.frames["dapi"], self.frames["he"]
        out = np.full_like(xy, np.nan)
        reason = np.zeros(len(xy), dtype=object)
        reason[:] = "ok"
        valid = _within(xy, d["level0_width"], d["level0_height"])
        reason[~valid] = "source_out_of_domain"
        for start in range(0, len(xy), chunk):
            ids = np.flatnonzero(valid[start:start + chunk]) + start
            if not len(ids):
                continue
            src = xy[ids] / self.dapi_scale
            if self.kind == "matrix":
                hom = np.column_stack([src, np.ones(len(src))]) @ self.matrix.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    dst = hom[:, :2] / hom[:, 2:]
                ok = np.isfinite(dst).all(axis=1)
                native_code = np.where(ok, 0, 1)
                result = dst * self.he_scale
            elif self.kind == "regwsi_pull_field":
                result, ok, native_code = self.native.apply(xy[ids])  # works in level 0 itself
            else:
                dst, ok, native_code = self.native.apply(src)
                result = dst * self.he_scale
            bad = ~ok
            reason[ids[bad]] = [f"map_undefined:{self.native_reasons[c]}" if self.native_reasons else "map_undefined"
                                for c in native_code[bad]]
            inside = _within(result, h["level0_width"], h["level0_height"])
            reason[ids[ok & ~inside]] = "target_outside_he"
            out[ids[ok]] = result[ok]
            valid[ids[bad]] = False
        return out, valid, reason
