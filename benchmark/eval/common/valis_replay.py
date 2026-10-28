"""DAPI -> H&E points by replaying a VALIS run's exported native transform.

The run stores VALIS's own matrices, full native displacement fields and shape arguments
(methods/valis/wrapper.py). Points are mapped with VALIS's own warp_tools arithmetic --
the same as Slide.warp_xy_from_to -- so this needs the valis-wsi version the run used.
Field domains are checked before VALIS's splines are evaluated, which would otherwise
extrapolate silently. For NumPy fields under VALIS 1.2.0 the splines are fitted only on the
field tiles the points fall in (eval/common/valis_splines.py); otherwise the package API
is called on the whole field.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .valis_splines import SUPPORTED_VALIS_VERSION, TiledFieldSpline
REASONS = ("ok", "source_field_out_of_domain", "target_field_out_of_domain")


def _within(xy, size_wh):
    return np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy <= np.asarray(size_wh) - 1).all(axis=1)


class ValisReplay:
    def __init__(self, run_dir: Path, artifact_path: str, frames: dict):
        from valis import warp_tools
        self.tools = warp_tools
        payload = json.loads((Path(run_dir) / artifact_path).read_text())
        version = importlib.metadata.version("valis-wsi")
        if version != payload["valis_version"]:
            raise ValueError(f"run used valis-wsi {payload['valis_version']}, this environment has {version}")
        self.slides = payload["slides"]
        for slide in self.slides.values():
            for name in ("fwd_dxdy", "bk_dxdy"):
                spec = slide[name]
                if spec is None:
                    continue
                path = Path(run_dir) / spec["path"]
                if spec["representation"] == "pyvips":
                    import pyvips
                    slide[name] = pyvips.Image.new_from_file(str(path))
                    shape = [2, slide[name].height, slide[name].width]
                else:
                    slide[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                    shape = list(slide[name].shape)
                if shape != spec["shape"]:
                    raise ValueError("native field shape disagrees with its record")
        self.dapi_rc = [frames["dapi"]["height"], frames["dapi"]["width"]]
        self.he_rc = [frames["he"]["height"], frames["he"]["width"]]
        a, b = self.slides["dapi"], self.slides["he"]
        self.cached = (version == SUPPORTED_VALIS_VERSION
                       and all(f is None or isinstance(f, np.ndarray) for f in (a["fwd_dxdy"], b["bk_dxdy"])))
        if self.cached:
            self.forward_spline = TiledFieldSpline(warp_tools, a["fwd_dxdy"]) if a["fwd_dxdy"] is not None else None
            self.backward_spline = TiledFieldSpline(warp_tools, b["bk_dxdy"]) if b["bk_dxdy"] is not None else None
        self.interpolation = "native_splines_tiled" if self.cached else "native_point_api"

    @staticmethod
    def _field_size(field):
        return (np.asarray(field.shape[1:][::-1]) if isinstance(field, np.ndarray)
                else np.array([field.width, field.height]))

    def _scales(self, slide, size_rc, field, backward=False):
        return self.tools.get_warp_scaling_factors(
            transformation_src_shape_rc=slide["processed_img_shape_rc"],
            transformation_dst_shape_rc=slide["reg_img_shape_rc"], src_shape_rc=size_rc,
            dst_shape_rc=self.slides["dapi"]["aligned_slide_shape_rc"],
            **({"bk_dxdy": field} if backward else {"fwd_dxdy": field}))

    def _forward(self, src):
        a = self.slides["dapi"]
        src_scale, dst_scale, field_scale, _ = self._scales(a, self.dapi_rc, a["fwd_dxdy"])
        points = src / src_scale if src_scale is not None else src
        points = self.tools.warp_xy_rigid(points, np.asarray(a["M"])).astype(float)
        if field_scale is not None:
            points *= field_scale
        if self.forward_spline is not None:
            points = self.forward_spline.apply(points)
        return points * dst_scale if dst_scale is not None else points

    def _inverse_target(self, common):
        b = self.slides["he"]
        src_scale, dst_scale, field_scale, _ = self._scales(b, self.he_rc, b["bk_dxdy"], backward=True)
        points = common / dst_scale if dst_scale is not None else common
        if self.backward_spline is not None:
            points = self.backward_spline.apply(points)  # a lookup, as warp_xy_inv does
        if field_scale is not None:
            points /= field_scale
        points = self.tools.warp_xy_rigid(points, np.linalg.inv(np.asarray(b["M"]))).astype(float)
        return points * src_scale if src_scale is not None else points

    def apply(self, src):
        """DAPI input-frame xy -> (H&E input-frame xy, valid, reason code)."""
        a, b = self.slides["dapi"], self.slides["he"]
        reason = np.zeros(len(src), dtype=np.int8)
        out = np.full_like(src, np.nan, dtype=float)
        rigid = self.tools.warp_xy_rigid(src / (np.asarray(self.dapi_rc[::-1]) / np.asarray(a["processed_img_shape_rc"])[::-1]),
                                         np.asarray(a["M"]))
        valid = np.ones(len(src), bool)
        if a["fwd_dxdy"] is not None:
            size = self._field_size(a["fwd_dxdy"])
            valid &= _within(rigid * size / np.asarray(a["reg_img_shape_rc"])[::-1], size)
        reason[~valid] = 1
        ids = np.flatnonzero(valid)
        if not len(ids):
            return out, valid, reason
        if self.cached:
            common = self._forward(src[ids])
        else:
            common = self.tools.warp_xy(src[ids], M=np.asarray(a["M"]),
                                        transformation_src_shape_rc=a["processed_img_shape_rc"],
                                        transformation_dst_shape_rc=a["reg_img_shape_rc"],
                                        src_shape_rc=self.dapi_rc, dst_shape_rc=a["aligned_slide_shape_rc"],
                                        fwd_dxdy=a["fwd_dxdy"])
        if b["bk_dxdy"] is not None:
            size = self._field_size(b["bk_dxdy"])
            keep = _within(common * size / np.asarray(a["aligned_slide_shape_rc"])[::-1], size)
            valid[ids[~keep]] = False
            reason[ids[~keep]] = 2
            ids, common = ids[keep], common[keep]
        if not len(ids):
            return out, valid, reason
        if self.cached:
            out[ids] = self._inverse_target(common)
        else:
            out[ids] = self.tools.warp_xy_from_to(
                src[ids], from_M=np.asarray(a["M"]),
                from_transformation_src_shape_rc=a["processed_img_shape_rc"],
                from_transformation_dst_shape_rc=a["reg_img_shape_rc"], from_src_shape_rc=self.dapi_rc,
                from_dst_shape_rc=a["aligned_slide_shape_rc"], from_fwd_dxdy=a["fwd_dxdy"],
                to_M=np.asarray(b["M"]), to_transformation_src_shape_rc=b["processed_img_shape_rc"],
                to_transformation_dst_shape_rc=b["reg_img_shape_rc"], to_src_shape_rc=self.he_rc,
                to_dst_shape_rc=a["aligned_slide_shape_rc"], to_bk_dxdy=b["bk_dxdy"])
        return out, valid, reason
