"""Shared Stage-6 image outputs, with no GUI imports or widget objects.

CLI calls this directly; the Tk viewer inherits the same computation and adds
only interactive display/error handling. Pillow/font support is loaded solely
when writing the existing optional QC GIF.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from my_utils import read_image, dapi_to_lut_rgb
import overlays
import run_roles
import local_residual
from run_artifacts import initial_alignment_path


def _read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def build_cache(run_dir, prefix, moving, level, tint_level=None, fixed="he"):
    """Output paths shared by CLI and the viewer, named by role token."""
    tint_level = level if tint_level is None else tint_level
    return {
        "moving_base": run_dir / f"{prefix}_cache_{moving}_base_L{level}.png",
        "moving_tint": run_dir / f"{prefix}_cache_{moving}_tint_L{tint_level}.png",
        "fixed_base": run_dir / f"{prefix}_cache_{fixed}_base_L{level}.png",
        "moving_kp": run_dir / f"{prefix}_cache_{moving}_kp_L{level}.png",
        "fixed_kp": run_dir / f"{prefix}_cache_{fixed}_kp_L{level}.png",
        "overlay": run_dir / f"{prefix}_overlay_final_L{level}.png",
        "manual": run_dir / f"{prefix}_overlay_manual_L{level}.png",
        "alternating": run_dir / f"{prefix}_alternating_L{level}.gif",
        "cells": run_dir / f"{prefix}_cells_centroids_L{level}.png",
    }


def add_watermark(img, text):
    """
    Add a simple text watermark to a PIL image.

    Used for the alternating GIF that compares the Stage-2 initial alignment
    with the current final alignment (Stage 5 or Stage 8).
    """
    from PIL import ImageDraw, ImageFont

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    padding = 20
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return img


def draw_points(img, pts, color=(0, 255, 0), r=4, max_points=6000, name=""):
    """
    Draw 2D points on an image.

    Parameters
    ----------
    img : ndarray
        Target image.
    pts : (N, 2) array
        Point coordinates in the same coordinate system as `img`.
    max_points : int
        Maximum number of points to draw directly. If exceeded,
        points are uniformly subsampled for GUI responsiveness.
    """
    out = img.copy()
    h, w = img.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        print(f"[DEBUG] draw_points({name}): empty", flush=True)
        return out

    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
        print(f"[DEBUG] draw_points({name}): subsample -> {len(pts)}", flush=True)

    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    print(f"[DEBUG] draw_points({name}) img=(h={h}, w={w}) pts={len(pts)} valid={valid.sum()}", flush=True)

    for (x, y) in pts[valid]:
        cv2.circle(out, (int(x), int(y)), int(r), color, -1)

    return out


def load_affine(path: Path):
    data = _read_json(path)
    print(f"[DEBUG] affine json keys: {list(data.keys())}", flush=True)

    A2 = None
    A3 = None

    if "affine_3x3" in data:
        A3 = np.array(data["affine_3x3"], dtype=np.float32)
        print(f"[DEBUG] affine_3x3 shape={A3.shape}", flush=True)

    if "affine_2x3" in data:
        A2 = np.array(data["affine_2x3"], dtype=np.float32)
        print(f"[DEBUG] affine_2x3 shape={A2.shape}", flush=True)

    # support stage5 affine output saved as homography_3x3
    if A3 is None and "homography_3x3" in data:
        A3 = np.array(data["homography_3x3"], dtype=np.float32)
        print(f"[DEBUG] homography_3x3 used as affine_3x3, shape={A3.shape}", flush=True)

    if A3 is not None and A3.shape == (3, 3):
        A2 = A3[:2, :].copy().astype(np.float32)
        return A2, A3.astype(np.float32)

    if A2 is not None and A2.shape == (2, 3):
        A3 = np.vstack([A2, [0, 0, 1]]).astype(np.float32)
        return A2, A3

    raise ValueError(
        f"[ERROR] Invalid affine shapes.\n"
        f"affine_2x3={None if A2 is None else A2.shape}\n"
        f"affine_3x3={None if A3 is None else A3.shape}\n"
        f"Raw affine_2x3 value head={str(data.get('affine_2x3'))[:200]}"
    )


def load_homography(path: Path):
    """
    Load perspective transform (homography) from json.
    Returns:
      H2 (2x3), H3 (3x3)
    Notes:
      - For true perspective warp, ALWAYS use H3 with cv2.warpPerspective.
      - H2 is provided only for backward compatibility; it drops the 3rd row.
    """
    data = _read_json(path)
    print(f"[DEBUG] homography json keys: {list(data.keys())}", flush=True)

    H3 = None

    # ---- accept a few possible keys ----
    for k in ("homography_3x3", "H_mat", "H", "matrix_3x3"):
        if k in data:
            H3 = np.array(data[k], dtype=np.float32)
            print(f"[DEBUG] {k} shape={H3.shape}", flush=True)
            break

    if H3 is None:
        raise ValueError(
            f"[ERROR] No homography found in json.\n"
            f"Expected one of keys: homography_3x3 / H_mat / H / matrix_3x3\n"
            f"Got keys: {list(data.keys())}"
        )

    if H3.shape != (3, 3):
        raise ValueError(
            f"[ERROR] Invalid homography shape: {H3.shape}, expected (3,3).\n"
            f"Raw value head={str(H3)[:200]}"
        )

    # Optional sanity: ensure bottom-right not 0 (scale ambiguity is OK, but all-zero is bad)
    if abs(float(H3[2, 2])) < 1e-12:
        print("[WARN] H[2,2] is ~0; homography scale may be unusual. Still returning H.", flush=True)

    # Provide 2x3 "compat" affine-like slice (NOT a true perspective transform)
    H2 = H3[:2, :].copy().astype(np.float32)

    return H2, H3


class AlignmentOutputs:
    """Image/keypoint state and output generation, independent of any viewer."""

    def __init__(self, run_dir, *, minimal_outputs=False, lut_path=None):
        self.run_dir = Path(run_dir)
        self.info = _read_json(self.run_dir / "images_info.json")
        self.roles = run_roles.read(self.info)
        self.case_id = int(self.roles.moving.orientation_case)
        # Preserve Stage 2's source resolution and fixed-image display canvas.
        self.display_level = int(self.roles.fixed.level)
        self.display_scale = float(2 ** self.display_level)
        self.overlay_src_level = int(self.roles.moving.level)
        self.overlay_src_scale = float(2 ** self.overlay_src_level)
        self.cache = build_cache(
            self.run_dir, "6", self.roles.moving.token, self.display_level,
            tint_level=self.overlay_src_level, fixed=self.roles.fixed.token,
        )
        self.lut_path = (Path(lut_path) if lut_path is not None else
                         Path(__file__).resolve().parent / "engine_dapi/glasbey_inverted.lut")
        self.minimal_outputs = bool(minimal_outputs)
        self.alignment_loaded = False
        self.H3 = None
        self.warped_dapi = None
        self.he_dapi_overlay = None
        self.dapi_pts0 = None
        self.he_pts0 = None
        self.selected_indices = None
        self.other_indices = None
        self.tps_inv = None
        self.dapi_rgb = None
        self.dapi_overlay_base = None
        self.he_rgb = None
        self.he16 = None

    def prepare_base_images(self):
        # The LUT display/keypoint panel does not contribute to the final overlay.
        # Minimal CLI therefore never reads its extra moving-image pyramid level.
        if not self.minimal_outputs:
            self.dapi_rgb = self._load_or_build_dapi_base()
        self.dapi_overlay_base = self._load_or_build_dapi_overlay_base()
        self.he_rgb, self.he16 = self._load_or_build_he_base()

    def load_alignment_and_keypoints(self):
        """
        Load nuclei keypoints and the final DAPI-to-H&E transform,
        then build display-space overlays.

        This function:
        1. loads nuclei centroids in level-0 coordinates
        2. loads either homography or TPS transform
        3. warps DAPI into H&E display space
        4. builds final overlay and manual-alignment overlay
        5. caches panel images for fast GUI refresh
        """

        # ---- local TPS helpers ----
        def load_tps(path: Path, direction="forward"):
            """
            direction:
              - 'forward' : dapi -> he
              - 'inverse' : he   -> dapi
            """
            data = _read_json(path)
            print(f"[DEBUG] TPS json keys: {list(data.keys())}", flush=True)

            if direction == "forward":
                key = "forward_tps"
            elif direction == "inverse":
                key = "inverse_tps"
            else:
                raise ValueError("direction must be 'forward' or 'inverse'")

            if key not in data:
                raise ValueError(f"[ERROR] Missing {key} in json. Keys: {list(data.keys())}")

            d = data[key]

            ctrl = np.array(d["control_points_src"], dtype=np.float64)
            w = np.array(d["weights"], dtype=np.float64)
            A2x3 = np.array(d["affine_2x3"], dtype=np.float64)

            if ctrl.ndim != 2 or ctrl.shape[1] != 2:
                raise ValueError(f"[ERROR] ctrl shape invalid: {ctrl.shape}")
            if w.shape[0] != ctrl.shape[0] or w.shape[1] != 2:
                raise ValueError(f"[ERROR] weights shape invalid: {w.shape}, ctrl={ctrl.shape}")
            if A2x3.shape != (2, 3):
                raise ValueError(f"[ERROR] affine_2x3 invalid: {A2x3.shape}")

            # internal TPS affine format: (3,2)
            a = np.zeros((3, 2), dtype=np.float64)
            a[0, :] = A2x3[:, 0]
            a[1, :] = A2x3[:, 1]
            a[2, :] = A2x3[:, 2]

            return {
                "ctrl": ctrl,
                "w": w,
                "a": a,
            }

        def _tps_kernel(r2: np.ndarray) -> np.ndarray:
            out = np.zeros_like(r2, dtype=np.float64)
            mask = r2 > 0
            out[mask] = r2[mask] * np.log(r2[mask])
            return out

        def apply_tps(xy: np.ndarray, model: dict) -> np.ndarray:
            xy = np.asarray(xy, dtype=np.float64)
            ctrl = model["ctrl"]
            w = model["w"]
            a = model["a"]

            diff = xy[:, None, :] - ctrl[None, :, :]
            r2 = np.sum(diff * diff, axis=2)
            K = _tps_kernel(r2)

            P = np.concatenate([np.ones((xy.shape[0], 1)), xy], axis=1)  # (N,3)
            return K @ w + P @ a

        def load_transform_mode(path: Path):
            data = _read_json(path)

            transform_type = str(data.get("transform_type", "")).strip().lower()

            if transform_type == "affine":
                return "affine"
            elif transform_type == "homography":
                return "homography"
            elif transform_type == local_residual.TRANSFORM_TYPE:
                return local_residual.TRANSFORM_TYPE
            elif transform_type in {"tps", "local_tps"}:
                return "tps"

            # fallback for older files
            if "inverse_tps" in data and "forward_tps" in data:
                return "tps"
            elif "homography_3x3" in data:
                return "homography"
            elif "initial_homography_3x3" in data:
                return "tps"

            raise ValueError(f"[ERROR] Unknown transform json format. Keys: {list(data.keys())}")

        def build_tps_remap_for_display(tps_model, out_shape_hw, display_scale):
            """
            Build map_x, map_y in DISPLAY space for cv2.remap.

            Parameters
            ----------
            tps_model : dict
                inverse TPS model: HE(level0) -> DAPI(level0)
            out_shape_hw : (H, W)
                output display image shape, typically he_rgb.shape[:2]
            display_scale : float
                level0_px / display_px
            """
            H, W = out_shape_hw
            s = float(display_scale)

            gx, gy = np.meshgrid(
                np.arange(W, dtype=np.float32),
                np.arange(H, dtype=np.float32)
            )
            pts_disp = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

            # display HE px -> level0 HE px
            pts_he_lvl0 = pts_disp * s

            # inverse TPS: HE(level0) -> DAPI(level0)
            pts_dapi_lvl0 = apply_tps(pts_he_lvl0, tps_model)

            # level0 DAPI px -> display DAPI px
            pts_dapi_disp = pts_dapi_lvl0 / s

            map_x = pts_dapi_disp[:, 0].reshape(H, W).astype(np.float32)
            map_y = pts_dapi_disp[:, 1].reshape(H, W).astype(np.float32)

            return map_x, map_y

        # ---- load nuclei centroids (level 0) ----
        nuclei_path = self.run_dir / "nuclei_centroids_global_used_for_stage5.json"
        if not nuclei_path.exists():
            nuclei_path = self.run_dir / "nuclei_patches/nuclei_centroids_global.json"
        nuclei = _read_json(nuclei_path)['data']
        m_key = f"{self.roles.moving.token}_centroid_global"
        f_key = f"{self.roles.fixed.token}_centroid_global"
        self.dapi_pts0 = np.array([x[m_key] for x in nuclei], np.float32)
        self.he_pts0   = np.array([x[f_key] for x in nuclei], np.float32)
        print(f"[INFO] Loaded nuclei: {len(self.dapi_pts0)}", flush=True)

        # ---- load final transform ----
        tf_path = self.run_dir / f"{self.roles.homography_stem}.json"
        mode = load_transform_mode(tf_path)

        tf_data = _read_json(tf_path)
        sampling = tf_data.get("sampling", {})
        selected_idx = sampling.get("selected_indices_from_original", None)

        n_pts = len(self.dapi_pts0)
        if selected_idx is None:
            # no balanced sampling info -> treat all as selected
            self.selected_indices = np.arange(n_pts, dtype=int)
            self.other_indices = np.array([], dtype=int)
            print("[INFO] No selected_indices_from_original found; treating all points as selected.", flush=True)
        else:
            self.selected_indices = np.array(selected_idx, dtype=int)
            self.selected_indices = self.selected_indices[
                (self.selected_indices >= 0) & (self.selected_indices < n_pts)
            ]

            mask = np.zeros(n_pts, dtype=bool)
            mask[self.selected_indices] = True
            self.other_indices = np.where(~mask)[0]

            sampling_type = str(sampling.get("type", "selected_subset"))
            print(
                f"[INFO] Stage-5 sampling ({sampling_type}): "
                f"selected={len(self.selected_indices)} "
                f"other={len(self.other_indices)} total={n_pts}",
                flush=True,
            )

        # ---- warp DAPI into H&E display space ----
        s = float(self.display_scale)
        src_s = float(self.overlay_src_scale)
        S = np.array([[src_s, 0, 0],
                      [0, src_s, 0],
                      [0, 0, 1]], dtype=np.float32)
        S_inv = np.array([[1 / s, 0, 0],
                          [0, 1 / s, 0],
                          [0, 0, 1]], dtype=np.float32)
        if mode == "affine":
            _, self.H3 = load_affine(tf_path)
            H_disp = (S_inv @ self.H3 @ S).astype(np.float32)

            self.warped_dapi = cv2.warpPerspective(
                self.dapi_overlay_base,
                H_disp,
                (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        elif mode == "homography":
            _, self.H3 = load_homography(tf_path)
            H_disp = (S_inv @ self.H3 @ S).astype(np.float32)

            self.warped_dapi = cv2.warpPerspective(
                self.dapi_overlay_base,
                H_disp,
                (self.he_rgb.shape[1], self.he_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        elif mode == "tps":
            # TPS path: for image warp we must use inverse TPS (HE -> DAPI)
            self.tps_inv = load_tps(tf_path, direction="inverse")

            map_x, map_y = build_tps_remap_for_display(
                self.tps_inv,
                out_shape_hw=self.he_rgb.shape[:2],
                display_scale=self.display_scale
            )

            tps_base = self.dapi_overlay_base
            if self.overlay_src_level != self.display_level:
                tps_base = cv2.resize(
                    tps_base,
                    (int(round(tps_base.shape[1] * self.overlay_src_scale
                               / self.display_scale)),
                     int(round(tps_base.shape[0] * self.overlay_src_scale
                               / self.display_scale))),
                    interpolation=cv2.INTER_AREA)
            self.warped_dapi = cv2.remap(
                tps_base,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

        elif mode == local_residual.TRANSFORM_TYPE:
            # Density alignment is a finite-support residual on top of Stage 2.
            # Build an inverse map on the fixed canvas; pixels outside every local
            # control's radius reduce exactly to the base homography.
            self.H3 = np.asarray(tf_data["base_homography_3x3"], dtype=np.float32)
            map_x, map_y = local_residual.display_remap(
                tf_data,
                out_shape_hw=self.he_rgb.shape[:2],
                fixed_display_scale=self.display_scale,
                moving_display_scale=self.overlay_src_scale,
            )
            self.warped_dapi = cv2.remap(
                self.dapi_overlay_base,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        else:
            raise ValueError(f"Unknown transform mode: {mode}")

        # ---- save final overlay ----
        self.he_dapi_overlay = overlays.blend(self.he_rgb, self.warped_dapi,
                                              self.roles.moving.modality)
        if getattr(self, "minimal_outputs", False):
            # The headless pipeline consumes the final overlay only.  A low
            # PNG compression level preserves every pixel while avoiding a
            # comparatively expensive maximum-compression encode.
            cv2.imwrite(
                str(self.cache["overlay"]),
                self.he_dapi_overlay,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            self.alignment_loaded = True
            return

        cv2.imwrite(str(self.cache["overlay"]), self.he_dapi_overlay)

        # ---- build Stage-2 initial-alignment overlay ----
        data_manual = _read_json(initial_alignment_path(self.run_dir))
        H3m = np.array(data_manual['H_mat_level_0'], dtype=np.float32)
        Hm_disp = (S_inv @ H3m @ S).astype(np.float32)
        warped_dapi_manual = cv2.warpPerspective(
            self.dapi_overlay_base,
            Hm_disp,
            (self.he_rgb.shape[1], self.he_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        he_dapi_overlay_manual = overlays.blend(self.he_rgb, warped_dapi_manual,
                                                self.roles.moving.modality)
        cv2.imwrite(str(self.cache["manual"]), he_dapi_overlay_manual)

        from PIL import Image

        # ---- compare Stage-2 initial against the current final matrix ----
        final_stage = int(tf_data.get("stage", 5))
        final_label = (
            "Stage 8 final refined alignment"
            if final_stage == 8 else f"Stage {final_stage} final alignment"
        )
        initial_frame = add_watermark(
            Image.fromarray(he_dapi_overlay_manual[..., ::-1]),
            "Stage 2 initial alignment",
        )
        final_frame = add_watermark(
            Image.fromarray(self.he_dapi_overlay[..., ::-1]), final_label
        )
        initial_frame.save(
            str(self.cache["alternating"]),
            save_all=True,
            append_images=[final_frame],
            duration=1000,
            loop=0
        )

        # ---- build cached keypoint panels ----
        # 1) moving kp
        p_dapi_kp = self.cache["moving_kp"]
        dapi_kp = self._imread(p_dapi_kp)
        if dapi_kp is None:
            pts = self.dapi_pts0 / self.display_scale
            dapi_kp = draw_points(self.dapi_rgb, pts, color=(0, 255, 0), r=6, name="DAPI nuclei")
            self._imsave(p_dapi_kp, dapi_kp)

        # 2) fixed kp
        p_he_kp = self.cache["fixed_kp"]
        he_kp = self._imread(p_he_kp)
        if he_kp is None:
            pts = self.he_pts0 / self.display_scale
            he_kp = draw_points(self.he_rgb, pts, color=(0, 255, 0), r=6, name="H&E nuclei")
            self._imsave(p_he_kp, he_kp)

        self.alignment_loaded = True

    def _imread(self, p: Path):
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return img
        return None

    def _imsave(self, p: Path, img_bgr: np.ndarray):
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img_bgr)

    def _render_side_base(self, image, level, modality, threshold):
        """A side's display image at `level`, rendered by what it is: DAPI through
        the Glasbey LUT above its Stage-1 cutoff, H&E as the picture it is."""
        if modality == "dapi":
            channel = image.channel_index if image.channel_index is not None else "dapi"
            dapi16, _ = read_image(
                image.path, keep_16bit=True, level=level, channel=channel
            )
            lut_path = self.lut_path
            lut = np.fromfile(str(lut_path), dtype=np.uint8).reshape(256, 3)
            rgb = dapi_to_lut_rgb(dapi16, lut, threshold=int(threshold))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dapi16
        he16, _ = read_image(image.path, keep_16bit=False, level=level, channel="he")
        he8 = cv2.normalize(he16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if he8.ndim == 2:
            return cv2.cvtColor(he8, cv2.COLOR_GRAY2BGR), he16
        return cv2.cvtColor(he8, cv2.COLOR_RGB2BGR), he16

    def _load_or_build_dapi_base(self):
        """The moving side's display image at DISPLAY_LEVEL, cached."""
        img = self._imread(self.cache["moving_base"])
        if img is not None:
            return img
        moving = self.roles.moving
        threshold = moving.threshold if moving.threshold is not None else 300
        bgr, _ = self._render_side_base(moving, self.display_level, moving.modality, threshold)
        if not getattr(self, "minimal_outputs", False):
            self._imsave(self.cache["moving_base"], bgr)
        return bgr

    def _load_or_build_dapi_overlay_base(self):
        """The DAPI as an overlay should show it: intensity, tinted, background at zero.

        Not the Glasbey LUT `self.dapi_rgb` carries. That palette is categorical --
        LUT[1] is [0, 88, 128] and LUT[2] is [136, 184, 0] -- so it renders continuous
        intensity as unrelated colours, and `dapi_to_lut_rgb` scales min to max, which
        lets one bright speck crush the rest of the range. Blended at 0.7/0.8 over the
        H&E the result is unreadable.

        The preparation is the one `2_initial_overlay_original.png` uses: a percentile window
        into a single channel, added into green over the untouched H&E, so the fixed
        image survives wherever the moving one has no signal.
        """
        cached = self.cache.get("moving_tint")
        img = self._imread(cached) if cached is not None else None
        if img is not None:
            return img

        moving = self.roles.moving
        if moving.modality == "dapi":
            channel = moving.channel_index if moving.channel_index is not None else "dapi"
            source, _ = read_image(moving.path, keep_16bit=True,
                                   level=self.overlay_src_level, channel=channel)
        else:
            source, _ = read_image(moving.path, keep_16bit=False,
                                   level=self.overlay_src_level, channel="he")
        bgr = overlays.moving_overlay_base(source, moving.modality)
        if cached is not None and not getattr(self, "minimal_outputs", False):
            self._imsave(cached, bgr)
        return bgr

    def _load_or_build_he_base(self):
        """
        Load cached display-level H&E image if available,
        otherwise read H&E at DISPLAY_LEVEL, normalize to uint8,
        and cache the display image.
        """
        img = self._imread(self.cache["fixed_base"])
        if img is not None:            return img, None

        fixed = self.roles.fixed
        threshold = fixed.threshold if fixed.threshold is not None else 300
        bgr, he16 = self._render_side_base(fixed, self.display_level, fixed.modality, threshold)
        if not getattr(self, "minimal_outputs", False):
            self._imsave(self.cache["fixed_base"], bgr)
        return bgr, he16
