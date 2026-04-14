from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import os
import json
import sys
from pathlib import Path
import time

def apply_orientation_to_patch(img, case_id):
    """
    img: np.ndarray (H,W) or (H,W,3)
    case_id: int in [0..7]
    """
    if case_id == 0:
        return img
    if case_id == 1:      # rot90 CW
        return np.rot90(img, k=3)
    if case_id == 2:      # rot180
        return np.rot90(img, k=2)
    if case_id == 3:      # rot90 CCW
        return np.rot90(img, k=1)
    if case_id == 4:      # flip vertical
        return np.flipud(img)
    if case_id == 5:      # flip horizontal
        return np.fliplr(img)
    if case_id == 6:      # rot90 CW + flip H (transpose)
        if img.ndim == 3:
            return np.transpose(np.rot90(img, k=3), (1, 0, 2))
        else:
            return np.transpose(np.rot90(img, k=3))
    if case_id == 7:      # rot90 CW + flip V
        return np.flipud(np.rot90(img, k=3))

    raise ValueError(f"Unknown orientation case: {case_id}")

def show_nucleus_patch_in_memory(
        tile_id,
        nucleus_id,
        output_folder,
        case_id,
        display_size=(256, 256),
        bg_color=(255, 255, 255)
):
    assert len(tile_id) == len(nucleus_id)

    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "Gallery.TButton",
        font=("Helvetica", 12),
        padding=(6, 6),
    )
    root.title("STEP 8: DAPI & H&E Patch Gallery")
    idx = [0]
    show_auto_centroids = [True]
    dapi_mode = ["luted"]
    # ---------- Labels ----------
    tk.Label(root, text="DAPI", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="DAPI nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)

    dapi_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    dapi_label.grid(row=1, column=0, padx=8, pady=8)
    dapi_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # ---------- Utils ----------
    def resolve_path(tile_name, nucleus_name, kind, show_overlay):

        # ---- DAPI image filename depends on mode ----
        if kind == "dapi_img":
            if dapi_mode[0] == "raw":
                plain_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_raw.png"
                overlay_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_raw_overlay.png"
            else:
                plain_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_patch.png"
                overlay_name = f"{tile_name}_nucleus_{nucleus_name}_dapi_patch_overlay.png"

            first = overlay_name if show_overlay else plain_name
            second = plain_name if show_overlay else overlay_name

            p1 = os.path.join(output_folder, first)
            if os.path.exists(p1):
                return p1
            return os.path.join(output_folder, second)

        # ---- other kinds: keep your existing naming ----
        overlay_map = {
            "dapi_mask": f"{tile_name}_nucleus_{nucleus_name}_dapi_mask_overlay.png",
            "he_img": f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png",
            "he_mask": f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png",
        }
        plain_map = {
            "dapi_mask": f"{tile_name}_nucleus_{nucleus_name}_dapi_mask.png",
            "he_img": f"{tile_name}_nucleus_{nucleus_name}_he_patch.png",
            "he_mask": f"{tile_name}_nucleus_{nucleus_name}_he_mask.png",
        }

        first = overlay_map[kind] if show_overlay else plain_map[kind]
        second = plain_map[kind] if show_overlay else overlay_map[kind]

        p1 = os.path.join(output_folder, first)
        if os.path.exists(p1):
            return p1
        return os.path.join(output_folder, second)

    manual_path = Path(output_folder) / "manual_centroids.json"

    def load_raw_numpy(path, is_mask=False, case_id=None, apply_case=False):
        if not os.path.exists(path):
            return None

        is_overlay = path.endswith("_overlay.png")

        if is_mask and not is_overlay:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read mask: {path}")
            if apply_case and case_id is not None:
                img = apply_orientation_to_patch(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if apply_case and case_id is not None:
                img = apply_orientation_to_patch(img, case_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def append_manual_centroid(tile_name, nucleus_name, image_kind, x, y, extra=None):
        rec = {
            "tile_id": tile_name,
            "nucleus_id": nucleus_name,
            "image_kind": image_kind,  # "dapi" / "he" / ...
            "manual_centroid_xy": [int(x), int(y)],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            rec.update(extra)
        with open(manual_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def open_zoom_picker(img_np_rgb, title, tile_name, nucleus_name, image_kind, pixel_zoom=3):
        """
        img_np_rgb: (H,W,3) RGB
        允许在放大图上点选，记录坐标（回到原图坐标系）。
        """
        if img_np_rgb is None:
            messagebox.showwarning("No image", "Image not available.")
            return

        top = tk.Toplevel(root)
        top.withdraw()  # ✅ 一开始就不显示
        top.title(title)
        top.configure(bg="black")
        top.transient(root)  # 可选，但推荐

        top.title(title)
        top.configure(bg="black")

        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()

        H, W = img_np_rgb.shape[:2]
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)

        fit_scale = min(max_w / W, max_h / H)
        scale = min(float(pixel_zoom), float(fit_scale))
        scale = max(scale, 1.0)

        scale_int = int(np.floor(scale))
        scale = max(scale_int, 1)

        disp_w = int(W * scale)
        disp_h = int(H * scale)

        # ---- 转 PIL 并缩放用于显示 ----
        pil = Image.fromarray(img_np_rgb)
        if scale != 1.0:
            pil_disp = pil.resize((disp_w, disp_h), resample=Image.NEAREST)
        else:
            pil_disp = pil

        tk_img = ImageTk.PhotoImage(pil_disp)

        # ---- Canvas 显示 ----
        canvas = tk.Canvas(top, width=disp_w, height=disp_h, highlightthickness=0, bg="black", cursor="crosshair")
        canvas.pack(padx=10, pady=10)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        # 保持引用
        canvas._img_ref = tk_img
        top._scale = scale
        top._orig_W = W
        top._orig_H = H

        info = tk.Label(top, text="", fg="white", bg="black", font=("Helvetica", 14))
        info.pack(pady=(0, 10))

        picked = {"x": None, "y": None, "mark": None}

        def to_orig_xy(cx, cy):
            # canvas坐标 -> 原图坐标
            x = int(cx // scale)
            y = int(cy // scale)
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            return float(x), float(y)

        def on_move(ev):
            x, y = to_orig_xy(ev.x, ev.y)
            info.config(text=f"{image_kind} | tile={tile_name} nucleus={nucleus_name} | x={int(x)}, y={int(y)}")

        def on_click(ev):
            x, y = to_orig_xy(ev.x, ev.y)
            picked["x"], picked["y"] = x, y

            # 画个小十字
            if picked["mark"] is not None:
                canvas.delete(picked["mark"])
            r = 10
            # 用两个line组成十字，tag存一起
            tag = "pick"
            canvas.delete(tag)
            canvas.create_line(ev.x - r, ev.y, ev.x + r, ev.y, fill="green", width=5, tags=tag)
            canvas.create_line(ev.x, ev.y - r, ev.x, ev.y + r, fill="green", width=5, tags=tag)
            picked["mark"] = tag

        def save_and_close(event=None):
            if picked["x"] is None:
                messagebox.showwarning("No point", "Click a point first.")
                return
            append_manual_centroid(
                tile_name, nucleus_name, image_kind,
                picked["x"], picked["y"],
                extra={"scale_used": float(scale)}
            )
            top.destroy()

        def reset_pick(event=None):
            picked["x"], picked["y"] = None, None
            canvas.delete("pick")
            info.config(text=f"{image_kind} | tile={tile_name} nucleus={nucleus_name} | (click to pick)")

        def cancel(event=None):
            top.destroy()

        def reopen(new_zoom):
            top.destroy()
            open_zoom_picker(img_np_rgb, title, tile_name, nucleus_name, image_kind, pixel_zoom=new_zoom)

        top.bind("1", lambda e: reopen(1))
        top.bind("2", lambda e: reopen(2))
        top.bind("3", lambda e: reopen(3))
        top.bind("4", lambda e: reopen(4))

        canvas.bind("<Motion>", on_move)
        canvas.bind("<Button-1>", on_click)
        top.bind("<Return>", save_and_close)
        top.bind("s", save_and_close)
        top.bind("r", reset_pick)
        top.bind("<Escape>", cancel)

        # ---- 居中弹窗 ----
        top.update_idletasks()

        win_w = top.winfo_reqwidth()
        win_h = top.winfo_reqheight()

        screen_w = top.winfo_screenwidth()
        screen_h = top.winfo_screenheight()

        x0 = (screen_w - win_w) // 2
        y0 = (screen_h - win_h) // 2

        top.geometry(f"{win_w}x{win_h}+{x0}+{y0}")  # 尺寸 + 位置一次性给
        top.deiconify()  # ✅ 现在才显示
        top.lift()
        top.focus_force()

        hint = tk.Label(
            top,
            text="Click to pick centroid • Press Enter/S to save • R to reset • Esc to close",
            fg="white", bg="black", font=("Helvetica", 12)
        )
        hint.pack(pady=(0, 10))

    def pad_to_fixed_size(img_pil):
        img_pil = ImageOps.contain(img_pil, display_size)
        canvas = Image.new("RGB", display_size, bg_color)
        x = (display_size[0] - img_pil.width) // 2
        y = (display_size[1] - img_pil.height) // 2
        canvas.paste(img_pil, (x, y))
        return canvas

    def load_optional(path, is_mask=False, case_id=None):
        if os.path.exists(path):
            is_overlay = path.endswith("_overlay.png")

            if is_mask and not is_overlay:
                # ✅ 真正的 mask：灰度
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read mask: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            else:
                # ✅ overlay 或普通 RGB patch
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                if case_id is not None:
                    img = apply_orientation_to_patch(img, case_id)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            img = np.full((display_size[1], display_size[0], 3), 240, np.uint8)
            text = "NOT AVAILABLE NOW"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (display_size[0] - text_w) // 2
            y = (display_size[1] + text_h) // 2
            cv2.putText(img, text, (x, y), font, font_scale, (120, 120, 120), thickness)

        return pad_to_fixed_size(Image.fromarray(img))

    def _tps_kernel(r2: np.ndarray) -> np.ndarray:
        """U(r) = r^2 log(r^2), safe at 0."""
        out = np.zeros_like(r2, dtype=np.float64)
        mask = r2 > 0
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    def fit_tps(src: np.ndarray, dst: np.ndarray, reg: float = 1e-3):
        """
        Fit 2D TPS mapping src -> dst.

        Parameters
        ----------
        src : (N,2)
        dst : (N,2)
        reg : float

        Returns
        -------
        dict with keys:
            ctrl : (N,2) control points
            w    : (N,2) TPS weights
            a    : (3,2) affine part
            reg  : float
        """
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)

        n = src.shape[0]
        if n < 3:
            raise ValueError("TPS needs at least 3 points.")
        if src.shape != dst.shape or src.shape[1] != 2:
            raise ValueError("src and dst must both be (N,2).")

        diff = src[:, None, :] - src[None, :, :]
        r2 = np.sum(diff * diff, axis=2)
        K = _tps_kernel(r2)

        P = np.concatenate([np.ones((n, 1)), src], axis=1)  # (N,3)

        L = np.zeros((n + 3, n + 3), dtype=np.float64)
        L[:n, :n] = K + reg * np.eye(n)
        L[:n, n:] = P
        L[n:, :n] = P.T

        Y = np.zeros((n + 3, 2), dtype=np.float64)
        Y[:n, :] = dst

        params = np.linalg.solve(L, Y)
        w = params[:n, :]  # (N,2)
        a = params[n:, :]  # (3,2)

        return {"ctrl": src, "w": w, "a": a, "reg": float(reg)}

    def apply_tps(xy: np.ndarray, model: dict) -> np.ndarray:
        """
        Apply TPS model to xy, shape (M,2).
        """
        xy = np.asarray(xy, dtype=np.float64)

        ctrl = model["ctrl"]
        w = model["w"]
        a = model["a"]

        diff = xy[:, None, :] - ctrl[None, :, :]
        r2 = np.sum(diff * diff, axis=2)
        K = _tps_kernel(r2)

        P = np.concatenate([np.ones((xy.shape[0], 1)), xy], axis=1)  # (M,3)

        return K @ w + P @ a


    def balanced_grid_sample_pairs(
        src,
        dst,
        grid_shape=(6, 6),
        max_per_cell=4,
        prefer="spread",
        score=None,
        use_dst_for_binning=False,
    ):
        """
        Spatially balanced sampling of point pairs.

        Parameters
        ----------
        src : (N,2)
            source points, usually DAPI
        dst : (N,2)
            destination points, usually HE
        grid_shape : (gx, gy)
            number of bins along x and y
        max_per_cell : int
            keep at most this many pairs per cell
        prefer : str
            "spread" -> farthest-point-like greedy inside each cell
            "random" -> random sampling inside each cell
            "score"  -> top score inside each cell (requires score)
        score : (N,) or None
            optional score, larger = better
        use_dst_for_binning : bool
            if True, bin on dst instead of src

        Returns
        -------
        keep_idx : np.ndarray of shape (M,)
            selected indices
        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        pts = dst if use_dst_for_binning else src

        n = len(pts)
        if n == 0:
            return np.array([], dtype=int)

        gx, gy = int(grid_shape[0]), int(grid_shape[1])
        gx = max(gx, 1)
        gy = max(gy, 1)

        x = pts[:, 0]
        y = pts[:, 1]

        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())

        # avoid zero division
        xr = max(xmax - xmin, 1e-6)
        yr = max(ymax - ymin, 1e-6)

        xi = np.floor((x - xmin) / xr * gx).astype(int)
        yi = np.floor((y - ymin) / yr * gy).astype(int)

        xi = np.clip(xi, 0, gx - 1)
        yi = np.clip(yi, 0, gy - 1)

        cell_to_indices = {}
        for i in range(n):
            key = (xi[i], yi[i])
            cell_to_indices.setdefault(key, []).append(i)

        keep = []

        for key, inds in cell_to_indices.items():
            inds = np.array(inds, dtype=int)

            if len(inds) <= max_per_cell:
                keep.extend(inds.tolist())
                continue

            if prefer == "random":
                picked = np.random.choice(inds, size=max_per_cell, replace=False)

            elif prefer == "score":
                if score is None:
                    raise ValueError("prefer='score' requires score")
                cell_scores = np.asarray(score)[inds]
                order = np.argsort(-cell_scores)
                picked = inds[order[:max_per_cell]]

            else:  # prefer == "spread"
                # greedy farthest-point selection within this cell
                cell_pts = pts[inds]
                picked_local = [0]  # start from first
                remaining = list(range(1, len(cell_pts)))

                while len(picked_local) < max_per_cell and remaining:
                    chosen = cell_pts[picked_local]
                    cand = cell_pts[remaining]

                    # distance to nearest chosen point
                    d2 = ((cand[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
                    min_d2 = d2.min(axis=1)
                    best_j = int(np.argmax(min_d2))
                    picked_local.append(remaining[best_j])
                    remaining.pop(best_j)

                picked = inds[np.array(picked_local, dtype=int)]

            keep.extend(picked.tolist())

        keep = np.array(sorted(set(keep)), dtype=int)
        return keep


    def print_sampling_summary(src, keep_idx, name="balanced sample"):
        src = np.asarray(src)
        kept = src[keep_idx]
        print(f"[INFO] {name}: kept {len(keep_idx)}/{len(src)} pairs", flush=True)
        if len(kept) > 0:
            print(
                f"[INFO] {name}: kept x=[{kept[:,0].min():.1f}, {kept[:,0].max():.1f}] "
                f"y=[{kept[:,1].min():.1f}, {kept[:,1].max():.1f}]",
                flush=True
            )

    def calculate_perspective_ransac(transform_type="tps"):
        """
        Fit transform from dapi_centroid_global -> he_centroid_global.

        Modes
        -----
        affine
        homography
        tps
        local_tps

        Uses spatially balanced sampling before global fitting.
        """
        try:
            run_dir = Path(output_folder).resolve().parent
            info_path = Path(output_folder) / "nuclei_centroids_global.json"
            if not info_path.exists():
                messagebox.showerror("Missing file", f"Cannot find:\n{info_path}")
                return

            with open(info_path, "r") as f:
                nuclei_info = json.load(f)

            transform_type = transform_type.lower().strip()
            if transform_type not in ("homography", "affine", "tps", "local_tps"):
                messagebox.showerror(
                    "Invalid transform_type",
                    "transform_type must be 'homography', 'affine', 'tps', or 'local_tps'"
                )
                return

            if len(nuclei_info) < 3 and transform_type == "affine":
                messagebox.showerror("Not enough points", "Need at least 3 nucleus pairs to estimate affine.")
                return
            if len(nuclei_info) < 4 and transform_type in ("homography", "tps", "local_tps"):
                messagebox.showerror("Not enough points",
                                     "Need at least 4 nucleus pairs to estimate homography / TPS init.")
                return

            # src: DAPI, dst: HE
            src_all = np.array([x["dapi_centroid_global"] for x in nuclei_info], dtype=np.float32)
            dst_all = np.array([x["he_centroid_global"] for x in nuclei_info], dtype=np.float32)

            # -----------------------------------
            # spatially balanced sampling
            # -----------------------------------
            grid_shape = (6, 6)
            max_per_cell = 4

            keep_idx = balanced_grid_sample_pairs(
                src_all,
                dst_all,
                grid_shape=grid_shape,
                max_per_cell=max_per_cell,
                prefer="spread",   # can change to "random" or "score"
                score=None,
                use_dst_for_binning=False
            )

            if transform_type == "affine" and len(keep_idx) < 3:
                keep_idx = np.arange(len(src_all), dtype=int)
            if transform_type in ("homography", "tps", "local_tps") and len(keep_idx) < 4:
                keep_idx = np.arange(len(src_all), dtype=int)

            print_sampling_summary(src_all, keep_idx, name="grid-balanced sample")

            src = src_all[keep_idx]
            dst = dst_all[keep_idx]

            # ---- RANSAC params ----
            ransac_thr = 8.0
            maxIters = 10000
            confidence = 0.995

            H = None
            inliers = None
            tps_model = None
            tps_model_fwd = None
            tps_model_inv = None
            src_in = None
            dst_in = None

            # ---- Estimate transform ----
            if transform_type == "affine":
                A, inliers = cv2.estimateAffine2D(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence,
                    refineIters=10
                )
                if A is not None:
                    H = np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64)
                method_str = "grid-balanced cv2.estimateAffine2D(RANSAC) -> 3x3"

                if H is None:
                    messagebox.showerror(
                        "Affine RANSAC failed",
                        "Affine estimation returned None. Try adjusting threshold or check point order / outliers."
                    )
                    return

                src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
                proj = (H @ src_h.T).T
                proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
                err = np.linalg.norm(proj_xy - dst, axis=1)

            elif transform_type == "homography":
                H, inliers = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence
                )
                method_str = "grid-balanced cv2.findHomography(RANSAC)"

                if H is None:
                    messagebox.showerror(
                        "Homography RANSAC failed",
                        "Homography estimation returned None. Try adjusting threshold or check point order / outliers."
                    )
                    return

                src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
                proj = (H.astype(np.float64) @ src_h.T).T
                proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-8, None)
                err = np.linalg.norm(proj_xy - dst, axis=1)

            else:  # tps or local_tps
                H, inliers = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence
                )

                if transform_type == "tps":
                    method_str = "grid-balanced cv2.findHomography(RANSAC) + TPS (forward+inverse)"
                else:
                    method_str = "grid-balanced cv2.findHomography(RANSAC) + TPS control pairs (LOCAL_TPS-ready)"

                if H is None:
                    messagebox.showerror(
                        "Homography RANSAC failed",
                        "Homography estimation returned None. Cannot initialize TPS."
                    )
                    return

                mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(src), dtype=bool)
                src_in = src[mask].astype(np.float64)
                dst_in = dst[mask].astype(np.float64)

                if len(src_in) < 3:
                    messagebox.showerror(
                        "Not enough inliers",
                        f"Need at least 3 inliers for TPS, got {len(src_in)}."
                    )
                    return

                tps_reg = 1e-2

                # forward: dapi -> he
                tps_model_fwd = fit_tps(src_in, dst_in, reg=tps_reg)

                # inverse: he -> dapi
                tps_model_inv = fit_tps(dst_in, src_in, reg=tps_reg)

                tps_model = tps_model_fwd

                proj_xy = apply_tps(src.astype(np.float64), tps_model_fwd)
                err = np.linalg.norm(proj_xy - dst.astype(np.float64), axis=1)

            # ---- Error summary ----
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            total_balanced = len(src)
            total_raw = len(src_all)

            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                err_in = err[mask]
                med_err = float(np.median(err_in)) if len(err_in) else float("nan")
                mean_err = float(np.mean(err_in)) if len(err_in) else float("nan")
            else:
                med_err = float(np.median(err))
                mean_err = float(np.mean(err))

            out = {
                "from": "dapi_level0",
                "to": "he_level0",
                "method": method_str,
                "transform_type": transform_type,
                "sampling": {
                    "type": "grid_balanced",
                    "grid_shape": list(grid_shape),
                    "max_per_cell": int(max_per_cell),
                    "num_points_before_sampling": int(total_raw),
                    "num_points_after_sampling": int(total_balanced),
                    "selected_indices_from_original": keep_idx.tolist(),
                },
                "ransacReprojThreshold": float(ransac_thr),
                "maxIters": int(maxIters),
                "confidence": float(confidence),
                "num_points": int(total_balanced),
                "num_inliers": int(inlier_count),
                "inlier_median_reproj_error_px": med_err,
                "inlier_mean_reproj_error_px": mean_err,
            }

            if transform_type in ("affine", "homography"):
                out["homography_3x3"] = H.tolist()

            else:
                out["initial_homography_3x3"] = H.tolist()
                out["tps_regularization"] = float(tps_reg)

                out["forward_tps"] = {
                    "from": "dapi_level0",
                    "to": "he_level0",
                    "control_points_src": tps_model_fwd["ctrl"].tolist(),
                    "control_points_dst": dst_in.tolist(),
                    "weights": tps_model_fwd["w"].tolist(),
                    "affine_2x3": tps_model_fwd["a"].T.tolist(),
                }

                out["inverse_tps"] = {
                    "from": "he_level0",
                    "to": "dapi_level0",
                    "control_points_src": tps_model_inv["ctrl"].tolist(),
                    "control_points_dst": src_in.tolist(),
                    "weights": tps_model_inv["w"].tolist(),
                    "affine_2x3": tps_model_inv["a"].T.tolist(),
                }

                if transform_type == "local_tps":
                    out["local_tps_ready"] = True
                    out["note"] = (
                        "This file stores forward/inverse TPS control pairs and global TPS parameters. "
                        "Downstream code may ignore stored weights and refit local TPS on neighborhoods at runtime."
                    )

            out_path = run_dir / "dapi_to_he_homography_level0.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            if transform_type in ("affine", "homography"):
                msg = (
                    f"Saved:\n{out_path}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Inliers: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"H (3x3):\n{H}"
                )
            elif transform_type == "tps":
                msg = (
                    f"Saved:\n{out_path}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Homography inliers used for TPS: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"Initial H (3x3):\n{H}\n\n"
                    f"TPS control points: {len(tps_model['ctrl'])}"
                )
            else:
                msg = (
                    f"Saved:\n{out_path}\n\n"
                    f"Raw points: {total_raw}\n"
                    f"Balanced sample: {total_balanced}\n"
                    f"Homography inliers used for LOCAL_TPS-ready control pairs: {inlier_count}/{total_balanced}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"Initial H (3x3):\n{H}\n\n"
                    f"Control points saved: {len(tps_model['ctrl'])}"
                )

            messagebox.showinfo("Transform estimated", msg)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------- Core ----------
    def update_images():
        i = idx[0]
        tile_name = tile_id[i]
        nucleus_name = nucleus_id[i]

        base = output_folder
        # 根据 toggle 选择 overlay / plain
        dapi_img_path = resolve_path(tile_name, nucleus_name, "dapi_img", show_auto_centroids[0])
        dapi_mask_path = resolve_path(tile_name, nucleus_name, "dapi_mask", show_auto_centroids[0])
        he_img_path = resolve_path(tile_name, nucleus_name, "he_img", show_auto_centroids[0])
        he_mask_path = resolve_path(tile_name, nucleus_name, "he_mask", show_auto_centroids[0])

        # 小图（展示）
        dapi_pil = load_optional(dapi_img_path, is_mask=False, case_id=case_id)
        dapi_mask_pil = load_optional(dapi_mask_path, is_mask=True, case_id=case_id)
        he_pil = load_optional(he_img_path, is_mask=False)
        he_mask_pil = load_optional(he_mask_path, is_mask=True)

        # 原图 numpy（放大拾取）
        raw_dapi = load_raw_numpy(dapi_img_path, is_mask=False, case_id=case_id, apply_case=True)
        raw_dapi_mask = load_raw_numpy(dapi_mask_path, is_mask=True, case_id=case_id, apply_case=True)
        raw_he = load_raw_numpy(he_img_path, is_mask=False, apply_case=False)
        raw_he_mask = load_raw_numpy(he_mask_path, is_mask=True, apply_case=False)

        # Tk images
        imgs = [
            ImageTk.PhotoImage(dapi_pil),
            ImageTk.PhotoImage(dapi_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]

        labels = [dapi_label, dapi_mask_label, he_label, he_mask_label]
        for lbl, im in zip(labels, imgs):
            lbl.config(image=im)
            lbl.image = im

        def bind_click(label_widget, raw_img, kind):
            label_widget.bind(
                "<Button-1>",
                lambda ev: open_zoom_picker(
                    raw_img,
                    title=f"{kind.upper()} | {tile_name} nucleus {nucleus_name}",
                    tile_name=tile_name,
                    nucleus_name=nucleus_name,
                    image_kind=kind,
                    pixel_zoom=10
                )
            )

        bind_click(dapi_label, raw_dapi, "dapi")
        bind_click(dapi_mask_label, raw_dapi_mask, "dapi_mask")
        bind_click(he_label, raw_he, "he")
        bind_click(he_mask_label, raw_he_mask, "he_mask")

        info_label.config(text=f"{i + 1}/{len(tile_id)} | {tile_name.split('_', 1)[0].capitalize()}")

    def toggle_centroids():
        show_auto_centroids[0] = not show_auto_centroids[0]
        btn_toggle.config(text=("🙈 Hide auto centroids" if show_auto_centroids[0] else "👁️ Unhide auto centroids"))
        update_images()

    def toggle_dapi_mode():
        dapi_mode[0] = "raw" if dapi_mode[0] == "luted" else "luted"
        btn_dapi_mode.config(text=("🧬 DAPI: RAW" if dapi_mode[0] == "raw" else "🎨 DAPI: LUT"))
        update_images()

    def next_patch():
        idx[0] = (idx[0] + 1) % len(tile_id)
        update_images()

    def prev_patch():
        idx[0] = (idx[0] - 1) % len(tile_id)
        update_images()

    # ---------- Buttons ----------
    btn_prev = ttk.Button(root, text="⟨ Previous", command=prev_patch)
    btn_next = ttk.Button(root, text="Next ⟩", command=next_patch)

    # ---- middle button group (under DAPI mask + HE) ----
    mid_btn_frame = ttk.Frame(root)
    mid_btn_frame.grid(row=3, column=1, columnspan=2, pady=(8, 8))

    btn_calc = ttk.Button(
        mid_btn_frame,
        text="🧮 Calculate alignment matrix",
        style="Gallery.TButton",
        command=calculate_perspective_ransac
    )

    btn_toggle = ttk.Button(
        mid_btn_frame,
        text="🙈 Hide auto centroids",
        style="Gallery.TButton",
        command=toggle_centroids
    )

    btn_dapi_mode = ttk.Button(
        mid_btn_frame,
        text="🎨 DAPI: LUT",  # 或你现在用的文字
        style="Gallery.TButton",
        command=toggle_dapi_mode
    )

    btn_calc.grid(row=0, column=0, padx=6)
    btn_toggle.grid(row=0, column=1, padx=6)
    btn_dapi_mode.grid(row=0, column=2, padx=6)

    # ---- left / right navigation buttons ----
    btn_prev.grid(row=3, column=0, sticky="w", padx=(20, 10), pady=(8, 8))
    btn_next.grid(row=3, column=3, sticky="e", padx=(10, 20), pady=(8, 8))

    # ---- optional: keep middle truly centered when window resizes ----
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)

    update_images()
    root.mainloop()

def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python 8_nucleus_gallery.py <patches_folder>")

    run_dir = Path(sys.argv[1]).resolve()
    nuclei_dir = run_dir / "nuclei_patches"

    info_path = nuclei_dir / "nuclei_centroids_global.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing nuclei info: {info_path}")

    with open(info_path, "r") as f:
        nuclei_info = json.load(f)

    with open(os.path.join(nuclei_dir, "../images_info.json"), "r") as f:
        case_id = json.load(f)['DAPI_orientation_case']

    tile_id = [x['tile_id'] for x in nuclei_info]
    nucleus_id = [x['nucleus_id'] for x in nuclei_info]

    show_nucleus_patch_in_memory(
        tile_id=tile_id,
        nucleus_id=nucleus_id,
        output_folder=nuclei_dir,
        case_id=case_id
    )

if __name__ == "__main__":
    main()