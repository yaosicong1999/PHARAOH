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
    root.title("STEP 8: HE0 & H&E Patch Gallery")
    idx = [0]
    show_auto_centroids = [True]
    # ---------- Labels ----------
    tk.Label(root, text="HE0", font=("Helvetica", 15)).grid(row=0, column=0)
    tk.Label(root, text="HE0 nuclei mask", font=("Helvetica", 15)).grid(row=0, column=1)
    tk.Label(root, text="H&E", font=("Helvetica", 15)).grid(row=0, column=2)
    tk.Label(root, text="H&E nuclei mask", font=("Helvetica", 15)).grid(row=0, column=3)

    he0_label = tk.Label(root)
    he0_mask_label = tk.Label(root)
    he_label = tk.Label(root)
    he_mask_label = tk.Label(root)

    he0_label.grid(row=1, column=0, padx=8, pady=8)
    he0_mask_label.grid(row=1, column=1, padx=8, pady=8)
    he_label.grid(row=1, column=2, padx=8, pady=8)
    he_mask_label.grid(row=1, column=3, padx=8, pady=8)

    info_label = tk.Label(root, font=("Helvetica", 15))
    info_label.grid(row=2, column=0, columnspan=5)

    # ---------- Utils ----------
    def resolve_path(tile_name, nucleus_name, kind, show_overlay):
        overlay_map = {
            "he0_img": f"{tile_name}_nucleus_{nucleus_name}_he0_patch_overlay.png",
            "he0_mask": f"{tile_name}_nucleus_{nucleus_name}_he0_mask_overlay.png",
            "he_img": f"{tile_name}_nucleus_{nucleus_name}_he_patch_overlay.png",
            "he_mask": f"{tile_name}_nucleus_{nucleus_name}_he_mask_overlay.png",
        }
        plain_map = {
            "he0_img": f"{tile_name}_nucleus_{nucleus_name}_he0_patch.png",
            "he0_mask": f"{tile_name}_nucleus_{nucleus_name}_he0_mask.png",
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
            "image_kind": image_kind,  # "he0" / "he" / ...
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

    def calculate_perspective_ransac(transform_type="tps"):
        """
        Fit transform from he0_centroid_global -> he_centroid_global.

        Modes
        -----
        affine:
            affine only
        homography:
            homography only
        tps:
            homography first, then TPS refinement on homography inliers

        Uses nuclei_centroids_global.json under output_folder.
        Saves to: RUN_DIR/he0_to_he_homography_level0.json
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
            if transform_type not in ("homography", "affine", "tps"):
                messagebox.showerror(
                    "Invalid transform_type",
                    "transform_type must be 'homography', 'affine', or 'tps'"
                )
                return

            if len(nuclei_info) < 3 and transform_type == "affine":
                messagebox.showerror("Not enough points", "Need at least 3 nucleus pairs to estimate affine.")
                return
            if len(nuclei_info) < 4 and transform_type in ("homography", "tps"):
                messagebox.showerror("Not enough points",
                                     "Need at least 4 nucleus pairs to estimate homography / TPS init.")
                return

            # src: HE0, dst: HE
            src = np.array([x["he0_centroid_global"] for x in nuclei_info], dtype=np.float32)
            dst = np.array([x["he_centroid_global"] for x in nuclei_info], dtype=np.float32)

            # ---- RANSAC params ----
            ransac_thr = 8.0
            maxIters = 10000
            confidence = 0.995

            H = None
            inliers = None
            tps_model = None

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
                method_str = "cv2.estimateAffine2D(RANSAC) -> 3x3"

                if H is None:
                    messagebox.showerror(
                        "Affine RANSAC failed",
                        "Affine estimation returned None. Try adjusting threshold or check point order / outliers."
                    )
                    return

                # reprojection with affine
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
                method_str = "cv2.findHomography(RANSAC)"

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

            else:  # tps
                # 1) homography init: he0 -> he
                H, inliers = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thr,
                    maxIters=maxIters,
                    confidence=confidence
                )
                method_str = "cv2.findHomography(RANSAC) + TPS (forward+inverse)"

                if H is None:
                    messagebox.showerror(
                        "Homography RANSAC failed",
                        "Homography estimation returned None. Cannot initialize TPS."
                    )
                    return

                mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(src), dtype=bool)
                src_in = src[mask].astype(np.float64)   # he0 inliers
                dst_in = dst[mask].astype(np.float64)   # he   inliers

                if len(src_in) < 3:
                    messagebox.showerror(
                        "Not enough inliers",
                        f"Need at least 3 inliers for TPS, got {len(src_in)}."
                    )
                    return

                # 2) fit BOTH directions
                tps_reg = 1e-3

                # forward: he0 -> he
                tps_model_fwd = fit_tps(src_in, dst_in, reg=tps_reg)

                # inverse: he -> he0
                tps_model_inv = fit_tps(dst_in, src_in, reg=tps_reg)

                # keep forward as primary model for point reprojection stats
                tps_model = tps_model_fwd

                # reprojection with forward TPS on all points
                proj_xy = apply_tps(src.astype(np.float64), tps_model_fwd)
                err = np.linalg.norm(proj_xy - dst.astype(np.float64), axis=1)

            # ---- Error summary ----
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            total = len(nuclei_info)

            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                err_in = err[mask]
                med_err = float(np.median(err_in)) if len(err_in) else float("nan")
                mean_err = float(np.mean(err_in)) if len(err_in) else float("nan")
            else:
                med_err = float(np.median(err))
                mean_err = float(np.mean(err))

            out = {
                "from": "he0_level0",
                "to": "he_level0",
                "method": method_str,
                "ransacReprojThreshold": float(ransac_thr),
                "maxIters": int(maxIters),
                "confidence": float(confidence),
                "num_points": int(total),
                "num_inliers": int(inlier_count),
                "inlier_median_reproj_error_px": med_err,
                "inlier_mean_reproj_error_px": mean_err,
            }

            if transform_type in ("affine", "homography"):
                out["homography_3x3"] = H.tolist()

            else:  # tps
                out["initial_homography_3x3"] = H.tolist()
                out["tps_regularization"] = float(tps_reg)
                # forward TPS: he0 -> he
                out["forward_tps"] = {
                    "from": "he0_level0",
                    "to": "he_level0",
                    "control_points_src": tps_model_fwd["ctrl"].tolist(),
                    "control_points_dst": dst_in.tolist(),
                    "weights": tps_model_fwd["w"].tolist(),
                    "affine_2x3": tps_model_fwd["a"].T.tolist(),
                }
                # inverse TPS: he -> he0
                out["inverse_tps"] = {
                    "from": "he_level0",
                    "to": "he0_level0",
                    "control_points_src": tps_model_inv["ctrl"].tolist(),
                    "control_points_dst": src_in.tolist(),
                    "weights": tps_model_inv["w"].tolist(),
                    "affine_2x3": tps_model_inv["a"].T.tolist(),
                }

            out_path = run_dir / "he0_to_he_homography_level0.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            if transform_type in ("affine", "homography"):
                msg = (
                    f"Saved:\n{out_path}\n\n"
                    f"Inliers: {inlier_count}/{total}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"H (3x3):\n{H}"
                )
            else:
                msg = (
                    f"Saved:\n{out_path}\n\n"
                    f"Homography inliers used for TPS: {inlier_count}/{total}\n"
                    f"Median inlier reproj err: {med_err:.2f}px\n\n"
                    f"Initial H (3x3):\n{H}\n\n"
                    f"TPS control points: {len(tps_model['ctrl'])}"
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
        he0_img_path = resolve_path(tile_name, nucleus_name, "he0_img", show_auto_centroids[0])
        he0_mask_path = resolve_path(tile_name, nucleus_name, "he0_mask", show_auto_centroids[0])
        he_img_path = resolve_path(tile_name, nucleus_name, "he_img", show_auto_centroids[0])
        he_mask_path = resolve_path(tile_name, nucleus_name, "he_mask", show_auto_centroids[0])

        # 小图（展示）
        he0_pil = load_optional(he0_img_path, is_mask=False, case_id=case_id)
        he0_mask_pil = load_optional(he0_mask_path, is_mask=True, case_id=case_id)
        he_pil = load_optional(he_img_path, is_mask=False)
        he_mask_pil = load_optional(he_mask_path, is_mask=True)

        # 原图 numpy（放大拾取）
        raw_he0 = load_raw_numpy(he0_img_path, is_mask=False, case_id=case_id, apply_case=True)
        raw_he0_mask = load_raw_numpy(he0_mask_path, is_mask=True, case_id=case_id, apply_case=True)
        raw_he = load_raw_numpy(he_img_path, is_mask=False, apply_case=False)
        raw_he_mask = load_raw_numpy(he_mask_path, is_mask=True, apply_case=False)

        # Tk images
        imgs = [
            ImageTk.PhotoImage(he0_pil),
            ImageTk.PhotoImage(he0_mask_pil),
            ImageTk.PhotoImage(he_pil),
            ImageTk.PhotoImage(he_mask_pil),
        ]

        labels = [he0_label, he0_mask_label, he_label, he_mask_label]
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

        bind_click(he0_label, raw_he0, "he0")
        bind_click(he0_mask_label, raw_he0_mask, "he0_mask")
        bind_click(he_label, raw_he, "he")
        bind_click(he_mask_label, raw_he_mask, "he_mask")

        info_label.config(text=f"{i + 1}/{len(tile_id)} | {tile_name.split('_', 1)[0].capitalize()}")

    def toggle_centroids():
        show_auto_centroids[0] = not show_auto_centroids[0]
        btn_toggle.config(text=("🙈 Hide auto centroids" if show_auto_centroids[0] else "👁️ Unhide auto centroids"))
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

    # ---- middle button group (under HE0 mask + HE) ----
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

    btn_calc.grid(row=0, column=0, padx=6)
    btn_toggle.grid(row=0, column=1, padx=6)

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
        case_id = json.load(f)['HE0_orientation_case']

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