import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys
import json, os
from tkinter import messagebox
from pathlib import Path


class BlobMatcherApp:
    def __init__(
        self,
        root,
        img_a_path,
        img_b_path,
        image_a_level,
        image_b_level,
        display_height=400,
        dapi_gui_affine=None,
    ):
        self.root = root
        self.root.title("STEP 2: Interactive Blob Matcher")
        self.display_height = display_height

        self.image_a_level = int(image_a_level)
        self.image_b_level = int(image_b_level)
        self.dapi_gui_affine = dapi_gui_affine  # ← 保存，仅用于 Confirm

        # Load masks (already GUI-transformed)
        self.img_A = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
        _, self.img_A_binary = cv2.threshold(self.img_A, 127, 255, cv2.THRESH_BINARY)

        self.img_B = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)
        _, self.img_B_binary = cv2.threshold(self.img_B, 127, 255, cv2.THRESH_BINARY)

        _, self.labels_A, _, self.centroids_A = cv2.connectedComponentsWithStats(self.img_A_binary)
        _, self.labels_B, _, self.centroids_B = cv2.connectedComponentsWithStats(self.img_B_binary)

        self.red_blob_labels = []
        self.blue_blob_labels = []
        self.red_centroids = []
        self.blue_centroids = []

        self.alignment_cache = {}

        self.setup_gui()
        self.update_displays()

    # ==========================================================
    # GUI
    # ==========================================================
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(expand=True, fill=tk.BOTH)

        frame_a = ttk.Frame(main_frame)
        frame_b = ttk.Frame(main_frame)
        frame_c = ttk.Frame(main_frame)
        frame_a.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        frame_b.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        frame_c.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)

        ttk.Label(frame_a, text="H&E (A)").pack()
        self.label_img_a = ttk.Label(frame_a, cursor="crosshair")
        self.label_img_a.pack(expand=True)
        self.label_img_a.bind("<Button-1>", lambda e: self.on_image_click(e, "A"))

        ttk.Label(frame_b, text="DAPI (B)").pack()
        self.label_img_b = ttk.Label(frame_b, cursor="crosshair")
        self.label_img_b.pack(expand=True)
        self.label_img_b.bind("<Button-1>", lambda e: self.on_image_click(e, "B"))

        ttk.Label(frame_c, text="Aligned (C)").pack()
        self.label_img_c = ttk.Label(frame_c)
        self.label_img_c.pack(expand=True)

        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.pack(fill=tk.X)

        self.status_var = tk.StringVar()
        ttk.Label(ctrl, textvariable=self.status_var).pack(side=tk.LEFT)

        ttk.Button(ctrl, text="Reset", command=self.reset).pack(side=tk.RIGHT)

        ttk.Button(
            ctrl,
            text="Confirm Initial Alignment",
            command=self.confirm_initial_alignment,
            state=tk.DISABLED,
        ).pack(side=tk.RIGHT, padx=6)

        self.confirm_button = ctrl.winfo_children()[-1]

    # ==========================================================
    # Interaction
    # ==========================================================
    def reset(self):
        self.red_blob_labels.clear()
        self.blue_blob_labels.clear()
        self.red_centroids.clear()
        self.blue_centroids.clear()
        self.alignment_cache.clear()
        self.update_displays()

    def on_image_click(self, event, img_type):
        widget = event.widget
        if img_type == "A":
            img = self.img_A
            labels = self.labels_A
            centroids = self.centroids_A
            sel_labels = self.red_blob_labels
            sel_centroids = self.red_centroids
        else:
            img = self.img_B
            labels = self.labels_B
            centroids = self.centroids_B
            sel_labels = self.blue_blob_labels
            sel_centroids = self.blue_centroids

        H, W = img.shape
        xi = int(event.x * W / max(1, widget.winfo_width()))
        yi = int(event.y * H / max(1, widget.winfo_height()))
        xi = np.clip(xi, 0, W - 1)
        yi = np.clip(yi, 0, H - 1)

        lbl = int(labels[yi, xi])
        if lbl == 0:
            return

        if lbl in sel_labels:
            i = sel_labels.index(lbl)
            sel_labels.pop(i)
            sel_centroids.pop(i)
        else:
            sel_labels.append(lbl)
            sel_centroids.append(tuple(centroids[lbl]))

        self.update_displays()

    # ==========================================================
    # Alignment (GUI only)
    # ==========================================================
    def compute_and_cache_alignment(self, red_lbls, blue_lbls, pts_A, pts_B):
        key = (tuple(red_lbls), tuple(blue_lbls))
        if key in self.alignment_cache:
            return self.alignment_cache[key].copy()

        H_mat, _ = cv2.estimateAffine2D(pts_B, pts_A, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H_mat is None:
            return None

        warped = cv2.warpAffine(
            self.img_B, H_mat, (self.img_A.shape[1], self.img_A.shape[0])
        )
        img_c = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        for lbl in red_lbls:
            img_c[self.labels_A == lbl] = (0, 0, 255)

        warped_lbls_B = cv2.warpAffine(
            self.labels_B.astype(np.float32),
            H_mat,
            (self.img_A.shape[1], self.img_A.shape[0]),
            flags=cv2.INTER_NEAREST,
        ).astype(np.int32)

        for lbl in blue_lbls:
            img_c[warped_lbls_B == lbl] = (255, 0, 0)
        # -------------------------------------------------
        # Draw numbers (click order)
        # -------------------------------------------------
        k = min(len(pts_A), len(pts_B))
        for i in range(k):
            rx, ry = pts_A[i]
            bx, by = pts_B[i]

            cv2.putText(
                img_c, str(i + 1),
                (int(round(rx)), int(round(ry))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        self.alignment_cache[key] = img_c.copy()
        return img_c

    # ==========================================================
    # Display
    # ==========================================================
    def update_displays(self):
        preview_a = self.generate_preview(self.img_A, self.labels_A, self.red_blob_labels, self.red_centroids, (0, 0, 255))
        preview_b = self.generate_preview(self.img_B, self.labels_B, self.blue_blob_labels, self.blue_centroids, (255, 0, 0))

        k = min(len(self.red_centroids), len(self.blue_centroids))
        img_c = np.zeros((*self.img_A.shape, 3), dtype=np.uint8)

        if k >= 3:
            img_tmp = self.compute_and_cache_alignment(
                self.red_blob_labels[:k],
                self.blue_blob_labels[:k],
                np.array(self.red_centroids[:k], np.float32),
                np.array(self.blue_centroids[:k], np.float32),
            )
            if img_tmp is not None:
                img_c = img_tmp

        self.photo_a = self.cv_to_photoimage(preview_a)
        self.photo_b = self.cv_to_photoimage(preview_b)
        self.photo_c = self.cv_to_photoimage(img_c)

        self.label_img_a.config(image=self.photo_a)
        self.label_img_b.config(image=self.photo_b)
        self.label_img_c.config(image=self.photo_c)

        self.status_var.set(f"A selected: {len(self.red_centroids)} | B selected: {len(self.blue_centroids)}")
        self.confirm_button.config(state=tk.NORMAL if k >= 3 else tk.DISABLED)

    # ==========================================================
    # CONFIRM — ONLY HERE we compose H_total
    # ==========================================================
    def confirm_initial_alignment(self):
        k = min(len(self.red_centroids), len(self.blue_centroids))
        if k < 3:
            return

        pts_A = np.array(self.red_centroids[:k], np.float32)
        pts_B_gui = np.array(self.blue_centroids[:k], np.float32)

        # ---- GUI affine -> 3x3 ----
        T_gui = np.array(self.dapi_gui_affine, dtype=np.float32)
        if T_gui.shape == (2, 3):
            T_gui = np.vstack([T_gui, [0, 0, 1]])

        T_gui_inv = np.linalg.inv(T_gui)

        # ---- GUI coords -> original DAPI coords ----
        pts_B_gui_h = np.hstack([
            pts_B_gui,
            np.ones((len(pts_B_gui), 1), dtype=np.float32)
        ])

        pts_B_orig = (T_gui_inv @ pts_B_gui_h.T).T[:, :2]

        # ---- Estimate affine in ORIGINAL coordinate system ----
        H_mat, _ = cv2.estimateAffine2D(
            pts_B_orig,
            pts_A,
            method=cv2.RANSAC
        )
        colored_HE = self.generate_preview(
            self.img_A,
            self.labels_A,
            self.red_blob_labels,
            self.red_centroids,
            color=(0, 0, 255)  # red
        )
        colored_DAPI = self.generate_preview(
            self.img_B,
            self.labels_B,
            self.blue_blob_labels,
            self.blue_centroids,
            color=(255, 0, 0)  # blue
        )
        # Use the most recent aligned overlay (reuse from cache)
        key = (tuple(self.red_blob_labels[:k]), tuple(self.blue_blob_labels[:k]))
        if key in self.alignment_cache:
            overlay_img = self.alignment_cache[key]
        else:
            overlay_img = self.compute_and_cache_alignment(
                self.red_blob_labels[:k],
                self.blue_blob_labels[:k],
                np.array(self.red_centroids[:k], dtype=np.float32),
                np.array(self.blue_centroids[:k], dtype=np.float32),
            )
        cv2.imwrite(os.path.join(RUN_DIR, "2_colored_HE.png"), colored_HE)
        cv2.imwrite(os.path.join(RUN_DIR, "2_colored_DAPI.png"), colored_DAPI)
        cv2.imwrite(os.path.join(RUN_DIR, "2_overlay_HE_DAPI.png"), overlay_img)

        if H_mat is None:
            messagebox.showerror("Error", "Affine estimation failed.")
            return

        H_total = H_mat
        with open(os.path.join(RUN_DIR, "clicked_blob_initial_alignment.json"), "w") as f:
            json.dump({"H_mat": H_total.tolist()}, f, indent=2)

        messagebox.showinfo(
            "Step 2 Saved",
            "Alignment confirmed and saved successfully.\n\n"
            "You can now return to the pipeline and run Step 3."
        )

        self.root.destroy()

    # ==========================================================
    # Utils
    # ==========================================================
    @staticmethod
    def generate_preview(img, labels, active, centroids, color):
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, lbl in enumerate(active):
            out[labels == lbl] = color
            cx, cy = map(int, centroids[i])
            cv2.putText(out, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return out

    def cv_to_photoimage(self, img):
        h, w = img.shape[:2]
        scale = self.display_height / h
        img = cv2.resize(img, (int(w * scale), self.display_height))
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


# ==========================================================
# main
# ==========================================================
if __name__ == "__main__":
    RUN_DIR = sys.argv[1]
    with open(os.path.join(RUN_DIR, "images_info.json"), "r") as f:
        info = json.load(f)

    RUN_ID = info.get("RUN_ID", Path(RUN_DIR).name.replace("runs_", "", 1))

    HE_MASK_PATH = os.path.join(RUN_DIR, "1_confirmed_he_dense_mask.png")
    DAPI_MASK_PATH = os.path.join(RUN_DIR, "1_confirmed_dapi_mask.png")

    output_folder = os.path.join(RUN_DIR, "tiles")
    os.makedirs(output_folder, exist_ok=True)

    dapi_gui_affine = np.array(info["DAPI_gui_affine"], dtype=np.float32)

    root = tk.Tk()
    app = BlobMatcherApp(
        root,
        HE_MASK_PATH,
        DAPI_MASK_PATH,
        info["HE_level"],
        info["DAPI_level"],
        dapi_gui_affine=dapi_gui_affine,
    )
    root.mainloop()