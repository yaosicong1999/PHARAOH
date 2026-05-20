import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from my_utils import read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel
from sklearn.cluster import DBSCAN
from scipy import ndimage as ndi
from pathlib import Path
import sys
import json
import tkinter.messagebox as messagebox
Image.MAX_IMAGE_PIXELS = None  # disable the check
from PIL import Image, ImageDraw, ImageOps

TILE_SIZE = (320, 320)   # 你可以改成 256×256，和 gallery 一致
length = TILE_SIZE[0]

def make_na_tile(size=TILE_SIZE, bg=240):
    img = Image.new("RGB", size, (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    text = "NOT AVAILABLE NOW"

    w, h = draw.textbbox((0, 0), text)[2:]
    draw.text(
        ((size[0] - w) // 2, (size[1] - h) // 2),
        text,
        fill=(120, 120, 120)
    )
    return img

def to_fixed_tile(img_np, size=TILE_SIZE, bg=240):
    """
    img_np: np.ndarray or None
    returns: PIL.Image (fixed size, padded)
    """
    if img_np is None:
        return make_na_tile(size, bg)

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 3:
        img_np = img_np.copy()

    if img_np.dtype == np.uint16:
        img_np = (img_np / 256).astype(np.uint8)

    pil = Image.fromarray(img_np)
    pil = ImageOps.contain(pil, size)

    canvas = Image.new("RGB", size, (bg, bg, bg))
    x = (size[0] - pil.width) // 2
    y = (size[1] - pil.height) // 2
    canvas.paste(pil, (x, y))
    return canvas

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

def on_close():
    print("Window closed, exiting process.")
    try:
        root.destroy()   # 关闭 Tk
    except:
        pass
    sys.exit(0)          # 结束 Python 进程

ORIENTATION_CASES = {
    0: np.array([[ 1,  0],
                 [ 0,  1]]),   # identity

    1: np.array([[ 0, -1],
                 [ 1,  0]]),   # rot90 CW

    2: np.array([[-1,  0],
                 [ 0, -1]]),   # rot180

    3: np.array([[ 0,  1],
                 [-1,  0]]),   # rot270 CW (90 CCW)

    4: np.array([[ 1,  0],
                 [ 0, -1]]),   # flip vertical

    5: np.array([[-1,  0],
                 [ 0,  1]]),   # flip horizontal

    6: np.array([[0, 1],
                 [1, 0]], np.float32),  # rot90 CW then flip H  (== transpose)

    7: np.array([[0, -1],
                 [-1, 0]], np.float32),  # rot90 CW then flip V  (== anti-transpose)
}

# ---- Load LUT once ----
lut_path = "glasbey_inverted.lut"
lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

# ---- Global storage ----
he_orig = None
he_h_proc = None
he_mask_img = None
he_dense_mask = None

he0_orig = None
he0_h_proc = None
he0_mask_img = None
he0_dense_mask = None

he0_gui_affine = np.eye(3, dtype=np.float32)
he0_orig_shape = None
he0_gui_shape = None
he0_transform_btn_frame = None

he_slider = None  # H&E slider
he0_slider = None  # H&E0 slider

he_level = None
he0_level = None
he_path = None
he0_path = None

he_blob_count_var = None
he0_blob_count_var = None


# ---- Helper Functions ----
def clean_and_cluster_mask(mask, top_k=15, bridge_kernel=15, min_area=5000, dist_thresh=50):
    # Ensure binary
    mask_bin = (mask > 0).astype(np.uint8) * 255
    # Fill holes
    mask_filled = ndi.binary_fill_holes(mask_bin > 0).astype(np.uint8) * 255
    # Break thin bridges
    kernel = np.ones((bridge_kernel, bridge_kernel), np.uint8)
    mask_open = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel)
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_open)
    if num_labels <= 1:
        return mask_open  # nothing to process
    # Extract centroids (ignore background)
    centroids = centroids[1:]
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Cluster components by spatial distance
    clustering = DBSCAN(eps=dist_thresh, min_samples=1).fit(centroids)
    cluster_masks = []
    for cluster_id in np.unique(clustering.labels_):
        members = np.where(clustering.labels_ == cluster_id)[0] + 1  # shift for bg
        cluster_mask = np.isin(labels, members).astype(np.uint8) * 255
        cluster_area = np.sum(cluster_mask > 0)
        if cluster_area >= min_area:
            cluster_masks.append(cluster_mask)
    # Sort clusters by area
    cluster_masks = sorted(cluster_masks, key=lambda m: np.sum(m > 0), reverse=True)
    # Keep top_k clusters
    mask_final = np.zeros_like(mask, dtype=np.uint8)
    for cm in cluster_masks[:top_k]:
        mask_final = cv2.bitwise_or(mask_final, cm)
    return mask_final

def filter_step(mask, min_area=5000):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    return filtered_mask

def create_blob_mask_from_dot_mask(binary_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    he_dense = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    he_opened = cv2.morphologyEx(he_dense, cv2.MORPH_OPEN, kernel)

    blur = cv2.GaussianBlur(he_opened, (9, 9), 0)
    _, mask = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    min_area = 100
    filtered_mask = filter_step(mask, min_area=min_area)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoised = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    _, smooth_mask = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)

    mask_filled = ndi.binary_fill_holes(smooth_mask).astype(np.uint8) * 255

    mask_clean = clean_and_cluster_mask(mask_filled, top_k=25, bridge_kernel=15, min_area=2000, dist_thresh=50)
    return mask_clean

def count_components(mask_u8, min_area=2000):
    """
    Count connected components in a binary/uint8 mask.
    Returns: int (number of components excluding background)
    """
    if mask_u8 is None:
        return 0
    m = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    return int(np.sum(areas >= min_area))

# ---- GUI Functions ----
def select_he():
    global he_orig, he_h_proc, he_mask_img, he_dense_mask, he_slider, he_level, he_path
    path = filedialog.askopenfilename(title="Select H&E Image")
    he_path = path
    if not path:
        return
    he_orig, he_level = read_image(path, channel="he")
    print(f"he_orig shape is", he_orig.shape)
    he_h = extract_hematoxylin_channel(he_orig)
    he_h_proc = enhance_hematoxylin_channel(he_h)
    update_grid()
    if he_slider is None:
        create_he_slider()
    else:
        update_he(int(he_slider.get()))
    update_confirm_button_state()

def select_he0():
    global he0_orig, he0_h_proc, he0_mask_img, he0_dense_mask
    global he0_slider, he0_level, he0_path
    global he0_gui_affine, he0_orig_shape, he0_gui_shape
    global he0_transform_btn_frame

    path = filedialog.askopenfilename(title="Select H&E0 Image")
    he0_path = path
    if not path:
        return

    he0_orig, he0_level = read_image(path, channel="he")
    print(f"he0_orig shape is", he0_orig.shape)

    he0_orig_shape = he0_orig.shape[:2]
    he0_gui_shape = he0_orig_shape
    he0_gui_affine = np.eye(3, dtype=np.float32)

    he0_h = extract_hematoxylin_channel(he0_orig)
    he0_h_proc = enhance_hematoxylin_channel(he0_h)

    update_grid()
    if he0_slider is None:
        create_he0_slider()
    else:
        update_he0(int(he0_slider.get()))

    if he0_transform_btn_frame is not None:
        he0_transform_btn_frame.grid(row=6, column=1, padx=5, pady=5, sticky="we")

    update_confirm_button_state()
    update_he0_transform_buttons_state()

def create_he_slider():
    global he_slider
    frame = tk.Frame(root, width=TILE_SIZE[0])
    frame.grid(row=3, column=0, padx=5, pady=6)
    frame.grid_propagate(False)
    tk.Label(frame, text="H&E Threshold").pack(anchor="w")
    he_val = tk.IntVar(value=240)
    he_slider = tk.Scale(
        frame,
        from_=150,
        to=255,
        orient=tk.HORIZONTAL,
        resolution=5,
        showvalue=False,
        length=TILE_SIZE[0] - 50,   # 给右侧数值留空间
        variable=he_val,
        command=lambda v: update_he(int(v))
    )
    he_slider.pack(side="left", fill="x", expand=True)
    tk.Label(
        frame,
        textvariable=he_val,
        width=4,
        anchor="e"
    ).pack(side="right")
    update_he(240)

def create_he0_slider():
    global he0_slider
    frame = tk.Frame(root, width=TILE_SIZE[0])
    frame.grid(row=3, column=1, padx=5, pady=6)
    frame.grid_propagate(False)
    tk.Label(frame, text="H&E0 Threshold").pack(anchor="w")
    he0_val = tk.IntVar(value=240)
    he0_slider = tk.Scale(
        frame,
        from_=150,
        to=255,
        orient=tk.HORIZONTAL,
        resolution=5,
        showvalue=False,
        length=TILE_SIZE[0] - 50,   # 给右侧数值留空间
        variable=he0_val,
        command=lambda v: update_he0(int(v))
    )
    he0_slider.pack(side="left", fill="x", expand=True)
    tk.Label(
        frame,
        textvariable=he0_val,
        width=4,
        anchor="e"
    ).pack(side="right")
    update_he0(240)


def update_he(threshold):
    global he_mask_img, he_dense_mask
    if he_h_proc is None:
        return
    threshold = int(threshold)

    # 1. threshold H channel
    _, he_mask = cv2.threshold(he_h_proc, threshold, 255, cv2.THRESH_BINARY)
    he_mask_img = he_mask.astype(np.uint8)

    # 2. dense mask
    he_dense_mask = create_blob_mask_from_dot_mask(he_mask_img)

    # 3. update HE mask display
    tile = to_fixed_tile(he_mask_img)
    tk_img = ImageTk.PhotoImage(tile)
    he_mask_label.configure(image=tk_img)
    he_mask_label.image = tk_img

    # 4. update HE dense display
    tile = to_fixed_tile(he_dense_mask)
    tk_img = ImageTk.PhotoImage(tile)
    he_dense_label.configure(image=tk_img)
    he_dense_label.image = tk_img

    # 5. update HE blob count display
    if he_blob_count_var is not None:
        n_he = count_components(he_dense_mask, min_area=2000)
        he_blob_count_var.set(f"HE blobs: {n_he}")

    update_confirm_button_state()

def update_he0(threshold):
    global he0_mask_img, he0_dense_mask
    if he0_h_proc is None:
        return
    threshold = int(threshold)

    # 1. threshold H channel
    _, he0_mask = cv2.threshold(he0_h_proc, threshold, 255, cv2.THRESH_BINARY)
    he0_mask_img = he0_mask.astype(np.uint8)

    # 2. dense mask
    he0_dense_mask = create_blob_mask_from_dot_mask(he0_mask_img)

    # 3. update he0 mask display
    tile = to_fixed_tile(he0_mask_img)
    tk_img = ImageTk.PhotoImage(tile)
    he0_mask_label.configure(image=tk_img)
    he0_mask_label.image = tk_img

    # 4. update he0 dense display
    tile = to_fixed_tile(he0_dense_mask)
    tk_img = ImageTk.PhotoImage(tile)
    he0_dense_label.configure(image=tk_img)
    he0_dense_label.image = tk_img

    # 5. update he0 blob count display
    if he0_blob_count_var is not None:
        n_he0 = count_components(he0_dense_mask, min_area=2000)
        he0_blob_count_var.set(f"he0 blobs: {n_he0}")

    update_he0_transform_buttons_state()
    update_confirm_button_state()


def update_grid():
    # ---- HE original ----
    tile = to_fixed_tile(he_orig) if he_orig is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he_orig_label.configure(image=tk_img)
    he_orig_label.image = tk_img

    # ---- HE mask ----
    tile = to_fixed_tile(he_mask_img) if he_mask_img is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he_mask_label.configure(image=tk_img)
    he_mask_label.image = tk_img

    # ---- HE dense ----
    tile = to_fixed_tile(he_dense_mask) if he_dense_mask is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he_dense_label.configure(image=tk_img)
    he_dense_label.image = tk_img

    # ---- HE0 original ----
    tile = to_fixed_tile(he0_orig) if he0_orig is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he0_orig_label.configure(image=tk_img)
    he0_orig_label.image = tk_img

    # ---- HE0 mask ----
    tile = to_fixed_tile(he0_mask_img) if he0_mask_img is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he0_mask_label.configure(image=tk_img)
    he0_mask_label.image = tk_img

    # ---- HE0 dense ----
    tile = to_fixed_tile(he0_dense_mask) if he0_dense_mask is not None else make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he0_dense_label.configure(image=tk_img)
    he0_dense_label.image = tk_img

    if he_blob_count_var is not None:
        he_blob_count_var.set(f"HE blobs: {count_components(he_dense_mask, min_area=2000)}")
    if he0_blob_count_var is not None:
        he0_blob_count_var.set(f"HE0 blobs: {count_components(he0_dense_mask, min_area=2000)}")

def save_rgb_png(img_np, out_path):
    """Save RGB (or gray) image as PNG in uint8."""
    if img_np is None:
        return
    arr = img_np
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.copy()
    # uint16 -> uint8 (简单线性缩放到 0-255)
    if arr.dtype == np.uint16:
        arr8 = (arr / 256).clip(0, 255).astype(np.uint8)
    else:
        arr8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    Image.fromarray(arr8).save(out_path)


def confirm_and_save():
    global he_orig, he0_orig, he_dense_mask, he0_dense_mask, RUN_DIR, RUN_ID

    if he_dense_mask is None or he0_dense_mask is None or he_orig is None or he0_orig is None:
        messagebox.showerror("Error", "Please load & threshold both H&E and H&E0 first.")
        return

    os.makedirs(RUN_DIR, exist_ok=True)

    # ======================================================
    # 1. Save low-level original images + masks INTO run folder
    # ======================================================
    he_img_path = os.path.join(RUN_DIR, "1_he_level_image.png")
    he0_img_path = os.path.join(RUN_DIR, "1_he0_level_image.png")

    he_mask_row2_path = os.path.join(RUN_DIR, "1_he_threshold_mask.png")
    he0_mask_row2_path = os.path.join(RUN_DIR, "1_he0_threshold_mask.png")

    he_dense_path = os.path.join(RUN_DIR, "1_confirmed_he_dense_mask.png")
    he0_dense_path = os.path.join(RUN_DIR, "1_confirmed_he0_dense_mask.png")

    save_rgb_png(he_orig, he_img_path)
    save_rgb_png(he0_orig, he0_img_path)

    cv2.imwrite(he_mask_row2_path, he_mask_img)
    cv2.imwrite(he0_mask_row2_path, he0_mask_img)

    cv2.imwrite(he_dense_path, he_dense_mask)
    cv2.imwrite(he0_dense_path, he0_dense_mask)

    # ======================================================
    # 2. Save images_info.json INTO run folder
    # ======================================================
    save_current_levels_json(
        json_path=os.path.join(RUN_DIR, "images_info.json"),
        RUN_ID=RUN_ID
    )

    messagebox.showinfo("Saved", f"Step 1 outputs saved to:\n{RUN_DIR}\n\nYou can now run Step 2 in 0_pipeline.")
    root.destroy()
    sys.exit(0)

def infer_he0_orientation_case(he0_gui_affine, tol=1e-4):
    """
    Infer which of the 8 orientation cases the GUI transform corresponds to.
    Returns: case_id (0..7)
    """
    A = np.array(he0_gui_affine, dtype=np.float32)[:2, :2]

    for case_id, A_ref in ORIENTATION_CASES.items():
        if np.allclose(A, A_ref, atol=tol):
            return case_id

    raise ValueError(
        f"Unknown he0 orientation.\nLinear part:\n{A}"
    )


def save_current_levels_json(json_path="images_info.json", RUN_ID=None):
    global he_path, he_level, he0_path, he0_level, he0_gui_affine
    global he_dense_mask, he0_mask_img, he0_dense_mask
    global he_slider, he0_slider

    if he_path is None or he0_path is None:
        return

    min_area = 2000
    he_blob_count = int(count_components(he_dense_mask, min_area=min_area)) if he_dense_mask is not None else 0
    he0_blob_count = int(count_components(he0_dense_mask, min_area=min_area)) if he0_dense_mask is not None else 0

    he0_orientation_case = infer_he0_orientation_case(he0_gui_affine)

    # --- slider values (recorded) ---
    he_threshold = int(he_slider.get()) if he_slider is not None else None
    he0_threshold = int(he0_slider.get()) if he0_slider is not None else None

    data = {
        "RUN_ID": RUN_ID,
        "HE_path": he_path,
        "HE_level": he_level,
        "HE0_path": he0_path,
        "HE0_level": he0_level,
        "HE0_gui_affine": he0_gui_affine.tolist(),
        "HE0_orientation_case": int(he0_orientation_case),

        # record slider choices
        "HE_threshold": he_threshold,
        "HE0_threshold": he0_threshold,

        "blob_count_min_area": int(min_area),
        "HE_blob_count": he_blob_count,
        "HE0_blob_count": he0_blob_count,
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def rotate_he0_cw():
    global he0_dense_mask, he0_mask_img, he0_orig, he0_h_proc, he0_gui_affine, he0_gui_shape
    if he0_dense_mask is None or he0_gui_shape is None:
        return

    H, W = he0_gui_shape
    R_cw = np.array([
        [0, -1, H - 1],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    he0_gui_affine = R_cw @ he0_gui_affine
    he0_dense_mask = np.rot90(he0_dense_mask, k=3)
    he0_mask_img = np.rot90(he0_mask_img, k=3) if he0_mask_img is not None else None
    he0_orig = np.rot90(he0_orig, k=3) if he0_orig is not None else None
    he0_h_proc = np.rot90(he0_h_proc, k=3) if he0_h_proc is not None else None

    he0_gui_shape = (W, H)
    update_grid()
    update_confirm_button_state()
    update_he0_transform_buttons_state()

def rotate_he0_ccw():
    global he0_dense_mask, he0_mask_img, he0_orig, he0_h_proc, he0_gui_affine, he0_gui_shape
    if he0_dense_mask is None or he0_gui_shape is None:
        return

    H, W = he0_gui_shape
    R_ccw = np.array([
        [0,  1, 0],
        [-1, 0, W - 1],
        [0,  0, 1]
    ], dtype=np.float32)

    he0_gui_affine = R_ccw @ he0_gui_affine
    he0_dense_mask = np.rot90(he0_dense_mask, k=1)
    he0_mask_img = np.rot90(he0_mask_img, k=1) if he0_mask_img is not None else None
    he0_orig = np.rot90(he0_orig, k=1) if he0_orig is not None else None
    he0_h_proc = np.rot90(he0_h_proc, k=1) if he0_h_proc is not None else None

    he0_gui_shape = (W, H)
    update_grid()
    update_confirm_button_state()
    update_he0_transform_buttons_state()

def flip_he0_vertical():
    global he0_dense_mask, he0_mask_img, he0_orig, he0_h_proc, he0_gui_affine, he0_gui_shape
    if he0_dense_mask is None or he0_gui_shape is None:
        return

    H0, W0 = he0_gui_shape
    Fv = np.array([
        [1, 0, 0],
        [0, -1, H0 - 1],
        [0, 0, 1]
    ], dtype=np.float32)

    he0_gui_affine = Fv @ he0_gui_affine
    he0_dense_mask = np.flipud(he0_dense_mask)
    he0_mask_img = np.flipud(he0_mask_img) if he0_mask_img is not None else None
    he0_orig = np.flipud(he0_orig) if he0_orig is not None else None
    he0_h_proc = np.flipud(he0_h_proc) if he0_h_proc is not None else None

    update_grid()
    update_confirm_button_state()
    update_he0_transform_buttons_state()

def flip_he0_horizontal():
    global he0_dense_mask, he0_mask_img, he0_orig, he0_h_proc, he0_gui_affine, he0_gui_shape
    if he0_dense_mask is None or he0_gui_shape is None:
        return

    H0, W0 = he0_gui_shape
    Fh = np.array([
        [-1, 0, W0 - 1],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    he0_gui_affine = Fh @ he0_gui_affine
    he0_dense_mask = np.fliplr(he0_dense_mask)
    he0_mask_img = np.fliplr(he0_mask_img) if he0_mask_img is not None else None
    he0_orig = np.fliplr(he0_orig) if he0_orig is not None else None
    he0_h_proc = np.fliplr(he0_h_proc) if he0_h_proc is not None else None

    update_grid()
    update_confirm_button_state()
    update_he0_transform_buttons_state()

def update_confirm_button_state():
    if (
        he_orig is not None and
        he_dense_mask is not None and
        he0_orig is not None and
        he0_dense_mask is not None
    ):
        confirm_btn.config(state=tk.NORMAL)
    else:
        confirm_btn.config(state=tk.DISABLED)

def update_he0_transform_buttons_state():
    enabled = (he0_orig is not None and he0_gui_shape is not None)
    state = tk.NORMAL if enabled else tk.DISABLED
    try:
        btn_rotate_cw.config(state=state)
        btn_rotate_ccw.config(state=state)
        btn_flip_v.config(state=state)
        btn_flip_h.config(state=state)
    except Exception:
        pass

def main():
    global root, RUN_DIR, RUN_ID
    global he0_orig_label, he0_mask_label, he0_dense_label
    global he_orig_label, he_mask_label, he_dense_label
    global confirm_btn
    global btn_rotate_cw, btn_rotate_ccw, btn_flip_v, btn_flip_h
    global he0_btn_frame
    global he0_transform_btn_frame

    RUN_DIR = Path(sys.argv[1]).resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    name = RUN_DIR.name
    if name.startswith("runs_"):
        RUN_ID = name.replace("runs_", "", 1)
    else:
        RUN_ID = name
    print(f"[INFO] RUN_DIR = {RUN_DIR}", flush=True)
    print(f"[INFO] RUN_ID  = {RUN_ID}", flush=True)

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.title("STEP 1: H&E0 & H&E Viewer")
    root.geometry("1000x1100")

    # -------------------------------
    # Top buttons (image-width)
    # -------------------------------
    he_btn_frame = tk.Frame(root, width=TILE_SIZE[0])
    he_btn_frame.grid(row=0, column=0, padx=5, pady=5)
    he_btn_frame.grid_propagate(False)
    he_btn = tk.Button(
        he_btn_frame,
        text="Select H&E Image",
        command=select_he
    )
    he_btn.pack(fill="x")
    he0_select_btn_frame = tk.Frame(root, width=TILE_SIZE[0])
    he0_select_btn_frame.grid(row=0, column=1, padx=5, pady=5)
    he0_select_btn_frame.grid_propagate(False)
    he0_btn = tk.Button(
        he0_select_btn_frame,
        text="Select H&E0 Image",
        command=select_he0
    )
    he0_btn.pack(fill="x")

    # -------------------------------
    # Image labels
    # -------------------------------
    he_orig_label = tk.Label(root)
    he_mask_label = tk.Label(root)
    he_dense_label = tk.Label(root)

    he0_orig_label = tk.Label(root)
    he0_mask_label = tk.Label(root)
    he0_dense_label = tk.Label(root)

    for lbl in [
        he_orig_label, he_mask_label, he_dense_label,
        he0_orig_label, he0_mask_label, he0_dense_label
    ]:
        tile = make_na_tile()
        tk_img = ImageTk.PhotoImage(tile)
        lbl.configure(image=tk_img)
        lbl.image = tk_img

    he_orig_label.grid(row=1, column=0, padx=5, pady=5)
    he_mask_label.grid(row=2, column=0, padx=5, pady=5)
    he_dense_label.grid(row=4, column=0, padx=5, pady=5)

    he0_orig_label.grid(row=1, column=1, padx=5, pady=5)
    he0_mask_label.grid(row=2, column=1, padx=5, pady=5)
    he0_dense_label.grid(row=4, column=1, padx=5, pady=5)

    # -------------------------------
    # Layout weights
    # -------------------------------
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(4, weight=1)

    # -------------------------------
    # Blob count display (centered under images)
    # -------------------------------
    global he_blob_count_var, he0_blob_count_var
    he_blob_count_var = tk.StringVar(value="HE blobs: 0")
    he0_blob_count_var = tk.StringVar(value="HE0 blobs: 0")

    # HE blob counter frame (same width as image)
    he_count_frame = tk.Frame(root, width=TILE_SIZE[0])
    he_count_frame.grid(row=5, column=0, padx=5, pady=(0, 6))
    he_count_frame.grid_propagate(False)
    he_count_label = tk.Label(
        he_count_frame,
        textvariable=he_blob_count_var,
        anchor="center"
    )
    he_count_label.pack(expand=True, fill="both")

    # he0 blob counter frame (same width as image)
    he0_count_frame = tk.Frame(root, width=TILE_SIZE[0])
    he0_count_frame.grid(row=5, column=1, padx=5, pady=(0, 6))
    he0_count_frame.grid_propagate(False)

    he0_count_label = tk.Label(
        he0_count_frame,
        textvariable=he0_blob_count_var,
        anchor="center"
    )
    he0_count_label.pack(expand=True, fill="both")

    # ============================================================
    # he0 Transform Buttons (RIGHT COLUMN, INITIALLY HIDDEN)
    # ============================================================
    he0_transform_btn_frame = tk.Frame(root, width=TILE_SIZE[0])
    he0_transform_btn_frame.grid(row=6, column=1, padx=5, pady=5)
    he0_transform_btn_frame.grid_propagate(False)

    btn_rotate_cw = tk.Button(
        he0_transform_btn_frame, text="Rotate CW", command=rotate_he0_cw, state=tk.DISABLED
    )
    btn_rotate_ccw = tk.Button(
        he0_transform_btn_frame, text="Rotate CCW", command=rotate_he0_ccw, state=tk.DISABLED
    )
    btn_flip_v = tk.Button(
        he0_transform_btn_frame, text="Flip V", command=flip_he0_vertical, state=tk.DISABLED
    )
    btn_flip_h = tk.Button(
        he0_transform_btn_frame, text="Flip H", command=flip_he0_horizontal, state=tk.DISABLED
    )

    for btn in [btn_rotate_cw, btn_rotate_ccw, btn_flip_v, btn_flip_h]:
        btn.pack(side="left", expand=True, fill="x", padx=2)

    # -------------------------------
    # Action buttons (Confirm + Manual Alignment)
    # -------------------------------
    action_frame = tk.Frame(root)
    action_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky="we")
    action_frame.columnconfigure(0, weight=1)
    action_frame.columnconfigure(1, weight=1)

    confirm_btn = tk.Button(
        action_frame,
        text="Confirm & Save Orientation",
        command=confirm_and_save,
        state=tk.DISABLED
    )
    confirm_btn.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10)

    update_confirm_button_state()
    update_he0_transform_buttons_state()
    root.mainloop()

if __name__ == "__main__":
    main()