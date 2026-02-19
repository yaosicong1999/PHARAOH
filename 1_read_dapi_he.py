import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from my_utils import read_image, extract_hematoxylin_channel, enhance_hematoxylin_channel, dapi_to_lut_rgb
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

dapi_img = None
dapi_mask_img = None  # new for 3rd DAPI
dapi_img_view = None  # 当前用于显示 & mask 的 DAPI（可旋转/翻转）
dapi_lut_img = None   # 用来保存第二行的 LUT 彩色图 (uint8, 3-channel)

dapi_btn_frame = None   # 容器，延迟显示
dapi_gui_affine = np.eye(3, dtype=np.float32)
dapi_orig_shape = None
dapi_gui_shape = None

he_slider = None  # H&E slider
dapi_slider = None       # DAPI LUT slider

he_level = None
dapi_level = None
he_path = None
dapi_path = None

he_blob_count_var = None
dapi_blob_count_var = None


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

def create_blob_mask_from_luted_dapi(luted_dapi):
    gray = cv2.cvtColor(luted_dapi, cv2.COLOR_BGR2GRAY)

    blur_ksize = 3;
    threshold = 10
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    min_area = 500
    filtered_mask = filter_step(mask, min_area=min_area)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoised = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blurred = cv2.GaussianBlur(denoised, (7, 7), 0)
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    mask_filled = ndi.binary_fill_holes(smooth_mask).astype(np.uint8) * 255

    mask_clean = clean_and_cluster_mask(mask_filled, top_k=15, bridge_kernel=15, min_area=2000, dist_thresh=50)
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

def select_dapi():
    global dapi_img, dapi_img_view, dapi_slider, dapi_level, dapi_path
    global dapi_btn_frame, dapi_gui_affine, dapi_orig_shape, dapi_gui_shape
    path = filedialog.askopenfilename(title="Select DAPI Image")
    dapi_path = path
    if not path:
        return
    dapi_img, dapi_level = read_image(path, keep_16bit=True, force_rgb=False, channel="dapi")
    dapi_img_view = dapi_img.copy()
    dapi_orig_shape = dapi_img.shape[:2]
    dapi_gui_shape = dapi_orig_shape
    dapi_gui_affine = np.eye(3, dtype=np.float32)

    update_grid()
    if dapi_slider is None:
        create_dapi_slider()
    else:
        update_dapi(int(dapi_slider.get()))
    if dapi_btn_frame is not None:
        dapi_btn_frame.grid(row=5, column=1, padx=5, pady=5, sticky="we")

    update_confirm_button_state()
    update_dapi_transform_buttons_state()

def create_dapi_slider():
    global dapi_slider
    frame = tk.Frame(root, width=TILE_SIZE[0])
    frame.grid(row=3, column=1, padx=5, pady=6)
    frame.grid_propagate(False)
    tk.Label(frame, text="DAPI LUT Threshold").pack(anchor="w")
    dapi_val = tk.IntVar(value=300)
    dapi_slider = tk.Scale(
        frame,
        from_=0,
        to=2000,
        orient=tk.HORIZONTAL,
        resolution=50,
        showvalue=False,
        length=TILE_SIZE[0] - 50,
        variable=dapi_val,
        command=lambda v: update_dapi(int(v))
    )
    dapi_slider.pack(side="left", fill="x", expand=True)
    tk.Label(
        frame,
        textvariable=dapi_val,
        width=4,
        anchor="e"
    ).pack(side="right")
    update_dapi(300)

def update_dapi(threshold):
    global dapi_mask_img

    if dapi_img_view is None:
        return

    threshold = int(threshold)

    # 1. LUT colored
    global dapi_lut_img
    dapi_rgb = dapi_to_lut_rgb(dapi_img_view, lut, threshold=threshold)
    dapi_lut_img = dapi_rgb

    # 2. blob mask
    dapi_mask_img = create_blob_mask_from_luted_dapi(dapi_rgb)

    # 3. update LUT display
    tile = to_fixed_tile(dapi_rgb)
    tk_img = ImageTk.PhotoImage(tile)
    dapi_lut_label.configure(image=tk_img)
    dapi_lut_label.image = tk_img

    # 4. update mask display
    tile = to_fixed_tile(dapi_mask_img)
    tk_img = ImageTk.PhotoImage(tile)
    dapi_mask_label.configure(image=tk_img)
    dapi_mask_label.image = tk_img

    # 5. update DAPI blob count display
    if dapi_blob_count_var is not None:
        n_dapi = count_components(dapi_mask_img, min_area=2000)
        dapi_blob_count_var.set(f"DAPI blobs: {n_dapi}")

    update_confirm_button_state()

def update_grid():
    # ---- HE original ----
    if he_orig is not None:
        tile = to_fixed_tile(he_orig)
    else:
        tile = make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he_orig_label.configure(image=tk_img)
    he_orig_label.image = tk_img

    # ---- HE dense ----
    if he_dense_mask is not None:
        tile = to_fixed_tile(he_dense_mask)
    else:
        tile = make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    he_dense_label.configure(image=tk_img)
    he_dense_label.image = tk_img

    # ---- DAPI gray ----
    if dapi_img_view is not None:
        dapi_gray = (
            dapi_img_view[..., 0]
            if dapi_img_view.ndim == 3
            else dapi_img_view
        )
        dapi_gray = normalize_to_uint8(dapi_gray)
        tile = to_fixed_tile(dapi_gray)
    else:
        tile = make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    dapi_gray_label.configure(image=tk_img)
    dapi_gray_label.image = tk_img

    # ---- DAPI mask ----
    if dapi_mask_img is not None:
        tile = to_fixed_tile(dapi_mask_img)
    else:
        tile = make_na_tile()
    tk_img = ImageTk.PhotoImage(tile)
    dapi_mask_label.configure(image=tk_img)
    dapi_mask_label.image = tk_img

    # ---- update blob count text ----
    if he_blob_count_var is not None:
        n_he = count_components(he_dense_mask, min_area=2000)
        he_blob_count_var.set(f"HE blobs: {n_he}")

    if dapi_blob_count_var is not None:
        n_dapi = count_components(dapi_mask_img, min_area=2000)
        dapi_blob_count_var.set(f"DAPI blobs: {n_dapi}")

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
    global he_orig, dapi_lut_img, he_dense_mask, dapi_mask_img, RUN_DIR, RUN_ID

    if he_dense_mask is None or dapi_mask_img is None or he_orig is None or dapi_lut_img is None:
        messagebox.showerror("Error", "Please load & threshold both H&E and DAPI first.")
        return

    os.makedirs(RUN_DIR, exist_ok=True)

    # ======================================================
    # 1. Save low-level original images + masks INTO run folder
    # ======================================================
    he_img_path   = os.path.join(RUN_DIR, "1_he_level_image.png")
    dapi_lut_path = os.path.join(RUN_DIR, "1_dapi_lut.png")
    he_mask_path  = os.path.join(RUN_DIR, "1_confirmed_he_dense_mask.png")
    dapi_mask_path= os.path.join(RUN_DIR, "1_confirmed_dapi_mask.png")

    # he_orig 可能是 uint8/uint16；用你已有逻辑保存也行
    Image.fromarray(he_orig).save(he_img_path)
    cv2.imwrite(dapi_lut_path, dapi_lut_img)
    cv2.imwrite(he_mask_path, he_dense_mask)
    cv2.imwrite(dapi_mask_path, dapi_mask_img)

    # ======================================================
    # 2. Save images_info.json INTO run folder
    # ======================================================
    update_he(int(he_slider.get()))
    update_dapi(int(dapi_slider.get()))
    save_current_levels_json(
        json_path=os.path.join(RUN_DIR, "images_info.json"),
        RUN_ID=RUN_ID
    )

    messagebox.showinfo("Saved", f"Step 1 outputs saved to:\n{RUN_DIR}\n\nYou can now run Step 2 in 0_pipeline.")
    root.destroy()
    sys.exit(0)



def infer_dapi_orientation_case(dapi_gui_affine, tol=1e-4):
    """
    Infer which of the 8 orientation cases the GUI transform corresponds to.
    Returns: case_id (0..7)
    """
    A = np.array(dapi_gui_affine, dtype=np.float32)[:2, :2]

    for case_id, A_ref in ORIENTATION_CASES.items():
        if np.allclose(A, A_ref, atol=tol):
            return case_id

    raise ValueError(
        f"Unknown DAPI orientation.\nLinear part:\n{A}"
    )


def save_current_levels_json(json_path="images_info.json", RUN_ID=None):
    global he_path, he_level, dapi_path, dapi_level, dapi_gui_affine
    global he_dense_mask, dapi_mask_img
    global he_slider, dapi_slider

    if he_path is None or dapi_path is None:
        return

    min_area = 2000
    he_blob_count = int(count_components(he_dense_mask, min_area=min_area)) if he_dense_mask is not None else 0
    dapi_blob_count = int(count_components(dapi_mask_img, min_area=min_area)) if dapi_mask_img is not None else 0

    dapi_orientation_case = infer_dapi_orientation_case(dapi_gui_affine)

    # --- slider values (recorded) ---
    he_threshold = int(he_slider.get()) if he_slider is not None else None
    dapi_lut_threshold = int(dapi_slider.get()) if dapi_slider is not None else None

    data = {
        "RUN_ID": RUN_ID,
        "HE_path": he_path,
        "HE_level": he_level,
        "DAPI_path": dapi_path,
        "DAPI_level": dapi_level,
        "DAPI_gui_affine": dapi_gui_affine.tolist(),
        "DAPI_orientation_case": int(dapi_orientation_case),

        # record slider choices
        "HE_threshold": he_threshold,
        "DAPI_LUT_threshold": dapi_lut_threshold,

        "blob_count_min_area": int(min_area),
        "HE_blob_count": he_blob_count,
        "DAPI_blob_count": dapi_blob_count,
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def rotate_dapi_cw():
    global dapi_img_view, dapi_gui_affine, dapi_gui_shape
    if dapi_img_view is None:
        return

    H, W = dapi_gui_shape

    R_cw = np.array([
        [0, -1, H-1],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    dapi_gui_affine = R_cw @ dapi_gui_affine
    dapi_img_view = np.rot90(dapi_img_view, k=3)

    dapi_gui_shape = (W, H)

    update_grid()
    update_dapi(dapi_slider.get() if dapi_slider else 300)

def rotate_dapi_ccw():
    global dapi_img_view, dapi_gui_affine, dapi_gui_shape
    if dapi_img_view is None:
        return

    H, W = dapi_gui_shape

    R_ccw = np.array([
        [0,  1, 0],
        [-1, 0, W - 1],
        [0,  0, 1]
    ], dtype=np.float32)

    dapi_gui_affine = R_ccw @ dapi_gui_affine
    dapi_img_view = np.rot90(dapi_img_view, k=1)

    dapi_gui_shape = (W, H)

    update_grid()
    update_dapi(dapi_slider.get() if dapi_slider else 300)

def flip_dapi_vertical():
    global dapi_img_view, dapi_gui_affine
    if dapi_img_view is None:
        return
    H0, W0 = dapi_gui_shape
    Fv = np.array([
        [1, 0, 0],
        [0, -1, H0 - 1],
        [0, 0, 1]
    ], dtype=np.float32)
    dapi_gui_affine = Fv @ dapi_gui_affine
    dapi_img_view = np.flipud(dapi_img_view)
    update_grid()
    update_dapi(dapi_slider.get() if dapi_slider else 300)

def flip_dapi_horizontal():
    global dapi_img_view, dapi_gui_affine
    if dapi_img_view is None:
        return
    H0, W0 = dapi_gui_shape
    Fh = np.array([
        [-1, 0, W0 - 1],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    dapi_gui_affine = Fh @ dapi_gui_affine
    dapi_img_view = np.fliplr(dapi_img_view)
    update_grid()
    update_dapi(dapi_slider.get() if dapi_slider else 300)

def update_confirm_button_state():
    if (
        he_orig is not None and
        he_dense_mask is not None and
        dapi_lut_img is not None and
        dapi_mask_img is not None
    ):
        confirm_btn.config(state=tk.NORMAL)
    else:
        confirm_btn.config(state=tk.DISABLED)

def update_dapi_transform_buttons_state():
    enabled = (dapi_img_view is not None)

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
    global he_orig_label, he_mask_label, he_dense_label
    global dapi_gray_label, dapi_lut_label, dapi_mask_label
    global confirm_btn
    global btn_rotate_cw, btn_rotate_ccw, btn_flip_v, btn_flip_h


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
    root.title("STEP 1: H&E & DAPI Viewer")
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
    dapi_btn_frame = tk.Frame(root, width=TILE_SIZE[0])
    dapi_btn_frame.grid(row=0, column=1, padx=5, pady=5)
    dapi_btn_frame.grid_propagate(False)
    dapi_btn = tk.Button(
        dapi_btn_frame,
        text="Select DAPI Image",
        command=select_dapi
    )
    dapi_btn.pack(fill="x")

    # -------------------------------
    # Image labels
    # -------------------------------
    he_orig_label = tk.Label(root)
    he_mask_label = tk.Label(root)
    he_dense_label = tk.Label(root)

    dapi_gray_label = tk.Label(root)
    dapi_lut_label = tk.Label(root)
    dapi_mask_label = tk.Label(root)

    for lbl in [
        he_orig_label, he_mask_label, he_dense_label,
        dapi_gray_label, dapi_lut_label, dapi_mask_label
    ]:
        tile = make_na_tile()
        tk_img = ImageTk.PhotoImage(tile)
        lbl.configure(image=tk_img)
        lbl.image = tk_img

    he_orig_label.grid(row=1, column=0, padx=5, pady=5)
    he_mask_label.grid(row=2, column=0, padx=5, pady=5)
    he_dense_label.grid(row=4, column=0, padx=5, pady=5)

    dapi_gray_label.grid(row=1, column=1, padx=5, pady=5)
    dapi_lut_label.grid(row=2, column=1, padx=5, pady=5)
    dapi_mask_label.grid(row=4, column=1, padx=5, pady=5)

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
    global he_blob_count_var, dapi_blob_count_var
    he_blob_count_var = tk.StringVar(value="HE blobs: 0")
    dapi_blob_count_var = tk.StringVar(value="DAPI blobs: 0")

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

    # DAPI blob counter frame (same width as image)
    dapi_count_frame = tk.Frame(root, width=TILE_SIZE[0])
    dapi_count_frame.grid(row=5, column=1, padx=5, pady=(0, 6))
    dapi_count_frame.grid_propagate(False)

    dapi_count_label = tk.Label(
        dapi_count_frame,
        textvariable=dapi_blob_count_var,
        anchor="center"
    )
    dapi_count_label.pack(expand=True, fill="both")

    # ============================================================
    # DAPI Transform Buttons (RIGHT COLUMN, INITIALLY HIDDEN)
    # ============================================================
    dapi_btn_frame = tk.Frame(root, width=TILE_SIZE[0])
    dapi_btn_frame.grid(row=6, column=1, padx=5, pady=5)
    dapi_btn_frame.grid_propagate(False)
    btn_rotate_cw = tk.Button(
        dapi_btn_frame, text="Rotate CW", command=rotate_dapi_cw, state=tk.DISABLED
    )
    btn_rotate_ccw = tk.Button(
        dapi_btn_frame, text="Rotate CCW", command=rotate_dapi_ccw, state=tk.DISABLED
    )
    btn_flip_v = tk.Button(
        dapi_btn_frame, text="Flip V", command=flip_dapi_vertical, state=tk.DISABLED
    )
    btn_flip_h = tk.Button(
        dapi_btn_frame, text="Flip H", command=flip_dapi_horizontal, state=tk.DISABLED
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
    update_dapi_transform_buttons_state()
    root.mainloop()

if __name__ == "__main__":
    main()