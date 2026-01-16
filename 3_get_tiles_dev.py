from PIL import Image, ImageTk, ImageOps
import os
from sklearn.cluster import DBSCAN
from scipy import ndimage as ndi
import sys
import json
Image.MAX_IMAGE_PIXELS = None  # disable the check
import numpy as np
import cv2
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from my_utils import read_image, dapi_to_lut_rgb
import tkinter as tk
import os
import time

class StepTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def mark(self, name):
        now = time.perf_counter()
        print(f"[TIMER] {name:<40s}: {now - self.last:8.2f} s   (total {now - self.t0:8.2f} s)")
        self.last = now


def apply_gui_affine_to_image(img, T_gui):
    """
    img: H x W x C  (RGB or BGR)
    T_gui: 3x3 affine matrix from GUI
    """
    h, w = img.shape[:2]
    M = T_gui[:2, :]  # OpenCV needs 2x3

    warped = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )
    return warped


def show_patch_gallery_in_memory(dapi_patches, he_patches, display_size=(256, 256), bg_color=(255, 255, 255)):
    """
    Display paired DAPI and H&E patches side by side in a Tkinter window.
    Each image is centered and padded to a fixed display size without distortion.
    """
    assert len(dapi_patches) == len(he_patches), "Patch lists must be of equal length"
    root = tk.Tk()
    root.title("DAPI ↔ H&E Patch Gallery")

    idx = [0]

    dapi_label = tk.Label(root)
    he_label = tk.Label(root)
    dapi_label.grid(row=0, column=0, padx=10, pady=10)
    he_label.grid(row=0, column=1, padx=10, pady=10)

    info_label = tk.Label(root, text="", font=("Arial", 12))
    info_label.grid(row=1, column=0, columnspan=2, pady=5)

    def pad_to_fixed_size(img_pil, target_size, bg_color=(255, 255, 255)):
        """
        Resize while keeping aspect ratio, then pad to target_size and center the image.
        """
        img_pil = ImageOps.contain(img_pil, target_size, Image.LANCZOS)
        new_img = Image.new("RGB", target_size, bg_color)
        x = (target_size[0] - img_pil.width) // 2
        y = (target_size[1] - img_pil.height) // 2
        new_img.paste(img_pil, (x, y))
        return new_img

    def update_images():
        i = idx[0]
        dapi_patch = dapi_patches[i]
        he_patch = he_patches[i]

        # Convert OpenCV (BGR) → PIL (RGB)
        dapi_img = Image.fromarray(cv2.cvtColor(dapi_patch["img"], cv2.COLOR_BGR2RGB))
        he_img = Image.fromarray(cv2.cvtColor(he_patch["img"], cv2.COLOR_BGR2RGB))

        # Resize and pad to same display size
        dapi_pil = pad_to_fixed_size(dapi_img, display_size, bg_color)
        he_pil = pad_to_fixed_size(he_img, display_size, bg_color)

        # Convert for Tkinter
        dapi_tk = ImageTk.PhotoImage(dapi_pil)
        he_tk = ImageTk.PhotoImage(he_pil)

        dapi_label.config(image=dapi_tk)
        he_label.config(image=he_tk)
        dapi_label.image = dapi_tk
        he_label.image = he_tk

        info_label.config(text=f"Patch {i+1}/{len(dapi_patches)}   |   Type: {dapi_patch['type']}")

    def next_patch():
        idx[0] = (idx[0] + 1) % len(dapi_patches)
        update_images()

    def prev_patch():
        idx[0] = (idx[0] - 1) % len(dapi_patches)
        update_images()

    # Buttons
    btn_prev = tk.Button(root, text="⟨ Previous", command=prev_patch, width=12)
    btn_next = tk.Button(root, text="Next ⟩", command=next_patch, width=12)
    btn_prev.grid(row=2, column=0, pady=10)
    btn_next.grid(row=2, column=1, pady=10)

    update_images()
    root.mainloop()

def clean_and_cluster_mask(mask, top_k=15, bridge_kernel=15, min_area=5000, dist_thresh=50):
    # Ensure binary
    mask_bin = (mask > 0).astype(np.uint8) * 255
    # Fill holes
    # mask_filled = ndi.binary_fill_holes(mask_bin > 0).astype(np.uint8) * 255
    # Break thin bridges
    kernel = np.ones((bridge_kernel, bridge_kernel), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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
    cv2.imwrite('3_dapi_mask_smooth.png', smooth_mask)

    # mask_filled = ndi.binary_fill_holes(smooth_mask).astype(np.uint8) * 255

    mask = (smooth_mask/255).astype(np.uint8)
    # Ensure binary 0/1
    # Invert mask to get holes as foreground
    holes = 1 - mask
    # Label connected components in the holes
    labeled_holes, num_holes = ndi.label(holes)
    # Count area of each hole
    hole_areas = ndi.sum(np.ones_like(holes), labeled_holes, index=np.arange(1, num_holes + 1))
    # Identify holes to fill (small ones)
    small_holes_labels = np.arange(1, num_holes + 1)[hole_areas <= 800]
    # Vectorized filling
    mask_filled = mask.copy()
    if len(small_holes_labels) > 0:
        mask_filled[np.isin(labeled_holes, small_holes_labels)] = 1
    # Convert to 0/255 and save
    mask_filled = (mask_filled * 255).astype(np.uint8)
    cv2.imwrite("3_dapi_mask_filled.png", mask_filled)

    mask_clean = clean_and_cluster_mask(mask_filled, top_k=50, bridge_kernel=15, min_area=2000, dist_thresh=50)
    return mask_clean

def get_valid_coords(mask_available, max_points=200_000):
    ys, xs = np.where(mask_available == 0)
    coords = np.column_stack((xs, ys))
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
    return coords

## For sampling:
# ============================================================
# --- 1. Create mask of available areas
# ============================================================
def create_available_mask(mask, giant_tiles_dict):
    """
    Create a binary mask for available areas:
      0 = available (can sample)
      255 = unavailable
    Uses:
      - only giant tiles from `giant_tiles_dict`
    Small patches are NOT marked as available.
    """
    mask_available = np.ones_like(mask, dtype=np.uint8) * 255  # start unavailable

    # Mark all giant blob tiles as available
    for squares in giant_tiles_dict.values():
        for sq in squares:
            x0, y0, w, h = int(sq['x0']), int(sq['y0']), int(sq['w']), int(sq['h'])
            mask_available[y0:y0+h, x0:x0+w] = 0

    return mask_available

# ============================================================
# --- 2. Initialize random points
# ============================================================
def initialize_points(mask_available, N_total, existing_points, MIN_DIST=35, MIN_HOLE_DIST=15):
    coords_valid = get_valid_coords(mask_available)

    hole_coords = np.column_stack(np.where(mask_available > 0))
    hole_tree = cKDTree(hole_coords) if len(hole_coords) > 0 else None

    points = existing_points.copy()
    attempts = 0
    while len(points) < N_total and attempts < 50000:
        idx = np.random.randint(len(coords_valid))
        x, y = coords_valid[idx]
        if any(np.linalg.norm(np.array([x, y]) - p) < MIN_DIST for p in points):
            attempts += 1
            continue
        if hole_tree is not None and hole_tree.query([x, y])[0] < MIN_HOLE_DIST:
            attempts += 1
            continue
        points = np.vstack([points, [x, y]])
        attempts += 1
    return points

# ============================================================
# --- 3. Enforce minimal spacing (asymmetric version)
# ============================================================
def enforce_min_distances(points, coords_valid, min_dist, n_existing=0):
    """
    Keep at least min_dist between points.
    Existing points (first n_existing) are fixed — only new ones move.
    """
    tree = cKDTree(points)
    pairs = tree.query_pairs(min_dist)

    for i, j in pairs:
        d = np.linalg.norm(points[i] - points[j])
        if d < min_dist:
            shift = (min_dist - d) / 2
            vec = points[j] - points[i]
            if np.all(vec == 0):
                vec = np.random.randn(2)
            vec = vec / np.linalg.norm(vec) * shift

            # existing vs new
            if i < n_existing and j >= n_existing:
                # move only new point j
                new_j = points[j] + vec * 2
                points[j] = coords_valid[np.argmin(np.sum((coords_valid - new_j) ** 2, axis=1))]
            elif j < n_existing and i >= n_existing:
                # move only new point i
                new_i = points[i] - vec * 2
                points[i] = coords_valid[np.argmin(np.sum((coords_valid - new_i) ** 2, axis=1))]
            else:
                # both are new -> move both
                new_i = points[i] - vec
                new_j = points[j] + vec
                points[i] = coords_valid[np.argmin(np.sum((coords_valid - new_i) ** 2, axis=1))]
                points[j] = coords_valid[np.argmin(np.sum((coords_valid - new_j) ** 2, axis=1))]

    return points

# ============================================================
# --- 4. Main CVT iteration
# ============================================================
def cvt_masked(mask_available, N_POINTS=60, existing_points=None, MIN_DIST=35, MIN_HOLE_DIST=15, ITERATIONS=50):
    ys_valid, xs_valid = np.where(mask_available == 0)
    coords_valid = np.column_stack((xs_valid, ys_valid))

    if existing_points is None:
        existing_points = np.zeros((0, 2), dtype=float)
    n_existing = len(existing_points)

    # Initialize only new points
    points_new = initialize_points(mask_available, N_POINTS - n_existing, existing_points, MIN_DIST, MIN_HOLE_DIST)
    points = np.vstack([existing_points, points_new])

    for it in range(ITERATIONS):
        tree_points = cKDTree(points)
        dist, idxs = tree_points.query(coords_valid)

        new_points = points.copy()
        # Only move the new points (keep existing ones fixed)
        for i in range(n_existing, N_POINTS):
            region_idx = np.where(idxs == i)[0]
            if len(region_idx) > 0:
                centroid = coords_valid[region_idx].mean(axis=0)
                nearest_idx = np.argmin(np.sum((coords_valid[region_idx] - centroid) ** 2, axis=1))
                new_points[i] = coords_valid[region_idx[nearest_idx]]
            else:
                new_points[i] = coords_valid[np.random.randint(len(coords_valid))]

        points = enforce_min_distances(new_points, coords_valid, MIN_DIST, n_existing)

    return points

# ============================================================
# --- 5. Evaluate uniformity (NDI metric)
# ============================================================
def normalized_dispersion_index_corrected(points, mask, alpha=0.4, beta=0.4, gamma=0.2):
    points = np.array(points, dtype=float)
    N = len(points)
    if N < 2:
        return 0.0
    ys, xs = np.where(mask == 0)
    available_poly = Polygon(np.column_stack((xs, ys))).convex_hull
    A_available = available_poly.area
    hull = MultiPoint(points).convex_hull
    A_hull = hull.area
    dist = np.sqrt(np.sum((points[None, :, :] - points[:, None, :]) ** 2, axis=-1))
    dists = dist[np.triu_indices(N, k=1)]
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    coords = np.column_stack((xs, ys))
    dmax = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    term1 = mean_d / dmax
    term2 = A_hull / A_available
    term3 = std_d / mean_d if mean_d > 0 else 0
    return alpha * term1 + beta * term2 - gamma * term3


def erode_available_region(mask_available, erosion_radius):
    """
    Shrink the available regions (0) by eroding them inward by `erosion_radius`.

    mask_available: np.ndarray, 0=available, 255=unavailable
    erosion_radius: number of pixels to shrink available regions
    """
    # Create a circular kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1)
    )

    # Dilate unavailable regions (255) to shrink available regions
    eroded = cv2.dilate(mask_available, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)

    return eroded

def combine_patches_with_sampled(patches, new_points, patch_size):
    """
    Create a combined patch list:
    - Existing small patches remain unchanged
    - New sampled points are converted into square patches
    """
    new_patches = []

    # --- Keep existing small patches ---
    for p in patches:
        if p['type'] == 'small':
            new_patches.append(p.copy())

    # --- Add new points as square patches ---
    half_size = patch_size // 2
    for i, pt in enumerate(new_points):
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        patch = {
            'x0': cx - half_size,
            'y0': cy - half_size,
            'w': patch_size,
            'h': patch_size,
            'cx': cx,
            'cy': cy,
            'area': patch_size**2,
            'type': 'sampled'
        }
        new_patches.append(patch)

    return new_patches

def save_dapi_patches(dapi_rgb, patches, output_folder, rescale_factor=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counts = {}
    saved_patches = []
    h_img, w_img = dapi_rgb.shape[:2]
    output_dict= {}

    for p in patches:
        if p['type'] == 'giant':
            continue

        patch_type = p['type']
        counts.setdefault(patch_type, 0)

        x0 = int(round(p["x0"] * rescale_factor))
        y0 = int(round(p["y0"] * rescale_factor))
        w = int(round(p["w"] * rescale_factor))
        h = int(round(p["h"] * rescale_factor))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x0 + w)
        y1 = min(h_img, y0 + h)

        patch_img = dapi_rgb[y0:y1, x0:x1]

        filename = f"{patch_type}_patch_{counts[patch_type]:02d}_dapi.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))

        info = {
            "x0": x0, "y0": y0,
            "w": x1 - x0, "h": y1 - y0,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
            "type": patch_type,
            "id": counts[patch_type],
            "filename": filename,
            "img": patch_img
        }
        saved_patches.append(info)
        output_dict[f"{patch_type}_patch_{counts[patch_type]:02d}"] = {k: v for k, v in info.items() if k != "img"}
        counts[patch_type] += 1


    print(f"Saved patches in '{output_folder}':")
    for t, c in counts.items():
        print(f"  {t.capitalize()} patches: {c}")
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Unknown type")
    # Save to JSON
    with open(f"{output_folder}/dapi_patch_info.json", "w") as f:
        json.dump(output_dict, f, default=convert_ndarray, indent=4)
    return saved_patches

def save_he_patches(he_rgb, patches, h_mat, output_folder, rescale_factor=1.0, margin_ratio=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counts = {}
    he_patches = []
    output_dict = {}
    H = np.array(h_mat)
    h_img, w_img = he_rgb.shape[:2]

    for p in patches:
        if p['type'] == 'giant':
            continue

        patch_type = p['type']
        counts.setdefault(patch_type, 0)

        x0, y0, w, h = p["x0"], p["y0"], p["w"], p["h"]
        mw, mh = int(round(w * (1 + margin_ratio))), int(round(h * (1 + margin_ratio)))
        x0_centered, y0_centered = int(round(x0 - (mw - w) / 2)), int(round(y0 - (mh - h) / 2))
        x1, y1 = x0_centered + mw, y0_centered + mh

        corners = np.array([[x0_centered, y0_centered],
                            [x1, y0_centered],
                            [x1, y1],
                            [x0_centered, y1]], dtype=float)

        transformed = np.dot(H[:, :2], corners.T).T + H[:, 2]
        min_x = max(0, int(round(transformed[:, 0].min() * rescale_factor)))
        max_x = min(w_img, int(round(transformed[:, 0].max() * rescale_factor)))
        min_y = max(0, int(round(transformed[:, 1].min() * rescale_factor)))
        max_y = min(h_img, int(round(transformed[:, 1].max() * rescale_factor)))

        patch_img = he_rgb[min_y:max_y, min_x:max_x]

        filename = f"{patch_type}_patch_{counts[patch_type]:02d}_he.png"
        cv2.imwrite(os.path.join(output_folder, filename), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))

        info = {
            "x0": min_x, "y0": min_y,
            "w": max_x - min_x, "h": max_y - min_y,
            "cx": (min_x + max_x) / 2, "cy": (min_y + max_y) / 2,
            "type": patch_type,
            "id": counts[patch_type],
            "filename": filename,
            "img": patch_img
        }
        he_patches.append(info)
        output_dict[f"{patch_type}_patch_{counts[patch_type]:02d}"] = {k: v for k, v in info.items() if k != "img"}
        counts[patch_type] += 1

    print(f"Saved H&E patches in '{output_folder}':")
    for t, c in counts.items():
        print(f"  {t.capitalize()} patches: {c}")
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Unknown type")
    # Save to JSON
    with open(f"{output_folder}/he_patch_info.json", "w") as f:
        json.dump(output_dict, f, default=convert_ndarray, indent=4)
    return he_patches


def save_patch_overlay_cv2(image, final_patches, mask,
                                 show_giant=True, show_small=True,
                                 save_path=None, alpha=0.4):
    """
    Draw blobs and patch centroids correctly with OpenCV.
    Returns RGB overlay.
    """
    # Ensure BGR
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        overlay = cv2.cvtColor(np.clip(image,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    # Map label → blob info
    blob_info = {}
    for i in range(1, num_labels):
        blob = (labels==i)
        M = cv2.moments(blob.astype(np.uint8))
        if M['m00']==0: continue
        cx = int(round(M['m10']/M['m00']))
        cy = int(round(M['m01']/M['m00']))
        blob_info[i] = {'mask': blob, 'cx': cx, 'cy': cy}

    # Prepare overlay layer
    overlay_layer = overlay.copy()
    small_count = giant_count = 0
    green = (0,255,0)

    for p in final_patches:
        ctype = p['type']
        if (ctype=='small' and not show_small) or (ctype=='giant' and not show_giant):
            continue

        px, py = int(round(p['cx'])), int(round(p['cy']))

        # Match patch to blob
        for info in blob_info.values():
            if info['cx']==px and info['cy']==py:
                # Choose color
                color = (0,255,255) if ctype=='small' else (0,0,255)  # BGR: yellow or red
                idx = np.where(info['mask'])
                overlay_layer[idx[0], idx[1]] = cv2.addWeighted(overlay_layer[idx[0], idx[1]],
                                                               1-alpha,
                                                               np.full_like(overlay_layer[idx[0], idx[1]], color),
                                                               alpha, 0)
                break
        # Centroid
        cv2.circle(overlay_layer, (px, py), 5, green, -1)

        # Rectangle only for small
        if ctype=='small':
            x0, y0, w, h = map(int, (p['x0'], p['y0'], p['w'], p['h']))
            cv2.rectangle(overlay_layer, (x0,y0), (x0+w, y0+h), (0,255,255), 2)  # yellow rectangle
            text = f"S{small_count}"
            small_count += 1
        else:
            text = f"G{giant_count}"
            giant_count += 1

        cv2.putText(overlay_layer, text, (px-10, py+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, overlay_layer)

    return cv2.cvtColor(overlay_layer, cv2.COLOR_BGR2RGB)

def draw_dapi_patches_cv2(dapi_rgb, all_patches,
                           show_small_only=True,
                           show_sampled_only=True,
                           save_path=None,
                           display=False):
    """
    Draw DAPI patches on the image with labels at patch centroids using OpenCV.

    Parameters:
    - dapi_rgb: np.ndarray (HxWx3), DAPI RGB image
    - all_patches: list of dicts with keys ['x0','y0','w','h','type']
    - show_small_only: bool, if True, draw only 'small' patches
    - show_sampled_only: bool, if True, draw only 'sampled' patches
    - save_path: str or None, if given, save the overlay
    - display: bool, if True, show window via cv2.imshow
    Returns:
    - overlay_rgb: np.ndarray, HxWx3 RGB image with overlays
    """

    # Make a copy and convert to BGR for OpenCV
    overlay_bgr = cv2.cvtColor(np.clip(dapi_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Define colors (BGR)
    color_dict = {
        'small': (0, 255, 255),   # yellow
        'sampled': (0, 0, 255),   # red
        'other': (0, 255, 0),      # green
    }

    for i, p in enumerate(all_patches):
        patch_type = p.get('type', 'other')

        # Skip based on flags
        if patch_type == 'small' and not show_small_only:
            continue
        if patch_type == 'sampled' and not show_sampled_only:
            continue

        color = color_dict.get(patch_type, color_dict['other'])
        x0, y0, w, h = map(int, (p['x0'], p['y0'], p['w'], p['h']))

        # Draw rectangle
        cv2.rectangle(overlay_bgr, (x0, y0), (x0 + w, y0 + h), color, 2)

        # Draw centroid label
        cx, cy = x0 + w // 2, y0 + h // 2
        cv2.putText(overlay_bgr, f"{i:02d}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Convert back to RGB for Tkinter or PIL
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Save if requested
    if save_path is not None:
        cv2.imwrite(save_path, overlay_bgr)

    # Display if requested
    if display:
        cv2.imshow("DAPI Patches", overlay_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return overlay_rgb

 
if __name__ == "__main__":
    timer = StepTimer()
    output_folder = "v9.0_patches"

    # ------------------------------
    # 0. Read configuration from JSON
    # ------------------------------
    HE_LEVEL = None
    DAPI_LEVEL = None
    json_path = "images_info.json"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            HE_LEVEL = data.get("HE_level", None)
            DAPI_LEVEL = data.get("DAPI_level", None)
            HE_PATH = data.get("HE_path", None)
            DAPI_PATH = data.get("DAPI_path", None)
            print(f"Read levels from JSON: HE_PATH={HE_PATH}, HE_level={HE_LEVEL}, DAPI_PATH={DAPI_PATH}, DAPI_level={DAPI_LEVEL}")
    except FileNotFoundError:
        print(f"Warning: {json_path} not found.")
    timer.mark("Read config & paths")

    # ------------------------------
    # 1. Load LUT and DAPI image
    # ------------------------------
    lut_path = "/Users/sicongy/Documents/GitHub/rotation_1/LUT/glasbey_inverted.lut"
    lut = np.fromfile(lut_path, dtype=np.uint8).reshape(256, 3)

    dapi_img, dapi_level = read_image(DAPI_PATH, keep_16bit=True, level=4)
    dapi_rgb = dapi_to_lut_rgb(dapi_img, lut, threshold=300)
    cv2.imwrite('3_dapi_rgb.png', dapi_rgb)
    timer.mark("Load DAPI + LUT + LUT RGB")

    # ------------------------------
    # 2. Compute DAPI mask
    # ------------------------------
    dapi_mask = create_blob_mask_from_luted_dapi(dapi_rgb)
    cv2.imwrite('3_dapi_mask_final.png', dapi_mask)
    timer.mark("Create DAPI blob mask")

    # ------------------------------
    # 3. Extract blobs and classify patches
    # ------------------------------
    min_area, max_area = 500, 10000
    margin_ratio = 0.5
    mask = dapi_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    patches = []
    square_size = 128
    giant_tiles_dict = {}

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        if area > max_area:
            # Giant blob: tile internally and record single patch
            patch_type = "giant"
            blob_mask = (labels == i).astype(np.uint8)
            ys, xs = np.where(blob_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            tiles = []
            for y0 in range(y_min, y_max, square_size):
                for x0 in range(x_min, x_max, square_size):
                    y1, x1 = min(y0 + square_size, mask.shape[0]), min(x0 + square_size, mask.shape[1])
                    if np.any(blob_mask[y0:y1, x0:x1] > 0):
                        tiles.append({'x0': x0, 'y0': y0, 'w': x1-x0, 'h': y1-y0})
            giant_tiles_dict[i] = tiles

            patches.append({"x0": x_min, "y0": y_min, "w": x_max-x_min, "h": y_max-y_min,
                            "cx": cx, "cy": cy, "area": area, "type": patch_type})
        else:
            # Small blob: add margin
            patch_type = "small"
            mw, mh = int(w * (1 + margin_ratio)), int(h * (1 + margin_ratio))
            x0, y0 = int(x - (mw - w)/2), int(y - (mh - h)/2)
            x0, y0 = max(0, x0), max(0, y0)
            mw, mh = min(mw, mask.shape[1] - x0), min(mh, mask.shape[0] - y0)
            patches.append({"x0": x0, "y0": y0, "w": mw, "h": mh,
                            "cx": cx, "cy": cy, "area": area, "type": patch_type})

    overlay_small = save_patch_overlay_cv2(dapi_rgb, patches, mask,
                                           show_giant=False, show_small=True, save_path='3_blobs_small.png')
    overlay_giant = save_patch_overlay_cv2(dapi_rgb, patches, mask,
                                           show_giant=True, show_small=False, save_path='3_blobs_giant.png')
    overlay_both = save_patch_overlay_cv2(dapi_rgb, patches, mask,
                                          show_giant=True, show_small=True, save_path='3_blobs_both.png')
    timer.mark("Extract blobs & classify patches")

    # ------------------------------
    # 4. Create mask of available regions for sampling
    # ------------------------------
    mask_available = create_available_mask(mask, giant_tiles_dict)
    cv2.imwrite('3_dapi_mask_available.png', mask_available)
    MIN_DIST = 100
    N_total = 60
    mask_available_eroded = erode_available_region(mask_available, erosion_radius=int(MIN_DIST/np.sqrt(2)))
    cv2.imwrite('3_dapi_mask_available_eroded.png', mask_available_eroded)
    timer.mark("Create & erode available mask")

    # ------------------------------
    # 5. Evenly sample new points with masked CVT
    # ------------------------------
    small_centroids = np.array([[int(round(p['cx'])), int(round(p['cy']))]
                                for p in patches if p['type'] == 'small'])

    all_points = cvt_masked(mask_available_eroded, N_POINTS=N_total, existing_points=small_centroids,
                            MIN_DIST=MIN_DIST, MIN_HOLE_DIST=15, ITERATIONS=50)
    ndi_score = normalized_dispersion_index_corrected(all_points, mask_available)
    print("NDI Score:", ndi_score)
    new_points = all_points[len(small_centroids):]

    # ------------------------------
    # 6. Combine patches with sampled points
    # ------------------------------
    patch_size = int(np.ceil(MIN_DIST/np.sqrt(2)))
    all_patches = combine_patches_with_sampled(patches, new_points, patch_size)
    timer.mark("Masked CVT sampling")

    # ------------------------------
    # 7. Save DAPI patches
    # ------------------------------
    # 1. 读 original DAPI
    dapi_img2, _ = read_image(DAPI_PATH, keep_16bit=True, level=1)
    dapi_rgb2 = dapi_to_lut_rgb(dapi_img2, lut, threshold=300)

    # 2. 用 GUI affine 摆正整张 DAPI
    with open("images_info.json", "r") as f:
        cfg = json.load(f)
    DAPI_GUI_AFFINE = np.array(cfg["DAPI_gui_affine"], dtype=np.float32)

    dapi_rgb2_gui = apply_gui_affine_to_image(dapi_rgb2, DAPI_GUI_AFFINE)

    # 3. 用“摆正后的 DAPI”去截 tile
    dapi_patches = save_dapi_patches(
        dapi_rgb2_gui,
        all_patches,
        output_folder,
        rescale_factor=8.0
    )
    draw_dapi_patches_cv2(dapi_rgb, all_patches, show_small_only=True, show_sampled_only=False, save_path='3_dapi_patches_overlay_small.png')
    draw_dapi_patches_cv2(dapi_rgb, all_patches, show_small_only=False, show_sampled_only=True, save_path='3_dapi_patches_overlay_sampled.png')
    draw_dapi_patches_cv2(dapi_rgb, all_patches, show_small_only=True, show_sampled_only=True, save_path='3_dapi_patches_overlay_both.png')
    timer.mark("Save DAPI patches")

    # ------------------------------
    # 8. Save HE patches using transformation
    # ------------------------------
    with open("clicked_blob_boxes.json", "r") as f:
        data = json.load(f)
    h_mat = data["H_mat"]
    he_img2, _ = read_image(HE_PATH, keep_16bit=True, level=1)
    he_patches = save_he_patches(he_img2, all_patches, h_mat, output_folder, rescale_factor=8.0, margin_ratio=0.1)
    timer.mark("Save HE patches using transformation")

    # ------------------------------
    # 8. Show patch gallery
    # ------------------------------
    show_patch_gallery_in_memory(dapi_patches, he_patches)



