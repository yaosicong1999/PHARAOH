# utils.py
import numpy as np
import tifffile
import cv2
from PIL import Image
from skimage import color, exposure
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
from skimage.morphology import local_maxima
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.transform import rescale
from PIL import Image


def read_image(path, keep_16bit=True, series=0, page=0, level=None,
               min_size=1000, max_size=2000):
    """
    Reads an image from path. Supports OME-TIFF with subIFDs, regular TIFF, and standard image formats.
    Auto-selects a pyramid level (or downsample factor) so that min(image_dim) is between min_size and max_size.

    Returns:
        img: np.ndarray
        selected_level: int or None (OME-TIFF level or pseudo-level for non-pyramidal images)
    """
    import numpy as np
    import cv2
    import tifffile
    from PIL import Image

    filename = path.lower()
    selected_level = None

    if filename.endswith(".ome.tif"):
        with tifffile.TiffFile(path) as tif:
            img_series = tif.series[series]
            img_page = img_series.pages[page]
            print(f"Selecting series {series} page {page}")
            pages = [img_page] + list(img_page.pages)
            print("Full resolution:", pages[0].shape)
            for j, p in enumerate(pages):
                print(f"  Level {j}: shape={p.shape}")

            # Auto-select level
            if level is None:
                selected_level = 0
                for j, p in enumerate(pages):
                    h, w = p.shape[:2]
                    if min_size <= min(h, w) <= max_size:
                        selected_level = j
                        break
                level = selected_level
                print(f"Auto-selected level {level} with shape {pages[level].shape}")
            else:
                selected_level = level
                print(f"Selected input level {level} with shape {pages[level].shape}")

            img = pages[level].asarray()

    else:
        # Regular TIFF or standard image
        img = tifffile.imread(path) if filename.endswith((".tif", ".tiff")) else np.array(Image.open(path))
        h, w = img.shape[:2]
        print(f"Original image shape: {img.shape}")

        # Compute pseudo-level (number of 2x downsamples)
        pseudo_level = 0
        target_h, target_w = h, w
        while min(target_h, target_w) > max_size:
            target_h //= 2
            target_w //= 2
            pseudo_level += 1

        # Only downsample if needed
        if pseudo_level > 0:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            print(f"Downsampled x(2^{pseudo_level}) → shape {img.shape}")
        else:
            print("No downsampling needed (pseudo-level 0)")

        selected_level = pseudo_level

    # Convert grayscale to RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Optional 8-bit conversion
    if not keep_16bit and img.dtype != np.uint8:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    return img, selected_level

def read_ome_tiff_subifd(path, series=0, page=0, min_size=1000, max_size=2000):
    """
    Read an OME-TIFF subIFD and automatically select the level
    where min(height, width) is within [min_size, max_size].
    """
    with tifffile.TiffFile(path) as tif:
        img_series = tif.series[series]
        img_page = img_series.pages[page]
        pages = [img_page] + list(img_page.pages)
        n_pages = len(pages)
        print(f"Total levels available: {n_pages}")

        # iterate levels to find one where min(height, width) is in desired range
        selected_level = 0
        for i, p in enumerate(pages):
            h, w = p.shape[:2]
            small_dim = min(h, w)
            print(f"Level {i}: shape={p.shape}, min_dim={small_dim}")
            if min_size <= small_dim <= max_size:
                selected_level = i
                break
            elif small_dim < min_size:
                # stop if below minimum
                selected_level = max(0, i-1)
                break

        print(f"Selected level {selected_level} with shape {pages[selected_level].shape}")
        return pages[selected_level].asarray()

def extract_hematoxylin_channel(img):
    img_float = img.astype(np.float32) + 1.0
    od = -np.log(img_float / 255.0)
    stain_matrix = np.array([[0.65, 0.07],
                             [0.70, 0.99],
                             [0.29, 0.11]])
    stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
    inv_matrix = np.linalg.pinv(stain_matrix)
    conc = np.dot(od.reshape((-1, 3)), inv_matrix.T)
    H_conc = conc[:, 0].reshape(img.shape[:2])
    H_conc = (H_conc - H_conc.min()) / (H_conc.max() - H_conc.min()) * 255.0
    return H_conc.astype(np.uint8)

def enhance_hematoxylin_channel(H_channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    H_clahe = clahe.apply(H_channel)
    p2, p98 = np.percentile(H_clahe, (2, 98))
    if p98 == p2:
        H_rescale = H_clahe
    else:
        H_rescale = np.clip((H_clahe - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    H_final = cv2.filter2D(H_rescale, -1, kernel)
    return H_final

def dapi_to_lut_rgb(dapi_img_local, lut_table, threshold=300):
    dapi = dapi_img_local[..., 0] if dapi_img_local.ndim == 3 else dapi_img_local
    dapi_clipped = np.clip(dapi, threshold, None)
    d_min, d_max = dapi_clipped.min(), dapi_clipped.max()
    if d_max > d_min:
        scaled = ((dapi_clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(dapi_clipped, dtype=np.uint8)
    rgb = lut_table[scaled]
    return rgb


def upsample_tile(tile_rgb, scale=2):
    up_tile = rescale(
        tile_rgb,
        scale=(scale, scale),
        anti_aliasing=True,
        channel_axis=2,
        preserve_range=True
    ).astype(np.uint8)
    return up_tile

# --- Step 1: Stain separation (Macenko normalization) ---
def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    h, w, c = img.shape
    img_flat = img.reshape((-1, 3))
    OD = -np.log((img_flat.astype(np.float32) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    HE = np.array((vMax[:, 0], vMin[:, 0])).T if vMin[0] <= vMax[0] else np.array((vMin[:, 0], vMax[:, 0])).T
    Y = OD.T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = maxC / maxCRef
    C2 = C / tmp[:, np.newaxis]
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    return Inorm, H, E


# --- Step 2: Morphological smoothing ---
def morphological_smooth(img_gray, n=2):
    se = disk(n)
    img_open = opening(img_gray, se)
    img_close = closing(img_open, se)
    return img_close


# --- Step 3: Thresholding super-dark nuclei ---
def threshold_super_dark(H_gray, intensity_threshold=0.6):
    # H_gray scaled 0-1
    mask_dark = H_gray > intensity_threshold
    mask_dark = remove_small_objects(mask_dark, min_size=30)
    mask_dark = remove_small_holes(mask_dark, area_threshold=60)
    mask_dark = closing(mask_dark, disk(1))
    return mask_dark


# --- Step 4: Separate touching nuclei & label ---
def separate_and_label(binary_mask, min_label_size=30):
    distance = ndi.distance_transform_edt(binary_mask)
    # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary_mask)
    local_maxi = local_maxima(distance) & (binary_mask > 0)
    markers = ndi.label(local_maxi)[0]
    labeled = watershed(-distance, markers, mask=binary_mask)
    # remove small labels
    for label in np.unique(labeled)[1:]:
        if np.sum(labeled == label) < min_label_size:
            labeled[labeled == label] = 0
    labeled, _ = ndi.label(labeled > 0)
    return labeled


# --- Overlay contours ---
def overlay_contours(rgb_tile, labeled_mask):
    import cv2
    overlay = rgb_tile.copy()
    for label in np.unique(labeled_mask)[1:]:
        mask = (labeled_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
    return overlay

# --- Full pipeline ---
def segment_super_dark_nuclei_full(rgb_tile, upsample_scale=2, n_smooth=2, intensity_threshold=0.7):
    tile_up = upsample_tile(rgb_tile, scale=upsample_scale)
    _, H, E = normalizeStaining(tile_up)
    H_gray = color.rgb2gray(H)
    H_smooth = morphological_smooth(H_gray, n=n_smooth)
    mask_dark = threshold_super_dark(H_smooth, intensity_threshold=intensity_threshold)
    labeled_mask = separate_and_label(mask_dark, min_label_size=30)
    overlay = overlay_contours(tile_up, labeled_mask)
    return labeled_mask, mask_dark