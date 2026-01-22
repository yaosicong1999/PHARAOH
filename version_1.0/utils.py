# utils.py
import numpy as np
import tifffile
import cv2
from PIL import Image

# def read_image(path, keep_16bit=True):
#     filename = path.lower()
#     if filename.endswith(".ome.tif"):
#         img = tifffile.imread(path, level=4)
#     elif filename.endswith((".tif", ".tiff")):
#         img = tifffile.imread(path)
#     else:
#         img = np.array(Image.open(path))
#     if not keep_16bit and img.ndim == 3 and img.shape[2] == 3:
#         img = img[::8, ::8]
#     if img.ndim == 2:
#         img = np.stack([img] * 3, axis=-1)
#     if not keep_16bit and img.dtype != np.uint8:
#         img_min, img_max = img.min(), img.max()
#         if img_max > img_min:
#             img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
#         else:
#             img = np.zeros_like(img, dtype=np.uint8)
#     return img

# def read_image(path, keep_16bit=True, series=0, page=0, level=None, min_size=1000, max_size=2000):
#     """
#     Reads an image from path. Supports OME-TIFF with subIFDs, regular TIFF, and standard image formats.
#     Auto-selects a pyramid level based on min_size and max_size if level is None.
#
#     Returns:
#         img: np.ndarray
#         selected_level: int or None (only for OME-TIFF)
#     """
#     filename = path.lower()
#     selected_level = None
#
#     if filename.endswith(".ome.tif"):
#         import tifffile
#         from PIL import Image
#         import numpy as np
#
#         with tifffile.TiffFile(path) as tif:
#             img_series = tif.series[series]
#             img_page = img_series.pages[page]
#             print(f"Selecting series {series} page {page}")
#             # Combine full-res + SubIFDs
#             pages = [img_page] + list(img_page.pages)
#             print("Full resolution:", pages[0].shape)
#             for j, p in enumerate(pages):
#                 print(f"  Level {j}: shape={p.shape}")
#             # Auto-select level if None
#             if level is None:
#                 selected_level = 0
#                 for j, p in enumerate(pages):
#                     h, w = p.shape[:2]
#                     if min_size <= min(h, w) <= max_size:
#                         selected_level = j
#                         break
#                 level = selected_level
#                 print(f"Auto-selected level {level} with shape {pages[level].shape}")
#             else:
#                 selected_level = level
#                 print(f"Selected input level {level} with shape {pages[level].shape}")
#
#             img = pages[level].asarray()
#
#     elif filename.endswith((".tif", ".tiff")):
#         import tifffile
#         img = tifffile.imread(path)
#     else:
#         from PIL import Image
#         import numpy as np
#         img = np.array(Image.open(path))
#
#     # Convert grayscale to RGB
#     if img.ndim == 2:
#         img = np.stack([img] * 3, axis=-1)
#
#     # Optional downscale for non-16bit images
#     if not keep_16bit and img.dtype != np.uint8:
#         img_min, img_max = img.min(), img.max()
#         if img_max > img_min:
#             img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
#         else:
#             img = np.zeros_like(img, dtype=np.uint8)
#
#     return img, selected_level

def read_image(path, keep_16bit=True, series=0, page=0, level=None,
               min_size=1000, max_size=2000, force_rgb=True):
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
    if force_rgb and img.ndim == 2:
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