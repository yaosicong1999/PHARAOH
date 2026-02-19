# utils.py
import numpy as np
import tifffile
import cv2
import zarr
from PIL import Image
import math
from skimage import color, exposure
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
from skimage.morphology import local_maxima
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.transform import rescale
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None  # disable the check

def read_image(
    path,
    keep_16bit=True,
    series=0,
    page=0,
    level=None,
    min_size=1000,
    max_size=2000,
    force_rgb=True,
    channel="auto",   # "auto" | "he" | "dapi" | int | None
):
    """
    Unified reader for:
      - OME-TIFF (with subIFDs or series.levels or no pyramid)
      - regular TIFF
      - jpg/png/other images

    Strategy for OME:
      1) Try page.pages (subIFD pyramid) first (tif.series[series].pages[page])
      2) Else try series.levels (OME pyramid API)
      3) Else treat as single image and do pseudo downsample (2x steps) like non-OME

    Channel policy (only matters for multichannel arrays):
      - channel="auto":
          * CYX: if C>=4 -> pick C0 as DAPI (print)
                 if C==3 -> treat as HE RGB (print)
          * YXC: keep as RGB if last dim in (3,4); else if force_rgb -> pick channel0 to gray->rgb
      - channel="he":
          * require RGB-like (CYX with C==3, or YXC with last dim==3/4), else raise
      - channel="dapi":
          * CYX: pick C0 (print)
          * YXC: pick channel0 (print)
      - channel=int:
          * pick that channel (CYX or YXC), print
      - channel=None:
          * keep all channels (convert CYX->YXC), no picking

    Returns:
      img: np.ndarray (H,W,C) if force_rgb or RGB image; else possibly (H,W)
      selected_level: for OME: chosen pyramid index (or pseudo_level if no pyramid)
                      for non-OME: pseudo_level
    """
    import numpy as np
    import cv2
    import tifffile
    from PIL import Image

    filename = str(path).lower()
    selected_level = None

    def _rep_to_rgb(arr2d):
        return np.repeat(arr2d[..., None], 3, axis=-1)

    def _normalize_to_uint8_if_needed(img):
        if keep_16bit or img.dtype == np.uint8:
            return img
        imgf = img.astype(np.float32)
        mn, mx = float(imgf.min()), float(imgf.max())
        if mx > mn:
            return ((imgf - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def _pick_level_by_size(get_hw_fn, n_levels):
        # choose first level where min(h,w) in [min_size, max_size], else fallback to last level
        if level is not None:
            lv = int(level)
            if lv < 0 or lv >= n_levels:
                raise ValueError(f"level out of range: {lv} for n_levels={n_levels}")
            return lv

        chosen = 0
        for j in range(n_levels):
            h, w = get_hw_fn(j)
            if min_size <= min(h, w) <= max_size:
                chosen = j
                break
        return chosen

    def _pseudo_downsample(img0):
        # downsample by powers of 2 until min(h,w) <= max_size
        h0, w0 = img0.shape[:2]
        if min(h0, w0) <= max_size:
            return img0, 0
        pseudo = 0
        th, tw = h0, w0
        while min(th, tw) > max_size:
            th //= 2
            tw //= 2
            pseudo += 1
        th = max(1, th)
        tw = max(1, tw)
        img_ds = cv2.resize(img0, (tw, th), interpolation=cv2.INTER_AREA)
        print(f"[INFO] pseudo downsample x(2^{pseudo}) -> shape={img_ds.shape}")
        return img_ds, pseudo

    def _apply_channel_policy(arr, axes):
        """
        arr: np.ndarray from tifffile (may be 2D or 3D)
        axes: e.g. "CYX" or "YXC" or "YX"
        returns image in either (H,W,C) or (H,W) depending on force_rgb + channel rules
        """
        axes = (axes or "").upper()

        # 2D
        if arr.ndim == 2:
            return _rep_to_rgb(arr) if force_rgb else arr

        # 3D
        if arr.ndim != 3:
            # weird dims: return as-is
            return arr

        # ---------- CYX ----------
        if axes == "CYX":
            C = arr.shape[0]

            # explicit channel index
            if isinstance(channel, int):
                ch = int(channel)
                if ch < 0 or ch >= C:
                    raise ValueError(f"[read_image] CYX has C={C}, cannot pick channel={ch}")
                out = arr[ch, ...]
                print(f"[INFO] CYX: picked channel={ch} (explicit)")
                return _rep_to_rgb(out) if force_rgb else out

            # keep all
            if channel is None:
                out = np.moveaxis(arr, 0, -1)
                return out

            # channel="dapi"
            if str(channel).lower() == "dapi":
                if C < 1:
                    raise ValueError("[read_image] CYX but C<1, cannot pick DAPI")
                out = arr[0, ...]
                print("[INFO] CYX mode=dapi: picked C0 as DAPI")
                return _rep_to_rgb(out) if force_rgb else out

            # channel="he"
            if str(channel).lower() == "he":
                if C != 3:
                    raise ValueError(f"[read_image] mode=he requires CYX C==3, got C={C}")
                out = np.moveaxis(arr, 0, -1)  # (H,W,3)
                print("[INFO] CYX mode=he: treated as HE RGB (C==3)")
                return out

            # channel="auto"
            if str(channel).lower() == "auto":
                if C >= 4:
                    out = arr[0, ...]
                    print(f"[INFO] CYX mode=auto: C={C} >=4, picked C0 as DAPI")
                    return _rep_to_rgb(out) if force_rgb else out
                if C == 3:
                    out = np.moveaxis(arr, 0, -1)
                    print("[INFO] CYX mode=auto: C==3, treated as HE RGB")
                    return out
                # other C: default moveaxis keep all
                out = np.moveaxis(arr, 0, -1)
                print(f"[INFO] CYX mode=auto: C={C}, kept all channels as YXC")
                return out

            # fallback
            out = np.moveaxis(arr, 0, -1)
            return out

        # ---------- YXC ----------
        if axes == "YXC":
            C = arr.shape[-1]

            if isinstance(channel, int):
                ch = int(channel)
                if ch < 0 or ch >= C:
                    raise ValueError(f"[read_image] YXC has C={C}, cannot pick channel={ch}")
                out = arr[..., ch]
                print(f"[INFO] YXC: picked channel={ch} (explicit)")
                return _rep_to_rgb(out) if force_rgb else out

            if channel is None:
                return arr

            if str(channel).lower() == "dapi":
                out = arr[..., 0]
                print("[INFO] YXC mode=dapi: picked channel 0")
                return _rep_to_rgb(out) if force_rgb else out

            if str(channel).lower() == "he":
                if C not in (3, 4):
                    raise ValueError(f"[read_image] mode=he requires YXC last dim 3/4, got {C}")
                print("[INFO] YXC mode=he: kept RGB(A)")
                return arr

            if str(channel).lower() == "auto":
                if C in (3, 4):
                    print("[INFO] YXC mode=auto: kept RGB(A)")
                    return arr
                out = arr[..., 0]
                print(f"[INFO] YXC mode=auto: C={C}, picked channel0 as grayscale")
                return _rep_to_rgb(out) if force_rgb else out

            return arr

        # ---------- unknown axes: guess ----------
        # try to guess channel axis
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            # treat as CYX
            return _apply_channel_policy(arr, "CYX")
        if arr.shape[-1] in (3, 4):
            return _apply_channel_policy(arr, "YXC")

        # fallback: pick first channel if force_rgb
        out = arr[..., 0]
        return _rep_to_rgb(out) if force_rgb else out

    # ============================================================
    # OME-TIFF
    # ============================================================
    if filename.endswith((".ome.tif", ".ome.tiff")):
        with tifffile.TiffFile(path) as tif:
            s = tif.series[series]

            # ---------- 1) Try page.pages (subIFDs pyramid) ----------
            pages = None
            try:
                # NOTE: s.pages can be multi-page; you asked to start from "page"
                base_page = s.pages[page]
                # base_page.pages is the subIFD pyramid list (often)
                sub_pages = list(getattr(base_page, "pages", []))
                if sub_pages:
                    pages = [base_page] + sub_pages
            except Exception as e:
                pages = None

            if pages is not None and len(pages) > 1:
                print("=== LEVELS (from page.pages) ===")
                for j, p in enumerate(pages[:20]):
                    print(f"  level[{j}] shape={getattr(p, 'shape', None)} dtype={getattr(p, 'dtype', None)}")
                if len(pages) > 20:
                    print(f"  ... total levels: {len(pages)}")

                def get_hw(j):
                    sh = pages[j].shape
                    # page.asarray() usually returns YX or YXC already, but shape here is YX
                    h, w = int(sh[0]), int(sh[1])
                    return h, w

                chosen = _pick_level_by_size(get_hw, len(pages))
                selected_level = int(chosen)
                print(f"[INFO] Picked level={selected_level} from page.pages")

                arr = pages[selected_level].asarray()
                # ---------- HE special-case: page.pages sometimes is grayscale replicated ----------
                # If user asked for HE and this looks like replicated RGB, skip pages and try series.levels.
                if (channel is not None) and (str(channel).lower() == "he"):
                    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        if (arr.ndim == 3) and (arr.shape[-1] in (3, 4)):
                            rgb = arr[..., :3]
                            if np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]):
                                print(
                                    "[WARN] page.pages returned replicated grayscale RGB for HE; fallback to series.levels",
                                    flush=True)
                                pages = None
                    elif arr.ndim == 2:
                        print(
                            "[WARN] page.pages returned replicated grayscale RGB for HE; fallback to series.levels",
                            flush=True)
                        pages = None

                if pages is not None:
                    # axes is not always available on TiffPage -> infer
                    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                        axes = "CYX"
                    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        axes = "YXC"
                    else:
                        axes = "YX"
                    img = _apply_channel_policy(arr, axes)
                    img = _normalize_to_uint8_if_needed(img)
                    return img, selected_level

            # ---------- 2) Fallback: series.levels ----------
            levels = list(getattr(s, "levels", [])) or []
            if levels:
                print("=== LEVELS (from series.levels) ===")
                for j, lv in enumerate(levels[:20]):
                    print(f"  level[{j}] axes={getattr(lv, 'axes', None)} shape={getattr(lv, 'shape', None)} dtype={getattr(lv, 'dtype', None)}")
                if len(levels) > 20:
                    print(f"  ... total levels: {len(levels)}")

                def get_hw(j):
                    lv = levels[j]
                    axes = (getattr(lv, "axes", "") or "").upper()
                    sh = lv.shape
                    if axes == "CYX":
                        return int(sh[1]), int(sh[2])
                    if axes == "YXC":
                        return int(sh[0]), int(sh[1])
                    if axes == "YX":
                        return int(sh[0]), int(sh[1])
                    return int(sh[-2]), int(sh[-1])

                chosen = _pick_level_by_size(get_hw, len(levels))
                selected_level = int(chosen)
                print(f"[INFO] Picked level={selected_level} from series.levels")

                arr = levels[selected_level].asarray()
                img = _apply_channel_policy(arr, getattr(levels[selected_level], "axes", None))
                img = _normalize_to_uint8_if_needed(img)
                return img, selected_level

            # ---------- 3) No pyramid: read full-res then pseudo downsample ----------
            print("[INFO] OME: no page.pages pyramid and no series.levels pyramid -> read full res then downsample")
            arr = s.asarray()
            img0 = _apply_channel_policy(arr, getattr(s, "axes", None))
            img0 = _normalize_to_uint8_if_needed(img0)

            img_ds, pseudo = _pseudo_downsample(img0)
            selected_level = int(pseudo)
            return img_ds, selected_level

    # ============================================================
    # Non-OME: regular TIFF / JPG / PNG
    # ============================================================
    def _infer_axes_non_ome(arr):
        """
        Infer axes for non-OME reads (tifffile.imread / PIL).
        Returns one of: "YX", "YXC", "CYX"
        """
        if arr.ndim == 2:
            return "YX"

        if arr.ndim == 3:
            # If last dim looks like color channels -> YXC
            if arr.shape[-1] in (3, 4):
                return "YXC"

            # Otherwise treat as stack-of-planes: IYX/CYX style
            # (e.g., IYX with I=24, or CYX with C=3/4 but stored first)
            return "CYX"

        # weird dims, return empty -> caller can handle
        return ""


    # read
    if filename.endswith((".tif", ".tiff")):
        img0 = tifffile.imread(path)
    else:
        img0 = np.array(Image.open(path))

    axes = _infer_axes_non_ome(img0)
    img0 = _apply_channel_policy(img0, axes)

    # ensure channel layout similar to others
    # if grayscale
    if img0.ndim == 2:
        img0 = _rep_to_rgb(img0) if force_rgb else img0

    # auto downsample (pseudo levels)
    if level is not None:
        used = int(level)
        if used < 0:
            raise ValueError(f"level must be >=0, got {used}")
        pseudo = used
        scale = 2 ** pseudo
        h0, w0 = img0.shape[:2]
        th = max(1, h0 // scale)
        tw = max(1, w0 // scale)
        if pseudo > 0:
            img0 = cv2.resize(img0, (tw, th), interpolation=cv2.INTER_AREA)
            print(f"[INFO] Non-OME: specified pseudo_level={pseudo} -> shape={img0.shape}")
        selected_level = pseudo
    else:
        img0, pseudo = _pseudo_downsample(img0)
        selected_level = int(pseudo)

    # optional uint8
    img0 = _normalize_to_uint8_if_needed(img0)
    return img0, selected_level

def read_image0(path, keep_16bit=True, series=0, page=0, level=None,
               min_size=1000, max_size=2000):
    """
    Reads an image from path. Supports OME-TIFF with subIFDs, regular TIFF, and standard image formats.
    Auto-selects a pyramid level (or downsample factor) so that min(image_dim) is between min_size and max_size.

    Returns:
        img: np.ndarray
        selected_level: int or None (OME-TIFF level or pseudo-level for non-pyramidal images)
    """

    filename = path.lower()
    selected_level = None

    if filename.endswith((".ome.tif", ".ome.tiff")):
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
        # decide level
        if level is None:
            # smallest k s.t. min(h,w)/2^k <= max_size  -> k = ceil(log2(min/max))
            m = min(h, w)
            if m <= max_size:
                used_level = 0
            else:
                used_level = int(math.ceil(math.log2(m / float(max_size))))
        else:
            used_level = int(level)
            if used_level < 0:
                raise ValueError(f"level must be >= 0, got {used_level}")
        # compute target size (clamp to >= 1)
        scale = 2 ** used_level
        target_h = max(1, h // scale)
        target_w = max(1, w // scale)
        # resize if needed
        if used_level > 0 and (target_h != h or target_w != w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            print(f"Downsampled x(2^{used_level}) → shape {img.shape}")
        else:
            print("No downsampling needed (level 0)")

        selected_level = used_level

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

def open_ome_level_lazy(path, series=0, level=0):
    tif = tifffile.TiffFile(path)
    s = tif.series[series]

    # levels: pyramidal -> s.levels[level]; non-pyramidal -> s itself
    lv = s.levels[level] if hasattr(s, "levels") else s

    root = zarr.open(lv.aszarr(), mode="r")

    # root may be Array or Group
    if isinstance(root, zarr.Array):
        arr = root
    else:
        # root is Group: pick the first array inside
        arrays = list(root.arrays())  # list of (name, zarr.Array)
        if not arrays:
            raise RuntimeError(f"No zarr arrays found in group. keys={list(root.group_keys())}")
        name, arr = arrays[0]
        print(f"[INFO] zarr root is Group, using first array: {name}")

    print(f"[INFO] ome-tiff lazy array's shape={arr.shape}")
    return tif, arr

def read_crop_patch(img_z, x0, y0, w, h):
    if img_z.ndim == 3 and img_z.shape[-1] in (3, 4):  # Y,X,C
        return img_z[y0:y0+h, x0:x0+w, :]
    elif img_z.ndim == 3:  # C,Y,X
        p = img_z[:, y0:y0+h, x0:x0+w]
        return np.moveaxis(p, 0, -1)
    else:  # grayscale
        return img_z[y0:y0+h, x0:x0+w]

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

def upsample_tile(tile, scale=2):
    """
    tile:
      - RGB: (H, W, 3)
      - DAPI u8/u16: (H, W)
    """
    if tile.ndim == 2:
        # grayscale (DAPI)
        up = rescale(
            tile,
            scale=scale,
            anti_aliasing=True,
            preserve_range=True
        )
    elif tile.ndim == 3:
        # RGB
        up = rescale(
            tile,
            scale=(scale, scale),
            anti_aliasing=True,
            channel_axis=2,
            preserve_range=True
        )
    else:
        raise ValueError(f"Unsupported shape: {tile.shape}")

    return up.astype(tile.dtype)


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
import numpy as np
import cv2
from skimage import color

# ----------------------------
# Helper: watershed split
# ----------------------------
def split_touching_objects_watershed(
    mask_dark,
    min_area=30,
    opening_ksize=3,
    dist_frac=0.45,
    peak_min_dist=6,
    debug=False,
    debug_prefix=None,
):
    """
    mask_dark: bool / {0,1} / {0,255} nuclei foreground mask
    Returns:
      labels: int32 (0=bg, 1..K instances)
      mask_clean: uint8 (0/255)
    """
    mask = mask_dark.astype(np.uint8)
    if mask.max() == 1:
        mask *= 255

    # --- optional: opening to cut thin bridges ---
    if opening_ksize and opening_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_ksize, opening_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # --- remove tiny components ---
    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    mask2 = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            mask2[lab == i] = 255
    mask = mask2

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32), mask

    # --- distance transform ---
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # --- sure foreground (seeds) ---
    sure_fg = (dist_norm > float(dist_frac)).astype(np.uint8) * 255

    # --- enforce min distance between seeds (simple erosion) ---
    if peak_min_dist and peak_min_dist > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * peak_min_dist + 1, 2 * peak_min_dist + 1))
        sure_fg = cv2.erode(sure_fg, k, iterations=1)

    n_markers, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
    markers = markers.astype(np.int32)

    # unknown = mask - sure_fg
    unknown = ((mask > 0) & (sure_fg == 0)).astype(np.uint8) * 255

    # watershed needs 3-channel image; use mask edges works OK, or pass original RGB if you want
    img_ws = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # background label must be 1; unknown=0
    markers = markers + 1
    markers[unknown > 0] = 0

    markers_ws = cv2.watershed(img_ws, markers)  # boundaries -> -1

    labels = np.zeros_like(markers_ws, dtype=np.int32)
    labels[markers_ws > 1] = markers_ws[markers_ws > 1] - 1

    # --- remove tiny labels again & relabel to 1..K ---
    out = np.zeros_like(labels, dtype=np.int32)
    cur = 1
    for lab_id in range(1, labels.max() + 1):
        area = int(np.sum(labels == lab_id))
        if area >= int(min_area):
            out[labels == lab_id] = cur
            cur += 1

    mask_clean = (out > 0).astype(np.uint8) * 255

    if debug:
        print(
            f"[DEBUG watershed] cc_in={num-1}, seeds={n_markers-1}, out_labels={out.max()}, "
            f"mask_px={int((mask_clean>0).sum())}",
            flush=True
        )

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_mask_in.png", mask)
        cv2.imwrite(f"{debug_prefix}_dist.png", (dist_norm * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_prefix}_sure_fg.png", sure_fg)
        cv2.imwrite(f"{debug_prefix}_mask_out.png", mask_clean)

    return out, mask_clean


# ----------------------------
# Main: your function, patched
# ----------------------------
def segment_super_dark_nuclei_full(
    rgb_tile,
    upsample_scale=2,
    n_smooth=2,
    intensity_threshold=0.7,
    # NEW knobs:
    split=True,
    split_opening_ksize=3,
    split_dist_frac=0.42,
    split_peak_min_dist=6,
    min_label_size=30,
    debug=False,
    debug_prefix=None,
):
    tile_up = upsample_tile(rgb_tile, scale=upsample_scale)

    # ---- your original H channel extraction ----
    try:
        _, H, E = normalizeStaining(tile_up)
        H_gray = color.rgb2gray(H)
    except Exception as e:
        H_gray = color.rgb2gray(tile_up)

    H_smooth = morphological_smooth(H_gray, n=n_smooth)
    mask_dark = threshold_super_dark(H_smooth, intensity_threshold=float(intensity_threshold))

    # ---- DEBUG prints ----
    if debug:
        md = mask_dark.astype(np.uint8)
        frac = float(md.mean()) if md.max() <= 1 else float((md > 0).mean())
        print(
            f"[DEBUG seg] tile_up={tile_up.shape} H_gray={H_gray.shape} "
            f"H_gray min/max=({H_gray.min():.3f},{H_gray.max():.3f}) "
            f"mask_dark_fg_frac={frac:.4f}",
            flush=True
        )

    # ---- KEY CHANGE: split touching nuclei here ----
    if split:
        labeled_mask, mask_clean = split_touching_objects_watershed(
            mask_dark,
            min_area=int(min_label_size),
            opening_ksize=int(split_opening_ksize),
            dist_frac=float(split_dist_frac),
            peak_min_dist=int(split_peak_min_dist),
            debug=debug,
            debug_prefix=debug_prefix,
        )
        # 你仍然想返回原始 mask_dark：就返回 mask_dark，不替换
    else:
        # keep your old behavior
        labeled_mask = separate_and_label(mask_dark, min_label_size=int(min_label_size))

    overlay = overlay_contours(tile_up, labeled_mask)
    return labeled_mask, mask_dark


def plot_cell_centroid(
    cells,
    he=None,
    color="red",
    save_name=None,
    save_fig=True,
    dot_size=5,
    dpi=100,          # ⭐ 统一 DPI
):
    print("Directly working on transformed coordinates...")

    if he is not None:
        height_pixels, width_pixels = he.shape[:2]
    else:
        width_pixels = int(np.ceil(cells.x_centroid.max()))
        height_pixels = int(np.ceil(cells.y_centroid.max()))

    # ⭐ 核心：figsize × dpi = 像素
    figsize = (width_pixels / dpi, height_pixels / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if he is not None:
        ax.imshow(he)

    ax.scatter(
        cells.x_centroid,
        cells.y_centroid,
        s=dot_size,
        c=color,
        marker="o",
        linewidths=0,
    )

    ax.set_xlim(0, width_pixels)
    ax.set_ylim(height_pixels, 0)  # y 轴向下
    ax.axis("off")

    print("saving the plot...")
    if save_fig:
        save_path = Path(save_name) if save_name else Path("cell_centroids.png")
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()

# def fill_holes_binary(mask255: np.ndarray) -> np.ndarray:
#     """
#     Fill holes inside foreground (mask is 0/255).
#     Robust even when outside background is split into multiple components.
#     """
#     mask255 = (mask255 > 0).astype(np.uint8) * 255
#     h, w = mask255.shape[:2]
#
#     # background image: bg=255, fg=0
#     bg = (mask255 == 0).astype(np.uint8) * 255
#
#     flood = bg.copy()
#     ffmask = np.zeros((h + 2, w + 2), np.uint8)
#
#     # floodFill ALL border background components to 0 (mark as outside)
#     # top/bottom rows
#     for x in range(w):
#         if flood[0, x] == 255:
#             cv2.floodFill(flood, ffmask, (x, 0), 0)
#         if flood[h - 1, x] == 255:
#             cv2.floodFill(flood, ffmask, (x, h - 1), 0)
#
#     # left/right cols
#     for y in range(h):
#         if flood[y, 0] == 255:
#             cv2.floodFill(flood, ffmask, (0, y), 0)
#         if flood[y, w - 1] == 255:
#             cv2.floodFill(flood, ffmask, (w - 1, y), 0)
#
#     # remaining bg==255 are holes
#     holes = (flood == 255)
#     out = mask255.copy()
#     out[holes] = 255
#     return out

def fill_holes_binary(
    mask255: np.ndarray,
    max_hole_area_frac=0.005,     # 相对整图
    max_hole_area_frac_fg=0.05,   # 相对前景面积
    max_hole_radius=6,            # NEW: 只填“半径 <= 6px”的小洞（强烈推荐）
    connectivity=8,
) -> np.ndarray:
    """
    mask255: 前景=255, 背景=0 的二值图（uint8）
    只填 “面积小 + 半径小” 的洞，避免把 ring 的大空腔填掉
    """
    mask255 = (mask255 > 0).astype(np.uint8) * 255
    h, w = mask255.shape[:2]

    # 背景图（255=背景）
    bg = (mask255 == 0).astype(np.uint8) * 255

    # flood fill：把与边界连通的背景清掉，剩下的背景就是“洞”
    flood = bg.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # 四条边界做 floodFill（更快也更稳：只挑边界上的255点做种子）
    for x in range(w):
        if flood[0, x] == 255:       cv2.floodFill(flood, ffmask, (x, 0), 0)
        if flood[h-1, x] == 255:     cv2.floodFill(flood, ffmask, (x, h-1), 0)
    for y in range(h):
        if flood[y, 0] == 255:       cv2.floodFill(flood, ffmask, (0, y), 0)
        if flood[y, w-1] == 255:     cv2.floodFill(flood, ffmask, (w-1, y), 0)

    holes = (flood == 255).astype(np.uint8)  # 1=hole
    if holes.max() == 0:
        return mask255

    num, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=connectivity)
    if num <= 1:
        return mask255

    fg_area = int((mask255 > 0).sum())
    max_area = int(max(max_hole_area_frac * (h * w),
                       max_hole_area_frac_fg * fg_area))

    # NEW: 用 distance transform 估计“洞的最大半径”
    # 注意：distanceTransform 输入要是 0/非0，这里 holes 是 0/1，OK
    dt = cv2.distanceTransform(holes, cv2.DIST_L2, 3)

    out = mask255.copy()
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_area:
            continue

        # 这个洞的“最大内切圆半径”
        max_r = float(dt[lab == i].max()) if area > 0 else 0.0

        # 只填“面积小 + 半径小”的洞
        if max_r <= float(max_hole_radius):
            out[lab == i] = 255

    return out

def remove_small_components(mask255: np.ndarray, min_area: int, connectivity: int = 8) -> tuple[np.ndarray, dict]:
    """Remove connected components with area < min_area. mask is 0/255."""
    bw = (mask255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=connectivity)

    keep = np.zeros_like(bw, dtype=np.uint8)
    kept_areas = []
    for k in range(1, num):  # skip background
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            keep[labels == k] = 1
            kept_areas.append(area)

    out = (keep * 255).astype(np.uint8)
    info = {
        "cc_total": int(num - 1),
        "cc_kept": int(len(kept_areas)),
        "kept_area_min": int(min(kept_areas)) if kept_areas else None,
        "kept_area_median": int(np.median(kept_areas)) if kept_areas else None,
        "kept_area_max": int(max(kept_areas)) if kept_areas else None,
    }
    return out, info


def mask_to_rgba(mask_bgr, color_rgb=(255, 0, 0), alpha=0.6):
    """
    mask_bgr: (H,W,3) uint8, foreground > 0
    color_rgb: foreground color
    alpha: 0~1

    return: rgba float32 in [0,1]
    """
    if mask_bgr.ndim == 3:
        mask = mask_bgr[..., 0] > 0
    else:
        mask = mask_bgr > 0

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    r, g, b = color_rgb
    rgba[mask, 0] = r / 255.0
    rgba[mask, 1] = g / 255.0
    rgba[mask, 2] = b / 255.0
    rgba[mask, 3] = alpha

    return rgba


def warp_mask(mask_bgr, H, out_shape):
    return cv2.warpPerspective(
        mask_bgr,
        H,
        out_shape,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def overlay_rgba_on_bgr(bg_bgr, fg_rgba):
    """
    bg_bgr: uint8 (H,W,3)
    fg_rgba: float32 (H,W,4) in [0,1]
    """
    bg = bg_bgr.astype(np.float32) / 255.0
    fg_rgb = fg_rgba[..., :3]
    alpha = fg_rgba[..., 3:4]

    out = bg * (1 - alpha) + fg_rgb * alpha
    return (out * 255).astype(np.uint8)

