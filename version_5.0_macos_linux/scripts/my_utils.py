# ============================================================
# my_utils.py - Shared utilities
#
# Common helpers used across all stages: image / pyramid reading,
# DAPI LUT coloring, H&E stain separation, nuclei mask operations,
# TPS-free geometry, and overlay helpers.
# Usage: imported as a module (not run directly).
# ============================================================
import numpy as np
import tifffile
import cv2
# import zarr
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def scalar_mpp(value):
    """OME spacing is reported per axis; collapse it to one number."""
    if value is None:
        return None
    array = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return None
    return float(np.sqrt(np.prod(array)) if array.size > 1 else array[0])


def choose_level_by_target_mpp(base_mpp, n_levels, target_mpp, *,
                               min_level=0, max_level=None):
    """Pick the pyramid level whose physical pixel size is closest to `target_mpp`.

    The ladder is geometric, so closeness is measured in log2 -- one level too fine and
    one level too coarse are then treated as equally far from the target instead of the
    coarse one always looking closer on a linear scale.

    Returns None when the base spacing is unknown, so the caller can fall back.
    """
    base_mpp = scalar_mpp(base_mpp)
    if base_mpp is None:
        return None
    if not target_mpp or target_mpp <= 0:
        raise ValueError("target_mpp must be positive")
    highest = int(n_levels) - 1 if max_level is None else min(int(max_level), int(n_levels) - 1)
    lowest = max(0, int(min_level))
    if highest < lowest:
        return None
    candidates = range(lowest, highest + 1)
    return int(min(candidates,
                   key=lambda level: abs(np.log2((base_mpp * 2 ** level) / float(target_mpp)))))


def ome_physical_pixel_size(tif, series=0):
    """Read one OME image's level-0 (x, y) pixel spacing in micrometers."""
    metadata = tif.ome_metadata
    if not metadata:
        return None

    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return None
        factors = {
            "pm": 1e-6,
            "nm": 1e-3,
            "µm": 1.0,
            "μm": 1.0,
            "um": 1.0,
            "mm": 1e3,
            "cm": 1e4,
            "m": 1e6,
        }
        values = []
        for axis in ("X", "Y"):
            size = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
            factor = factors.get(unit)
            if factor is None or not np.isfinite(size) or size <= 0:
                return None
            values.append(size * factor)
        return tuple(values)
    except (IndexError, KeyError, TypeError, ValueError, ET.ParseError):
        return None


def ome_channel_names(tif, series=0):
    """Return one OME image's channel names in C order.

    Missing ``Name`` attributes are represented by empty strings. The returned
    list is padded to ``SizeC`` so its indices remain channel indices even when
    the producer omitted one or more ``Channel`` elements.
    """
    metadata = getattr(tif, "ome_metadata", None)
    if not metadata:
        return []
    try:
        images = ET.fromstring(metadata).findall("./{*}Image")
        pixels = images[int(series)].find("./{*}Pixels")
        if pixels is None:
            return []
        size_c = int(pixels.attrib.get("SizeC", 0))
        names = [
            str(channel.attrib.get("Name", "") or "").strip()
            for channel in pixels.findall("./{*}Channel")
        ]
        if size_c > len(names):
            names.extend([""] * (size_c - len(names)))
        return names
    except (IndexError, TypeError, ValueError, ET.ParseError):
        return []


def _dapi_name_rank(name):
    """Rank a channel name by the pipeline's nuclear-stain preference."""
    text = str(name or "").strip().casefold()
    if not text:
        return None
    # Word-like matching accepts common producer spellings such as ``01_DAPI``
    # and ``01_Hoechst`` without letting an unrelated substring win.
    if re.search(r"(?<![a-z])dapi(?![a-z])", text):
        return 0
    if re.search(r"(?<![a-z])hoechst(?![a-z])", text):
        return 1
    if text == "dna":
        return 2
    if re.fullmatch(r"dna\s*\(\s*0\s*\)", text):
        return 3
    if re.search(r"(?<![a-z])dna(?![a-z])", text):
        return 4
    return None


def _tiff_channel_count(tif, series=0, names=()):
    """Best-effort channel count without decoding image pixels."""
    metadata = getattr(tif, "ome_metadata", None)
    if metadata:
        try:
            images = ET.fromstring(metadata).findall("./{*}Image")
            pixels = images[int(series)].find("./{*}Pixels")
            if pixels is not None:
                return max(1, int(pixels.attrib.get("SizeC", 1)))
        except (IndexError, TypeError, ValueError, ET.ParseError):
            pass
    selected = tif.series[int(series)]
    axes = str(getattr(selected, "axes", "") or "").upper().replace("S", "C")
    if "C" in axes:
        return int(selected.shape[axes.index("C")])
    if "I" in axes and len(selected.shape) == len(axes):
        return int(selected.shape[axes.index("I")])
    if names:
        return len(names)
    return 1


def resolve_dapi_channel(path, requested=None, *, series=0, log=print):
    """Resolve a DAPI-equivalent channel index and describe why it was chosen.

    An explicit zero-based index wins. Otherwise OME names are ranked as
    DAPI > Hoechst > DNA > DNA (0) > the earliest remaining DNA-like name.
    Inputs with no recognized name fall back to C0 and emit an INFO warning.
    """
    path = Path(path)
    names = []
    count = 1
    if str(path).lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tif:
            names = ome_channel_names(tif, series)
            count = _tiff_channel_count(tif, series, names)
    else:
        try:
            from PIL import Image
            with Image.open(path) as image:
                count = max(1, len(image.getbands()))
        except Exception:
            count = 1

    if requested is not None:
        index = int(requested)
        if index < 0 or index >= count:
            raise ValueError(
                f"DAPI channel C{index} is out of range for {path.name}; "
                f"available channels are C0..C{count - 1}"
            )
        name = names[index] if index < len(names) else ""
        source = "explicit"
        if _dapi_name_rank(name) is None:
            label = repr(name) if name else "<unnamed>"
            log(
                f"[INFO] WARNING: explicitly selected DAPI channel C{index} "
                f"({label}) in {path.name}, but its name is not a common "
                "DAPI/Hoechst/DNA name.",
                flush=True,
            )
    else:
        candidates = [
            (rank, index, name)
            for index, name in enumerate(names[:count])
            if (rank := _dapi_name_rank(name)) is not None
        ]
        if candidates:
            rank, index, name = min(candidates, key=lambda item: (item[0], item[1]))
            source = "ome_name"
        else:
            index = 0
            name = names[0] if names else ""
            source = "fallback_c0"
            available = ", ".join(
                f"C{i}={channel_name!r}"
                for i, channel_name in enumerate(names[:count])
                if channel_name
            ) or "no channel names"
            log(
                f"[INFO] WARNING: {path.name} has no recognized "
                "DAPI/Hoechst/DNA channel name ({available}); falling back to C0.",
                flush=True,
            )

    label = f" ({name})" if name else ""
    log(
        f"[INFO] DAPI channel selection: C{index}{label}, source={source}",
        flush=True,
    )
    return {
        "index": int(index),
        "name": str(name or ""),
        "source": source,
        "count": int(count),
    }


def _pyramid_level_candidates(level_shapes, base_mpp=None):
    """Describe available levels without decoding any image pixels."""
    if not level_shapes:
        raise ValueError("image pyramid has no levels")
    base_height, base_width = map(int, level_shapes[0])
    candidates = []
    for index, (height, width) in enumerate(level_shapes):
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid pyramid dimensions: {height}x{width}")
        mpp = None
        if base_mpp is not None:
            mpp_x = float(base_mpp[0]) * base_width / width
            mpp_y = float(base_mpp[1]) * base_height / height
            mpp = float(np.sqrt(mpp_x * mpp_y))
        candidates.append({
            "level": index,
            "height": height,
            "width": width,
            "short_side": min(height, width),
            "mpp": mpp,
        })
    return candidates


def _smallest_level(candidates):
    return min(
        candidates,
        key=lambda item: (
            item["short_side"],
            item["height"] * item["width"],
        ),
    )


# 1024-2048 rather than 1000-2000: the competing methods now express a working
# resolution as a power of two (VALIS caps at 1024/2048/4096, and the VoltRon and
# RegWSI tiers follow), so Stage 1 accepts the same numbers. The rule itself is
# unchanged -- this is still the short side, and still chosen automatically.
def select_pyramid_level(level_shapes, *, requested_level=None,
                         min_size=1024, max_size=2048, base_mpp=None,
                         min_mpp=3.0, max_mpp=9.0):
    """Prefer 1024-2048 px short side at 3-9 MPP, then dimensions, then valid MPP."""
    candidates = _pyramid_level_candidates(level_shapes, base_mpp)
    if requested_level is not None:
        chosen = int(requested_level)
        if chosen < 0 or chosen >= len(candidates):
            raise ValueError(
                f"level out of range: {chosen} for n_levels={len(candidates)}"
            )
        return chosen

    preferred_size = [
        candidate for candidate in candidates
        if min_size <= candidate["short_side"] <= max_size
    ]
    preferred_mpp = [
        candidate for candidate in candidates
        if candidate["mpp"] is not None
        and min_mpp <= candidate["mpp"] <= max_mpp
    ]
    preferred_both = [
        candidate for candidate in preferred_size
        if candidate["mpp"] is not None
        and min_mpp <= candidate["mpp"] <= max_mpp
    ]

    selected = _smallest_level(
        preferred_both or preferred_size or preferred_mpp or [candidates[0]]
    )
    mpp_text = (
        "unknown" if selected["mpp"] is None
        else f"{selected['mpp']:.4f} µm/px"
    )
    if preferred_both:
        reason = "preferred size + MPP"
    elif preferred_size:
        reason = "preferred size"
    elif preferred_mpp:
        reason = "smallest resolution in MPP range"
    else:
        reason = "level-0 fallback"
    print(
        f"[INFO] Level selection: level={selected['level']}, "
        f"resolution={selected['width']}x{selected['height']}, "
        f"mpp={mpp_text} ({reason})"
    )
    return selected["level"]


def inspect_image_pyramid(path, *, series=0, page=0):
    """Inspect level dimensions and OME physical spacing without loading images."""
    path = Path(path)
    filename = str(path).lower()

    def shape_hw(shape, axes=""):
        axes = (axes or "").upper()
        if "Y" in axes and "X" in axes:
            return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
        if len(shape) >= 3 and shape[0] in (2, 3, 4) and shape[-1] not in (2, 3, 4):
            return int(shape[1]), int(shape[2])
        return int(shape[0]), int(shape[1])

    if filename.endswith((".ome.tif", ".ome.tiff")):
        with tifffile.TiffFile(path) as tif:
            selected_series = tif.series[int(series)]
            base_mpp = ome_physical_pixel_size(tif, series)
            base_page = selected_series.pages[int(page)]
            sub_pages = list(getattr(base_page, "pages", []) or [])
            if sub_pages:
                levels = [base_page] + sub_pages
            else:
                levels = list(getattr(selected_series, "levels", []) or [])
            if levels:
                return {
                    "level_shapes": [
                        shape_hw(level.shape, getattr(level, "axes", ""))
                        for level in levels
                    ],
                    "base_mpp": base_mpp,
                }
            height, width = shape_hw(
                selected_series.shape, getattr(selected_series, "axes", "")
            )
    elif filename.endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tif:
            selected_series = tif.series[int(series)]
            height, width = shape_hw(
                selected_series.shape, getattr(selected_series, "axes", "")
            )
        base_mpp = None
    else:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        base_mpp = None

    level_shapes = [(int(height), int(width))]
    while min(level_shapes[-1]) > 1:
        level_shapes.append((
            max(1, level_shapes[-1][0] // 2),
            max(1, level_shapes[-1][1] // 2),
        ))
    return {"level_shapes": level_shapes, "base_mpp": base_mpp}


def select_paired_pyramid_levels(moving_shapes, fixed_shapes, *,
                                 moving_base_mpp=None, fixed_base_mpp=None,
                                 requested_moving_level=None,
                                 requested_fixed_level=None,
                                 min_size=1024, max_size=2048,
                                 min_mpp=3.0, max_mpp=9.0):
    """Jointly select moving/fixed levels using size first and matched MPP."""
    moving = _pyramid_level_candidates(moving_shapes, moving_base_mpp)
    fixed = _pyramid_level_candidates(fixed_shapes, fixed_base_mpp)

    def in_size(candidate):
        return min_size <= candidate["short_side"] <= max_size

    def in_mpp(candidate):
        return (candidate["mpp"] is not None
                and min_mpp <= candidate["mpp"] <= max_mpp)

    def independent(candidates, base_mpp):
        return candidates[select_pyramid_level(
            [(item["height"], item["width"]) for item in candidates],
            base_mpp=base_mpp,
            min_size=min_size,
            max_size=max_size,
            min_mpp=min_mpp,
            max_mpp=max_mpp,
        )]

    def closest_to(anchor, candidates, base_mpp):
        if anchor["mpp"] is None:
            return independent(candidates, base_mpp)
        with_mpp = [item for item in candidates if item["mpp"] is not None]
        if not with_mpp:
            return independent(candidates, base_mpp)
        return min(
            with_mpp,
            key=lambda item: (
                abs(item["mpp"] - anchor["mpp"]),
                not in_size(item),
                item["short_side"],
            ),
        )

    moving_requested = None
    fixed_requested = None
    if requested_moving_level is not None:
        moving_requested = moving[select_pyramid_level(
            moving_shapes, requested_level=requested_moving_level,
            base_mpp=moving_base_mpp,
        )]
    if requested_fixed_level is not None:
        fixed_requested = fixed[select_pyramid_level(
            fixed_shapes, requested_level=requested_fixed_level,
            base_mpp=fixed_base_mpp,
        )]

    if moving_requested is not None and fixed_requested is not None:
        selected_moving, selected_fixed = moving_requested, fixed_requested
    elif moving_requested is not None:
        selected_moving = moving_requested
        selected_fixed = closest_to(selected_moving, fixed, fixed_base_mpp)
    elif fixed_requested is not None:
        selected_fixed = fixed_requested
        selected_moving = closest_to(selected_fixed, moving, moving_base_mpp)
    else:
        moving_preferred = [item for item in moving if in_size(item) and in_mpp(item)]
        fixed_preferred = [item for item in fixed if in_size(item) and in_mpp(item)]
        if moving_preferred and fixed_preferred:
            selected_moving = _smallest_level(moving_preferred)
            selected_fixed = _smallest_level(fixed_preferred)
        elif moving_preferred:
            selected_moving = _smallest_level(moving_preferred)
            selected_fixed = closest_to(selected_moving, fixed, fixed_base_mpp)
        elif fixed_preferred:
            selected_fixed = _smallest_level(fixed_preferred)
            selected_moving = closest_to(selected_fixed, moving, moving_base_mpp)
        else:
            moving_sized = [item for item in moving if in_size(item)]
            fixed_sized = [item for item in fixed if in_size(item)]
            if moving_sized and fixed_sized:
                selected_moving = _smallest_level(moving_sized)
                selected_fixed = _smallest_level(fixed_sized)
            elif moving_sized:
                selected_moving = _smallest_level(moving_sized)
                selected_fixed = closest_to(selected_moving, fixed, fixed_base_mpp)
            elif fixed_sized:
                selected_fixed = _smallest_level(fixed_sized)
                selected_moving = closest_to(selected_fixed, moving, moving_base_mpp)
            else:
                selected_moving = independent(moving, moving_base_mpp)
                selected_fixed = closest_to(selected_moving, fixed, fixed_base_mpp)

    def describe(candidate):
        mpp = "unknown" if candidate["mpp"] is None else f"{candidate['mpp']:.4f}"
        return (
            f"level={candidate['level']}, "
            f"resolution={candidate['width']}x{candidate['height']}, "
            f"mpp={mpp}"
        )

    print(
        f"[INFO] Paired Stage-1 level selection: moving({describe(selected_moving)}) "
        f"fixed({describe(selected_fixed)})"
    )
    return selected_moving["level"], selected_fixed["level"]


def select_paired_image_levels(moving_path, fixed_path, *,
                               requested_moving_level=None,
                               requested_fixed_level=None,
                               min_size=1024, max_size=2048,
                               min_mpp=3.0, max_mpp=9.0):
    """Inspect both slides first, then choose their Stage-1 levels jointly."""
    moving = inspect_image_pyramid(moving_path)
    fixed = inspect_image_pyramid(fixed_path)
    return select_paired_pyramid_levels(
        moving["level_shapes"],
        fixed["level_shapes"],
        moving_base_mpp=moving["base_mpp"],
        fixed_base_mpp=fixed["base_mpp"],
        requested_moving_level=requested_moving_level,
        requested_fixed_level=requested_fixed_level,
        min_size=min_size,
        max_size=max_size,
        min_mpp=min_mpp,
        max_mpp=max_mpp,
    )


def read_image(
    path,
    keep_16bit=True,
    series=0,
    page=0,
    level=None,
    min_size=1024,
    max_size=2048,
    force_rgb=True,
    channel="auto",   # "auto" | "he" | "dapi" | int | None
    maxworkers=None,  # tile decoding threads; None lets tifffile decide
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
          * resolve an OME channel name as DAPI > Hoechst > DNA > DNA (0) >
            earliest remaining DNA-like name; warn and use C0 if none match
      - channel=int:
          * pick that zero-based channel (CYX/IYX or YXC), print
      - channel=None:
          * keep all channels (convert CYX->YXC), no picking

    Returns:
      img: np.ndarray (H,W,C) if force_rgb or RGB image; else possibly (H,W)
      selected_level: for OME: chosen pyramid index (or pseudo_level if no pyramid)
                      for non-OME: pseudo_level
    """
    filename = str(path).lower()
    selected_level = None
    base_mpp = None

    # ``channel='dapi'`` is semantic: resolve it from OME names before any pixels
    # are decoded. An integer remains an explicit zero-based channel selection.
    if str(channel).lower() == "dapi":
        channel = resolve_dapi_channel(path, series=series)["index"]

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
        return select_pyramid_level(
            [get_hw_fn(index) for index in range(n_levels)],
            requested_level=level,
            min_size=min_size,
            max_size=max_size,
            base_mpp=base_mpp,
        )

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

        # ---------- CYX / IYX ----------
        # Some OME producers expose a channel-first stack as IYX even though the
        # OME XML records SizeC and one Channel element per plane.
        if axes in ("CYX", "IYX"):
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

            # channel="dapi" has already been resolved to an integer above.
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
            base_mpp = ome_physical_pixel_size(tif, series)

            # ---------- 1) Try page.pages (subIFDs pyramid) ----------
            pages = None
            try:
                # NOTE: s.pages can be multi-page; you asked to start from "page"
                page_index = int(page)
                series_axes = str(getattr(s, "axes", "") or "").upper()
                series_levels = list(getattr(s, "levels", []) or [])
                # Channel-first OME writers store one base page (and one SubIFD
                # ladder) per C/I plane. Select that page before decoding; an
                # interleaved YXC/YXS image still uses page 0 and is sliced below.
                if isinstance(channel, int) and (
                    "C" in series_axes or "I" in series_axes
                ) and len(s.pages) > 1:
                    page_index = int(channel)
                if page_index < 0 or page_index >= len(s.pages):
                    raise ValueError(
                        f"channel/page index {page_index} is out of range for "
                        f"{path}; pages={len(s.pages)}"
                    )
                # For channel-first pyramids, tifffile exposes the target page at
                # every series level. Using it avoids materializing all C planes
                # merely to select one (40-channel CycIF can otherwise use tens of
                # gigabytes at level 0).
                if (
                    isinstance(channel, int)
                    and ("C" in series_axes or "I" in series_axes)
                    and len(series_levels) > 1
                    and all(page_index < len(item.pages) for item in series_levels)
                ):
                    pages = [item.pages[page_index] for item in series_levels]
                else:
                    base_page = s.pages[page_index]
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

                arr = pages[selected_level].asarray(maxworkers=maxworkers)
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

                arr = levels[selected_level].asarray(maxworkers=maxworkers)
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
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
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

# def open_ome_level_lazy(path, series=0, level=0):
#     tif = tifffile.TiffFile(path)
#     s = tif.series[series]
#
#     # levels: pyramidal -> s.levels[level]; non-pyramidal -> s itself
#     lv = s.levels[level] if hasattr(s, "levels") else s
#
#     root = zarr.open(lv.aszarr(), mode="r")
#
#     # root may be Array or Group
#     if isinstance(root, zarr.Array):
#         arr = root
#     else:
#         # root is Group: pick the first array inside
#         arrays = list(root.arrays())  # list of (name, zarr.Array)
#         if not arrays:
#             raise RuntimeError(f"No zarr arrays found in group. keys={list(root.group_keys())}")
#         name, arr = arrays[0]
#         print(f"[INFO] zarr root is Group, using first array: {name}")
#
#     print(f"[INFO] ome-tiff lazy array's shape={arr.shape}")
#     return tif, arr

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

def resolve_dapi_lut_threshold(dapi_img_local, threshold=None):
    """Return a LUT cutoff represented safely in the image's native dtype.

    The historical 300-count default is useful for detector-native (>8-bit)
    fluorescence, but it cannot even be represented by uint8.  For an 8-bit
    image, use its median as a robust background estimate; bright nuclei do not
    pull that percentile upward.  An explicit threshold remains authoritative,
    but receives a clear validation error instead of NumPy's dtype-dependent
    ``OverflowError``.
    """
    dapi = np.asarray(
        dapi_img_local[..., 0] if dapi_img_local.ndim == 3 else dapi_img_local
    )
    if dapi.size == 0:
        raise ValueError("DAPI LUT threshold cannot be resolved from an empty image")

    if threshold is None:
        if np.issubdtype(dapi.dtype, np.integer) and np.iinfo(dapi.dtype).max > 255:
            threshold = 300
        else:
            sample = dapi.reshape(-1)
            if sample.size > 2_000_000:
                sample = sample[::int(np.ceil(sample.size / 2_000_000))]
            finite = sample[np.isfinite(sample)]
            if finite.size == 0:
                raise ValueError("DAPI image has no finite intensities")
            threshold = int(round(float(np.percentile(finite, 50.0))))

    threshold = int(threshold)
    if np.issubdtype(dapi.dtype, np.integer):
        bounds = np.iinfo(dapi.dtype)
        if threshold < bounds.min or threshold > bounds.max:
            raise ValueError(
                f"DAPI LUT threshold {threshold} is outside {dapi.dtype} range "
                f"[{bounds.min}, {bounds.max}]"
            )
    return threshold


def dapi_to_lut_rgb(dapi_img_local, lut_table, threshold=None):
    dapi = dapi_img_local[..., 0] if dapi_img_local.ndim == 3 else dapi_img_local
    threshold = resolve_dapi_lut_threshold(dapi, threshold)
    dapi_clipped = np.clip(dapi, threshold, None)
    d_min, d_max = dapi_clipped.min(), dapi_clipped.max()
    if d_max > d_min:
        scaled = ((dapi_clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(dapi_clipped, dtype=np.uint8)
    rgb = lut_table[scaled]
    return rgb

def upsample_tile(tile, scale=2):
    from skimage.transform import rescale

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
def find_pigment(rgb, max_channel=70, neutrality=0.60):
    """Near-black, colourless pixels -- anthracotic pigment, ink, dense hemosiderin.

    These are not stain, but `normalizeStaining` has nowhere to put them: it projects
    every pixel onto two stain vectors it estimates from the image's own optical
    densities, and a near-black pixel has an optical density several times any
    nucleus'. On `geo_lung_l8`, 2% of a tile being soot cost 85% of its nucleus mask
    (PLAN_4.8.md section 10).

    The test is darkness *and* colourlessness. Carbon absorbs flat across the
    spectrum, so its three channels are nearly equal; hematoxylin is purple, and a
    dense nucleus keeps a blue-over-green margin however dark it gets. Testing
    darkness alone would eat the darkest real nuclei.

    `neutrality` was measured rather than chosen: at 0.35 the test flags 0.4% of an
    l8 pigment tile and buys nothing (31 isolated nuclei against a baseline of 31);
    at 0.60 it flags 2.5% and the same tiles reach 78. Loosening further to 1.00 --
    darkness alone -- changes neither number, so 0.60 keeps a guard that costs
    nothing.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros(rgb.shape[:2], dtype=bool)
    channels = rgb[..., :3].astype(np.float32)
    brightest = channels.max(axis=2)
    spread = brightest - channels.min(axis=2)
    return (brightest < float(max_channel)) & (
        spread < float(neutrality) * np.maximum(brightest, 1.0)
    )


def inpaint_pigment(rgb, max_channel=70, neutrality=0.60, radius=5):
    """Replace pigment with its surroundings, so it cannot set the stain scale.

    Returns ``(repaired_rgb, pigment_fraction)``. Inpainting rather than masking out:
    dropping the pixels changes the shape of the optical-density cloud that Macenko
    fits its vectors to, which overshot into segmenting whole tissue regions when it
    was tried (PLAN_4.8.md section 10). Filling them with neighbouring tissue leaves the
    cloud looking like the slide without the soot.
    """
    pigment = find_pigment(rgb, max_channel=max_channel, neutrality=neutrality)
    fraction = float(pigment.mean())
    if not pigment.any():
        return np.asarray(rgb), fraction
    bgr = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, pigment.astype(np.uint8), float(radius), cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB), fraction


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
    from skimage.morphology import closing, disk, opening

    se = disk(n)
    img_open = opening(img_gray, se)
    img_close = closing(img_open, se)
    return img_close


# --- Step 3: Thresholding super-dark nuclei ---
def threshold_super_dark(H_gray, intensity_threshold=0.6):
    from skimage.morphology import (
        closing, disk, remove_small_holes, remove_small_objects,
    )

    # H_gray scaled 0-1
    mask_dark = H_gray > intensity_threshold
    mask_dark = remove_small_objects(mask_dark, min_size=30)
    mask_dark = remove_small_holes(mask_dark, area_threshold=60)
    mask_dark = closing(mask_dark, disk(1))
    return mask_dark


# --- Step 4: Separate touching nuclei & label ---
def separate_and_label(binary_mask, min_label_size=30):
    from scipy import ndimage as ndi
    from skimage.morphology import local_maxima
    from skimage.segmentation import watershed

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
    from skimage import color

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
    import matplotlib.pyplot as plt
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

    # areas of components 1..num-1 (label 0 is background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_mask = areas >= int(min_area)
    kept_labels = np.nonzero(keep_mask)[0] + 1  # +1 to skip background label

    # single-pass relabel via a label lookup table (avoids one full scan per component)
    lut = np.zeros(num, dtype=np.uint8)
    lut[kept_labels] = 255
    out = lut[labels]

    kept_areas = areas[keep_mask]
    info = {
        "cc_total": int(num - 1),
        "cc_kept": int(kept_areas.size),
        "kept_area_min": int(kept_areas.min()) if kept_areas.size else None,
        "kept_area_median": int(np.median(kept_areas)) if kept_areas.size else None,
        "kept_area_max": int(kept_areas.max()) if kept_areas.size else None,
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



def cli_params_block(engine=None):
    """CLI: return this engine's block of `scripts/parameters.json`.

    `engine` is a pair name -- 'dapi-he', 'he-he', ... -- and defaults to the pair
    `run_pharaoh.py` exported in the environment (`PHARAOH_PAIR`), the way it exports
    the QC-image setting: every stage is its own process and inherits it. With one
    engine serving every pair there is no directory left to infer from, so with
    neither an argument nor the variable the default pair's block, `dapi-he`, is
    returned. This file used to be copied into each engine and keyed off its own
    `__file__`, and the second copy looked its block up by the `he0` spelling when the
    key was `he0fg`, which returned an empty dict with nothing in the log to say so.

    Returns {} if the file or block is unavailable.
    """
    import os, json

    if engine is None:
        engine = os.environ.get(PAIR_ENV_VAR, "").strip() or "dapi-he"

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "parameters.json"),
                  os.path.join(here, os.pardir, "parameters.json")]
    for top in candidates:
        try:
            with open(top) as f:
                data = json.load(f)
        except Exception:
            continue
        block = data.get(engine, {}) if isinstance(data, dict) else {}
        return _with_stage4_tile_fraction_override(block)
    return {}


def _with_stage4_tile_fraction_override(block):
    """Apply `run_pharaoh.py --stage4-tile-fraction`, which every stage process inherits
    as STAGE4_TILE_FRACTION_ENV_VAR, to `stage3.stage4_tile_fraction` of the run's block."""
    import os

    value = os.environ.get(STAGE4_TILE_FRACTION_ENV_VAR, "").strip()
    if not value or not isinstance(block, dict):
        return block
    block = dict(block)
    block["stage3"] = {**block.get("stage3", {}), "stage4_tile_fraction": float(value)}
    return block


def percentile_stretch_to_u8(image, low_percentile=1.0, high_percentile=99.5):
    """Map an image's occupied intensity range onto 0-255 for display.

    A 16-bit DAPI slide typically fills only the bottom few percent of the uint16
    range -- one Xenium morphology image here peaks at 8,297 of 65,535 with 99% of
    its pixels below 2,202 -- so a viewer that maps 0-65535 to black-white shows an
    almost black picture. Windowing on percentiles rather than min/max keeps a few
    hot pixels from collapsing the contrast again.

    Returns uint8. Never use this on an image a later stage matches on: it discards
    the intensities Stage 2 registers with.
    """
    data = np.asarray(image)
    if data.ndim == 3:
        data = data[..., 0]
    data = data.astype(np.float32)
    low = float(np.percentile(data, low_percentile))
    high = float(np.percentile(data, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(data.min()), float(data.max())
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (data - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


QC_IMAGE_GROUPS = (
    "stage1_extras",
    "stage3_debug",
    "stage4b_overlays",
    "stage4b_density",
    "stage4c_patches",
    "stage4c_overlays",
    "stage6_overlays",
    "stage7_anchors",
    "stage7_tree",
)
QC_IMAGES_ENV_VAR = "PHARAOH_QC_IMAGES"
PAIR_ENV_VAR = "PHARAOH_PAIR"
STAGE4_TILE_FRACTION_ENV_VAR = "PHARAOH_STAGE4_TILE_FRACTION"
_QC_IMAGES_ALL = ("all", "1", "true", "on", "yes")
_QC_IMAGES_NONE = ("none", "0", "false", "off", "no")


def normalize_qc_images_setting(value):
    """Canonicalize a `--qc-images` value into what the env var carries.

    Accepts "all", "none", or a comma-separated subset of `QC_IMAGE_GROUPS`, and
    raises ValueError naming the bad token otherwise -- a mistyped group has to be a
    loud CLI error, because the quiet version is a run that writes no QC images and
    looks exactly like a run that was never asked for any.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(
            "QC image selection is empty; use 'all', 'none', or "
            f"a comma-separated subset of {', '.join(QC_IMAGE_GROUPS)}"
        )
    lowered = text.lower()
    if lowered in _QC_IMAGES_ALL:
        return "all"
    if lowered in _QC_IMAGES_NONE:
        return "none"
    groups = [
        token.strip()
        for token in lowered.replace(" ", ",").split(",")
        if token.strip()
    ]
    unknown = [name for name in groups if name not in QC_IMAGE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown QC image group(s): {', '.join(unknown)}; choose from "
            f"{', '.join(QC_IMAGE_GROUPS)}, or 'all'/'none'"
        )
    return ",".join(dict.fromkeys(groups))


def _qc_images_env_override(group):
    """Resolve `PHARAOH_QC_IMAGES` for one group, or None when it does not apply.

    The pipeline runs every stage as a separate process, so a per-run override has to
    travel in the environment: `run_pharaoh.py` sets this variable once and each child
    inherits it, which is what lets `--qc-images` be a per-run choice instead of an
    edit to the shared parameters.json.
    """
    import os

    raw = os.environ.get(QC_IMAGES_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        normalized = normalize_qc_images_setting(raw)
    except ValueError as exc:
        import sys

        print(f"[qc_images] ignoring {QC_IMAGES_ENV_VAR}={raw!r}: {exc}",
              file=sys.stderr, flush=True)
        return None
    if normalized == "all":
        return True
    if normalized == "none":
        return False
    return group in normalized.split(",")


def qc_images_enabled(group, default=False):
    """Whether a group of quality-control images should be written.

    These overlays and debug dumps are for a human to look at; nothing downstream reads
    them. On one CRC slide they are 172 MB of a 263 MB run directory, so a headless run
    leaves them off. Files later stages actually read -- `1_dapi_level_image.png` and
    `1_he_level_image.png`, which Stage 2 matches on -- are never gated here.

    `PHARAOH_QC_IMAGES` (set by `--qc-images` / `--no-qc-images`) wins over
    parameters.json, so one run can ask for overlays without changing the defaults
    every other run and every other dataset in a parallel batch reads.
    """
    import os, json

    override = _qc_images_env_override(group)
    if override is not None:
        return override

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "parameters.json"),
                      os.path.join(here, os.pardir, "parameters.json")):
        try:
            with open(candidate) as handle:
                data = json.load(handle)
        except Exception:
            continue
        settings = data.get("qc_images") if isinstance(data, dict) else None
        if isinstance(settings, dict):
            return bool(settings.get(group, default))
        return bool(default)
    return bool(default)
