# ============================================================
# dapi_lut_window.py - Choose a DAPI display window from the image itself
#
# The Stage-1 GUI ships one fixed LUT cutoff (300 counts) and one fixed slider
# range (0-2000). That pair only suits a slide whose background sits just above
# zero and whose nuclei peak inside two thousand counts. A dataset that reaches
# the manual-alignment queue is frequently the opposite case: a high or drifting
# background, a heavily skewed intensity histogram, or a few saturated specks
# that flatten every real nucleus to black once the LUT rescales on the maximum.
#
# The window here is derived per image instead:
#
#   cutoff = median + k * sigma_robust     (background level, not a constant)
#   high   = a high percentile             (so hot pixels cannot own the top)
#
# The median and the MAD are used rather than the mean and the standard
# deviation because background pixels dominate a DAPI level image while the
# bright tail is exactly what must not move the estimate -- that skew is the
# failure mode this replaces.
# ============================================================
from dataclasses import dataclass

import numpy as np

MAD_TO_SIGMA = 1.4826  # MAD -> standard deviation for normally distributed noise


@dataclass
class LutWindow:
    """A display window for one DAPI image, in the image's own intensity units."""

    cutoff: int          # LUT threshold: everything at or below this is background
    high: int            # top of the display range; brighter pixels are clipped
    background: float    # estimated background level (the median)
    sigma: float         # robust background spread
    slider_max: int      # a slider range that actually contains `cutoff`
    slider_step: int

    def as_event(self):
        """The numbers worth recording in pipeline_times.json."""
        return {
            "cutoff": self.cutoff,
            "high": self.high,
            "background": round(self.background, 3),
            "sigma": round(self.sigma, 3),
        }


def _sample(image, target_pixels):
    """Flatten to one channel and subsample on a stride, to keep this instant."""
    data = np.asarray(image)
    if data.ndim == 3:
        data = data[..., 0]
    data = data.reshape(-1)
    if data.size > target_pixels:
        stride = int(np.ceil(data.size / target_pixels))
        data = data[::stride]
    return data.astype(np.float32, copy=False)


def auto_lut_window(image, *, sigmas=3.0, high_percentile=99.9,
                    target_pixels=2_000_000):
    """Return the `LutWindow` to open a DAPI image with.

    `sigmas` sets how far above the background the cutoff sits. Three robust
    sigmas keeps ordinary background noise out of the picture while leaving dim
    nuclei visible; the caller can still move the slider.
    """
    data = _sample(image, target_pixels)
    if data.size == 0:
        return LutWindow(0, 1, 0.0, 0.0, 2000, 10)

    background = float(np.median(data))
    sigma = float(np.median(np.abs(data - background)) * MAD_TO_SIGMA)
    if sigma <= 0.0:
        # A quantized or largely empty background gives a zero MAD. Fall back to
        # the upper half-width of the distribution, which survives that case.
        sigma = max(float(np.percentile(data, 84.13)) - background, 1.0)

    cutoff = background + sigmas * sigma
    # Never below the background itself, and never so high that the tissue is
    # thresholded away along with the background.
    cutoff = min(max(cutoff, background), float(np.percentile(data, 99.0)))

    high = float(np.percentile(data, high_percentile))
    if high <= cutoff:
        high = max(float(data.max()), cutoff + 1.0)

    cutoff_i = int(round(cutoff))
    high_i = int(round(high))
    # The slider must reach past the chosen cutoff on both sides, so a user who
    # disagrees with the estimate can actually correct it.
    slider_max = max(high_i, cutoff_i * 2, 100)
    slider_step = max(1, int(round(slider_max / 400)))
    # Snap onto the slider's own grid, so the value the window opens at is the
    # value the slider reports rather than the next tick along.
    cutoff_i = int(round(cutoff_i / slider_step) * slider_step)
    return LutWindow(
        cutoff=cutoff_i,
        high=high_i,
        background=background,
        sigma=sigma,
        slider_max=slider_max,
        slider_step=slider_step,
    )


def clip_to_window(image, high):
    """Clip the top of a DAPI image before it is rendered through the LUT.

    `dapi_to_lut_rgb` rescales on the clipped image's maximum, so one saturated
    speck otherwise compresses every nucleus into the first LUT entries.
    """
    data = np.asarray(image)
    return np.minimum(data, np.asarray(high, dtype=data.dtype))
