"""Common utility helpers shared across blueprints.

Centralizes small duplicated snippets so updates (e.g. image encoding or
camera folder normalization) propagate consistently.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Union

import numpy as np
from loguru import logger
from PIL import Image


def camera_number(camera: Union[str, int]) -> int:
    """Return the numeric camera id from a value like 1, "1", "Cam1".

    Raises ValueError if it cannot parse a positive int.
    """
    if isinstance(camera, int):
        return camera
    s = str(camera).strip()
    if s.lower().startswith("cam"):
        s = s[3:]
    try:
        cam_int = int(s)
    except (TypeError, ValueError):
        logger.error(f"Invalid camera identifier (non-parsable): {camera!r}")
        raise
    if cam_int < 0:
        raise ValueError("camera must be positive integer")
    return cam_int


def numpy_to_png_base64(arr: np.ndarray, compress_level: int = 1) -> str:
    """Convert a numpy array (uint8 or convertible) to a base64 PNG string.

    Args:
        arr: Input numpy array
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
                       Default is 1 for speed. Use 6 for better compression.
    """
    return numpy_to_base64(arr, format="png", compress_level=compress_level)


def get_display_contrast_stats(arr: np.ndarray) -> dict:
    """Compute contrast slider positions for the sqrt-normalised 8-bit encoding.

    Returns vmin_pct/vmax_pct as percentages (0-100) of the encoded 8-bit
    range, representing a tighter window than the full [0, 100] default.
    This gives the frontend slider a meaningful starting position: the user
    can widen toward [0, 100] to see more, or tighten to boost contrast.

    The percentiles (5th–95th) are computed in the sqrt domain to match
    the encoding in numpy_to_base64().
    """
    if arr.dtype == np.uint8:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    a = np.sqrt(np.maximum(arr.astype(np.float32, copy=False), 0))
    if not a.size:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    total = a.size
    if total > 4_000_000:
        stride = max(2, int(np.sqrt(total / 1_000_000)))
        sampled = a[::stride, ::stride].ravel()
    else:
        sampled = a.ravel()

    # The encoding window used by numpy_to_base64
    enc_lo = float(np.percentile(sampled, 0.5))
    enc_hi = float(np.percentile(sampled, 99.5))
    enc_range = enc_hi - enc_lo
    if enc_range <= 0:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    # Tighter auto-scale window within the encoded range
    p5 = float(np.percentile(sampled, 5))
    p95 = float(np.percentile(sampled, 95))

    vmin_pct = max(0.0, 100.0 * (p5 - enc_lo) / enc_range)
    vmax_pct = min(100.0, 100.0 * (p95 - enc_lo) / enc_range)

    return {"vmin_pct": round(vmin_pct, 2), "vmax_pct": round(vmax_pct, 2)}


def numpy_to_base64(
    arr: np.ndarray,
    format: str = "png",
    compress_level: int = 1,
    jpeg_quality: int = 85,
    vmin: float = None,
    vmax: float = None,
) -> str:
    """Convert a numpy array to a base64 encoded image string.

    For high-bit-depth images (12/16-bit), applies a sqrt transform before
    the linear percentile clip.  This is the variance-stabilising transform
    for Poisson (photon-counting) noise and compresses the dynamic range so
    that dim PIV particles are visible in the 8-bit output without affecting
    well-exposed images (sqrt is nearly linear for large values).

    When vmin/vmax are not provided, automatic 0.5th–99.5th percentile
    clipping is applied on the sqrt-transformed data.  This rejects hot
    pixels and sensor defects while preserving the full particle signal.

    Args:
        arr: Input numpy array
        format: Image format - "png" or "jpeg"
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
        jpeg_quality: JPEG quality (1-95). Higher = better quality, larger files.
        vmin: If provided, clip sqrt-domain values below this before normalising
              to 0-255. When omitted, the 0.5th percentile is used.
        vmax: If provided, clip sqrt-domain values above this before normalising
              to 0-255. When omitted, the 99.5th percentile is used.
    """
    if arr.dtype != np.uint8:
        # Apply sqrt for variance-stabilised normalisation of photon-count data.
        # This dramatically improves contrast for low-count PIV images (12/16-bit
        # cameras using <10% of sensor range) while being nearly invisible on
        # well-exposed images where sqrt(x) ≈ linear for large x.
        a = np.sqrt(np.maximum(arr.astype(np.float32, copy=False), 0))
        if a.size:
            if vmin is not None and vmax is not None:
                mn, mx = float(vmin), float(vmax)
            else:
                # Auto percentile clipping in sqrt domain
                total = a.size
                if total > 4_000_000:
                    stride = max(2, int(np.sqrt(total / 1_000_000)))
                    sampled = a[::stride, ::stride].ravel()
                else:
                    sampled = a.ravel()
                mn = float(np.percentile(sampled, 0.5))
                mx = float(np.percentile(sampled, 99.5))
            if mx > mn:
                a = (255 * (np.clip(a, mn, mx) - mn) / (mx - mn)).astype(np.uint8)
            else:
                logger.debug("Flat image (min==max); producing black output")
                a = np.zeros_like(a, dtype=np.uint8)
        else:
            logger.debug("Empty array; substituting 1x1 black pixel")
            a = np.zeros((1, 1), dtype=np.uint8)
        arr = a

    img = Image.fromarray(arr)
    buf = BytesIO()

    if format.lower() == "jpeg":
        # PIL handles 'L' (grayscale) mode JPEG natively — no need to convert to RGB.
        # Skipping the conversion avoids tripling the data size before compression.
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        # Default to PNG
        img.save(buf, format="PNG", compress_level=compress_level, optimize=False)

    return base64.b64encode(buf.getvalue()).decode("utf-8")
