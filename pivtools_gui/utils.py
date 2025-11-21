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


def camera_folder(camera: Union[str, int]) -> str:
    """
    Return canonical folder name (e.g. Cam1) for a camera reference.

    DEPRECATED: This function returns hardcoded 'CamN' folders and does not respect
    custom camera_subfolders configuration. Use config.get_camera_folder() instead.

    This function is kept for backward compatibility only.
    """
    return f"Cam{camera_number(camera)}"


def numpy_to_png_base64(arr: np.ndarray, compress_level: int = 1) -> str:
    """Convert a numpy array (uint8 or convertible) to a base64 PNG string.

    Args:
        arr: Input numpy array
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
                       Default is 1 for speed. Use 6 for better compression.
    """
    return numpy_to_base64(arr, format="png", compress_level=compress_level)


def numpy_to_base64(arr: np.ndarray, format: str = "png", compress_level: int = 1, jpeg_quality: int = 85) -> str:
    """Convert a numpy array to a base64 encoded image string.

    Args:
        arr: Input numpy array
        format: Image format - "png" or "jpeg"
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
        jpeg_quality: JPEG quality (1-95). Higher = better quality, larger files.
    """
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32, copy=False)
        if a.size:
            mn = float(a.min())
            mx = float(a.max())
            if mx > mn:
                a = (255 * (a - mn) / (mx - mn)).astype(np.uint8)
            else:
                # Completely flat -> black; log once so caller can trace
                logger.debug("Flat image (min==max); producing black output")
                a = np.zeros_like(a, dtype=np.uint8)
        else:
            logger.debug("Empty array; substituting 1x1 black pixel")
            a = np.zeros((1, 1), dtype=np.uint8)
        arr = a

    img = Image.fromarray(arr)
    buf = BytesIO()

    if format.lower() == "jpeg":
        # Convert grayscale to RGB for JPEG (JPEG doesn't support all grayscale modes well)
        if img.mode == "L":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        # Default to PNG
        img.save(buf, format="PNG", compress_level=compress_level, optimize=False)

    return base64.b64encode(buf.getvalue()).decode("utf-8")
