import os
import logging
from pathlib import Path

import cv2
import numpy as np


def read_tiff(file_path: str) -> np.ndarray:
    """Read TIFF images using tifffile."""
    import tifffile

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    img = tifffile.imread(file_path)
    if img.ndim > 2 and img.shape[-1] > 1:
        logging.debug(f"Converting color TIFF image to grayscale: {Path(file_path).name}")
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = np.mean(img, axis=-1).astype(img.dtype)
    return img


def read_png_jpeg(file_path: str) -> np.ndarray:
    """Read PNG/JPEG images using PIL or opencv."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        from PIL import Image

        img = Image.open(file_path)
        if img.mode in ("RGB", "RGBA", "P", "L") and img.mode != "L":
            logging.debug(f"Converting color PNG/JPEG image to grayscale: {Path(file_path).name}")
            img = img.convert("L")
        img_array = np.array(img)
        return img_array
    except ImportError:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image file could not be read: {file_path}")
        if img.ndim > 2 and img.shape[-1] > 1:
            logging.debug(f"Converting color PNG/JPEG image to grayscale: {Path(file_path).name}")
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = np.mean(img, axis=-1).astype(img.dtype)
        return img


def read_raw(file_path: str) -> np.ndarray:
    """Read RAW images using rawpy."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        import rawpy

        with rawpy.imread(file_path) as raw:
            img = raw.postprocess()
            if img.ndim > 2 and img.shape[-1] > 1:
                logging.debug(f"Converting color RAW image to grayscale: {Path(file_path).name}")
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    img = np.mean(img, axis=-1).astype(img.dtype)
            return img
    except ImportError:
        raise ImportError("rawpy is required for RAW image support")
