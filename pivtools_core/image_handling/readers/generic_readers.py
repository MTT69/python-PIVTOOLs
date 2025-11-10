import os

import numpy as np


def read_tiff(file_path: str) -> np.ndarray:
    """Read TIFF images using tifffile."""
    import tifffile

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return tifffile.imread(file_path)


def read_png_jpeg(file_path: str) -> np.ndarray:
    """Read PNG/JPEG images using PIL or opencv."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        from PIL import Image

        img = Image.open(file_path)
        return np.array(img)
    except ImportError:
        try:
            import cv2

            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Image file could not be read: {file_path}")
            return img
        except ImportError:
            raise ImportError(
                "Either PIL or opencv-python is required for PNG/JPEG support"
            )


def read_raw(file_path: str) -> np.ndarray:
    """Read RAW images using rawpy."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        import rawpy

        with rawpy.imread(file_path) as raw:
            return raw.postprocess()
    except ImportError:
        raise ImportError("rawpy is required for RAW image support")
