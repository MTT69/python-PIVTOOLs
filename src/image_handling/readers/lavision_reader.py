import os

import numpy as np

from . import register_reader


def read_lavision_im7(
    file_path: str, camera_no: int = 1, frames: int = 2
) -> np.ndarray:
    """Read LaVision .im7 files."""
    import sys

    if sys.platform == "darwin":
        raise ImportError(
            "lvpyio is not shipped or supported on macOS (darwin). Please use a supported platform for LaVision .im7 reading."
        )
    try:
        import lvpyio as lv
    except ImportError:
        raise ImportError(
            "LaVision library not available. Please install lavisionlib or equivalent."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    # Read the buffer
    p1 = lv.read_buffer(file_path)
    im_list = list(p1)
    first_img = im_list[(camera_no - 1) * 2]
    height, width = first_img.components["PIXEL"].planes[0].shape
    data = np.zeros((frames, height, width), dtype=np.float64)
    for j in range(frames):
        img_idx = int(camera_no - 1) * 2 + j
        img = im_list[img_idx]
        i_scale = img.scales.i.slope
        i_offset = img.scales.i.offset
        u_arr = img.components["PIXEL"].planes[0] * i_scale + i_offset
        data[j, :, :] = u_arr
    del p1
    return data.astype(np.float32)


def read_lavision_pair(file_path: str, camera_no: int = 1) -> np.ndarray:
    """Read LaVision .im7 file and return as frame pair."""
    return read_lavision_im7(file_path, camera_no, frames=2)


register_reader([".im7"], read_lavision_pair)
