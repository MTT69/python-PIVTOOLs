"""LaVision .im7 and .set file readers.

Thin wrappers over the pure-Python im7_reader and set_reader modules.
No dependency on lvpyio — works on macOS, Linux, and Windows.
"""
import os
from typing import Optional

import numpy as np

from .im7_reader import read_im7_camera
from .set_reader import read_set_pair


def read_lavision_im7(
    file_path: str,
    camera_no: int = 1,
    frames: int = 2,
    frames_per_camera: int = 2,
) -> np.ndarray:
    """Read LaVision .im7 files.

    Args:
        file_path: Path to the .im7 file
        camera_no: Camera number (1-based indexing)
        frames: Number of frames to return from this camera
        frames_per_camera: Frames stored per camera in the file (2=PIV, 1=single)

    Returns:
        np.ndarray: Array of shape (frames, H, W) float32
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    data = read_im7_camera(file_path, camera_no, frames_per_camera)
    return data[:frames]


def read_lavision_pair(file_path: str, camera_no: int = 1, **kwargs) -> np.ndarray:
    """Read LaVision .im7 file and return as frame pair.

    Args:
        file_path: Path to the .im7 file
        camera_no: Camera number (1-based)
        **kwargs: frames (default 2), frames_per_camera (default 2)

    Returns:
        np.ndarray: Array of shape (frames, H, W) float32
    """
    frames = kwargs.get("frames", 2)
    frames_per_camera = kwargs.get("frames_per_camera", 2)
    return read_lavision_im7(
        file_path, camera_no, frames=frames, frames_per_camera=frames_per_camera
    )


def read_lavision_ims(
    file_path: str,
    camera_no: Optional[int] = None,
    im_no: Optional[int] = None,
    time_resolved: bool = False,
    im_no_b: Optional[int] = None,
) -> np.ndarray:
    """Read LaVision images from a .set file.

    Pre-paired mode (time_resolved=False):
        Reads frames[2*(camera_no-1)] and frames[2*(camera_no-1)+1]
        from entry im_no.

    Time-resolved mode (time_resolved=True):
        Reads frames[camera_no-1] from entries im_no and im_no_b.

    Args:
        file_path: Path to the .set file
        camera_no: Camera number (1-based)
        im_no: Image/entry number (1-based)
        time_resolved: If True, read single frames from two entries
        im_no_b: Second entry number (1-based, time_resolved only)

    Returns:
        np.ndarray: Array of shape (2, H, W) float32
    """
    if camera_no is None or im_no is None:
        raise ValueError("camera_no and im_no must be provided for .set files")

    return read_set_pair(
        file_path,
        camera_no=camera_no,
        im_no=im_no,
        time_resolved=time_resolved,
        im_no_b=im_no_b,
    )


def read_lavision_ims_pair(file_path: str, **kwargs) -> np.ndarray:
    """Read LaVision .set file and return as frame pair."""
    return read_lavision_ims(file_path, **kwargs)
