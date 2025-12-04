import os
from typing import Optional
from loguru import logger
import numpy as np

from . import register_reader


def read_lavision_im7(
    file_path: str,
    camera_no: int = 1,
    frames: int = 2,
    frames_per_camera: int = 2
) -> np.ndarray:
    """Read LaVision .im7 files.

    LaVision .im7 files store all cameras in a single file per time instance.

    Standard PIV mode (frames_per_camera=2):
        Each file contains frame pairs (A and B) for all cameras.
        Structure: For N cameras, the file contains 2*N frames:
        - Frames 0,1: Camera 1, frames A and B
        - Frames 2,3: Camera 2, frames A and B
        - etc.

    Single frame mode (frames_per_camera=1):
        Each file contains one frame per camera (e.g., time-resolved/snapshot).
        Structure: For N cameras, the file contains N frames:
        - Frame 0: Camera 1
        - Frame 1: Camera 2
        - etc.

    Args:
        file_path: Path to the .im7 file
        camera_no: Camera number (1-based indexing)
        frames: Number of frames to read from this camera
        frames_per_camera: Frames stored per camera in the file (2=PIV, 1=single)

    Returns:
        np.ndarray: Array of shape (frames, H, W) containing the image data
    """
    import sys
    if sys.platform == "darwin":
        raise ImportError(
            "lvpyio is not supported on macOS. "
            "Please use Windows for LaVision .im7 reading."
        )
    try:
        import lvpyio as lv
    except ImportError:
        raise ImportError(
            "LaVision library not available. Please install."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Read the buffer as a generator
    buffer = lv.read_buffer(file_path)

    # Calculate which frames we need for this camera
    # For standard PIV: camera 1 -> frames 0,1; camera 2 -> frames 2,3
    # For single frame: camera 1 -> frame 0; camera 2 -> frame 1
    start_frame = (camera_no - 1) * frames_per_camera
    end_frame = start_frame + frames
    
    # Iterate through the generator, only processing the frames we need
    data = None
    for idx, img in enumerate(buffer):
        if idx < start_frame:
            # Skip frames before our camera
            continue
        elif idx < end_frame:
            # This is one of our frames
            if data is None:
                # Initialize array on first needed frame
                height, width = img.components["PIXEL"].planes[0].shape
                data = np.zeros((frames, height, width), dtype=np.float64)
            
            i_scale = img.scales.i.slope
            i_offset = img.scales.i.offset
            u_arr = img.components["PIXEL"].planes[0] * i_scale + i_offset
            data[idx - start_frame, :, :] = u_arr
        else:
            # We've got all our frames, stop iterating
            break
    
    if data is None:
        raise ValueError(f"Camera {camera_no} not found in file {file_path}")
    
    return data.astype(np.float32)


def read_lavision_pair(file_path: str, camera_no: int = 1, **kwargs) -> np.ndarray:
    """Read LaVision .im7 file and return as frame pair.

    Args:
        file_path: Path to the .im7 file (contains all cameras for one time instance)
        camera_no: Camera number (1-based) to extract from the file
        **kwargs: Additional args for time-resolved mode:
            - frames: Number of frames to read (default 2)
            - frames_per_camera: Frames stored per camera in file (default 2)

    Returns:
        np.ndarray: Array of shape (frames, H, W) containing image data
    """
    frames = kwargs.get('frames', 2)
    frames_per_camera = kwargs.get('frames_per_camera', 2)
    return read_lavision_im7(file_path, camera_no, frames=frames, frames_per_camera=frames_per_camera)


def read_lavision_ims(
    file_path: str,
    camera_no: Optional[int] = None,
    im_no: Optional[int] = None,
    time_resolved: bool = False,
    im_no_b: Optional[int] = None
) -> np.ndarray:
    """Read LaVision images from a .set file.

    LaVision .set files contain all cameras and frames in a single container.
    This function supports two modes:

    Pre-paired mode (time_resolved=False):
        Structure: set_file[im_no].frames[2*camera + 0/1]
        Each entry contains A+B frame pairs for all cameras.
        im_no specifies which entry to read from.

    Time-resolved mode (time_resolved=True):
        Structure: set_file[time_step].frames[camera_idx]
        Each entry has ONE frame per camera (0-indexed camera).
        Reads from im_no (frame A) and im_no_b (frame B).
        Used for sequential PIV pairing: t1+t2, t2+t3, etc.

    Args:
        file_path: Path to the .set file
        camera_no: Camera number (1-based). If None, extracted from file_path (legacy)
        im_no: Image/time step number for frame A (1-based)
        time_resolved: If True, read single frames from two entries
        im_no_b: Image/time step number for frame B (1-based, only used when time_resolved=True)

    Returns:
        np.ndarray: Array of shape (2, H, W) containing frame A and B
    """
    import sys
    from pathlib import Path

    if sys.platform == "darwin":
        raise ImportError(
            "LaVision libraries are not supported on macOS (darwin). Please use a supported platform."
        )

    try:
        import lvpyio as lv
    except ImportError:
        raise ImportError(
            "LaVision library not available. Please install lvpyio."
        )

    path = Path(file_path)

    # For .set files, camera_no and im_no must be provided
    if path.suffix.lower() == '.set' or (camera_no is not None and im_no is not None):
        # Modern format: file_path is the .set file
        set_file_path = file_path
        if camera_no is None or im_no is None:
            raise ValueError("camera_no and im_no must be provided for .set files")
    else:
        # Legacy path parsing for backward compatibility
        # Extract camera number from path (e.g., "Cam1" -> 1)
        if camera_no is None:
            camera_match = None
            for part in path.parts:
                if part.startswith("Cam") and part[3:].isdigit():
                    camera_match = int(part[3:])
                    break
            if camera_match is None:
                raise ValueError(f"Could not extract camera number from path: {file_path}")
            camera_no = camera_match

        # Extract image number from filename
        if im_no is None:
            stem = path.stem
            if stem.isdigit():
                im_no = int(stem)
            else:
                raise ValueError(f"Could not extract image number from filename: {path.name}")

        # Source directory is typically the parent of the CamX directory
        source_dir = path.parent.parent
        set_file_path = str(source_dir)

    if not Path(set_file_path).exists():
        raise FileNotFoundError(f"Set file path not found: {set_file_path}")

    # Read the set file
    try:
        set_file = lv.read_set(set_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read set file from {set_file_path}: {e}")

    if time_resolved:
        # Time-resolved mode: read single frame from two different time steps
        # Structure: set_file[time_step].frames[camera_idx] where camera_idx is 0-based
        if im_no_b is None:
            raise ValueError("im_no_b must be provided for time_resolved mode")

        camera_idx = camera_no - 1  # Convert to 0-based

        try:
            im_a = set_file[im_no - 1]  # 0-based indexing
            im_b = set_file[im_no_b - 1]
        except IndexError as e:
            raise ValueError(f"Time step index out of range: {e}")

        frame_a = im_a.frames[camera_idx]
        frame_b = im_b.frames[camera_idx]

        # Get shape from first frame
        shape = frame_a.components["PIXEL"].planes[0].shape
        data = np.zeros((2, *shape), dtype=np.float64)

        # Apply scaling to both frames
        for i, frame in enumerate([frame_a, frame_b]):
            i_scale = frame.scales.i.slope
            i_offset = frame.scales.i.offset
            u_arr = frame.components["PIXEL"].planes[0] * i_scale + i_offset
            data[i, :, :] = u_arr

    else:
        # Pre-paired mode: read A+B frames from single entry
        # Structure: set_file[im_no].frames[2*(camera_no-1)] and frames[2*(camera_no-1)+1]
        try:
            im = set_file[im_no - 1]  # 0-based indexing in Python
        except IndexError as e:
            raise ValueError(f"Image number {im_no} out of range: {e}")

        # Extract frames for this camera
        data = np.zeros((2, *im.frames[0].components["PIXEL"].planes[0].shape), dtype=np.float64)

        for j in range(2):
            # Frame indexing: 2*cameraNo-(2-j) = 2*(camera_no-1) + j
            frame_idx = 2 * camera_no - (2 - j)
            frame = im.frames[frame_idx]

            # Apply scaling
            i_scale = frame.scales.i.slope
            i_offset = frame.scales.i.offset
            u_arr = frame.components["PIXEL"].planes[0] * i_scale + i_offset

            data[j, :, :] = u_arr

    set_file.close()
    return data.astype(np.float32)


def read_lavision_ims_pair(file_path: str, **kwargs) -> np.ndarray:
    """Read LaVision .set file and return as frame pair."""
    return read_lavision_ims(file_path, **kwargs)


register_reader([".im7"], read_lavision_pair)
register_reader([".set"], read_lavision_ims_pair)
