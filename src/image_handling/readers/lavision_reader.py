import os
from typing import Optional
from loguru import logger
import numpy as np

from . import register_reader


def read_lavision_im7(
    file_path: str, camera_no: int = 1, frames: int = 2
) -> np.ndarray:
    """Read LaVision .im7 files.
    
    LaVision .im7 files store all cameras in a single file per time instance.
    Each file contains frame pairs (A and B) for all cameras.
    
    Structure: For N cameras, the file contains 2*N frames in sequence:
    - Frames 0,1: Camera 1, frames A and B
    - Frames 2,3: Camera 2, frames A and B
    - etc.
    
    Args:
        file_path: Path to the .im7 file
        camera_no: Camera number (1-based indexing)
        frames: Number of frames to read (typically 2 for PIV)
        
    Returns:
        np.ndarray: Array of shape (frames, H, W) containing the image data
    """
    import sys
    if sys.platform == "darwin":
        raise ImportError(
            "lvpyio is not shipped or supported on macOS (darwin). Please use a supported platform for LaVision .im7 reading."
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
    start_frame = (camera_no - 1) * 2
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


def read_lavision_pair(file_path: str, camera_no: int = 1) -> np.ndarray:
    """Read LaVision .im7 file and return as frame pair.
    
    Args:
        file_path: Path to the .im7 file (contains all cameras for one time instance)
        camera_no: Camera number (1-based) to extract from the file
        
    Returns:
        np.ndarray: Array of shape (2, H, W) containing frame A and B for the specified camera
    """
    return read_lavision_im7(file_path, camera_no, frames=2)


def read_lavision_ims(file_path: str, camera_no: Optional[int] = None, im_no: Optional[int] = None) -> np.ndarray:
    """Read LaVision images from a .set file.
    
    LaVision .set files contain all cameras and frames in a single container.
    This function reads the set file and extracts the appropriate frames for the
    specified camera and image number.
    
    Args:
        file_path: Path to the .set file
        camera_no: Camera number (1-based). If None, extracted from file_path (legacy)
        im_no: Image number (1-based). If None, extracted from file_path (legacy)
        
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
        im = set_file[im_no - 1]  # 0-based indexing in Python
    except Exception as e:
        raise RuntimeError(f"Failed to read set file from {set_file_path}: {e}")
    
    # Extract frames for this camera
    data = np.zeros((2, *im.frames[0].components["PIXEL"].planes[0].shape), dtype=np.float64)
    
    for j in range(2):
        # Frame indexing: 2*cameraNo-(2-j)
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
