"""Calibration image loading utilities.

This module provides functions for loading and validating calibration images
using the centralized image handling system. It supports all image formats
including standard formats (TIFF, PNG, JPEG) and container formats
(.set, .im7, .cine).

Key Functions:
- read_calibration_image: Read a single calibration image
- validate_calibration_images: Validate calibration images exist and are readable
- get_calibration_frame_count: Auto-detect number of calibration images
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import Config
from .load_images import read_single_frame
from .path_utils import build_calibration_camera_path, resolve_file_path, validate_images_generic


def read_calibration_image(
    idx: int,
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    subfolder: Optional[str] = None,
    image_type: Optional[str] = None,
) -> np.ndarray:
    """Read a single calibration image.

    This function uses the unified read_single_frame() core reader,
    eliminating duplicated format handling. It handles all image formats:
    - Standard formats (.tif, .png, .jpg) with numbered patterns
    - LaVision .set containers (all cameras in one file)
    - LaVision .im7 files (one per frame)
    - Phantom .cine video files (one per camera)

    Parameters
    ----------
    idx : int
        Image index (1-based unless calibration_zero_based_indexing is True)
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object with calibration settings
    source_path_idx : int, optional
        Index into source_paths list, defaults to 0
    image_format : str, optional
        Override for calibration_image_format from config
    subfolder : str, optional
        Override for calibration_subfolder from config
    image_type : str, optional
        Override for calibration_image_type from config

    Returns
    -------
    np.ndarray
        Image data as 2D array (H, W)

    Raises
    ------
    FileNotFoundError
        If the image file does not exist
    ValueError
        If the image cannot be read
    """
    # Use passed values or fall back to config
    cal_image_type = image_type if image_type is not None else config.calibration_image_type
    fmt = image_format if image_format is not None else config.calibration_image_format

    # Build calibration path using shared utility
    camera_path = build_calibration_camera_path(config, source_path_idx, camera, subfolder)

    # Resolve file path based on image type
    file_path = resolve_file_path(
        camera_path=camera_path,
        camera=camera,
        frame_idx=idx,
        format_pattern=fmt,
        image_type=cal_image_type,
        zero_based_indexing=config.calibration_zero_based_indexing,
    )

    # For IM7 with camera subfolders, each file is single-camera - don't pass camera_no
    if cal_image_type == "lavision_im7" and config.calibration_use_camera_subfolders:
        from .load_images import read_image
        img = read_image(str(file_path))
        if img.ndim == 3:
            img = img[0]  # Extract single frame
        return img

    # Use the unified core reader (passes camera_no for multi-camera containers)
    return read_single_frame(
        file_path=file_path,
        camera=camera,
        frame_idx=idx,
        image_type=cal_image_type,
        time_resolved=True,  # Calibration always reads single frames
    )


def validate_calibration_images(
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    num_images: Optional[int] = None,
    subfolder: Optional[str] = None,
    image_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate calibration images exist and are readable.

    Uses the generic validate_images_generic() function with calibration-specific
    parameters.

    Parameters
    ----------
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object with calibration settings
    source_path_idx : int, optional
        Index into source_paths list, defaults to 0
    image_format : str, optional
        Override for calibration_image_format from config
    num_images : int, optional
        Override for calibration_image_count from config
    subfolder : str, optional
        Override for calibration_subfolder from config
    image_type : str, optional
        Override for calibration_image_type from config

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - Overall validation result
        - found_count: int or 'container' - Number of files found
        - expected_count: int - Expected number of files
        - camera_path: str - Path to camera directory
        - first_image_preview: str - Base64 PNG of first image (if valid)
        - image_size: tuple - (width, height) of images
        - sample_files: list - Sample of matching filenames
        - format_detected: str - Detected file format
        - error: str or None - Error message if validation failed
        - suggested_pattern: str or None - Suggested pattern if files don't match
    """
    # Use passed values or fall back to config
    cal_image_type = image_type if image_type is not None else config.calibration_image_type
    fmt = image_format if image_format is not None else config.calibration_image_format
    expected_count = num_images if num_images is not None else config.calibration_image_count
    cal_subfolder = subfolder

    # Build calibration path using shared utility
    camera_path = build_calibration_camera_path(config, source_path_idx, camera, cal_subfolder)

    # Create a frame reader function for preview generation
    def read_frame(idx: int) -> np.ndarray:
        return read_calibration_image(
            idx, camera, config, source_path_idx,
            image_format=fmt, subfolder=cal_subfolder, image_type=cal_image_type
        )

    # Use the generic validator
    return validate_images_generic(
        camera_path=camera_path,
        camera=camera,
        image_format=fmt,
        image_type=cal_image_type,
        expected_count=expected_count,
        zero_based_indexing=config.calibration_zero_based_indexing,
        read_frame_fn=read_frame,
    )


def get_calibration_frame_count(
    camera: int,
    config: Config,
    source_path_idx: int = 0
) -> int:
    """Auto-detect number of calibration images from directory.

    Counts matching files or returns container frame count.

    Parameters
    ----------
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object with calibration settings
    source_path_idx : int, optional
        Index into source_paths list, defaults to 0

    Returns
    -------
    int
        Number of calibration images found
    """
    # Build calibration path using shared utility
    camera_path = build_calibration_camera_path(config, source_path_idx, camera)

    image_type = config.calibration_image_type
    fmt = config.calibration_image_format

    if not camera_path.exists():
        return 0

    if image_type == "lavision_set":
        # For .set files, we would need to read the file to get count
        # Return configured count as fallback
        return config.calibration_image_count

    elif image_type == "cine":
        # Get frame count from .cine file
        try:
            from .readers.cine_reader import get_cine_frame_count
            if "%" in fmt:
                cine_filename = fmt % camera
            else:
                cine_filename = fmt
            cine_path = camera_path / cine_filename
            if cine_path.exists():
                return get_cine_frame_count(str(cine_path))
        except Exception:
            pass
        return config.calibration_image_count

    elif image_type == "lavision_im7":
        pattern = fmt.replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
        return len(list(camera_path.glob(pattern)))

    else:
        # Standard formats
        pattern = fmt.replace("%05d", "*").replace("%04d", "*").replace("%03d", "*").replace("%d", "*")
        return len(list(camera_path.glob(pattern)))


