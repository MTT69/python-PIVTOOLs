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
from .path_utils import build_calibration_camera_path, format_to_glob, resolve_file_path, validate_images_generic


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image array to uint8 for OpenCV detection.

    Parameters
    ----------
    img : np.ndarray
        Input image of any dtype

    Returns
    -------
    np.ndarray
        Image normalized to uint8 (0-255)
    """
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    return img.astype(np.uint8)


def read_calibration_image(
    idx: int,
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    image_type: Optional[str] = None,
    normalize_uint8: bool = True,
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
        Index into calibration_sources list, defaults to 0
    image_format : str, optional
        Override for calibration_image_format from config
    image_type : str, optional
        Override for calibration_image_type from config
    normalize_uint8 : bool, optional
        If True, normalize output to uint8 for OpenCV detection (default True)

    Returns
    -------
    np.ndarray
        Image data as 2D array (H, W), normalized to uint8 if normalize_uint8=True

    Raises
    ------
    FileNotFoundError
        If the image file does not exist
    ValueError
        If the image cannot be read or calibration_sources not configured
    """
    # Use passed values or fall back to config
    cal_image_type = image_type if image_type is not None else config.calibration_image_type
    fmt = image_format if image_format is not None else config.calibration_image_format

    # Build calibration path using shared utility
    camera_path = build_calibration_camera_path(config, source_path_idx, camera)

    return read_calibration_frame_at(
        camera_path=camera_path,
        camera=camera,
        frame_idx=idx,
        image_format=fmt,
        image_type=cal_image_type,
        zero_based_indexing=config.calibration_zero_based_indexing,
        use_camera_subfolders=config.calibration_use_camera_subfolders,
        normalize_uint8=normalize_uint8,
        num_cameras=config.camera_count,
    )


def read_calibration_frame_at(
    camera_path: Path,
    camera: int,
    frame_idx: int,
    image_format: str,
    image_type: str,
    *,
    zero_based_indexing: bool = False,
    use_camera_subfolders: bool = False,
    normalize_uint8: bool = True,
    num_cameras: Optional[int] = None,
) -> np.ndarray:
    """Read one calibration frame from an ALREADY-RESOLVED camera path/container.

    The format-dispatch half of :func:`read_calibration_image`, factored out so callers
    that resolve the camera directory themselves can share it. The GUI/config path goes
    through ``read_calibration_image`` (resolves ``camera_path`` from ``config`` +
    ``source_path_idx``); the calibration CLI calls this directly with its
    ``--source``-derived directory, so both read the same formats (standard tif/png,
    LaVision ``.im7``/``.set``, Phantom ``.cine``) through one code path.

    Parameters mirror :func:`read_calibration_image` except ``camera_path`` is supplied
    directly (a directory for per-file formats, or the container file for ``.set``).
    """
    # Resolve file path based on image type.
    file_path = resolve_file_path(
        camera_path=camera_path,
        camera=camera,
        frame_idx=frame_idx,
        format_pattern=image_format,
        image_type=image_type,
        zero_based_indexing=zero_based_indexing,
    )

    # For IM7 with camera subfolders, each file is single-camera - don't pass camera_no.
    if image_type == "lavision_im7" and use_camera_subfolders:
        from .load_images import read_image
        img = read_image(str(file_path))
        if img.ndim == 3:
            img = img[0]  # Extract single frame
        return _normalize_to_uint8(img) if normalize_uint8 else img

    # For multi-camera .im7 (all cameras in one buffer), detect how many frames
    # each camera occupies so this camera's slice is located correctly — same
    # rule the PIV path uses. Only when the caller supplies a camera count;
    # otherwise (e.g. direct CLI use without one) keep the default stride of 1.
    fpc = 1
    if image_type == "lavision_im7" and num_cameras is not None:
        from .load_images import _detect_im7_frames_per_camera
        fpc = _detect_im7_frames_per_camera(Path(file_path), num_cameras)

    # Unified core reader (passes camera_no for multi-camera containers).
    img = read_single_frame(
        file_path=file_path,
        camera=camera,
        frame_idx=frame_idx,
        image_type=image_type,
        time_resolved=True,  # Calibration always reads single frames
        frames_per_camera=fpc,
    )

    return _normalize_to_uint8(img) if normalize_uint8 else img


def validate_calibration_images(
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    num_images: Optional[int] = None,
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
        Index into calibration_sources list, defaults to 0
    image_format : str, optional
        Override for calibration_image_format from config
    num_images : int, optional
        Override for calibration_image_count from config
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

    # Build calibration path using shared utility
    camera_path = build_calibration_camera_path(config, source_path_idx, camera)

    # Create a frame reader function for preview generation
    def read_frame(idx: int) -> np.ndarray:
        return read_calibration_image(
            idx, camera, config, source_path_idx,
            image_format=fmt, image_type=cal_image_type
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
        pattern = format_to_glob(fmt)
        return len(list(camera_path.glob(pattern)))

    else:
        # Standard formats
        pattern = format_to_glob(fmt)
        return len(list(camera_path.glob(pattern)))


