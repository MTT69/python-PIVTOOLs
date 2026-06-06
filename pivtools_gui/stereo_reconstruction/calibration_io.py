"""
calibration_io.py

Shared I/O utilities and constants for all calibration modules (planar dotboard,
planar charuco, stereo dotboard, stereo charuco). Single canonical source.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


# Standard ArUco dictionaries mapping — used by both planar and stereo ChArUco
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def is_container_format(config, file_pattern: str) -> bool:
    """Check if file pattern is a container format (.set, .cine).

    Uses config.calibration_is_container_format when available, otherwise
    falls back to pattern-based detection.

    Note: IM7 files with % patterns (e.g., B%05d.im7) are individual files,
    not containers. Only treat as container if it's a single file without
    a printf-style pattern.
    """
    if config is not None:
        return config.calibration_is_container_format

    # Fallback: pattern-based detection
    pattern_lower = file_pattern.lower()
    if "%" in file_pattern:
        return False
    return '.set' in pattern_lower or '.cine' in pattern_lower


def read_calibration_image_with_fallback(
    img_path,
    camera: int,
    img_index: int,
    config,
    source_path_idx: int,
    file_pattern: str,
) -> Optional[np.ndarray]:
    """Read calibration image using core reader with CLI fallback.

    When config is available, uses the unified core reader which handles
    all formats consistently. Falls back to direct reading for CLI mode.

    Returns the image in its native dtype (float32/uint16/uint8) to
    preserve full dynamic range for flat-fielding in grid_detection.py.

    Parameters
    ----------
    img_path : str or Path or None
        Path to image file (used for fallback/CLI mode)
    camera : int
        Camera number (1-based)
    img_index : int
        Image index (1-based)
    config : Config or None
        Configuration object
    source_path_idx : int
        Index into config.calibration_sources
    file_pattern : str
        Image file naming pattern

    Returns
    -------
    np.ndarray or None
        Image as 2D array in native dtype, or None if reading failed
    """
    if config is not None:
        from pivtools_core.image_handling.calibration_loader import (
            read_calibration_image as core_read_calibration_image,
        )
        try:
            img = core_read_calibration_image(
                idx=img_index,
                camera=camera,
                config=config,
                source_path_idx=source_path_idx,
                normalize_uint8=False,
            )
            if img is not None and img.ndim == 3:
                img = img[0]  # Extract single frame if needed
            logger.info(f"Read image {img_index} shape: {img.shape}, dtype: {img.dtype}")
            return img
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Error reading image {img_index}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error reading image {img_index}: {e}")
            return None

    # Fallback: direct reading for CLI mode without config
    return read_calibration_image_direct(img_path, camera, img_index, file_pattern, config)


def read_calibration_image_direct(
    img_path,
    camera: int,
    img_index: int,
    file_pattern: str,
    config,
) -> Optional[np.ndarray]:
    """Direct image reading fallback for CLI mode without config.

    Returns the image in its native dtype to preserve full dynamic range
    for flat-fielding in grid_detection.py.
    """
    from pivtools_core.image_handling.load_images import read_image

    try:
        img_path_lower = str(img_path).lower()
        container = is_container_format(config, file_pattern)

        if container:
            if '.set' in img_path_lower:
                img = read_image(str(img_path), camera_no=camera, im_no=img_index)
            elif '.cine' in img_path_lower:
                from pivtools_core.image_handling.readers.cine_reader import read_cine_single
                img = read_cine_single(str(img_path), idx=img_index)
            else:
                img = read_image(str(img_path))
        elif '.im7' in img_path_lower:
            img = read_image(str(img_path), camera_no=camera, frames=1, frames_per_camera=1)
        else:
            img = read_image(str(img_path))

        if img is None:
            return None

        logger.info(f"Read image shape: {img.shape}, dtype: {img.dtype}")

        # Only normalize bool (binary masks) — preserve all other dtypes
        if img.dtype == np.bool_:
            img = img.astype(np.uint8) * 255

        return img
    except Exception as e:
        logger.warning(f"Error reading image {img_path}: {e}")
        return None


def find_calibration_images(
    cam_input_dir: Path,
    file_pattern: str,
    config=None,
) -> List[Path]:
    """Find all calibration images matching the pattern.

    Handles container formats, glob patterns, numbered patterns, and single files.
    Canonical source from stereo_calibration_base (most complete set of branches).

    Parameters
    ----------
    cam_input_dir : Path
        Directory to search
    file_pattern : str
        File naming pattern
    config : Config or None
        Configuration object (for container format detection)
    """
    if is_container_format(config, file_pattern):
        container_file = cam_input_dir / file_pattern
        if container_file.exists():
            return [container_file]
        return []

    # Glob pattern matching
    if "*" in file_pattern or "?" in file_pattern:
        return sorted(cam_input_dir.glob(file_pattern))

    # Numbered pattern (e.g., "calib%05d.tif")
    if "%" in file_pattern:
        files = []
        i = 1
        while True:
            try:
                filename = file_pattern % i
            except TypeError as e:
                raise ValueError(
                    f"file_pattern {file_pattern!r} contains '%' but is not a numbered format "
                    f"(expected e.g. 'calib%05d.tif'): {e}"
                ) from e
            filepath = cam_input_dir / filename
            if filepath.exists():
                files.append(filepath)
                i += 1
            else:
                break
        return files

    # Single file
    single = cam_input_dir / file_pattern
    return [single] if single.exists() else []


def get_camera_input_dir(
    cam_num: int,
    config,
    source_path_idx: int,
    source_dir: Path,
    calibration_input_path: Optional[Path] = None,
) -> Path:
    """Get the input directory for calibration images.

    Uses build_calibration_camera_path for path resolution from
    calibration_sources config.

    Parameters
    ----------
    cam_num : int
        Camera number (1-based)
    config : Config or None
        Configuration object
    source_path_idx : int
        Index into config.calibration_sources
    source_dir : Path
        Fallback source directory (used if config is None)
    calibration_input_path : Path or None
        Explicit override path (used by ChArUco CLI)
    """
    if calibration_input_path is not None:
        return calibration_input_path

    if config is not None:
        from pivtools_core.image_handling.path_utils import build_calibration_camera_path
        return build_calibration_camera_path(config, source_path_idx=source_path_idx, camera=cam_num)

    return source_dir


def create_charuco_detector(
    squares_h: int,
    squares_v: int,
    square_size: float,
    marker_ratio: float,
    aruco_dict_name: str,
) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    """Create ChArUco board and detector.

    Parameters
    ----------
    squares_h : int
        Number of squares horizontally
    squares_v : int
        Number of squares vertically
    square_size : float
        Physical square size in meters
    marker_ratio : float
        Ratio of marker size to square size
    aruco_dict_name : str
        ArUco dictionary name (e.g., "DICT_4X4_1000")

    Returns
    -------
    tuple
        (CharucoBoard, CharucoDetector)
    """
    marker_size = square_size * marker_ratio
    dict_id = ARUCO_DICT_MAP.get(aruco_dict_name, cv2.aruco.DICT_4X4_1000)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        (squares_h, squares_v),
        square_size,
        marker_size,
        dictionary,
    )

    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters(),
    )

    return board, detector
