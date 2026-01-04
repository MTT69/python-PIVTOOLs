"""Shared path utilities and validation for image loading.

This module provides centralized path-building and validation functions used by both
PIV image loading (load_images.py) and calibration image loading
(calibration_loader.py), eliminating code duplication.
"""

import base64
import io
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config


def build_calibration_camera_path(
    config: "Config",
    source_path_idx: int = 0,
    camera: int = 1,
    subfolder_override: Optional[str] = None,
) -> Path:
    """Build the path to calibration images for a specific camera.

    Path structure depends on calibration_image_type, use_camera_subfolders,
    and calibration_path_order:

    path_order='camera_first' (default):
    - Container formats (.set, .cine): source_directory / calibration_subfolder
    - IM7 with use_camera_subfolders=False: source_directory / calibration_subfolder
    - IM7 with use_camera_subfolders=True: source_directory / camera_folder / calibration_subfolder
    - Standard formats: source_directory / camera_folder / calibration_subfolder

    path_order='calibration_first':
    - Container formats (.set, .cine): source_directory / calibration_subfolder
    - IM7 with use_camera_subfolders=False: source_directory / calibration_subfolder
    - IM7 with use_camera_subfolders=True: source_directory / calibration_subfolder / camera_folder
    - Standard formats: source_directory / calibration_subfolder / camera_folder

    Note: For .set PIV files, source_path is the file itself, so we use
    get_source_directory() to get the parent directory for calibration images.
    This allows calibration images to be in a different format (e.g., .tif).

    This is the single source of truth for calibration path building,
    used by calibration_loader.py and the Flask calibration views.

    Args:
        config: Configuration object with calibration settings
        source_path_idx: Index into source_paths list (default: 0)
        camera: Camera number (1-based, default: 1)
        subfolder_override: Override for calibration_subfolder from config.
                           If None, uses config.calibration_subfolder.

    Returns:
        Path: Full path to calibration image directory

    Examples:
        >>> # camera_first (default): source/Cam1/calibration/
        >>> path = build_calibration_camera_path(config, 0, 1)

        >>> # calibration_first: source/calibration/Cam1/
        >>> # (when config.calibration_path_order = 'calibration_first')
        >>> path = build_calibration_camera_path(config, 0, 1)
    """
    # Use get_source_directory() which returns parent for .set files
    # This allows calibration images to be in a different format than PIV
    source_dir = config.get_source_directory(source_path_idx)
    cal_image_type = config.calibration_image_type
    path_order = config.calibration_path_order
    subfolder = subfolder_override if subfolder_override is not None else config.calibration_subfolder

    # Container formats (SET, CINE) never use camera subfolders
    if cal_image_type in ("lavision_set", "cine"):
        camera_path = source_dir
        if subfolder:
            camera_path = camera_path / subfolder
        return camera_path

    # Determine if camera folder should be used
    # Standard and IM7 formats respect the use_camera_subfolders setting
    # Container formats (SET, CINE) are handled above and never use camera subfolders
    use_camera_folder = config.calibration_use_camera_subfolders

    # Get camera folder name
    camera_folder = config.get_calibration_camera_folder(camera) if use_camera_folder else ""

    # Build path based on order preference
    if path_order == "calibration_first":
        # source_dir / calibration_subfolder / camera_folder
        camera_path = source_dir
        if subfolder:
            camera_path = camera_path / subfolder
        if camera_folder:
            camera_path = camera_path / camera_folder
    else:
        # camera_first (default): source_dir / camera_folder / calibration_subfolder
        camera_path = source_dir
        if camera_folder:
            camera_path = camera_path / camera_folder
        if subfolder:
            camera_path = camera_path / subfolder

    return camera_path


def build_piv_camera_path(
    config: "Config",
    source_path_idx: int = 0,
    camera: int = 1,
) -> Path:
    """Build the path to PIV images for a specific camera.

    Path structure depends on image type and use_camera_subfolders setting:
    - .set files: source_path IS the .set file (return it directly)
    - .cine files: source_path directory (no camera subfolder)
    - IM7 with use_camera_subfolders=False: source_path (all cameras in file)
    - IM7 with use_camera_subfolders=True: source_path / camera_folder
    - Standard formats: source_path / camera_folder

    Args:
        config: Configuration object
        source_path_idx: Index into source_paths list (default: 0)
        camera: Camera number (1-based, default: 1)

    Returns:
        Path: Path to PIV image file (.set) or directory for other formats
    """
    source_path = config.source_paths[source_path_idx]
    image_type = config.image_type

    # SET: source_path IS the .set file - return it directly
    if image_type == "lavision_set":
        return source_path

    # CINE: source_path is directory containing .cine files
    if image_type == "cine":
        return source_path

    # IM7: check if using camera subfolders
    if image_type == "lavision_im7":
        if config.images_use_camera_subfolders:
            # Single-camera IM7 files in camera subdirectories
            camera_folder = config.get_camera_folder(camera)
            if camera_folder:
                return source_path / camera_folder
        # Multi-camera IM7 files (default): no subdirectory
        return source_path

    # Standard formats use camera subdirectories
    camera_folder = config.get_camera_folder(camera)
    if camera_folder:
        return source_path / camera_folder
    return source_path


def resolve_file_path(
    camera_path: Path,
    camera: int,
    frame_idx: int,
    format_pattern: str,
    image_type: str,
    zero_based_indexing: bool = False,
) -> Path:
    """Resolve the file path for a specific frame.

    Handles the different file path patterns for each image type:
    - lavision_set: camera_path IS the .set file (return directly)
    - lavision_im7: camera_path / (format_pattern % frame_idx)
    - cine: camera_path / (format_pattern % camera)
    - standard: camera_path / (format_pattern % frame_idx)

    Args:
        camera_path: Base path - for .set this IS the file, otherwise directory
        camera: Camera number (1-based)
        frame_idx: Frame index (1-based)
        format_pattern: File format pattern (e.g., "%05d.tif", "Camera%d.cine")
        image_type: One of "lavision_set", "lavision_im7", "cine", "standard"
        zero_based_indexing: Whether file indices are 0-based

    Returns:
        Path: Full path to the image file
    """
    # Apply zero-based indexing if needed
    file_idx = frame_idx - 1 if zero_based_indexing else frame_idx

    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file - return directly
        # (format_pattern is ignored as source_path contains the full file path)
        return camera_path

    elif image_type == "cine":
        # .cine files: pattern uses camera number
        if "%" in format_pattern:
            return camera_path / (format_pattern % camera)
        return camera_path / format_pattern

    elif image_type == "lavision_im7":
        # .im7 files: pattern uses frame index
        if "%" in format_pattern:
            return camera_path / (format_pattern % file_idx)
        return camera_path / format_pattern

    else:
        # Standard formats: pattern uses frame index
        if "%" in format_pattern:
            return camera_path / (format_pattern % file_idx)
        return camera_path / format_pattern


def _image_to_base64(img: np.ndarray, max_size: int = 512) -> str:
    """Convert image array to base64 PNG string.

    Parameters
    ----------
    img : np.ndarray
        Image array (H, W) or (H, W, C)
    max_size : int
        Maximum dimension for resizing

    Returns
    -------
    str
        Base64-encoded PNG image
    """
    from PIL import Image

    # Normalize to 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == bool:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(img)

    # Resize if too large
    if max(pil_img.size) > max_size:
        ratio = max_size / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _suggest_pattern(filename: str, forced_ext: Optional[str] = None) -> str:
    """Suggest a filename pattern based on a sample filename.

    Parameters
    ----------
    filename : str
        Sample filename to analyze
    forced_ext : str, optional
        Force a specific extension

    Returns
    -------
    str
        Suggested pattern with appropriate %0Nd placeholder
    """
    path = Path(filename)
    ext = forced_ext or path.suffix.lstrip(".")
    stem = path.stem

    # Try to find numeric portion
    match = re.search(r'(\d+)', stem)
    if match:
        num_str = match.group(1)
        num_len = len(num_str)
        prefix = stem[:match.start()]
        suffix = stem[match.end():]

        # Create pattern with appropriate zero-padding
        # If the number has leading zeros, preserve that width
        if num_len >= 5:
            pattern = f"{prefix}%05d{suffix}.{ext}"
        elif num_len == 4:
            pattern = f"{prefix}%04d{suffix}.{ext}"
        elif num_len == 3:
            pattern = f"{prefix}%03d{suffix}.{ext}"
        elif num_len == 2:
            # Check if it has leading zero (e.g., "01", "02")
            if num_str.startswith("0"):
                pattern = f"{prefix}%02d{suffix}.{ext}"
            else:
                pattern = f"{prefix}%d{suffix}.{ext}"
        else:
            pattern = f"{prefix}%d{suffix}.{ext}"

        return pattern

    # No number found, return as-is
    return filename


def _detect_ab_pair_pattern(
    sample_files: List[str], forced_ext: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Detect if files follow A/B naming convention and suggest both patterns.

    Looks for paired files with _A/_B or _a/_b suffixes in the filename.
    Returns both pattern_a and pattern_b if A/B pairs are detected.

    Parameters
    ----------
    sample_files : List[str]
        List of filenames to analyze (just names, not full paths)
    forced_ext : str, optional
        Force a specific extension

    Returns
    -------
    Optional[Dict[str, Any]]
        If A/B pairs detected:
        {
            "pattern_a": str,  # Pattern for A files (e.g., "B%05d_A.tif")
            "pattern_b": str,  # Pattern for B files (e.g., "B%05d_B.tif")
            "mode": "ab_format"
        }
        If not A/B format, returns None
    """
    if not sample_files:
        return None

    # Look for A/B pattern in filenames
    # Common patterns: _A.tif/_B.tif, _a.png/_b.png, -A.jpg/-B.jpg
    a_pattern = re.compile(r'[_-][Aa]\.[a-zA-Z]+$')
    b_pattern = re.compile(r'[_-][Bb]\.[a-zA-Z]+$')

    a_files = [f for f in sample_files if a_pattern.search(f)]
    b_files = [f for f in sample_files if b_pattern.search(f)]

    # Need both A and B files present, with similar counts
    if not a_files or not b_files:
        return None

    # Check that counts are roughly similar (within factor of 2)
    if max(len(a_files), len(b_files)) > 2 * min(len(a_files), len(b_files)):
        return None

    # Generate patterns from the first A and B files
    first_a = sorted(a_files)[0]
    first_b = sorted(b_files)[0]

    # Use existing _suggest_pattern but preserve the _A/_B suffix
    pattern_a = _suggest_pattern(first_a, forced_ext)
    pattern_b = _suggest_pattern(first_b, forced_ext)

    # Verify the patterns differ only in A/B
    pattern_a_normalized = re.sub(r'[_-][Aa]\.', '_X.', pattern_a)
    pattern_b_normalized = re.sub(r'[_-][Bb]\.', '_X.', pattern_b)

    if pattern_a_normalized != pattern_b_normalized:
        # Patterns don't match in structure - not a valid A/B pair
        return None

    return {
        "pattern_a": pattern_a,
        "pattern_b": pattern_b,
        "mode": "ab_format",
    }


def validate_images_generic(
    camera_path: Path,
    camera: int,
    image_format: str,
    image_type: str,
    expected_count: int,
    zero_based_indexing: bool,
    read_frame_fn: Callable[[int], np.ndarray],
) -> Dict[str, Any]:
    """Generic image validation for both PIV and calibration images.

    This is the core validation logic used by both validate_files() in app.py
    and validate_calibration_images() in calibration_loader.py.

    Parameters
    ----------
    camera_path : Path
        Path to the camera directory or source directory
    camera : int
        Camera number (1-based)
    image_format : str
        Image format pattern (e.g., "%05d.tif", "data.set")
    image_type : str
        One of "lavision_set", "lavision_im7", "cine", "standard"
    expected_count : int
        Expected number of images/frames
    zero_based_indexing : bool
        Whether file indices are 0-based
    read_frame_fn : Callable[[int], np.ndarray]
        Function to read a single frame by index, for preview generation.
        Should accept frame index (1-based) and return image array.

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
    result = {
        "valid": False,
        "found_count": 0,
        "expected_count": expected_count,
        "camera_path": str(camera_path),
        "first_image_preview": None,
        "image_size": None,
        "sample_files": [],
        "format_detected": None,
        "error": None,
        "suggested_pattern": None,
        "suggested_pattern_b": None,  # For A/B pair detection
        "suggested_mode": None,  # "ab_format" or None
    }

    start_idx = 0 if zero_based_indexing else 1

    # Handle container formats
    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file itself
        set_file = camera_path
        if not set_file.exists():
            result["error"] = f"Set file not found: {set_file}"
            # Try to find .set files in parent directory and suggest
            parent_dir = set_file.parent
            if parent_dir.exists():
                set_files = list(parent_dir.glob("*.set"))
                if set_files:
                    result["suggested_pattern"] = str(set_files[0])
                    result["sample_files"] = [f.name for f in set_files[:5]]
            return result

        if not set_file.suffix.lower() == ".set":
            result["error"] = f"Expected .set file but got: {set_file}"
            return result

        result["valid"] = True
        result["found_count"] = "container"
        result["format_detected"] = "set"
        result["sample_files"] = [set_file.name]

        # Try to read first frame for preview
        try:
            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])  # (W, H)
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not read preview from .set file: {e}")

        return result

    # For non-.set types, camera_path must be a directory
    if not camera_path.exists():
        result["error"] = f"Camera path does not exist: {camera_path}"
        return result

    if image_type == "cine":
        if "%" in image_format:
            cine_filename = image_format % camera
        else:
            cine_filename = image_format
        cine_file = camera_path / cine_filename

        if not cine_file.exists():
            result["error"] = f"CINE file not found: {cine_file}"
            # Try to find .cine files and suggest
            cine_files = list(camera_path.glob("*.cine"))
            if cine_files:
                result["suggested_pattern"] = cine_files[0].name
                result["sample_files"] = [f.name for f in cine_files[:5]]
            return result

        result["valid"] = True
        result["found_count"] = "container"
        result["format_detected"] = "cine"
        result["sample_files"] = [cine_filename]

        # Try to get frame count and preview
        try:
            from .readers.cine_reader import get_cine_frame_count
            frame_count = get_cine_frame_count(str(cine_file))
            result["found_count"] = frame_count

            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not read preview from .cine file: {e}")

        return result

    elif image_type == "lavision_im7":
        # Count .im7 files in directory
        # Replace any printf-style integer format (%d, %02d, %5d, %05d, etc.) with *
        pattern = re.sub(r'%\d*d', '*', image_format)
        matching_files = sorted(camera_path.glob(pattern))

        if not matching_files:
            try:
                first_expected = image_format % start_idx
            except (TypeError, ValueError):
                first_expected = image_format

            error_msg = f"First frame not found. Looking for: {first_expected}"
            error_msg = _append_folder_contents(error_msg, camera_path)
            result["error"] = error_msg

            # Try to find .im7 files and suggest
            im7_files = list(camera_path.glob("*.im7"))
            if im7_files:
                suggested = _suggest_pattern(im7_files[0].name, "im7")
                result["suggested_pattern"] = suggested
                result["sample_files"] = [f.name for f in im7_files[:5]]
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = "im7"

        # Validate count
        if len(matching_files) < expected_count:
            result["error"] = f"Found {len(matching_files)} files, expected {expected_count}"
            result["valid"] = False
        else:
            result["valid"] = True

        # Try to read first for preview
        try:
            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not read preview from .im7 file: {e}")

        return result

    else:
        # Standard formats
        # Replace any printf-style integer format (%d, %02d, %5d, %05d, etc.) with *
        pattern = re.sub(r'%\d*d', '*', image_format)
        matching_files = sorted(camera_path.glob(pattern))

        if not matching_files:
            try:
                first_expected = image_format % start_idx
            except (TypeError, ValueError):
                first_expected = image_format

            error_msg = f"First frame not found. Looking for: {first_expected}"
            error_msg = _append_folder_contents(error_msg, camera_path)
            result["error"] = error_msg

            # Try to find image files and suggest pattern
            all_images = []
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                all_images.extend(camera_path.glob(ext))

            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]

                # First try to detect A/B pairs
                ab_result = _detect_ab_pair_pattern(all_image_names)
                if ab_result:
                    result["suggested_pattern"] = ab_result["pattern_a"]
                    result["suggested_pattern_b"] = ab_result["pattern_b"]
                    result["suggested_mode"] = ab_result["mode"]
                else:
                    # Fall back to single pattern suggestion
                    suggested = _suggest_pattern(all_image_names[0])
                    result["suggested_pattern"] = suggested
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = matching_files[0].suffix.lstrip(".")

        # Validate count
        if len(matching_files) < expected_count:
            result["error"] = f"Found {len(matching_files)} files, expected {expected_count}"
            result["valid"] = False
        else:
            result["valid"] = True

        # Try to read first for preview
        try:
            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            result["error"] = f"Could not read first image: {e}"
            result["valid"] = False

        return result


def _append_folder_contents(error_msg: str, folder_path: Path) -> str:
    """Append folder contents to error message for debugging."""
    if folder_path.exists() and folder_path.is_dir():
        all_files = sorted([f.name for f in folder_path.iterdir() if f.is_file()])[:10]
        if all_files:
            error_msg += f". Found {len(all_files)} files: {', '.join(all_files[:5])}"
            if len(all_files) > 5:
                error_msg += f" and {len(all_files) - 5} more..."
        else:
            error_msg += f". Folder is empty: {folder_path}"
    return error_msg
