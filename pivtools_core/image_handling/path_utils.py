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


def format_to_glob(fmt: str) -> str:
    """Convert printf-style format string to glob pattern.

    Replaces any %Nd format specifier (%d, %02d, %03d, %05d, etc.) with *.
    """
    return re.sub(r'%\d*d', '*', fmt)


def infer_image_type(image_format: str) -> str:
    """Infer the image type from a format string's extension.

    Standalone twin of ``Config._detect_image_type`` for callers that hold a format
    string but not a ``Config`` (e.g. the calibration CLI, which takes ``--source`` /
    ``--image-format`` directly). Returns one of ``standard``, ``cine``,
    ``lavision_set``, ``lavision_im7``.
    """
    fmt = (image_format or "").lower()
    if ".cine" in fmt:
        return "cine"
    if ".set" in fmt:
        return "lavision_set"
    if ".im7" in fmt or ".ims" in fmt:
        return "lavision_im7"
    return "standard"


def _glob_images(directory: Path, pattern: str) -> List[Path]:
    """glob() a directory, excluding dotfiles.

    macOS writes AppleDouble sidecar files ("._foo.tif") next to real images on
    FAT/exFAT/network volumes. These sort before the real files (".") and would
    otherwise poison pattern suggestion (yielding "._foo%05d.tif") and file counts.
    Any leading-dot name (hidden file or AppleDouble sidecar) is excluded.
    """
    return [f for f in directory.glob(pattern) if not f.name.startswith(".")]


def build_calibration_camera_path(
    config: "Config",
    source_path_idx: int = 0,
    camera: int = 1,
) -> Path:
    """Build path to calibration images for a specific camera.

    Uses calibration_sources directly - no legacy subfolder logic.

    Path structure:
    - Container formats (.set, .cine): calibration_source path directly (no camera subfolders)
    - IM7 with use_camera_subfolders=False: calibration_source path directly
    - IM7/Standard with use_camera_subfolders=True: calibration_source / camera_folder

    Args:
        config: Configuration object with calibration settings
        source_path_idx: Index into calibration_sources list (default: 0)
        camera: Camera number (1-based, default: 1)

    Returns:
        Path: Full path to calibration image directory or container file

    Raises:
        ValueError: If calibration_sources is not configured
        IndexError: If source_path_idx is out of range

    Examples:
        >>> # Standard format with camera subfolders: /data/calibration/Cam1/
        >>> path = build_calibration_camera_path(config, 0, 1)

        >>> # Container format: /data/calibration/data.set (no camera folder)
        >>> path = build_calibration_camera_path(config, 0, 1)
    """
    cal_source = config.get_calibration_source(source_path_idx)
    cal_image_type = config.calibration_image_type

    # Container formats: path is directly to file, no camera subfolder
    if cal_image_type in ("lavision_set", "cine"):
        return cal_source

    # IM7 without camera subfolders: return source directly
    if cal_image_type == "lavision_im7" and not config.calibration_use_camera_subfolders:
        return cal_source

    # Standard/IM7 with subfolders: apply camera folder
    if config.calibration_use_camera_subfolders:
        camera_folder = config.get_calibration_camera_folder(camera)
        if camera_folder:
            return cal_source / camera_folder

    return cal_source


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
        "suggested_subfolder": None,
    }

    start_idx = 0 if zero_based_indexing else 1

    # Check for empty pattern first (glob('') throws ValueError)
    if not image_format or not image_format.strip():
        result["error"] = "Pattern is empty"
        # Still provide suggestions from files in directory
        if camera_path.exists() and camera_path.is_dir():
            all_images = []
            if image_type == "lavision_im7":
                all_images = _glob_images(camera_path, "*.im7")
            else:
                for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                    all_images.extend(_glob_images(camera_path, ext))
            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]
                # Try to detect A/B pairs
                ab_result = _detect_ab_pair_pattern(all_image_names)
                if ab_result:
                    result["suggested_pattern"] = ab_result["pattern_a"]
                    result["suggested_pattern_b"] = ab_result["pattern_b"]
                    result["suggested_mode"] = ab_result["mode"]
                else:
                    suggested = _suggest_pattern(all_image_names[0])
                    result["suggested_pattern"] = suggested
        return result

    # Handle container formats
    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file itself
        set_file = camera_path
        if not set_file.exists():
            result["error"] = f"Set file not found: {set_file}"
            # Try to find .set files in parent directory and suggest
            parent_dir = set_file.parent
            if parent_dir.exists():
                set_files = _glob_images(parent_dir, "*.set")
                if set_files:
                    result["suggested_pattern"] = str(set_files[0])
                    result["sample_files"] = [f.name for f in set_files[:5]]
            return result

        if not set_file.suffix.lower() == ".set":
            result["error"] = f"Expected .set file but got: {set_file}"
            return result

        result["valid"] = True
        result["format_detected"] = "set"
        result["sample_files"] = [set_file.name]

        # Get entry count from .set file (opens file briefly, no pixel decode)
        try:
            from .readers import get_set_entry_count
            result["found_count"] = get_set_entry_count(str(set_file))
        except Exception as e:
            logging.warning(f"Could not read .set entry count: {e}")
            result["found_count"] = "container"

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
        suggestion = _suggest_camera_subfolder(camera_path, camera)
        if suggestion:
            result["suggested_subfolder"] = suggestion
            result["error"] += f'. Did you mean "{suggestion}"?'
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
            cine_files = _glob_images(camera_path, "*.cine")
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
        pattern = format_to_glob(image_format)
        matching_files = sorted(_glob_images(camera_path, pattern))

        if not matching_files:
            try:
                first_expected = image_format % start_idx
            except (TypeError, ValueError):
                first_expected = image_format

            error_msg = f"First frame not found. Looking for: {first_expected}"
            error_msg = _append_folder_contents(error_msg, camera_path)
            result["error"] = error_msg

            # Try to find .im7 files and suggest
            im7_files = _glob_images(camera_path, "*.im7")
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
        pattern = format_to_glob(image_format)
        matching_files = sorted(_glob_images(camera_path, pattern))

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
                all_images.extend(_glob_images(camera_path, ext))

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

        # Try to read first for preview (non-fatal if it fails)
        try:
            img = read_frame_fn(1)
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not read preview for camera: {e}")

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


def _suggest_pattern_for_role(
    sample_files: List[str],
    role: Optional[str] = None,
    forced_ext: Optional[str] = None,
) -> Optional[str]:
    """Suggest a pattern from sample files, optionally filtering by role (A/B).

    Parameters
    ----------
    sample_files : List[str]
        List of filenames to analyze
    role : str, optional
        If "A", filter to files with _A suffix. If "B", filter to _B suffix.
    forced_ext : str, optional
        Force a specific extension

    Returns
    -------
    Optional[str]
        Suggested pattern, or None if no matching files found
    """
    if not sample_files:
        return None

    candidates = sample_files

    # Patterns to match A/B suffixes:
    # - [_-][Aa] matches _A, -A, _a, -a (with separator)
    # - \d[Aa] matches 1A, 2A, etc. (directly after digit, like IMG00001A.tif)
    pattern_a = r'(?:[_-]|(?<=\d))[Aa]\.[a-zA-Z0-9]+$'
    pattern_b = r'(?:[_-]|(?<=\d))[Bb]\.[a-zA-Z0-9]+$'

    # Filter by role if specified
    if role == "A":
        candidates = [f for f in sample_files if re.search(pattern_a, f)]
    elif role == "B":
        candidates = [f for f in sample_files if re.search(pattern_b, f)]

    if not candidates and role:
        # No files match this role - try to derive from opposite role
        opposite_pattern = pattern_b if role == "A" else pattern_a
        opposite_files = [f for f in sample_files if re.search(opposite_pattern, f)]

        if opposite_files:
            # Generate pattern from opposite role, then transform A<->B
            opposite_suggestion = _suggest_pattern(sorted(opposite_files)[0], forced_ext)
            if opposite_suggestion:
                if role == "A":
                    # Transform B to A (preserving case) - handles both _B and digit+B
                    return re.sub(
                        r'([Bb])(\.[a-zA-Z0-9]+$)',
                        lambda m: ('A' if m.group(1) == 'B' else 'a') + m.group(2),
                        opposite_suggestion
                    )
                else:
                    # Transform A to B (preserving case) - handles both _A and digit+A
                    return re.sub(
                        r'([Aa])(\.[a-zA-Z0-9]+$)',
                        lambda m: ('B' if m.group(1) == 'A' else 'b') + m.group(2),
                        opposite_suggestion
                    )

        # No A/B files at all - fall back to first file (non-A/B naming)
        candidates = sample_files

    # Sort and use first file to generate pattern
    candidates = sorted(candidates)
    return _suggest_pattern(candidates[0], forced_ext)


def _suggest_camera_subfolder(camera_path: Path, camera_num: int) -> Optional[str]:
    """Suggest a camera subfolder name when the expected one doesn't exist.

    Scores candidate subdirectories in the parent folder by similarity to the
    expected camera folder name, returning the best match or None.
    """
    parent = camera_path.parent
    expected_name = camera_path.name

    if not parent.exists() or not parent.is_dir():
        return None

    # List subdirectories (cap at 100 to avoid huge dirs)
    subdirs = []
    for entry in parent.iterdir():
        if entry.is_dir():
            subdirs.append(entry.name)
            if len(subdirs) >= 100:
                break

    if not subdirs:
        return None

    best_score = 0
    best_name = None

    for candidate in subdirs:
        score = 0

        # Exact case-insensitive match (e.g. Cam1 vs cam1)
        if expected_name.lower() == candidate.lower():
            score = 100
        else:
            # Extract numbers from both names
            expected_nums = re.findall(r'\d+', expected_name)
            candidate_nums = re.findall(r'\d+', candidate)

            if expected_nums and candidate_nums:
                # Check if the camera number matches
                expected_cam_num = str(camera_num)
                has_matching_num = expected_cam_num in candidate_nums

                if has_matching_num:
                    # Camera-number match with camera-related prefix
                    camera_prefixes = re.compile(
                        r'(?:cam|camera|view|c)\s*[_\-]?\s*\d',
                        re.IGNORECASE,
                    )
                    if camera_prefixes.search(candidate):
                        score = 80
                    else:
                        # Number-only match (any directory name)
                        score = 40
            else:
                # Fuzzy match: check if candidate contains the camera number
                # and has a similar prefix (handles typos like "Camrr" → "Cam1")
                expected_cam_num = str(camera_num)
                if expected_cam_num in re.findall(r'\d+', candidate):
                    # Candidate has the right camera number
                    # Check prefix similarity (strip trailing digits/typos)
                    exp_prefix = re.sub(r'[\d]+.*$', '', expected_name).lower()
                    cand_prefix = re.sub(r'[\d]+.*$', '', candidate).lower()
                    if exp_prefix and cand_prefix and (
                        exp_prefix.startswith(cand_prefix) or cand_prefix.startswith(exp_prefix)
                        or exp_prefix[:2] == cand_prefix[:2]
                    ):
                        score = 60

        if score > best_score:
            best_score = score
            best_name = candidate

    return best_name


def validate_single_pattern(
    camera_path: Path,
    pattern: str,
    image_type: str,
    expected_count: int,
    zero_based_indexing: bool,
    role: Optional[str] = None,
    camera_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate a single image pattern.

    This function validates one pattern (e.g., pattern A or pattern B) independently,
    returning pattern-specific validation results including suggestions filtered by role.

    Parameters
    ----------
    camera_path : Path
        Path to the camera directory
    pattern : str
        Image format pattern to validate (e.g., "B%05d_A.tif")
    image_type : str
        One of "lavision_set", "lavision_im7", "cine", "standard"
    expected_count : int
        Expected number of files matching this pattern
    zero_based_indexing : bool
        Whether file indices are 0-based
    role : str, optional
        "A", "B", or None. Used to filter suggestions to matching role.

    Returns
    -------
    Dict[str, Any]
        Validation result with keys:
        - valid: bool - Whether pattern validation passed
        - found_count: int - Number of files matching pattern
        - error: str or None - Error message if validation failed
        - suggested_pattern: str or None - Role-aware suggestion if pattern doesn't match
        - sample_files: List[str] - Sample of matching filenames
    """
    result = {
        "valid": False,
        "found_count": 0,
        "error": None,
        "suggested_pattern": None,
        "suggested_subfolder": None,
        "sample_files": [],
    }

    # Check for empty pattern first
    if not pattern or not pattern.strip():
        result["error"] = "Pattern is empty"
        # Still provide suggestions from files in directory
        if camera_path.exists() and camera_path.is_dir():
            all_images = []
            if image_type == "lavision_im7":
                all_images = _glob_images(camera_path, "*.im7")
            else:
                for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                    all_images.extend(_glob_images(camera_path, ext))
            if all_images:
                all_image_names = [f.name for f in sorted(all_images)]
                result["sample_files"] = all_image_names[:5]
                suggested = _suggest_pattern_for_role(all_image_names, role)
                if suggested:
                    result["suggested_pattern"] = suggested
        return result

    # Container formats don't use per-pattern validation
    if image_type in ("lavision_set", "cine"):
        result["valid"] = True
        result["found_count"] = "container"
        return result

    if not camera_path.exists():
        result["error"] = f"Camera path does not exist: {camera_path}"
        cam = camera_num
        if cam is None:
            m = re.search(r'(\d+)', camera_path.name)
            cam = int(m.group(1)) if m else 1
        suggestion = _suggest_camera_subfolder(camera_path, cam)
        if suggestion:
            result["suggested_subfolder"] = suggestion
        return result

    if not camera_path.is_dir():
        result["error"] = f"Camera path is not a directory: {camera_path}"
        return result

    # Convert pattern to glob pattern
    glob_pattern = format_to_glob(pattern)
    matching_files = sorted(_glob_images(camera_path, glob_pattern))

    # Calculate the first expected filename
    start_idx = 0 if zero_based_indexing else 1
    try:
        first_expected = pattern % start_idx
    except (TypeError, ValueError):
        first_expected = pattern

    # Check if the first expected file actually exists
    first_file_path = camera_path / first_expected
    first_file_exists = first_file_path.exists()

    # Find all image files (for suggestions)
    all_images = []
    if image_type == "lavision_im7":
        all_images = _glob_images(camera_path, "*.im7")
    else:
        for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
            all_images.extend(_glob_images(camera_path, ext))
    all_image_names = [f.name for f in sorted(all_images)] if all_images else []

    if not matching_files:
        result["error"] = f"No files matching pattern '{pattern}'"
        if all_image_names:
            result["sample_files"] = all_image_names[:5]
            # Get role-aware suggestion
            suggested = _suggest_pattern_for_role(all_image_names, role)
            if suggested:
                result["suggested_pattern"] = suggested
        return result

    # Files match the glob, but check if first expected file actually exists
    # This catches cases like "B%05d" matching "B00001_A.tif" via glob "B*"
    # but the actual file "B00001" doesn't exist
    if not first_file_exists:
        result["error"] = f"Pattern incomplete: '{first_expected}' not found"
        result["sample_files"] = all_image_names[:5] if all_image_names else [f.name for f in matching_files[:5]]
        # Suggest a corrected pattern based on role
        suggested = _suggest_pattern_for_role(all_image_names, role)
        if suggested and suggested != pattern:
            result["suggested_pattern"] = suggested
        return result

    result["found_count"] = len(matching_files)
    result["sample_files"] = [f.name for f in matching_files[:5]]

    # Validate count
    if len(matching_files) < expected_count:
        result["error"] = f"Found {len(matching_files)} files, expected {expected_count}"
        result["valid"] = False
    else:
        result["valid"] = True

    return result
