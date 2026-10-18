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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:
    from ..config import Config


def format_to_glob(fmt: str) -> str:
    """Convert printf-style format string to glob pattern.

    Replaces any %Nd format specifier (%d, %02d, %03d, %05d, etc.) with *.
    """
    return re.sub(r"%\d*d", "*", fmt)


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


def calibration_camera_folder(
    image_settings: dict,
    image_type: str,
    camera: int,
    num_cameras: int,
) -> str:
    """Subfolder name for one camera's calibration images (settings-sidecar based).

    Container formats (.cine, .set) never use camera subfolders. Standard and
    IM7 formats respect ``image.use_camera_subfolders``. A single-camera setup
    (``num_cameras == 1``, the configured camera count) always resolves at the
    source root. Otherwise an explicit ``camera_subfolders`` entry wins, else
    the ``Cam{N}`` default.
    """
    if image_type in ("lavision_set", "cine"):
        return ""
    if not image_settings.get("use_camera_subfolders", False):
        return ""
    # Single camera resolves at the source root — a stored subfolder entry is stale
    # state from a previous multi-camera session, never a valid override. Mirrors
    # Config.get_camera_folder for PIV images.
    if num_cameras == 1:
        return ""
    subfolders = image_settings.get("camera_subfolders") or []
    idx = camera - 1  # camera is 1-based
    if idx < len(subfolders) and subfolders[idx]:
        return str(subfolders[idx])
    return f"Cam{camera}"


def build_calibration_camera_path(
    source: Path,
    image_settings: dict,
    camera: int = 1,
    num_cameras: int = 1,
) -> Path:
    """Path to one camera's calibration images for an explicit source.

    ``image_settings`` is the ``image`` block of the source's settings sidecar
    (``pivtools_core.calibration_settings``) — the persisted YAML config plays
    no part in calibration image resolution.

    Path structure:
    - Container formats (.set, .cine): the source path directly (no camera subfolders)
    - IM7 with use_camera_subfolders=False: the source path directly
    - IM7/Standard with use_camera_subfolders=True: source / camera_folder
    """
    source = Path(source)
    fmt = image_settings.get("image_format") or ""
    image_type = image_settings.get("image_type") or infer_image_type(fmt)

    # Container formats: path is directly to file, no camera subfolder
    if image_type in ("lavision_set", "cine"):
        return source

    # IM7 without camera subfolders: return source directly
    if image_type == "lavision_im7" and not image_settings.get(
        "use_camera_subfolders", False
    ):
        return source

    # Standard/IM7 with subfolders: apply camera folder
    folder = calibration_camera_folder(image_settings, image_type, camera, num_cameras)
    if folder:
        return source / folder

    return source


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


AMBIGUOUS_FRAME_INDEX_MSG = "could not tell which number is the frame index in '{name}'"
"""Reason given when no single digit run can be identified as the frame index."""

NO_FRAME_NUMBER_MSG = "no frame number found in '{name}'"
"""Reason given when a filename contains no digits at all."""

_DIGIT_RUN = re.compile(r"(\d+)")


class PatternSuggestion(NamedTuple):
    """A suggested filename pattern, or the reason none could be made.

    Exactly one of the two fields is set: ``pattern`` is the printf-style format
    (e.g. ``"Cam1_%05d.tif"``), ``reason`` explains why no pattern was suggested
    (e.g. the frame number is ambiguous). Callers surface ``reason`` to the user
    rather than dropping it.
    """

    pattern: Optional[str]
    reason: Optional[str]


def _split_digit_runs(stem: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split a filename stem into its non-digit skeleton and its digit runs.

    ``"run3_img0001_cam1"`` -> ``(("run", "_img", "_cam", ""), ("3", "0001", "1"))``.
    The skeleton always has one more element than the runs.
    """
    parts = _DIGIT_RUN.split(stem)
    return tuple(parts[0::2]), tuple(parts[1::2])


def _run_placeholder(values: List[str]) -> str:
    """printf placeholder for a digit run given its observed string values.

    A leading zero in any value proves zero padding, so the width is that run's
    true length (``%0Nd`` for any N). Otherwise the run is unpadded (``%d``).
    """
    padded = [v for v in values if len(v) > 1 and v.startswith("0")]
    if padded:
        return f"%0{max(len(v) for v in padded)}d"
    return "%d"


def _suggest_pattern(
    sample_files: List[str], forced_ext: Optional[str] = None
) -> PatternSuggestion:
    """Suggest a filename pattern from a folder's sample filenames.

    The frame index is the digit run whose value varies between files. The first
    sorted file fixes the naming skeleton (non-digit text, run count and
    extension); only files sharing that skeleton are compared, so other file
    families in the folder (e.g. the B frames of an A/B pair) do not interfere.

    - Exactly one run position varies: it is the frame index.
    - Several vary: the one strictly increasing in sorted order is chosen; if
      none or more than one is, the frame index is ambiguous.
    - None vary (single file or identical names): a name with exactly one digit
      run uses it; otherwise the frame index is ambiguous.

    Parameters
    ----------
    sample_files : List[str]
        Filenames (not paths) found in the folder.
    forced_ext : str, optional
        Extension to write into the pattern instead of the file's own.

    Returns
    -------
    PatternSuggestion
        The pattern, or the reason no pattern could be suggested.
    """
    if not sample_files:
        raise ValueError("_suggest_pattern needs at least one filename")
    names = sorted(sample_files)
    first = Path(names[0])
    skeleton, first_runs = _split_digit_runs(first.stem)
    ext = forced_ext or first.suffix.lstrip(".")

    if not first_runs:
        return PatternSuggestion(None, NO_FRAME_NUMBER_MSG.format(name=first.name))

    group: List[Tuple[str, ...]] = []
    for name in names:
        path = Path(name)
        name_skeleton, runs = _split_digit_runs(path.stem)
        if name_skeleton == skeleton and path.suffix == first.suffix:
            group.append(runs)

    varying = [
        i for i in range(len(first_runs)) if len({runs[i] for runs in group}) > 1
    ]
    if len(varying) > 1:
        varying = [
            i
            for i in varying
            if all(int(a[i]) < int(b[i]) for a, b in zip(group[:-1], group[1:]))
        ]
    elif not varying and len(first_runs) == 1:
        varying = [0]

    if len(varying) != 1:
        return PatternSuggestion(
            None, AMBIGUOUS_FRAME_INDEX_MSG.format(name=first.name)
        )

    frame_run = varying[0]
    placeholder = _run_placeholder([runs[frame_run] for runs in group])
    pieces = [skeleton[0]]
    for i, run in enumerate(first_runs):
        pieces.append(placeholder if i == frame_run else run)
        pieces.append(skeleton[i + 1])
    suffix = f".{ext}" if ext else ""
    return PatternSuggestion(f"{''.join(pieces)}{suffix}", None)


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
    a_pattern = re.compile(r"[_-][Aa]\.[a-zA-Z]+$")
    b_pattern = re.compile(r"[_-][Bb]\.[a-zA-Z]+$")

    a_files = [f for f in sample_files if a_pattern.search(f)]
    b_files = [f for f in sample_files if b_pattern.search(f)]

    # Need both A and B files present, with similar counts
    if not a_files or not b_files:
        return None

    # Check that counts are roughly similar (within factor of 2)
    if max(len(a_files), len(b_files)) > 2 * min(len(a_files), len(b_files)):
        return None

    # Suggest each role's pattern from that role's files; the _A/_B suffix is
    # non-digit skeleton text, so it is preserved.
    pattern_a = _suggest_pattern(a_files, forced_ext).pattern
    pattern_b = _suggest_pattern(b_files, forced_ext).pattern
    if pattern_a is None or pattern_b is None:
        # No frame index identifiable; the caller's single-pattern suggestion
        # reports why.
        return None

    # Verify the patterns differ only in A/B
    pattern_a_normalized = re.sub(r"[_-][Aa]\.", "_X.", pattern_a)
    pattern_b_normalized = re.sub(r"[_-][Bb]\.", "_X.", pattern_b)

    if pattern_a_normalized != pattern_b_normalized:
        # Patterns don't match in structure - not a valid A/B pair
        return None

    return {
        "pattern_a": pattern_a,
        "pattern_b": pattern_b,
        "mode": "ab_format",
    }


_PRINTF_INT = re.compile(r"%(0?)(\d*)d")


def _pattern_to_regex(pattern: str) -> Optional["re.Pattern[str]"]:
    """Compile a single-placeholder printf pattern into a filename regex.

    ``"B%05d_A.tif"`` -> ``^B(\d{5,})_A\.tif$``; ``%d`` -> ``(\d+)``. A number
    wider than the pad width still formats wider, hence the open upper bound.
    Returns None unless the pattern has exactly one ``%d``-style placeholder.
    """
    placeholders = list(_PRINTF_INT.finditer(pattern))
    if len(placeholders) != 1:
        return None
    m = placeholders[0]
    width = int(m.group(2)) if m.group(2) else 0
    digits = rf"(\d{{{width},}})" if width > 1 else r"(\d+)"
    return re.compile(
        "^"
        + re.escape(pattern[: m.start()])
        + digits
        + re.escape(pattern[m.end() :])
        + "$"
    )


def detect_start_index(sample_files: List[str], pattern: str) -> Optional[int]:
    """Smallest frame index among the filenames that match ``pattern``.

    The index is parsed with the pattern itself (its ``%0Nd`` placeholder), so a
    camera number elsewhere in the name (``Cam1_00000.tif``) is never mistaken for
    the frame index. Pass the FULL matching listing, not a truncated sample: the
    minimum is only meaningful over every file.

    Returns None when the pattern has no single placeholder or nothing matches.
    """
    regex = _pattern_to_regex(pattern)
    if regex is None:
        return None
    indices = [int(m.group(1)) for m in map(regex.match, sample_files) if m]
    return min(indices) if indices else None


def start_index_hint_message(found: int, start: int) -> str:
    """User-facing text for files whose first index differs from the start index."""
    message = f"Files start at {found} but the start index is {start}"
    if found in (0, 1):
        message += f" — switch to {found}-based indexing"
    return message


def _start_index_mismatch(
    camera_path: Path,
    pattern: str,
    matching_names: List[str],
    start_idx: int,
    expected_count: int,
) -> Optional[int]:
    """The files' first index when a wrong 0/1 start index explains a failure.

    Returns the detected first index only when the first or last expected file
    is missing AND the files start at the other of 0/1. Otherwise None (no
    mismatch, or a missing file the start index does not explain).
    """
    found = detect_start_index(matching_names, pattern)
    if found is None or found == start_idx or found not in (0, 1):
        return None
    try:
        first = pattern % start_idx
        last = pattern % (start_idx + max(expected_count, 1) - 1)
    except (TypeError, ValueError):
        return None
    if (camera_path / first).exists() and (camera_path / last).exists():
        return None
    return found


def _validate_file_listing(
    result: Dict[str, Any],
    camera_path: Path,
    image_format: str,
    matching_files: List[Path],
    start_idx: int,
    expected_count: int,
) -> None:
    """Set ``valid``/``error`` for a non-empty listing of per-frame files.

    Records ``detected_start_index`` over the full listing. A wrong 0/1 start
    index that leaves the first or last expected file missing fails with the
    start-index hint and ``suggested_zero_based``; otherwise the count decides.
    """
    names = [f.name for f in matching_files]
    result["detected_start_index"] = detect_start_index(names, image_format)
    found = _start_index_mismatch(
        camera_path, image_format, names, start_idx, expected_count
    )
    if found is not None:
        result["valid"] = False
        result["suggested_zero_based"] = found == 0
        result["error"] = start_index_hint_message(found, start_idx)
    elif len(matching_files) < expected_count:
        result["valid"] = False
        result["error"] = (
            f"Found {len(matching_files)} files, expected {expected_count}"
        )
    else:
        result["valid"] = True


def _apply_folder_suggestion(
    result: Dict[str, Any], image_names: List[str], forced_ext: Optional[str] = None
) -> None:
    """Fill a validation result's pattern suggestion from the folder's filenames.

    A/B pairs set both patterns and ``suggested_mode``. Otherwise a single pattern
    is suggested; when none can be (ambiguous frame index), the reason goes into
    ``suggestion_error`` and is appended to ``error`` so it is never dropped.
    """
    ab_result = _detect_ab_pair_pattern(image_names, forced_ext)
    if ab_result:
        result["suggested_pattern"] = ab_result["pattern_a"]
        result["suggested_pattern_b"] = ab_result["pattern_b"]
        result["suggested_mode"] = ab_result["mode"]
        return
    _apply_single_suggestion(result, _suggest_pattern(image_names, forced_ext))


def _apply_single_suggestion(
    result: Dict[str, Any], suggestion: PatternSuggestion
) -> None:
    """Record a single-pattern suggestion, or append its reason to the error."""
    if suggestion.pattern is not None:
        result["suggested_pattern"] = suggestion.pattern
        return
    result["suggestion_error"] = suggestion.reason
    result["error"] = (
        f"{result['error']}; {suggestion.reason}"
        if result["error"]
        else suggestion.reason
    )


def _container_count_failure(
    result: Dict[str, Any], container: Path, exc: Exception
) -> Dict[str, Any]:
    """Fail a container validation whose image count could not be read.

    The message names the file and the reader's reason; it also goes into
    ``read_error`` so callers that rebuild the message (the PIV route) keep it.
    """
    message = f"Could not read the image count of {container}: {exc}"
    logging.warning(message)
    result["valid"] = False
    result["error"] = message
    result["read_error"] = message
    return result


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
        - found_count: int - Number of files (or container entries/frames) found
        - expected_count: int - Expected number of files
        - camera_path: str - Path to camera directory
        - first_image_preview: str - Base64 PNG of first image (if valid)
        - image_size: tuple - (width, height) of images
        - sample_files: list - Sample of matching filenames
        - format_detected: str - Detected file format
        - error: str or None - Error message if validation failed
        - read_error: str or None - Why a container file that EXISTS would not
          decode. Callers that build their own message must check this: a status
          flag alone cannot tell "file absent" from "file present but
          undecodable", and reporting the former for the latter sends the user
          hunting for a missing file that is right there.
          Kept as a string, not the exception object: this dict is returned
          straight to `jsonify` by the calibration validate route
          (`pivtools_gui/calibration/app/views.py`), and an Exception in it
          raises TypeError there — turning the very failure this field exists to
          report into an HTTP 500 with no message.
        - suggested_pattern: str or None - Suggested pattern if files don't match
        - suggestion_error: str or None - Why no pattern could be suggested
          (e.g. the frame index is ambiguous); also appended to ``error``
        - detected_start_index: int or None - First frame index among the files
          matching the pattern (full listing, standard/im7 only)
        - suggested_zero_based: bool or None - Set when the files start at the
          other of 0/1 and that leaves an expected file missing
    """
    result = {
        "valid": False,
        "found_count": 0,
        "expected_count": expected_count,
        "camera_path": str(camera_path),
        "first_image_preview": None,
        "read_error": None,
        "image_size": None,
        "sample_files": [],
        "format_detected": None,
        "error": None,
        "suggested_pattern": None,
        "suggested_pattern_b": None,  # For A/B pair detection
        "suggested_mode": None,  # "ab_format" or None
        "suggested_subfolder": None,
        "suggestion_error": None,  # why no pattern could be suggested
        "detected_start_index": None,  # first frame index among matching files
        "suggested_zero_based": None,  # set when a wrong 0/1 start index is the cause
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
                _apply_folder_suggestion(result, all_image_names)
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
                    result["suggested_pattern"] = sorted(set_files)[0].name
                    result["sample_files"] = [f.name for f in set_files[:5]]
            return result

        if not set_file.suffix.lower() == ".set":
            result["error"] = f"Expected .set file but got: {set_file}"
            return result

        result["format_detected"] = "set"
        result["sample_files"] = [set_file.name]

        # Get entry count from .set file (opens file briefly, no pixel decode).
        # Import from the concrete submodule (mirrors the .cine branch below) —
        # `get_set_entry_count` is NOT re-exported from the .readers package.
        # A count that cannot be read fails validation visibly: the file exists
        # but its index is unreadable, and an unknown count must not pass.
        try:
            from .readers.set_reader import get_set_entry_count

            result["found_count"] = get_set_entry_count(str(set_file))
        except Exception as e:
            return _container_count_failure(result, set_file, e)
        result["valid"] = True

        # Read the first frame. A read failure is a genuine error (corrupt/locked
        # container, wrong camera), so surface it instead of passing silently.
        try:
            img = read_frame_fn(1)
        except Exception as e:
            # Log as well as return: the caller may replace `error` with its own
            # text, and without this the real reason reached neither the screen
            # nor pypiv.log. An unsupported .set pixel encoding was invisible in
            # both for exactly this reason.
            logging.warning(f"Could not read first frame of {set_file}: {e}")
            result["valid"] = False
            result["error"] = f"Set file found but could not read the first frame: {e}"
            result["read_error"] = str(e)
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])  # (W, H)
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .set file: {e}")

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

        result["format_detected"] = "cine"
        result["sample_files"] = [cine_filename]

        # A header whose frame count cannot be read fails validation visibly.
        try:
            from .readers.cine_reader import get_cine_frame_count

            result["found_count"] = get_cine_frame_count(str(cine_file))
        except Exception as e:
            return _container_count_failure(result, cine_file, e)
        result["valid"] = True

        # Surface a genuine first-frame read failure.
        try:
            img = read_frame_fn(1)
        except Exception as e:
            logging.warning(f"Could not read first frame of {cine_file}: {e}")
            result["valid"] = False
            result["error"] = f"CINE file found but could not read the first frame: {e}"
            result["read_error"] = str(e)
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .cine file: {e}")

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
            im7_files = sorted(_glob_images(camera_path, "*.im7"))
            if im7_files:
                im7_names = [f.name for f in im7_files]
                _apply_single_suggestion(result, _suggest_pattern(im7_names, "im7"))
                result["sample_files"] = im7_names[:5]
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = "im7"

        _validate_file_listing(
            result, camera_path, image_format, matching_files, start_idx, expected_count
        )

        # Read the first frame. A failure here is a genuine read error (e.g. a
        # multi-camera .im7 frame-count mismatch), not a cosmetic preview issue —
        # surface it and fail validation instead of passing with a missing preview.
        try:
            img = read_frame_fn(1)
        except Exception as e:
            result["valid"] = False
            # The reason already names the file (e.g. the frame-count mismatch).
            # A start-index hint already explains a missing first file; keep it.
            if result["suggested_zero_based"] is None:
                result["error"] = str(e)
            return result
        try:
            result["image_size"] = (img.shape[1], img.shape[0])
            result["first_image_preview"] = _image_to_base64(img)
        except Exception as e:
            logging.warning(f"Could not render preview from .im7 file: {e}")

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
                _apply_folder_suggestion(result, all_image_names)
            return result

        result["found_count"] = len(matching_files)
        result["sample_files"] = [f.name for f in matching_files[:5]]
        result["format_detected"] = matching_files[0].suffix.lstrip(".")

        _validate_file_listing(
            result, camera_path, image_format, matching_files, start_idx, expected_count
        )

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
) -> PatternSuggestion:
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
    PatternSuggestion
        The suggested pattern, or the reason none could be made.
    """
    if not sample_files:
        raise ValueError("_suggest_pattern_for_role needs at least one filename")

    candidates = sample_files

    # Patterns to match A/B suffixes:
    # - [_-][Aa] matches _A, -A, _a, -a (with separator)
    # - \d[Aa] matches 1A, 2A, etc. (directly after digit, like IMG00001A.tif)
    pattern_a = r"(?:[_-]|(?<=\d))[Aa]\.[a-zA-Z0-9]+$"
    pattern_b = r"(?:[_-]|(?<=\d))[Bb]\.[a-zA-Z0-9]+$"

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
            opposite = _suggest_pattern(opposite_files, forced_ext)
            if opposite.pattern is None:
                return opposite
            if role == "A":
                # Transform B to A (preserving case) - handles both _B and digit+B
                return PatternSuggestion(
                    re.sub(
                        r"([Bb])(\.[a-zA-Z0-9]+$)",
                        lambda m: ("A" if m.group(1) == "B" else "a") + m.group(2),
                        opposite.pattern,
                    ),
                    None,
                )
            # Transform A to B (preserving case) - handles both _A and digit+A
            return PatternSuggestion(
                re.sub(
                    r"([Aa])(\.[a-zA-Z0-9]+$)",
                    lambda m: ("B" if m.group(1) == "A" else "b") + m.group(2),
                    opposite.pattern,
                ),
                None,
            )

        # No A/B files at all - fall back to first file (non-A/B naming)
        candidates = sample_files

    return _suggest_pattern(candidates, forced_ext)


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
            expected_nums = re.findall(r"\d+", expected_name)
            candidate_nums = re.findall(r"\d+", candidate)

            if expected_nums and candidate_nums:
                # Check if the camera number matches
                expected_cam_num = str(camera_num)
                has_matching_num = expected_cam_num in candidate_nums

                if has_matching_num:
                    # Camera-number match with camera-related prefix
                    camera_prefixes = re.compile(
                        r"(?:cam|camera|view|c)\s*[_\-]?\s*\d",
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
                if expected_cam_num in re.findall(r"\d+", candidate):
                    # Candidate has the right camera number
                    # Check prefix similarity (strip trailing digits/typos)
                    exp_prefix = re.sub(r"[\d]+.*$", "", expected_name).lower()
                    cand_prefix = re.sub(r"[\d]+.*$", "", candidate).lower()
                    if (
                        exp_prefix
                        and cand_prefix
                        and (
                            exp_prefix.startswith(cand_prefix)
                            or cand_prefix.startswith(exp_prefix)
                            or exp_prefix[:2] == cand_prefix[:2]
                        )
                    ):
                        score = 60

        if score > best_score:
            best_score = score
            best_name = candidate

    return best_name


def detect_flat_source_files(source_path: Path, patterns: List[str]) -> List[str]:
    """Detect image files matching the configured patterns directly in the source folder.

    Used to spot a stale multi-camera config: when camera subfolder paths (e.g.
    ``source/Cam1``) don't exist but files matching the pattern sit directly in
    ``source_path``, the dataset is a single-camera layout and the caller can
    suggest switching camera_count to 1.

    Returns up to 5 matching filenames, or an empty list if none / not a directory.
    """
    if not source_path.exists() or not source_path.is_dir():
        return []

    matches: List[Path] = []
    for pattern in patterns:
        if not pattern or not pattern.strip():
            continue
        matches.extend(_glob_images(source_path, format_to_glob(pattern)))

    return [f.name for f in sorted(set(matches))[:5]]


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
        - suggestion_error: str or None - Why no pattern could be suggested
        - suggested_zero_based: bool or None - Set when the files start at the
          other of 0/1 and that leaves an expected file missing
        - sample_files: List[str] - Sample of matching filenames
    """
    result = {
        "valid": False,
        "found_count": 0,
        "error": None,
        "suggested_pattern": None,
        "suggested_subfolder": None,
        "suggestion_error": None,  # why no pattern could be suggested
        "suggested_zero_based": None,  # set when a wrong 0/1 start index is the cause
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
                _apply_single_suggestion(
                    result, _suggest_pattern_for_role(all_image_names, role)
                )
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
            m = re.search(r"(\d+)", camera_path.name)
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
            _apply_single_suggestion(
                result, _suggest_pattern_for_role(all_image_names, role)
            )
        return result

    # The pattern matches but the first or last expected file is missing because
    # the files start at the other of 0/1: name the start index as the cause.
    # Checked before "Pattern incomplete", whose same-pattern suggestion would
    # otherwise be dropped and leave the user with no reason at all.
    matching_names = [f.name for f in matching_files]
    found_start = _start_index_mismatch(
        camera_path, pattern, matching_names, start_idx, expected_count
    )
    if found_start is not None:
        result["found_count"] = len(matching_files)
        result["sample_files"] = matching_names[:5]
        result["suggested_zero_based"] = found_start == 0
        result["error"] = start_index_hint_message(found_start, start_idx)
        return result

    # Files match the glob, but check if first expected file actually exists
    # This catches cases like "B%05d" matching "B00001_A.tif" via glob "B*"
    # but the actual file "B00001" doesn't exist
    if not first_file_exists:
        result["error"] = f"Pattern incomplete: '{first_expected}' not found"
        result["sample_files"] = (
            all_image_names[:5] if all_image_names else matching_names[:5]
        )
        # Suggest a corrected pattern based on role (an identical one says nothing)
        if all_image_names:
            suggestion = _suggest_pattern_for_role(all_image_names, role)
            if suggestion.pattern != pattern:
                _apply_single_suggestion(result, suggestion)
        return result

    result["found_count"] = len(matching_files)
    result["sample_files"] = [f.name for f in matching_files[:5]]

    # Validate count
    if len(matching_files) < expected_count:
        result["error"] = (
            f"Found {len(matching_files)} files, expected {expected_count}"
        )
        result["valid"] = False
    else:
        result["valid"] = True

    return result
