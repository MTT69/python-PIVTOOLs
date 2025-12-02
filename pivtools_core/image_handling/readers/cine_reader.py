"""
CINE file reader for Phantom high-speed cameras.

.cine files are single-camera containers with custom frame indexing.
Key difference from .im7/.set: one .cine file per camera, not all cameras in one file.

Directory structure:
    source_dir/Camera1.cine
    source_dir/Camera2.cine

Frame indexing:
    .cine files have a FirstImageNo offset (can be negative for pre-trigger frames)
    User sees: Pair 1, Pair 2, Pair 3...
    Internal: actual_frame = user_frame + FirstImageNo - 1 (handled transparently)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Metadata cache: {file_path: (metadata, last_modified_time)}
_metadata_cache = {}


def _get_cached_metadata(file_path: str):
    """Get cached cine metadata, reload if file changed.

    Caching avoids re-reading the header for each frame pair,
    which significantly improves performance for large datasets.
    """
    import cinereader as cr

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CINE file not found: {file_path}")

    mtime = path.stat().st_mtime

    if file_path in _metadata_cache:
        cached_meta, cached_mtime = _metadata_cache[file_path]
        if cached_mtime == mtime:
            return cached_meta

    logging.debug(f"Reading CINE metadata: {file_path}")
    metadata = cr.read_metadata(file_path)
    _metadata_cache[file_path] = (metadata, mtime)
    return metadata


def read_cine_pair(file_path: str, idx: int = 1, frames: int = 2) -> np.ndarray:
    """Read frame pair from .cine file.

    This function is called by load_images.read_pair() with the frame_a index
    already computed by config.get_frame_pair_indices().

    The idx parameter is the user-facing frame number (1-based). This function
    handles the translation to internal frame numbers using FirstImageNo.

    Args:
        file_path: Path to .cine file
        idx: Frame A index (1-based, from pairing logic)
        frames: Number of frames to read (always 2 for PIV)

    Returns:
        np.ndarray: Shape (2, H, W) with frame A and B as float32

    Raises:
        FileNotFoundError: If the .cine file doesn't exist
        ValueError: If requested frames are out of range
    """
    import cinereader as cr

    metadata = _get_cached_metadata(file_path)

    # Convert user frame number to internal frame number
    # User sees frames starting at 1, internal uses FirstImageNo
    # FirstImageNo is typically 1 but can be negative for pre-trigger
    internal_a = idx + metadata.FirstImageNo - 1
    internal_b = internal_a + 1

    # Validate frame range
    first_frame = metadata.FirstImageNo
    last_frame = metadata.FirstImageNo + metadata.ImageCount - 1

    if internal_a < first_frame:
        raise ValueError(
            f"Frame {idx} (internal {internal_a}) is before first available frame. "
            f"CINE file has frames {first_frame} to {last_frame}."
        )

    if internal_b > last_frame:
        raise ValueError(
            f"Frame pair starting at {idx} (internal {internal_a}, {internal_b}) "
            f"exceeds available frames. CINE file has frames {first_frame} to {last_frame}."
        )

    frame_a = cr.read_image(metadata, file_path, internal_a)
    frame_b = cr.read_image(metadata, file_path, internal_b)

    return np.stack([frame_a, frame_b], axis=0).astype(np.float32)


def read_cine_single(file_path: str, idx: int = 1) -> np.ndarray:
    """Read a single frame from .cine file.

    Args:
        file_path: Path to .cine file
        idx: Frame index (1-based, user-facing)

    Returns:
        np.ndarray: Shape (H, W) single frame as float32
    """
    import cinereader as cr

    metadata = _get_cached_metadata(file_path)
    internal_idx = idx + metadata.FirstImageNo - 1

    # Validate frame range
    first_frame = metadata.FirstImageNo
    last_frame = metadata.FirstImageNo + metadata.ImageCount - 1

    if internal_idx < first_frame or internal_idx > last_frame:
        raise ValueError(
            f"Frame {idx} (internal {internal_idx}) is out of range. "
            f"CINE file has frames {first_frame} to {last_frame}."
        )

    frame = cr.read_image(metadata, file_path, internal_idx)
    return frame.astype(np.float32)


def get_cine_frame_count(file_path: str) -> int:
    """Get total number of frames in .cine file.

    Args:
        file_path: Path to .cine file

    Returns:
        int: Total frame count
    """
    metadata = _get_cached_metadata(file_path)
    return metadata.ImageCount


def get_cine_first_frame_no(file_path: str) -> int:
    """Get FirstImageNo for a .cine file.

    This is mainly useful for debugging - user-facing code should
    use 1-based indexing which is translated automatically.

    Args:
        file_path: Path to .cine file

    Returns:
        int: First frame number in camera's numbering scheme
    """
    metadata = _get_cached_metadata(file_path)
    return metadata.FirstImageNo


def get_cine_image_shape(file_path: str) -> Tuple[int, int]:
    """Get image dimensions from .cine file metadata.

    Args:
        file_path: Path to .cine file

    Returns:
        Tuple[int, int]: (height, width) of images
    """
    import cinereader as cr

    metadata = _get_cached_metadata(file_path)
    # Read first frame to get actual shape (metadata has biWidth/biHeight but
    # reading a frame ensures we get the exact array dimensions)
    first_frame = cr.read_image(metadata, file_path, metadata.FirstImageNo)
    return first_frame.shape


def get_cine_metadata_info(file_path: str) -> dict:
    """Get comprehensive metadata info for a .cine file.

    Useful for debugging and validation.

    Args:
        file_path: Path to .cine file

    Returns:
        dict: Metadata information including frame count, dimensions, etc.
    """
    metadata = _get_cached_metadata(file_path)

    return {
        "first_image_no": metadata.FirstImageNo,
        "image_count": metadata.ImageCount,
        "last_image_no": metadata.FirstImageNo + metadata.ImageCount - 1,
        "width": metadata.biWidth,
        "height": metadata.biHeight,
        "bit_depth": metadata.biBitCount,
        "real_bpp": getattr(metadata, 'RealBPP', metadata.biBitCount),
    }


def clear_metadata_cache():
    """Clear the metadata cache.

    Useful for testing or when files have been modified.
    """
    global _metadata_cache
    _metadata_cache = {}
