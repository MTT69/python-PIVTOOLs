from pathlib import Path
from typing import Tuple, Optional, List
import logging

import dask
import dask.array as da
import numpy as np
from dask.delayed import Delayed
from scipy.ndimage import convolve

from ..config import Config
from ..vector_loading import read_mask_from_mat
from ..window_utils import compute_window_centers, compute_window_centers_single_mode

# Import all readers to register them
from .readers import get_reader


def read_image(file_path: str, **kwargs) -> np.ndarray:
    """Read an image file using appropriate reader based on file extension.

    Args:
        file_path (str): Path to the image file
        **kwargs: Additional arguments passed to the specific reader

    Returns:
        np.ndarray: The image data
    """
    reader_func = get_reader(file_path)
    return reader_func(file_path, **kwargs)


def read_single_frame(
    file_path: Path,
    camera: int,
    frame_idx: int,
    image_type: str,
    time_resolved: bool = True,
) -> np.ndarray:
    """Read a single frame from any supported image format.

    This is the core reader that handles all format-specific logic for reading
    ONE frame. Both PIV pair reading and calibration image reading use this
    function, eliminating duplicated format handling.

    Supported formats:
    - lavision_set: All cameras/frames in one .set container
    - lavision_im7: Per-frame .im7 files (or multi-frame)
    - cine: Phantom .cine video containers
    - standard: Individual image files (.tif, .png, .jpg, etc.)

    Args:
        file_path: Path to the image file or container
        camera: Camera number (1-based)
        frame_idx: Frame index within the file/container (1-based)
        image_type: One of "lavision_set", "lavision_im7", "cine", "standard"
        time_resolved: For .set files, whether to read single frame (True) or
                      expect A+B pair in one entry (False)

    Returns:
        np.ndarray: Single frame of shape (H, W)

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the image cannot be read or format is unsupported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    if image_type == "lavision_set":
        # .set container: all cameras and frames in one file
        if time_resolved:
            # Single frame per entry - read just one frame
            img = read_image(
                str(file_path),
                camera_no=camera,
                im_no=frame_idx,
                time_resolved=True
            )
        else:
            # Pre-paired A+B in one entry - read both, return first
            img = read_image(str(file_path), camera_no=camera, im_no=frame_idx)

        # If returned as pair (2, H, W), extract single frame
        if img.ndim == 3 and img.shape[0] == 2:
            img = img[0]
        return img

    elif image_type == "lavision_im7":
        # .im7 file: may contain single frame or A+B pair
        img = read_image(
            str(file_path),
            camera_no=camera,
            frames=1,
            frames_per_camera=1
        )
        # read_lavision_im7 returns (frames, H, W) for single frame
        if img.ndim == 3:
            img = img[0]
        return img

    elif image_type == "cine":
        # .cine video container: frames extracted by index
        # Reader handles FirstImageNo translation internally
        img = read_image(str(file_path), idx=frame_idx, frames=1)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        return img

    else:
        # Standard formats (.tif, .png, .jpg, etc.)
        return read_image(str(file_path))


def read_pair(idx: int, camera_path: Path, camera: int, config: Config) -> np.ndarray:
    """Read a pair of images (A and B frames).

    This function handles four main file organization strategies:

    1. Multi-camera container files (.set):
       - All cameras stored in ONE file for all time instances
       - Organized as: source_dir/xxx.set contains all cameras and all time instances
       - No camera subdirectories (Cam1/, Cam2/, etc.)

    2. Multi-camera per-file container (.im7):
       - All cameras stored in ONE file per time instance
       - Organized as: source_dir/B00001.im7 contains all cameras for time instance 1
       - No camera subdirectories

    3. Single-camera video containers (.cine):
       - One video file per camera in source directory
       - Organized as: source_dir/Camera1.cine, source_dir/Camera2.cine
       - Pattern uses %d for camera number (not frame index)

    4. Camera-specific directories with standard formats (.tif, .png, .jpg):
       - Organized as: source_dir/Cam1/00001.tif, source_dir/Cam2/00001.tif
       - Each camera has its own subdirectory

    Frame Pairing:
        The idx parameter is ALWAYS 1-based internally (idx=1 means first pair).
        The actual file indices to read are determined by config.get_frame_pair_indices(),
        which handles:
        - Zero-based vs 1-based file indexing
        - Sequential vs skip pairing modes
        - Non-time-resolved A/B pairs vs time-resolved sequences

    Args:
        idx (int): Pair number (1-based, where 1 = first pair)
        camera_path (Path): Path to camera directory or source directory (for containers)
        camera (int): Camera number (1-based)
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) containing frame A and B
    """
    format_str = config.image_format[0]
    image_type = config.image_type
    frame_a_idx, frame_b_idx = config.get_frame_pair_indices(idx)

    # Handle container formats (single file contains multiple frames/cameras)
    if image_type == "lavision_set":
        # For .set files, camera_path IS the .set file itself
        # (source_path is the full path to the .set file)
        set_file_path = camera_path

        if config.time_resolved:
            # Time-resolved: read two separate frames from container
            frame_a = read_single_frame(set_file_path, camera, frame_a_idx, image_type, time_resolved=True)
            frame_b = read_single_frame(set_file_path, camera, frame_b_idx, image_type, time_resolved=True)
            return np.stack([frame_a, frame_b], axis=0)
        else:
            # Pre-paired: A+B frames in one entry - read directly
            return read_image(str(set_file_path), camera_no=camera, im_no=idx)

    elif image_type == "lavision_im7":
        # Check if single-camera or multi-camera IM7 files
        single_camera_im7 = config.images_use_camera_subfolders

        if config.time_resolved:
            # Time-resolved: each file has one frame, read two files
            im7_file_a = camera_path / (format_str % frame_a_idx)
            im7_file_b = camera_path / (format_str % frame_b_idx)

            if single_camera_im7:
                # Single-camera file: don't pass camera_no
                frame_a = read_image(str(im7_file_a), frames=1, frames_per_camera=1)
                frame_b = read_image(str(im7_file_b), frames=1, frames_per_camera=1)
                # Handle shape (returns (frames, H, W) for single frame)
                if frame_a.ndim == 3:
                    frame_a = frame_a[0]
                if frame_b.ndim == 3:
                    frame_b = frame_b[0]
            else:
                # Multi-camera file: pass camera_no to extract specific camera
                frame_a = read_single_frame(im7_file_a, camera, frame_a_idx, image_type)
                frame_b = read_single_frame(im7_file_b, camera, frame_b_idx, image_type)
            return np.stack([frame_a, frame_b], axis=0)
        else:
            # Non-time-resolved: each file contains A+B pair
            im7_file_path = camera_path / (format_str % frame_a_idx)
            if single_camera_im7:
                # Single-camera file: don't pass camera_no
                return read_image(str(im7_file_path))
            else:
                # Multi-camera file: pass camera_no
                return read_image(str(im7_file_path), camera_no=camera)

    elif image_type == "cine":
        # .cine: one video file per camera, frames extracted by index
        cine_filename = format_str % camera
        cine_path = camera_path / cine_filename
        # For pairs, read 2 consecutive frames starting at frame_a_idx
        return read_image(str(cine_path), idx=frame_a_idx, frames=2)

    else:
        # Standard formats: separate files per frame
        if len(config.image_format) == 2:
            # Non-time-resolved: separate A and B format patterns
            image_format_A, image_format_B = config.image_format
            file_a = camera_path / (image_format_A % frame_a_idx)
            file_b = camera_path / (image_format_B % frame_b_idx)
        else:
            # Time-resolved: single format pattern
            file_a = camera_path / (format_str % frame_a_idx)
            file_b = camera_path / (format_str % frame_b_idx)

        frame_a = read_single_frame(file_a, camera, frame_a_idx, image_type)
        frame_b = read_single_frame(file_b, camera, frame_b_idx, image_type)
        return np.stack([frame_a, frame_b], axis=0)


def delayed_image_pair(idx: int, camera_path: Path, camera: int, config: Config) -> Delayed:
    """Create a delayed task to read a pair of images.

    Args:
        idx (int): Index of the image pair to read
        camera_path (Path): Path to camera directory or set file
        camera (int): Camera number
        config (Config): Configuration object

    Returns:
        Delayed: A delayed task representing the image pair
    """

    return dask.delayed(read_pair)(idx, camera_path, camera, config)


def to_dask_array(delayed_pair: Delayed, config: Config) -> da.Array:
    """

    Args:
        delayed_pair (dask.delayed): _description_
        config (Config): _description_

    Returns:
        dask.array.Array: _description_
    """
    arr = dask.array.from_delayed(
        delayed_pair,
        shape=(2, *config.image_shape), 
        dtype=config.image_dtype,
    )
    return arr


def load_images(camera: int, config: Config, source: Path = None) -> da.Array:
    """Load images for a specific camera using pure lazy loading.
    
    This function creates one delayed task per image pair. Each task is
    completely independent and only loads when computed on a worker.
    
    Memory Efficiency - True Lazy Loading:
    - Creates N delayed objects (~1 KB each) for N images
    - Main process memory: ~N KB (minimal, just task graph)
    - Worker memory: Only 1 image pair at a time (~80 MB)
    - Each worker: load → process → save → free → next
    - Peak worker memory: ~280 MB (1 image + PIV overhead)
    
    This is the OPTIMAL Dask pattern:
    - No pre-loading of batches
    - No memory accumulation
    - Workers process images one-by-one
    - Dask scheduler handles distribution naturally

    Args:
        camera (int): The camera number.
        config (Config): The configuration object.
        source (Path, optional): The root directory for camera folders.
            If None, uses first source_path from config.

    Returns:
        da.Array: A Dask array containing the loaded image pairs.
            Shape: (num_frame_pairs, 2, H, W)
            Note: This is a lazy array - no actual image data loaded yet.
            Each element is an independent delayed task.
    """
    if source is None:
        source = config.source_paths[0]

    # Determine camera_path based on image type
    # Container formats don't use camera subdirectories:
    # - .set: all cameras in one file
    # - .cine: separate files per camera, but in source directory (not subdirs)
    # - .im7: depends on images_use_camera_subfolders setting
    image_type = config.image_type

    if image_type == "lavision_set":
        # For .set files, source IS the .set file itself (full path)
        camera_path = source
    elif image_type == "lavision_im7":
        if config.images_use_camera_subfolders:
            # Single-camera IM7 files in camera subdirectories
            folder = config.get_camera_folder(camera)
            camera_path = source / folder if folder else source
        else:
            # Multi-camera IM7 files in source directory (default)
            camera_path = source
    elif image_type == "cine":
        camera_path = source  # .cine files in source directory (no subdirs)
    else:
        # Standard formats use camera subdirectories
        folder = config.get_camera_folder(camera)
        camera_path = source / folder if folder else source
    
    num_pairs = config.num_frame_pairs

    # Create one delayed task per image pair (pure lazy loading)
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, camera, config)
        for idx in range(1, num_pairs + 1)
    ]

    # Convert each delayed task to a Dask array
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]

    # Stack into single array - still lazy, no computation yet!
    pairs_stack = da.stack(dask_pairs, axis=0)

    logging.info(
        f"Lazy loading complete: {num_pairs} independent delayed tasks created "
        f"(~{num_pairs} KB memory footprint)"
    )
    
    return pairs_stack


def create_rectangular_mask(config: Config) -> np.ndarray:
    """
    Create a rectangular edge mask based on config settings.
    
    Parameters
    ----------
    config : Config
        Configuration object containing image shape and rectangular mask settings
        
    Returns
    -------
    np.ndarray
        Boolean mask array of shape (H, W) where True = masked region
    """
    H, W = config.image_shape
    mask = np.zeros((H, W), dtype=bool)
    
    rect_settings = config.mask_rectangular_settings
    top = rect_settings.get("top", 0)
    bottom = rect_settings.get("bottom", 0)
    left = rect_settings.get("left", 0)
    right = rect_settings.get("right", 0)
    
    # Apply edge masks
    if top > 0:
        mask[:top, :] = True
    if bottom > 0:
        mask[-bottom:, :] = True
    if left > 0:
        mask[:, :left] = True
    if right > 0:
        mask[:, -right:] = True
    
    masked_pixels = np.sum(mask)
    total_pixels = mask.size
    mask_fraction = masked_pixels / total_pixels if total_pixels > 0 else 0
    
    logging.debug(
        "Created rectangular mask: top={}, bottom={}, left={}, right={} "
        "({}/{:.0f} pixels = {:.1f}%)",
        top, bottom, left, right, masked_pixels, total_pixels, mask_fraction * 100
    )
    
    return mask


def load_mask_for_camera(
    camera_num: int, config: Config, source_path_idx: int = 0
) -> Optional[np.ndarray]:
    """
    Load or create a mask for a specific camera.
    
    The mask is a boolean array of shape (H, W) where True indicates
    regions to mask out (invalid regions). 
    
    Supports two modes:
    - 'file': Load mask from .mat file (created by Flask masking endpoint)
    - 'rectangular': Create mask from edge pixel specifications
    
    Parameters
    ----------
    camera_num : int
        Camera number (e.g., 1 for Cam1)
    config : Config
        Configuration object
    source_path_idx : int, optional
        Index into source_paths list, defaults to 0
        
    Returns
    -------
    Optional[np.ndarray]
        Boolean mask array of shape (H, W) where True = masked region,
        or None if masking is disabled or mask cannot be loaded
    """
    if not config.masking_enabled:
        logging.debug("Masking is disabled in config")
        return None
    
    mask_mode = config.mask_mode
    
    # Rectangular mode: create mask from edge specifications
    if mask_mode == "rectangular":
        logging.debug("Using rectangular edge masking")
        return create_rectangular_mask(config)
    
    # File mode: load from .mat file
    elif mask_mode == "file":
        try:
            mask_path = config.get_mask_path(camera_num, source_path_idx)
            
            if not mask_path.exists():
                logging.warning(
                    "Mask file not found for Cam%s at %s. Proceeding without mask.",
                    camera_num, mask_path
                )
                return None

            logging.debug("Loading mask for Cam%s from %s", camera_num, mask_path)
            mask, polygons = read_mask_from_mat(str(mask_path))
            
            # Ensure mask is boolean
            mask = np.asarray(mask, dtype=bool)
            
            # Log mask statistics
            masked_pixels = np.sum(mask)
            total_pixels = mask.size
            mask_fraction = masked_pixels / total_pixels if total_pixels > 0 else 0
            
            logging.debug(
                "Mask loaded: {}/{} pixels masked ({:.1f}%)",
                masked_pixels, total_pixels, mask_fraction * 100
            )
            
            return mask
            
        except Exception as e:
            logging.error(
                "Failed to load mask for Cam{}: {}. Proceeding without mask.",
                camera_num, e
            )
            return None
    
    else:
        logging.warning(
            "Unknown mask mode '{}'. Must be 'file' or 'rectangular'. "
            "Proceeding without mask.", mask_mode
        )
        return None


def compute_vector_mask(
    pixel_mask: np.ndarray,
    config: Config,
    ensemble: bool = None,
) -> List[np.ndarray]:
    """
    Compute binary vector masks for each PIV pass based on pixel mask.

    This function is analogous to MATLAB's compute_b_mask. It convolves the
    pixel mask with box filters matching the interrogation window size for
    each pass, then interpolates at window center positions and applies a
    threshold to determine which vectors should be masked.

    Uses centralized window_utils for consistency with PIV processing.
    Supports both standard and single mode ensemble PIV.

    The process:
    1. For each pass, get the window size and overlap
    2. Compute window center positions using centralized utilities
    3. Convolve pixel mask with box filter of window size
    4. Interpolate the filtered mask at window centers
    5. Apply threshold to create binary mask (True = masked)

    Parameters
    ----------
    pixel_mask : np.ndarray
        Boolean pixel mask of shape (H, W) where True indicates masked regions
    config : Config
        Configuration object containing window sizes, overlap, and mask threshold
    ensemble : bool, optional
        If True, use ensemble config (ensemble_window_sizes, etc.).
        If False, use instantaneous config (window_sizes, etc.).
        If None (default), auto-detect from config.ensemble_piv flag.

    Returns
    -------
    List[np.ndarray]
        List of binary masks, one per pass. Each mask has shape (n_win_y, n_win_x)
        where True indicates this vector should be masked (set to 0/NaN)

    Notes
    -----
    The mask threshold (config.mask_threshold) determines the sensitivity:
    - 0.0: mask vector if any pixel in window is masked
    - 0.5: mask vector if >50% of pixels in window are masked
    - 1.0: only mask vector if all pixels in window are masked

    A typical value is 0.5, meaning vectors are masked if more than half
    of the interrogation window overlaps with masked regions.
    """
    if pixel_mask is None:
        return []

    # CRITICAL: Use config.image_shape for window grid computation to match
    # correlator/accumulator, NOT pixel_mask.shape. This ensures the vector
    # mask grid dimensions match the PIV data grid.
    H, W = config.image_shape
    logging.info(f"compute_vector_mask: Using config.image_shape = ({H}, {W})")
    logging.info(f"compute_vector_mask: pixel_mask.shape = {pixel_mask.shape}")

    # Validate mask dimensions match config.image_shape
    if pixel_mask.shape != (H, W):
        logging.warning(
            f"Pixel mask shape {pixel_mask.shape} differs from config.image_shape {(H, W)}. "
            f"Resizing mask to match image dimensions."
        )
        from scipy.ndimage import zoom
        zoom_factors = (H / pixel_mask.shape[0], W / pixel_mask.shape[1])
        # Use order=0 (nearest neighbor) to preserve binary nature of mask
        pixel_mask = zoom(pixel_mask.astype(np.float32), zoom_factors, order=0) > 0.5

    vector_masks = []
    threshold = config.mask_threshold

    # Determine if we're in ensemble mode
    # If explicit parameter is provided, use it; otherwise use config.ensemble_piv flag
    if ensemble is not None:
        is_ensemble = ensemble
    else:
        # Fall back to config flag (should be set correctly by caller)
        is_ensemble = hasattr(config, 'ensemble_piv') and config.ensemble_piv
    logging.info(f"compute_vector_mask: is_ensemble={is_ensemble} (explicit={ensemble}, config.ensemble_piv={getattr(config, 'ensemble_piv', None)})")

    if is_ensemble:
        num_passes = len(config.ensemble_window_sizes)
    else:
        num_passes = config.num_passes

    for pass_idx in range(num_passes):
        # Get window size and overlap for this pass
        if is_ensemble:
            win_y, win_x = config.ensemble_window_sizes[pass_idx]
            overlap = config.ensemble_overlaps[pass_idx]
            runtype = config.ensemble_type[pass_idx]
        else:
            win_y, win_x = config.window_sizes[pass_idx]
            overlap = config.overlap[pass_idx]
            runtype = 'standard'

        # Use centralized window center computation
        if runtype == 'single':
            # Single mode: use sum window for positioning
            result = compute_window_centers_single_mode(
                image_shape=(H, W),
                window_size=(win_y, win_x),
                sum_window=tuple(config.ensemble_sum_window),
                overlap=overlap,
                validate=True
            )
        else:
            # Standard mode
            result = compute_window_centers(
                image_shape=(H, W),
                window_size=(win_y, win_x),
                overlap=overlap,
                validate=True
            )

        win_ctrs_x = result.win_ctrs_x
        win_ctrs_y = result.win_ctrs_y
        n_win_x = result.n_win_x
        n_win_y = result.n_win_y

        # Use geometric overlap check instead of box filter convolution
        # This avoids asymmetry issues with even-sized kernels
        b_mask_pass = np.zeros((n_win_y, n_win_x), dtype=bool)

        # Calculate window bounds for each window center
        # A window of size W centered at C covers pixels [C - W/2, C + W/2)
        # Using floor to match C code: floor(C - (W-1)/2 + 0.5) to floor(C + (W-1)/2 + 0.5)
        half_win_y = (win_y - 1) / 2.0
        half_win_x = (win_x - 1) / 2.0

        for iy in range(n_win_y):
            y_center = win_ctrs_y[iy]
            y_min = int(np.floor(y_center - half_win_y + 0.5))
            y_max = int(np.floor(y_center + half_win_y + 0.5))

            for ix in range(n_win_x):
                x_center = win_ctrs_x[ix]
                x_min = int(np.floor(x_center - half_win_x + 0.5))
                x_max = int(np.floor(x_center + half_win_x + 0.5))

                # Check overlap: window [y_min, y_max] × [x_min, x_max] with pixel mask
                # Count masked pixels in window region
                y_min_clip = max(0, y_min)
                y_max_clip = min(H, y_max + 1)
                x_min_clip = max(0, x_min)
                x_max_clip = min(W, x_max + 1)

                if y_max_clip > y_min_clip and x_max_clip > x_min_clip:
                    window_region = pixel_mask[y_min_clip:y_max_clip, x_min_clip:x_max_clip]
                    overlap_fraction = np.sum(window_region) / (win_y * win_x)
                    b_mask_pass[iy, ix] = overlap_fraction > threshold

        vector_masks.append(b_mask_pass)

        # Log mask shape for debugging rectangular window support
        logging.info(
            f"compute_vector_mask: Pass {pass_idx}: window=({win_y}, {win_x}), "
            f"grid=({n_win_y}, {n_win_x}), mask_shape={b_mask_pass.shape}"
        )

        # Log statistics for this pass (debug level only)
        masked_vectors = np.sum(b_mask_pass)
        total_vectors = b_mask_pass.size
        mask_fraction = masked_vectors / total_vectors if total_vectors > 0 else 0

        # Find which rows are masked for debugging
        masked_rows_y = np.any(b_mask_pass, axis=1)  # Which Y indices have any masks
        masked_row_indices = np.where(masked_rows_y)[0]

        logging.debug(
            "Pass {}: {}/{} vectors masked ({:.1f}%), window size: ({}, {})",
            pass_idx + 1, masked_vectors, total_vectors,
            mask_fraction * 100, win_y, win_x
        )



    
    return vector_masks
