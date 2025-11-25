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


def read_pair(idx: int, camera_path: Path, camera: int, config: Config) -> np.ndarray:
    """Read a pair of images (A and B frames).

    This function handles three main file organization strategies:

    1. Multi-camera container files (.set, .im7):
       - All cameras stored in ONE file per time instance
       - .set: source_dir/xxx.set contains all cameras and all time instances
       - .im7: source_dir/B00001.im7 contains all cameras for time instance 1
       - No camera subdirectories (Cam1/, Cam2/, etc.)

    2. Camera-specific directories with standard formats (.tif, .png, .jpg):
       - Organized as: source_dir/Cam1/00001.tif, source_dir/Cam2/00001.tif
       - Each camera has its own subdirectory

    3. Time-resolved formats (.cine):
       - Camera-specific directories with video files
       - Organized as: source_dir/Cam1/recording.cine

    Frame Pairing:
        The idx parameter is ALWAYS 1-based internally (idx=1 means first pair).
        The actual file indices to read are determined by config.get_frame_pair_indices(),
        which handles:
        - Zero-based vs 1-based file indexing
        - Sequential vs skip pairing modes
        - Non-time-resolved A/B pairs vs time-resolved sequences

    Args:
        idx (int): Pair number (1-based, where 1 = first pair)
        camera_path (Path): Path to camera directory or source directory (for .set/.im7)
        camera (int): Camera number (1-based)
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) containing frame A and B
    """
    # Get image format - now always a tuple
    format_str = config.image_format[0]

    # Special handling for .set and .im7 files (all cameras in one file per time instance)
    if '.set' in str(format_str):
        # For .set files, camera_path is the source directory
        set_file_path = camera_path / format_str
        return read_image(str(set_file_path), camera_no=camera, im_no=idx)

    if '.im7' in str(format_str):
        # For .im7 files, camera_path is the source directory
        # Each .im7 file contains all cameras for one time instance
        # Use frame pairing logic for im7 files
        frame_a_idx, frame_b_idx = config.get_frame_pair_indices(idx)
        # Read frame A (im7 returns pair from single file, so we read it once)
        im7_file_path = camera_path / (format_str % frame_a_idx)
        return read_image(str(im7_file_path), camera_no=camera)

    # Get the file indices for this pair using new pairing logic
    frame_a_idx, frame_b_idx = config.get_frame_pair_indices(idx)

    # Check if we have A/B pair (len > 1) or single format
    if len(config.image_format) == 2:
        # Non-time-resolved: separate A and B formats
        # Both use the same file index (pair 1 = file 1A + file 1B)
        image_format_A, image_format_B = config.image_format
        file_paths = [
            camera_path / (image_format_A % frame_a_idx),
            camera_path / (image_format_B % frame_b_idx),
        ]
    else:
        # Time-resolved: single format, read two consecutive (or skipped) frames
        file_paths = [
            camera_path / (format_str % frame_a_idx),
            camera_path / (format_str % frame_b_idx),
        ]

    # Check if it's a proprietary format that reads pairs natively
    file_ext = Path(file_paths[0]).suffix.lower()
    if file_ext == ".cine":
        # For .cine files, use the first frame index
        # The frames parameter will read consecutive frames starting from that index
        return read_image(str(file_paths[0]), idx=frame_a_idx - 1, frames=2)
    else:
        # Read individual frames (e.g., .tif, .png, .jpg)
        frame_a = read_image(str(file_paths[0]))
        frame_b = read_image(str(file_paths[1]))
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
    
    # For .set and .im7 files, there are no camera subdirectories
    # All cameras are stored in a single file per time instance in the source directory
    # File format: source_directory/B00001.im7 (contains all cameras for time instance 1)
    format_str = config.image_format[0]
    if '.set' in format_str:
        camera_path = source  # No camera subdirectory for set files
    elif '.im7' in format_str:
        camera_path = source  # No camera subdirectory for im7 files
    else:
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
                    "Mask file not found for Cam{} at {}. Proceeding without mask.",
                    camera_num, mask_path
                )
                return None
            
            logging.debug("Loading mask for Cam{} from {}", camera_num, mask_path)
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

    # Ensure mask is float for convolution
    im_mask = pixel_mask.astype(np.float32)
    H, W = im_mask.shape

    vector_masks = []
    threshold = config.mask_threshold

    # Determine if we're in ensemble mode
    is_ensemble = hasattr(config, 'ensemble_piv') and config.ensemble_piv
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
