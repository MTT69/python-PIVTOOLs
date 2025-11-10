from pathlib import Path
from typing import Tuple, Optional, List
from loguru import logger

import dask
import dask.array as da
import numpy as np
from dask.delayed import Delayed
from scipy.ndimage import convolve

from ..config import Config
from ..vector_loading import read_mask_from_mat

# Import all readers to register them
from .readers import get_reader

try:
    from line_profiler import profile
except ImportError:
    profile = lambda f: f


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

    Args:
        idx (int): Index of the image pair to read (1-based)
        camera_path (Path): Path to camera directory or source directory (for .set/.im7)
        camera (int): Camera number (1-based)
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) containing frame A and B
    """
    # Get image format - handle both time-resolved (single format) and non-time-resolved (A/B pair)
    image_format = config.image_format
    
    # Special handling for .set and .im7 files (all cameras in one file per time instance)
    # Check if image_format (string or tuple) contains '.set' or '.im7'
    if isinstance(image_format, tuple):
        format_str = image_format[0]  # Check first element for tuple
    else:
        format_str = image_format  # Single string for time-resolved
    
    if '.set' in str(format_str):
        # For .set files, camera_path is the source directory
        set_file_path = camera_path / format_str
        return read_image(str(set_file_path), camera_no=camera, im_no=idx)
    
    if '.im7' in str(format_str):
        # For .im7 files, camera_path is the source directory
        # Each .im7 file contains all cameras for one time instance
        im7_file_path = camera_path / (format_str % idx)
        return read_image(str(im7_file_path), camera_no=camera)
    
    if isinstance(image_format, tuple):
        # Non-time-resolved: separate A and B formats
        image_format_A, image_format_B = image_format
        file_paths = [
            camera_path / (image_format_A % idx),
            camera_path / (image_format_B % idx),
        ]
    else:
        file_paths = [
            camera_path / (image_format % idx),
            camera_path / (image_format % (idx + 1)),
        ]

    # Check if it's a proprietary format that reads pairs natively
    file_ext = Path(file_paths[0]).suffix.lower()
    if file_ext == ".cine":
        return read_image(str(file_paths[0]), idx=idx - 1, frames=2)
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


@profile
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
            Shape: (num_images, 2, H, W)
            Note: This is a lazy array - no actual image data loaded yet.
            Each element is an independent delayed task.
    """
    if source is None:
        source = config.source_paths[0]
    
    # For .set and .im7 files, there are no camera subdirectories
    # All cameras are stored in a single file per time instance in the source directory
    # File format: source_directory/B00001.im7 (contains all cameras for time instance 1)
    if '.set' in str(config.image_format):
        camera_path = source  # No camera subdirectory for set files
    elif '.im7' in str(config.image_format):
        camera_path = source  # No camera subdirectory for set files
    else:
        camera_path = source / f"Cam{camera}"
    
    num_images = config.num_images
    
    logger.info(f"Creating {num_images} delayed tasks for lazy loading (Cam{camera})")
    
    # Create one delayed task per image pair (pure lazy loading)
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, camera, config)
        for idx in range(1, num_images + 1)
    ]
    
    # Convert each delayed task to a Dask array
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
    
    # Stack into single array - still lazy, no computation yet!
    pairs_stack = da.stack(dask_pairs, axis=0)
    
    logger.info(
        f"Lazy loading complete: {num_images} independent delayed tasks created "
        f"(~{num_images} KB memory footprint)"
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
    
    logger.debug(
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
        logger.debug("Masking is disabled in config")
        return None
    
    mask_mode = config.mask_mode
    
    # Rectangular mode: create mask from edge specifications
    if mask_mode == "rectangular":
        logger.debug("Using rectangular edge masking")
        return create_rectangular_mask(config)
    
    # File mode: load from .mat file
    elif mask_mode == "file":
        try:
            mask_path = config.get_mask_path(camera_num, source_path_idx)
            
            if not mask_path.exists():
                logger.warning(
                    "Mask file not found for Cam{} at {}. Proceeding without mask.",
                    camera_num, mask_path
                )
                return None
            
            logger.debug("Loading mask for Cam{} from {}", camera_num, mask_path)
            mask, polygons = read_mask_from_mat(str(mask_path))
            
            # Ensure mask is boolean
            mask = np.asarray(mask, dtype=bool)
            
            # Log mask statistics
            masked_pixels = np.sum(mask)
            total_pixels = mask.size
            mask_fraction = masked_pixels / total_pixels if total_pixels > 0 else 0
            
            logger.debug(
                "Mask loaded: {}/{} pixels masked ({:.1f}%)",
                masked_pixels, total_pixels, mask_fraction * 100
            )
            
            return mask
            
        except Exception as e:
            logger.error(
                "Failed to load mask for Cam{}: {}. Proceeding without mask.",
                camera_num, e
            )
            return None
    
    else:
        logger.warning(
            "Unknown mask mode '{}'. Must be 'file' or 'rectangular'. "
            "Proceeding without mask.", mask_mode
        )
        return None

@profile
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
    
    The process:
    1. For each pass, get the window size and overlap
    2. Compute window center positions (same as PIV does)
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
    
    for pass_idx in range(config.num_passes):
        # Get window size and overlap for this pass
        # config.window_sizes is in (H, W) format = (win_y, win_x)
        win_y, win_x = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]
        
        # Calculate window spacing
        win_spacing_x = round((1 - overlap / 100) * win_x)
        win_spacing_y = round((1 - overlap / 100) * win_y)
        
        # Calculate window center positions (matching PIV computation exactly)
        # For a 128-pixel window (indices 0-127), center is at 63.5
        # First window center in X (width dimension) - 0-based array indexing
        first_ctr_x = (win_x - 1) / 2.0  # For 128: (127)/2 = 63.5
        # Last possible window center in X
        last_ctr_x = W - (win_x + 1) / 2.0  # For W=4872, win=128: 4872 - 64.5 = 4807.5
        
        # First window center in Y (height dimension) - 0-based array indexing
        first_ctr_y = (win_y - 1) / 2.0
        # Last possible window center in Y
        last_ctr_y = H - (win_y + 1) / 2.0
        
        # Calculate number of windows
        n_win_x = int(np.floor((last_ctr_x - first_ctr_x) / win_spacing_x)) + 1
        n_win_y = int(np.floor((last_ctr_y - first_ctr_y) / win_spacing_y)) + 1
        
        # Ensure at least one window
        n_win_x = max(1, n_win_x)
        n_win_y = max(1, n_win_y)
        
        # Window center positions using linspace (matches MATLAB's colon operator)
        win_ctrs_x = np.linspace(
            first_ctr_x, first_ctr_x + win_spacing_x * (n_win_x - 1), n_win_x
        )
        win_ctrs_y = np.linspace(
            first_ctr_y, first_ctr_y + win_spacing_y * (n_win_y - 1), n_win_y
        )
        
        box_filter_y = np.ones((win_y, 1), dtype=np.float32) / win_y
        f_mask = convolve(im_mask, box_filter_y, mode='constant', cval=0.0)
        
        # Convolve along x (columns)
        box_filter_x = np.ones((1, win_x), dtype=np.float32) / win_x
        f_mask = convolve(f_mask, box_filter_x, mode='constant', cval=0.0)
        
        # Interpolate at window center positions using nearest neighbor
        # Create grid of window centers
        win_y_grid, win_x_grid = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing='ij')
        
        # Convert to integer indices for nearest neighbor
        win_y_idx = np.clip(np.round(win_y_grid).astype(int), 0, H - 1)
        win_x_idx = np.clip(np.round(win_x_grid).astype(int), 0, W - 1)
        
        # Sample the filtered mask
        b_mask_pass = f_mask[win_y_idx, win_x_idx] > threshold
        
        vector_masks.append(b_mask_pass)
        
        # Log statistics for this pass (debug level only)
        masked_vectors = np.sum(b_mask_pass)
        total_vectors = b_mask_pass.size
        mask_fraction = masked_vectors / total_vectors if total_vectors > 0 else 0
        
        logger.debug(
            "Pass {}: {}/{} vectors masked ({:.1f}%), window size: ({}, {})",
            pass_idx + 1, masked_vectors, total_vectors,
            mask_fraction * 100, win_y, win_x
        )
    
    return vector_masks
