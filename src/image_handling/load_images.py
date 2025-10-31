from pathlib import Path
from typing import Tuple, Optional, List
import logging

import dask
import dask.array as da
import numpy as np
from dask.delayed import Delayed
from scipy.ndimage import convolve

from config import Config
from vector_loading import read_mask_from_mat

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

    Args:
        idx (int): Index of the image pair to read
        camera_path (Path): Path to camera directory or set file
        camera (int): Camera number
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) containing frame A and B
    """
    # Get image format - handle both time-resolved (single format) and non-time-resolved (A/B pair)
    image_format = config.image_format
    
    # Special handling for .ims/.set files (set files in source directory)
    if str(image_format).endswith(('.set')):
        # For .ims/.set files, camera_path is the source directory
        # image_format should be a string for .set files
        set_file_path = camera_path / image_format
        return read_image(str(set_file_path), camera_no=camera, im_no=idx)
    
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
    if file_ext == ".im7":
        camera_no = camera  # Use the passed camera number
        return read_image(str(file_paths[0]), camera_no=camera_no)
    elif file_ext == ".cine":
        return read_image(str(file_paths[0]), idx=idx - 1, frames=2)
    else:
        # Read individual frames
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
    """Load images for a specific camera.

    Args:
        camera (int): The camera number.
        config (Config): The configuration object.
        source (Path, optional): The root directory for camera folders.
            If None, uses first source_path from config.

    Returns:
        da.Array: A Dask array containing the loaded image pairs.
    """
    if source is None:
        source = config.source_paths[0]
    
    # For .ims/.set files, there are no camera subdirectories
    if str(config.image_format).endswith(('.ims', '.set')):
        camera_path = source  # No camera subdirectory for set files
        logging.info("Loading images from set file in directory: %s for camera: Cam%d", source, camera)
    else:
        camera_path = source / f"Cam{camera}"
        logging.info("Lazily loading images with Dask for camera: Cam%d from %s", camera, camera_path)
    
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, camera, config)
        for idx in range(1, config.num_images + 1)
    ]
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
    pairs_stack = da.stack(dask_pairs, axis=0)
    pairs_stack = pairs_stack.rechunk(
        (config.piv_chunk_size, 2, *config.image_shape)
    )  # Always 2 for frame pairs
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
        "Created rectangular mask: top=%d, bottom=%d, left=%d, right=%d "
        "(%d/%d pixels = %.1f%%)",
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
                    "Mask file not found for Cam%d at %s. Proceeding without mask.",
                    camera_num, mask_path
                )
                return None
            
            logging.debug("Loading mask for Cam%d from %s", camera_num, mask_path)
            mask, polygons = read_mask_from_mat(str(mask_path))
            
            # Ensure mask is boolean
            mask = np.asarray(mask, dtype=bool)
            
            # Log mask statistics
            masked_pixels = np.sum(mask)
            total_pixels = mask.size
            mask_fraction = masked_pixels / total_pixels if total_pixels > 0 else 0
            
            logging.debug(
                "Mask loaded: %d/%d pixels masked (%.1f%%)",
                masked_pixels, total_pixels, mask_fraction * 100
            )
            
            return mask
            
        except Exception as e:
            logging.error(
                "Failed to load mask for Cam%d: %s. Proceeding without mask.",
                camera_num, e
            )
            return None
    
    else:
        logging.warning(
            "Unknown mask mode '%s'. Must be 'file' or 'rectangular'. "
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
        
        # Calculate window center positions (matching PIV computation)
        # Using 0-based indexing: centers start at half window size
        start_x = win_x / 2 - 0.5
        start_y = win_y / 2 - 0.5
        
        # Temporarily disable edge margin (set to full image bounds)
        # Original code (commented):
        # # Apply edge margin if needed (matching PIV's EDGE_MARGIN = 32)
        # EDGE_MARGIN = 0
        # min_x = EDGE_MARGIN
        # max_x = W - EDGE_MARGIN - 1
        # min_y = EDGE_MARGIN
        # max_y = H - EDGE_MARGIN - 1
        min_x = 0
        max_x = W - 1
        min_y = 0
        max_y = H - 1
        
        start_x = max(start_x, min_x)
        start_y = max(start_y, min_y)
        
        # Calculate number of windows
        n_win_x = int(np.floor((max_x - start_x) / win_spacing_x)) + 1
        n_win_y = int(np.floor((max_y - start_y) / win_spacing_y)) + 1
        
        n_win_x = max(1, n_win_x)
        n_win_y = max(1, n_win_y)
        
        # Window center positions
        win_ctrs_x = np.linspace(
            start_x, start_x + win_spacing_x * (n_win_x - 1), n_win_x
        )
        win_ctrs_y = np.linspace(
            start_y, start_y + win_spacing_y * (n_win_y - 1), n_win_y
        )
        
        # Perform 2D convolution with box filter (separable for efficiency)
        # Convolve along y (rows) first
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
        
        logging.debug(
            "Pass %d: %d/%d vectors masked (%.1f%%), window size: (%d, %d)",
            pass_idx + 1, masked_vectors, total_vectors,
            mask_fraction * 100, win_y, win_x
        )
    
    return vector_masks
