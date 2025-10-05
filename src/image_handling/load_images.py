from pathlib import Path
from typing import Tuple
import logging

import dask
import dask.array as da
import numpy as np
from dask.delayed import Delayed

from config import Config

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


def read_pair(idx: int, camera_path: Path, config: Config) -> np.ndarray:
    """Read a pair of images (A and B frames).

    Args:
        idx (int): Index of the image pair to read
        camera_path (Path): Path to camera directory
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) containing frame A and B
    """
    # Get image format - handle both time-resolved (single format) and non-time-resolved (A/B pair)
    image_format = config.image_format
    
    if isinstance(image_format, tuple):
        # Non-time-resolved: separate A and B formats
        image_format_A, image_format_B = image_format
        file_paths = [
            camera_path / (image_format_A % idx),
            camera_path / (image_format_B % idx),
        ]
    else:
        # Time-resolved: single format (shouldn't happen for PIV pairs, but handle it)
        logging.warning("Time-resolved format detected for PIV pairs, this may not work correctly")
        file_paths = [
            camera_path / (image_format % idx),
            camera_path / (image_format.replace("_A", "_B") % idx),
        ]

    # Check if it's a proprietary format that reads pairs natively
    file_ext = Path(file_paths[0]).suffix.lower()
    if file_ext == ".im7":
        # LaVision files contain both frames, read once
        # Extract camera number from path if needed
        camera_no = (
            int(str(camera_path).split("Cam")[-1]) if "Cam" in str(camera_path) else 1
        )
        return read_image(str(file_paths[0]), camera_no=camera_no)
    elif file_ext == ".cine":
        # For .cine, pass frame index (idx-1 for 0-based)
        return read_image(str(file_paths[0]), idx=idx - 1, frames=2)
    else:
        # Read individual frames
        frame_a = read_image(str(file_paths[0]))
        frame_b = read_image(str(file_paths[1]))
        return np.stack([frame_a, frame_b], axis=0)


def delayed_image_pair(idx: int, camera_path: Path, config: Config) -> Delayed:
    """Create a delayed task to read a pair of images.

    Args:
        idx (int): Index of the image pair to read

    Returns:
        Delayed: A delayed task representing the image pair
    """

    return dask.delayed(read_pair)(idx, camera_path, config)


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
        shape=(2, *config.image_shape),  # Always 2 for frame pairs
        dtype=config.image_dtype,
    )
    return arr


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
    
    camera_path = source / f"Cam{camera}"
    logging.info("Lazily loading images with Dask for camera: Cam%d from %s", camera, camera_path)
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, config)
        for idx in range(1, config.num_images + 1)
    ]
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
    pairs_stack = da.stack(dask_pairs, axis=0)
    pairs_stack = pairs_stack.rechunk(
        (config.piv_chunk_size, 2, *config.image_shape)
    )  # Always 2 for frame pairs
    return pairs_stack
