import logging
from pathlib import Path
from typing import Tuple, Union
import sys

import dask
import dask.array as da
import numpy as np
from dask.delayed import Delayed
import tifffile

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from config import Config

# Try to import advanced readers from src
HAS_ADVANCED_READERS = False
get_reader = None
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from image_handling.readers import get_reader
    HAS_ADVANCED_READERS = True
except ImportError:
    HAS_ADVANCED_READERS = False
    logging.warning("Advanced image readers not available, falling back to tifffile")


def read_image(file_path: str, **kwargs) -> np.ndarray:
    """Read an image file using appropriate reader.

    Args:
        file_path (str): Path to the image file
        **kwargs: Additional arguments for specific readers

    Returns:
        np.ndarray: The image data
    """
    if HAS_ADVANCED_READERS and get_reader is not None:
        reader_func = get_reader(file_path)
        return reader_func(file_path, **kwargs)
    else:
        return tifffile.imread(file_path)


def read_pair(
    idx: int, camera_path: Path, config: Config
) -> np.ndarray:
    """Read a pair of images (A and B frames).

    Args:
        idx (int): Index of the image pair to read
        camera_path (Path): Path to camera directory
        config (Config): Configuration object

    Returns:
        np.ndarray: Stacked array of shape (2, H, W) with frame A and B
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
        # Time-resolved: single format, derive B from A
        file_paths = [
            camera_path / (image_format % idx),
            camera_path / (image_format.replace("_A", "_B") % idx),
        ]

    # Check if it's a proprietary format that reads pairs natively
    file_ext = Path(file_paths[0]).suffix.lower()
    if HAS_ADVANCED_READERS and file_ext == ".im7":
        # LaVision files contain both frames, read once
        camera_no = (
            int(str(camera_path).split("Cam")[-1]) if "Cam" in str(camera_path) else 1
        )
        return read_image(str(file_paths[0]), camera_no=camera_no)
    elif HAS_ADVANCED_READERS and file_ext == ".cine":
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
        dask.delayed: A delayed task representing the image pair
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
        shape=(2, *config.image_shape),
        dtype=config.image_dtype,
    )
    return arr


def load_images(camera: str, config: Config) -> da.Array:
    """Load images for a specific camera.

    Args:
        camera (str): The camera folder name.
        config (Config): The configuration object.

    Returns:
        da.Array: A Dask array containing the loaded image pairs.
    """
    logging.info("Lazily loading images with Dask for camera: %s", camera)
    camera_path = config.base_path / camera
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, config)
        for idx in range(1, config.num_images + 1)
    ]
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
    pairs_stack = da.stack(dask_pairs, axis=0)
    pairs_stack = pairs_stack.rechunk((config.piv_chunk_size, 2, *config.image_shape))
    logging.info("Finished lazy loading images with Dask for camera: %s", camera)
    return pairs_stack
