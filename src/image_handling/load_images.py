from pathlib import Path
from typing import Tuple

import dask
import dask.array as da
import numpy as np
import tifffile
from dask.delayed import Delayed

from config import Config


def read_image(file_path: str) -> np.ndarray:
    """Read an image file using tifffile.

    Args:
        file_path (str): Path to the image file

    Returns:
        tifffile.TiffFile: The image file
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    return tifffile.imread(file_path)


def read_pair(idx: int, camera_path: Path, config: Config) -> Tuple[np.ndarray, ...]:
    """Read a pair of images (A and B frames).

    Args:
        idx (int): Index of the image pair to read

    Returns:
        tuple: A tuple containing two numpy arrays representing the images
    """
    image_format_A, image_format_B = (
        config.image_format
    )  # Should be a tuple of two formats
    file_paths = [
        camera_path / (image_format_A % idx),
        camera_path / (image_format_B % idx),
    ]
    return np.stack([read_image(file_paths[0]), read_image(file_paths[1])], axis=0)


def delayed_image_pair(idx: int, camera_path: Path, config: Config) -> dask.delayed:
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

    Returns:
        da.Array: A Dask array containing the loaded image pairs.
    """

    camera_path = source / f"Cam{camera}"
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
