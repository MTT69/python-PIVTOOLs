import logging
from pathlib import Path
from typing import Tuple

import dask
import dask.array as da
import numpy as np
import tifffile
from dask.delayed import Delayed
from dask.distributed import get_worker
from typeguard import config

from pypivtools.config import Config


def read_image(file_path: str) -> np.ndarray:
    """Read an image file using tifffile.

    Args:
        file_path (str): Path to the image file

    Returns:
        tifffile.TiffFile: The image file
    """
    return tifffile.imread(file_path)


def read_pair(
    idx: int, camera_path: Path, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a pair of images

    Args:
        idx (int): Index of the image pair to read

    Returns:
        tuple: A tuple containing two numpy arrays representing the images
    """

    file_paths = [f"{idx:04d}_A.tif", f"{idx:04d}_B.tif"]
    image_format = config.image_format

    file_paths = [
        camera_path / (image_format % idx),
        camera_path / (image_format.replace("_A", "_B") % idx),
    ]
    return np.stack(
        [
            read_image(file_paths[0]).astype(config.image_dtype),
            read_image(file_paths[1]).astype(config.image_dtype),
        ],
        axis=0,
    )


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
