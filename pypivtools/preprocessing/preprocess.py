import sys
from pathlib import Path

import dask.array as da

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from config import Config
from image_handling.load_images import load_images

from pypivtools.preprocessing.filters import filter_images


def preprocess_images(images: da.Array, config: Config):
    """
    Preprocess images based on the provided configuration.

    Args:
        images (da.Array): Dask array containing the images.
        config (Config): Configuration object.
    """

    images = filter_images(images, config)

    return images
