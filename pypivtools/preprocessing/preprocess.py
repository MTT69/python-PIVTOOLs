import dask.array as da

from pypivtools.config import Config
from pypivtools.image_handling.load_images import load_images
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
