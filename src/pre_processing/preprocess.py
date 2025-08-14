import dask.array as da

from config import Config
from pre_processing.filters import filter_images


def preprocess_images(images: da.Array, config: Config):
    """
    Preprocess images based on the provided configuration.

    Args:
        images (da.Array): Dask array containing the images.
        config (Config): Configuration object.
    """

    return filter_images(images, config)
