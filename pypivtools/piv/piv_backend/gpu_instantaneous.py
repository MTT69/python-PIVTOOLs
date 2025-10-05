import dask.array as da
import numpy as np

from pypivtools.config import Config
from pypivtools.piv.piv_backend.base import CrossCorrelator


class InstantaneousCorrelatorGPU(CrossCorrelator):
    def correlate_batch(self, images: np.ndarray, config: Config) -> da.Array:

        pass
