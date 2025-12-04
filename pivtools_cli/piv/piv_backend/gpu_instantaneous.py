import sys
from pathlib import Path

import dask.array as da
import numpy as np

from pivtools_core.config import Config

from pivtools_cli.piv.piv_backend.base import CrossCorrelator


class InstantaneousCorrelatorGPU(CrossCorrelator):
    def correlate_batch(self, images: np.ndarray, config: Config) -> da.Array:

        pass
