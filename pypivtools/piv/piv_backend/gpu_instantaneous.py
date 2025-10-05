import sys
from pathlib import Path

import dask.array as da
import numpy as np

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from config import Config

from pypivtools.piv.piv_backend.base import CrossCorrelator


class InstantaneousCorrelatorGPU(CrossCorrelator):
    def correlate_batch(self, images: np.ndarray, config: Config) -> da.Array:

        pass
