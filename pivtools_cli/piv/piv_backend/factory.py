import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.piv_backend.gpu_instantaneous import InstantaneousCorrelatorGPU

# Global cache for correlator instances to avoid redundant caching
_correlator_cache = {}
_correlator_cache_data = {}


def make_correlator_backend(
    config: Config,
    precomputed_cache: Optional[dict] = None,
    ensemble: bool = False,
    vector_masks: Optional[List[np.ndarray]] = None,
):
    """Create correlator backend, optionally with precomputed cache.

    :param config: Configuration object
    :param precomputed_cache: Optional precomputed cache data to avoid redundant computation
    :param ensemble: If True, create ensemble correlator instead of instantaneous
    :param vector_masks: Pre-computed vector masks for each pass (ensemble only)
    :return: Correlator backend instance
    """
    backend = getattr(config, "backend", "cpu").lower()

    if ensemble:
        # Import here to avoid circular imports and allow graceful failure if GSL not installed
        try:
            from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU
            return EnsembleCorrelatorCPU(
                config=config,
                precomputed_cache=precomputed_cache,
                vector_masks=vector_masks,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Ensemble PIV requires the marquadt library which depends on GSL. "
                f"Install GSL: brew install gsl (macOS) or apt-get install libgsl-dev (Linux), "
                f"then rebuild with 'pip install -e .'. Error: {e}"
            )
        except ImportError as e:
            raise RuntimeError(f"Failed to import ensemble correlator: {e}")

    if backend == "cpu":
        return InstantaneousCorrelatorCPU(config=config, precomputed_cache=precomputed_cache)
    elif backend == "gpu":
        return InstantaneousCorrelatorGPU()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def make_ensemble_correlator(
    config: Config,
    precomputed_cache: Optional[dict] = None,
    vector_masks: Optional[List[np.ndarray]] = None,
):
    """Create ensemble correlator backend.

    Convenience function specifically for ensemble PIV.

    :param config: Configuration object
    :param precomputed_cache: Optional precomputed cache data
    :param vector_masks: Pre-computed vector masks for each pass
    :return: EnsembleCorrelatorCPU instance
    """
    return make_correlator_backend(
        config=config,
        precomputed_cache=precomputed_cache,
        ensemble=True,
        vector_masks=vector_masks,
    )


def make_stereo_ensemble_correlator(
    config: Config,
    cam1,
    cam2,
    output_size,
    world_bounds,
    self_cal_z: float = 0.0,
    self_cal_tilt_x: float = 0.0,
    self_cal_tilt_y: float = 0.0,
    precomputed_cache: Optional[dict] = None,
    vector_masks: Optional[List[np.ndarray]] = None,
):
    """Create stereo ensemble correlator with Correlation-of-Correlations.

    :param config: Configuration object
    :param cam1, cam2: PinholeCamera instances
    :param output_size: (H, W) of dewarped output
    :param world_bounds: (x_min, x_max, y_min, y_max) in mm
    :param self_cal_z/tilt_x/tilt_y: Self-calibration corrections
    :param precomputed_cache: Optional precomputed cache
    :param vector_masks: Per-pass vector masks
    :return: StereoEnsembleCorrelatorCPU instance
    """
    from pivtools_cli.piv.piv_backend.cpu_stereo_ensemble import (
        StereoEnsembleCorrelatorCPU,
    )

    return StereoEnsembleCorrelatorCPU(
        config=config,
        cam1=cam1,
        cam2=cam2,
        output_size=output_size,
        world_bounds=world_bounds,
        self_cal_z=self_cal_z,
        self_cal_tilt_x=self_cal_tilt_x,
        self_cal_tilt_y=self_cal_tilt_y,
        precomputed_cache=precomputed_cache,
        vector_masks=vector_masks,
    )
