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
    active_pass_idx: Optional[int] = None,
):
    """Create correlator backend, optionally with precomputed cache.

    :param config: Configuration object
    :param precomputed_cache: Optional precomputed cache data to avoid redundant computation
    :param ensemble: If True, create ensemble correlator instead of instantaneous
    :param vector_masks: Pre-computed vector masks for each pass (ensemble only)
    :param active_pass_idx: If set, only allocate correlation buffers for this pass (ensemble only)
    :return: Correlator backend instance
    """
    backend = getattr(config, "backend", "cpu").lower()

    if ensemble:
        # Import here to avoid circular imports and surface a clear build error
        try:
            from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU
            return EnsembleCorrelatorCPU(
                config=config,
                precomputed_cache=precomputed_cache,
                vector_masks=vector_masks,
                active_pass_idx=active_pass_idx,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Ensemble PIV requires the compiled C libraries (libbulkxcorr2d, "
                f"libfusedwarp). Rebuild with 'pip install -e .'. Error: {e}"
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
    active_pass_idx: Optional[int] = None,
):
    """Create ensemble correlator backend.

    Convenience function specifically for ensemble PIV.

    :param config: Configuration object
    :param precomputed_cache: Optional precomputed cache data
    :param vector_masks: Pre-computed vector masks for each pass
    :param active_pass_idx: If set, only allocate correlation buffers for this pass
    :return: EnsembleCorrelatorCPU instance
    """
    return make_correlator_backend(
        config=config,
        precomputed_cache=precomputed_cache,
        ensemble=True,
        vector_masks=vector_masks,
        active_pass_idx=active_pass_idx,
    )
