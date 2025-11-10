import sys
from pathlib import Path
from typing import Optional

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.piv_backend.gpu_instantaneous import InstantaneousCorrelatorGPU

# Global cache for correlator instances to avoid redundant caching
_correlator_cache = {}
_correlator_cache_data = {}


def make_correlator_backend(config: Config, precomputed_cache: Optional[dict] = None):
    """Create correlator backend, optionally with precomputed cache.
    
    :param config: Configuration object
    :param precomputed_cache: Optional precomputed cache data to avoid redundant computation
    :return: Correlator backend instance
    """
    backend = getattr(config, "backend", "cpu").lower()

    if backend == "cpu":
        return InstantaneousCorrelatorCPU(config=config, precomputed_cache=precomputed_cache)
    elif backend == "gpu":
        return InstantaneousCorrelatorGPU()
    else:
        raise ValueError(f"Unknown backend: {backend}")
