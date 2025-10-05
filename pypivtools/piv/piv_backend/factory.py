from pypivtools.config import Config
from pypivtools.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pypivtools.piv.piv_backend.gpu_instantaneous import InstantaneousCorrelatorGPU


def make_correlator_backend(config: Config):
    backend = getattr(config, "backend", "cpu").lower()

    if config.backend == "cpu":
        return InstantaneousCorrelatorCPU(config=config)
    elif config.backend == "gpu":
        return InstantaneousCorrelatorGPU()
    else:
        raise ValueError(f"Unknown backend: {backend}")
