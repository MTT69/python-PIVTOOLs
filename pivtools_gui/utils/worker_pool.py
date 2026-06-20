"""Shared worker pool utilities for post-processing scripts.

Provides consistent worker count capping and thread-control initialization
for ProcessPoolExecutor workers. Prevents thread oversubscription on HPC
nodes where N worker processes × M internal threads would otherwise compete
for N cores.
"""

import os


def worker_initializer():
    """Initialize a ProcessPoolExecutor worker with single-threaded internals.

    Sets environment variables and library configs so that NumPy (OpenBLAS/MKL),
    SciPy, OpenCV, and OpenMP each use exactly 1 thread inside the worker process.
    Must be called before any of these libraries are imported in the worker, which
    is guaranteed when passed as ``initializer=`` to ProcessPoolExecutor.
    """
    # Pin all internal thread pools to 1 thread per worker process.
    # Process-level parallelism (N workers) handles concurrency;
    # internal threading would cause N×M oversubscription.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    # OpenCV's internal threading (TBB/pthreads) is controlled via API,
    # not environment variable. Safe to call even if cv2 isn't imported yet.
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def get_max_workers(n_items=None, config=None):
    """Determine the number of ProcessPoolExecutor workers from the
    ``processing.post_processing_workers`` config knob.

    Parameters
    ----------
    n_items : int, optional
        Number of work items (files, frames). When given, workers are capped at
        this value since more workers than items is wasteful. ``None`` applies no
        item cap (a long-lived pool reused across many items).
    config : Config, optional
        PIVTOOLs Config object. Fetched via ``get_config()`` (a cached singleton)
        when not supplied, so the knob is honoured without every call site having
        to thread a config through.

    Returns
    -------
    int
        Number of workers, at least 1.
    """
    if config is None:
        # Lazy import avoids any pivtools_core <-> pivtools_gui import cycle.
        # Always called in the parent process before the pool is built, so the
        # singleton lookup is safe here.
        from pivtools_core.config import get_config

        config = get_config()

    # Property resolves the knob to an int, falling back to min(cpu_count, 16).
    cap = config.post_processing_workers
    if n_items is not None:
        cap = min(cap, n_items)

    return max(1, cap)
