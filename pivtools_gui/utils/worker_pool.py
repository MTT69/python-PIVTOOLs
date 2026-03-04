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


def get_max_workers(n_items, config=None):
    """Determine the number of ProcessPoolExecutor workers.

    Parameters
    ----------
    n_items : int
        Number of work items (files, frames). Workers are capped at this value
        since more workers than items is wasteful.
    config : Config, optional
        PIVTOOLs Config object. If provided, reads
        ``config.data["processing"]["post_processing_workers"]``.

    Returns
    -------
    int
        Number of workers, at least 1.
    """
    # Read from config if available
    cap = None
    if config is not None:
        try:
            cap = config.data.get("processing", {}).get("post_processing_workers")
        except (AttributeError, TypeError):
            pass

    # Fall back to min(cpu_count, 16)
    if cap is None:
        cap = min(os.cpu_count() or 4, 16)

    return max(1, min(cap, n_items))
