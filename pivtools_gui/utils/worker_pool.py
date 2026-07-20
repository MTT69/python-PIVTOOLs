"""Shared worker pool utilities for post-processing scripts.

Provides consistent worker count capping and thread-control initialization
for ProcessPoolExecutor workers. Prevents thread oversubscription on HPC
nodes where N worker processes × M internal threads would otherwise compete
for N cores.

IMPORT-LIGHT ON PURPOSE: under Windows spawn, a pool child unpickles the
initializer *function reference* (and its initargs) during bootstrap, BEFORE the
initializer body runs. Resolving that reference imports this module — so if this
module (or anything in initargs) pulled in numpy, BLAS thread pools would load
from the environment before ``worker_initializer`` can pin them to 1. Keep this
module stdlib-only and ship payloads as pre-pickled bytes
(``payload_worker_initializer``).
"""

import os
import pickle

# Holds the worker's threadpoolctl limiter for the process lifetime (see
# worker_initializer); module-level so it is never garbage collected.
_THREADPOOL_LIMITER = None


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

    # Env vars only affect libraries loaded AFTER this point — but under Windows
    # spawn the child resolves the initializer's module reference during bootstrap,
    # and the pivtools_gui.utils package __init__ imports numpy, so BLAS is already
    # loaded (at its default thread count) by the time this runs. Clamp any
    # already-loaded pools via the threadpoolctl API, which works post-import.
    # The limiter is kept referenced module-level: the limit must hold for the
    # worker's whole life, not until the object is collected.
    global _THREADPOOL_LIMITER
    from threadpoolctl import threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)

    # OpenCV's internal threading (TBB/pthreads) is controlled via API,
    # not environment variable. Safe to call even if cv2 isn't imported yet.
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


# Shared per-job payload for a payload-initialized pool worker (one pool per job;
# worker processes are pool-private, so this global cannot cross-contaminate jobs).
_WORKER_PAYLOAD = None


def payload_worker_initializer(payload_bytes):
    """Pool initializer: pin internal thread pools, THEN unpickle the job payload.

    The payload arrives pre-pickled (bytes) so nothing numpy-laden rides in
    ``initargs`` — raw ndarrays there would import numpy (loading BLAS thread
    pools from the environment) during child bootstrap, before the pinning env
    vars are set. Bytes unpickle without importing anything.
    """
    global _WORKER_PAYLOAD
    worker_initializer()
    _WORKER_PAYLOAD = pickle.loads(payload_bytes)


def get_worker_payload():
    """Return the payload installed by ``payload_worker_initializer``.

    Raises loudly when called outside a payload-initialized pool worker — a task
    function running with no payload is a wiring bug, not a case to default.
    """
    if _WORKER_PAYLOAD is None:
        raise RuntimeError(
            "no worker payload installed — task ran outside a "
            "payload_worker_initializer pool and was given no explicit context"
        )
    return _WORKER_PAYLOAD


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
