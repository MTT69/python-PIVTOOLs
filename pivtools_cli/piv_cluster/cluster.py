import logging
import os
import warnings
from collections import defaultdict
from typing import List, Tuple

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

from pivtools_core.config import Config


class _DaskNoiseFilter(logging.Filter):
    """Filter out noisy Dask worker messages."""

    _SUPPRESSED = (
        "Event loop was unresponsive",
        "Unmanaged memory use is high",
    )

    def filter(self, record):
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)


def _suppress_dask_verbose_logging():
    """Suppress verbose Dask internal logging to reduce noise."""
    import dask

    # Set Dask config-level logging — propagates to worker subprocesses
    # before they emit any log messages (unlike silence_logs which may be too late)
    dask.config.set({"logging": {"distributed": "warning"}})

    # Suppress worker startup/shutdown messages in main process
    for logger_name in [
        "distributed",
        "distributed.worker",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.core",
        "distributed.comm",
        "distributed.http.proxy",
        "bokeh.server.views.ws",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Filter out noisy Dask warnings (unresponsive event loop, unmanaged memory)
    logging.getLogger("distributed").addFilter(_DaskNoiseFilter())

    # Suppress "Sending large graph" UserWarning from distributed.client
    warnings.filterwarnings(
        "ignore", message="Sending large graph", category=UserWarning
    )


def make_cluster(
    config: Config,
) -> Tuple[object, Client]:
    """
    Create a Dask cluster (local or Slurm) based on config.cluster_type.

    Returns:
        cluster: Dask Cluster object
        client: Dask Client
    """
    if getattr(config, "cluster_type", "local") == "local":
        cluster = LocalCluster(
            n_workers=config.dask_workers_per_node,
            threads_per_worker=1,
            memory_limit=config.dask_memory_limit,
            nanny=config.dask_nanny,
            processes=True,
            silence_logs=logging.WARNING,
            config={
                "distributed.worker.profile.enabled": False,
                # Increase heartbeat tolerance for long-running C calls
                # that block the worker thread (e.g., dense PIV passes with 700+ windows)
                "distributed.scheduler.worker-ttl": "30m",  # Default: 60s
                "distributed.worker.heartbeat": "5s",  # Default: 1s
                "distributed.comm.timeouts.connect": "60s",  # Default: 30s
                "distributed.comm.timeouts.tcp": "600s",  # Default: 30s
            },
            dashboard_address=":8788",
        )
        client = Client(cluster)
        logging.info(f"Local Dask cluster started with {len(cluster.workers)} workers.")
        return cluster, client
    elif config.cluster_type == "slurm":
        if not hasattr(config, "n_nodes"):
            raise ValueError("config.n_nodes must be set for Slurm cluster")
        import socket

        import dask

        # Set heartbeat tolerance for long-running C calls on SLURM too
        dask.config.set(
            {
                "distributed.scheduler.worker-ttl": "30m",
                "distributed.worker.heartbeat": "5s",
                "distributed.comm.timeouts.connect": "60s",
                "distributed.comm.timeouts.tcp": "600s",
            }
        )
        cluster = SLURMCluster(
            queue=config.slurm_partition,
            walltime=config.slurm_walltime,
            cores=1,
            processes=config.dask_workers_per_node,
            memory=config.slurm_memory_limit,
            interface=config.slurm_interface,
            job_extra=config.slurm_job_extra,
            job_script_prologue=config.slurm_job_prologue,
            scheduler_options={"host": socket.gethostname()},
        )

        if config.n_nodes is not None:
            cluster.scale(jobs=config.n_nodes)

            client = Client(cluster)
            return cluster, client
    else:
        raise ValueError(f"Unknown cluster_type: {config.cluster_type}")


def group_workers_by_host(client: Client) -> dict[str, List[str]]:
    workers = client.scheduler_info()["workers"]
    grouped = defaultdict(list)
    for addr, info in workers.items():
        grouped[info["host"]].append(addr)
    return dict(grouped)


def select_workers_per_node(client: Client, n_workers_per_node: int = 1) -> List[str]:
    grouped = group_workers_by_host(client)
    selected = []
    for node_workers in grouped.values():
        selected.extend(node_workers[:n_workers_per_node])
    return selected


def start_cluster(
    n_workers_per_node: int = 1,
    memory_limit: str = "auto",
    config: Config = Config(),
    worker_omp_threads: str = None,
) -> tuple[LocalCluster, Client]:
    """
    Start a local Dask cluster.

    Returns:
        client: Dask Client
        piv_workers: list of workers to use for PIV
    """
    # Suppress verbose Dask internal logging
    _suppress_dask_verbose_logging()

    cluster = None
    client = None

    try:
        cluster, client = make_cluster(
            config=config  # n_workers_per_node=n_workers_per_node,
        )
        client.run(
            setup_worker_logging,
            log_level=getattr(logging, config.log_level, logging.INFO),
            log_file=config.log_file if hasattr(config, "log_file") else None,
            log_console=True,
        )

        if worker_omp_threads is not None:
            client.run(set_worker_omp_threads, omp_threads=worker_omp_threads)

        return cluster, client

    except Exception as e:
        print(f"Error starting Dask cluster: {e}")
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
        raise


def setup_worker_logging(log_level=logging.INFO, log_file=None, log_console=True):
    """
    Configure logging inside a Dask worker process.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if log_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Suppress noisy Dask internal logging inside worker processes too
    _suppress_dask_verbose_logging()

    logger.debug("Worker logging configured successfully")


def set_worker_omp_threads(omp_threads: str):
    """
    Pin native threadpools (OpenMP + BLAS/MKL/NumExpr) in worker processes.

    Each Dask worker already runs one task at a time (threads_per_worker=1), so
    every native pool a worker opens multiplies against the worker count. Pinning
    them all to the same value keeps total threads bounded by the worker count
    and avoids core oversubscription (the pure-NumPy k-space fitter's FFT/solves
    are additionally capped via threadpoolctl, but these env vars are the
    defence-in-depth set, honoured by libraries that read them at import).
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = omp_threads
    logging.debug(
        f"Set OMP/BLAS/MKL/NumExpr threads to {omp_threads} in worker process"
    )
