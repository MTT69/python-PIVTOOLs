import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from pivtools_core.config import Config


def _suppress_dask_verbose_logging():
    """Suppress verbose Dask internal logging to reduce noise."""
    # Suppress worker startup/shutdown messages
    for logger_name in [
        "distributed.worker",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.core",
        "distributed.comm",
        "distributed.http.proxy",
        "bokeh.server.views.ws",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


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
            nanny=False,
            processes=True,
            config={
                "distributed.worker.profile.enabled": False,
                # Increase heartbeat tolerance for long-running C calls
                # that block the worker thread (e.g., dense PIV passes with 700+ windows)
                "distributed.scheduler.worker-ttl": "120s",  # Default: 60s
                "distributed.worker.heartbeat": "5s",  # Default: 1s
                "distributed.comm.timeouts.connect": "60s",  # Default: 30s
            },
            dashboard_address=":8788"
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
        dask.config.set({
            "distributed.scheduler.worker-ttl": "120s",
            "distributed.worker.heartbeat": "5s",
            "distributed.comm.timeouts.connect": "60s",
        })
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
            config= config#n_workers_per_node=n_workers_per_node,
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

    logger.info("Worker logging configured successfully")


def set_worker_omp_threads(omp_threads: str):
    """
    Set OMP_NUM_THREADS in worker processes.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = omp_threads
    logging.info(f"Set OMP_NUM_THREADS to {omp_threads} in worker process")
