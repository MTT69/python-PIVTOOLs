import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from pivtools_core.config import Config


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
            threads_per_worker=config.dask_threads_per_worker,
            memory_limit=config.dask_memory_limit,
            nanny=False,
            processes=True,
            config={"distributed.worker.profile.enabled": False},
            dashboard_address=":8788"
        )

    elif config.cluster_type == "slurm":
        if not hasattr(config, "n_nodes"):
            raise ValueError("config.n_nodes must be set for Slurm cluster")
        slurm_env = [
            'export DASK_DISTRIBUTED__COMM__ALLOWED_TRANSPORTS=["tcp://[::]:0"]',
            'echo "Landed on $HOSTNAME"',
            f'source {os.getenv("HOME")}/.bashrc',
            f"source /home/co1f23/scratch/MorganTaylor/PyPIVtools/examples/setup_env.sh",
        ]
        import socket

        cluster = SLURMCluster(
            queue=getattr(config, "queue", None),
            project=getattr(config, "project", None),
            cores=1,  # config.dask_workers_per_node,
            processes=config.dask_workers_per_node,
            memory=config.dask_memory_limit,
            walltime=getattr(config, "walltime", "01:00:00"),
            interface="ib0",
            job_extra=[
                "--qos=expert",
                f"--nodes={config.n_nodes}",
                "--output=dask_job_output_%j.out",
                "--error=dask_job_output_%j.err",
                "--exclusive",
            ],
            job_script_prologue=slurm_env,
            scheduler_options={"host": socket.gethostname()},
        )
        logging.info(f"Number of nodes in Slurm cluster: {config.n_nodes}")

        cluster.scale(jobs=config.n_nodes)
        logging.info(f"Slurm job script:\n{cluster.job_script()}")

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
    cluster = None
    client = None

    try:
        cluster, client = make_cluster(
            n_workers_per_node=n_workers_per_node,
            threads_per_worker=1,
            memory_limit=memory_limit,
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
