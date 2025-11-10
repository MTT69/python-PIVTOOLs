import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from dask.distributed import Client, LocalCluster

from pivtools_core.config import Config


def make_cluster(
    threads_per_worker: int = 1,  # None,
    n_workers_per_node: int = 2,
    memory_limit: str = "auto",
) -> Tuple[LocalCluster, Client]:
    cluster = LocalCluster(
        n_workers=n_workers_per_node,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        nanny=True,
    )
    client = Client(cluster)
    return cluster, client


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
    threads_per_worker: int = None,
    memory_limit: str = "auto",
    config: Config = Config(),
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
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        )
        client.run(
            setup_worker_logging,
            log_level=getattr(logging, config.log_level, logging.INFO),
            log_file=config.log_file if hasattr(config, "log_file") else None,
            log_console=True,
        )

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
