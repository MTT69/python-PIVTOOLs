import logging
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional
from dask import array as da
from dask.distributed import Client

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from config import Config

from pypivtools.piv.piv_backend.factory import make_correlator_backend
from pypivtools.piv.piv_result import PIVResult
from pypivtools.piv.save_results import save_piv_result_distributed


def _batch_policy(config: Config) -> int:
    if config.backend == "gpu":
        # ~ (batch_size * H * W * bytes_per_pixel * intermediates)
        # < GPU_RAM * safety
        return 8  # e.g. start with 8 and tune
    else:
        return 1


def perform_piv_and_save(
    images: da.Array,
    config: Config,
    client: Client,
    output_path: Path,
    start_frame: int = 1,
    pass_index: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> List:
    """
    Perform PIV and save results in parallel on workers.
    
    This function chains PIV computation with saving, avoiding the memory
    bottleneck of gathering all results to the main process before saving.
    Each worker processes and saves its results independently.
    
    Parameters
    ----------
    images : da.Array
        Dask array of shape (N, 2, H, W) containing image pairs.
    config : Config
        Configuration object.
    client : Client
        Dask distributed client.
    output_path : Path
        Directory where .mat files will be saved.
    start_frame : int
        Starting frame number (1-based) for filenames.
    pass_index : Optional[int]
        If specified, save only this pass (0-based).
        If None, save all passes.
    mask : Optional[np.ndarray]
        Boolean mask array of shape (H, W) where True indicates masked regions.
        If provided, vectors in masked regions will be invalidated (set to NaN).
        
    Returns
    -------
    List
        List of Future objects that will resolve to saved file paths.
        Use client.gather() or wait() to ensure all files are saved.
    """
    save_futures = []
    for i in range(images.shape[0]):
        block = images[i]
        frame_number = start_frame + i
        
        # Submit PIV task with mask
        piv_future = client.submit(_piv_single_pass, block, config, mask)
        
        # Chain save task to PIV result
        save_future = client.submit(
            save_piv_result_distributed,
            piv_future,  # Takes result from PIV task
            output_path,
            frame_number,
            pass_index,
        )
        save_futures.append(save_future)
    
    return save_futures


def perform_piv(images: da.Array, config: Config, client: Client) -> List:
    """
    Perform PIV on a batch of image pairs.
    
    Parameters
    ----------
    images : da.Array
        Dask array of shape (N, 2, H, W) containing image pairs.
    config : Config
        Configuration object.
    client : Client
        Dask distributed client.
        
    Returns
    -------
    List
        List of Future objects that will resolve to PIVResult objects.
        Use client.gather() to collect results or simply iterate and
        call .result() on each future.
    """
    # Submit tasks to the cluster and return futures
    futures = []
    for i in range(images.shape[0]):
        block = images[i]  # Get each block
        future = client.submit(_piv_single_pass, block, config)
        futures.append(future)
    return futures


def _piv_single_pass(
    image_block: da.Array,
    config: Config,
    mask: Optional[np.ndarray] = None,
) -> PIVResult:
    try:
        image_block = image_block.compute()
        if image_block.ndim == 3:
            # Shape: (2, H, W)
            image_block = image_block[np.newaxis, ...]  # Shape: (1, 2, H, W)
        correlator = make_correlator_backend(config)
        piv_results = correlator.correlate_batch(image_block, config=config, mask=mask)
    except Exception as e:
        # Return a PIVResult containing error information
        error_result = PIVResult()
        # We could add error information to the result if needed
        logging.error(f"PIV processing failed: {str(e)}")
        # For now, return an empty result to maintain consistent typing
        return error_result
    return piv_results
