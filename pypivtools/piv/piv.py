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
    vector_masks: Optional[List[np.ndarray]] = None,
) -> List:
    """
    Perform PIV and save results in parallel on workers.
    
    This function chains PIV computation with saving, avoiding the memory
    bottleneck of gathering all results to the main process before saving.
    Each worker processes and saves its results independently.
    
    Memory-efficient design:
    - PIV results stay on workers (no gather to main)
    - Direct serialization to disk on each worker
    - Minimal memory footprint for large batches
    
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
    vector_masks : Optional[List[np.ndarray]]
        Pre-computed vector masks for each PIV pass. Each mask should be a boolean
        array of shape (n_win_y, n_win_x) where True indicates vectors to mask.
        If provided, these are used directly instead of computing from pixel masks.
        
    Returns
    -------
    List
        List of Future objects that will resolve to saved file paths.
        Use client.gather() or wait() to ensure all files are saved.
    """
    # Pre-compute correlator cache once to avoid redundant caching on workers
    temp_correlator = make_correlator_backend(config)
    correlator_cache = temp_correlator.get_cache_data()
    
    # Broadcast cache to all workers once using scatter (more efficient than sending with each task)
    scattered_cache = client.scatter(correlator_cache, broadcast=True)
    logging.info("Pre-computed and broadcast correlator cache for distributed workers")
    
    save_futures = []
    for i in range(int(images.shape[0])):
        block = images[i]
        frame_number = start_frame + i
        
        # Submit PIV task with scattered cache reference (not the full data)
        piv_future = client.submit(_piv_single_pass, block, config, vector_masks, scattered_cache)
        
        # Chain save task to PIV result
        # All required data (win_ctrs, b_mask) is in PIVResult
        save_future = client.submit(
            save_piv_result_distributed,
            piv_future,  # Takes result from PIV task
            output_path,
            frame_number,
            pass_index,
            config.vector_format,
        )
        save_futures.append(save_future)
    
    return save_futures, scattered_cache


# def perform_piv(images: da.Array, config: Config, client: Client) -> List:
#     """
#     Perform PIV on a batch of image pairs.
    
#     Parameters
#     ----------
#     images : da.Array
#         Dask array of shape (N, 2, H, W) containing image pairs.
#     config : Config
#         Configuration object.
#     client : Client
#         Dask distributed client.
        
#     Returns
#     -------
#     List
#         List of Future objects that will resolve to PIVResult objects.
#         Use client.gather() to collect results or simply iterate and
#         call .result() on each future.
#     """
#     # Submit tasks to the cluster and return futures
#     futures = []
#     for i in range(images.shape[0]):
#         block = images[i]  # Get each block
#         future = client.submit(_piv_single_pass, block, config)
#         futures.append(future)
#     return futures


def _piv_single_pass(
    image_block: da.Array,
    config: Config,
    vector_masks: Optional[List[np.ndarray]] = None,
    correlator_cache: Optional[dict] = None,
) -> PIVResult:
    try:
        image_block = image_block.compute()
        if image_block.ndim == 3:
            # Shape: (2, H, W)
            image_block = image_block[np.newaxis, ...]  # Shape: (1, 2, H, W)
        correlator = make_correlator_backend(config, precomputed_cache=correlator_cache)
        piv_results = correlator.correlate_batch(image_block, config=config, vector_masks=vector_masks)
    except Exception as e:
        # Return a PIVResult containing error information
        error_result = PIVResult()
        # We could add error information to the result if needed
        logging.error(f"PIV processing failed: {str(e)}")
        # For now, return an empty result to maintain consistent typing
        return error_result
    return piv_results
