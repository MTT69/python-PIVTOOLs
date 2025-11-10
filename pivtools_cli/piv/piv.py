import logging
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional
from dask import array as da
from dask.distributed import Client
from pivtools_core.config import Config

from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.piv.piv_result import PIVResult
from pivtools_cli.piv.save_results import save_piv_result_distributed


def _batch_policy(config: Config) -> int:
    if config.backend == "gpu":
        # ~ (batch_size * H * W * bytes_per_pixel * intermediates)
        # < GPU_RAM * safety
        return 8  # e.g. start with 8 and tune
    else:
        return 1


def _process_and_save_single_pair(
    image_pair: da.Array,
    frame_number: int,
    config: Config,
    scattered_masks,
    scattered_cache,
    output_path: Path,
    runs_to_save: Optional[List[int]],
    vector_format: str,
) -> str:
    """
    Combined PIV processing and saving for a single image pair.
    
    This function is designed to be called via client.map() for efficient
    batch submission of tasks. It combines PIV computation and saving into
    a single atomic operation to reduce task graph complexity.
    
    Parameters
    ----------
    image_pair : da.Array
        Dask array slice of shape (2, H, W) containing one image pair.
    frame_number : int
        Frame number (1-based) for output filename.
    config : Config
        Configuration object.
    scattered_masks : Future or None
        Scattered reference to vector masks.
    scattered_cache : Future
        Scattered reference to correlator cache.
    output_path : Path
        Directory where .mat file will be saved.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save.
    vector_format : str
        Format string for output filenames.
        
    Returns
    -------
    str
        Path to the saved file.
    """
    # Process PIV
    piv_result = _piv_single_pass(image_pair, config, scattered_masks, scattered_cache)
    
    # Save immediately to avoid accumulating results in memory
    saved_path = save_piv_result_distributed(
        piv_result, output_path, frame_number, runs_to_save, vector_format
    )
    
    return saved_path


def perform_piv_and_save(
    images: da.Array,
    config: Config,
    client: Client,
    output_path: Path,
    start_frame: int = 1,
    runs_to_save: Optional[List[int]] = None,
    vector_masks: Optional[List[np.ndarray]] = None,
    batch_size: int = None,  # Deprecated, kept for compatibility
) -> List:
    """
    Perform PIV and save results in parallel using TRUE lazy loading.
    
    This is the OPTIMAL Dask pattern:
    1. Images are already delayed tasks (from load_images)
    2. We convert to delayed format and submit to workers
    3. Each worker receives ONE delayed task at a time
    4. Worker: load → process → save → free → next
    5. No memory accumulation, no manual batching
    
    Memory footprint per worker:
    - 1 image pair: ~80 MB
    - PIV processing: ~200 MB peak
    - Total: ~280 MB (constant, regardless of total images!)
    
    Scaling:
    - 4 workers × 280 MB = ~1.1 GB total
    - Can process 10,000 images with same memory!
    - Main process: ~10 MB (just task graph)
    
    Parameters
    ----------
    images : da.Array
        Dask array of shape (N, 2, H, W) containing image pairs.
        Should be created by load_images() - already delayed!
    config : Config
        Configuration object.
    client : Client
        Dask distributed client.
    output_path : Path
        Directory where .mat files will be saved.
    start_frame : int
        Starting frame number (1-based) for filenames.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save. If None, save all passes.
    vector_masks : Optional[List[np.ndarray]]
        Pre-computed vector masks for each PIV pass.
    batch_size : int
        DEPRECATED. Kept for compatibility but ignored.
        Dask scheduler handles distribution automatically.
        
    Returns
    -------
    tuple
        (all_saved_paths, scattered_cache) where:
        - all_saved_paths: List of paths to saved files
        - scattered_cache: Scattered correlator cache (for coordinate saving)
    """
    # Pre-compute correlator cache once to avoid redundant caching on workers
    temp_correlator = make_correlator_backend(config)
    correlator_cache = temp_correlator.get_cache_data()
    
    # Broadcast cache to all workers once (efficient, happens once)
    scattered_cache = client.scatter(correlator_cache, broadcast=True)
    logging.info("Broadcast correlator cache to all workers")
    
    scattered_masks = None
    if vector_masks is not None:
        scattered_masks = client.scatter(vector_masks, broadcast=True)
        total_mask_size = sum(m.nbytes for m in vector_masks) / 1024
        logging.info(f"Broadcast vector masks to all workers ({total_mask_size:.1f} KB)")
    
    num_images = int(images.shape[0])
    
    # Convert Dask array to delayed objects (still lazy!)
    # This gives us individual delayed tasks, one per image
    delayed_blocks = images.to_delayed().ravel()
    
    # Prepare frame numbers
    frame_numbers = list(range(start_frame, start_frame + num_images))
    
    logging.info(f"Submitting {num_images} independent tasks to cluster")
    
    # Use client.map() to submit all tasks at once
    # Dask scheduler will distribute to workers efficiently
    # Each worker will process tasks one-by-one as they become available
    futures = client.map(
        _process_and_save_single_pair,
        delayed_blocks,  # List of delayed tasks
        frame_numbers,
        config=config,
        scattered_masks=scattered_masks,
        scattered_cache=scattered_cache,
        output_path=output_path,
        runs_to_save=runs_to_save,
        vector_format=config.vector_format,
    )
    
    logging.info(
        f"Tasks submitted. Dask scheduler managing distribution across workers. "
        f"Each worker processes ONE image at a time."
    )
    
    # Gather results (this will block until all complete)
    # Workers process in parallel but each holds only 1 image at a time
    try:
        all_saved_paths = client.gather(futures)
        logging.info(f"All {num_images} images processed successfully")
    except Exception as e:
        logging.error(f"PIV processing failed: {e}")
        raise
    
    return all_saved_paths, scattered_cache


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
