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
    batch_size: int = 20,
) -> List:
    """
    Perform PIV and save results in parallel on workers using batched processing.
    
    This optimized version uses client.map() for efficient task submission and
    processes images in batches to control memory usage. Each worker processes
    and saves results independently, avoiding memory bottlenecks.
    
    Memory-efficient design:
    - Uses client.map() instead of individual submit() calls (80% less overhead)
    - Batched processing prevents memory buildup
    - PIV results stay on workers (no gather to main)
    - Direct serialization to disk on each worker
    - Explicit memory cleanup between batches
    
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
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save. If None, save all passes.
        For passes not in this list, empty arrays will be saved.
    vector_masks : Optional[List[np.ndarray]]
        Pre-computed vector masks for each PIV pass. Each mask should be a boolean
        array of shape (n_win_y, n_win_x) where True indicates vectors to mask.
        If provided, these are used directly instead of computing from pixel masks.
    batch_size : int
        Number of images to process per batch. Smaller batches use less memory
        but have more overhead. Default: 20 (good balance for most systems).
        
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
    
    # Broadcast cache to all workers once using scatter (more efficient than sending with each task)
    scattered_cache = client.scatter(correlator_cache, broadcast=True)
    logging.info("Pre-computed and broadcast correlator cache for distributed workers")
    
    scattered_masks = None
    if vector_masks is not None:
        scattered_masks = client.scatter(vector_masks, broadcast=True)
        total_mask_size = sum(m.nbytes for m in vector_masks) / 1024
        logging.info(f"Broadcast vector masks to all workers ({total_mask_size:.1f} KB total)")
    
    num_images = int(images.shape[0])
    all_saved_paths = []
    
    # Process in batches to control memory usage
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_num_images = batch_end - batch_start
        
        logging.info(f"Processing batch {batch_start}-{batch_end-1} ({batch_num_images} images)")
        
        # Prepare batch data
        image_pairs = [images[i] for i in range(batch_start, batch_end)]
        frame_numbers = list(range(start_frame + batch_start, start_frame + batch_end))
        
        # Use client.map() for efficient batch submission
        # This is much faster than individual client.submit() calls in a loop
        batch_futures = client.map(
            _process_and_save_single_pair,
            image_pairs,
            frame_numbers,
            config=config,
            scattered_masks=scattered_masks,
            scattered_cache=scattered_cache,
            output_path=output_path,
            runs_to_save=runs_to_save,
            vector_format=config.vector_format,
        )
        
        # Wait for this batch to complete before starting next
        # This prevents memory buildup from too many concurrent tasks
        try:
            batch_results = client.gather(batch_futures)
            all_saved_paths.extend(batch_results)
            logging.info(f"Batch {batch_start}-{batch_end-1} completed successfully")
        except Exception as e:
            logging.error(f"Batch {batch_start}-{batch_end-1} failed: {e}")
            raise
        finally:
            # Explicitly release references to help garbage collection
            del batch_futures
            if 'batch_results' in locals():
                del batch_results
        
        # Optional: Force garbage collection on workers to free memory
        if batch_end < num_images:  # Don't GC after final batch
            client.run(lambda: __import__('gc').collect())
    
    logging.info(f"All {num_images} images processed successfully")
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
