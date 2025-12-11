import logging
import os
import signal
import sys
import tracemalloc
import time
from pathlib import Path

import yaml

# Global references for clean shutdown
_client = None
_cluster = None
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle termination signals for clean shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logging.info(f"Received signal {sig_name}, initiating clean shutdown...")
    print(f"\n[CANCELLED] Received signal {sig_name}, shutting down...", flush=True)

    # Close Dask client and cluster if they exist
    try:
        if _client is not None:
            logging.info("Closing Dask client...")
            _client.close()
    except Exception as e:
        logging.warning(f"Error closing client: {e}")

    try:
        if _cluster is not None:
            logging.info("Closing Dask cluster...")
            _cluster.close()
    except Exception as e:
        logging.warning(f"Error closing cluster: {e}")

    logging.info("Shutdown complete.")
    print("[CANCELLED] Shutdown complete.", flush=True)
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Add src to path for unified imports
from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import load_images, load_mask_for_camera
from pivtools_core.image_handling.load_images import compute_vector_mask

from pivtools_cli.piv.save_results import (
    save_coordinates_from_config_distributed,
    get_output_path,
    get_ensemble_output_path,
)
from pivtools_cli.piv_cluster.cluster import start_cluster
from pivtools_cli.processing.batch_pipeline import UnifiedBatchPipeline


def validate_config(config: Config) -> tuple[bool, str, list[str]]:
    """
    Validate configuration before starting PIV processing.

    Returns:
        tuple: (is_valid, error_message, warnings)
    """
    errors = []
    warnings = []

    # Check source_paths and base_paths have the same length for paired processing
    if len(config.source_paths) != len(config.base_paths):
        errors.append(
            f"source_paths ({len(config.source_paths)}) and base_paths ({len(config.base_paths)}) "
            "must have the same number of entries for paired processing"
        )

    # Check at least one active path
    if not config.active_paths:
        errors.append("No active paths configured. Set active_paths in config or check indices are valid.")

    # Check source paths exist
    for i, source_path in enumerate(config.source_paths):
        if not source_path.exists():
            errors.append(f"Source path {i+1} does not exist: {source_path}")

    # Check base paths exist (if used) - create if missing
    for i, base_path in enumerate(config.base_paths):
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created base path {i+1}: {base_path}")
            except Exception as e:
                errors.append(f"Failed to create base path {i+1}: {base_path} - {e}")

    if errors:
        return False, "\n".join(errors), warnings

    # Check image files for each camera
    camera_numbers = config.camera_numbers
    source_path = config.source_paths[0]

    for camera_num in camera_numbers:
        # Determine camera path
        format_str = config.image_format[0]
        if '.set' in str(format_str) or '.im7' in str(format_str):
            camera_path = source_path
        else:
            folder = config.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        if not camera_path.exists():
            errors.append(f"Camera {camera_num} path does not exist: {camera_path}")
            continue

        # Count files
        if '.set' in str(format_str):
            # Set files: single file
            if camera_path.is_file():
                set_file = camera_path
            else:
                set_file = camera_path / format_str
            
            if not set_file.exists():
                errors.append(f"Camera {camera_num}: Set file not found: {set_file}")
        elif '.im7' in str(format_str):
            # IM7 files
            pattern = format_str.replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
            matching_files = list(camera_path.glob(pattern))
            expected = config.num_images
            if len(matching_files) != expected:
                errors.append(
                    f"Camera {camera_num}: Found {len(matching_files)} IM7 files, expected {expected}. "
                    f"Path: {camera_path}, Pattern: {pattern}"
                )
        else:
            # Standard files
            expected = config.num_images
            if len(config.image_format) == 2:
                # A/B format: count A files
                pattern_a = config.image_format[0].replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
                matching_files = list(camera_path.glob(pattern_a))
            else:
                # Time-resolved: count all files
                pattern = format_str.replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
                matching_files = list(camera_path.glob(pattern))

            # Check for indexing mismatch
            if not ('.set' in str(format_str) or '.im7' in str(format_str)) and matching_files:
                indices = []
                for f in matching_files:
                    try:
                        import re
                        match = re.search(r'(\d+)', f.name)
                        if match:
                            idx = int(match.group(1))
                            indices.append(idx)
                    except Exception:
                        pass
                if indices:
                    min_idx = min(indices)
                    expected_min = 0 if config.zero_based_indexing else 1
                    if min_idx != expected_min:
                        warnings.append(
                            f"Camera {camera_num}: File indexing mismatch - found files starting at {min_idx}, "
                            f"but zero_based_indexing is {'enabled' if config.zero_based_indexing else 'disabled'} "
                            f"(expects {expected_min})"
                        )

            if len(matching_files) < expected:
                # ERROR: Not enough files
                all_files = sorted([f.name for f in camera_path.iterdir() if f.is_file()])[:5]
                file_list = ', '.join(all_files) if all_files else "(empty folder)"
                errors.append(
                    f"Camera {camera_num}: Missing files - found {len(matching_files)}, expected {expected}.\n"
                    f"  Path: {camera_path}\n"
                    f"  Pattern: {format_str}\n"
                    f"  Found files: {file_list}"
                )
            elif len(matching_files) > expected:
                # WARNING: Processing subset (this is fine!)
                warnings.append(
                    f"Camera {camera_num}: Processing subset - using {expected} of {len(matching_files)} available files"
                )

    if errors:
        return False, "\n".join(errors), warnings

    return True, "", warnings


def main():
    """Main PIV processing function"""
    start_time = time.time()  # Start timer

    config = Config()

    # Validate configuration before starting
    logging.info("=" * 80)
    logging.info("VALIDATING CONFIGURATION")
    logging.info("=" * 80)

    is_valid, error_msg, warnings = validate_config(config)
    if not is_valid:
        logging.error("Configuration validation failed!")
        logging.error("=" * 80)
        logging.error("ERRORS:")
        logging.error(error_msg)
        logging.error("=" * 80)
        logging.error("\nPlease fix the configuration errors in config.yaml and try again.")
        sys.exit(1)

    logging.info("✓ Configuration validated successfully")
    logging.info(f"  Source paths: {config.source_paths}")
    logging.info(f"  Cameras: {config.camera_numbers}")
    logging.info(f"  Image files: {config.num_images}")
    logging.info(f"  Frame pairs: {config.num_frame_pairs}")
    logging.info(f"  Image format: {config.image_format}")

    if warnings:
        logging.info("")
        logging.info("NOTES:")
        for warning in warnings:
            logging.info(f"  ℹ {warning}")

    logging.info("=" * 80)
    logging.info("")

    # Store original OMP_NUM_THREADS for workers
    
    # For batch filtering in main process, use all CPU cores
    max_threads = str(os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = max_threads
    logging.info(f"Set OMP_NUM_THREADS to {max_threads} for main process batch filtering")
    
    # For workers, use config value (set later when starting cluster)
    worker_omp_threads = str(config.omp_threads)
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    if config.debug:
        tracemalloc.start()

    global _client, _cluster

    try:
        cluster, client = start_cluster(
            n_workers_per_node=config.dask_workers_per_node,
            threads_per_worker=config.dask_threads_per_worker,
            memory_limit=config.dask_memory_limit,
            config=config,
            worker_omp_threads=worker_omp_threads,
        )
        # Store references for signal handler
        _cluster = cluster
        _client = client
        logging.info("Dask cluster started successfully")

    except Exception as e:
        logging.error("Error starting Dask cluster: %s", e)
        exit(1)

    try:

        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logging.info("Dask Worker Info:")
            logging.info("Worker %s", w)
            logging.info("  pid: %s", meta.get("pid"))
            logging.info("  host: %s", meta.get("host"))
            logging.info("  local_dir: %s", meta.get("local_directory"))
            logging.info("  nanny: %s", meta.get("nanny"))

        camera_numbers = config.camera_numbers
        active_path_indices = config.active_paths

        # Log multi-path processing info
        logging.info("")
        logging.info("=" * 80)
        logging.info("MULTI-PATH PROCESSING")
        logging.info(f"Processing {len(active_path_indices)} path combination(s): {active_path_indices}")
        logging.info("=" * 80)

        for path_set_num, path_idx in enumerate(active_path_indices, start=1):
            # Check if shutdown was requested
            if _shutdown_requested:
                logging.info("Shutdown requested, stopping processing...")
                break

            source_path = config.source_paths[path_idx]
            base_path = config.base_paths[path_idx]

            logging.info("")
            logging.info("=" * 80)
            logging.info(f"PATH SET {path_set_num} of {len(active_path_indices)}")
            logging.info(f"  Index: {path_idx}")
            logging.info(f"  Source: {source_path}")
            logging.info(f"  Base: {base_path}")
            logging.info("=" * 80)

            for camera_num in camera_numbers:
                # Check if shutdown was requested
                if _shutdown_requested:
                    logging.info("Shutdown requested, stopping processing...")
                    break

                logging.info("Processing camera: Cam%d", camera_num)

                # Load images from source path (lazy loading - no memory consumption yet)
                images = load_images(camera_num, config, source=source_path)

                # Load mask once per camera (if masking is enabled)
                mask = load_mask_for_camera(camera_num, config, source_path_idx=path_idx)

                # Pre-compute vector masks once per camera (if masking is enabled)
                vector_masks = None
                if config.masking_enabled and mask is not None:
                    logging.info("Pre-computing vector masks for Cam%d", camera_num)
                    vector_masks = compute_vector_mask(mask, config)
                    logging.info("Vector masks computed: %d passes", len(vector_masks))

                # Determine processing mode
                is_ensemble = config.data.get("processing", {}).get("ensemble", False)
                mode = "ensemble" if is_ensemble else "instantaneous"

                # Get output path based on mode
                if is_ensemble:
                    output_path = get_ensemble_output_path(
                        config,
                        camera_num,
                        use_uncalibrated=True,
                        base_path_idx=path_idx,
                    )
                else:
                    output_path = get_output_path(
                        config,
                        camera_num,
                        use_uncalibrated=True,
                        base_path_idx=path_idx,
                    )

                # Log processing configuration
                logging.info("=" * 80)
                logging.info(f"UNIFIED BATCH PIPELINE: {mode.upper()} MODE")
                logging.info("=" * 80)
                logging.info(f"Image files: {config.num_images}")
                logging.info(f"Frame pairs: {config.num_frame_pairs}")

                if is_ensemble:
                    logging.info(f"Ensemble passes: {config.ensemble_num_passes}")
                    logging.info(f"Ensemble window sizes: {config.ensemble_window_sizes}")
                    logging.info(f"Ensemble types: {config.ensemble_type}")
                    logging.info(f"Runs to save: {config.ensemble_runs_0based}")
                else:
                    logging.info(f"Runs to save: {config.instantaneous_runs_0based}")

                if config.filters:
                    filter_names = [f.get('type') for f in config.filters]
                    logging.info(f"Filters: {filter_names}")

                logging.info(f"Output path: {output_path}")
                logging.info("=" * 80)

                # Create unified pipeline
                pipeline = UnifiedBatchPipeline(
                    client=client,
                    config=config,
                    mode=mode,
                )

                # Process with unified pipeline
                # Pass both pixel_mask (for preprocessing) and vector_masks (for correlation validation)
                logging.info(f"Starting {mode} PIV processing with unified batch pipeline...")
                result = pipeline.process(
                    images,
                    output_path,
                    vector_masks=vector_masks,
                    pixel_mask=mask,  # Apply pixel mask during preprocessing
                )

                # Save coordinates
                if is_ensemble:
                    # For ensemble, save coordinates from cached correlator
                    from pivtools_cli.piv.save_results import save_ensemble_coordinates_from_config_distributed
                    from pivtools_cli.piv.piv_backend.factory import make_correlator_backend

                    temp_correlator = make_correlator_backend(config, ensemble=True)
                    correlator_cache = temp_correlator.get_cache_data()

                    coords_path = save_ensemble_coordinates_from_config_distributed(
                        config,
                        output_path,
                        correlator_cache=correlator_cache,
                        runs_to_save=config.ensemble_runs_0based,
                    )
                    logging.info(f"Ensemble coordinates saved to {coords_path}")
                else:
                    # For instantaneous, use scatter approach
                    from pivtools_cli.piv.piv_backend.factory import make_correlator_backend

                    temp_correlator = make_correlator_backend(config)
                    correlator_cache = temp_correlator.get_cache_data()
                    scattered_cache = client.scatter(correlator_cache, broadcast=True)

                    coords_future = client.submit(
                        save_coordinates_from_config_distributed,
                        config,
                        output_path,
                        scattered_cache,
                        config.instantaneous_runs_0based,
                    )
                    coords_future.result()
                    logging.info(f"Coordinates saved to {output_path}")

                logging.info("")
                logging.info("=" * 80)
                logging.info(f"{mode.upper()} PIV PROCESSING COMPLETE!")
                if is_ensemble:
                    logging.info(f"Ensemble result saved to {result}")
                else:
                    logging.info(f"Total images processed: {len(result)}")
                logging.info("=" * 80)

        if config.debug:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 10**6:.2f} MB")
            print(f"Peak memory usage: {peak / 10**6:.2f} MB")

            tracemalloc.stop()
    except Exception as e:
        import traceback
        print(f"Error: {e}", flush=True)
        print("Traceback:", flush=True)
        traceback.print_exc()
    finally:
        # Clean shutdown of Dask resources
        try:
            if client is not None:
                client.close()
        except Exception as e:
            logging.warning(f"Error closing client in finally: {e}")

        try:
            if cluster is not None:
                cluster.close()
        except Exception as e:
            logging.warning(f"Error closing cluster in finally: {e}")

        end_time = time.time()  # End timer
        elapsed = end_time - start_time
        if _shutdown_requested:
            print(f"[CANCELLED] Run cancelled after {elapsed:.2f} seconds", flush=True)
        else:
            print(f"Total elapsed time: {elapsed:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
