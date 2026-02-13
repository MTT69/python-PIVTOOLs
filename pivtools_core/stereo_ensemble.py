"""
Dask-Native Stereo Ensemble PIV — Correlation-of-Correlations Pipeline

Resolves all 6 Reynolds stress components by cross-correlating per-frame
correlation maps from two cameras. Standard stereo PIV only gets 5 of 6
(R_xx and R_zz are coupled). CoC provides the missing constraint.

Architecture mirrors ensemble.py:
- Lazy image loading from BOTH cameras
- Sliding window accumulation with bounded memory
- Multi-pass predictor-corrector refinement
- Stereo-specific: dewarping, dual-camera correlation, CoC

Usage:
    python -m pivtools_core.stereo_ensemble
"""
from pivtools_core.config import Config
import os
config = Config()
omp_threads = str(config.omp_threads)
os.environ["OMP_NUM_THREADS"] = omp_threads

import cv2
import gc
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.io
from dask.distributed import Client

from pivtools_core.validation import validate_config, log_validation_result
from pivtools_core.image_handling.load_images import (
    load_images,
    load_mask_for_camera,
    compute_vector_mask,
)
from pivtools_cli.piv_cluster.cluster import start_cluster
from pivtools_cli.piv.save_results import (
    get_output_path,
    save_stereo_ensemble_result,
    save_stereo_ensemble_coordinates,
)
from pivtools_cli.piv.stereo_ensemble_result import PIVStereoEnsembleResult
from pivtools_cli.piv.piv_backend.factory import make_stereo_ensemble_correlator
from pivtools_cli.piv.piv_backend.stereo_ensemble_accumulator import (
    StereoEnsembleAccumulator,
)
from pivtools_cli.processing.dask_pipeline import (
    rechunk_for_batched_processing,
    create_filter_pipeline,
    reduce_stereo_ensemble_results,
    correlate_stereo_batch_and_accumulate,
)
from pivtools_gui.stereo_reconstruction.self_calibration import PinholeCamera


logger = logging.getLogger(__name__)


def _load_stereo_cameras(
    config: Config, base_path: Path
) -> Tuple[PinholeCamera, PinholeCamera, int, int]:
    """
    Load stereo calibration model and build PinholeCamera instances.

    Camera 1 is the reference (R=I, t=0). Camera 2 has relative R and t.

    Returns
    -------
    (cam1, cam2, cam_num_a, cam_num_b)
    """
    cam_pair = config.stereo_ensemble_camera_pair
    cam_a, cam_b = int(cam_pair[0]), int(cam_pair[1])

    # Find stereo model — try both calibration methods
    model_path = None
    for method in ["stereo_dotboard", "stereo_charuco"]:
        candidate = (
            base_path / "calibration"
            / f"stereo_cam{cam_a}_cam{cam_b}"
            / "model" / "stereo_model.mat"
        )
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(
            f"No stereo calibration model found for cameras {cam_a},{cam_b} "
            f"under {base_path / 'calibration'}"
        )

    logger.info(f"Loading stereo model from: {model_path}")
    stereo_data = scipy.io.loadmat(
        str(model_path), squeeze_me=True, struct_as_record=False
    )

    # Validate required fields
    for field in ["camera_matrix_1", "camera_matrix_2",
                  "dist_coeffs_1", "dist_coeffs_2",
                  "rotation_matrix", "translation_vector"]:
        if field not in stereo_data:
            raise ValueError(f"Missing required field '{field}' in stereo model")

    K1 = np.array(stereo_data["camera_matrix_1"], dtype=np.float64)
    K2 = np.array(stereo_data["camera_matrix_2"], dtype=np.float64)
    dist1 = np.array(stereo_data["dist_coeffs_1"], dtype=np.float64).flatten()
    dist2 = np.array(stereo_data["dist_coeffs_2"], dtype=np.float64).flatten()
    R = np.array(stereo_data["rotation_matrix"], dtype=np.float64)
    t = np.array(stereo_data["translation_vector"], dtype=np.float64).reshape(3, 1)

    image_size = tuple(reversed(config.image_shape))  # (W, H)

    cam1 = PinholeCamera(
        K=K1, dist=dist1,
        R=np.eye(3, dtype=np.float64),
        t=np.zeros((3, 1), dtype=np.float64),
        image_size=image_size,
    )
    cam2 = PinholeCamera(
        K=K2, dist=dist2,
        R=R, t=t,
        image_size=image_size,
    )

    logger.info(
        f"  Camera pair: {cam_a},{cam_b}, "
        f"image_size={image_size}"
    )

    return cam1, cam2, cam_a, cam_b


def _compute_world_bounds(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    output_size: Tuple[int, int],
    config_bounds,
) -> Tuple[float, float, float, float]:
    """
    Compute world coordinate bounds for dewarping.

    If config_bounds is set, use those. Otherwise auto-compute from
    the intersection of both cameras' fields of view at Z=0.

    Returns (x_min, x_max, y_min, y_max) in mm.
    """
    if config_bounds is not None:
        return tuple(config_bounds)

    # Auto-compute: project image corners to Z=0 plane for both cameras
    W, H = cam1.image_size
    corners = np.array([
        [0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]
    ], dtype=np.float64)

    # Unproject image corners through each camera's model
    # Use a simpler approach: project 4 image corners from each camera
    # to the Z=0 world plane
    all_x, all_y = [], []
    for cam in [cam1, cam2]:
        img_pts = np.array([
            [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]
        ], dtype=np.float64)

        # Undistort points
        undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2), cam.K, cam.dist
        ).reshape(-1, 2)

        # Project to Z=0 plane in camera frame
        # Ray direction = R^T @ [undist_x, undist_y, 1]^T
        # Intersection with Z=0: solve for t where R^T(ray*t - t_cam)[2] = 0
        R_inv = cam.R.T
        t_world = -R_inv @ cam.t.flatten()

        for pt in undist:
            ray_cam = np.array([pt[0], pt[1], 1.0])
            ray_world = R_inv @ ray_cam
            # Z=0 intersection: t_world[2] + s * ray_world[2] = 0
            if abs(ray_world[2]) > 1e-10:
                s = -t_world[2] / ray_world[2]
                world_pt = t_world + s * ray_world
                all_x.append(world_pt[0])
                all_y.append(world_pt[1])

    if not all_x:
        raise ValueError("Could not auto-compute world bounds from camera models")

    # Use the overlap region
    x_min = max(min(all_x), -500)
    x_max = min(max(all_x), 500)
    y_min = max(min(all_y), -500)
    y_max = min(max(all_y), 500)

    logger.info(
        f"  Auto world bounds: x=[{x_min:.1f}, {x_max:.1f}], "
        f"y=[{y_min:.1f}, {y_max:.1f}] mm"
    )

    return (x_min, x_max, y_min, y_max)


def _get_stereo_ensemble_output_path(
    config: Config,
    cam_a: int,
    cam_b: int,
    base_path_idx: int = 0,
    create: bool = True,
) -> Path:
    """
    Get output path for stereo ensemble results.

    Structure: base_path/uncalibrated_piv/{N}/Stereo Cam{A}_Cam{B}/stereo_ensemble/
    """
    base_path = config.base_paths[base_path_idx]
    num = config.num_frame_pairs
    out_dir = (
        base_path / "uncalibrated_piv" / str(num)
        / f"Stereo Cam{cam_a}_Cam{cam_b}" / "stereo_ensemble"
    )
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def process_stereo_pass_sliding_window(
    client,
    images_cam1,
    images_cam2,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
):
    """
    Process one stereo ensemble pass using sliding window for parallel I/O.

    Mirrors process_pass_sliding_window from ensemble.py but submits
    paired camera chunks to each worker.

    Memory: ~4 batches per worker (2 cameras × 2 in flight), bounded.
    """
    from dask.distributed import wait

    num_workers = len(workers)
    max_in_flight = min(num_workers * 2, num_chunks)

    # State tracking
    pending_cam1 = {}  # chunk_idx -> Future
    pending_cam2 = {}  # chunk_idx -> Future
    accumulated_futures = {w: None for w in workers}
    next_to_correlate = 0
    next_to_submit = 0
    last_reported_pct = -1

    logger.debug(f"  Stereo sliding window: max_in_flight={max_in_flight}")

    # PHASE 1: Fill initial window — NO worker pinning for parallel I/O
    while next_to_submit < min(max_in_flight, num_chunks):
        pending_cam1[next_to_submit] = client.compute(
            images_cam1.blocks[next_to_submit]
        )
        pending_cam2[next_to_submit] = client.compute(
            images_cam2.blocks[next_to_submit]
        )
        next_to_submit += 1

    logger.debug(
        f"  Initial window: {len(pending_cam1)} chunk pairs in flight"
    )

    # PHASE 2: Process all chunks
    while next_to_correlate < num_chunks:
        # Wait for BOTH cameras' chunks to be ready
        cam1_future = pending_cam1[next_to_correlate]
        cam2_future = pending_cam2[next_to_correlate]
        wait([cam1_future, cam2_future])

        # Submit correlation (WITH worker pinning for accumulation chain)
        worker = workers[next_to_correlate % num_workers]
        is_first = (
            (next_to_correlate % num_workers == 0)
            and (next_to_correlate < num_workers)
        )

        accumulated_futures[worker] = client.submit(
            correlate_stereo_batch_and_accumulate,
            accumulated_futures[worker],
            cam1_future,
            cam2_future,
            scattered_config,
            pass_idx,
            scattered_predictor,
            scattered['cache'],
            scattered['masks'],
            scattered['stereo_setup'],
            is_first,
            workers=[worker],
            pure=False,
        )

        # Release filter references → memory freed
        del pending_cam1[next_to_correlate]
        del pending_cam2[next_to_correlate]
        next_to_correlate += 1

        # Submit replacement chunks to keep window full
        if next_to_submit < num_chunks:
            pending_cam1[next_to_submit] = client.compute(
                images_cam1.blocks[next_to_submit]
            )
            pending_cam2[next_to_submit] = client.compute(
                images_cam2.blocks[next_to_submit]
            )
            next_to_submit += 1

        # Progress logging
        current_pct = round((next_to_correlate / num_chunks) * 100)
        if current_pct > last_reported_pct:
            logger.info(f"  Correlation progress: {current_pct}%")
            last_reported_pct = current_pct

    # PHASE 3: Gather final results
    worker_results = []
    for worker in workers:
        if accumulated_futures[worker] is not None:
            result = accumulated_futures[worker].result()
            worker_results.append(result)
            logger.debug(f"  Worker complete: {result['n_images']} images")

    # Reduce across workers
    accumulated = worker_results[0]
    for r in worker_results[1:]:
        accumulated = reduce_stereo_ensemble_results(accumulated, r)

    return accumulated


def run_stereo_ensemble_piv(
    config: Config,
    client: Client,
    cam_a: int,
    cam_b: int,
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    source_path: Path,
    output_path: Path,
    base_path: Path,
    world_bounds: Tuple[float, float, float, float],
    vector_masks: Optional[List] = None,
    pixel_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Run Dask-native stereo ensemble PIV for one camera pair.

    Flow:
    1. Load images from BOTH cameras (lazy)
    2. Rechunk identically
    3. Apply filters via map_blocks
    4. For each pass:
       - Sliding window: submit paired camera chunks to workers
       - Workers: dewarp + per-camera correlation + CoC
       - Accumulate + finalize → 3D velocity + 6 stresses
       - Extract predictor for next pass
    5. Save stereo_ensemble_result.mat + stereo_coordinates.mat
    """
    output_size = tuple(config.stereo_ensemble_dewarp_output_size)

    # 1. Load images from both cameras (lazy)
    logger.info(f"Loading images for camera {cam_a}...")
    images_cam1 = load_images(cam_a, config, source=source_path)
    logger.info(f"  Cam {cam_a}: shape={images_cam1.shape}")

    logger.info(f"Loading images for camera {cam_b}...")
    images_cam2 = load_images(cam_b, config, source=source_path)
    logger.info(f"  Cam {cam_b}: shape={images_cam2.shape}")

    # 2. Build stereo correlator to get cache
    logger.info("Building stereo correlator and scattering data...")
    stereo_correlator = make_stereo_ensemble_correlator(
        config=config,
        cam1=cam1, cam2=cam2,
        output_size=output_size,
        world_bounds=world_bounds,
        self_cal_z=config.stereo_ensemble_self_cal_z,
        self_cal_tilt_x=config.stereo_ensemble_self_cal_tilt_x,
        self_cal_tilt_y=config.stereo_ensemble_self_cal_tilt_y,
    )
    correlator_cache = stereo_correlator.get_cache_data()
    scattered_cache = client.scatter(correlator_cache, broadcast=True)

    # Scatter vector masks
    scattered_masks = None
    if vector_masks:
        scattered_masks = client.scatter(vector_masks, broadcast=True)

    # Scatter stereo setup (cam models, geometry, etc.) — small objects
    stereo_setup = {
        'cam1': cam1,
        'cam2': cam2,
        'output_size': output_size,
        'world_bounds': world_bounds,
        'self_cal_z': config.stereo_ensemble_self_cal_z,
        'self_cal_tilt_x': config.stereo_ensemble_self_cal_tilt_x,
        'self_cal_tilt_y': config.stereo_ensemble_self_cal_tilt_y,
    }
    scattered_stereo_setup = client.scatter(stereo_setup, broadcast=True)

    scattered = {
        'cache': scattered_cache,
        'masks': scattered_masks,
        'stereo_setup': scattered_stereo_setup,
    }

    # 3. Rechunk both cameras identically
    batch_size = config.batch_size
    logger.info(f"Rechunking to batch_size={batch_size}...")
    images_cam1 = rechunk_for_batched_processing(images_cam1, batch_size)
    images_cam2 = rechunk_for_batched_processing(images_cam2, batch_size)
    logger.info(
        f"  Rechunked: {len(images_cam1.chunks[0])} chunks of size "
        f"{images_cam1.chunks[0][0]}"
    )

    # 4. Apply filters to both cameras
    logger.info("Creating filter pipelines...")
    images_cam1 = create_filter_pipeline(images_cam1, config, pixel_mask)
    images_cam2 = create_filter_pipeline(images_cam2, config, pixel_mask)

    # 5. Memory strategy
    num_chunks = len(images_cam1.chunks[0])
    if num_chunks == 1:
        from dask.distributed import wait as dask_wait
        logger.info(f"Small dataset ({num_chunks} chunk): persisting both cameras")
        images_cam1 = images_cam1.persist()
        images_cam2 = images_cam2.persist()
        dask_wait([images_cam1, images_cam2])
    else:
        logger.info(
            f"Large dataset ({num_chunks} chunks): "
            f"sliding window for parallel I/O"
        )

    num_passes = config.stereo_ensemble_num_passes
    accumulator = StereoEnsembleAccumulator(
        config=config,
        stereo_half_angle=stereo_correlator.stereo_half_angle,
        mm_per_pixel=stereo_correlator.mm_per_pixel,
        vector_masks=vector_masks,
    )
    predictor_field = None

    # Import for predictor extraction (used both for resume and after each pass)
    from pivtools_cli.processing.dask_pipeline import extract_predictor_field

    # Check for resume from previous pass
    resume_from_pass = config.stereo_ensemble_resume_from_pass  # 1-based, 0 = no resume
    start_pass_idx = 0  # 0-based

    if resume_from_pass > 0:
        # Convert to 0-based: resume_from_pass=3 means start at pass_idx=2
        start_pass_idx = resume_from_pass - 1

        # Validation
        if start_pass_idx < 1:
            raise ValueError(
                f"resume_from_pass={resume_from_pass} invalid: must resume from pass 2 or higher"
            )
        if start_pass_idx >= num_passes:
            raise ValueError(
                f"resume_from_pass={resume_from_pass} exceeds num_passes={num_passes}"
            )

        # Load existing stereo_ensemble_result.mat
        existing_result_path = output_path / "stereo_ensemble_result.mat"
        if not existing_result_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: {existing_result_path} not found. "
                f"Ensure previous passes completed successfully."
            )

        logger.info(f"Resuming from pass {resume_from_pass} (loading passes 1-{resume_from_pass-1})...")

        from pivtools_cli.piv.save_results import load_stereo_ensemble_result
        loaded_result, n_loaded = load_stereo_ensemble_result(
            existing_result_path,
            passes_to_load=list(range(start_pass_idx))  # Load passes 0..start_pass_idx-1
        )

        # Validate loaded passes
        if n_loaded < start_pass_idx:
            raise ValueError(
                f"Loaded only {n_loaded} passes but need {start_pass_idx} for resume_from_pass={resume_from_pass}"
            )

        # Load previous passes into accumulator
        accumulator.load_previous_passes(loaded_result, config.num_images)

        # Extract predictor from last loaded pass
        last_pass = loaded_result.passes[-1]
        predictor_field = extract_predictor_field(last_pass)

        logger.info(f"  Loaded {n_loaded} passes from {existing_result_path}")
        logger.info(f"  Predictor extracted from pass {start_pass_idx} (shape: {predictor_field.shape})")

    # Scatter config once
    scattered_config = client.scatter(config, broadcast=True)

    logger.info(
        f"Processing passes {start_pass_idx + 1} to {num_passes} with {num_chunks} chunks each..."
    )

    for pass_idx in range(start_pass_idx, num_passes):
        if _shutdown_requested:
            logger.info("Shutdown requested, stopping...")
            break

        logger.info("")
        logger.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

        # Scatter predictor for this pass
        scattered_predictor = None
        if predictor_field is not None:
            scattered_predictor = client.scatter(
                predictor_field, broadcast=True
            )
            logger.info("  Broadcast predictor field from previous pass")

        workers = list(client.ncores().keys())
        num_workers = len(workers)
        logger.info(
            f"  Distributing {num_chunks} chunk pairs across "
            f"{num_workers} workers..."
        )

        pass_start = time.time()

        # Process pass using stereo sliding window
        accumulated = process_stereo_pass_sliding_window(
            client=client,
            images_cam1=images_cam1,
            images_cam2=images_cam2,
            num_chunks=num_chunks,
            workers=workers,
            scattered_config=scattered_config,
            pass_idx=pass_idx,
            scattered_predictor=scattered_predictor,
            scattered=scattered,
            config=config,
        )

        # Accumulate and finalize pass
        accumulator.accumulate_batch(accumulated, pass_idx=pass_idx)
        pass_result = accumulator.finalize_pass(
            client=client, pass_idx=pass_idx, predictor_field=predictor_field,
            output_path=output_path,
        )

        # Extract predictor for next pass (in-plane displacements)
        if pass_idx < num_passes - 1:
            uy = pass_result.uy_mat.copy()
            ux = pass_result.ux_mat.copy()
            predictor_field = np.stack([uy, ux], axis=-1).astype(np.float32)
            logger.info(
                f"  Predictor: ux mean={np.nanmean(ux):.4f}, "
                f"uy mean={np.nanmean(uy):.4f}"
            )

        # Clean up
        accumulator.clear_pass_data(pass_idx)
        del accumulated
        if scattered_predictor is not None:
            del scattered_predictor
        gc.collect()

        pass_elapsed = time.time() - pass_start
        logger.info(f"  Pass {pass_idx + 1} complete in {pass_elapsed:.1f}s")

    # 6. Build and save result
    logger.info("")
    logger.info("Building stereo ensemble result...")
    stereo_result = PIVStereoEnsembleResult()
    for pass_result in accumulator.passes_results:
        stereo_result.add_pass(pass_result)

    runs_to_save = list(range(num_passes))

    # Backup existing result if resuming (safety measure)
    if resume_from_pass > 0:
        import shutil
        existing_path = output_path / "stereo_ensemble_result.mat"
        if existing_path.exists():
            backup_path = output_path / f"stereo_ensemble_result_before_pass{resume_from_pass}.mat.bak"
            shutil.copy2(existing_path, backup_path)
            logger.info(f"Backed up previous result to {backup_path}")

    logger.info("Saving stereo ensemble result...")
    save_stereo_ensemble_result(
        stereo_result, output_path, runs_to_save=runs_to_save,
    )

    # Save coordinates
    logger.info("Saving stereo coordinates...")
    win_ctrs_x_list = [p.win_ctrs_x for p in stereo_result.passes]
    win_ctrs_y_list = [p.win_ctrs_y for p in stereo_result.passes]
    save_stereo_ensemble_coordinates(
        config, output_path,
        win_ctrs_x_list=win_ctrs_x_list,
        win_ctrs_y_list=win_ctrs_y_list,
        mm_per_pixel=stereo_correlator.mm_per_pixel,
        world_bounds=world_bounds,
        runs_to_save=runs_to_save,
    )

    final_path = output_path / "stereo_ensemble_result.mat"
    logger.info(f"  Saved to {final_path}")

    return str(final_path)


# Global references for clean shutdown
_client = None
_cluster = None
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle termination signals for clean shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = (
        signal.Signals(signum).name
        if hasattr(signal, 'Signals') else str(signum)
    )
    logger.info(f"Received signal {sig_name}, initiating clean shutdown...")
    print(
        f"\n[CANCELLED] Received signal {sig_name}, shutting down...",
        flush=True,
    )

    # Suppress noisy distributed logs during teardown
    import logging as _logging
    for name in [
        "distributed.worker", "distributed.scheduler",
        "distributed.nanny", "distributed.core",
        "distributed.comm", "distributed.comm.tcp",
        "distributed.batched", "tornado.application",
        "tornado.general",
    ]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL)

    try:
        if _client is not None:
            logger.info("Cancelling pending futures...")
            try:
                _client.cancel(_client.futures, force=True)
            except Exception:
                pass
            logger.info("Closing Dask client...")
            _client.close(timeout=5)
    except Exception as e:
        logger.warning(f"Error closing client: {e}")

    import time as _time
    _time.sleep(0.5)

    try:
        if _cluster is not None:
            logger.info("Closing Dask cluster...")
            _cluster.close(timeout=5)
    except Exception as e:
        logger.warning(f"Error closing cluster: {e}")

    logger.info("Shutdown complete.")
    print("[CANCELLED] Shutdown complete.", flush=True)
    os._exit(1)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main entry point for stereo ensemble PIV processing."""
    start_time = time.time()

    # Validate configuration
    is_valid, error_msg, warnings = validate_config(config)
    log_validation_result(is_valid, error_msg, warnings, config)

    if not is_valid:
        sys.exit(1)

    # Set up environment
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

    global _client, _cluster

    try:
        # Start Dask cluster
        logger.info("Starting Dask cluster...")
        cluster, client = start_cluster(config=config)
        _cluster = cluster
        _client = client
        logger.info("Dask cluster started successfully")

        # Log worker info
        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logger.info(
                f"Worker {w}: pid={meta.get('pid')}, host={meta.get('host')}"
            )

        # Generate run timestamp for config traceability
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        active_path_indices = config.active_paths
        cam_pair = config.stereo_ensemble_camera_pair

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEREO ENSEMBLE PIV — CORRELATION-OF-CORRELATIONS")
        logger.info(
            f"Processing {len(active_path_indices)} path(s), "
            f"camera pair {cam_pair}"
        )
        logger.info("=" * 80)

        for path_set_num, path_idx in enumerate(active_path_indices, start=1):
            if _shutdown_requested:
                break

            source_path = config.source_paths[path_idx]
            base_path = config.base_paths[path_idx]

            # Save timestamped config copy
            config_copy_path = config.save_timestamped_copy(
                base_path, timestamp=run_timestamp
            )
            logger.info(f"Config saved to: {config_copy_path}")

            logger.info("")
            logger.info(
                f"PATH SET {path_set_num} of {len(active_path_indices)}"
            )
            logger.info(f"  Source: {source_path}")
            logger.info(f"  Base: {base_path}")

            if _shutdown_requested:
                break

            # Load stereo cameras
            cam1, cam2, cam_a, cam_b = _load_stereo_cameras(
                config, base_path
            )

            # Compute world bounds
            output_size = tuple(config.stereo_ensemble_dewarp_output_size)
            world_bounds = _compute_world_bounds(
                cam1, cam2, output_size,
                config.stereo_ensemble_world_bounds,
            )

            # Load mask (use cam_a's mask)
            mask = load_mask_for_camera(
                cam_a, config, source_path_idx=path_idx
            )
            vector_masks = None
            if config.masking_enabled and mask is not None:
                logger.info("Computing vector masks...")
                vector_masks = compute_vector_mask(
                    mask, config, ensemble=True
                )

            # Get output path
            output_path = _get_stereo_ensemble_output_path(
                config, cam_a, cam_b, base_path_idx=path_idx,
            )

            # Run
            logger.info("=" * 60)
            logger.info(
                f"STEREO ENSEMBLE PIV: Cameras {cam_a},{cam_b}"
            )
            logger.info(f"  Frame pairs: {config.num_frame_pairs}")
            logger.info(f"  Batch size: {config.batch_size}")
            logger.info(f"  Passes: {config.stereo_ensemble_num_passes}")
            logger.info(
                f"  Window sizes: {config.stereo_ensemble_window_sizes}"
            )
            logger.info(f"  Dewarp size: {output_size}")
            logger.info(f"  World bounds: {world_bounds}")
            logger.info(f"  Output: {output_path}")
            logger.info("=" * 60)

            result_path = run_stereo_ensemble_piv(
                config=config,
                client=client,
                cam_a=cam_a,
                cam_b=cam_b,
                cam1=cam1,
                cam2=cam2,
                source_path=source_path,
                output_path=output_path,
                base_path=base_path,
                world_bounds=world_bounds,
                vector_masks=vector_masks,
                pixel_mask=mask,
            )

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"STEREO ENSEMBLE PIV COMPLETE: {result_path}")
            logger.info("=" * 60)

        elapsed = time.time() - start_time
        logger.info("")
        logger.info(
            f"All stereo ensemble processing complete in {elapsed:.1f}s"
        )

    except Exception as e:
        logger.error(f"Stereo ensemble PIV failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            if _client is not None:
                _client.close(timeout=5)
            if _cluster is not None:
                _cluster.close(timeout=5)
        except Exception:
            pass


if __name__ == "__main__":
    main()
