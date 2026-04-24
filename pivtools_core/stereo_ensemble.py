"""
Dask-Native Stereo Ensemble PIV Processing (CoC Method)

Dewarps images to common world-XY plane, performs multi-pass ensemble
correlation with predictor refinement, and uses Correlation-of-Correlations
to extract all 6 Reynolds stress components.

Calibration is baked into the dewarping — output is directly in physical
units (m/s, mm). No separate calibration step needed.

Usage:
    python -m pivtools_core.stereo_ensemble
"""
from pivtools_core.config import Config
import os
config = Config()
omp_threads = str(config.omp_threads)
os.environ["OMP_NUM_THREADS"] = omp_threads

import gc
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from dask.distributed import Client, Variable, as_completed

from pivtools_core.validation import (
    validate_config,
    validate_stereo_ensemble_config,
    log_validation_result,
)
from pivtools_core.image_handling.load_images import (
    load_images,
    load_mask_for_camera,
    compute_vector_mask,
)
from pivtools_cli.piv_cluster.cluster import start_cluster
from pivtools_cli.piv.stereo_ensemble_result import PIVStereoEnsembleResult
from pivtools_cli.piv.piv_backend.stereo_ensemble_accumulator import (
    StereoEnsembleAccumulator,
    extract_stereo_predictor_field,
)
from pivtools_cli.processing.dask_pipeline import (
    create_filter_pipeline,
    scatter_immutable_data,
    correlate_stereo_worker_batches,
    reduce_stereo_ensemble_results,
)


logger = logging.getLogger(__name__)

_shutdown_requested = False
_client = None
_cluster = None


def signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info(f"Signal {signum} received, requesting shutdown...")
    if _client is not None:
        try:
            _client.close(timeout=5)
        except Exception:
            pass
    os._exit(1)


def _compute_stereo_angle(cam1, cam2) -> float:
    """Compute sin(half-angle) between two camera optical axes.

    Parameters
    ----------
    cam1, cam2 : PinholeCamera
        Camera models with R (rotation world→camera).

    Returns
    -------
    sin_theta : float
        sin of the half-angle between camera viewing directions.
    """
    # Optical axis in world frame: R^T @ [0, 0, 1]
    axis1 = cam1.R.T @ np.array([0.0, 0.0, 1.0])
    axis2 = cam2.R.T @ np.array([0.0, 0.0, 1.0])

    # Normalize
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    # Full angle between axes
    cos_full = np.clip(np.dot(axis1, axis2), -1.0, 1.0)
    full_angle = np.arccos(cos_full)
    half_angle = full_angle / 2.0

    return float(np.sin(half_angle))


def process_stereo_pass(
    client,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
    output_path,
    camera_pair,
    source_path,
    pixel_mask,
    dewarp_maps_cam1,
    dewarp_maps_cam2,
    mm_per_pixel,
    stereo_angle,
):
    """Process one stereo ensemble pass using per-worker accumulation + tree reduction."""
    import threading

    num_workers = len(workers)

    # Assign chunks round-robin
    worker_chunks = {w: [] for w in workers}
    for chunk_idx in range(num_chunks):
        w = workers[chunk_idx % num_workers]
        worker_chunks[w].append(chunk_idx)
    worker_chunks = {w: idxs for w, idxs in worker_chunks.items() if idxs}

    logger.info(
        f"  Worker accumulation: {len(worker_chunks)} workers, "
        f"{num_chunks} chunks ({[len(c) for c in worker_chunks.values()]} per worker)"
    )

    # Progress variables
    progress_vars = []
    worker_list = list(worker_chunks.keys())
    for i in range(len(worker_list)):
        var = Variable(f"_stereo_p{pass_idx}_{i}", client)
        var.set(0)
        progress_vars.append(var)

    # Submit one task per worker
    worker_futures = []
    for wi, (worker, chunk_indices) in enumerate(worker_chunks.items()):
        fut = client.submit(
            correlate_stereo_worker_batches,
            batch_indices=chunk_indices,
            config=scattered_config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
            cache=scattered['cache'],
            masks=scattered['masks'],
            camera_pair=camera_pair,
            source_path=str(source_path),
            dewarp_maps_cam1=dewarp_maps_cam1,
            dewarp_maps_cam2=dewarp_maps_cam2,
            mm_per_pixel=mm_per_pixel,
            stereo_angle=stereo_angle,
            pixel_mask=pixel_mask,
            output_path=str(output_path) if config.stereo_ensemble_save_diagnostics and 0 in chunk_indices else None,
            progress_var_name=f"_stereo_p{pass_idx}_{wi}",
            workers=[worker],
            pure=False,
        )
        worker_futures.append(fut)

    # Progress polling
    stop_progress = threading.Event()
    last_pct = [0]

    def poll_progress():
        while not stop_progress.is_set():
            try:
                total_done = sum(v.get(timeout=2) for v in progress_vars)
                pct = min(100, int(100 * total_done / num_chunks))
                if pct > last_pct[0]:
                    logger.info(f"  Pass {pass_idx + 1} correlation: {pct}%")
                    last_pct[0] = pct
            except Exception:
                pass
            stop_progress.wait(3)

    progress_thread = threading.Thread(target=poll_progress, daemon=True)
    progress_thread.start()

    # Wait for completion
    completed_count = 0
    ac = as_completed(worker_futures)
    for completed in ac:
        completed_count += 1
        exc = completed.exception()
        if exc is not None:
            for f in worker_futures:
                f.cancel()
            raise exc

    stop_progress.set()
    progress_thread.join(timeout=5)
    for v in progress_vars:
        try:
            v.delete()
        except Exception:
            pass

    # Tree reduction
    futures = list(worker_futures)
    logger.info(f"  Tree reduction: {len(futures)} worker results")

    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), 2):
            if i + 1 < len(futures):
                merged = client.submit(
                    reduce_stereo_ensemble_results, futures[i], futures[i + 1],
                    pure=False,
                )
                new_futures.append(merged)
            else:
                new_futures.append(futures[i])
        futures = new_futures

    final = futures[0].result()
    logger.info(f"  Accumulated: {final['n_images']} images")
    return final


def run_stereo_ensemble_piv(
    config: Config,
    client: Client,
    camera_pair: tuple,
    source_path: Path,
    output_path: Path,
    base_path: Path,
    vector_masks: Optional[List] = None,
    pixel_mask: Optional = None,
) -> str:
    """Run stereo ensemble CoC PIV for one camera pair.

    Flow:
    1. Load camera models + compute dewarp maps
    2. Load images for both cameras (lazy)
    3. Apply filters
    4. Multi-pass: scatter → worker accumulation → finalize → predictor
    5. Save calibrated results (dewarping IS calibration)
    """
    cam_a, cam_b = camera_pair

    # ── 1. Load camera models ───────────────────────────────────────────
    from pivtools_gui.calibration.camera_model_utils import load_cameras_from_stereo_model
    from pivtools_gui.stereo_reconstruction.self_calibration import (
        compute_dewarp_maps,
        estimate_pixel_scale,
    )

    logger.info(f"Loading stereo model for cameras {cam_a}, {cam_b}...")
    cam1, cam2, _, _ = load_cameras_from_stereo_model(
        str(base_path), cam_a, cam_b
    )

    # Self-calibration corrections
    z_offset = config.self_calibration_z_offset
    tilt_x = config.self_calibration_tilt_x
    tilt_y = config.self_calibration_tilt_y
    logger.info(f"  Self-cal: z_offset={z_offset:.3f}mm, tilt_x={tilt_x:.4f}rad, tilt_y={tilt_y:.4f}rad")

    # Stereo geometry
    sin_theta = _compute_stereo_angle(cam1, cam2)
    logger.info(f"  Stereo half-angle: sin(θ)={sin_theta:.4f} (θ={np.degrees(np.arcsin(sin_theta)):.1f}°)")

    # ── 2. Compute world bounds and dewarp maps ─────────────────────────
    world_bounds = config.stereo_ensemble_world_bounds
    if world_bounds is None:
        from pivtools_gui.calibration.camera_model_utils import compute_camera_world_bounds
        import cv2 as cv
        # Auto-detect from camera FOV overlap
        rvec1, _ = cv.Rodrigues(cam1.R)
        rvec2, _ = cv.Rodrigues(cam2.R)
        w1, h1 = cam1.image_size
        w2, h2 = cam2.image_size
        bounds1 = compute_camera_world_bounds(cam1.K, cam1.dist, rvec1, cam1.t, w1, h1)
        bounds2 = compute_camera_world_bounds(cam2.K, cam2.dist, rvec2, cam2.t, w2, h2)
        # Intersection of both cameras' FOV
        world_bounds = (
            max(bounds1[0], bounds2[0]),  # x_min
            min(bounds1[1], bounds2[1]),  # x_max
            max(bounds1[2], bounds2[2]),  # y_min
            min(bounds1[3], bounds2[3]),  # y_max
        )
        logger.info(f"  Auto world bounds: {world_bounds}")

    mm_per_pixel = estimate_pixel_scale(cam1, cam2, world_bounds, z=z_offset)
    logger.info(f"  mm_per_pixel: {mm_per_pixel:.4f}")

    # Compute dewarp maps (ONCE, fixed for all frames and passes)
    logger.info("  Computing dewarp maps...")
    maps_cam1 = compute_dewarp_maps(cam1, world_bounds, mm_per_pixel, z_offset, tilt_x, tilt_y)
    maps_cam2 = compute_dewarp_maps(cam2, world_bounds, mm_per_pixel, z_offset, tilt_x, tilt_y)

    dw_h, dw_w = maps_cam1[0].shape
    logger.info(f"  Dewarped image size: {dw_h}x{dw_w}")

    # Dewarp pixel masks to world-XY space (if provided)
    if pixel_mask is not None:
        import cv2
        # Dewarp mask: 1=masked. Use INTER_NEAREST to keep binary.
        # borderValue=1 means outside-FOV is masked.
        dw_mask = cv2.remap(
            pixel_mask.astype(np.float32),
            maps_cam1[0], maps_cam1[1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
        pixel_mask = (dw_mask > 0.5).astype(np.uint8)
        logger.info(f"  Dewarped pixel mask: {pixel_mask.sum()}/{pixel_mask.size} masked pixels")

        # Recompute vector masks from the dewarped mask
        # Override image_shape for the dewarped dimensions
        orig_shape = getattr(config, '_detected_image_shape', None)
        config._detected_image_shape = (dw_h, dw_w)
        try:
            vector_masks = compute_vector_mask(pixel_mask, config, ensemble=True)
        except Exception:
            vector_masks = None
        if orig_shape is not None:
            config._detected_image_shape = orig_shape

    # ── 3. Load images for both cameras ─────────────────────────────────
    batch_size = config.batch_size
    logger.info(f"Loading images for cameras {cam_a} and {cam_b} (batch_size={batch_size})...")

    images_cam1 = load_images(cam_a, config, source=source_path, batch_size=batch_size)
    images_cam2 = load_images(cam_b, config, source=source_path, batch_size=batch_size)
    num_chunks = len(images_cam1.chunks[0])
    logger.info(f"  Loaded: {images_cam1.shape}, {num_chunks} chunks")

    # ── 4. Scatter immutable data ───────────────────────────────────────
    logger.info("Scattering immutable data...")
    scattered = scatter_immutable_data(
        client, config, vector_masks, pixel_mask, ensemble=True
    )
    scattered_config = client.scatter(config, broadcast=True)

    # Scatter dewarp maps (numpy arrays, serializable)
    scattered_maps_cam1 = client.scatter(maps_cam1, broadcast=True)
    scattered_maps_cam2 = client.scatter(maps_cam2, broadcast=True)

    scattered_pixel_mask = None
    if pixel_mask is not None:
        scattered_pixel_mask = client.scatter(pixel_mask, broadcast=True)

    # ── 5. Multi-pass loop ──────────────────────────────────────────────
    num_passes = config.stereo_ensemble_num_passes
    accumulator = StereoEnsembleAccumulator(
        config, mm_per_pixel=mm_per_pixel,
        stereo_angle=sin_theta, vector_masks=vector_masks,
    )
    predictor_field = None

    from pivtools_cli.piv.piv_backend.cpu_stereo_ensemble import StereoEnsembleCorrelatorCPU
    from pivtools_cli.piv.save_results import (
        save_stereo_ensemble_result,
        save_stereo_ensemble_coordinates,
    )

    # Build a local correlator for finalize_pass (fitting, window info)
    local_correlator = StereoEnsembleCorrelatorCPU(
        config,
        vector_masks=vector_masks,
        dewarp_maps={cam_a: maps_cam1, cam_b: maps_cam2},
        mm_per_pixel=mm_per_pixel,
        stereo_angle=sin_theta,
    )

    logger.info(f"Processing {num_passes} passes with {num_chunks} chunks each...")

    for pass_idx in range(num_passes):
        if _shutdown_requested:
            logger.info("Shutdown requested, stopping...")
            break

        logger.info("")
        logger.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

        # Scatter predictor
        scattered_predictor = None
        if predictor_field is not None:
            scattered_predictor = client.scatter(predictor_field, broadcast=True)
            logger.info(f"  Broadcast predictor field (shape: {predictor_field.shape})")

        workers = list(client.ncores().keys())
        pass_start = time.time()

        # Worker accumulation + tree reduction
        accumulated = process_stereo_pass(
            client=client,
            num_chunks=num_chunks,
            workers=workers,
            scattered_config=scattered_config,
            pass_idx=pass_idx,
            scattered_predictor=scattered_predictor,
            scattered=scattered,
            config=config,
            output_path=output_path,
            camera_pair=camera_pair,
            source_path=source_path,
            pixel_mask=scattered_pixel_mask,
            dewarp_maps_cam1=scattered_maps_cam1,
            dewarp_maps_cam2=scattered_maps_cam2,
            mm_per_pixel=mm_per_pixel,
            stereo_angle=sin_theta,
        )

        corr_elapsed = time.time() - pass_start

        # Finalize pass
        finalize_start = time.time()
        accumulator.set_pass_data(pass_idx, accumulated)
        pass_result = accumulator.finalize_pass(
            pass_idx=pass_idx,
            correlator=local_correlator,
            config=config,
            output_path=str(output_path),
        )
        finalize_elapsed = time.time() - finalize_start

        # Intermediate save (enables debugging mid-run, protects against failures)
        intermediate_result = PIVStereoEnsembleResult()
        for pr in accumulator.passes_results:
            intermediate_result.add_pass(pr)
        output_path.mkdir(parents=True, exist_ok=True)
        save_stereo_ensemble_result(intermediate_result, output_path)
        logger.info(f"  Saved intermediate result (pass {pass_idx + 1}/{num_passes})")
        del intermediate_result

        # Extract predictor for next pass
        if pass_idx < num_passes - 1:
            predictor_field = extract_stereo_predictor_field(
                pass_result, mm_per_pixel, config.dt,
            )

        # Clean up
        del accumulated
        if scattered_predictor is not None:
            del scattered_predictor
        gc.collect()

        pass_elapsed = time.time() - pass_start
        logger.info(
            f"  Pass {pass_idx + 1} complete in {pass_elapsed:.1f}s "
            f"(correlation: {corr_elapsed:.1f}s, finalize: {finalize_elapsed:.1f}s)"
        )

    # ── 6. Save results ─────────────────────────────────────────────────
    logger.info("")
    logger.info("Saving stereo ensemble results...")

    # Build result container
    stereo_result = PIVStereoEnsembleResult()
    for pr in accumulator.passes_results:
        stereo_result.add_pass(pr)

    output_path.mkdir(parents=True, exist_ok=True)

    save_stereo_ensemble_result(stereo_result, output_path)
    save_stereo_ensemble_coordinates(
        output_path, world_bounds, mm_per_pixel,
        local_correlator.win_ctrs_x, local_correlator.win_ctrs_y,
        num_passes,
    )

    final_path = output_path / "stereo_ensemble_result.mat"
    logger.info(f"  Saved to {final_path}")
    return str(final_path)


def main():
    """Main entry point for stereo ensemble PIV processing."""
    start_time = time.time()

    # Validate
    is_valid, error_msg, warnings = validate_config(config)
    log_validation_result(is_valid, error_msg, warnings, config)
    if not is_valid:
        sys.exit(1)

    is_valid_se, errors_se, warnings_se = validate_stereo_ensemble_config(config)
    if not is_valid_se:
        logger.error("Stereo ensemble validation failed:")
        for e in errors_se:
            logger.error(f"  - {e}")
        sys.exit(1)
    for w in warnings_se:
        logger.warning(f"  - {w}")

    # Environment
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

    global _client, _cluster

    # Register signal handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start cluster
        logger.info("Starting Dask cluster...")
        cluster, client = start_cluster(config)
        _cluster, _client = cluster, client
        logger.info(f"  {len(client.ncores())} workers ready")

        # Process each active path
        # CLI --camera-pair override via environment variable
        camera_pair_env = os.environ.get("PIV_STEREO_CAMERA_PAIR")
        if camera_pair_env:
            cam_a, cam_b = (int(c.strip()) for c in camera_pair_env.split(","))
            camera_pair = (cam_a, cam_b)
        else:
            camera_pair = config.stereo_ensemble_camera_pair
        cam_a, cam_b = camera_pair

        for path_idx in config.active_paths:
            source_path = config.source_paths[path_idx]
            base_path = config.base_paths[path_idx]

            # Output: calibrated_piv/{N}/Stereo Cam{A}_Cam{B}/stereo_ensemble/
            n_frames = config.num_frame_pairs
            output_path = (
                base_path / "calibrated_piv" / str(n_frames)
                / f"Stereo Cam{cam_a}_Cam{cam_b}" / "stereo_ensemble"
            )

            logger.info(f"\nProcessing path {path_idx}: {source_path}")
            logger.info(f"  Output: {output_path}")

            # Load raw masks (dewarping happens inside run_stereo_ensemble_piv)
            pixel_mask_raw = load_mask_for_camera(cam_a, config, path_idx)

            run_stereo_ensemble_piv(
                config, client, camera_pair,
                source_path, output_path, base_path,
                vector_masks=None,
                pixel_mask=pixel_mask_raw,
            )

        elapsed = time.time() - start_time
        logger.info(f"\nStereo ensemble processing complete in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if _client:
            _client.close()
        if _cluster:
            _cluster.close()


if __name__ == "__main__":
    main()
