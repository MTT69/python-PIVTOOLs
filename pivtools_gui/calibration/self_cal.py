"""calibration.self_cal — stereo self-calibration (Wieneke 2005) for calibration.

Bridges the calibration stereo model to the self-contained algorithm core in
``stereo_reconstruction.self_calibration`` (imported unchanged — no math is
reimplemented here). Self-calibration drives cross-camera disparity to zero on the
recorded PIV particle images and recovers the laser sheet's
``(z_offset, tilt_x, tilt_y)``; those three numbers then move the reconstructed
velocities onto the true sheet.

Frame consistency (the load-bearing fact): the ``PinholeCamera`` handed to the
algorithm is built directly from the stored ``CameraModel`` (same ``K, R, t, dist``),
so its world frame IS the calibration clicked origin/+X/+Y frame. The algorithm's
dewarp plane ``z = z_offset + wx*tan(tilt_y) + wy*tan(tilt_x)`` is the same equation
as ``CameraModel.back_project_to_plane``. So the recovered parameters plug straight
into ``stereo_model.reconstruct_3c_field(..., z_world, tilt_x, tilt_y)`` with no
convention translation.

Requires the ``libbulkxcorr2d`` C extension (the ensemble correlation routine). The
particle-image loader and figure writers import the core/cli primitives directly so
calibration stays independent of the v1 calibration service layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    SelfCalibrationResult,
    _load_xcorr_library,
    estimate_pixel_scale,
    run_self_calibration,
)

from .camera_model import CameraModel
from .record import StereoRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model bridge + dewarp world bounds
# ---------------------------------------------------------------------------

def pinhole_from_model(model: CameraModel) -> PinholeCamera:
    """Adapt a calibration ``CameraModel`` to the algorithm core's ``PinholeCamera``.

    A pure field copy (same ``K, dist, R, t`` world->camera and ``image_size``), so
    the resulting camera's world frame is the calibration clicked frame.
    """
    return PinholeCamera(
        K=np.asarray(model.K, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(model.dist, dtype=np.float64).reshape(-1),
        R=np.asarray(model.R, dtype=np.float64).reshape(3, 3),
        t=np.asarray(model.t, dtype=np.float64).reshape(3, 1),
        image_size=(int(model.image_size[0]), int(model.image_size[1])),
    )


def _camera_world_bounds(model: CameraModel) -> Tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` mm of a camera's FOV on the Z=0 plane.

    Samples 20 points along each image edge and back-projects to Z=0 via the
    calibration ray-plane intersection. Mirrors the legacy
    ``camera_model_utils.compute_camera_world_bounds`` but off the calibration model.
    """
    w, h = int(model.image_size[0]), int(model.image_size[1])
    n = 20
    top = np.column_stack([np.linspace(0, w - 1, n), np.zeros(n)])
    bottom = np.column_stack([np.linspace(0, w - 1, n), np.full(n, h - 1)])
    left = np.column_stack([np.zeros(n), np.linspace(0, h - 1, n)])
    right = np.column_stack([np.full(n, w - 1), np.linspace(0, h - 1, n)])
    pts_px = np.vstack([top, bottom, left, right]).astype(np.float64)

    world = model.back_project_to_plane(pts_px, 0.0, 0.0, 0.0)
    valid = ~np.isnan(world).any(axis=1)
    world = world[valid]
    if world.size == 0:
        raise ValueError("all edge projections returned NaN — check the camera model")
    return (
        float(world[:, 0].min()), float(world[:, 0].max()),
        float(world[:, 1].min()), float(world[:, 1].max()),
    )


def stereo_world_bounds(
    model1: CameraModel, model2: CameraModel
) -> Tuple[float, float, float, float]:
    """Intersection of the two cameras' Z=0 FOV boxes — the dewarp grid bounds.

    Matches the grid ``coordinates.mat`` stores, so self-cal sees the same window
    layout production reconstruction will. Raises if the FOVs do not overlap.
    """
    b1 = _camera_world_bounds(model1)
    b2 = _camera_world_bounds(model2)
    xmin, xmax = max(b1[0], b2[0]), min(b1[1], b2[1])
    ymin, ymax = max(b1[2], b2[2]), min(b1[3], b2[3])
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "cameras have no overlapping FOV: "
            f"cam1 x=[{b1[0]:.1f},{b1[1]:.1f}] y=[{b1[2]:.1f},{b1[3]:.1f}]; "
            f"cam2 x=[{b2[0]:.1f},{b2[1]:.1f}] y=[{b2[2]:.1f},{b2[3]:.1f}]"
        )
    return xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------
# Particle-image loader (recorded PIV frames from a base_path dataset)
# ---------------------------------------------------------------------------

def _to_pair_gray(pair: np.ndarray) -> np.ndarray:
    """Coerce a read_pair result to a ``(2, H, W)`` grayscale stack."""
    if pair.ndim == 2:                       # single frame -> duplicate
        return np.stack([pair, pair])
    if pair.ndim == 4:                       # colour (2, H, W, C) -> gray
        return np.stack([
            cv2.cvtColor(pair[0], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(pair[1], cv2.COLOR_BGR2GRAY),
        ])
    return pair                              # already (2, H, W)


def load_particle_pairs(
    config,
    base_path_idx: int,
    cam1: int,
    cam2: int,
    n_images: int,
    apply_filters: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load N frame-A particle images per camera from a base_path PIV dataset.

    Evenly samples across the available frame pairs. When ``apply_filters`` is True
    (default), the same spatial/temporal filters main PIV uses are applied first —
    essential for transmission rigs where a static background dominates the raw
    correlation. Returns ``(images_cam1, images_cam2)`` as lists of 2D float32 frames.

    Ported from the v1 ``self_calibration_service.load_source_images``, importing the
    core/cli loaders directly so calibration does not depend on the v1 service.
    """
    from pivtools_core.image_handling.load_images import (
        load_mask_for_camera,
        read_pair,
    )
    from pivtools_core.image_handling.path_utils import build_piv_camera_path

    total = int(config.num_frame_pairs)
    if n_images >= total:
        indices = list(range(1, total + 1))
    else:
        step = total / n_images
        indices = [min(int(round(1 + i * step)), total) for i in range(n_images)]

    cam1_path = build_piv_camera_path(config, base_path_idx, cam1)
    cam2_path = build_piv_camera_path(config, base_path_idx, cam2)

    pairs1: List[np.ndarray] = []
    pairs2: List[np.ndarray] = []
    for idx in indices:
        try:
            pair1 = _to_pair_gray(read_pair(idx, cam1_path, cam1, config))
            pair2 = _to_pair_gray(read_pair(idx, cam2_path, cam2, config))
            pairs1.append(pair1.astype(np.float32))
            pairs2.append(pair2.astype(np.float32))
        except Exception as e:  # one bad frame must not abort the whole load
            logger.warning("self-cal: failed to load frame %s: %s", idx, e)
            continue

    if not pairs1:
        raise ValueError("no particle images could be loaded for self-calibration")

    stack1 = np.stack(pairs1)  # (N, 2, H, W)
    stack2 = np.stack(pairs2)
    logger.info(
        "self-cal: loaded %d particle pairs from %s / %s",
        stack1.shape[0], cam1_path, cam2_path,
    )

    if apply_filters:
        from pivtools_cli.processing.dask_pipeline import (
            apply_all_filters_slim,
            get_filter_specs,
        )

        filter_specs = get_filter_specs(config)
        mask1 = load_mask_for_camera(cam1, config, base_path_idx)
        mask2 = load_mask_for_camera(cam2, config, base_path_idx)
        if filter_specs or mask1 is not None or mask2 is not None:
            stack1 = apply_all_filters_slim(
                stack1, filter_specs=filter_specs, pixel_mask=mask1
            )
            stack2 = apply_all_filters_slim(
                stack2, filter_specs=filter_specs, pixel_mask=mask2
            )
            logger.info("self-cal: applied %d main-pipeline filters", len(filter_specs))

    # Frame A only — self-cal correlates cam1 vs cam2 at the same time instant.
    images_cam1 = [stack1[i, 0] for i in range(stack1.shape[0])]
    images_cam2 = [stack2[i, 0] for i in range(stack2.shape[0])]
    return images_cam1, images_cam2


# ---------------------------------------------------------------------------
# Run + record-block packing
# ---------------------------------------------------------------------------

def run(
    record: StereoRecord,
    images_cam1: List[np.ndarray],
    images_cam2: List[np.ndarray],
    *,
    window_size: int = 64,
    overlap: float = 50.0,
    max_iterations: int = 10,
    convergence_threshold: float = 0.1,
    quality_threshold: float = 0.3,
    world_bounds: Optional[Tuple[float, float, float, float]] = None,
    figure_dir: Optional[Path] = None,
) -> SelfCalibrationResult:
    """Run self-calibration for a calibration stereo record.

    Builds ``PinholeCamera`` objects from the record's two models, computes the
    dewarp world bounds (or uses ``world_bounds`` if given) + pixel scale, runs the
    iterative Wieneke loop, and (when ``figure_dir`` is given) writes the diagnostic
    figures there. Raises a clear error if the ``libbulkxcorr2d`` C extension is
    unavailable.
    """
    try:
        _load_xcorr_library()
    except Exception as e:
        raise RuntimeError(
            "stereo self-calibration needs the libbulkxcorr2d C extension; the "
            f"editable install appears to have been built without it ({e})"
        ) from e

    cam1 = pinhole_from_model(record.model1)
    cam2 = pinhole_from_model(record.model2)
    bounds = world_bounds if world_bounds is not None else stereo_world_bounds(
        record.model1, record.model2
    )
    logger.info(
        "self-cal FOV intersection: x=[%.1f,%.1f] y=[%.1f,%.1f] mm", *bounds
    )

    result = run_self_calibration(
        cam1, cam2, images_cam1, images_cam2,
        world_bounds=bounds,
        window_size=window_size,
        overlap=overlap,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        quality_threshold=quality_threshold,
    )

    if figure_dir is not None:
        try:
            from . import figures as c2figs

            mm_per_pixel = estimate_pixel_scale(cam1, cam2, bounds)
            c2figs.write_self_cal_figures(
                result, cam1, cam2, images_cam1, images_cam2,
                world_bounds=bounds, mm_per_pixel=mm_per_pixel,
                figure_dir=Path(figure_dir),
                cam1_num=int(record.cam1), cam2_num=int(record.cam2),
            )
        except Exception as e:  # figures are diagnostic — never fail the run on them
            logger.warning("self-cal: failed to write figures: %s", e)

    return result


def rebake_record(
    record: StereoRecord, z_offset: float, tilt_x: float, tilt_y: float
) -> None:
    """Bake a recovered sheet correction into both cameras' extrinsics, in place.

    Redefines the world frame so the sheet ``Z = z_offset + X*tan(tilt_y) +
    Y*tan(tilt_x)`` becomes the new Z=0 plane (the DaVis convention): for each camera
    ``R' = R @ R_corr``, ``t' = R @ t_corr + t`` from
    ``self_cal_frame.plane_to_world_correction``. The cross-camera pose is invariant,
    but ``R_stereo``/``T_stereo`` are recomputed so the stored values stay exactly
    consistent with the rebaked poses.

    Pinhole only: the rebake is a pose redefinition and is undefined for the direct
    image->world polynomial map, so a polynomial record raises rather than silently
    skipping.
    """
    import dataclasses

    from .self_cal_frame import plane_to_world_correction, rebake_pose
    from .stereo_model import compose_stereo

    for label, model in (("model1", record.model1), ("model2", record.model2)):
        if not (hasattr(model, "R") and hasattr(model, "t")):
            raise ValueError(
                f"self-cal rebake requires pinhole models; {label} is "
                f"{type(model).__name__} (no extrinsics to rebake)")

    R_corr, t_corr = plane_to_world_correction(z_offset, tilt_x, tilt_y)
    R1, t1 = rebake_pose(record.model1.R, record.model1.t, R_corr, t_corr)
    R2, t2 = rebake_pose(record.model2.R, record.model2.t, R_corr, t_corr)
    record.model1 = dataclasses.replace(record.model1, R=R1, t=t1)
    record.model2 = dataclasses.replace(record.model2, R=R2, t=t2)
    record.R_stereo, record.T_stereo = compose_stereo(record.model1, record.model2)


def baked_block(
    result: SelfCalibrationResult,
    *,
    n_images: int,
    window_size: int,
    overlap: float,
    source: str = "auto",
) -> dict:
    """Provenance ``self_cal`` block for a correction baked into the extrinsics.

    The applied correction (``z_offset``/``tilt_x``/``tilt_y``) is zero — it now lives
    in the rebaked poses, so reconstruction consumers (``record.sc_*``,
    ``reconstruct_3c_field``) apply nothing further. The recovered sheet is preserved
    under ``fitted_*`` and ``baked=1`` marks the record as carrying the correction in
    its extrinsics. Scalars + one string only, so it round-trips through
    ``record._meta_to_dict`` / ``_meta_from`` unchanged.
    """
    return {
        "z_offset": 0.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
        "baked": 1,
        "fitted_z_offset": float(result.z_offset),
        "fitted_tilt_x": float(result.tilt_x),
        "fitted_tilt_y": float(result.tilt_y),
        "converged": int(bool(result.converged)),
        "final_rms_disparity": float(result.final_rms_disparity),
        "n_iterations": int(result.n_iterations),
        "n_images": int(n_images),
        "window_size": int(window_size),
        "overlap": float(overlap),
        "source": source,
    }
