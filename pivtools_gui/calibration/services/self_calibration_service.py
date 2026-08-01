"""
self_calibration_service.py

Service layer for stereo self-calibration. Used by both GUI routes and CLI.
Handles camera loading, image loading, dewarp overlay generation, and
orchestrating the self-calibration pipeline.
"""

import logging
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from pivtools_gui.calibration.camera_model_utils import (
    compute_camera_world_bounds,
    load_cameras_from_stereo_model,
)
from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    SelfCalibrationResult,
    compute_dewarp_maps,
    dewarp_image,
    estimate_pixel_scale,
    run_self_calibration,
)

logger = logging.getLogger(__name__)


def load_stereo_cameras(
    base_dir: str, cam1_num: int, cam2_num: int, method: str
) -> Tuple[PinholeCamera, PinholeCamera, dict, dict]:
    """Load PinholeCamera objects from the stereo model.

    Returns (cam1, cam2, model_data1, model_data2).
    """
    return load_cameras_from_stereo_model(base_dir, cam1_num, cam2_num)


def compute_stereo_world_bounds(
    md1: dict, md2: dict
) -> Tuple[float, float, float, float]:
    """World-XY bounds for the stereo dewarp grid.

    Mirrors the production vector-calibration logic: project each camera's
    image edges to the Z=0 world plane via ``_pixels_to_world_mm`` (the same
    function used by ``stereo_reconstruction_production.py`` to build the
    output ``coordinates.mat``), then take the intersection of the two AABBs.

    The resulting bounds match the grid that `coordinates.mat` stores, so
    self-cal operates on the same window layout production PIV will see.
    Windows that fall outside both cameras' valid-data region will become
    disparity outliers and are cleaned by the existing median outlier +
    infill pass.

    Returns (xmin, xmax, ymin, ymax) in mm.
    """
    b1 = compute_camera_world_bounds(
        md1["camera_matrix"], md1["dist_coeffs"],
        md1["rvec"], md1["tvec"],
        md1["image_width"], md1["image_height"],
    )
    b2 = compute_camera_world_bounds(
        md2["camera_matrix"], md2["dist_coeffs"],
        md2["rvec"], md2["tvec"],
        md2["image_width"], md2["image_height"],
    )
    xmin = max(b1[0], b2[0])
    xmax = min(b1[1], b2[1])
    ymin = max(b1[2], b2[2])
    ymax = min(b1[3], b2[3])

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"Cameras have no overlapping FOV. "
            f"Cam1: x=[{b1[0]:.1f},{b1[1]:.1f}], y=[{b1[2]:.1f},{b1[3]:.1f}]. "
            f"Cam2: x=[{b2[0]:.1f},{b2[1]:.1f}], y=[{b2[2]:.1f},{b2[3]:.1f}]."
        )
    return xmin, xmax, ymin, ymax


def load_source_images(
    config, source_path_idx: int, cam1_num: int, cam2_num: int, n_images: int,
    apply_filters: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load N source images (frame A only) for self-cal.

    Evenly samples across available frames if n_images < total.

    When ``apply_filters`` is True (default), the same filter pipeline that
    main PIV processing uses (``apply_all_filters_slim`` with config's
    spatial+temporal specs and per-camera pixel mask) is applied before
    extracting frame A. This is essential for transmission setups where the
    static background dominates the raw correlation — the time/POD filters
    remove it and let the actual particle correspondence drive self-cal.

    Returns lists of frame-A images (numpy 2D float32), one per loaded pair.
    """
    from pivtools_core.image_handling.load_images import (
        read_pair, load_mask_for_camera,
    )
    from pivtools_core.image_handling.path_utils import build_piv_camera_path

    total = config.num_frame_pairs
    if n_images >= total:
        indices = list(range(1, total + 1))
    else:
        # Evenly sample
        step = total / n_images
        indices = [int(round(1 + i * step)) for i in range(n_images)]
        indices = [min(i, total) for i in indices]

    cam1_path = build_piv_camera_path(config, source_path_idx, cam1_num)
    cam2_path = build_piv_camera_path(config, source_path_idx, cam2_num)
    pairs1 = []  # (2, H, W) per index — frames A and B
    pairs2 = []

    for idx in indices:
        try:
            pair1 = read_pair(idx, cam1_path, cam1_num, config)
            pair2 = read_pair(idx, cam2_path, cam2_num, config)
            # Coerce to (2, H, W) float32 grayscale
            if pair1.ndim == 2:
                pair1 = np.stack([pair1, pair1])
            if pair2.ndim == 2:
                pair2 = np.stack([pair2, pair2])
            if pair1.ndim == 4:  # color: (2, H, W, C)
                pair1 = np.stack([
                    cv2.cvtColor(pair1[0], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(pair1[1], cv2.COLOR_BGR2GRAY),
                ])
            if pair2.ndim == 4:
                pair2 = np.stack([
                    cv2.cvtColor(pair2[0], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(pair2[1], cv2.COLOR_BGR2GRAY),
                ])
            pairs1.append(pair1.astype(np.float32))
            pairs2.append(pair2.astype(np.float32))
        except Exception as e:
            logger.warning(f"Failed to load frame {idx}: {e}")
            continue

    if not pairs1:
        raise ValueError("No source images could be loaded")

    # Stack as (N, 2, H, W) — the shape apply_all_filters_slim expects
    stack1 = np.stack(pairs1)
    stack2 = np.stack(pairs2)
    logger.info(
        f"Loaded {stack1.shape[0]} source image pairs from "
        f"{cam1_path} / {cam2_path}, raw range "
        f"cam{cam1_num}=[{stack1.min():.0f},{stack1.max():.0f}] "
        f"cam{cam2_num}=[{stack2.min():.0f},{stack2.max():.0f}]"
    )

    if apply_filters:
        from pivtools_cli.processing.dask_pipeline import (
            apply_all_filters_slim,
            get_filter_specs,
        )
        filter_specs = get_filter_specs(config)
        filter_names = [f.get("type") for f in filter_specs]
        mask1 = load_mask_for_camera(cam1_num, config, source_path_idx)
        mask2 = load_mask_for_camera(cam2_num, config, source_path_idx)
        logger.info(
            f"Applying main-pipeline filters for self-cal: "
            f"filters={filter_names}, "
            f"mask cam{cam1_num}={'yes' if mask1 is not None else 'no'}, "
            f"mask cam{cam2_num}={'yes' if mask2 is not None else 'no'}"
        )
        if filter_specs or mask1 is not None or mask2 is not None:
            stack1 = apply_all_filters_slim(stack1, filter_specs=filter_specs, pixel_mask=mask1)
            stack2 = apply_all_filters_slim(stack2, filter_specs=filter_specs, pixel_mask=mask2)
            logger.info(
                f"Filtered range "
                f"cam{cam1_num}=[{stack1.min():.1f},{stack1.max():.1f}] "
                f"cam{cam2_num}=[{stack2.min():.1f},{stack2.max():.1f}]"
            )
        else:
            logger.info("No filters or masks configured — skipping filter pass")

    # Frame A only — self-cal correlates cam1 vs cam2 at the same time instant
    images_cam1 = [stack1[i, 0] for i in range(stack1.shape[0])]
    images_cam2 = [stack2[i, 0] for i in range(stack2.shape[0])]
    return images_cam1, images_cam2


def generate_dewarp_overlay(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    img1: np.ndarray,
    img2: np.ndarray,
    world_bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    z_offset: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dewarp and overlay: cam1=red, cam2=cyan.

    Returns (overlay_rgb, cam1_normalized_u8, cam2_normalized_u8).
    """
    map1_x, map1_y = compute_dewarp_maps(
        cam1, world_bounds, mm_per_pixel,
        z_offset=z_offset, tilt_x=tilt_x, tilt_y=tilt_y,
    )
    map2_x, map2_y = compute_dewarp_maps(
        cam2, world_bounds, mm_per_pixel,
        z_offset=z_offset, tilt_x=tilt_x, tilt_y=tilt_y,
    )

    dw1 = dewarp_image(img1, map1_x, map1_y)
    dw2 = dewarp_image(img2, map2_x, map2_y)

    # Normalize to 0-255 using percentile contrast stretch.
    # PIV particle images have a dark background (~95% of pixels) with sparse
    # bright particles. Linear min-max normalization crushes everything to
    # near-zero. Percentile-based clipping gives visible contrast.
    def _norm_u8(img):
        # Use only positive pixels (zero/negative = out-of-bounds from remap)
        pos = img[img > 0]
        if pos.size == 0:
            return np.zeros(img.shape, dtype=np.uint8)
        lo = float(np.percentile(pos, 1))
        hi = float(np.percentile(pos, 99.5))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        scaled = (img - lo) / (hi - lo) * 255
        return np.clip(scaled, 0, 255).astype(np.uint8)

    r = _norm_u8(dw1)
    c = _norm_u8(dw2)

    # Red-cyan: cam1 → red, cam2 → green+blue
    overlay = np.stack([r, c, c], axis=-1)
    return overlay, r, c


def run_self_cal_job(
    config,
    base_dir: str,
    source_path_idx: int,
    cam1_num: int,
    cam2_num: int,
    method: str,
    n_images: int = 20,
    window_size: int = 64,
    overlap: float = 50.0,
    convergence_threshold: float = 0.1,
    quality_threshold: float = 0.3,
    skip_below_px: float = 0.0,
    progress_callback: Optional[Callable] = None,
    save_figures: bool = True,
) -> SelfCalibrationResult:
    """Full pipeline: load cameras, load images, run self-calibration, return result.

    When ``save_figures`` is True (default), four diagnostic PNGs are written to
    ``{base_dir}/calibration/stereo_cam{A}_cam{B}/self_cal_figures/``:
      - ``fig1_convergence.png``           — RMS / Z / tilt convergence history
      - ``fig2_disparity_before_after.png`` — disparity field before & after
      - ``fig3_disparity_histograms.png``   — disparity distributions
      - ``fig4_overlay_before_after.png``   — red/cyan dewarped overlay
    """
    if progress_callback:
        progress_callback({"status": "loading_cameras", "progress": 5})

    cam1, cam2, md1, md2 = load_stereo_cameras(base_dir, cam1_num, cam2_num, method)

    if progress_callback:
        progress_callback({"status": "computing_bounds", "progress": 10})

    world_bounds = compute_stereo_world_bounds(md1, md2)
    logger.info(
        f"Stereo FOV intersection: x=[{world_bounds[0]:.1f},{world_bounds[1]:.1f}], "
        f"y=[{world_bounds[2]:.1f},{world_bounds[3]:.1f}] mm"
    )

    if progress_callback:
        progress_callback({"status": "loading_images", "progress": 15})

    images_cam1, images_cam2 = load_source_images(
        config, source_path_idx, cam1_num, cam2_num, n_images
    )

    if progress_callback:
        progress_callback({"status": "running_self_calibration", "progress": 25})

    result = run_self_calibration(
        cam1, cam2,
        images_cam1, images_cam2,
        world_bounds=world_bounds,
        window_size=window_size,
        overlap=overlap,
        convergence_threshold=convergence_threshold,
        quality_threshold=quality_threshold,
        skip_below_px=skip_below_px,
    )

    if save_figures:
        if progress_callback:
            progress_callback({"status": "saving_figures", "progress": 90})
        try:
            mm_per_pixel = estimate_pixel_scale(cam1, cam2, world_bounds)
            save_self_cal_figures(
                result,
                cam1, cam2,
                images_cam1, images_cam2,
                world_bounds=world_bounds,
                mm_per_pixel=mm_per_pixel,
                base_dir=base_dir,
                cam1_num=cam1_num,
                cam2_num=cam2_num,
            )
        except Exception as e:
            # Figures are diagnostic — don't fail the whole job if they fail
            logger.warning(f"Failed to save self-cal figures: {e}")

    if progress_callback:
        progress_callback({"status": "complete", "progress": 100})

    return result


def save_self_cal_figures(
    result: SelfCalibrationResult,
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    images_cam1: List[np.ndarray],
    images_cam2: List[np.ndarray],
    world_bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    base_dir: str,
    cam1_num: int,
    cam2_num: int,
) -> Path:
    """Save the four standard self-calibration diagnostic figures.

    Output directory: ``{base_dir}/calibration/stereo_cam{A}_cam{B}/self_cal_figures/``

    Returns the output directory path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = (
        Path(base_dir) / "calibration"
        / f"stereo_cam{cam1_num}_cam{cam2_num}" / "self_cal_figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = result.history
    if not hist:
        logger.warning("No iteration history — skipping self-cal figures")
        return out_dir

    # ----- fig1: convergence history -----
    iters = [h.iteration for h in hist]
    rms_vals = [h.rms_disparity for h in hist]
    z_vals = [h.cumulative_z for h in hist]
    tx_vals = [math.degrees(h.cumulative_tilt_x) for h in hist]
    ty_vals = [math.degrees(h.cumulative_tilt_y) for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Self-Calibration Convergence — cam{cam1_num} vs cam{cam2_num}",
        fontsize=14, fontweight="bold",
    )
    axes[0, 0].semilogy(iters, rms_vals, "bo-", lw=2, ms=8)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("RMS disparity (px)")
    axes[0, 0].set_title("RMS Disparity Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, z_vals, "rs-", lw=2, ms=8)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Z offset (mm)")
    axes[0, 1].set_title(f"Z Recovery (final: {result.z_offset:.3f} mm)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, tx_vals, "g^-", lw=2, ms=8, label="Tilt X")
    axes[1, 0].plot(iters, ty_vals, "mD-", lw=2, ms=8, label="Tilt Y")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Tilt (deg)")
    axes[1, 0].set_title("Tilt Recovery")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    rows = [
        ["Parameter", "Value"],
        ["Z offset", f"{result.z_offset:.4f} mm"],
        ["Tilt X", f"{math.degrees(result.tilt_x):.4f} deg"],
        ["Tilt Y", f"{math.degrees(result.tilt_y):.4f} deg"],
        ["Final RMS", f"{result.final_rms_disparity:.4f} px"],
        ["Iterations", f"{result.n_iterations}"],
        ["Converged", f"{result.converged}"],
        ["Initial RMS", f"{hist[0].rms_disparity:.2f} px"],
        ["Reduction", f"{hist[0].rms_disparity / max(result.final_rms_disparity, 0.001):.1f}x"],
    ]
    table = axes[1, 1].table(
        cellText=rows, loc="center", cellLoc="center", colWidths=[0.45, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for j in range(2):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#d0d0d0")

    fig.tight_layout()
    fig.savefig(str(out_dir / "fig1_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- fig2 + fig3: disparity field before/after + histograms -----
    dx_b, dy_b = result.dx_before, result.dy_before
    dx_a, dy_a = result.dx_after, result.dy_after

    if dx_b is not None and dx_a is not None:
        mag_b = np.sqrt(
            np.where(np.isfinite(dx_b), dx_b, 0) ** 2
            + np.where(np.isfinite(dy_b), dy_b, 0) ** 2
        )
        mag_a = np.sqrt(
            np.where(np.isfinite(dx_a), dx_a, 0) ** 2
            + np.where(np.isfinite(dy_a), dy_a, 0) ** 2
        )
        vmax = float(np.nanpercentile(mag_b, 95)) if mag_b.size else 1.0
        rms_b = float(np.sqrt(np.nanmean(dx_b ** 2 + dy_b ** 2)))
        rms_a = float(np.sqrt(np.nanmean(dx_a ** 2 + dy_a ** 2)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Disparity Fields: Before vs After Correction",
            fontsize=14, fontweight="bold",
        )
        for row, (dx, dy, mag, rms, lbl) in enumerate([
            (dx_b, dy_b, mag_b, rms_b, "BEFORE"),
            (dx_a, dy_a, mag_a, rms_a, "AFTER"),
        ]):
            im = axes[row, 0].imshow(dx, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 0].set_title(f"dx {lbl}\nmean={np.nanmean(dx):.2f} px")
            plt.colorbar(im, ax=axes[row, 0], shrink=0.8)
            im = axes[row, 1].imshow(dy, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 1].set_title(f"dy {lbl}\nmean={np.nanmean(dy):.2f} px")
            plt.colorbar(im, ax=axes[row, 1], shrink=0.8)
            im = axes[row, 2].imshow(mag, cmap="hot", vmin=0, vmax=vmax)
            axes[row, 2].set_title(f"|d| {lbl}\nRMS={rms:.2f} px")
            plt.colorbar(im, ax=axes[row, 2], shrink=0.8)
        for ax in axes.flat:
            ax.set_xlabel("Window X")
            ax.set_ylabel("Window Y")
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig2_disparity_before_after.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # Mean/std of finite values — these are what self-cal actually
        # minimises (mean → 0) and the irreducible noise floor (std).
        dx_b_mean = float(np.nanmean(dx_b)); dx_b_std = float(np.nanstd(dx_b))
        dx_a_mean = float(np.nanmean(dx_a)); dx_a_std = float(np.nanstd(dx_a))
        dy_b_mean = float(np.nanmean(dy_b)); dy_b_std = float(np.nanstd(dy_b))
        dy_a_mean = float(np.nanmean(dy_a)); dy_a_std = float(np.nanstd(dy_a))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Disparity Distributions: Before vs After\n"
            "(vertical lines mark mean — self-cal's minimisation target)",
            fontsize=13, fontweight="bold",
        )
        bins = np.linspace(-vmax * 1.5, vmax * 1.5, 60)
        ax1.hist(
            dx_b[np.isfinite(dx_b)].ravel(), bins=bins, alpha=0.6, color="red",
            label=f"Before (μ={dx_b_mean:+.3f}, σ={dx_b_std:.2f})",
        )
        ax1.hist(
            dx_a[np.isfinite(dx_a)].ravel(), bins=bins, alpha=0.6, color="green",
            label=f"After (μ={dx_a_mean:+.3f}, σ={dx_a_std:.2f})",
        )
        ax1.axvline(dx_b_mean, color="darkred", lw=2, ls="--")
        ax1.axvline(dx_a_mean, color="darkgreen", lw=2, ls="--")
        ax1.set_xlabel("dx disparity (px)")
        ax1.set_ylabel("Count")
        ax1.set_title("dx Disparity")
        ax1.legend(fontsize=9)
        ax1.axvline(0, color="k", ls="-", alpha=0.3)

        ax2.hist(
            dy_b[np.isfinite(dy_b)].ravel(), bins=bins, alpha=0.6, color="red",
            label=f"Before (μ={dy_b_mean:+.3f}, σ={dy_b_std:.2f})",
        )
        ax2.hist(
            dy_a[np.isfinite(dy_a)].ravel(), bins=bins, alpha=0.6, color="green",
            label=f"After (μ={dy_a_mean:+.3f}, σ={dy_a_std:.2f})",
        )
        ax2.axvline(dy_b_mean, color="darkred", lw=2, ls="--")
        ax2.axvline(dy_a_mean, color="darkgreen", lw=2, ls="--")
        ax2.set_xlabel("dy disparity (px)")
        ax2.set_ylabel("Count")
        ax2.set_title("dy Disparity")
        ax2.legend(fontsize=9)
        ax2.axvline(0, color="k", ls="-", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig3_disparity_histograms.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # ----- fig4: dewarped red/cyan overlay before/after -----
    if images_cam1 and images_cam2:
        img1 = images_cam1[0]
        img2 = images_cam2[0]
        m1b = compute_dewarp_maps(cam1, world_bounds, mm_per_pixel)
        m2b = compute_dewarp_maps(cam2, world_bounds, mm_per_pixel)
        dw1b = dewarp_image(img1, m1b[0], m1b[1])
        dw2b = dewarp_image(img2, m2b[0], m2b[1])
        m1a = compute_dewarp_maps(
            cam1, world_bounds, mm_per_pixel,
            result.z_offset, result.tilt_x, result.tilt_y,
        )
        m2a = compute_dewarp_maps(
            cam2, world_bounds, mm_per_pixel,
            result.z_offset, result.tilt_x, result.tilt_y,
        )
        dw1a = dewarp_image(img1, m1a[0], m1a[1])
        dw2a = dewarp_image(img2, m2a[0], m2a[1])

        def _norm(im):
            pos = im[im > 0]
            if pos.size == 0:
                return np.zeros(im.shape, dtype=np.uint8)
            lo = float(np.percentile(pos, 1))
            hi = float(np.percentile(pos, 99.5))
            if hi - lo < 1e-6:
                hi = lo + 1.0
            return np.clip((im - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        def _rc(d1, d2):
            r = _norm(d1); c = _norm(d2)
            return np.stack([r, c, c], axis=-1)

        ov_before = _rc(dw1b, dw2b)
        ov_after = _rc(dw1a, dw2a)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Dewarped Particle Overlay: Before vs After Self-Cal",
            fontsize=13, fontweight="bold",
        )
        x_min, x_max, y_min, y_max = world_bounds
        extent = [x_min, x_max, y_min, y_max]
        ax1.imshow(ov_before, extent=extent, origin="lower", aspect="equal")
        ax1.set_title("BEFORE (Z=0, no tilt)")
        ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
        ax2.imshow(ov_after, extent=extent, origin="lower", aspect="equal")
        ax2.set_title(
            f"AFTER (Z={result.z_offset:.2f} mm, "
            f"tX={math.degrees(result.tilt_x):.2f}°)"
        )
        ax2.set_xlabel("X (mm)")
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig4_overlay_before_after.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # ----- fig5: BEFORE vs AFTER correlation planes at 6 world positions -----
        # Pulls correlation planes directly from the C-library output stored
        # in the result (corr_first_iter = iter 1 = before corrections;
        # corr_last_iter = final iter = after corrections). No Python
        # recomputation — the planes shown are pixel-exact what self-cal
        # optimised on. Each probe snaps to the nearest C-library window
        # centre so the plane is a real self-cal measurement, not an interp.
        try:
            if (
                result.corr_first_iter is None
                or result.corr_last_iter is None
                or result.win_ctrs_x is None
                or result.win_ctrs_y is None
                or result.window_size_used is None
            ):
                raise RuntimeError(
                    "correlation planes missing from SelfCalibrationResult"
                )

            corr_before = result.corr_first_iter
            corr_after = result.corr_last_iter
            win_ctrs_x = np.asarray(result.win_ctrs_x, dtype=np.float32)
            win_ctrs_y = np.asarray(result.win_ctrs_y, dtype=np.float32)
            ws_full = int(result.window_size_used)

            # Central crop for display: at auto-bumped window sizes the peak
            # is always within a few pixels of centre, so cropping to 64×64
            # makes the peak visually prominent instead of a single pixel.
            crop = min(64, ws_full)
            c0 = (ws_full - crop) // 2
            c1 = c0 + crop

            x_min, x_max, y_min, y_max = world_bounds
            mx = (x_max - x_min) * 0.15
            my = (y_max - y_min) * 0.15
            xs_probe = np.linspace(x_min + mx, x_max - mx, 3)
            ys_probe = np.linspace(y_min + my, y_max - my, 2)

            # Dewarp frame 0 once at corrected position for context panels
            img1_a = dewarp_image(images_cam1[0], m1a[0], m1a[1])
            img2_a = dewarp_image(images_cam2[0], m2a[0], m2a[1])
            out_h, out_w = img1_a.shape

            W_thumb = 64
            half_thumb = W_thumb // 2

            def _stretch(im, lo_pct=2.0, hi_pct=99.5):
                lo = float(np.percentile(im, lo_pct))
                hi = float(np.percentile(im, hi_pct))
                if hi - lo < 1e-6:
                    hi = lo + 1.0
                return np.clip((im - lo) / (hi - lo), 0, 1)

            n_iters = len(result.history)
            crop_label = (
                f" (display crop: {crop}×{crop})" if crop < ws_full else ""
            )

            # Layout: 2 rows × (3 x-positions × 5 panels) = 2 × 15 cols
            fig, axes = plt.subplots(2, 15, figsize=(32, 7))
            fig.suptitle(
                f"Correlation planes BEFORE vs AFTER self-cal — "
                f"cam{cam1_num} vs cam{cam2_num} (z={result.z_offset:.4f} mm, "
                f"tx={math.degrees(result.tilt_x):+.4f}°, "
                f"ty={math.degrees(result.tilt_y):+.4f}°)\n"
                f"per position: cam{cam1_num} f0 | cam{cam2_num} f0 | "
                f"R/G overlay | corr BEFORE (iter 1) | "
                f"corr AFTER (iter {n_iters}) — "
                f"C library window: {ws_full}×{ws_full}{crop_label}",
                fontsize=11, fontweight="bold",
            )

            for row, y_t in enumerate(ys_probe[::-1]):
                for col, x_t in enumerate(xs_probe):
                    base_col = col * 5

                    # Snap probe world coord to nearest C-library window centre
                    probe_px_x = (x_t - x_min) / mm_per_pixel
                    probe_px_y = (y_t - y_min) / mm_per_pixel
                    ix = int(np.argmin(np.abs(win_ctrs_x - probe_px_x)))
                    iy = int(np.argmin(np.abs(win_ctrs_y - probe_px_y)))
                    snap_px_x = float(win_ctrs_x[ix])
                    snap_px_y = float(win_ctrs_y[iy])
                    snap_mm_x = x_min + snap_px_x * mm_per_pixel
                    snap_mm_y = y_min + snap_px_y * mm_per_pixel

                    # Thumbnail crop around snapped window centre
                    cx_p = int(round(snap_px_x))
                    cy_p = int(round(snap_px_y))
                    x0 = cx_p - half_thumb
                    x1 = cx_p + half_thumb
                    y0 = cy_p - half_thumb
                    y1 = cy_p + half_thumb
                    if x0 < 0 or y0 < 0 or x1 > out_w or y1 > out_h:
                        for k in range(5):
                            axes[row, base_col + k].axis("off")
                        continue

                    s1 = _stretch(img1_a[y0:y1, x0:x1])
                    s2 = _stretch(img2_a[y0:y1, x0:x1])
                    axes[row, base_col + 0].imshow(
                        s1, origin="lower", cmap="inferno", vmin=0, vmax=1,
                    )
                    axes[row, base_col + 0].set_title(
                        f"({snap_mm_x:+.0f},{snap_mm_y:+.0f}) c{cam1_num} f0",
                        fontsize=9,
                    )
                    axes[row, base_col + 1].imshow(
                        s2, origin="lower", cmap="inferno", vmin=0, vmax=1,
                    )
                    axes[row, base_col + 1].set_title(
                        f"c{cam2_num} f0", fontsize=9,
                    )
                    rgb = np.zeros((W_thumb, W_thumb, 3), dtype=np.float32)
                    rgb[..., 0] = s1
                    rgb[..., 1] = s2
                    axes[row, base_col + 2].imshow(rgb, origin="lower")
                    axes[row, base_col + 2].set_title(
                        f"R=c{cam1_num}, G=c{cam2_num}", fontsize=9,
                    )

                    # Pull BEFORE/AFTER planes straight from the C library
                    # result, crop central region, auto-scale each plane
                    plane_before = corr_before[iy, ix, c0:c1, c0:c1]
                    plane_after = corr_after[iy, ix, c0:c1, c0:c1]
                    vmin_b = float(np.min(plane_before))
                    vmax_b = float(np.max(plane_before))
                    vmin_a = float(np.min(plane_after))
                    vmax_a = float(np.max(plane_after))

                    axes[row, base_col + 3].imshow(
                        plane_before, origin="lower", cmap="viridis",
                        vmin=vmin_b, vmax=vmax_b,
                    )
                    axes[row, base_col + 3].set_title(
                        f"BEFORE iter 1\npeak={vmax_b:.3g}", fontsize=8,
                    )

                    axes[row, base_col + 4].imshow(
                        plane_after, origin="lower", cmap="viridis",
                        vmin=vmin_a, vmax=vmax_a,
                    )
                    axes[row, base_col + 4].set_title(
                        f"AFTER iter {n_iters}\npeak={vmax_a:.3g}",
                        fontsize=8,
                    )

                    logger.info(
                        f"  fig5 probe ({snap_mm_x:+6.1f},{snap_mm_y:+6.1f}) "
                        f"mm [win {iy},{ix}]: "
                        f"peak_before={vmax_b:.3g} peak_after={vmax_a:.3g}"
                    )

                    for k in range(5):
                        axes[row, base_col + k].set_xticks([])
                        axes[row, base_col + k].set_yticks([])

            fig.tight_layout()
            fig.savefig(
                str(out_dir / "fig5_correlation_probes.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"fig5_correlation_probes failed: {e}")

    # ----- fig6: forward-model decomposition -----
    # Forward-predict the iter-1 BEFORE disparity field from iter-1's
    # recovered (delta_z, delta_tilt_x, delta_tilt_y) using the SAME linear
    # model that fit_disparity_plane inverts:
    #   D_proj = disp_px_per_mm * (dz + tan(dty)*X + tan(dtx)*Y)
    # then back-project onto (dx, dy) via disp_direction.
    #
    # By least-squares construction the predicted field IS the best-fit plane
    # through the observed iter-1 field, so the residual (observed − predicted)
    # is exactly what self-cal's model cannot explain — the noise floor. A
    # good converged self-cal has residual.mean ≈ 0 and residual looks
    # statistically identical to dy_after.
    #
    # Why iter-1 and not cumulative-final: the dewarp is nonlinear in Z, so
    # the cumulative correction spans several linearisations and does NOT
    # predict iter-1's observation. Iter 1's linear fit does.
    if (
        result.dx_before is not None
        and result.dy_before is not None
        and result.grid_x_mm is not None
        and result.grid_y_mm is not None
        and result.disp_px_per_mm is not None
        and result.disp_direction is not None
        and result.history
    ):
        try:
            dx_o = result.dx_before
            dy_o = result.dy_before
            Xm = result.grid_x_mm
            Ym = result.grid_y_mm
            dpm = float(result.disp_px_per_mm)
            ddir = np.asarray(result.disp_direction, dtype=np.float64)

            # Iter-1's recovered corrections (for iter 1, cumulative == delta)
            h1 = result.history[0]
            z_i1 = float(h1.cumulative_z)
            tx_i1 = float(h1.cumulative_tilt_x)
            ty_i1 = float(h1.cumulative_tilt_y)

            # Forward model — linear, exactly inverts fit_disparity_plane:
            # tilt about Y → X-gradient, tilt about X → Y-gradient (both +).
            disp_mag_pred = dpm * (
                z_i1 + math.tan(ty_i1) * Xm + math.tan(tx_i1) * Ym
            )
            dx_pred = disp_mag_pred * ddir[0]
            dy_pred = disp_mag_pred * ddir[1]

            # Residuals = observed BEFORE - predicted BEFORE
            dx_res = dx_o - dx_pred
            dy_res = dy_o - dy_pred

            # Shared colour scale for the dy column (dominant component) so
            # the three panels are visually comparable.
            finite_obs = dy_o[np.isfinite(dy_o)]
            if finite_obs.size:
                v = float(np.nanpercentile(np.abs(finite_obs), 98))
            else:
                v = 1.0
            if v < 0.1:
                v = 0.1

            # Statistics for the titles
            def _stats(a):
                finite = a[np.isfinite(a)]
                if finite.size == 0:
                    return 0.0, 0.0
                return float(np.mean(finite)), float(np.std(finite))

            dy_o_m, dy_o_s = _stats(dy_o)
            dy_p_m, dy_p_s = _stats(dy_pred)
            dy_r_m, dy_r_s = _stats(dy_res)
            dx_o_m, dx_o_s = _stats(dx_o)
            dx_p_m, dx_p_s = _stats(dx_pred)
            dx_r_m, dx_r_s = _stats(dx_res)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(
                "Forward-Model Decomposition — "
                f"cam{cam1_num} vs cam{cam2_num}\n"
                f"Iter-1 fit: dz={z_i1:+.4f} mm, "
                f"dtx={math.degrees(tx_i1):+.4f}°, "
                f"dty={math.degrees(ty_i1):+.4f}° | "
                f"sensitivity={dpm:.2f} px/mm, "
                f"direction=({ddir[0]:+.3f}, {ddir[1]:+.3f}) | "
                f"Final converged: z={result.z_offset:+.4f} mm, "
                f"tx={math.degrees(result.tilt_x):+.4f}°, "
                f"ty={math.degrees(result.tilt_y):+.4f}°\n"
                "PREDICTED is the linear fit through iter-1 OBSERVED, "
                "so RESIDUAL = observation - fit = noise floor. "
                "Small residual mean and structure-less residual = model "
                "explains the observation.",
                fontsize=11, fontweight="bold",
            )

            panels = [
                ("dy OBSERVED (iter 1)", dy_o, dy_o_m, dy_o_s),
                ("dy PREDICTED from (z,tx,ty)", dy_pred, dy_p_m, dy_p_s),
                ("dy RESIDUAL = obs − pred", dy_res, dy_r_m, dy_r_s),
            ]
            for i, (title, fld, mean_v, std_v) in enumerate(panels):
                im = axes[0, i].imshow(
                    fld, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                )
                axes[0, i].set_title(
                    f"{title}\nμ={mean_v:+.3f} px, σ={std_v:.2f} px",
                    fontsize=11,
                )
                axes[0, i].set_xlabel("Window X")
                axes[0, i].set_ylabel("Window Y")
                plt.colorbar(im, ax=axes[0, i], shrink=0.8)

            panels_x = [
                ("dx OBSERVED (iter 1)", dx_o, dx_o_m, dx_o_s),
                ("dx PREDICTED", dx_pred, dx_p_m, dx_p_s),
                ("dx RESIDUAL", dx_res, dx_r_m, dx_r_s),
            ]
            for i, (title, fld, mean_v, std_v) in enumerate(panels_x):
                im = axes[1, i].imshow(
                    fld, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                )
                axes[1, i].set_title(
                    f"{title}\nμ={mean_v:+.3f} px, σ={std_v:.2f} px",
                    fontsize=11,
                )
                axes[1, i].set_xlabel("Window X")
                axes[1, i].set_ylabel("Window Y")
                plt.colorbar(im, ax=axes[1, i], shrink=0.8)

            fig.tight_layout()
            fig.savefig(
                str(out_dir / "fig6_forward_model.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

            logger.info(
                f"fig6 forward-model: dy mean OBS={dy_o_m:+.3f} -> "
                f"PRED={dy_p_m:+.3f} -> RES={dy_r_m:+.3f} px "
                f"(residual std={dy_r_s:.2f}); "
                f"dx mean OBS={dx_o_m:+.3f} -> PRED={dx_p_m:+.3f} -> "
                f"RES={dx_r_m:+.3f} px"
            )
        except Exception as e:
            logger.warning(f"fig6_forward_model failed: {e}")

    # ----- correlation_planes.mat (first + last iteration C-correlator output) -----
    try:
        from scipy.io import savemat
        if result.corr_first_iter is not None and result.corr_last_iter is not None:
            mat_path = out_dir / "correlation_planes.mat"
            mat_data = {
                "corr_first_iter": result.corr_first_iter.astype(np.float32),
                "corr_last_iter": result.corr_last_iter.astype(np.float32),
                "win_ctrs_x": np.asarray(result.win_ctrs_x, dtype=np.float32),
                "win_ctrs_y": np.asarray(result.win_ctrs_y, dtype=np.float32),
                "n_win_x": np.int32(result.n_win_x or 0),
                "n_win_y": np.int32(result.n_win_y or 0),
                "window_size": np.int32(result.window_size_used or 0),
                "mm_per_pixel": np.float64(result.mm_per_pixel or 0.0),
                "world_bounds": np.asarray(world_bounds, dtype=np.float64),
                "z_offset_final": np.float64(result.z_offset),
                "tilt_x_final": np.float64(result.tilt_x),
                "tilt_y_final": np.float64(result.tilt_y),
                "n_iterations": np.int32(result.n_iterations),
            }
            if result.grid_x_mm is not None:
                mat_data["grid_x_mm"] = np.asarray(result.grid_x_mm, dtype=np.float32)
            if result.grid_y_mm is not None:
                mat_data["grid_y_mm"] = np.asarray(result.grid_y_mm, dtype=np.float32)
            savemat(str(mat_path), mat_data, do_compression=True)
            logger.info(
                f"Saved correlation_planes.mat ({result.n_win_y}x"
                f"{result.n_win_x} windows × {result.window_size_used}² px) "
                f"to {mat_path}"
            )
        else:
            logger.warning(
                "Skipping correlation_planes.mat — "
                "no correlation planes captured (run aborted before iter 1?)"
            )
    except Exception as e:
        logger.warning(f"Failed to save correlation_planes.mat: {e}")

    logger.info(f"Saved self-cal figures to {out_dir}")
    return out_dir


def _self_cal_file_path(config, cam1: int = None, cam2: int = None) -> Path:
    """Return path to the self_calibration.yaml file for the active stereo pair."""
    base = Path(str(config.base_paths[0]))
    if cam1 is None or cam2 is None:
        pairs = config.stereo_pairs
        if pairs:
            cam1, cam2 = pairs[0]
        else:
            cam1, cam2 = 1, 2
    return base / "calibration" / f"stereo_cam{cam1}_cam{cam2}" / "self_calibration.yaml"


def load_self_cal_from_file(config, cam1: int = None, cam2: int = None) -> dict:
    """Load self-calibration results from the stereo model directory.

    Returns empty dict if no file exists.
    """
    path = _self_cal_file_path(config, cam1, cam2)
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load self-cal from {path}: {e}")
        return {}


def save_self_cal_to_file(
    config, result: SelfCalibrationResult,
    cam1: int = None, cam2: int = None, **params
):
    """Save self-calibration results alongside the stereo model."""
    import yaml

    sc_data = {
        "z_offset": float(result.z_offset),
        "tilt_x": float(result.tilt_x),
        "tilt_y": float(result.tilt_y),
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "final_rms_disparity": float(result.final_rms_disparity),
    }
    sc_data.update(params)

    path = _self_cal_file_path(config, cam1, cam2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(sc_data, f, default_flow_style=False)

    logger.info(f"Saved self-cal to {path}")


def clear_self_cal_file(config, cam1: int = None, cam2: int = None):
    """Delete the self_calibration.yaml file (invalidates old results)."""
    path = _self_cal_file_path(config, cam1, cam2)
    if path.exists():
        path.unlink()
        logger.info(f"Cleared self-cal: {path}")


def save_self_cal_to_config(config, result: SelfCalibrationResult, **params):
    """Save self-calibration results to file alongside stereo model.

    Results are saved to the canonical file location only
    ({base_path}/calibration/stereo_cam{A}_cam{B}/self_calibration.yaml).
    The config.py property reads from this file directly.
    """
    save_self_cal_to_file(config, result, **params)
    logger.info(
        f"Saved self-cal to file: z={result.z_offset:.4f} mm, "
        f"tilt_x={math.degrees(result.tilt_x):.4f}°, "
        f"tilt_y={math.degrees(result.tilt_y):.4f}°"
    )


def result_to_dict(result: SelfCalibrationResult) -> dict:
    """Convert SelfCalibrationResult to JSON-serializable dict."""
    # Prepend iteration 0: baseline RMS before any corrections were applied
    initial_rms = result.history[0].rms_disparity if result.history else 0.0
    history = [
        {
            "iteration": 0,
            "rms_disparity": initial_rms,
            "delta_z": 0.0,
            "delta_tilt_x": 0.0,
            "delta_tilt_y": 0.0,
            "cumulative_z": 0.0,
            "cumulative_tilt_x": 0.0,
            "cumulative_tilt_y": 0.0,
        },
    ] + [
        {
            "iteration": rec.iteration,
            "rms_disparity": rec.rms_disparity,
            "delta_z": rec.delta_z,
            "delta_tilt_x": rec.delta_tilt_x,
            "delta_tilt_y": rec.delta_tilt_y,
            "cumulative_z": rec.cumulative_z,
            "cumulative_tilt_x": rec.cumulative_tilt_x,
            "cumulative_tilt_y": rec.cumulative_tilt_y,
        }
        for rec in result.history
    ]
    return {
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "z_offset": float(result.z_offset),
        "tilt_x": float(result.tilt_x),
        "tilt_y": float(result.tilt_y),
        "tilt_x_deg": float(math.degrees(result.tilt_x)),
        "tilt_y_deg": float(math.degrees(result.tilt_y)),
        "final_rms_disparity": float(result.final_rms_disparity),
        "history": history,
    }
