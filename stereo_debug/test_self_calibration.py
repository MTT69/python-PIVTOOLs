"""
Synthetic test for stereo PIV self-calibration.

Creates a synthetic stereo setup with known laser-sheet misalignment,
runs self-calibration, and generates 5 diagnostic figures.

Usage
-----
    python scripts/test_self_calibration.py
"""

import logging
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Ensure the parent package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    compute_dewarp_maps,
    dewarp_image,
    run_self_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRUE_Z_OFFSET = 0.3      # mm
TRUE_TILT_X = 0.002       # rad
TRUE_TILT_Y = -0.001      # rad
STEREO_ANGLE_DEG = 30.0
N_IMAGE_PAIRS = 20
N_PARTICLES = 2000
WORLD_BOUNDS = (-40.0, 40.0, -40.0, 40.0)  # mm
OUTPUT_SIZE = (512, 512)
WINDOW_SIZE = 64
OVERLAP = 50.0
FOCAL_LENGTH_PX = 1000.0
IMAGE_SIZE = (1024, 1024)  # (width, height)
BASELINE_MM = 200.0
PARTICLE_SIGMA = 2.0
PARTICLE_INTENSITY = 200

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "self_cal_figures")


# ---------------------------------------------------------------------------
# Synthetic camera + particle generation
# ---------------------------------------------------------------------------

def create_stereo_cameras(
    stereo_angle_deg: float = STEREO_ANGLE_DEG,
    focal_length_px: float = FOCAL_LENGTH_PX,
    image_size: tuple = IMAGE_SIZE,
    baseline_mm: float = BASELINE_MM,
) -> tuple:
    """Create a symmetric stereo camera pair.

    Cameras are placed symmetrically about the Y-axis at +/-theta,
    looking at the origin.
    """
    w, h = image_size
    theta = math.radians(stereo_angle_deg / 2.0)

    K = np.array([
        [focal_length_px, 0.0, w / 2.0],
        [0.0, focal_length_px, h / 2.0],
        [0.0, 0.0, 1.0],
    ])
    dist = np.zeros(5)

    # Camera positions in world frame
    z_cam = baseline_mm / (2.0 * math.tan(theta))
    cam1_pos = np.array([baseline_mm / 2.0, 0.0, z_cam])
    cam2_pos = np.array([-baseline_mm / 2.0, 0.0, z_cam])

    def look_at_rotation(cam_pos, target=np.array([0, 0, 0])):
        """Compute rotation matrix for camera looking at target."""
        z_axis = target - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        up = np.array([0.0, -1.0, 0.0])  # Y-down in image
        x_axis = np.cross(z_axis, up)
        norm = np.linalg.norm(x_axis)
        if norm < 1e-10:
            up = np.array([0.0, 0.0, -1.0])
            x_axis = np.cross(z_axis, up)
            norm = np.linalg.norm(x_axis)
        x_axis = x_axis / norm
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        return R

    R1 = look_at_rotation(cam1_pos)
    R2 = look_at_rotation(cam2_pos)

    t1 = (-R1 @ cam1_pos).reshape(3, 1)
    t2 = (-R2 @ cam2_pos).reshape(3, 1)

    cam1 = PinholeCamera(K=K.copy(), dist=dist.copy(), R=R1, t=t1, image_size=image_size)
    cam2 = PinholeCamera(K=K.copy(), dist=dist.copy(), R=R2, t=t2, image_size=image_size)
    return cam1, cam2


def generate_particles(
    n_particles: int,
    x_range: tuple,
    y_range: tuple,
    z_offset: float,
    tilt_x: float,
    tilt_y: float,
) -> np.ndarray:
    """Generate random particles on a tilted laser sheet."""
    x = np.random.uniform(x_range[0], x_range[1], n_particles)
    y = np.random.uniform(y_range[0], y_range[1], n_particles)
    z = z_offset + x * np.tan(tilt_y) + y * np.tan(tilt_x)
    return np.column_stack([x, y, z])


def render_particles(
    particles: np.ndarray,
    camera: PinholeCamera,
    particle_sigma: float = PARTICLE_SIGMA,
    intensity: int = PARTICLE_INTENSITY,
) -> np.ndarray:
    """Render particles as Gaussian spots on an image."""
    w, h = camera.image_size
    image = np.zeros((h, w), dtype=np.float32)

    pts2d = camera.project(particles)

    # Pre-compute Gaussian kernel
    half_size = int(math.ceil(4 * particle_sigma))
    kernel_size = 2 * half_size + 1
    kx = np.arange(kernel_size) - half_size
    ky = np.arange(kernel_size) - half_size
    kxx, kyy = np.meshgrid(kx, ky)
    kernel_base = np.exp(-(kxx ** 2 + kyy ** 2) / (2 * particle_sigma ** 2))

    for pt in pts2d:
        px, py = pt
        ix, iy = int(round(px)), int(round(py))
        if ix < half_size or ix >= w - half_size:
            continue
        if iy < half_size or iy >= h - half_size:
            continue

        # Sub-pixel offset
        dx = px - ix
        dy = py - iy
        kernel = intensity * np.exp(
            -((kxx - dx) ** 2 + (kyy - dy) ** 2) / (2 * particle_sigma ** 2)
        )

        image[iy - half_size:iy + half_size + 1,
              ix - half_size:ix + half_size + 1] += kernel

    # Add noise and clip
    noise = np.random.normal(0, 5, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 255)
    return image.astype(np.uint8)


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

def figure_1_dewarped_overlay(
    cam1, cam2, images_cam1, images_cam2, output_size, world_bounds,
    z_before, tx_before, ty_before, z_after, tx_after, ty_after, save_dir,
):
    """Figure 1: Dewarped image overlay (2x2)."""
    img1 = images_cam1[0]
    img2 = images_cam2[0]

    # Before correction (Z=0)
    m1b = compute_dewarp_maps(cam1, output_size, world_bounds, z_before, tx_before, ty_before)
    m2b = compute_dewarp_maps(cam2, output_size, world_bounds, z_before, tx_before, ty_before)
    dw1b = dewarp_image(img1, m1b[0], m1b[1])
    dw2b = dewarp_image(img2, m2b[0], m2b[1])

    # After correction
    m1a = compute_dewarp_maps(cam1, output_size, world_bounds, z_after, tx_after, ty_after)
    m2a = compute_dewarp_maps(cam2, output_size, world_bounds, z_after, tx_after, ty_after)
    dw1a = dewarp_image(img1, m1a[0], m1a[1])
    dw2a = dewarp_image(img2, m2a[0], m2a[1])

    # Build RGB overlays
    overlay_before = np.zeros((*output_size, 3), dtype=np.uint8)
    overlay_before[..., 0] = dw1b  # red = cam1
    overlay_before[..., 1] = dw2b  # green = cam2

    overlay_after = np.zeros((*output_size, 3), dtype=np.uint8)
    overlay_after[..., 0] = dw1a
    overlay_after[..., 1] = dw1a  # both channels → yellow when aligned
    overlay_after[..., 1] = np.maximum(overlay_after[..., 1], dw2a)
    overlay_after[..., 0] = np.maximum(overlay_after[..., 0], dw2a)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(dw1b, cmap="gray")
    axes[0, 0].set_title("Camera 1 (dewarped)")
    axes[0, 1].imshow(dw2b, cmap="gray")
    axes[0, 1].set_title("Camera 2 (dewarped)")
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title("BEFORE: Red=Cam1, Green=Cam2")
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title("AFTER: Yellow = aligned")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("Figure 1: Dewarped Image Overlay", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_dewarped_overlay.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved Figure 1")


def figure_2_disparity_before(dx, dy, grid_x_mm, grid_y_mm, save_dir, vmax=None):
    """Figure 2: Disparity field BEFORE correction (1x3)."""
    mag = np.sqrt(dx ** 2 + dy ** 2)
    if vmax is None:
        vmax = max(np.nanmax(np.abs(dx)), np.nanmax(np.abs(dy)), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(dx, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    axes[0].set_title(f"dx disparity (px)\nmean={np.nanmean(dx):.3f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dy, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    axes[1].set_title(f"dy disparity (px)\nmean={np.nanmean(dy):.3f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(mag, cmap="hot", vmin=0, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    # Quiver overlay (subsample)
    step = max(1, dx.shape[0] // 8)
    qx = grid_x_mm[::step, ::step]
    qy = grid_y_mm[::step, ::step]
    qdx = dx[::step, ::step]
    qdy = dy[::step, ::step]
    axes[2].quiver(qx, qy, qdx, -qdy, color="cyan", scale=vmax * 10, width=0.003)
    rms = float(np.sqrt(np.nanmean(dx ** 2 + dy ** 2)))
    axes[2].set_title(f"Magnitude + quiver\nRMS={rms:.3f} px")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle("Figure 2: Disparity Field BEFORE Self-Calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_disparity_before.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved Figure 2")
    return vmax


def figure_3_disparity_after(dx, dy, grid_x_mm, grid_y_mm, save_dir, vmax):
    """Figure 3: Disparity field AFTER correction (1x3), same scale as Fig 2."""
    mag = np.sqrt(dx ** 2 + dy ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(dx, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    axes[0].set_title(f"dx disparity (px)\nmean={np.nanmean(dx):.3f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dy, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    axes[1].set_title(f"dy disparity (px)\nmean={np.nanmean(dy):.3f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(mag, cmap="hot", vmin=0, vmax=vmax,
                          extent=[grid_x_mm[0, 0], grid_x_mm[0, -1],
                                  grid_y_mm[-1, 0], grid_y_mm[0, 0]])
    step = max(1, dx.shape[0] // 8)
    qx = grid_x_mm[::step, ::step]
    qy = grid_y_mm[::step, ::step]
    qdx = dx[::step, ::step]
    qdy = dy[::step, ::step]
    axes[2].quiver(qx, qy, qdx, -qdy, color="cyan", scale=vmax * 10, width=0.003)
    rms = float(np.sqrt(np.nanmean(dx ** 2 + dy ** 2)))
    axes[2].set_title(f"Magnitude + quiver\nRMS={rms:.3f} px")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle("Figure 3: Disparity Field AFTER Self-Calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_disparity_after.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved Figure 3")


def figure_4_summary(result, save_dir):
    """Figure 4: Summary comparison (2x2)."""
    dx_b = result.dx_before
    dy_b = result.dy_before
    dx_a = result.dx_after
    dy_a = result.dy_after

    mag_b = np.sqrt(dx_b ** 2 + dy_b ** 2)
    mag_a = np.sqrt(dx_a ** 2 + dy_a ** 2)
    vmax = np.nanmax(mag_b)

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Magnitude before
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(mag_b, cmap="hot", vmin=0, vmax=vmax)
    ax0.set_title(f"Magnitude BEFORE\nRMS={np.sqrt(np.nanmean(mag_b**2)):.3f} px")
    plt.colorbar(im0, ax=ax0)

    # Magnitude after
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(mag_a, cmap="hot", vmin=0, vmax=vmax)
    ax1.set_title(f"Magnitude AFTER\nRMS={np.sqrt(np.nanmean(mag_a**2)):.3f} px")
    plt.colorbar(im1, ax=ax1)

    # Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    valid_b = dx_b[np.isfinite(dx_b)]
    valid_a = dx_a[np.isfinite(dx_a)]
    bins = np.linspace(-vmax, vmax, 50)
    ax2.hist(valid_b, bins=bins, alpha=0.6, label="Before", color="red")
    ax2.hist(valid_a, bins=bins, alpha=0.6, label="After", color="green")
    ax2.set_xlabel("dx disparity (px)")
    ax2.set_ylabel("Count")
    ax2.set_title("dx Histogram")
    ax2.legend()

    # Parameter table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Parameter", "True", "Estimated", "Error"],
        ["Z offset (mm)",
         f"{TRUE_Z_OFFSET:.4f}",
         f"{result.z_offset:.4f}",
         f"{abs(result.z_offset - TRUE_Z_OFFSET):.4f}"],
        ["Tilt X (deg)",
         f"{math.degrees(TRUE_TILT_X):.4f}",
         f"{math.degrees(result.tilt_x):.4f}",
         f"{abs(math.degrees(result.tilt_x - TRUE_TILT_X)):.4f}"],
        ["Tilt Y (deg)",
         f"{math.degrees(TRUE_TILT_Y):.4f}",
         f"{math.degrees(result.tilt_y):.4f}",
         f"{abs(math.degrees(result.tilt_y - TRUE_TILT_Y)):.4f}"],
        ["RMS before (px)", f"{np.sqrt(np.nanmean(mag_b**2)):.3f}", "", ""],
        ["RMS after (px)", f"{result.final_rms_disparity:.3f}", "",
         f"{(1 - result.final_rms_disparity / np.sqrt(np.nanmean(mag_b**2))) * 100:.1f}% reduction"],
    ]
    table = ax3.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    # Bold header row
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")

    fig.suptitle("Figure 4: Summary Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_summary.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved Figure 4")


def figure_5_convergence(result, save_dir):
    """Figure 5: Convergence history (2x2)."""
    hist = result.history
    iters = [h.iteration for h in hist]
    rms_vals = [h.rms_disparity for h in hist]
    z_vals = [h.cumulative_z for h in hist]
    tx_vals = [math.degrees(h.cumulative_tilt_x) for h in hist]
    ty_vals = [math.degrees(h.cumulative_tilt_y) for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMS disparity (log Y)
    ax = axes[0, 0]
    ax.semilogy(iters, rms_vals, "bo-", linewidth=2, markersize=8)
    ax.axhline(0.1, color="r", linestyle="--", label="Threshold (0.1 px)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS disparity (px)")
    ax.set_title("RMS Disparity Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z offset
    ax = axes[0, 1]
    ax.plot(iters, z_vals, "rs-", linewidth=2, markersize=8)
    ax.axhline(TRUE_Z_OFFSET, color="k", linestyle="--", label=f"True ({TRUE_Z_OFFSET} mm)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Z offset (mm)")
    ax.set_title("Z Offset Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tilt X
    ax = axes[1, 0]
    ax.plot(iters, tx_vals, "g^-", linewidth=2, markersize=8)
    ax.axhline(math.degrees(TRUE_TILT_X), color="k", linestyle="--",
               label=f"True ({math.degrees(TRUE_TILT_X):.4f} deg)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tilt X (deg)")
    ax.set_title("Tilt X Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tilt Y
    ax = axes[1, 1]
    ax.plot(iters, ty_vals, "mD-", linewidth=2, markersize=8)
    ax.axhline(math.degrees(TRUE_TILT_Y), color="k", linestyle="--",
               label=f"True ({math.degrees(TRUE_TILT_Y):.4f} deg)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tilt Y (deg)")
    ax.set_title("Tilt Y Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 5: Convergence History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_convergence.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved Figure 5")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demonstration():
    """Main demonstration driver."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("Stereo PIV Self-Calibration — Synthetic Test")
    logger.info("=" * 60)
    logger.info(f"True misalignment: Z={TRUE_Z_OFFSET} mm, "
                f"tilt_x={math.degrees(TRUE_TILT_X):.4f} deg, "
                f"tilt_y={math.degrees(TRUE_TILT_Y):.4f} deg")

    # Create cameras
    logger.info("Creating stereo camera pair...")
    cam1, cam2 = create_stereo_cameras()

    # Generate synthetic particle images
    logger.info(f"Rendering {N_IMAGE_PAIRS} image pairs with {N_PARTICLES} particles each...")
    images_cam1 = []
    images_cam2 = []

    for i in range(N_IMAGE_PAIRS):
        particles = generate_particles(
            N_PARTICLES,
            x_range=(WORLD_BOUNDS[0], WORLD_BOUNDS[1]),
            y_range=(WORLD_BOUNDS[2], WORLD_BOUNDS[3]),
            z_offset=TRUE_Z_OFFSET,
            tilt_x=TRUE_TILT_X,
            tilt_y=TRUE_TILT_Y,
        )
        img1 = render_particles(particles, cam1)
        img2 = render_particles(particles, cam2)
        images_cam1.append(img1)
        images_cam2.append(img2)

    # Run self-calibration (use Python fallback so no C library is needed)
    logger.info("Running self-calibration...")
    result = run_self_calibration(
        cam1, cam2,
        images_cam1, images_cam2,
        output_size=OUTPUT_SIZE,
        world_bounds=WORLD_BOUNDS,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        max_iterations=10,
        convergence_threshold=0.1,
        quality_threshold=0.1,
        use_c_library=True,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Converged: {result.converged}")
    logger.info(f"Iterations: {result.n_iterations}")
    logger.info(f"Final RMS disparity: {result.final_rms_disparity:.4f} px")
    logger.info(f"Z offset:  True={TRUE_Z_OFFSET:.4f}, "
                f"Est={result.z_offset:.4f}, "
                f"Err={abs(result.z_offset - TRUE_Z_OFFSET):.4f} mm")
    logger.info(f"Tilt X:    True={math.degrees(TRUE_TILT_X):.4f}, "
                f"Est={math.degrees(result.tilt_x):.4f}, "
                f"Err={abs(math.degrees(result.tilt_x - TRUE_TILT_X)):.4f} deg")
    logger.info(f"Tilt Y:    True={math.degrees(TRUE_TILT_Y):.4f}, "
                f"Est={math.degrees(result.tilt_y):.4f}, "
                f"Err={abs(math.degrees(result.tilt_y - TRUE_TILT_Y)):.4f} deg")

    # Generate figures
    logger.info("Generating diagnostic figures...")

    figure_1_dewarped_overlay(
        cam1, cam2, images_cam1, images_cam2,
        OUTPUT_SIZE, WORLD_BOUNDS,
        z_before=0, tx_before=0, ty_before=0,
        z_after=result.z_offset, tx_after=result.tilt_x, ty_after=result.tilt_y,
        save_dir=FIGURE_DIR,
    )

    vmax = figure_2_disparity_before(
        result.dx_before, result.dy_before,
        result.grid_x_mm, result.grid_y_mm,
        FIGURE_DIR,
    )

    figure_3_disparity_after(
        result.dx_after, result.dy_after,
        result.grid_x_mm, result.grid_y_mm,
        FIGURE_DIR, vmax,
    )

    figure_4_summary(result, FIGURE_DIR)
    figure_5_convergence(result, FIGURE_DIR)

    logger.info(f"All figures saved to {FIGURE_DIR}/")
    logger.info("Done.")

    return result


if __name__ == "__main__":
    run_demonstration()
