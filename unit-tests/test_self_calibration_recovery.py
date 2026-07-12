#!/usr/bin/env python3
"""
test_self_calibration_recovery.py

Verifies that stereo PIV self-calibration recovers known ground-truth
laser-sheet misalignment from synthetic stereo particle images.

Usage:
    pytest unit-tests/test_self_calibration_recovery.py -v
    pytest unit-tests/test_self_calibration_recovery.py -v --make-figures
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    run_self_calibration,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUE_Z_OFFSET = 0.3  # mm
TRUE_TILT_X = 0.002  # rad
TRUE_TILT_Y = -0.001  # rad
STEREO_ANGLE_DEG = 30.0
N_IMAGE_PAIRS = 20
N_PARTICLES = 2000
WORLD_BOUNDS = (-40.0, 40.0, -40.0, 40.0)  # mm
WINDOW_SIZE = 64
OVERLAP = 50.0
FOCAL_LENGTH_PX = 1000.0
IMAGE_SIZE = (1024, 1024)  # (width, height)
BASELINE_MM = 200.0
PARTICLE_SIGMA = 2.0
PARTICLE_INTENSITY = 200
CONVERGENCE_THRESHOLD = 0.1  # px


# ---------------------------------------------------------------------------
# Synthetic camera + particle generation
# ---------------------------------------------------------------------------


def create_stereo_cameras(
    stereo_angle_deg=STEREO_ANGLE_DEG,
    focal_length_px=FOCAL_LENGTH_PX,
    image_size=IMAGE_SIZE,
    baseline_mm=BASELINE_MM,
):
    """Create a symmetric stereo camera pair."""
    w, h = image_size
    theta = math.radians(stereo_angle_deg / 2.0)

    K = np.array(
        [
            [focal_length_px, 0.0, w / 2.0],
            [0.0, focal_length_px, h / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.zeros(5)

    z_cam = baseline_mm / (2.0 * math.tan(theta))
    cam1_pos = np.array([baseline_mm / 2.0, 0.0, z_cam])
    cam2_pos = np.array([-baseline_mm / 2.0, 0.0, z_cam])

    def look_at_rotation(cam_pos, target=np.array([0, 0, 0])):
        z_axis = target - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        up = np.array([0.0, -1.0, 0.0])
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

    cam1 = PinholeCamera(
        K=K.copy(), dist=dist.copy(), R=R1, t=t1, image_size=image_size
    )
    cam2 = PinholeCamera(
        K=K.copy(), dist=dist.copy(), R=R2, t=t2, image_size=image_size
    )
    return cam1, cam2


def generate_particles(n_particles, x_range, y_range, z_offset, tilt_x, tilt_y):
    """Generate random particles on a tilted laser sheet."""
    x = np.random.uniform(x_range[0], x_range[1], n_particles)
    y = np.random.uniform(y_range[0], y_range[1], n_particles)
    z = z_offset + x * np.tan(tilt_y) + y * np.tan(tilt_x)
    return np.column_stack([x, y, z])


def render_particles(
    particles, camera, particle_sigma=PARTICLE_SIGMA, intensity=PARTICLE_INTENSITY
):
    """Render particles as Gaussian spots on an image."""
    w, h = camera.image_size
    image = np.zeros((h, w), dtype=np.float32)
    pts2d = camera.project(particles)

    half_size = int(math.ceil(4 * particle_sigma))
    kernel_size = 2 * half_size + 1
    kx = np.arange(kernel_size) - half_size
    ky = np.arange(kernel_size) - half_size
    kxx, kyy = np.meshgrid(kx, ky)

    for pt in pts2d:
        px, py = pt
        ix, iy = int(round(px)), int(round(py))
        if ix < half_size or ix >= w - half_size:
            continue
        if iy < half_size or iy >= h - half_size:
            continue

        dx = px - ix
        dy = py - iy
        kernel = intensity * np.exp(
            -((kxx - dx) ** 2 + (kyy - dy) ** 2) / (2 * particle_sigma**2)
        )
        image[
            iy - half_size : iy + half_size + 1, ix - half_size : ix + half_size + 1
        ] += kernel

    noise = np.random.normal(0, 5, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 255)
    return image.astype(np.uint8)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped for expensive computation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stereo_cameras():
    """Create synthetic stereo camera pair."""
    return create_stereo_cameras()


@pytest.fixture(scope="module")
def synthetic_images(stereo_cameras):
    """Generate synthetic particle images with known misalignment."""
    np.random.seed(42)
    cam1, cam2 = stereo_cameras
    images_cam1 = []
    images_cam2 = []

    for _ in range(N_IMAGE_PAIRS):
        particles = generate_particles(
            N_PARTICLES,
            x_range=(WORLD_BOUNDS[0], WORLD_BOUNDS[1]),
            y_range=(WORLD_BOUNDS[2], WORLD_BOUNDS[3]),
            z_offset=TRUE_Z_OFFSET,
            tilt_x=TRUE_TILT_X,
            tilt_y=TRUE_TILT_Y,
        )
        images_cam1.append(render_particles(particles, cam1))
        images_cam2.append(render_particles(particles, cam2))

    return images_cam1, images_cam2


@pytest.fixture(scope="module")
def self_calibration_result(stereo_cameras, synthetic_images):
    """Run self-calibration once, shared across all tests."""
    cam1, cam2 = stereo_cameras
    images_cam1, images_cam2 = synthetic_images

    return run_self_calibration(
        cam1,
        cam2,
        images_cam1,
        images_cam2,
        world_bounds=WORLD_BOUNDS,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        max_iterations=10,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        quality_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_self_calibration_converges(self_calibration_result):
    """Self-calibration should converge."""
    assert self_calibration_result.converged is True


def test_z_offset_recovery(self_calibration_result):
    """|recovered Z - true Z| < 0.05 mm."""
    err = abs(self_calibration_result.z_offset - TRUE_Z_OFFSET)
    assert err < 0.05, f"Z offset error {err:.4f} mm exceeds 0.05 mm"


def test_tilt_x_recovery(self_calibration_result):
    """|recovered tilt_x - true tilt_x| < 0.001 rad."""
    err = abs(self_calibration_result.tilt_x - TRUE_TILT_X)
    assert err < 0.001, f"Tilt X error {err:.6f} rad exceeds 0.001 rad"


def test_tilt_y_recovery(self_calibration_result):
    """|recovered tilt_y - true tilt_y| < 0.001 rad."""
    err = abs(self_calibration_result.tilt_y - TRUE_TILT_Y)
    assert err < 0.001, f"Tilt Y error {err:.6f} rad exceeds 0.001 rad"


def test_rms_disparity_reduced(self_calibration_result):
    """Final RMS disparity below convergence threshold."""
    assert self_calibration_result.final_rms_disparity < CONVERGENCE_THRESHOLD, (
        f"Final RMS {self_calibration_result.final_rms_disparity:.4f} px "
        f"exceeds threshold {CONVERGENCE_THRESHOLD} px"
    )


def test_iterations_reasonable(self_calibration_result):
    """Should converge in < 10 iterations."""
    assert (
        self_calibration_result.n_iterations < 10
    ), f"Took {self_calibration_result.n_iterations} iterations (expected < 10)"


# ---------------------------------------------------------------------------
# Diagnostic figures (gated by --make-figures)
# ---------------------------------------------------------------------------


class TestDiagnosticFigures:
    """Generates diagnostic figures when --make-figures is passed."""

    def test_make_figures(self, self_calibration_result, make_figures, output_dir):
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic figures")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self_calibration_result
        hist = result.history

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RMS disparity convergence
        iters = [h.iteration for h in hist]
        rms_vals = [h.rms_disparity for h in hist]
        ax = axes[0, 0]
        ax.semilogy(iters, rms_vals, "bo-", linewidth=2, markersize=8)
        ax.axhline(
            CONVERGENCE_THRESHOLD,
            color="r",
            linestyle="--",
            label=f"Threshold ({CONVERGENCE_THRESHOLD} px)",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS disparity (px)")
        ax.set_title("RMS Disparity Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Z offset recovery
        z_vals = [h.cumulative_z for h in hist]
        ax = axes[0, 1]
        ax.plot(iters, z_vals, "rs-", linewidth=2, markersize=8)
        ax.axhline(
            TRUE_Z_OFFSET, color="k", linestyle="--", label=f"True ({TRUE_Z_OFFSET} mm)"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Z offset (mm)")
        ax.set_title("Z Offset Recovery")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tilt X recovery
        tx_vals = [math.degrees(h.cumulative_tilt_x) for h in hist]
        ax = axes[1, 0]
        ax.plot(iters, tx_vals, "g^-", linewidth=2, markersize=8)
        ax.axhline(
            math.degrees(TRUE_TILT_X),
            color="k",
            linestyle="--",
            label=f"True ({math.degrees(TRUE_TILT_X):.4f} deg)",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tilt X (deg)")
        ax.set_title("Tilt X Recovery")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary table
        ax = axes[1, 1]
        ax.axis("off")
        rows = [
            ["Parameter", "True", "Recovered", "Error"],
            [
                "Z (mm)",
                f"{TRUE_Z_OFFSET:.4f}",
                f"{result.z_offset:.4f}",
                f"{abs(result.z_offset - TRUE_Z_OFFSET):.4f}",
            ],
            [
                "Tilt X (deg)",
                f"{math.degrees(TRUE_TILT_X):.4f}",
                f"{math.degrees(result.tilt_x):.4f}",
                f"{abs(math.degrees(result.tilt_x - TRUE_TILT_X)):.4f}",
            ],
            [
                "Tilt Y (deg)",
                f"{math.degrees(TRUE_TILT_Y):.4f}",
                f"{math.degrees(result.tilt_y):.4f}",
                f"{abs(math.degrees(result.tilt_y - TRUE_TILT_Y)):.4f}",
            ],
            ["RMS (px)", "", f"{result.final_rms_disparity:.4f}", ""],
            ["Iterations", "", f"{result.n_iterations}", ""],
        ]
        table = ax.table(cellText=rows, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        for j in range(4):
            table[0, j].set_text_props(fontweight="bold")

        fig.suptitle("Self-Calibration Recovery", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(output_dir / "self_calibration_recovery.png"), dpi=150)
        plt.close(fig)
