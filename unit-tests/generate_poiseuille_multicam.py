#!/usr/bin/env python3
"""
Generate synthetic PIV image pairs for a 2-camera Poiseuille flow test.

Same physical velocity field as generate_poiseuille_cam_model.py:
  ux(y_phys) = u_max * (1 - y_norm^2)   m/s, rightward
  uy          = 0                         m/s

Two cameras with the same intrinsics but different extrinsics (tvec)
see overlapping portions of the physical domain:
  Cam1: sees top portion (positive y_phys)
  Cam2: sees bottom portion, offset 20px up + 10px left

Camera model:
  fx = fy = 5000 px,  cx = 550, cy = 350
  Image size 700 x 1100 (H x W), padded from ~600x1000 useful region
  tz = 0.333 m  (~15 px/mm)
  rvec = per-camera (small tilts ~2-3 deg)
  dist = [k1, 0, 0, 0, 0]  barrel distortion (k1 = -0.15)

Output:
  unit-tests/poiseuille_multicam/
    Cam1/B00001_A.tif, B00001_B.tif
    Cam2/B00001_A.tif, B00001_B.tif
    ground_truth.npz

Usage:
    python unit-tests/generate_poiseuille_multicam.py
"""

from pathlib import Path

import cv2
import numpy as np
import tifffile
from synthetic_piv import render_particles

# ---------------------------------------------------------------------------
# Physical velocity field
# ---------------------------------------------------------------------------
U_MAX = 0.833  # m/s peak centreline velocity
DT = 1e-3  # seconds between frames

# ---------------------------------------------------------------------------
# Image & particle parameters
# ---------------------------------------------------------------------------
IMAGE_H, IMAGE_W = 700, 1100
IMAGE_SHAPE = (IMAGE_H, IMAGE_W)
NUM_PARTICLES = 50_000
PARTICLE_DIAMETER = 3.0  # px (FWHM)
SIGMA = PARTICLE_DIAMETER / 2.355
NUM_PAIRS = 1
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent / "poiseuille_multicam"

# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
FX = FY = 5000.0
CX, CY = 550.0, 350.0
TZ = FX / (15.0 * 1000)  # 0.333 m  → ~15 px/mm
K1 = -0.15

# Per-camera rotation vectors (~5 deg tilts, stress-testing calibration)
RVEC_CAM1 = np.array([0.070, -0.047, 0.023], dtype=np.float64)  # ~5 deg total
RVEC_CAM2 = np.array(
    [-0.043, 0.065, -0.033], dtype=np.float64
)  # ~5 deg total, different axes

# Camera placement: tvec controls which part of the physical domain is visible
# Camera1: image centre maps to physical y = +13.3mm (top of domain)
TVEC_CAM1 = np.array([0.0, -0.01333, TZ], dtype=np.float64)

# Camera2: image centre maps to physical y ≈ -13.3mm (bottom of domain)
# Plus offset: 20px up → ty decreases by 20*TZ/5000, 10px left → tx increases by 10*TZ/5000
TVEC_CAM2 = np.array([10 * TZ / FX, 0.01333 - 20 * TZ / FY, TZ], dtype=np.float64)


def _normalize_uint16(img):
    mx = img.max()
    if mx > 0:
        return (img / mx * 65535).astype(np.uint16)
    return img.astype(np.uint16)


def _undistort_points_independent(pts_px, camera_matrix, dist_coeffs):
    """Independent undistortion — NO cv2.undistortPoints().

    Iterative fixed-point method to invert the radial distortion model.
    Only uses k1 (matching our synthetic data's single-coefficient model).
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1 = dist_coeffs[0]

    pts = pts_px.reshape(-1, 2).astype(np.float64)
    result = np.zeros((len(pts), 2), dtype=np.float64)

    for i, (u, v) in enumerate(pts):
        xd = (u - cx) / fx
        yd = (v - cy) / fy

        # Iterative undistortion (fixed-point iteration)
        x, y = xd, yd
        for _ in range(20):  # converges in ~5 iterations for |k1|<0.3
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2
            x = xd / radial
            y = yd / radial

        result[i] = [x, y]
    return result


def _pixel_to_world(pts_px, camera_matrix, dist_coeffs, rvec, tvec):
    """Independent ground-truth projection — NO cv2.undistortPoints()."""
    normalised = _undistort_points_independent(pts_px, camera_matrix, dist_coeffs)

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t = tvec.flatten()
    t_world = R_inv @ t

    world_pts = np.zeros((len(normalised), 3), dtype=np.float64)
    for i, (xn, yn) in enumerate(normalised):
        ray_cam = np.array([xn, yn, 1.0])
        ray_world = R_inv @ ray_cam
        s = t_world[2] / ray_world[2]
        world_pts[i] = s * ray_world - t_world

    return world_pts


def _generate_camera_images(
    camera_matrix, dist_coeffs, rvec, tvec, H_phys, rng, output_dir
):
    """Generate image pair for one camera. Returns pixel displacement stats."""
    H, W = IMAGE_SHAPE
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx in range(1, NUM_PAIRS + 1):
        x_pos = rng.uniform(0, W, NUM_PARTICLES).astype(np.float64)
        y_pos = rng.uniform(0, H, NUM_PARTICLES).astype(np.float64)
        intensities = rng.uniform(200, 255, NUM_PARTICLES)

        pts_px = np.column_stack([x_pos, y_pos])
        world_pts = _pixel_to_world(pts_px, camera_matrix, dist_coeffs, rvec, tvec)

        # Poiseuille velocity: y_norm = y_world / (H_phys/2)
        y_norm = world_pts[:, 1] / (H_phys / 2.0)
        ux_phys = U_MAX * (1.0 - y_norm**2)
        uy_phys = np.zeros_like(ux_phys)

        dx_world_m = ux_phys * DT
        dy_world_m = -uy_phys * DT

        displaced_world = world_pts.copy()
        displaced_world[:, 0] += dx_world_m
        displaced_world[:, 1] += dy_world_m

        displaced_px, _ = cv2.projectPoints(
            displaced_world,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        displaced_px = displaced_px.reshape(-1, 2)

        dx_px = displaced_px[:, 0] - x_pos
        dy_px = displaced_px[:, 1] - y_pos

        img_a = render_particles(IMAGE_SHAPE, x_pos, y_pos, intensities, SIGMA)
        img_b = render_particles(
            IMAGE_SHAPE, x_pos + dx_px, y_pos + dy_px, intensities, SIGMA
        )

        tifffile.imwrite(
            output_dir / f"B{pair_idx:05d}_A.tif", _normalize_uint16(img_a)
        )
        tifffile.imwrite(
            output_dir / f"B{pair_idx:05d}_B.tif", _normalize_uint16(img_b)
        )

    return {
        "dx_range": (float(np.min(dx_px)), float(np.max(dx_px))),
        "dy_range": (float(np.min(dy_px)), float(np.max(dy_px))),
        "ux_range": (float(np.min(ux_phys)), float(np.max(ux_phys))),
    }


def main():
    camera_matrix = np.array(
        [
            [FX, 0, CX],
            [0, FY, CY],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array([K1, 0, 0, 0, 0], dtype=np.float64)

    px_per_mm = FX / (TZ * 1000)
    # Physical FOV based on larger image dimension (width)
    fov_w_mm = IMAGE_W / px_per_mm
    fov_h_mm = IMAGE_H / px_per_mm
    # Physical channel height spans total visible y-range across both cameras
    # Use a generous H_phys that covers both cameras' FOV
    H_phys = 80.0 / 1000.0  # 80mm in metres — covers both cameras' combined y-extent

    print(f"Camera: fx={FX}, tz={TZ:.4f} m, {px_per_mm:.1f} px/mm")
    print(f"FOV per camera: {fov_w_mm:.1f} x {fov_h_mm:.1f} mm")
    print(f"Distortion: k1={K1}")
    print(f"Physical channel height: {H_phys*1000:.1f} mm")
    print(f"Poiseuille u_max={U_MAX} m/s, dt={DT} s")
    print(f"Cam1 rvec: {RVEC_CAM1}")
    print(f"Cam2 rvec: {RVEC_CAM2}")

    # Compute overlap feature point: physical origin [0,0,0] projected to both cameras
    phys_origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    overlap_on_cam1, _ = cv2.projectPoints(
        phys_origin, RVEC_CAM1, TVEC_CAM1, camera_matrix, dist_coeffs
    )
    overlap_on_cam2, _ = cv2.projectPoints(
        phys_origin, RVEC_CAM2, TVEC_CAM2, camera_matrix, dist_coeffs
    )
    overlap_on_cam1 = overlap_on_cam1.reshape(2)
    overlap_on_cam2 = overlap_on_cam2.reshape(2)

    print("\nOverlap feature point (physical origin):")
    print(f"  Cam1 raw pixel: ({overlap_on_cam1[0]:.1f}, {overlap_on_cam1[1]:.1f})")
    print(f"  Cam2 raw pixel: ({overlap_on_cam2[0]:.1f}, {overlap_on_cam2[1]:.1f})")

    # Generate images for each camera
    rng = np.random.default_rng(SEED)

    print("\nGenerating Cam1 images...")
    stats1 = _generate_camera_images(
        camera_matrix,
        dist_coeffs,
        RVEC_CAM1,
        TVEC_CAM1,
        H_phys,
        rng,
        OUTPUT_DIR / "Cam1",
    )
    print(f"  dx: [{stats1['dx_range'][0]:.2f}, {stats1['dx_range'][1]:.2f}] px")
    print(f"  ux: [{stats1['ux_range'][0]:.3f}, {stats1['ux_range'][1]:.3f}] m/s")

    print("\nGenerating Cam2 images...")
    stats2 = _generate_camera_images(
        camera_matrix,
        dist_coeffs,
        RVEC_CAM2,
        TVEC_CAM2,
        H_phys,
        rng,
        OUTPUT_DIR / "Cam2",
    )
    print(f"  dx: [{stats2['dx_range'][0]:.2f}, {stats2['dx_range'][1]:.2f}] px")
    print(f"  ux: [{stats2['ux_range'][0]:.3f}, {stats2['ux_range'][1]:.3f}] m/s")

    # Save ground truth
    gt_path = OUTPUT_DIR / "ground_truth.npz"
    np.savez(
        str(gt_path),
        u_max=U_MAX,
        dt=DT,
        H_phys=H_phys,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec_cam1=RVEC_CAM1,
        rvec_cam2=RVEC_CAM2,
        tvec_cam1=TVEC_CAM1,
        tvec_cam2=TVEC_CAM2,
        overlap_on_cam1=overlap_on_cam1,
        overlap_on_cam2=overlap_on_cam2,
        image_width=IMAGE_W,
        image_height=IMAGE_H,
        num_pairs=NUM_PAIRS,
    )

    print(f"\nGround truth saved to {gt_path}")
    print(f"Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
