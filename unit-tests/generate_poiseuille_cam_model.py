#!/usr/bin/env python3
"""
Generate synthetic PIV image pairs with a Poiseuille (parabolic) velocity
profile projected through a pinhole camera with barrel distortion.

Physical velocity field (in the Z=0 world plane):
  ux(y_phys) = u_max * (1 - (2*y_phys/H_phys - 1)^2)   m/s, rightward
  uy          = 0                                          m/s

Camera model:
  fx = fy = 5000 px
  cx, cy  = 500, 500  (image centre)
  tz      = 0.333 m   (~15 px/mm)
  rvec    = [0, 0, 0]  (frontal)
  dist    = [k1, 0, 0, 0, 0]   barrel distortion (k1 < 0)

The physical FOV is ~66.7 mm × 66.7 mm.  Poiseuille is defined over the
full physical height, so the channel walls are at y_phys = 0 and y_phys = H_phys.

Ground-truth velocity is stored analytically via (u_max, H_phys, dt) so the
test can compute the exact expected velocity at any calibrated coordinate.

Output:
  unit-tests/poiseuille_cam_model/
    Cam1/B00001_A.tif, B00001_B.tif
    ground_truth.npz

Usage:
    python unit-tests/generate_poiseuille_cam_model.py
"""

from pathlib import Path

import cv2
import numpy as np
import tifffile

from synthetic_piv import render_particles

# ---------------------------------------------------------------------------
# Physical velocity field
# ---------------------------------------------------------------------------
U_MAX = 0.833     # m/s peak centreline velocity (~12.5 px at 15 px/mm, dt=1ms)
DT = 1e-3         # seconds between frames

# ---------------------------------------------------------------------------
# Image & particle parameters
# ---------------------------------------------------------------------------
IMAGE_SHAPE = (1000, 1000)   # (H, W), 1 MP
NUM_PARTICLES = 40_000
PARTICLE_DIAMETER = 3.0      # px (FWHM)
SIGMA = PARTICLE_DIAMETER / 2.355
NUM_PAIRS = 1
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent / "poiseuille_cam_model" / "Cam1"

# ---------------------------------------------------------------------------
# Camera model: ~15 px/mm with mild barrel distortion
# ---------------------------------------------------------------------------
FX = FY = 5000.0
CX, CY = 500.0, 500.0
TZ = FX / (15.0 * 1000)   # 0.333 m  → ~15 px/mm
K1 = -0.15                 # barrel distortion (negative = barrel)


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


def main():
    camera_matrix = np.array([
        [FX,  0, CX],
        [ 0, FY, CY],
        [ 0,  0,  1],
    ], dtype=np.float64)
    dist_coeffs = np.array([K1, 0, 0, 0, 0], dtype=np.float64)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.array([0.0, 0.0, TZ], dtype=np.float64)

    H, W = IMAGE_SHAPE
    px_per_mm = FX / (TZ * 1000)
    fov_mm = W / px_per_mm

    # Physical FOV: the world region visible in the image
    # For frontal camera centred on origin: spans [-fov/2, +fov/2] in both axes
    H_phys = fov_mm / 1000.0   # physical channel height in metres
    print(f"Camera: fx={FX}, tz={TZ:.4f} m, {px_per_mm:.1f} px/mm, FOV={fov_mm:.1f} mm")
    print(f"Distortion: k1={K1}")
    print(f"Physical channel height: {H_phys*1000:.1f} mm")
    print(f"Poiseuille u_max={U_MAX} m/s, dt={DT} s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for pair_idx in range(1, NUM_PAIRS + 1):
        # Random particle positions in raw pixel coords
        x_pos = rng.uniform(0, W, NUM_PARTICLES).astype(np.float64)
        y_pos = rng.uniform(0, H, NUM_PARTICLES).astype(np.float64)
        intensities = rng.uniform(200, 255, NUM_PARTICLES)

        # Project pixels to world coordinates (metres)
        pts_px = np.column_stack([x_pos, y_pos])
        world_pts = _pixel_to_world(pts_px, camera_matrix, dist_coeffs, rvec, tvec)

        # Poiseuille velocity in physical coords
        # World y range: [-H_phys/2, +H_phys/2]  (camera centred on origin)
        # Map to normalised channel coord: y_norm = y_world / (H_phys/2)
        # u_x(y_norm) = u_max * (1 - y_norm^2)
        y_norm = world_pts[:, 1] / (H_phys / 2.0)
        ux_phys = U_MAX * (1.0 - y_norm ** 2)  # m/s, rightward
        uy_phys = np.zeros_like(ux_phys)        # m/s

        # Physical displacement over dt (metres)
        dx_world_m = ux_phys * DT          # rightward in world x
        dy_world_m = -uy_phys * DT         # world y-down convention (negated)

        # Displace in world and project back to pixels
        displaced_world = world_pts.copy()
        displaced_world[:, 0] += dx_world_m
        displaced_world[:, 1] += dy_world_m

        displaced_px, _ = cv2.projectPoints(
            displaced_world, rvec, tvec, camera_matrix, dist_coeffs,
        )
        displaced_px = displaced_px.reshape(-1, 2)

        dx_px = displaced_px[:, 0] - x_pos
        dy_px = displaced_px[:, 1] - y_pos

        print(f"  Pair {pair_idx}: pixel displacement range: "
              f"dx=[{np.min(dx_px):.3f}, {np.max(dx_px):.3f}], "
              f"dy=[{np.min(dy_px):.3f}, {np.max(dy_px):.3f}] px")
        print(f"  Physical velocity range: "
              f"ux=[{np.min(ux_phys):.3f}, {np.max(ux_phys):.3f}] m/s")

        # Render frames
        img_a = render_particles(IMAGE_SHAPE, x_pos, y_pos, intensities, SIGMA)
        img_b = render_particles(IMAGE_SHAPE, x_pos + dx_px, y_pos + dy_px,
                                 intensities, SIGMA)

        tifffile.imwrite(OUTPUT_DIR / f"B{pair_idx:05d}_A.tif", _normalize_uint16(img_a))
        tifffile.imwrite(OUTPUT_DIR / f"B{pair_idx:05d}_B.tif", _normalize_uint16(img_b))

    # Save ground truth: analytical parameters + camera model
    gt_path = OUTPUT_DIR.parent / "ground_truth.npz"
    np.savez(
        str(gt_path),
        u_max=U_MAX,
        dt=DT,
        H_phys=H_phys,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        image_width=W,
        image_height=H,
        num_pairs=NUM_PAIRS,
    )

    print(f"\nGround truth saved to {gt_path}")
    print(f"Images saved to {OUTPUT_DIR}")
    print(f"\nTo use with PIVTOOLs:")
    print(f"  source_paths: ['{OUTPUT_DIR.parent}']")
    print(f"  image_format: ['B%05d_A.tif', 'B%05d_B.tif']")
    print(f"  num_images: {NUM_PAIRS}")


if __name__ == "__main__":
    main()
