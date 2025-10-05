import numpy as np
import os
from numba import njit, prange
import time

# ---------- SETTINGS ----------
output_dir = "piv_output_shear_numba"
os.makedirs(output_dir, exist_ok=True)

# imaging
pixel_density = 1024 / 300  # px per mm ≈ 3.413333
pixel_size = 300 / 1024     # mm per px ≈ 0.29296875

# displacements in pixels (means now 0)
pix_u_top, pix_u_bot = 0, 0
pix_v, pix_w = 0, 0

# convert to mm
disp_u_top = pix_u_top * pixel_size
disp_u_bot = pix_u_bot * pixel_size
disp_v = pix_v * pixel_size
disp_w = pix_w * pixel_size

# time spacing
dt = 0.02

# mean velocities (now 0)
mean_u_top = disp_u_top / dt   # 0 mm/s
mean_u_bot = disp_u_bot / dt   # 0 mm/s
mean_v = disp_v / dt           # 0 mm/s
mean_w = disp_w / dt           # 0 mm/s

# turbulence intensities (RMS) in mm/s for desired pixels/timestep
RMS_u = 2.0 * pixel_size / dt  # ≈29.296875 mm/s for 2 pixels/timestep
RMS_v = 1.0 * pixel_size / dt  # ≈14.6484375 mm/s for 1 pixel/timestep
RMS_w = 3.0 * pixel_size / dt  # ≈43.9453125 mm/s for 3 pixels/timestep

# Calculate N_particles
image_pixels = 1024
window_size = 16
particles_per_pixel = 8

num_windows = (image_pixels // window_size) ** 2  # 64 x 64 = 4096 windows
N_particles = num_windows * window_size * window_size * particles_per_pixel // (window_size * window_size)
# Simplifies to:
N_particles = num_windows * particles_per_pixel  # 4096 * 8 = 32,768

n_pairs = 1000  # full dataset
rng = np.random.default_rng(12345)

# ---------- JIT FUNCTION ----------
@njit(parallel=True)
def advance_particles(x, y, z, mean_u_top, mean_u_bot, mean_v, mean_w, dt,
                      RMS_u, RMS_v, RMS_w):
    N = x.shape[0]
    xB = np.empty_like(x)
    yB = np.empty_like(y)
    zB = np.empty_like(z)

    for i in prange(N):
        # pick mean u depending on y (now both 0)
        mu = mean_u_top if y[i] >= 50.0 else mean_u_bot

        # Gaussian fluctuations
        u_inst = mu + np.random.normal(0.0, RMS_u)
        v_inst = mean_v + np.random.normal(0.0, RMS_v)
        w_inst = mean_w + np.random.normal(0.0, RMS_w)

        # advect
        xB[i] = x[i] + u_inst * dt
        yB[i] = y[i] + v_inst * dt
        zB[i] = z[i] + w_inst * dt

    return xB, yB, zB

# ---------- MAIN LOOP ----------
print("pixel_size (mm/px):", pixel_size)
print("mean velocities (top half):", mean_u_top, mean_v, mean_w)
print("mean velocities (bottom half):", mean_u_bot, mean_v, mean_w)

start_time = time.time()

for i in range(1, n_pairs + 1):
    # positions at t (frame A)
    x = rng.uniform(0.0, 300.0, size=N_particles)
    y = rng.uniform(-150.0, 150.0, size=N_particles)
    z = rng.uniform(-5.0, 5.0, size=N_particles)

    # advance to frame B with JIT function
    xB, yB, zB = advance_particles(
        x, y, z,
        mean_u_top, mean_u_bot,
        mean_v, mean_w,
        dt, RMS_u, RMS_v, RMS_w
    )

    # save
    fileA = os.path.join(output_dir, f"B{i:05d}_A.data")
    fileB = os.path.join(output_dir, f"B{i:05d}_B.data")
    np.savetxt(fileA, np.column_stack([x, y, z]), fmt="%.6f")
    np.savetxt(fileB, np.column_stack([xB, yB, zB]), fmt="%.6f")

end_time = time.time()
print(f"Elapsed time for {n_pairs} pairs: {end_time - start_time:.2f} seconds")

print(f"Saved {n_pairs} pairs in {output_dir}")
