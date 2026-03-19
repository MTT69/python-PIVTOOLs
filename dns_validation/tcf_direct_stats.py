#!/usr/bin/env python3
"""
tcf_direct_stats.py — Direct binning statistics for JHTDB channel flow data.

Reimplements the tcf_mean.m direct binning approach in Python:
  - Cosine-stretched (Chebyshev) y-bins for wall-normal resolution
  - Single-pass accumulation of velocity moments
  - Self-consistent variance: Var = <uu> - <u><u>
  - No spatial smoothing bias from convolution
  - 2D uniform grid for spatial mean/stress fields

Reads particle position pairs (B#####_A.data, B#####_B.data) and computes
turbulence statistics (mean velocity, Reynolds stress tensor) in wall units.
"""

import numpy as np
from pathlib import Path
import time

from scipy.io import loadmat, savemat

# ============================================================================
# CONFIGURATION
# ============================================================================

data_folder = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel"
    r"\particle_positions\normalized"
)
params_file = data_folder.parent / "download_parameters.mat"
output_folder = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel"
    r"\ensemble_statistics_direct_4000"
)

n_y = 90  # Number of y-bins (was 64; ~2x near-wall resolution since dy_wall ~ 1/n_y²)

# 2D grid bin size (mm) — uniform grid for spatial fields
dx_2d = 1.0
dy_2d = 1.0

# DNS Re_tau=1000 fallback constants (used if params file missing fields)
u_tau_nondim = 0.0499       # Friction velocity (non-dimensional)
nu_nondim = 5e-5            # Kinematic viscosity (non-dimensional)

# ============================================================================
# STEP 1: Load parameters from download_parameters.mat
# ============================================================================

print("=" * 60)
print("  Direct Binning Statistics — JHTDB Channel Flow")
print("=" * 60)
print()

print(f"Loading parameters from:\n  {params_file}")
mat_data = loadmat(str(params_file))
p = mat_data["params"].flat[0]

n_frames = int(p["n_frames"].flat[0])
dt = float(p["dt"].flat[0])
h_mm = float(p["h_mm"].flat[0])
Nx = int(p["Nx"].flat[0])
Ny = int(p["Ny"].flat[0])
mm_per_pixel = float(p["mm_per_pixel"].flat[0])
velocity_conv = float(p["velocity_conv"].flat[0])
length_conv = float(p["length_conv"].flat[0])

# Compute wall units
u_tau = u_tau_nondim * velocity_conv                # mm/s
nu = nu_nondim * length_conv * velocity_conv        # mm^2/s
delta_nu = nu / u_tau                               # viscous length scale (mm)
Re_tau = h_mm / delta_nu                            # friction Reynolds number

print(f"\n--- Physical Parameters ---")
print(f"  Channel half-height h     = {h_mm:.1f} mm")
print(f"  Time step dt              = {dt:.6e}")
print(f"  Friction velocity u_tau   = {u_tau:.4f} mm/s")
print(f"  Kinematic viscosity nu    = {nu:.2e} mm^2/s")
print(f"  Viscous length delta_nu   = {delta_nu:.4f} mm")
print(f"  Friction Reynolds Re_tau  = {Re_tau:.0f}")
print()

# ============================================================================
# STEP 2: Scan for frame pairs
# ============================================================================

print("Scanning for frame pairs...")
a_files = sorted(data_folder.glob("B*_A.data"))

frame_pairs = []
for af in a_files:
    stem = af.stem                          # e.g. "B00001_A"
    frame_str = stem.split("_")[0][1:]      # e.g. "00001"
    bf = af.parent / f"B{frame_str}_B.data"
    if bf.exists():
        frame_pairs.append((af, bf, int(frame_str)))

frame_pairs.sort(key=lambda x: x[2])
n_available = len(frame_pairs)

if n_available == 0:
    raise RuntimeError(f"No frame pairs found in {data_folder}")

print(f"  Found {n_available} complete frame pairs.")
print(f"  Frame range: {frame_pairs[0][2]} to {frame_pairs[-1][2]}")
print()

# ============================================================================
# STEP 3: Create cosine-stretched (Chebyshev) y-bins
# ============================================================================
# Based on tcf_mean.m but restricted to the bottom channel [-h, 0].
# phi from pi to pi/2 maps cos(phi) from -1 to 0, giving y in [-h, 0].
# All n_y bins cover the data range — no wasted bins in the empty top channel.

phi = np.linspace(np.pi, np.pi / 2, n_y + 1)
y_edges = h_mm * np.cos(phi)                           # -h … 0
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

print(f"Chebyshev y-bins: {n_y} bins spanning [{y_edges[0]:.1f}, {y_edges[-1]:.1f}] mm")
print(f"  First bin width (near wall):  {y_edges[1] - y_edges[0]:.3f} mm")
print(f"  Mid bin width (near centre):  {y_edges[n_y // 2 + 1] - y_edges[n_y // 2]:.3f} mm")
print()

# ============================================================================
# STEP 3b: Create uniform 2D grid for spatial fields
# ============================================================================
# x in [0, Lx], y in [-h, 0].  Lx determined from params grid.

Lx = Nx * mm_per_pixel
x2d_edges = np.arange(0, Lx + dx_2d, dx_2d)
y2d_edges = np.arange(-h_mm, 0 + dy_2d, dy_2d)
nx_2d = len(x2d_edges) - 1
ny_2d = len(y2d_edges) - 1
x2d_centers = 0.5 * (x2d_edges[:-1] + x2d_edges[1:])
y2d_centers = 0.5 * (y2d_edges[:-1] + y2d_edges[1:])

print(f"2D uniform grid: {nx_2d} x {ny_2d} bins ({dx_2d} x {dy_2d} mm)")
print(f"  x range: [0, {Lx:.1f}] mm    y range: [{-h_mm:.1f}, 0] mm")
print(f"  Resolution: {dx_2d} x {dy_2d} mm = "
      f"{dx_2d / mm_per_pixel:.2f} x {dy_2d / mm_per_pixel:.2f} px")
print()

# Pixel-space coordinate arrays
x2d_centers_px = x2d_centers / mm_per_pixel
y2d_centers_px = (y2d_centers + h_mm) / mm_per_pixel   # wall at 0 px, centreline at Ny px
x2d_edges_px = x2d_edges / mm_per_pixel
y2d_edges_px = (y2d_edges + h_mm) / mm_per_pixel

# ============================================================================
# STEP 4: Single-pass accumulation (1D Chebyshev + 2D uniform)
# ============================================================================

print("Accumulating statistics (single pass)...")

# --- 1D Chebyshev accumulators ---
n_sum = np.zeros(n_y, dtype=np.float64)
u_sum = np.zeros((n_y, 3), dtype=np.float64)
uu_sum = np.zeros((n_y, 3, 3), dtype=np.float64)

# --- Per-frame accumulators for bootstrap CI ---
n_per_frame = np.zeros((n_available, n_y), dtype=np.float64)
u_sum_per_frame = np.zeros((n_available, n_y, 3), dtype=np.float64)
uu_sum_per_frame = np.zeros((n_available, n_y, 3, 3), dtype=np.float64)

# --- 2D uniform grid accumulators (individual components) ---
n_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
# Means
u_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
v_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
w_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
# Second moments (6 unique components of symmetric tensor)
uu_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
vv_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
ww_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
uv_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
uw_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)
vw_sum_2d = np.zeros((nx_2d, ny_2d), dtype=np.float64)

t_start = time.time()
total_particles = 0
progress_interval = max(1, n_available // 20)

for idx, (file_a, file_b, frame_num) in enumerate(frame_pairs):
    # Load positions — 3 columns: x, y, z in mm
    pos_A = np.loadtxt(str(file_a))
    pos_B = np.loadtxt(str(file_b))

    if pos_A.shape[0] != pos_B.shape[0]:
        print(f"  Warning: frame {frame_num} particle count mismatch, skipping.")
        continue

    # Velocity = displacement / dt  (mm/s)
    vel = (pos_B - pos_A) / dt
    u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]

    # Midpoint positions for bin assignment
    x_mid = 0.5 * (pos_A[:, 0] + pos_B[:, 0])
    y_mid = 0.5 * (pos_A[:, 1] + pos_B[:, 1])

    # --- 1D Chebyshev accumulation ---
    bin_idx = np.digitize(y_mid, y_edges) - 1          # 0-based bin index
    valid = (bin_idx >= 0) & (bin_idx < n_y)
    bi = bin_idx[valid]
    vel_v = vel[valid]

    n_particles = int(valid.sum())
    total_particles += n_particles

    n_frame = np.bincount(bi, minlength=n_y).astype(np.float64)
    n_per_frame[idx] = n_frame
    n_sum += n_frame
    for i in range(3):
        u_frame_i = np.bincount(bi, weights=vel_v[:, i], minlength=n_y)
        u_sum_per_frame[idx, :, i] = u_frame_i
        u_sum[:, i] += u_frame_i
        for j in range(3):
            uu_frame_ij = np.bincount(
                bi, weights=vel_v[:, i] * vel_v[:, j], minlength=n_y
            )
            uu_sum_per_frame[idx, :, i, j] = uu_frame_ij
            uu_sum[:, i, j] += uu_frame_ij

    # --- 2D uniform grid accumulation ---
    ix = np.floor(x_mid / dx_2d).astype(int)
    iy = np.floor((y_mid - y2d_edges[0]) / dy_2d).astype(int)
    ok = (ix >= 0) & (ix < nx_2d) & (iy >= 0) & (iy < ny_2d)
    lin = ix[ok] * ny_2d + iy[ok]
    u_ok, v_ok, w_ok = u[ok], v[ok], w[ok]
    flat_sz = nx_2d * ny_2d

    n_sum_2d  += np.bincount(lin, minlength=flat_sz).reshape(nx_2d, ny_2d)
    u_sum_2d  += np.bincount(lin, weights=u_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    v_sum_2d  += np.bincount(lin, weights=v_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    w_sum_2d  += np.bincount(lin, weights=w_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    uu_sum_2d += np.bincount(lin, weights=u_ok * u_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    vv_sum_2d += np.bincount(lin, weights=v_ok * v_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    ww_sum_2d += np.bincount(lin, weights=w_ok * w_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    uv_sum_2d += np.bincount(lin, weights=u_ok * v_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    uw_sum_2d += np.bincount(lin, weights=u_ok * w_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)
    vw_sum_2d += np.bincount(lin, weights=v_ok * w_ok, minlength=flat_sz).reshape(nx_2d, ny_2d)

    if (idx + 1) % progress_interval == 0 or idx == n_available - 1:
        elapsed = time.time() - t_start
        print(
            f"  {idx + 1:>5d}/{n_available} frames "
            f"({100 * (idx + 1) / n_available:5.1f}%) — "
            f"{total_particles / 1e6:.2f}M particles — {elapsed:.1f}s"
        )

elapsed = time.time() - t_start
print(f"\n  Finished: {n_available} frames, "
      f"{total_particles / 1e6:.3f}M particles in {elapsed:.1f}s")
print()

# ============================================================================
# STEP 5: Compute statistics
# ============================================================================

print("Computing statistics...")

# --- 1D Chebyshev profiles ---
valid_bins = n_sum > 0

u_mean = np.full((n_y, 3), np.nan)
uu_mean = np.full((n_y, 3, 3), np.nan)
u_var = np.full((n_y, 3, 3), np.nan)

u_mean[valid_bins] = u_sum[valid_bins] / n_sum[valid_bins, None]
uu_mean[valid_bins] = uu_sum[valid_bins] / n_sum[valid_bins, None, None]

# Reynolds stress tensor:  <u_i' u_j'> = <u_i u_j> - <u_i><u_j>
u_var[valid_bins] = (
    uu_mean[valid_bins]
    - u_mean[valid_bins, :, None] * u_mean[valid_bins, None, :]
)

print(f"  1D bins with data: {int(valid_bins.sum())} / {n_y}")

# --- 2D fields: individual named variables ---
min_count_2d = 10
ok2d = n_sum_2d >= min_count_2d
N = np.where(ok2d, n_sum_2d, np.nan)

# Mean velocities  (nx_2d x ny_2d each)
U_mean_2d = np.where(ok2d, u_sum_2d / N, np.nan)
V_mean_2d = np.where(ok2d, v_sum_2d / N, np.nan)
W_mean_2d = np.where(ok2d, w_sum_2d / N, np.nan)

# Mean second moments
UU_mom_2d = np.where(ok2d, uu_sum_2d / N, np.nan)
VV_mom_2d = np.where(ok2d, vv_sum_2d / N, np.nan)
WW_mom_2d = np.where(ok2d, ww_sum_2d / N, np.nan)
UV_mom_2d = np.where(ok2d, uv_sum_2d / N, np.nan)
UW_mom_2d = np.where(ok2d, uw_sum_2d / N, np.nan)
VW_mom_2d = np.where(ok2d, vw_sum_2d / N, np.nan)

# Reynolds stresses:  <u'v'> = <uv> - <u><v>
uu_stress_2d = UU_mom_2d - U_mean_2d * U_mean_2d
vv_stress_2d = VV_mom_2d - V_mean_2d * V_mean_2d
ww_stress_2d = WW_mom_2d - W_mean_2d * W_mean_2d
uv_stress_2d = UV_mom_2d - U_mean_2d * V_mean_2d
uw_stress_2d = UW_mom_2d - U_mean_2d * W_mean_2d
vw_stress_2d = VW_mom_2d - V_mean_2d * W_mean_2d

print(f"  2D grid cells with data: {int(ok2d.sum())} / {nx_2d * ny_2d}")
print()

# ============================================================================
# STEP 5b: Bootstrap 95% confidence intervals (1D profiles)
# ============================================================================

N_boot = 2000
print(f"Bootstrap CI: {N_boot} resamples over {n_available} frames...")
t_boot = time.time()

rng_boot = np.random.default_rng(42)
stress_boot = np.full((N_boot, n_y, 3, 3), np.nan, dtype=np.float64)
umean_boot = np.full((N_boot, n_y, 3), np.nan, dtype=np.float64)

for b in range(N_boot):
    idx_b = rng_boot.integers(0, n_available, size=n_available)
    n_b = n_per_frame[idx_b].sum(axis=0)
    u_b = u_sum_per_frame[idx_b].sum(axis=0)
    uu_b = uu_sum_per_frame[idx_b].sum(axis=0)

    ok_b = n_b > 0
    mean_b = np.full((n_y, 3), np.nan)
    mean_b[ok_b] = u_b[ok_b] / n_b[ok_b, None]

    mom2_b = np.full((n_y, 3, 3), np.nan)
    mom2_b[ok_b] = uu_b[ok_b] / n_b[ok_b, None, None]

    var_b = np.full((n_y, 3, 3), np.nan)
    var_b[ok_b] = mom2_b[ok_b] - mean_b[ok_b, :, None] * mean_b[ok_b, None, :]

    stress_boot[b] = var_b / u_tau**2
    umean_boot[b] = mean_b / u_tau

# 95% CI (2.5th and 97.5th percentiles)
stress_ci_lo = np.nanpercentile(stress_boot, 2.5, axis=0)
stress_ci_hi = np.nanpercentile(stress_boot, 97.5, axis=0)
umean_ci_lo = np.nanpercentile(umean_boot, 2.5, axis=0)
umean_ci_hi = np.nanpercentile(umean_boot, 97.5, axis=0)

print(f"  Done in {time.time() - t_boot:.1f}s")
if valid_bins.any():
    mid_bin = n_y // 2
    ci_width = stress_ci_hi[mid_bin, 0, 0] - stress_ci_lo[mid_bin, 0, 0]
    print(f"  Example u'u'+ CI width at bin {mid_bin}: {ci_width:.4f} wall units")
print()

# ============================================================================
# STEP 6: Wall-unit normalization
# ============================================================================

y_plus = (h_mm + y_centers) / delta_nu      # wall distance in wall units
U_plus = u_mean / u_tau                      # mean velocity in wall units
stress_plus = u_var / u_tau**2               # Reynolds stresses in wall units

rng = valid_bins  # all bins are in [-h, 0]

print("Wall-unit normalization:")
print(f"  y+ range (with data): "
      f"[{np.nanmin(y_plus[rng]):.1f}, {np.nanmax(y_plus[rng]):.1f}]")
print(f"  U+ max (streamwise):  {np.nanmax(U_plus[rng, 0]):.2f}")
if rng.any():
    peak_uu = np.nanmax(stress_plus[rng, 0, 0])
    peak_yp = y_plus[rng][np.nanargmax(stress_plus[rng, 0, 0])]
    print(f"  <u'u'>+ peak:         {peak_uu:.2f} at y+ ~ {peak_yp:.0f}")
print()

# ============================================================================
# STEP 7: Save outputs
# ============================================================================

output_folder.mkdir(parents=True, exist_ok=True)

# Collect all 2D fields in a dict for easy saving
fields_2d = {
    # Coordinates in mm
    "x2d_centers": x2d_centers,
    "y2d_centers": y2d_centers,
    "x2d_edges": x2d_edges,
    "y2d_edges": y2d_edges,
    # Coordinates in pixels
    "x2d_centers_px": x2d_centers_px,
    "y2d_centers_px": y2d_centers_px,
    "x2d_edges_px": x2d_edges_px,
    "y2d_edges_px": y2d_edges_px,
    # Counts
    "n_sum_2d": n_sum_2d,
    # Mean velocities
    "U_mean_2d": U_mean_2d,
    "V_mean_2d": V_mean_2d,
    "W_mean_2d": W_mean_2d,
    # Reynolds stresses
    "uu_stress_2d": uu_stress_2d,
    "vv_stress_2d": vv_stress_2d,
    "ww_stress_2d": ww_stress_2d,
    "uv_stress_2d": uv_stress_2d,
    "uw_stress_2d": uw_stress_2d,
    "vw_stress_2d": vw_stress_2d,
}

fields_1d = {
    "y_centers": y_centers,
    "y_edges": y_edges,
    "y_plus": y_plus,
    "n_sum": n_sum,
    "u_mean": u_mean,
    "uu_mean": uu_mean,
    "u_var": u_var,
    "U_plus": U_plus,
    "stress_plus": stress_plus,
    # Bootstrap 95% CI
    "stress_ci_lo": stress_ci_lo,
    "stress_ci_hi": stress_ci_hi,
    "umean_ci_lo": umean_ci_lo,
    "umean_ci_hi": umean_ci_hi,
}

scalars = {
    "h_mm": h_mm,
    "u_tau": u_tau,
    "delta_nu": delta_nu,
    "Re_tau": Re_tau,
    "dt": dt,
    "n_frames": np.array(n_available),
    "mm_per_pixel": mm_per_pixel,
    "dx_2d_mm": dx_2d,
    "dy_2d_mm": dy_2d,
    "dx_2d_px": dx_2d / mm_per_pixel,
    "dy_2d_px": dy_2d / mm_per_pixel,
}

# --- .npz file ---
npz_file = output_folder / "direct_stats.npz"
np.savez(str(npz_file), **fields_1d, **fields_2d, **scalars)
print(f"Saved: {npz_file}")

# --- .mat file ---
mat_file = output_folder / "direct_stats.mat"
mat_dict = {}
mat_dict.update(fields_1d)
mat_dict.update(fields_2d)
mat_dict.update(scalars)
mat_dict["n_frames"] = n_available  # savemat prefers plain int
savemat(str(mat_file), mat_dict)
print(f"Saved: {mat_file}")
print()

# ============================================================================
# STEP 8: Validation plots
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Generating validation plots...")

kappa = 0.41
B_const = 5.2

# ---- Figure 1: Mean velocity profile (semilog U+ vs y+) ----
fig1, ax1 = plt.subplots(figsize=(8, 5))

yp_visc = np.linspace(0.1, 5, 100)
ax1.semilogx(yp_visc, yp_visc, "r-", lw=2, label=r"$U^+ = y^+$")

yp_log = np.logspace(1, 3, 100)
up_log = (1.0 / kappa) * np.log(yp_log) + B_const
ax1.semilogx(
    yp_log, up_log, "k--", lw=2,
    label=rf"Log law ($\kappa={kappa}$, $B={B_const}$)",
)

comp_names = [r"$\langle U_1 \rangle^+$",
              r"$\langle U_2 \rangle^+$",
              r"$\langle U_3 \rangle^+$"]
for ic in range(3):
    ax1.semilogx(y_plus[rng], U_plus[rng, ic], ".-", ms=4, label=comp_names[ic])

ax1.set_xlabel(r"$y^+$")
ax1.set_ylabel(r"$U^+$")
ax1.set_title(rf"Mean Velocity Profile ($Re_{{\tau}} = {Re_tau:.0f}$)")
ax1.legend(loc="upper left")
ax1.set_xlim(0.1, Re_tau)
ax1.set_ylim(-0.5, 25)
ax1.grid(True, which="both", alpha=0.3)
fig1.tight_layout()
fig1.savefig(str(output_folder / "fig1_mean_velocity_semilog.png"), dpi=200)
plt.close(fig1)
print("  Fig 1: Mean velocity (semilog U+ vs y+)")

# ---- Figure 2: Reynolds normal stresses vs y/h ----
fig2, ax2 = plt.subplots(figsize=(8, 5))

diag_labels = [r"$\langle u'_1 u'_1 \rangle^+$",
               r"$\langle u'_2 u'_2 \rangle^+$",
               r"$\langle u'_3 u'_3 \rangle^+$"]
for i in range(3):
    ax2.plot(
        stress_plus[rng, i, i], y_centers[rng] / h_mm, ".-", ms=4,
        label=diag_labels[i],
    )

ax2.set_xlabel(r"$\langle u'_i u'_i \rangle^+$")
ax2.set_ylabel(r"$y / h$")
ax2.set_title("Reynolds Normal Stresses")
ax2.legend(loc="best")
ax2.set_xlim(0, 10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(str(output_folder / "fig2_normal_stresses_yh.png"), dpi=200)
plt.close(fig2)
print("  Fig 2: Reynolds normal stresses vs y/h")

# ---- Figure 3: Reynolds shear stresses vs y/h ----
fig3, ax3 = plt.subplots(figsize=(8, 5))

shear_pairs = [(0, 1), (0, 2), (1, 2)]
for i, j in shear_pairs:
    label = rf"$\langle u'_{i+1} u'_{j+1} \rangle^+$"
    ax3.plot(
        stress_plus[rng, i, j], y_centers[rng] / h_mm, ".-", ms=4,
        label=label,
    )

yh_theory = np.linspace(-1, 0, 100)
ax3.plot(-(1 + yh_theory), yh_theory, "k-", lw=2, label=r"$-(1+y/h)$")

ax3.set_xlabel(r"$\langle u'_i u'_j \rangle^+$")
ax3.set_ylabel(r"$y / h$")
ax3.set_title("Reynolds Shear Stresses")
ax3.legend(loc="best")
ax3.set_xlim(-1, 1)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(str(output_folder / "fig3_shear_stresses_yh.png"), dpi=200)
plt.close(fig3)
print("  Fig 3: Reynolds shear stresses vs y/h")

# ---- Figure 4: Normal stresses vs y+ (semilog x-axis) with 95% CI ----
fig4, ax4 = plt.subplots(figsize=(8, 5))

ci_colors = ["C0", "C1", "C2"]
for i in range(3):
    ax4.semilogx(
        y_plus[rng], stress_plus[rng, i, i], ".-", ms=4,
        color=ci_colors[i], label=diag_labels[i],
    )
    ax4.fill_between(
        y_plus[rng],
        stress_ci_lo[rng, i, i],
        stress_ci_hi[rng, i, i],
        color=ci_colors[i], alpha=0.2,
        label=f"95% CI" if i == 0 else None,
    )

ax4.set_xlabel(r"$y^+$")
ax4.set_ylabel(r"$\langle u'_i u'_i \rangle^+$")
ax4.set_title("Reynolds Normal Stresses (wall units) — 95% bootstrap CI")
ax4.legend(loc="best")
ax4.set_xlim(0.1, Re_tau)
ax4.set_ylim(0, 10)
ax4.grid(True, which="both", alpha=0.3)
fig4.tight_layout()
fig4.savefig(str(output_folder / "fig4_normal_stresses_yplus.png"), dpi=200)
plt.close(fig4)
print("  Fig 4: Normal stresses vs y+ with 95% CI (semilog)")

# ---- Figure 5: 2D mean streamwise velocity field ----
# x horizontal (streamwise), y vertical (wall-normal, wall at bottom)
fig5, ax5 = plt.subplots(figsize=(12, 5))
im = ax5.pcolormesh(
    x2d_centers, y2d_centers + h_mm, U_mean_2d.T,
    cmap="viridis", shading="auto", rasterized=True,
)
fig5.colorbar(im, ax=ax5, label=r"$\langle u_x \rangle$ (mm/s)")
ax5.set_xlabel("x (mm)")
ax5.set_ylabel("y (mm, wall at 0)")
ax5.set_title(r"2D mean streamwise velocity $\langle u_x \rangle$")
ax5.set_aspect("equal")
fig5.tight_layout()
fig5.savefig(str(output_folder / "fig5_mean_ux_2d.png"), dpi=250)
plt.close(fig5)
print("  Fig 5: 2D mean streamwise velocity (mm)")

# ---- Figure 6: 2D mean streamwise velocity in pixel space ----
fig6, ax6 = plt.subplots(figsize=(12, 5))
im6 = ax6.pcolormesh(
    x2d_centers_px, y2d_centers_px, U_mean_2d.T,
    cmap="viridis", shading="auto", rasterized=True,
)
fig6.colorbar(im6, ax=ax6, label=r"$\langle u_x \rangle$ (mm/s)")
ax6.set_xlabel("x (px)")
ax6.set_ylabel("y (px, wall at 0)")
ax6.set_title(r"2D mean streamwise velocity $\langle u_x \rangle$ (pixel coords)")
ax6.set_aspect("equal")
fig6.tight_layout()
fig6.savefig(str(output_folder / "fig6_mean_ux_2d_px.png"), dpi=250)
plt.close(fig6)
print("  Fig 6: 2D mean streamwise velocity (px)")

print()
print("=" * 60)
print("  All done.")
print(f"  Outputs in: {output_folder}")
print("=" * 60)
