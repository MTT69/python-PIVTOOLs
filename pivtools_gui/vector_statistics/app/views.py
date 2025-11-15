"""
Vector Statistics API views
Provides endpoints for computing instantaneous statistics (mean and Reynolds stresses)
with progress tracking.
"""
import threading
import time
import uuid
from pathlib import Path

import dask
import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger
from scipy.io import savemat

matplotlib.use("Agg")

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ...plotting.plot_maker import make_scalar_settings, plot_scalar_field
from ...utils import camera_number
from pivtools_core.vector_loading import load_coords_from_directory, load_vectors_from_directory

statistics_bp = Blueprint("statistics", __name__)

# Global job tracking
statistics_jobs = {}


def create_disk_structuring_element(radius: int):
    """
    Create a disk structuring element and return neighbor offsets.
    Returns I, J offset arrays similar to MATLAB's strel('disk', radius).
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    I, J = np.where(mask)
    I = I - radius
    J = J - radius
    return I, J


def nansurround(arr, d):
    """Pad array with NaN values of width d on all sides."""
    if arr.ndim == 2:
        padded = np.pad(arr, d, mode='constant', constant_values=np.nan)
    else:
        # For 3D arrays, pad only first two dimensions
        pad_width = [(d, d), (d, d)] + [(0, 0)] * (arr.ndim - 2)
        padded = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
    return padded


def unsurround(arr, d):
    """Remove padding of width d from all sides."""
    if arr.ndim == 2:
        return arr[d:-d, d:-d]
    else:
        return arr[d:-d, d:-d, ...]


def localvel(u, v, d):
    """
    Compute local mean velocity within disk of radius d.
    Returns Vp with shape (H, W, 2) where last dimension is [u_local, v_local].
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    local_u = []
    local_v = []
    for i_offset, j_offset in zip(I, J):
        local_u.append(np.roll(u, (i_offset, j_offset), axis=(0, 1)))
        local_v.append(np.roll(v, (i_offset, j_offset), axis=(0, 1)))

    local_u = np.stack(local_u, axis=-1)
    local_v = np.stack(local_v, axis=-1)

    u_mean = np.nanmean(local_u, axis=-1)
    v_mean = np.nanmean(local_v, axis=-1)

    # Create ROI mask
    roi = np.ones(s)
    ind = np.isnan(u) & np.isnan(v)
    roi[ind] = np.nan
    roi[:d, :] = np.nan
    roi[-d:, :] = np.nan
    roi[:, :d] = np.nan
    roi[:, -d:] = np.nan

    u_mean = u_mean * roi
    v_mean = v_mean * roi

    return np.stack([u_mean, v_mean], axis=-1)


def gamma1(x, y, u, v, d=10):
    """
    Compute gamma1 vortex detection criterion.
    Based on Graftieaux et al. (2001).

    Parameters:
    - x, y: coordinate grids (H, W)
    - u, v: velocity components (H, W)
    - d: disk radius in grid points

    Returns:
    - G1: gamma1 field (H, W)
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    # Get center point
    i0, j0 = s[0] // 2, s[1] // 2
    ind0 = np.ravel_multi_index((i0, j0), s)

    # Get neighbor indices
    IN = np.ravel_multi_index((I + i0, J + j0), s)

    # Compute angles from center to neighbors
    x_flat = x.ravel()
    y_flat = y.ravel()
    A = np.arctan2(y_flat[IN] - y_flat[ind0], x_flat[IN] - x_flat[ind0])

    # Pad arrays
    U = nansurround(u, d)
    V = nansurround(v, d)
    S = U.shape

    local_s = []
    for k, (i_offset, j_offset) in enumerate(zip(I, J)):
        angle_grid = np.ones(S) * A[k]
        u_shifted = np.roll(U, (i_offset, j_offset), axis=(0, 1))
        v_shifted = np.roll(V, (i_offset, j_offset), axis=(0, 1))
        T = np.arctan2(v_shifted, u_shifted)
        local_s.append(np.sin(angle_grid - T))

    local_s = np.stack(local_s, axis=-1)
    G1 = -np.nansum(local_s, axis=-1) / len(I)
    G1 = unsurround(G1, d)

    return G1


def gamma2(x, y, u, v, d=10):
    """
    Compute gamma2 vortex detection criterion.
    Based on Graftieaux et al. (2001).

    Parameters:
    - x, y: coordinate grids (H, W)
    - u, v: velocity components (H, W)
    - d: disk radius in grid points

    Returns:
    - G2: gamma2 field (H, W)
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    # Get center point
    i0, j0 = s[0] // 2, s[1] // 2
    ind0 = np.ravel_multi_index((i0, j0), s)

    # Get neighbor indices
    IN = np.ravel_multi_index((I + i0, J + j0), s)

    # Compute angles from center to neighbors
    x_flat = x.ravel()
    y_flat = y.ravel()
    A = np.arctan2(y_flat[IN] - y_flat[ind0], x_flat[IN] - x_flat[ind0])

    # Pad arrays
    U = nansurround(u, d)
    V = nansurround(v, d)
    S = U.shape

    # Compute local velocity
    Vp = localvel(U, V, d)

    local_s = []
    for k, (i_offset, j_offset) in enumerate(zip(I, J)):
        angle_grid = np.ones(S) * A[k]
        u_shifted = np.roll(U, (i_offset, j_offset), axis=(0, 1))
        v_shifted = np.roll(V, (i_offset, j_offset), axis=(0, 1))
        u_rel = u_shifted - Vp[..., 0]
        v_rel = v_shifted - Vp[..., 1]
        T = np.arctan2(v_rel, u_rel)
        local_s.append(np.sin(angle_grid - T))

    local_s = np.stack(local_s, axis=-1)
    G2 = -np.nansum(local_s, axis=-1) / len(I)
    G2 = unsurround(G2, d)

    return G2


def find_non_empty_runs_in_file(data_dir: Path, vector_format: str) -> list:
    """
    Find which runs have non-empty vector data by checking the first vector file.
    Returns list of 1-based run numbers that contain valid data.
    """
    if not data_dir.exists():
        return []

    # Get first vector file to check run structure
    first_file = data_dir / (vector_format % 1)
    if not first_file.exists():
        return []

    try:
        mat = scipy.io.loadmat(str(first_file), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in mat:
            return []
        
        piv_result = mat["piv_result"]
        valid_runs = []

        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            # Multiple runs
            for run_idx in range(piv_result.size):
                pr = piv_result[run_idx]
                try:
                    # Check if ux has valid data
                    ux = np.asarray(getattr(pr, "ux", np.array([])))
                    if ux.size > 0 and not np.all(np.isnan(ux)):
                        valid_runs.append(run_idx + 1)  # Convert to 1-based
                except Exception:
                    pass
        else:
            # Single run
            try:
                ux = np.asarray(getattr(piv_result, "ux", np.array([])))
                if ux.size > 0 and not np.all(np.isnan(ux)):
                    valid_runs.append(1)
            except Exception:
                pass

        return valid_runs
    except Exception as e:
        logger.error(f"Error checking runs in {first_file}: {e}")
        return []


def compute_statistics_for_camera(
    base_dir: Path,
    camera: int,
    use_merged: bool,
    num_images: int,
    type_name: str,
    endpoint: str,
    vector_format: str,
    job_id: str,
):
    """
    Compute instantaneous statistics for a single camera or merged data.
    Updates job status in statistics_jobs dictionary.
    """
    try:
        cam_folder = "Merged" if use_merged else f"Cam{camera}"
        logger.info(f"[Statistics] Starting for {cam_folder}, endpoint={endpoint}")

        # Update job status
        statistics_jobs[job_id]["status"] = "running"
        statistics_jobs[job_id]["camera"] = cam_folder
        statistics_jobs[job_id]["progress"] = 5

        # Get paths - use cam (number) not cam_folder (string)
        paths = get_data_paths(
            base_dir=base_dir,
            num_images=num_images,
            cam=camera,
            type_name=type_name,
            endpoint=endpoint,
            use_merged=use_merged,
        )

        data_dir = paths["data_dir"]
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        statistics_jobs[job_id]["progress"] = 10

        # Find non-empty runs
        valid_runs = find_non_empty_runs_in_file(data_dir, vector_format)
        if not valid_runs:
            raise ValueError(f"No valid runs found in {data_dir}")

        logger.info(f"[Statistics] Found {len(valid_runs)} valid runs: {valid_runs}")
        statistics_jobs[job_id]["valid_runs"] = valid_runs
        statistics_jobs[job_id]["progress"] = 15

        # Create a minimal config object for loading vectors
        class MinimalConfig:
            def __init__(self, num_images, vector_format, piv_chunk_size=100):
                self.num_images = num_images
                self.vector_format = vector_format
                self.piv_chunk_size = piv_chunk_size

        config = MinimalConfig(num_images, vector_format)

        # Load and process each run separately (can't stack due to different grid sizes)
        logger.info(f"[Statistics] Loading vectors from {data_dir} for runs {valid_runs}")
        statistics_jobs[job_id]["progress"] = 20
        
        # Load coordinates for all valid runs first
        coords_x_list, coords_y_list = load_coords_from_directory(data_dir, runs=valid_runs)
        statistics_jobs[job_id]["progress"] = 25
        
        # Process each run independently
        mean_ux_all = []
        mean_uy_all = []
        mean_uz_all = [] if False else None  # Will set based on first run
        b_mask_all = []
        uu_all = []
        uv_all = []
        vv_all = []
        uw_all = [] if False else None
        vw_all = [] if False else None
        ww_all = [] if False else None

        # Additional statistics
        urms_all = []
        vrms_all = []
        wrms_all = [] if False else None
        tke_all = []
        Iu_all = []
        Iv_all = []
        Iw_all = [] if False else None
        divergence_all = []
        vorticity_all = []
        gamma1_all = []
        gamma2_all = []

        stereo = None  # Will be determined from first run
        
        for run_idx, run_num in enumerate(valid_runs):
            logger.info(f"[Statistics] Processing run {run_num} ({run_idx + 1}/{len(valid_runs)})")
            
            # Load this run's data
            arr_run = load_vectors_from_directory(data_dir, config, runs=[run_num])
            # Shape: (N_files, 1, 3_or_4, H, W)
            arr_run = arr_run[:, 0, :, :, :]  # (N_files, 3_or_4, H, W)
            
            # Check for stereo on first run
            if stereo is None:
                stereo = arr_run.shape[1] >= 4
                if stereo:
                    logger.info("[Statistics] Detected stereo data (4 components)")
                    mean_uz_all = []
                    uw_all = []
                    vw_all = []
                    ww_all = []
                    wrms_all = []
                    Iw_all = []
            
            # Extract components
            ux = arr_run[:, 0, :, :]  # (N, H, W)
            uy = arr_run[:, 1, :, :]
            if stereo:
                uz = arr_run[:, 2, :, :]
                bmask = arr_run[:, 3, :, :]
            else:
                bmask = arr_run[:, 2, :, :]
            
            # Compute statistics for this run
            mean_ux = ux.mean(axis=0)
            mean_uy = uy.mean(axis=0)
            b_mask = bmask[0]
            E_ux2 = (ux**2).mean(axis=0)
            E_uy2 = (uy**2).mean(axis=0)
            E_uxuy = (ux * uy).mean(axis=0)
            
            if stereo:
                mean_uz = uz.mean(axis=0)
                E_uz2 = (uz**2).mean(axis=0)
                E_uxuz = (ux * uz).mean(axis=0)
                E_uyuz = (uy * uz).mean(axis=0)
                
                # Compute all at once
                mean_ux_c, mean_uy_c, mean_uz_c, b_mask_c, E_ux2_c, E_uy2_c, E_uxuy_c, E_uz2_c, E_uxuz_c, E_uyuz_c = dask.compute(
                    mean_ux, mean_uy, mean_uz, b_mask, E_ux2, E_uy2, E_uxuy, E_uz2, E_uxuz, E_uyuz
                )
                
                # Compute Reynolds stresses
                uu = E_ux2_c - mean_ux_c**2
                uv = E_uxuy_c - (mean_ux_c * mean_uy_c)
                vv = E_uy2_c - mean_uy_c**2
                uw = E_uxuz_c - (mean_ux_c * mean_uz_c)
                vw = E_uyuz_c - (mean_uy_c * mean_uz_c)
                ww = E_uz2_c - mean_uz_c**2

                # Compute RMS velocities
                urms = np.sqrt(np.abs(uu))
                vrms = np.sqrt(np.abs(vv))
                wrms = np.sqrt(np.abs(ww))

                # Compute TKE (turbulent kinetic energy)
                tke = 0.5 * (uu + vv + ww)

                # Compute turbulent intensities (normalized by mean velocity)
                # Avoid division by zero
                mean_vel_mag = np.sqrt(mean_ux_c**2 + mean_uy_c**2 + mean_uz_c**2)
                mean_vel_mag = np.where(np.abs(mean_vel_mag) < 1e-10, np.nan, mean_vel_mag)
                Iu = urms / mean_vel_mag
                Iv = vrms / mean_vel_mag
                Iw = wrms / mean_vel_mag

                # Get coordinates for this run
                cx = coords_x_list[run_idx] if run_idx < len(coords_x_list) else None
                cy = coords_y_list[run_idx] if run_idx < len(coords_y_list) else None

                # Compute divergence: ∂u/∂x + ∂v/∂y + ∂w/∂z
                # For 2D measurements, we assume ∂w/∂z ≈ 0
                if cx is not None and cy is not None:
                    dx = np.gradient(cx, axis=1)
                    dy = np.gradient(cy, axis=0)
                    dudx = np.gradient(mean_ux_c, axis=1) / dx
                    dvdy = np.gradient(mean_uy_c, axis=0) / dy
                    divergence = dudx + dvdy
                else:
                    # Assume uniform grid spacing = 1
                    dudx = np.gradient(mean_ux_c, axis=1)
                    dvdy = np.gradient(mean_uy_c, axis=0)
                    divergence = dudx + dvdy

                # Compute vorticity (z-component): ∂v/∂x - ∂u/∂y
                if cx is not None and cy is not None:
                    dx_vort = np.gradient(cx, axis=1)
                    dy_vort = np.gradient(cy, axis=0)
                    dvdx = np.gradient(mean_uy_c, axis=1) / dx_vort
                    dudy = np.gradient(mean_ux_c, axis=0) / dy_vort
                    vorticity = dvdx - dudy
                else:
                    dvdx = np.gradient(mean_uy_c, axis=1)
                    dudy = np.gradient(mean_ux_c, axis=0)
                    vorticity = dvdx - dudy

                # Compute gamma1 and gamma2
                if cx is not None and cy is not None:
                    g1 = gamma1(cx, cy, mean_ux_c, mean_uy_c, d=10)
                    g2 = gamma2(cx, cy, mean_ux_c, mean_uy_c, d=10)
                else:
                    # Create simple meshgrid if coordinates not available
                    y_grid, x_grid = np.meshgrid(np.arange(mean_ux_c.shape[1]), np.arange(mean_ux_c.shape[0]))
                    g1 = gamma1(x_grid, y_grid, mean_ux_c, mean_uy_c, d=10)
                    g2 = gamma2(x_grid, y_grid, mean_ux_c, mean_uy_c, d=10)

                mean_uz_all.append(mean_uz_c)
                uw_all.append(uw)
                vw_all.append(vw)
                ww_all.append(ww)
                urms_all.append(urms)
                vrms_all.append(vrms)
                wrms_all.append(wrms)
                tke_all.append(tke)
                Iu_all.append(Iu)
                Iv_all.append(Iv)
                Iw_all.append(Iw)
                divergence_all.append(divergence)
                vorticity_all.append(vorticity)
                gamma1_all.append(g1)
                gamma2_all.append(g2)
            else:
                # Compute all at once
                mean_ux_c, mean_uy_c, b_mask_c, E_ux2_c, E_uy2_c, E_uxuy_c = dask.compute(
                    mean_ux, mean_uy, b_mask, E_ux2, E_uy2, E_uxuy
                )

                # Compute Reynolds stresses
                uu = E_ux2_c - mean_ux_c**2
                uv = E_uxuy_c - (mean_ux_c * mean_uy_c)
                vv = E_uy2_c - mean_uy_c**2

                # Compute RMS velocities
                urms = np.sqrt(np.abs(uu))
                vrms = np.sqrt(np.abs(vv))

                # Compute TKE (turbulent kinetic energy) - 2D case
                tke = 0.5 * (uu + vv)

                # Compute turbulent intensities (normalized by mean velocity)
                # Avoid division by zero
                mean_vel_mag = np.sqrt(mean_ux_c**2 + mean_uy_c**2)
                mean_vel_mag = np.where(np.abs(mean_vel_mag) < 1e-10, np.nan, mean_vel_mag)
                Iu = urms / mean_vel_mag
                Iv = vrms / mean_vel_mag

                # Get coordinates for this run
                cx = coords_x_list[run_idx] if run_idx < len(coords_x_list) else None
                cy = coords_y_list[run_idx] if run_idx < len(coords_y_list) else None

                # Compute divergence: ∂u/∂x + ∂v/∂y
                if cx is not None and cy is not None:
                    dx = np.gradient(cx, axis=1)
                    dy = np.gradient(cy, axis=0)
                    dudx = np.gradient(mean_ux_c, axis=1) / dx
                    dvdy = np.gradient(mean_uy_c, axis=0) / dy
                    divergence = dudx + dvdy
                else:
                    # Assume uniform grid spacing = 1
                    dudx = np.gradient(mean_ux_c, axis=1)
                    dvdy = np.gradient(mean_uy_c, axis=0)
                    divergence = dudx + dvdy

                # Compute vorticity (z-component): ∂v/∂x - ∂u/∂y
                if cx is not None and cy is not None:
                    dx_vort = np.gradient(cx, axis=1)
                    dy_vort = np.gradient(cy, axis=0)
                    dvdx = np.gradient(mean_uy_c, axis=1) / dx_vort
                    dudy = np.gradient(mean_ux_c, axis=0) / dy_vort
                    vorticity = dvdx - dudy
                else:
                    dvdx = np.gradient(mean_uy_c, axis=1)
                    dudy = np.gradient(mean_ux_c, axis=0)
                    vorticity = dvdx - dudy

                # Compute gamma1 and gamma2
                if cx is not None and cy is not None:
                    g1 = gamma1(cx, cy, mean_ux_c, mean_uy_c, d=10)
                    g2 = gamma2(cx, cy, mean_ux_c, mean_uy_c, d=10)
                else:
                    # Create simple meshgrid if coordinates not available
                    y_grid, x_grid = np.meshgrid(np.arange(mean_ux_c.shape[1]), np.arange(mean_ux_c.shape[0]))
                    g1 = gamma1(x_grid, y_grid, mean_ux_c, mean_uy_c, d=10)
                    g2 = gamma2(x_grid, y_grid, mean_ux_c, mean_uy_c, d=10)

                urms_all.append(urms)
                vrms_all.append(vrms)
                tke_all.append(tke)
                Iu_all.append(Iu)
                Iv_all.append(Iv)
                divergence_all.append(divergence)
                vorticity_all.append(vorticity)
                gamma1_all.append(g1)
                gamma2_all.append(g2)

            # Store results
            mean_ux_all.append(mean_ux_c)
            mean_uy_all.append(mean_uy_c)
            b_mask_all.append(b_mask_c)
            uu_all.append(uu)
            uv_all.append(uv)
            vv_all.append(vv)
            
            # Update progress
            progress = 30 + int((run_idx + 1) / len(valid_runs) * 45)  # 30-75%
            statistics_jobs[job_id]["progress"] = progress
        
        logger.info("[Statistics] Completed statistics computation for all runs")
        statistics_jobs[job_id]["progress"] = 75

        # Create output directory
        stats_dir = paths["stats_dir"]
        stats_dir.mkdir(parents=True, exist_ok=True)
        mean_stats_dir = stats_dir / "mean_stats"
        mean_stats_dir.mkdir(parents=True, exist_ok=True)

        # Get config for plotting
        cfg = get_config()
        plot_extension = getattr(cfg, "plot_save_extension", ".png")
        save_pickle = getattr(cfg, "plot_save_pickle", False)

        # Generate plots for each run
        # logger.info("[Statistics] Generating plots")
        # statistics_jobs[job_id]["progress"] = 80

        # for idx, run_label in enumerate(valid_runs):
        #     mask_bool = np.asarray(b_mask_all[idx]).astype(bool)
        #     cx = coords_x_list[idx] if idx < len(coords_x_list) else None
        #     cy = coords_y_list[idx] if idx < len(coords_y_list) else None

        #     # Plot mean velocities (ux, uy, uz if stereo)
        #     save_base_ux = mean_stats_dir / f"ux_{run_label}"
        #     settings_ux = make_scalar_settings(
        #         cfg,
        #         variable="ux",
        #         run_label=run_label,
        #         save_basepath=save_base_ux,
        #         variable_units="m/s",
        #         coords_x=cx,
        #         coords_y=cy,
        #     )
        #     fig_ux, _, _ = plot_scalar_field(mean_ux_all[idx], mask_bool, settings_ux)
        #     fig_ux.savefig(f"{save_base_ux}{plot_extension}", dpi=1200, bbox_inches="tight")
        #     if save_pickle:
        #         import pickle
        #         with open(f"{save_base_ux}.pkl", "wb") as f:
        #             pickle.dump(fig_ux, f)
        #     plt.close(fig_ux)

        #     save_base_uy = mean_stats_dir / f"uy_{run_label}"
        #     settings_uy = make_scalar_settings(
        #         cfg,
        #         variable="uy",
        #         run_label=run_label,
        #         save_basepath=save_base_uy,
        #         variable_units="m/s",
        #         coords_x=cx,
        #         coords_y=cy,
        #     )
        #     fig_uy, _, _ = plot_scalar_field(mean_uy_all[idx], mask_bool, settings_uy)
        #     fig_uy.savefig(f"{save_base_uy}{plot_extension}", dpi=1200, bbox_inches="tight")
        #     if save_pickle:
        #         import pickle
        #         with open(f"{save_base_uy}.pkl", "wb") as f:
        #             pickle.dump(fig_uy, f)
        #     plt.close(fig_uy)

        #     if stereo:
        #         save_base_uz = mean_stats_dir / f"uz_{run_label}"
        #         settings_uz = make_scalar_settings(
        #             cfg,
        #             variable="uz",
        #             run_label=run_label,
        #             save_basepath=save_base_uz,
        #             variable_units="m/s",
        #             coords_x=cx,
        #             coords_y=cy,
        #         )
        #         fig_uz, _, _ = plot_scalar_field(mean_uz_all[idx], mask_bool, settings_uz)
        #         fig_uz.savefig(f"{save_base_uz}{plot_extension}", dpi=1200, bbox_inches="tight")
        #         if save_pickle:
        #             import pickle
        #             with open(f"{save_base_uz}.pkl", "wb") as f:
        #                 pickle.dump(fig_uz, f)
        #         plt.close(fig_uz)

        #     # Plot Reynolds stresses (uu, uv, vv, and uw, vw, ww if stereo)
        #     save_base_uu = mean_stats_dir / f"uu_{run_label}"
        #     settings_uu = make_scalar_settings(
        #         cfg,
        #         variable="uu",
        #         run_label=run_label,
        #         save_basepath=save_base_uu,
        #         variable_units="m²/s²",
        #         coords_x=cx,
        #         coords_y=cy,
        #     )
        #     fig_uu, _, _ = plot_scalar_field(uu_all[idx], mask_bool, settings_uu)
        #     fig_uu.savefig(f"{save_base_uu}{plot_extension}", dpi=1200, bbox_inches="tight")
        #     if save_pickle:
        #         import pickle
        #         with open(f"{save_base_uu}.pkl", "wb") as f:
        #             pickle.dump(fig_uu, f)
        #     plt.close(fig_uu)

        #     save_base_uv = mean_stats_dir / f"uv_{run_label}"
        #     settings_uv = make_scalar_settings(
        #         cfg,
        #         variable="uv",
        #         run_label=run_label,
        #         save_basepath=save_base_uv,
        #         variable_units="m²/s²",
        #         coords_x=cx,
        #         coords_y=cy,
        #     )
        #     fig_uv, _, _ = plot_scalar_field(uv_all[idx], mask_bool, settings_uv)
        #     fig_uv.savefig(f"{save_base_uv}{plot_extension}", dpi=1200, bbox_inches="tight")
        #     if save_pickle:
        #         import pickle
        #         with open(f"{save_base_uv}.pkl", "wb") as f:
        #             pickle.dump(fig_uv, f)
        #     plt.close(fig_uv)

        #     save_base_vv = mean_stats_dir / f"vv_{run_label}"
        #     settings_vv = make_scalar_settings(
        #         cfg,
        #         variable="vv",
        #         run_label=run_label,
        #         save_basepath=save_base_vv,
        #         variable_units="m²/s²",
        #         coords_x=cx,
        #         coords_y=cy,
        #     )
        #     fig_vv, _, _ = plot_scalar_field(vv_all[idx], mask_bool, settings_vv)
        #     fig_vv.savefig(f"{save_base_vv}{plot_extension}", dpi=1200, bbox_inches="tight")
        #     if save_pickle:
        #         import pickle
        #         with open(f"{save_base_vv}.pkl", "wb") as f:
        #             pickle.dump(fig_vv, f)
        #     plt.close(fig_vv)

        #     if stereo:
        #         save_base_uw = mean_stats_dir / f"uw_{run_label}"
        #         settings_uw = make_scalar_settings(
        #             cfg,
        #             variable="uw",
        #             run_label=run_label,
        #             save_basepath=save_base_uw,
        #             variable_units="m²/s²",
        #             coords_x=cx,
        #             coords_y=cy,
        #         )
        #         fig_uw, _, _ = plot_scalar_field(uw_all[idx], mask_bool, settings_uw)
        #         fig_uw.savefig(f"{save_base_uw}{plot_extension}", dpi=1200, bbox_inches="tight")
        #         if save_pickle:
        #             import pickle
        #             with open(f"{save_base_uw}.pkl", "wb") as f:
        #                 pickle.dump(fig_uw, f)
        #         plt.close(fig_uw)

        #         save_base_vw = mean_stats_dir / f"vw_{run_label}"
        #         settings_vw = make_scalar_settings(
        #             cfg,
        #             variable="vw",
        #             run_label=run_label,
        #             save_basepath=save_base_vw,
        #             variable_units="m²/s²",
        #             coords_x=cx,
        #             coords_y=cy,
        #         )
        #         fig_vw, _, _ = plot_scalar_field(vw_all[idx], mask_bool, settings_vw)
        #         fig_vw.savefig(f"{save_base_vw}{plot_extension}", dpi=1200, bbox_inches="tight")
        #         if save_pickle:
        #             import pickle
        #             with open(f"{save_base_vw}.pkl", "wb") as f:
        #                 pickle.dump(fig_vw, f)
        #         plt.close(fig_vw)

        #         save_base_ww = mean_stats_dir / f"ww_{run_label}"
        #         settings_ww = make_scalar_settings(
        #             cfg,
        #             variable="ww",
        #             run_label=run_label,
        #             save_basepath=save_base_ww,
        #             variable_units="m²/s²",
        #             coords_x=cx,
        #             coords_y=cy,
        #         )
        #         fig_ww, _, _ = plot_scalar_field(ww_all[idx], mask_bool, settings_ww)
        #         fig_ww.savefig(f"{save_base_ww}{plot_extension}", dpi=1200, bbox_inches="tight")
        #         if save_pickle:
        #             import pickle
        #             with open(f"{save_base_ww}.pkl", "wb") as f:
        #                 pickle.dump(fig_ww, f)
        #         plt.close(fig_ww)

        statistics_jobs[job_id]["progress"] = 85

        # Build piv_result structured array
        logger.info("[Statistics] Building piv_result structure")
        n_passes = len(valid_runs)
        dt_fields = [
            ("ux", object),
            ("uy", object),
            ("b_mask", object),
            ("uu", object),
            ("uv", object),
            ("vv", object),
            ("urms", object),
            ("vrms", object),
            ("tke", object),
            ("Iu", object),
            ("Iv", object),
            ("divergence", object),
            ("vorticity", object),
            ("gamma1", object),
            ("gamma2", object),
        ]
        if stereo:
            dt_fields.extend([
                ("uz", object),
                ("uw", object),
                ("vw", object),
                ("ww", object),
                ("wrms", object),
                ("Iw", object),
            ])

        dt = np.dtype(dt_fields)
        # Create piv_result array with size = max run number (preserving run positions)
        max_run = max(valid_runs)
        piv_result = np.empty((max_run,), dtype=dt)
        
        # Initialize all positions with empty arrays (for runs that don't exist)
        for i in range(max_run):
            piv_result["ux"][i] = np.array([])
            piv_result["uy"][i] = np.array([])
            piv_result["b_mask"][i] = np.array([])
            piv_result["uu"][i] = np.array([])
            piv_result["uv"][i] = np.array([])
            piv_result["vv"][i] = np.array([])
            piv_result["urms"][i] = np.array([])
            piv_result["vrms"][i] = np.array([])
            piv_result["tke"][i] = np.array([])
            piv_result["Iu"][i] = np.array([])
            piv_result["Iv"][i] = np.array([])
            piv_result["divergence"][i] = np.array([])
            piv_result["vorticity"][i] = np.array([])
            piv_result["gamma1"][i] = np.array([])
            piv_result["gamma2"][i] = np.array([])
            if stereo:
                piv_result["uz"][i] = np.array([])
                piv_result["uw"][i] = np.array([])
                piv_result["vw"][i] = np.array([])
                piv_result["ww"][i] = np.array([])
                piv_result["wrms"][i] = np.array([])
                piv_result["Iw"][i] = np.array([])

        # Fill piv_result only at positions corresponding to valid runs
        # This preserves run positions (e.g., if valid_runs=[3,4], indices 0,1 stay empty, 2,3 get data)
        for list_idx, run_num in enumerate(valid_runs):
            piv_idx = run_num - 1  # Convert run number to 0-based index
            piv_result["ux"][piv_idx] = mean_ux_all[list_idx]
            piv_result["uy"][piv_idx] = mean_uy_all[list_idx]
            piv_result["b_mask"][piv_idx] = b_mask_all[list_idx]
            piv_result["uu"][piv_idx] = uu_all[list_idx]
            piv_result["uv"][piv_idx] = uv_all[list_idx]
            piv_result["vv"][piv_idx] = vv_all[list_idx]
            piv_result["urms"][piv_idx] = urms_all[list_idx]
            piv_result["vrms"][piv_idx] = vrms_all[list_idx]
            piv_result["tke"][piv_idx] = tke_all[list_idx]
            piv_result["Iu"][piv_idx] = Iu_all[list_idx]
            piv_result["Iv"][piv_idx] = Iv_all[list_idx]
            piv_result["divergence"][piv_idx] = divergence_all[list_idx]
            piv_result["vorticity"][piv_idx] = vorticity_all[list_idx]
            piv_result["gamma1"][piv_idx] = gamma1_all[list_idx]
            piv_result["gamma2"][piv_idx] = gamma2_all[list_idx]
            if stereo:
                piv_result["uz"][piv_idx] = mean_uz_all[list_idx]
                piv_result["uw"][piv_idx] = uw_all[list_idx]
                piv_result["vw"][piv_idx] = vw_all[list_idx]
                piv_result["ww"][piv_idx] = ww_all[list_idx]
                piv_result["wrms"][piv_idx] = wrms_all[list_idx]
                piv_result["Iw"][piv_idx] = Iw_all[list_idx]

        # Build coordinates structure (also preserving run positions)
        dt_coords = np.dtype([("x", object), ("y", object)])
        coordinates = np.empty((max_run,), dtype=dt_coords)
        
        # Initialize all positions with empty arrays
        for i in range(max_run):
            coordinates["x"][i] = np.array([])
            coordinates["y"][i] = np.array([])
        
        # Fill only valid run positions
        for list_idx, run_num in enumerate(valid_runs):
            piv_idx = run_num - 1  # Convert run number to 0-based index
            coordinates["x"][piv_idx] = coords_x_list[list_idx]
            coordinates["y"][piv_idx] = coords_y_list[list_idx]

        statistics_jobs[job_id]["progress"] = 95

        # Save results in the format expected by the plotting system
        out_file = mean_stats_dir / "mean_stats.mat"
        logger.info(f"[Statistics] Saving results to {out_file}")

        meta_dict = {
            "endpoint": endpoint,
            "use_merged": use_merged,
            "camera": cam_folder,
            "selected_passes": valid_runs,
            "n_passes": int(n_passes),
            "stereo": stereo,
            "definitions": (
                "ux=<u>, uy=<v>, uu=<u'^2>, uv=<u'v'>, vv=<v'^2>, "
                "urms=sqrt(<u'^2>), vrms=sqrt(<v'^2>), "
                "tke=0.5*(uu+vv+ww), "
                "Iu=urms/|U|, Iv=vrms/|U|, "
                "divergence=du/dx+dv/dy, "
                "vorticity=dv/dx-du/dy, "
                "gamma1=vortex criterion (Graftieaux), "
                "gamma2=vortex criterion with local velocity"
                + (", uz=<w>, uw=<u'w'>, vw=<v'w'>, ww=<w'^2>, wrms=sqrt(<w'^2>), Iw=wrms/|U|" if stereo else "")
            ),
        }

        # Save in the same file with both piv_result, coordinates, and meta
        savemat(
            out_file, 
            {
                "piv_result": piv_result, 
                "coordinates": coordinates,
                "meta": meta_dict,
            }
        )

        statistics_jobs[job_id]["progress"] = 100
        statistics_jobs[job_id]["status"] = "completed"
        statistics_jobs[job_id]["output_file"] = str(out_file)
        logger.info(f"[Statistics] Completed successfully for {cam_folder}")

    except Exception as e:
        logger.error(f"[Statistics] Error: {e}", exc_info=True)
        statistics_jobs[job_id]["status"] = "failed"
        statistics_jobs[job_id]["error"] = str(e)


@statistics_bp.route("/statistics/calculate", methods=["POST"])
def calculate_statistics():
    """
    Start statistics calculation job.
    Expects JSON with: base_path_idx, cameras (list), include_merged (bool), 
                       image_count, type_name, endpoint
    """
    data = request.get_json() or {}
    logger.info(f"Received statistics calculation request: {data}")
    base_path_idx = int(data.get("base_path_idx", 0))
    cameras = data.get("cameras", [])  # List of camera numbers
    include_merged = bool(data.get("include_merged", False))
    image_count = int(data.get("image_count", 1000))
    type_name = data.get("type_name", "instantaneous")
    endpoint = data.get("endpoint", "")

    try:
        cfg = get_config()
        base_paths = getattr(cfg, "base_paths", getattr(cfg, "source_paths", []))
        if not base_paths or base_path_idx >= len(base_paths):
            return jsonify({"error": "Invalid base_path_idx"}), 400

        base_dir = Path(base_paths[base_path_idx])
        vector_format = getattr(cfg, "vector_format", "%05d.mat")

        # Create a parent job to track all sub-jobs
        parent_job_id = str(uuid.uuid4())
        sub_jobs = []

        # Check if merged data exists if requested
        if include_merged:
            # For merged data, we still need to pass a camera number (use first camera)
            # but with use_merged=True flag
            first_cam = cameras[0] if cameras else 1
            merged_paths = get_data_paths(
                base_dir=base_dir,
                num_images=image_count,
                cam=first_cam,
                type_name=type_name,
                endpoint=endpoint,
                use_merged=True,
            )
            if merged_paths["data_dir"].exists():
                job_id = str(uuid.uuid4())
                sub_jobs.append({"job_id": job_id, "type": "merged"})
                statistics_jobs[job_id] = {
                    "status": "starting",
                    "progress": 0,
                    "start_time": time.time(),
                    "camera": "Merged",
                    "parent_job_id": parent_job_id,
                }

                # Start thread for merged - pass first camera number
                thread = threading.Thread(
                    target=compute_statistics_for_camera,
                    args=(
                        base_dir,
                        first_cam,  # Use first camera number
                        True,  # use_merged
                        image_count,
                        type_name,
                        endpoint,
                        vector_format,
                        job_id,
                    ),
                )
                thread.daemon = True
                thread.start()
            else:
                logger.warning(f"Merged data directory not found: {merged_paths['data_dir']}")

        # Start jobs for each camera
        for cam in cameras:
            cam_num = camera_number(cam)
            job_id = str(uuid.uuid4())
            sub_jobs.append({"job_id": job_id, "type": f"camera_{cam_num}"})
            statistics_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "start_time": time.time(),
                "camera": f"Cam{cam_num}",
                "parent_job_id": parent_job_id,
            }

            # Start thread
            thread = threading.Thread(
                target=compute_statistics_for_camera,
                args=(
                    base_dir,
                    cam_num,
                    False,  # use_merged
                    image_count,
                    type_name,
                    endpoint,
                    vector_format,
                    job_id,
                ),
            )
            thread.daemon = True
            thread.start()

        # Store parent job
        statistics_jobs[parent_job_id] = {
            "status": "running",
            "sub_jobs": sub_jobs,
            "start_time": time.time(),
        }

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "status": "starting",
            "message": f"Statistics calculation started for {len(cameras)} camera(s)" + 
                      (" and merged data" if include_merged else ""),
        })

    except Exception as e:
        logger.error(f"Error starting statistics calculation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@statistics_bp.route("/statistics/status/<job_id>", methods=["GET"])
def get_statistics_status(job_id):
    """Get statistics calculation job status"""
    if job_id not in statistics_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = statistics_jobs[job_id].copy()

    # If this is a parent job, aggregate sub-job status
    if "sub_jobs" in job_data:
        sub_job_statuses = []
        all_completed = True
        any_failed = False
        total_progress = 0

        for sub_job in job_data["sub_jobs"]:
            sub_id = sub_job["job_id"]
            if sub_id in statistics_jobs:
                sub_status = statistics_jobs[sub_id].copy()
                sub_status["type"] = sub_job["type"]
                sub_job_statuses.append(sub_status)
                
                if sub_status["status"] != "completed":
                    all_completed = False
                if sub_status["status"] == "failed":
                    any_failed = True
                
                total_progress += sub_status.get("progress", 0)

        job_data["sub_job_statuses"] = sub_job_statuses
        job_data["overall_progress"] = total_progress / max(1, len(sub_job_statuses))
        
        if any_failed:
            job_data["status"] = "failed"
        elif all_completed:
            job_data["status"] = "completed"
        else:
            job_data["status"] = "running"

    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed

        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100)
            job_data["estimated_remaining"] = estimated_total - elapsed

    return jsonify(job_data)
