#!/usr/bin/env python3
"""
instantaneous_statistics.py

Production module for computing instantaneous PIV statistics.
- Computes mean velocity fields, Reynolds stresses, TKE, vorticity, divergence
- Computes per-frame instantaneous statistics (fluctuations, gamma functions)
- Uses ProcessPoolExecutor for frame-level parallelism
- Can be run from command line or called from GUI

Pattern matches: planar_calibration_production.py
"""

import concurrent.futures
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.io import savemat

matplotlib.use("Agg")

from pivtools_core.vector_loading import (
    find_valid_piv_runs,
    load_coords_from_directory,
    load_vectors_from_directory,
)

# ===================== CONFIGURATION =====================

# SOURCE_DIR: Root directory containing vector data
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing"

# BASE_DIR: Output directory for statistics
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing"

# Processing parameters
CAMERA_NUMS = [1]
NUM_FRAME_PAIRS = 100
VECTOR_FORMAT = "%05d.mat"
TYPE_NAME = "instantaneous"
USE_MERGED = False

# Statistics to compute
REQUESTED_STATISTICS = [
    "mean_velocity",
    "mean_vorticity",
    "mean_divergence",
    "reynolds_stress",
    "normal_stress",
    "mean_tke",
    "inst_velocity",
    "inst_fluctuations",
    "inst_vorticity",
    "inst_divergence",
]

# LOGGING SETUP
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===================== HELPER FUNCTIONS =====================


def create_disk_structuring_element(radius: int):
    """
    Create a disk structuring element and return neighbor offsets.
    Returns I, J offset arrays similar to MATLAB's strel('disk', radius).
    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    I, J = np.where(mask)
    I = I - radius
    J = J - radius
    return I, J


def nansurround(arr, d):
    """Pad array with NaN values of width d on all sides."""
    if arr.ndim == 2:
        padded = np.pad(arr, d, mode="constant", constant_values=np.nan)
    else:
        # For 3D arrays, pad only first two dimensions
        pad_width = [(d, d), (d, d)] + [(0, 0)] * (arr.ndim - 2)
        padded = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
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
    Optimized for memory: uses iterative summation instead of stacking.
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    # Initialize accumulators
    sum_u = np.zeros(s, dtype=np.float64)
    sum_v = np.zeros(s, dtype=np.float64)
    count_u = np.zeros(s, dtype=np.float64)
    count_v = np.zeros(s, dtype=np.float64)

    for i_offset, j_offset in zip(I, J):
        # Get shifted arrays
        u_shifted = np.roll(u, (i_offset, j_offset), axis=(0, 1))
        v_shifted = np.roll(v, (i_offset, j_offset), axis=(0, 1))

        # Determine valid (non-NaN) locations
        valid_u = ~np.isnan(u_shifted)
        valid_v = ~np.isnan(v_shifted)

        # Accumulate sums and counts in-place
        sum_u[valid_u] += u_shifted[valid_u]
        count_u[valid_u] += 1

        sum_v[valid_v] += v_shifted[valid_v]
        count_v[valid_v] += 1

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        u_mean = sum_u / count_u
        v_mean = sum_v / count_v

    # Create ROI mask (same as original logic)
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
    Optimized for memory: uses iterative summation.
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    # Get center point indices
    i0, j0 = s[0] // 2, s[1] // 2
    ind0 = np.ravel_multi_index((i0, j0), s)
    IN = np.ravel_multi_index((I + i0, J + j0), s)

    # Compute angles A (1D array of length N_neighbors)
    x_flat = x.ravel()
    y_flat = y.ravel()
    A = np.arctan2(y_flat[IN] - y_flat[ind0], x_flat[IN] - x_flat[ind0])

    # Pad arrays
    U = nansurround(u, d)
    V = nansurround(v, d)
    S = U.shape

    # Accumulator for the sine sum
    sum_s = np.zeros(S, dtype=np.float64)

    # Iterate over neighbors
    for k, (i_offset, j_offset) in enumerate(zip(I, J)):
        u_shifted = np.roll(U, (i_offset, j_offset), axis=(0, 1))
        v_shifted = np.roll(V, (i_offset, j_offset), axis=(0, 1))

        T = np.arctan2(v_shifted, u_shifted)

        # Calculate sine component
        # A[k] is a scalar for this specific neighbor offset
        sine_val = np.sin(A[k] - T)

        # Treat NaNs as zero for the sum (matching nansum behavior)
        sum_s += np.nan_to_num(sine_val)

    # Final calculation (normalized by number of neighbors, not valid count, per Graftieaux)
    G1 = -sum_s / len(I)
    G1 = unsurround(G1, d)

    return G1


def gamma2(x, y, u, v, d=10):
    """
    Compute gamma2 vortex detection criterion.
    Optimized for memory: uses iterative summation.
    """
    s = u.shape
    I, J = create_disk_structuring_element(d)

    # Get center point indices
    i0, j0 = s[0] // 2, s[1] // 2
    ind0 = np.ravel_multi_index((i0, j0), s)
    IN = np.ravel_multi_index((I + i0, J + j0), s)

    # Compute angles A
    x_flat = x.ravel()
    y_flat = y.ravel()
    A = np.arctan2(y_flat[IN] - y_flat[ind0], x_flat[IN] - x_flat[ind0])

    # Pad arrays
    U = nansurround(u, d)
    V = nansurround(v, d)
    S = U.shape

    # Compute local velocity
    Vp_raw = localvel(u, v, d)  # Returns (H,W,2)
    Vp_u = nansurround(Vp_raw[..., 0], d)
    Vp_v = nansurround(Vp_raw[..., 1], d)

    # Accumulator
    sum_s = np.zeros(S, dtype=np.float64)

    for k, (i_offset, j_offset) in enumerate(zip(I, J)):
        u_shifted = np.roll(U, (i_offset, j_offset), axis=(0, 1))
        v_shifted = np.roll(V, (i_offset, j_offset), axis=(0, 1))

        # Subtract local convective velocity
        u_rel = u_shifted - Vp_u
        v_rel = v_shifted - Vp_v

        T = np.arctan2(v_rel, u_rel)
        sine_val = np.sin(A[k] - T)

        # Accumulate
        sum_s += np.nan_to_num(sine_val)

    G2 = -sum_s / len(I)
    G2 = unsurround(G2, d)

    return G2


# ===================== PROCESSOR CLASS =====================


class VectorStatisticsProcessor:
    """
    Production class for computing PIV statistics.

    Designed for:
    - GUI integration with progress callbacks
    - Command-line execution via __main__
    - Frame-level parallelism using ProcessPoolExecutor
    """

    # Valid statistics that can be requested
    VALID_STATISTICS = {
        "mean_velocity": ["ux", "uy"],
        "mean_vorticity": ["vorticity_mean"],
        "mean_divergence": ["divergence_mean"],
        "reynolds_stress": ["uv"],
        "normal_stress": ["uu", "vv"],
        "mean_tke": ["tke_mean"],
        "inst_velocity": ["u", "v"],
        "inst_fluctuations": ["u_prime", "v_prime"],
        "inst_vorticity": ["vorticity_inst"],
        "inst_divergence": ["divergence_inst"],
        "inst_gamma": ["gamma1", "gamma2"],
    }

    def __init__(
        self,
        data_dir: Path,
        base_dir: Path,
        num_frame_pairs: int,
        vector_format: str,
        type_name: str = "instantaneous",
        use_merged: bool = False,
        camera: int = 1,
    ):
        """
        Initialize the statistics processor.

        Parameters
        ----------
        data_dir : Path
            Directory containing vector .mat files
        base_dir : Path
            Base directory for output
        num_frame_pairs : int
            Number of frame pairs to process
        vector_format : str
            Printf-style format for vector files (e.g., "%05d.mat")
        type_name : str
            Type name for output directory structure
        use_merged : bool
            Whether processing merged stereo data
        camera : int
            Camera number (1-based)
        """
        self.data_dir = Path(data_dir)
        self.base_dir = Path(base_dir)
        self.num_frame_pairs = num_frame_pairs
        self.vector_format = vector_format
        self.type_name = type_name
        self.use_merged = use_merged
        self.camera = camera

        # Determine camera folder name
        self.cam_folder = "Merged" if use_merged else f"Cam{camera}"

        # Setup output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create output directory structure."""
        # Stats output: base_dir/statistics/{type_name}/{cam_folder}/
        self.stats_dir = self.base_dir / "statistics" / self.type_name / self.cam_folder
        self.mean_stats_dir = self.stats_dir / "mean_stats"
        self.figures_dir = self.stats_dir / "figures"
        self.inst_stats_dir = self.stats_dir / "instantaneous_stats"

    def process(
        self,
        requested_statistics: Optional[list] = None,
        save_figures: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        """
        Process statistics for a single camera/merged data.

        Uses ProcessPoolExecutor for frame-level parallelism.

        Parameters
        ----------
        requested_statistics : list, optional
            List of statistic names to compute. If None, compute all.
        save_figures : bool
            Whether to save visualization figures
        progress_callback : callable, optional
            Function called with progress percentage (0-100)

        Returns
        -------
        dict
            success: bool
            output_file: str (path to saved results)
            error: str (if failed)
            stats: dict with computed statistics summary
        """
        try:
            logger.info(f"[Statistics] Starting for {self.cam_folder}")

            # Determine which statistics to compute
            if requested_statistics is None or len(requested_statistics) == 0:
                active_stats = set(self.VALID_STATISTICS.keys())
            else:
                active_stats = set(requested_statistics)

            logger.info(f"[Statistics] Active statistics: {active_stats}")

            if progress_callback:
                progress_callback(5)

            # Check data directory exists
            if not self.data_dir.exists():
                return {"success": False, "error": f"Data directory not found: {self.data_dir}"}

            # Find valid runs using centralized utility
            first_file = self.data_dir / (self.vector_format % 1)
            if not first_file.exists():
                return {"success": False, "error": f"No vector files found in {self.data_dir}"}

            validation_result = find_valid_piv_runs(first_file, one_based=True)
            valid_runs = validation_result.valid_runs

            if not valid_runs:
                return {"success": False, "error": f"No valid runs found in {self.data_dir}"}

            logger.info(f"[Statistics] Found {len(valid_runs)} valid runs: {valid_runs}")

            if progress_callback:
                progress_callback(10)

            # Create output directories
            self.mean_stats_dir.mkdir(parents=True, exist_ok=True)
            if save_figures:
                self.figures_dir.mkdir(parents=True, exist_ok=True)

            # Load coordinates
            coords_x_list, coords_y_list = load_coords_from_directory(
                self.data_dir, runs=valid_runs
            )

            if progress_callback:
                progress_callback(15)

            # Determine computation flags
            calc_flags = self._determine_calc_flags(active_stats)

            # Phase 1: Compute mean statistics
            mean_results = self._compute_mean_statistics(
                valid_runs, coords_x_list, coords_y_list, calc_flags, progress_callback
            )

            if progress_callback:
                progress_callback(50)

            # Phase 2: Compute instantaneous statistics (if requested)
            should_save_inst = self._should_compute_instantaneous(active_stats)
            if should_save_inst:
                self._compute_instantaneous_statistics(
                    valid_runs,
                    mean_results,
                    coords_x_list,
                    coords_y_list,
                    calc_flags,
                    progress_callback,
                )

            if progress_callback:
                progress_callback(75)

            # Phase 3: Save figures (if requested)
            if save_figures:
                self._save_figures(
                    valid_runs,
                    mean_results,
                    coords_x_list,
                    coords_y_list,
                    active_stats,
                )

            if progress_callback:
                progress_callback(85)

            # Phase 4: Save results to mat file
            output_file = self._save_results(valid_runs, mean_results, coords_x_list, coords_y_list)

            if progress_callback:
                progress_callback(100)

            logger.info(f"[Statistics] Completed successfully for {self.cam_folder}")

            return {
                "success": True,
                "output_file": str(output_file),
                "num_runs": len(valid_runs),
                "stats": list(active_stats),
            }

        except Exception as e:
            logger.error(f"[Statistics] Error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _determine_calc_flags(self, active_stats: set) -> dict:
        """Determine which calculations are needed based on requested statistics."""
        calc_reynolds = "reynolds_stress" in active_stats
        calc_normal = "normal_stress" in active_stats
        calc_tke = "mean_tke" in active_stats

        # Second moments needed for reynolds, normal stress, or tke
        calc_second_moments = calc_reynolds or calc_normal or calc_tke

        return {
            "calc_second_moments": calc_second_moments,
            "calc_mean_tke": calc_tke,
            "calc_mean_divergence": "mean_divergence" in active_stats,
            "calc_mean_vorticity": "mean_vorticity" in active_stats,
            "save_inst_velocity": "inst_velocity" in active_stats,
            "save_inst_fluctuations": "inst_fluctuations" in active_stats,
            "calc_inst_vorticity": "inst_vorticity" in active_stats,
            "calc_inst_divergence": "inst_divergence" in active_stats,
            "calc_inst_gamma": "inst_gamma" in active_stats,
        }

    def _should_compute_instantaneous(self, active_stats: set) -> bool:
        """Check if any instantaneous statistics are requested."""
        inst_stats = {
            "inst_velocity",
            "inst_fluctuations",
            "inst_vorticity",
            "inst_divergence",
            "inst_gamma",
        }
        return bool(active_stats & inst_stats)

    def _compute_mean_statistics(
        self,
        valid_runs: list,
        coords_x_list: list,
        coords_y_list: list,
        calc_flags: dict,
        progress_callback: Optional[Callable[[int], None]],
    ) -> dict:
        """
        Compute mean statistics for all valid runs.

        Returns dict with arrays for each statistic keyed by run number.
        """
        logger.info("[Statistics] Computing mean statistics...")

        # Create minimal config for vector loading
        class MinimalConfig:
            def __init__(self, num_frame_pairs, vector_format, piv_chunk_size=100):
                self.num_frame_pairs = num_frame_pairs
                self.vector_format = vector_format
                self.piv_chunk_size = piv_chunk_size

        config = MinimalConfig(self.num_frame_pairs, self.vector_format)

        results = {}
        total_runs = len(valid_runs)

        for idx, run_num in enumerate(valid_runs):
            logger.info(f"[Statistics] Processing run {run_num} ({idx + 1}/{total_runs})")

            cx = coords_x_list[idx] if idx < len(coords_x_list) else None
            cy = coords_y_list[idx] if idx < len(coords_y_list) else None

            run_result = self._process_single_run(run_num, config, cx, cy, calc_flags)
            results[run_num] = run_result

            if progress_callback:
                progress = 15 + int((idx + 1) / total_runs * 35)  # 15-50%
                progress_callback(progress)

        return results

    def _process_single_run(
        self,
        run_num: int,
        config,
        cx: Optional[np.ndarray],
        cy: Optional[np.ndarray],
        calc_flags: dict,
    ) -> dict:
        """Process statistics for a single run."""
        # Load vectors for this run
        arr_run = load_vectors_from_directory(self.data_dir, config, runs=[run_num])
        # Shape: (N_files, 1, 3_or_4, H, W)
        arr_run = np.array(arr_run)[:, 0, :, :, :]  # (N_files, 3_or_4, H, W)

        stereo = arr_run.shape[1] >= 4

        # Extract components
        ux = arr_run[:, 0, :, :]  # (N, H, W)
        uy = arr_run[:, 1, :, :]
        uz = None
        if stereo:
            uz = arr_run[:, 2, :, :]
            bmask = arr_run[:, 3, :, :]
        else:
            bmask = arr_run[:, 2, :, :]

        # Compute mean statistics
        mean_ux = np.nanmean(ux, axis=0)
        mean_uy = np.nanmean(uy, axis=0)
        b_mask = bmask[0]

        result = {
            "run_num": run_num,
            "stereo": stereo,
            "mean_ux": mean_ux,
            "mean_uy": mean_uy,
            "b_mask": b_mask,
            "n_frames": ux.shape[0],
        }

        # Second moments for Reynolds stresses
        if calc_flags.get("calc_second_moments", False):
            E_ux2 = np.nanmean(ux**2, axis=0)
            E_uy2 = np.nanmean(uy**2, axis=0)
            E_uxuy = np.nanmean(ux * uy, axis=0)

            result["uu"] = E_ux2 - mean_ux**2
            result["uv"] = E_uxuy - (mean_ux * mean_uy)
            result["vv"] = E_uy2 - mean_uy**2

            if stereo and uz is not None:
                mean_uz = np.nanmean(uz, axis=0)
                E_uz2 = np.nanmean(uz**2, axis=0)
                E_uxuz = np.nanmean(ux * uz, axis=0)
                E_uyuz = np.nanmean(uy * uz, axis=0)

                result["mean_uz"] = mean_uz
                result["uw"] = E_uxuz - (mean_ux * mean_uz)
                result["vw"] = E_uyuz - (mean_uy * mean_uz)
                result["ww"] = E_uz2 - mean_uz**2
        elif stereo and uz is not None:
            result["mean_uz"] = np.nanmean(uz, axis=0)

        # TKE
        if calc_flags.get("calc_mean_tke", False) and "uu" in result:
            if stereo and "ww" in result:
                result["tke"] = 0.5 * (result["uu"] + result["vv"] + result["ww"])
            else:
                result["tke"] = 0.5 * (result["uu"] + result["vv"])

        # Divergence
        if calc_flags.get("calc_mean_divergence", False):
            if cx is not None and cy is not None:
                dx = np.gradient(cx, axis=1)
                dy = np.gradient(cy, axis=0)
                dudx = np.gradient(mean_ux, axis=1) / dx
                dvdy = np.gradient(mean_uy, axis=0) / dy
            else:
                dudx = np.gradient(mean_ux, axis=1)
                dvdy = np.gradient(mean_uy, axis=0)
            result["divergence"] = dudx + dvdy

        # Vorticity
        if calc_flags.get("calc_mean_vorticity", False):
            if cx is not None and cy is not None:
                dx = np.gradient(cx, axis=1)
                dy = np.gradient(cy, axis=0)
                dvdx = np.gradient(mean_uy, axis=1) / dx
                dudy = np.gradient(mean_ux, axis=0) / dy
            else:
                dvdx = np.gradient(mean_uy, axis=1)
                dudy = np.gradient(mean_ux, axis=0)
            result["vorticity"] = dvdx - dudy

        return result

    def _compute_instantaneous_statistics(
        self,
        valid_runs: list,
        mean_results: dict,
        coords_x_list: list,
        coords_y_list: list,
        calc_flags: dict,
        progress_callback: Optional[Callable[[int], None]],
    ):
        """
        Compute per-frame instantaneous statistics using parallel processing.
        """
        logger.info("[Statistics] Computing instantaneous statistics...")

        self.inst_stats_dir.mkdir(parents=True, exist_ok=True)

        # Build means dict for parallel workers
        means_dict = {}
        for run_num in valid_runs:
            if run_num in mean_results:
                res = mean_results[run_num]
                means_dict[run_num] = {
                    "mean_ux": res["mean_ux"],
                    "mean_uy": res["mean_uy"],
                    "stereo": res["stereo"],
                }
                if res["stereo"]:
                    means_dict[run_num]["mean_uz"] = res.get("mean_uz")

        # Get coordinates (use first run's coords)
        cx = coords_x_list[0] if coords_x_list else None
        cy = coords_y_list[0] if coords_y_list else None

        # Find all frame files
        glob_pattern = re.sub(r"%[0-9]*[diuoxXfFeEgG]", "*", self.vector_format)
        frame_files = sorted(list(self.data_dir.glob(glob_pattern)))

        if not frame_files:
            logger.warning("[Statistics] No frame files found for instantaneous processing")
            return

        n_frames = len(frame_files)
        logger.info(f"[Statistics] Processing {n_frames} frames with parallelism...")

        # Process frames in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {}
            for i, file_path in enumerate(frame_files):
                out_name = f"{i + 1:05d}.mat"
                out_path = self.inst_stats_dir / out_name

                future = executor.submit(
                    _process_frame_parallel,
                    str(file_path),
                    str(out_path),
                    means_dict,
                    calc_flags,
                    cx,
                    cy,
                )
                futures[future] = i

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if progress_callback and completed % 10 == 0:
                        progress = 50 + int(completed / n_frames * 25)  # 50-75%
                        progress_callback(progress)
                except Exception as e:
                    logger.error(f"Frame processing failed: {e}")

        logger.info(f"[Statistics] Completed {completed}/{n_frames} frames")

    def _save_figures(
        self,
        valid_runs: list,
        mean_results: dict,
        coords_x_list: list,
        coords_y_list: list,
        active_stats: set,
    ):
        """Save visualization figures for mean statistics."""
        from pivtools_gui.plotting.plot_maker import make_scalar_settings, plot_scalar_field

        logger.info("[Statistics] Generating figures...")

        # Create minimal config for plotting
        class PlotConfig:
            plot_fontsize = 12
            plot_title_fontsize = 14
            plot_save_extension = ".png"
            plot_save_pickle = False

        config = PlotConfig()

        for idx, run_num in enumerate(valid_runs):
            res = mean_results.get(run_num)
            if not res:
                continue

            cx = coords_x_list[idx] if idx < len(coords_x_list) else None
            cy = coords_y_list[idx] if idx < len(coords_y_list) else None
            mask = np.asarray(res["b_mask"]).astype(bool)

            def save_field(field_data, var_name, units, cmap=None, symmetric=True):
                if field_data is None:
                    return

                save_base = self.figures_dir / f"Run_{run_num}_{var_name}"

                settings = make_scalar_settings(
                    config,
                    variable=var_name,
                    run_label=run_num,
                    save_basepath=save_base,
                    variable_units=units,
                    coords_x=cx,
                    coords_y=cy,
                    cmap=cmap,
                    symmetric_around_zero=symmetric,
                )

                fig, ax, im = plot_scalar_field(field_data, mask, settings)
                fig.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

            # Plot requested statistics
            if "mean_velocity" in active_stats:
                save_field(res["mean_ux"], "Mean_Ux", "m/s", symmetric=True)
                save_field(res["mean_uy"], "Mean_Uy", "m/s", symmetric=True)
                if res.get("stereo") and "mean_uz" in res:
                    save_field(res["mean_uz"], "Mean_Uz", "m/s", symmetric=True)

            if "mean_vorticity" in active_stats and "vorticity" in res:
                save_field(res["vorticity"], "Mean_Vorticity", "1/s", cmap="seismic", symmetric=True)

            if "mean_divergence" in active_stats and "divergence" in res:
                save_field(res["divergence"], "Mean_Divergence", "1/s", cmap="seismic", symmetric=True)

            if "mean_tke" in active_stats and "tke" in res:
                save_field(res["tke"], "Mean_TKE", "m^2/s^2", cmap="viridis", symmetric=False)

            if "reynolds_stress" in active_stats and "uv" in res:
                save_field(res["uv"], "Reynolds_Stress_uv", "m^2/s^2", cmap="seismic", symmetric=True)

    def _save_results(
        self,
        valid_runs: list,
        mean_results: dict,
        coords_x_list: list,
        coords_y_list: list,
    ) -> Path:
        """Save results to mat file."""
        logger.info("[Statistics] Saving results...")

        # Check stereo from first result
        first_res = mean_results[valid_runs[0]]
        stereo = first_res["stereo"]

        # Build piv_result structured array
        max_run = max(valid_runs)
        dt_fields = [
            ("ux", object),
            ("uy", object),
            ("b_mask", object),
            ("uu", object),
            ("uv", object),
            ("vv", object),
            ("tke", object),
            ("divergence", object),
            ("vorticity", object),
        ]
        if stereo:
            dt_fields.extend([
                ("uz", object),
                ("uw", object),
                ("vw", object),
                ("ww", object),
            ])

        dt = np.dtype(dt_fields)
        piv_result = np.empty((max_run,), dtype=dt)

        # Initialize with empty arrays
        empty = np.empty((0, 0))
        for i in range(max_run):
            for field, _ in dt_fields:
                piv_result[i][field] = empty

        # Fill valid runs
        for run_num in valid_runs:
            res = mean_results[run_num]
            idx = run_num - 1

            piv_result[idx]["ux"] = res["mean_ux"]
            piv_result[idx]["uy"] = res["mean_uy"]
            piv_result[idx]["b_mask"] = res["b_mask"]

            for field in ["uu", "uv", "vv", "tke", "divergence", "vorticity"]:
                if field in res and res[field] is not None:
                    piv_result[idx][field] = res[field]

            if stereo:
                for field in ["uz", "uw", "vw", "ww"]:
                    key = "mean_uz" if field == "uz" else field
                    if key in res and res[key] is not None:
                        piv_result[idx][field] = res[key]

        # Build coordinates array
        dt_coords = np.dtype([("x", object), ("y", object)])
        coordinates = np.empty((max_run,), dtype=dt_coords)

        for i in range(max_run):
            coordinates[i]["x"] = empty
            coordinates[i]["y"] = empty

        for list_idx, run_num in enumerate(valid_runs):
            idx = run_num - 1
            if list_idx < len(coords_x_list):
                coordinates[idx]["x"] = coords_x_list[list_idx]
                coordinates[idx]["y"] = coords_y_list[list_idx]

        # Build metadata
        meta_dict = {
            "use_merged": self.use_merged,
            "camera": self.cam_folder,
            "selected_passes": valid_runs,
            "n_passes": len(valid_runs),
            "stereo": stereo,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        out_file = self.mean_stats_dir / "mean_stats.mat"
        savemat(out_file, {
            "piv_result": piv_result,
            "coordinates": coordinates,
            "meta": meta_dict,
        })

        logger.info(f"[Statistics] Saved to {out_file}")
        return out_file


# ===================== PARALLEL WORKER FUNCTION =====================


def _process_frame_parallel(
    file_path: str,
    out_path: str,
    means_dict: dict,
    calc_flags: dict,
    cx: Optional[np.ndarray],
    cy: Optional[np.ndarray],
):
    """
    Process a single frame for instantaneous statistics.
    Designed to run in a separate process.
    """
    try:
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)

        if "piv_result" not in mat_data:
            return

        piv_result_in = mat_data["piv_result"]

        # Ensure array
        if not isinstance(piv_result_in, np.ndarray):
            piv_result_in = np.array([piv_result_in])
        elif piv_result_in.ndim == 0:
            piv_result_in = np.array([piv_result_in.item()])

        if piv_result_in.ndim > 1:
            piv_result_in = piv_result_in.flatten()

        n_runs = piv_result_in.size
        results_list = []

        for i in range(n_runs):
            run_num = i + 1
            run_data = {}
            src_obj = piv_result_in[i]

            # Copy existing fields
            if hasattr(src_obj, "_fieldnames"):
                for field in src_obj._fieldnames:
                    run_data[field] = getattr(src_obj, field)

            if run_num in means_dict:
                means = means_dict[run_num]
                mean_ux = means["mean_ux"]
                mean_uy = means["mean_uy"]
                stereo = means["stereo"]

                u_t = run_data.get("ux", np.nan)
                v_t = run_data.get("uy", np.nan)
                w_t = run_data.get("uz", None) if stereo else None

                # Fluctuations
                if calc_flags.get("save_inst_fluctuations", False):
                    run_data["u_prime"] = u_t - mean_ux
                    run_data["v_prime"] = v_t - mean_uy
                    if stereo and w_t is not None:
                        mean_uz = means.get("mean_uz")
                        if mean_uz is not None:
                            run_data["w_prime"] = w_t - mean_uz

                # Gradients for vorticity/divergence/gamma
                need_grads = (
                    calc_flags.get("calc_inst_divergence", False)
                    or calc_flags.get("calc_inst_vorticity", False)
                    or calc_flags.get("calc_inst_gamma", False)
                )

                if need_grads and isinstance(u_t, np.ndarray):
                    if cx is not None and cy is not None:
                        dx = np.gradient(cx, axis=1)
                        dy = np.gradient(cy, axis=0)
                        dudx = np.gradient(u_t, axis=1) / dx
                        dudy = np.gradient(u_t, axis=0) / dy
                        dvdx = np.gradient(v_t, axis=1) / dx
                        dvdy = np.gradient(v_t, axis=0) / dy
                    else:
                        y_grid, x_grid = np.meshgrid(
                            np.arange(u_t.shape[1]), np.arange(u_t.shape[0])
                        )
                        cx, cy = x_grid, y_grid
                        dudx = np.gradient(u_t, axis=1)
                        dudy = np.gradient(u_t, axis=0)
                        dvdx = np.gradient(v_t, axis=1)
                        dvdy = np.gradient(v_t, axis=0)

                    if calc_flags.get("calc_inst_divergence", False):
                        run_data["divergence"] = dudx + dvdy

                    if calc_flags.get("calc_inst_vorticity", False):
                        run_data["vorticity"] = dvdx - dudy

                    if calc_flags.get("calc_inst_gamma", False):
                        run_data["gamma1"] = gamma1(cx, cy, u_t, v_t, d=10)
                        run_data["gamma2"] = gamma2(cx, cy, u_t, v_t, d=10)

            results_list.append(run_data)

        if not results_list:
            return

        # Build output structured array
        all_keys = set()
        for res in results_list:
            all_keys.update(res.keys())

        dt_fields = [(k, object) for k in sorted(list(all_keys))]
        dt = np.dtype(dt_fields)

        out_piv_result = np.empty((n_runs,), dtype=dt)

        for i, res in enumerate(results_list):
            for k in all_keys:
                out_piv_result[i][k] = res.get(k, np.empty((0, 0)))

        savemat(out_path, {"piv_result": out_piv_result}, do_compression=True)

    except Exception as e:
        logger.error(f"Error processing frame {file_path}: {e}")
        raise


# ===================== MAIN =====================


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Vector Statistics Processing - Starting")
    logger.info("=" * 60)
    logger.info(f"Source: {SOURCE_DIR}")
    logger.info(f"Output: {BASE_DIR}")
    logger.info(f"Cameras: {CAMERA_NUMS}")
    logger.info(f"Frame pairs: {NUM_FRAME_PAIRS}")
    logger.info(f"Statistics: {REQUESTED_STATISTICS}")

    failed_cameras = []

    for camera_num in CAMERA_NUMS:
        logger.info(f"\nProcessing Camera {camera_num}...")

        try:
            # Build data directory path
            data_dir = Path(SOURCE_DIR) / f"Cam{camera_num}" / TYPE_NAME

            processor = VectorStatisticsProcessor(
                data_dir=data_dir,
                base_dir=Path(BASE_DIR),
                num_frame_pairs=NUM_FRAME_PAIRS,
                vector_format=VECTOR_FORMAT,
                type_name=TYPE_NAME,
                use_merged=USE_MERGED,
                camera=camera_num,
            )

            result = processor.process(
                requested_statistics=REQUESTED_STATISTICS,
                save_figures=True,
            )

            if result["success"]:
                logger.info(
                    f"Camera {camera_num} completed: {result['num_runs']} runs, "
                    f"output: {result['output_file']}"
                )
            else:
                logger.error(f"Camera {camera_num} failed: {result.get('error', 'Unknown error')}")
                failed_cameras.append(camera_num)

        except Exception as e:
            logger.error(f"Camera {camera_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cameras.append(camera_num)

    logger.info("=" * 60)
    if failed_cameras:
        logger.error(f"Processing failed for cameras: {failed_cameras}")
    else:
        logger.info("Vector statistics completed successfully for all cameras")
