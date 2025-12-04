import unittest
from pathlib import Path
from unittest import skip
from unittest.mock import patch

import dask.array as da
import h5py
import numpy as np
import pytest
import tifffile
from scipy.io import loadmat

from pivtools_cli.config import Config
from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.tests.helpers import assert_arrays_close


def _fake_inpaint_biharm(self, x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=1.0)


class InstantaneousPIVTestCase(unittest.TestCase):

    def setUp(self):
        test_dir = Path(__file__).parent
        config_path = test_dir / "config.yaml"
        self.config = Config(config_path)
        self.config.data["paths"]["base_path"] = str(
            test_dir.parent / "data" / "instantaneous_piv"
        )
        camera_path = self.config.base_path / self.config.cameras[0]
        file_paths = [
            camera_path / (self.config.image_format % 1),
            camera_path / (self.config.image_format.replace("_A", "_B") % 1),
        ]

        self.img_pair = np.stack(
            [
                tifffile.imread(file_paths[0]).astype(self.config.image_dtype),
                tifffile.imread(file_paths[1]).astype(self.config.image_dtype),
            ],
            axis=0,
        )
        C, H, W = self.img_pair.shape
        self.img_pair = self.img_pair.reshape((1, C, H, W))
        self.base_path = Path(self.config.data["paths"]["base_path"]) / "Matlab"
        patcher = patch(
            "pivtools_cli.piv.piv_backend.cpu_instantaneous.InstantaneousCorrelatorCPU._inpaint_nans_biharm",
            new=_fake_inpaint_biharm,
        )

        self._mock_inpaint = patcher.start()

    #    def tearDown(self):
    #        self._mock_inpaint.stop()

    def test_x_peaks(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 2):  # len(self.config.window_sizes)):
            peak_x_mat_file = self.base_path / f"MATLAB_peak_loc_x_{i}.mat"

            with h5py.File(peak_x_mat_file, "r") as f:

                matlab_peaks = np.array(f["peak_loc_x"], dtype=np.float32)
                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                # compare_matrices(
                #    matlab_peaks[0], piv_result.passes[i - 1].peak_loc_x[0]
                # )
                assert_arrays_close(
                    self,
                    matlab_peaks,
                    piv_result.passes[i - 1].peak_loc_x,
                    tol=0.001,
                )

    def test_x_peaks_after_bulk(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # len(self.config.window_sizes) + 1):
            peak_x_mat_file = self.base_path / f"MATLAB_peak_loc_x_after_bulk_{i}.mat"

            with h5py.File(peak_x_mat_file, "r") as f:

                matlab_peaks = np.array(f["peak_loc_x"], dtype=np.float32)
                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                compare_matrices(
                    matlab_peaks[0], piv_result.passes[i - 1].peak_loc_x_after_bulk[0]
                )
                assert_arrays_close(
                    self,
                    matlab_peaks,
                    piv_result.passes[i - 1].peak_loc_x_after_bulk,
                    tol=0.001,
                )

    def test_y_peaks_after_bulk(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # len(self.config.window_sizes) + 1):
            peak_y_mat_file = self.base_path / f"MATLAB_peak_loc_y  _after_bulk_{i}.mat"

            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_peaks = np.array(f["peak_loc_y"], dtype=np.float32)
                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                compare_matrices(
                    matlab_peaks[0], piv_result.passes[i - 1].peak_loc_y_after_bulk[0]
                )
                assert_arrays_close(
                    self,
                    matlab_peaks,
                    piv_result.passes[i - 1].peak_loc_y_after_bulk,
                    tol=0.001,
                )

    def test_y_peaks(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, len(self.config.window_sizes) + 1):
            peak_y_mat_file = self.base_path / f"MATLAB_peak_loc_y_{i}.mat"
            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_peaks = np.array(f["peak_loc_y"], dtype=np.float32)
                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                assert_arrays_close(
                    self, matlab_peaks, piv_result.passes[i - 1].peak_loc_y, tol=0.0001
                )

    def test_a_prime(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, len(self.config.window_sizes) + 1):
            peak_y_mat_file = self.base_path / f"MATLAB_A_prime_{i}.mat"
            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_image_a = np.array(f["A_prime"], dtype=np.float32)
                compare_matrices(
                    matlab_image_a.T[0],
                    piv_result.passes[i - 1].image_a_prime[0],
                )
                assert_arrays_close(
                    self,
                    matlab_image_a.T,
                    piv_result.passes[i - 1].image_a_prime,
                    tol=0.01,
                )

    def test_a_prime_subset(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            peak_y_mat_file = self.base_path / f"MATLAB_a_prime_subset_{i}.mat"
            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_image_a = np.array(f["A_prime_subset"], dtype=np.float32)
                compare_matrices(
                    matlab_image_a.T[0],
                    piv_result.passes[i - 1].image_a_prime_subset[0],
                )
                assert_arrays_close(
                    self,
                    matlab_image_a.T,
                    piv_result.passes[i - 1].image_a_prime_subset,
                    tol=0.005,
                )

    def test_a(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            peak_y_mat_file = self.base_path / f"MATLAB_image_a_{i}.mat"
            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_image_a = np.array(f["A"], dtype=np.float32)
                compare_matrices(
                    matlab_image_a.T[2], piv_result.passes[i - 1].image_a_before[2]
                )
                assert_arrays_close(
                    self,
                    matlab_image_a.T,
                    piv_result.passes[i - 1].image_a_before,
                    tol=0.05,
                )

    def test_a_mesh(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            peak_y_mat_file = self.base_path / f"MATLAB_image_mesh_A_{i}.mat"
            with h5py.File(peak_y_mat_file, "r") as f:

                matlab_image_mesh_a = np.array(f["im_mesh_A"], dtype=np.float32)
                matlab_image_mesh_a = np.transpose(matlab_image_mesh_a, (2, 1, 0))
                print(matlab_image_mesh_a.shape)
                print(piv_result.passes[i - 1].im_mesh_A.shape)
                ii, jj = 10, 20
                print(
                    "Python mesh:",
                    piv_result.passes[i - 1].im_mesh_A[ii, jj, 0],
                    piv_result.passes[i - 1].im_mesh_A[ii, jj, 1],
                )
                print("MATLAB mesh:", matlab_image_mesh_a[ii, jj, :])
                compare_matrices(
                    matlab_image_mesh_a.T[2], piv_result.passes[i - 1].im_mesh_A[2] + 1
                )
                assert_arrays_close(
                    self,
                    matlab_image_mesh_a,
                    piv_result.passes[i - 1].im_mesh_A + 1,
                    tol=0.0005,
                )

    def test_peak_height(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, self.MAX_PASS):  # len(self.config.window_sizes)+1):
            peak_height_mat_file = self.base_path / f"MATLAB_peak_height_{i}.mat"
            with h5py.File(peak_height_mat_file, "r") as f:
                matlab_peaks = np.array(f["peak_height"], dtype=np.float32)

                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                print("PEAKS", matlab_peaks[0, :, :])
                assert_arrays_close(
                    self, matlab_peaks, piv_result.passes[i - 1].peak_mag, tol=0.0001
                )

    def test_ux_mat(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 4):  # len(self.config.window_sizes)+1):
            ux_mat_file = self.base_path / f"MATLAB_ux_mat_{i}.mat"

            with h5py.File(ux_mat_file, "r") as f:
                matlab_ux_mat = np.array(f["ux_mat"], dtype=np.float32)
                matlab_ux_mat = np.transpose(matlab_ux_mat, (1, 0))
                print("UX", matlab_ux_mat[0], piv_result.passes[i - 1].ux_mat[0])
                assert_arrays_close(
                    self, matlab_ux_mat, piv_result.passes[i - 1].ux_mat, tol=0.0001
                )

    def test_uy_mat(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 5):  # len(self.config.window_sizes)+1):
            uy_mat_file = self.base_path / f"MATLAB_uy_mat_{i}.mat"
            with h5py.File(uy_mat_file, "r") as f:
                matlab_uy_mat = np.array(f["uy_mat"], dtype=np.float32)
                matlab_uy_mat = np.transpose(matlab_uy_mat, (1, 0))
                compare_matrices(matlab_uy_mat, piv_result.passes[i - 1].uy_mat)
                assert_arrays_close(
                    self, matlab_uy_mat, piv_result.passes[i - 1].uy_mat, tol=1.001
                )

    def test_ux_mat_secondary(self):
        self.config.data["piv"]["secondary_peak"] = True
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # len(self.config.window_sizes)+1):
            ux_mat_file = self.base_path / f"MATLAB_ux_mat_{i}_secondary.mat"
            with h5py.File(ux_mat_file, "r") as f:
                matlab_ux_mat = np.array(f["ux_mat"], dtype=np.float32)
                matlab_ux_mat = np.transpose(matlab_ux_mat, (1, 0))
                assert_arrays_close(
                    self, matlab_ux_mat, piv_result.passes[i - 1].ux_mat, tol=0.0001
                )

    def test_uy_mat_secondary(self):
        self.config.data["piv"]["secondary_peak"] = True
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # len(self.config.window_sizes)+1):
            uy_mat_file = self.base_path / f"MATLAB_uy_mat_{i}_secondary.mat"
            with h5py.File(uy_mat_file, "r") as f:
                matlab_uy_mat = np.array(f["uy_mat"], dtype=np.float32)
                matlab_uy_mat = np.transpose(matlab_uy_mat, (1, 0))

                assert_arrays_close(
                    self, matlab_uy_mat, piv_result.passes[i - 1].uy_mat, tol=0.0001
                )
                assert True

    def test_nan(self):
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)

        for i in range(1, self.MAX_PASS):  # len(self.config.window_sizes)+1):
            nan_file = self.base_path / f"MATLAB_nan_{i}.mat"
            with h5py.File(nan_file, "r") as f:
                matlab_nan = np.array(f["nan_mask"], dtype=bool)

            matlab_nan = np.transpose(matlab_nan, (1, 0))
            python_nan = piv_result.passes[i - 1].nan_mask
            try:
                np.testing.assert_array_equal(matlab_nan, python_nan)
            except AssertionError:
                diff = matlab_nan != python_nan
                idx = np.argwhere(diff)

                print(f"nan_mask mismatch: {len(idx)} elements differ")
                for r, c in idx[:10]:
                    print(
                        f"({r},{c}): MATLAB={matlab_nan[r,c]}, Python={python_nan[r,c]}"
                    )
                raise

    def test_Q_mat_secondary(self):

        self.config.data["piv"]["secondary_peak"] = True
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_Qmat_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_q_mat = np.array(f["Q_mat"], dtype=np.float32)
                matlab_q_mat = np.transpose(matlab_q_mat, (0, 2, 1))
                assert_arrays_close(
                    self, matlab_q_mat, piv_result.passes[i - 1].Q_mat, tol=0.0001
                )

    def test_delta_ab_gauss(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_old_gauss_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_old = np.array(f["delta_ab_old"], dtype=np.float32)
                matlab_delta_ab_old = np.transpose(matlab_delta_ab_old, (2, 1, 0))
                print(matlab_delta_ab_old.shape)

                compare_matrices(
                    matlab_delta_ab_old[0],
                    piv_result.passes[i - 1].delta_ab_old_gauss[0],
                )
                assert_arrays_close(
                    self,
                    matlab_delta_ab_old,
                    piv_result.passes[i - 1].delta_ab_old_gauss,
                    tol=0.0001,
                )

    def test_delta_ab_dense(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_dense_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_dense = np.array(f["delta_ab_dense"], dtype=np.float32)
                matlab_delta_ab_dense = np.transpose(matlab_delta_ab_dense, (2, 1, 0))
                for numb in range(1):
                    compare_matrices(
                        matlab_delta_ab_dense[numb],
                        piv_result.passes[i - 1].delta_ab_dense_test[numb],
                    )

                assert_arrays_close(
                    self,
                    matlab_delta_ab_dense,
                    piv_result.passes[i - 1].delta_ab_dense_test,
                    tol=0.001,
                )

    def test_delta_ab_filt(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_filt_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_filt = np.array(f["delta_ab_filt"], dtype=np.float32)
                matlab_delta_ab_filt = np.transpose(matlab_delta_ab_filt, (2, 1, 0))
                assert_arrays_close(
                    self,
                    matlab_delta_ab_filt,
                    piv_result.passes[i - 1].delta_ab_filt,
                    tol=0.0001,
                )

    def test_delta_ab_pred_test(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_pred_test_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_pred = np.array(f["delta_ab_pred"], dtype=np.float32)
                matlab_delta_ab_pred = np.transpose(matlab_delta_ab_pred, (2, 1, 0))
                print(f"MATLAB shape {i}", matlab_delta_ab_pred.shape)
                # print("MATLAB", matlab_delta_ab_pred.shape)
                # print("PYTHON", piv_result.passes[i - 1].delta_ab_pred_test.shape)
                for numb in range(8):
                    compare_matrices(
                        matlab_delta_ab_pred[numb],
                        piv_result.passes[i - 1].delta_ab_pred_test[numb],
                    )
                assert_arrays_close(
                    self,
                    matlab_delta_ab_pred,
                    piv_result.passes[i - 1].delta_ab_pred_test,
                    tol=0.0001,
                )

    def test_delta_ab_old(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        print("PIV", piv_result.passes[0].delta_ab_old[:, 2, 0])
        for i in range(1, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_old_padded_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_old = np.array(f["delta_ab_old"], dtype=np.float32)
                matlab_delta_ab_old = np.transpose(matlab_delta_ab_old, (2, 1, 0))
                print("MATLAB shape", matlab_delta_ab_old.shape)
                # print("MATLAB", matlab_delta_ab_pred.shape)
                # print("PYTHON", piv_result.passes[i - 1].delta_ab_pred_test.shape)
                for numb in range(1):
                    compare_matrices(
                        matlab_delta_ab_old[numb],
                        piv_result.passes[i - 1].delta_ab_old[numb],
                    )
                print(matlab_delta_ab_old[:, 0, 0])
                print(matlab_delta_ab_old[:, 1, 0])
                print(matlab_delta_ab_old[:, 2, 0])

                print(piv_result.passes[i - 1].delta_ab_old[:, 0, 0])
                print(piv_result.passes[i - 1].delta_ab_old[:, 1, 0])
                print(piv_result.passes[i - 1].delta_ab_old[:, 2, 0])

                assert_arrays_close(
                    self,
                    matlab_delta_ab_old,
                    piv_result.passes[i - 1].delta_ab_old,
                    tol=0.01,
                )

    def test_delta_ab_old_not_padded(self):

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_old_not_padded_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_old = np.array(f["delta_ab_old"], dtype=np.float32)
                matlab_delta_ab_old = np.transpose(matlab_delta_ab_old, (2, 1, 0))
                print("MATLAB shape", matlab_delta_ab_old.shape)
                # print("MATLAB", matlab_delta_ab_pred.shape)
                # print("PYTHON", piv_result.passes[i - 1].delta_ab_pred_test.shape)
                for numb in range(3):
                    compare_matrices(
                        matlab_delta_ab_old[numb],
                        piv_result.passes[i - 1].delta_ab_old_not_padded[numb],
                    )
                assert_arrays_close(
                    self,
                    matlab_delta_ab_old,
                    piv_result.passes[i - 1].delta_ab_old_not_padded,
                    tol=0.0001,
                )


import numpy as np
import pandas as pd


def compare_matrices(matlab_uy, python_uy, precision=6):
    """
    Compare two UY matrices (MATLAB vs Python) and summarize differences.

    Parameters
    ----------
    matlab_uy : 2D array_like
        MATLAB UY matrix.
    python_uy : 2D array_like
        Python UY matrix (same shape as matlab_uy).
    precision : int
        Number of decimal places to print.

    Returns
    -------
    summary_df : pandas DataFrame
        A table with MATLAB, Python, Diff, and Relative Diff for each element.
    """
    if matlab_uy.shape != python_uy.shape:
        raise ValueError(
            f"Matrices must have the same shape: {matlab_uy.shape} != {python_uy.shape}"
        )

    # Flatten for element-wise comparison
    matlab_flat = matlab_uy.ravel()
    python_flat = python_uy.ravel()

    # Differences
    diff = python_flat - matlab_flat
    rel_diff = np.where(matlab_flat != 0, diff / matlab_flat, 0)

    # Prepare a DataFrame for easy inspection
    summary_df = pd.DataFrame(
        {
            "MATLAB": np.round(matlab_flat, precision),
            "Python": np.round(python_flat, precision),
            "Diff": np.round(diff, precision),
            "RelDiff (%)": np.round(100 * rel_diff, precision),
        }
    )

    # Summary stats
    stats = {
        "Mean Absolute Diff": np.mean(np.abs(diff)),
        "Max Absolute Diff": np.max(np.abs(diff)),
        "Min Absolute Diff": np.min(np.abs(diff)),
        "Mean Relative Diff (%)": np.mean(np.abs(rel_diff)) * 100,
        "Max Relative Diff (%)": np.max(np.abs(rel_diff)) * 100,
        "Min Relative Diff (%)": np.min(np.abs(rel_diff)) * 100,
    }

    print("=== Full Element-wise Comparison ===")
    pd.set_option("display.max_rows", None)

    # Show all columns
    pd.set_option("display.max_columns", None)
    print(summary_df)
    print("\n=== Summary Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    print()
    return summary_df, stats


# Example usage:
# summary, stats = compare_uy_matrices(matlab_uy_mat, piv_result.passes[i-1].uy_mat)
