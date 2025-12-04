import unittest
from pathlib import Path
from unittest import skip
from unittest.mock import MagicMock, patch

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.distributed import get_worker
from scipy.io import loadmat

from pivtools_cli.config import Config
from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.tests.helpers import assert_arrays_close, compare_matrices


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

        inpaint_patcher = patch(
            "pivtools_cli.piv.piv_backend.cpu_instantaneous.InstantaneousCorrelatorCPU._inpaint_nans_biharm",
            new=_fake_inpaint_biharm,
        )
        self._mock_inpaint = inpaint_patcher.start()
        fake_worker = MagicMock()
        fake_worker.name = "worker-123"
        self._worker_patcher = patch(
            "pivtools_cli.piv.piv_backend.cpu_instantaneous.get_worker",
            return_value=fake_worker,
        )
        self._worker_patcher.start()

    def test_peak_height(self):
        """Test peak heights against MATLAB results"""
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, len(self.config.window_sizes) + 1):
            peak_height_mat_file = self.base_path / f"MATLAB_peak_height_{i}.mat"
            with h5py.File(peak_height_mat_file, "r") as f:
                matlab_peaks = np.array(f["peak_height"], dtype=np.float32)

                matlab_peaks = np.transpose(matlab_peaks, (2, 1, 0))
                assert_arrays_close(
                    self, matlab_peaks, piv_result.passes[i - 1].peak_mag, tol=0.0001
                )

    def test_ux_mat(self):
        """Test x displacement field against MATLAB results"""
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(2, len(self.config.window_sizes) + 1):
            ux_mat_file = self.base_path / f"MATLAB_ux_mat_{i}.mat"

            with h5py.File(ux_mat_file, "r") as f:
                matlab_ux_mat = np.array(f["ux_mat"], dtype=np.float32)
                matlab_ux_mat = np.transpose(matlab_ux_mat, (1, 0))
                print("UX", matlab_ux_mat[0], piv_result.passes[i - 1].ux_mat[0])
                assert_arrays_close(
                    self, matlab_ux_mat, piv_result.passes[i - 1].ux_mat, tol=0.0001
                )

    def test_uy_mat(self):
        """Test y displacement field against MATLAB results"""
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, 3):  # len(self.config.window_sizes)+1):
            uy_mat_file = self.base_path / f"MATLAB_uy_mat_{i}.mat"
            with h5py.File(uy_mat_file, "r") as f:
                matlab_uy_mat = np.array(f["uy_mat"], dtype=np.float32)
                matlab_uy_mat = np.transpose(matlab_uy_mat, (1, 0))
                compare_matrices(matlab_uy_mat, piv_result.passes[i - 1].uy_mat)
                assert_arrays_close(
                    self, matlab_uy_mat, piv_result.passes[i - 1].uy_mat, tol=1.001
                )

    def test_nan(self):
        """Test nan mask against MATLAB results"""
        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)

        for i in range(1, 3):  # self.MAX_PASS):  # len(self.config.window_sizes)+1):
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
                for r, c in idx[:]:
                    print(
                        f"({r},{c}): MATLAB={matlab_nan[r,c]}, Python={python_nan[r,c]}"
                    )
                raise

    def test_q_mat(self):
        """Test Q matrix against MATLAB results"""
        self.config.data["piv"]["secondary_peak"] = True
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, len(self.config.window_sizes) + 1):
            q_mat_file = self.base_path / f"MATLAB_Qmat_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_q_mat = np.array(f["Q_mat"], dtype=np.float32)
                matlab_q_mat = np.transpose(matlab_q_mat, (0, 2, 1))
                assert_arrays_close(
                    self, matlab_q_mat, piv_result.passes[i - 1].Q_mat, tol=0.1
                )

    def test_delta_ab_old(self):
        """Test delta_ab_old matrix against MATLAB results"""

        self.config.data["piv"]["secondary_peak"] = False
        cpu_backend = make_correlator_backend(config=self.config)
        piv_result = cpu_backend.correlate_batch(self.img_pair, self.config)
        for i in range(1, len(self.config.window_sizes) + 1):
            q_mat_file = self.base_path / f"MATLAB_delta_ab_old_padded_{i}.mat"
            with h5py.File(q_mat_file, "r") as f:
                matlab_delta_ab_old = np.array(f["delta_ab_old"], dtype=np.float32)
                matlab_delta_ab_old = np.transpose(matlab_delta_ab_old, (2, 1, 0))
                assert_arrays_close(
                    self,
                    matlab_delta_ab_old,
                    piv_result.passes[i - 1].predictor_field,
                    tol=0.01,
                )
