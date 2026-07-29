#!/usr/bin/env python3
"""
test_multipass_convergence.py

End-to-end integration tests for multipass PIV convergence using synthetic
Poiseuille flow images. Tests both instantaneous (single pair) and ensemble
(20 pairs, mixed std/single mode) pipelines.

Ensemble tests run the production fit_method ('kspace', the joint LM fitter —
the only selectable method) and verify it produces converged results.

Key verification:
  - Predictor field converges to measured velocity (proves warping works)
  - Recovered velocity matches analytical Poiseuille profile
  - Repeated final pass gives same answer (steady-state convergence)

Usage:
    cd unit-tests && python generate_poiseuille_ensemble.py   # generate data first
    pytest unit-tests/test_multipass_convergence.py -v -s
    pytest unit-tests/test_multipass_convergence.py -v -s --make-figures
"""

import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_cli.piv.save_results import (
    get_ensemble_output_path,
    get_output_path,
    load_ensemble_result,
)
from pivtools_core.config import Config, reload_config
from pivtools_core.instantaneous import run_instantaneous_piv
from pivtools_core.vector_loading import read_mat_contents

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UNIT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DATA_DIR = UNIT_DIR / "poiseuille_ensemble"
PARAMS_PATH = ENSEMBLE_DATA_DIR / "params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _robust_rmtree(path):
    """Remove directory tree, handling Windows/OneDrive file locking."""

    def _onerror(func, fpath, _exc_info):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    for attempt in range(3):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            time.sleep(0.5)
    shutil.rmtree(path, ignore_errors=True)


def _load_params():
    """Load ground truth parameters from the generated data."""
    with open(PARAMS_PATH) as f:
        return json.load(f)


def _poiseuille_ux_pixels(row_centers, image_height, u_max):
    """Analytical Poiseuille ux displacement at window center rows (pixels).

    row_centers : 0-based pixel row indices of window centers
    image_height : total image height in pixels
    u_max : peak displacement in pixels

    Returns ux in pixels (positive = rightward).
    """
    y_norm = 2.0 * row_centers / image_height - 1.0
    return u_max * (1.0 - y_norm**2)


def _interior_mask(grid_shape, exclude_fraction=0.15):
    """Create mask excluding near-wall rows (top/bottom exclude_fraction)."""
    nrows = grid_shape[0]
    margin = max(1, int(nrows * exclude_fraction))
    mask = np.zeros(grid_shape, dtype=bool)
    mask[margin:-margin, :] = True
    return mask


def _run_ensemble_piv(fit_method):
    """
    Run 3-pass ensemble PIV on 20 Poiseuille image pairs.

    Passes: 32×32 std -> 16×16 single -> 16×16 single
    Parametrized by fit_method to test both gaussian and kspace.

    Returns dict with per-pass results and ground truth.
    """
    params = _load_params()
    workspace = ENSEMBLE_DATA_DIR / f"test_workspace_{fit_method}"

    if workspace.exists():
        _robust_rmtree(workspace)
    workspace.mkdir(exist_ok=True)

    source_dir = workspace / "source"
    source_dir.mkdir(exist_ok=True)
    base_path = workspace / "output"
    base_path.mkdir(exist_ok=True)

    # Copy all image pairs
    num_pairs = params["num_pairs"]
    for idx in range(1, num_pairs + 1):
        for suffix in ("_A.tif", "_B.tif"):
            name = f"B{idx:05d}{suffix}"
            shutil.copy2(ENSEMBLE_DATA_DIR / name, source_dir / name)

    # Config: 3-pass ensemble — 32 std, 16 single, 16 single
    config_dict = {
        "paths": {
            "source_paths": [str(source_dir)],
            "base_paths": [str(base_path)],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {
            "num_images": num_pairs,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
        },
        "processing": {
            "backend": "cpu",
            "omp_threads": 4,
            "dask_workers_per_node": 2,
            "dask_memory_limit": "4GB",
            "dask_max_in_flight_per_worker": 3,
        },
        "batches": {"batch_size": 10},
        "ensemble_piv": {
            "window_size": [[32, 32], [16, 16], [16, 16]],
            "overlap": [50, 50, 50],
            "type": ["std", "single", "single"],
            "sum_window": [32, 32],
            "sum_fitting_window": [16, 16],
            "sum_fitting_window_enabled": True,
            "runs": [2, 3],  # save the two 16×16 single passes (1-based)
            "peak_finder": "gaussian",
            "fit_method": fit_method,
            "gradient_correction": True,
            "predictor_smoothing": False,
        },
        "outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 3.0, "epsilon": 0.1}],
        },
        "infilling": {
            "mid_pass": {"method": "nearest", "parameters": {}},
            "final_pass": {"method": "biharmonic", "parameters": {}},
        },
        "ensemble_outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 3.0, "epsilon": 0.1}],
        },
        "ensemble_infilling": {
            "mid_pass": {"method": "nearest", "parameters": {}},
            "final_pass": {"method": "biharmonic", "parameters": {}},
        },
        "filters": [],
        "masking": {"enabled": False},
    }

    config_path = workspace / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    reload_config()
    config = Config(path=str(config_path))

    output_path = get_ensemble_output_path(config, camera=1, use_uncalibrated=True)

    from dask.distributed import Client

    from pivtools_core.ensemble import run_ensemble_piv

    print(f"\n{'='*60}")
    print(f"  Running ensemble PIV — fit_method={fit_method}")
    print("  3 passes: 32std -> 16single -> 16single")
    print(f"  {num_pairs} pairs, batch_size=10, omp_threads=4, workers=2")
    print(f"{'='*60}")

    client = Client(processes=False, n_workers=2, threads_per_worker=1)
    try:
        run_ensemble_piv(
            config=config,
            client=client,
            camera_num=1,
            source_path=source_dir,
            output_path=output_path,
            base_path=base_path,
        )
    finally:
        client.close()

    # Load ensemble result
    result_file = output_path / "ensemble_result.mat"
    assert result_file.exists(), f"Ensemble result not found: {result_file}"
    ensemble_result, n_loaded = load_ensemble_result(result_file)

    # Build per-pass dict keyed by 0-based pass index.
    # load_ensemble_result returns ALL passes (including empty ones for non-saved).
    # runs=[2,3] (1-based) -> passes 1,2 (0-based) have data; pass 0 is empty.
    pass_data = {}
    for pass_idx, p in enumerate(ensemble_result.passes):
        if p.ux_mat is None:
            continue  # skip empty passes
        pass_data[pass_idx] = {
            "ux": p.ux_mat,
            "uy": p.uy_mat,
            "pred_x": p.pred_x,
            "pred_y": p.pred_y,
            "peakheight": p.peakheight,
            "UU_stress": p.UU_stress,
            "VV_stress": p.VV_stress,
            "UV_stress": p.UV_stress,
            "b_mask": p.b_mask,
        }

    print(
        f"\n  Loaded passes: {sorted(pass_data.keys())} "
        f"(grid shapes: {[pass_data[k]['ux'].shape for k in sorted(pass_data.keys())]})"
    )

    return {
        "pass_data": pass_data,
        "params": params,
        "output_path": output_path,
        "config_dict": config_dict,
        "fit_method": fit_method,
    }


# ===================================================================
# Part A: Instantaneous PIV (single image pair)
# ===================================================================


@pytest.fixture(scope="module")
def instantaneous_workspace():
    """
    Run 3-pass instantaneous PIV on the first Poiseuille ensemble image pair.
    Returns dict with per-pass results and ground truth.
    """
    # params.json is git-tracked but the rendered tifs are NOT — a fresh
    # worktree has the former without the latter, so check an actual image too
    # (otherwise setup dies in shutil.copy2 with a bare FileNotFoundError).
    if not PARAMS_PATH.exists() or not (ENSEMBLE_DATA_DIR / "B00001_A.tif").exists():
        pytest.skip(
            f"Synthetic data not found at {ENSEMBLE_DATA_DIR}. "
            "Run: cd unit-tests && python generate_poiseuille_ensemble.py"
        )

    params = _load_params()
    workspace = ENSEMBLE_DATA_DIR / "test_workspace_inst"

    if workspace.exists():
        _robust_rmtree(workspace)
    workspace.mkdir(exist_ok=True)

    source_dir = workspace / "source"
    source_dir.mkdir(exist_ok=True)
    base_path = workspace / "output"
    base_path.mkdir(exist_ok=True)

    # Copy first image pair
    for name in ("B00001_A.tif", "B00001_B.tif"):
        shutil.copy2(ENSEMBLE_DATA_DIR / name, source_dir / name)

    # Config: 3-pass instantaneous
    config_dict = {
        "paths": {
            "source_paths": [str(source_dir)],
            "base_paths": [str(base_path)],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {
            "num_images": 1,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "vector_format": ["B%05d.mat"],
        },
        "processing": {
            "backend": "cpu",
            "omp_threads": 4,
            "dask_workers_per_node": 1,
            "dask_memory_limit": "2GB",
            "dask_max_in_flight_per_worker": 3,
        },
        "batches": {"batch_size": 1},
        "instantaneous_piv": {
            "window_size": [[64, 64], [32, 32], [16, 16]],
            "overlap": [50],
            "runs": [1, 2, 3],  # save all passes (1-based)
            "peak_finder": "gauss6",
        },
        "outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 3.0, "epsilon": 0.1}],
        },
        "infilling": {
            "mid_pass": {"method": "nearest", "parameters": {}},
            "final_pass": {"method": "biharmonic", "parameters": {}},
        },
        "filters": [],
        "masking": {"enabled": False},
    }

    config_path = workspace / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    reload_config()
    config = Config(path=str(config_path))

    output_path = get_output_path(
        config, camera=1, use_uncalibrated=True, piv_type="instantaneous"
    )

    from dask.distributed import Client

    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    try:
        run_instantaneous_piv(
            config=config,
            client=client,
            camera_num=1,
            source_path=source_dir,
            output_path=output_path,
            base_path=base_path,
        )
    finally:
        client.close()

    # Load per-pass results
    pass_results = {}
    for run_idx in range(3):
        arr = read_mat_contents(str(output_path / "B00001.mat"), run_index=run_idx)
        ux = arr[0, 0].astype(float)
        uy = arr[0, 1].astype(float)
        b_mask = arr[0, 2]
        valid = (b_mask == 0) & np.isfinite(ux) & np.isfinite(uy)
        pass_results[run_idx] = {"ux": ux, "uy": uy, "valid": valid}

    return {
        "pass_results": pass_results,
        "params": params,
        "output_path": output_path,
    }


class TestInstantaneousWarping:
    """Verify instantaneous multipass PIV on Poiseuille flow."""

    def test_ux_matches_poiseuille(self, instantaneous_workspace):
        """Recovered ux matches analytical Poiseuille profile."""
        ws = instantaneous_workspace
        params = ws["params"]
        # Use final pass (pass 2, 0-indexed)
        res = ws["pass_results"][2]
        ux, valid = res["ux"], res["valid"]
        H = params["image_shape"][0]
        u_max = params["u_max"]

        nrows, ncols = ux.shape
        interior = _interior_mask(ux.shape) & valid

        pixel_rows = np.linspace(0, H - 1, nrows)
        gt_ux = _poiseuille_ux_pixels(pixel_rows, H, u_max)
        gt_grid = np.broadcast_to(gt_ux[:, np.newaxis], ux.shape)

        err = np.abs(ux[interior] - gt_grid[interior])
        median_err = np.median(err)
        assert (
            median_err < 0.3
        ), f"median |ux - gt| = {median_err:.3f} px, expected < 0.3"

    def test_uy_near_zero(self, instantaneous_workspace):
        """Vertical displacement should be near zero for Poiseuille."""
        res = instantaneous_workspace["pass_results"][2]
        uy, valid = res["uy"], res["valid"]
        interior = _interior_mask(uy.shape) & valid
        median_uy = np.median(np.abs(uy[interior]))
        assert median_uy < 0.15, f"median |uy| = {median_uy:.3f} px, expected < 0.15"


# ===================================================================
# Part B: Ensemble PIV (20 image pairs, parametrized by fit_method)
# ===================================================================

# Passes: 32×32 std -> 16×16 single -> 16×16 single
# Saved passes (1-based): [2, 3] -> 0-based: [1, 2]
PASS_2_IDX = 1  # first 16×16 single pass (0-based)
PASS_3_IDX = 2  # second 16×16 single pass (0-based, final)


@pytest.fixture(scope="module", params=["kspace"])
def ensemble_workspace(request):
    """
    Run 3-pass ensemble PIV with the production fit_method ('kspace' — the
    LM fitter is the only selectable method; the dormant closed-form fitter
    is covered by its direct unit tests, not this e2e path).
    """
    fit_method = request.param

    # params.json is git-tracked but the rendered tifs are NOT — a fresh
    # worktree has the former without the latter, so check an actual image too
    # (otherwise setup dies in shutil.copy2 with a bare FileNotFoundError).
    if not PARAMS_PATH.exists() or not (ENSEMBLE_DATA_DIR / "B00001_A.tif").exists():
        pytest.skip(
            f"Synthetic data not found at {ENSEMBLE_DATA_DIR}. "
            "Run: cd unit-tests && python generate_poiseuille_ensemble.py"
        )

    return _run_ensemble_piv(fit_method)


class TestPredictorConvergence:
    """Core convergence test: predictor should match measured velocity."""

    def test_final_pred_x_matches_ux(self, ensemble_workspace):
        """At final pass, predictor_x should match measured ux (convergence)."""
        ws = ensemble_workspace
        p = ws["pass_data"][PASS_3_IDX]
        ux, pred_x = p["ux"], p["pred_x"]

        if pred_x is None:
            pytest.skip("pred_x not saved for final pass")

        valid = np.isfinite(ux) & np.isfinite(pred_x)
        interior = _interior_mask(ux.shape) & valid

        err = np.abs(pred_x[interior] - ux[interior])
        median_err = np.median(err)
        assert median_err < 0.15, (
            f"[{ws['fit_method']}] median |pred_x - ux| = {median_err:.3f} px "
            f"at final pass, expected < 0.15"
        )

    def test_final_pred_y_matches_uy(self, ensemble_workspace):
        """At final pass, predictor_y should match measured uy (~0)."""
        ws = ensemble_workspace
        p = ws["pass_data"][PASS_3_IDX]
        uy, pred_y = p["uy"], p["pred_y"]

        if pred_y is None:
            pytest.skip("pred_y not saved for final pass")

        valid = np.isfinite(uy) & np.isfinite(pred_y)
        interior = _interior_mask(uy.shape) & valid

        err = np.abs(pred_y[interior] - uy[interior])
        median_err = np.median(err)
        assert median_err < 0.1, (
            f"[{ws['fit_method']}] median |pred_y - uy| = {median_err:.3f} px "
            f"at final pass, expected < 0.1"
        )

    def test_predictor_error_decreases_across_passes(self, ensemble_workspace):
        """Predictor-velocity mismatch should decrease from pass 2 to pass 3."""
        ws = ensemble_workspace
        errors = {}
        for pidx in [PASS_2_IDX, PASS_3_IDX]:
            p = ws["pass_data"][pidx]
            ux, pred_x = p["ux"], p["pred_x"]
            if pred_x is None:
                pytest.skip(f"pred_x not saved for pass {pidx+1}")

            valid = np.isfinite(ux) & np.isfinite(pred_x)
            interior = _interior_mask(ux.shape) & valid
            errors[pidx] = np.median(np.abs(pred_x[interior] - ux[interior]))

        assert errors[PASS_3_IDX] <= errors[PASS_2_IDX], (
            f"[{ws['fit_method']}] Predictor error should decrease: "
            f"pass2={errors[PASS_2_IDX]:.4f}, pass3={errors[PASS_3_IDX]:.4f}"
        )


class TestFinalPassConvergence:
    """Verify repeating the same pass config gives the same answer."""

    def test_repeated_final_pass_converged(self, ensemble_workspace):
        """Pass 2 and pass 3 (both 16px single) should give same velocity."""
        ws = ensemble_workspace
        p2 = ws["pass_data"][PASS_2_IDX]
        p3 = ws["pass_data"][PASS_3_IDX]

        valid = (
            np.isfinite(p2["ux"])
            & np.isfinite(p3["ux"])
            & np.isfinite(p2["uy"])
            & np.isfinite(p3["uy"])
        )
        interior = _interior_mask(p2["ux"].shape) & valid

        ux_diff = np.median(np.abs(p3["ux"][interior] - p2["ux"][interior]))
        uy_diff = np.median(np.abs(p3["uy"][interior] - p2["uy"][interior]))

        # Tolerance 0.05 -> 0.06 (2026-07-06): the pair-count envelope divide in
        # finalize_pass re-weights the k-space fit (autos-only in single mode), and
        # with only 20 pairs the two passes' different predictors give ~0.053 px
        # median disagreement. Absolute accuracy vs the analytic profile is checked
        # separately (TestVelocityAccuracy, 0.2 px) and unaffected.
        assert ux_diff < 0.06, (
            f"[{ws['fit_method']}] median |ux_p3 - ux_p2| = {ux_diff:.4f} px, "
            f"expected < 0.06"
        )
        assert uy_diff < 0.06, (
            f"[{ws['fit_method']}] median |uy_p3 - uy_p2| = {uy_diff:.4f} px, "
            f"expected < 0.06"
        )


class TestVelocityAccuracy:
    """Verify ensemble velocity matches analytical Poiseuille profile."""

    def test_ux_matches_poiseuille_profile(self, ensemble_workspace):
        """Final pass ux matches analytical Poiseuille within 0.2 px."""
        ws = ensemble_workspace
        params = ws["params"]
        p = ws["pass_data"][PASS_3_IDX]
        ux = p["ux"]
        H = params["image_shape"][0]
        u_max = params["u_max"]

        valid = np.isfinite(ux)
        interior = _interior_mask(ux.shape) & valid

        nrows = ux.shape[0]
        pixel_rows = np.linspace(0, H - 1, nrows)
        gt_ux = _poiseuille_ux_pixels(pixel_rows, H, u_max)
        gt_grid = np.broadcast_to(gt_ux[:, np.newaxis], ux.shape)

        err = np.abs(ux[interior] - gt_grid[interior])
        median_err = np.median(err)
        assert median_err < 0.2, (
            f"[{ws['fit_method']}] median |ux - gt| = {median_err:.3f} px, "
            f"expected < 0.2"
        )

    def test_uy_near_zero(self, ensemble_workspace):
        """Vertical displacement should be near zero."""
        ws = ensemble_workspace
        p = ws["pass_data"][PASS_3_IDX]
        uy = p["uy"]
        valid = np.isfinite(uy)
        interior = _interior_mask(uy.shape) & valid
        median_uy = np.median(np.abs(uy[interior]))
        assert median_uy < 0.1, (
            f"[{ws['fit_method']}] median |uy| = {median_uy:.3f} px, " f"expected < 0.1"
        )


class TestSingleModeSpecific:
    """Verify single-mode passes ran end-to-end."""

    def test_single_mode_passes_have_results(self, ensemble_workspace):
        """Both 16×16 single-mode passes should have valid fields."""
        ws = ensemble_workspace
        for pidx in [PASS_2_IDX, PASS_3_IDX]:
            p = ws["pass_data"][pidx]
            assert p["ux"] is not None, f"[{ws['fit_method']}] Pass {pidx+1} ux is None"
            assert p["uy"] is not None, f"[{ws['fit_method']}] Pass {pidx+1} uy is None"
            assert (
                p["peakheight"] is not None
            ), f"[{ws['fit_method']}] Pass {pidx+1} peakheight is None"
            valid = np.isfinite(p["ux"])
            assert valid.sum() > 10, (
                f"[{ws['fit_method']}] Pass {pidx+1} has too few valid "
                f"windows ({valid.sum()})"
            )


class TestDiagnosticFigures:
    """Generate diagnostic plots (gated by --make-figures)."""

    def test_make_figures(self, ensemble_workspace, make_figures, output_dir):
        """Generate 3-panel convergence diagnostic figure."""
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ws = ensemble_workspace
        fit_method = ws["fit_method"]
        params = ws["params"]
        H = params["image_shape"][0]
        u_max = params["u_max"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Multipass Ensemble PIV Convergence — {fit_method} "
            f"(Poiseuille, 32std->16single->16single)",
            fontweight="bold",
        )

        # Panel 1: Row-averaged ux profile vs analytical
        ax = axes[0]
        pass_labels = {
            PASS_2_IDX: "Pass 2 (16px single)",
            PASS_3_IDX: "Pass 3 (16px single)",
        }
        for pidx, label in pass_labels.items():
            p = ws["pass_data"][pidx]
            ux = p["ux"]
            row_mean = np.nanmean(ux, axis=1)
            nrows = ux.shape[0]
            pixel_rows = np.linspace(0, H - 1, nrows)
            ax.plot(row_mean, pixel_rows, "o-", markersize=3, label=label)

        # Analytical
        y_fine = np.linspace(0, H - 1, 200)
        ax.plot(
            _poiseuille_ux_pixels(y_fine, H, u_max),
            y_fine,
            "k--",
            linewidth=2,
            label="Analytical",
        )
        ax.set_xlabel("ux (px)")
        ax.set_ylabel("Row (px)")
        ax.set_title("ux profile")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        # Panel 2: Predictor-velocity error vs pass
        ax = axes[1]
        pass_indices = sorted(ws["pass_data"].keys())
        pred_errors = []
        for pidx in pass_indices:
            p = ws["pass_data"][pidx]
            if p["pred_x"] is not None and p["ux"] is not None:
                valid = np.isfinite(p["ux"]) & np.isfinite(p["pred_x"])
                interior = _interior_mask(p["ux"].shape) & valid
                err = np.median(np.abs(p["pred_x"][interior] - p["ux"][interior]))
                pred_errors.append((pidx + 1, err))

        if pred_errors:
            passes, errs = zip(*pred_errors)
            ax.bar(passes, errs, color=["#4c72b0", "#dd8452"][: len(passes)])
            ax.set_xlabel("Pass number")
            ax.set_ylabel("median |pred_x - ux| (px)")
            ax.set_title("Predictor convergence")

        # Panel 3: Pass 2 vs Pass 3 scatter
        ax = axes[2]
        p2 = ws["pass_data"][PASS_2_IDX]
        p3 = ws["pass_data"][PASS_3_IDX]
        valid = np.isfinite(p2["ux"]) & np.isfinite(p3["ux"])
        ax.scatter(p2["ux"][valid], p3["ux"][valid], s=5, alpha=0.5, c="#4c72b0")
        lims = [
            min(np.nanmin(p2["ux"][valid]), np.nanmin(p3["ux"][valid])),
            max(np.nanmax(p2["ux"][valid]), np.nanmax(p3["ux"][valid])),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("Pass 2 ux (px)")
        ax.set_ylabel("Pass 3 ux (px)")
        ax.set_title("Pass convergence (y=x)")
        ax.set_aspect("equal")

        fig.tight_layout()
        out_path = output_dir / f"multipass_convergence_{fit_method}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Figure saved: {out_path}")
