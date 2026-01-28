#!/usr/bin/env python3
"""
Apply gradient corrections to Reynolds stress measurements.

Corrections based on geometric smearing theory:
- UU_corrected = (sig_AB_x - sig_A_x) - 0.5 * sig_A_x * (dU/dy)²
- VV_corrected = (sig_AB_y - sig_A_y) - 0.5 * sig_A_y * (dV/dx)²
- UV_corrected = sig_xy_AB - 0.5 * sig_A_x * (dU/dy + dV/dx)
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
import sys
import copy

def load_ensemble_result(result_path: Path, pass_idx: int = -1) -> dict:
    """Load ensemble result .mat file.

    Args:
        result_path: Path to ensemble_result.mat
        pass_idx: Which pass to load (-1 for final pass)

    Returns:
        Dictionary with result arrays for the specified pass
    """
    data = loadmat(str(result_path))

    # Data is stored as structured array: ensemble_result with shape (1, num_passes)
    ensemble = data['ensemble_result']
    num_passes = ensemble.shape[1]

    if pass_idx == -1:
        pass_idx = num_passes - 1

    # Extract data for specified pass
    result = ensemble[0, pass_idx]

    return {
        'U': np.squeeze(result['ux']),
        'V': np.squeeze(result['uy']),
        'sig_A_x': np.squeeze(result['sig_A_x']),
        'sig_A_y': np.squeeze(result['sig_A_y']),
        'sig_AB_x': np.squeeze(result['sig_AB_x']),
        'sig_AB_y': np.squeeze(result['sig_AB_y']),
        'UU_stress': np.squeeze(result['UU_stress']),
        'VV_stress': np.squeeze(result['VV_stress']),
        'UV_stress': np.squeeze(result['UV_stress']) if 'UV_stress' in result.dtype.names else None,
        'window_size': np.squeeze(result['window_size']),
        'num_passes': num_passes,
        'pass_idx': pass_idx,
    }

def load_coordinates(coord_path: Path, pass_idx: int = -1) -> dict:
    """Load coordinates .mat file.

    Args:
        coord_path: Path to coordinates.mat
        pass_idx: Which pass to load (-1 for final pass)
    """
    data = loadmat(str(coord_path))

    # Data is stored as structured array: coordinates with shape (1, num_passes)
    coords = data['coordinates']
    num_passes = coords.shape[1]

    if pass_idx == -1:
        pass_idx = num_passes - 1

    result = coords[0, pass_idx]

    return {
        'X': np.squeeze(result['x']),
        'Y': np.squeeze(result['y']),
    }

def compute_gradients(U: np.ndarray, V: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Compute velocity gradients using central differences.

    Returns dU/dy and dV/dx (the gradients that cause geometric smearing).

    NOTE: Uses SIGNED spacing to correctly handle coordinate orientation.
    If Y decreases with row index (image coordinates), dy will be negative,
    and np.gradient will compute the correct physical gradient direction.
    This matches MATLAB convention: dy = -2 when Y decreases with row index.
    """
    # Get grid spacing WITH SIGN (not absolute value!)
    # Sign is critical for correct gradient direction
    if X.ndim == 2:
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    else:
        dx = X[1] - X[0] if len(X) > 1 else 1.0
        dy = Y[1] - Y[0] if len(Y) > 1 else 1.0

    # Compute gradients using numpy gradient (central differences)
    # np.gradient with signed spacing gives correct physical gradient
    # dU/dy: gradient of U in the y direction
    dU_dy = np.gradient(U, dy, axis=0)

    # dV/dx: gradient of V in the x direction
    dV_dx = np.gradient(V, dx, axis=1)

    # Also compute dU/dx and dV/dy for completeness
    dU_dx = np.gradient(U, dx, axis=1)
    dV_dy = np.gradient(V, dy, axis=0)

    return {
        'dU_dy': dU_dy,
        'dV_dx': dV_dx,
        'dU_dx': dU_dx,
        'dV_dy': dV_dy,
        'dx': dx,
        'dy': dy,
    }

def apply_gradient_correction(data: dict, grads: dict) -> dict:
    """
    Apply gradient corrections to Reynolds stresses.

    In our fitter:
    - sig_A_x = autocorrelation width (2 * particle variance)
    - sig_AB_x = ADDITIONAL width from displacement variance (already subtracted)
    - UU_stress = sig_AB_x (the displacement variance component)

    Correction formula:
    - UU_corrected = UU_stress - 0.5 * sig_A_x * (dU/dy)²
    - VV_corrected = VV_stress - 0.5 * sig_A_y * (dV/dx)²
    - UV_corrected = UV_stress - 0.5 * sig_A_x * (dU/dy + dV/dx)
    """
    sig_A_x = data['sig_A_x']
    sig_A_y = data['sig_A_y']
    UU_stress = data['UU_stress']
    VV_stress = data['VV_stress']
    UV_stress = data['UV_stress']

    dU_dy = grads['dU_dy']
    dV_dx = grads['dV_dx']

    # Gradient corrections
    UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)
    VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)
    UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)

    # Corrected stresses
    UU_corrected = UU_stress - UU_correction
    VV_corrected = VV_stress - VV_correction
    UV_corrected = UV_stress - UV_correction if UV_stress is not None else None

    return {
        'UU_raw': UU_stress,
        'VV_raw': VV_stress,
        'UV_raw': UV_stress,
        'UU_correction': UU_correction,
        'VV_correction': VV_correction,
        'UV_correction': UV_correction,
        'UU_corrected': UU_corrected,
        'VV_corrected': VV_corrected,
        'UV_corrected': UV_corrected,
    }

def analyze_result(name: str, result_path: Path, coord_path: Path, true_uu: float, true_vv: float, pass_idx: int = -1):
    """Analyze a single result with gradient corrections."""
    print(f"\n{'='*70}")
    print(f"Result: {name}")
    print(f"{'='*70}")

    # Load data
    data = load_ensemble_result(result_path, pass_idx)
    coords = load_coordinates(coord_path, data['pass_idx'])

    print(f"  Pass {data['pass_idx']+1} of {data['num_passes']}, window_size={data['window_size']}")

    # Compute gradients
    grads = compute_gradients(data['U'], data['V'], coords['X'], coords['Y'])

    # Print mean flow statistics
    print(f"\nMean Flow:")
    print(f"  U mean: {np.nanmean(data['U']):.4f} px/frame")
    print(f"  V mean: {np.nanmean(data['V']):.4f} px/frame")
    print(f"  Grid spacing: dx={grads['dx']:.1f}, dy={grads['dy']:.1f} px (negative dy = Y decreases with row)")

    # Print gradient statistics
    print(f"\nVelocity Gradients:")
    print(f"  dU/dy: mean={np.nanmean(grads['dU_dy']):.6f}, max={np.nanmax(np.abs(grads['dU_dy'])):.6f}")
    print(f"  dV/dx: mean={np.nanmean(grads['dV_dx']):.6f}, max={np.nanmax(np.abs(grads['dV_dx'])):.6f}")
    print(f"  dU/dx: mean={np.nanmean(grads['dU_dx']):.6f}, max={np.nanmax(np.abs(grads['dU_dx'])):.6f}")
    print(f"  dV/dy: mean={np.nanmean(grads['dV_dy']):.6f}, max={np.nanmax(np.abs(grads['dV_dy'])):.6f}")

    # Print sigma values
    print(f"\nCorrelation Widths (mean across field):")
    print(f"  sig_A_x:  {np.nanmean(data['sig_A_x']):.4f} (particle size)")
    print(f"  sig_A_y:  {np.nanmean(data['sig_A_y']):.4f}")
    print(f"  UU_stress (sig_AB_x): {np.nanmean(data['UU_stress']):.4f} (displacement variance)")
    print(f"  VV_stress (sig_AB_y): {np.nanmean(data['VV_stress']):.4f}")

    # Apply corrections
    corrected = apply_gradient_correction(data, grads)

    # Print correction magnitudes
    print(f"\nGradient Corrections (mean across field):")
    print(f"  UU correction: {np.nanmean(corrected['UU_correction']):.6f}")
    print(f"  VV correction: {np.nanmean(corrected['VV_correction']):.6f}")
    print(f"  UV correction: {np.nanmean(corrected['UV_correction']):.6f}")

    # Compare raw vs corrected
    print(f"\nReynolds Stress Comparison:")
    print(f"  {'Quantity':<12} {'Raw':<12} {'Correction':<12} {'Corrected':<12} {'True':<12} {'Raw Err%':<12} {'Corr Err%':<12}")
    print(f"  {'-'*90}")

    uu_raw_mean = np.nanmean(corrected['UU_raw'])
    uu_corr_val = np.nanmean(corrected['UU_correction'])
    uu_corr_mean = np.nanmean(corrected['UU_corrected'])
    uu_raw_err = (uu_raw_mean - true_uu) / true_uu * 100 if true_uu != 0 else 0
    uu_corr_err = (uu_corr_mean - true_uu) / true_uu * 100 if true_uu != 0 else 0
    print(f"  {'UU':<12} {uu_raw_mean:<12.4f} {uu_corr_val:<12.4f} {uu_corr_mean:<12.4f} {true_uu:<12.4f} {uu_raw_err:<12.1f} {uu_corr_err:<12.1f}")

    vv_raw_mean = np.nanmean(corrected['VV_raw'])
    vv_corr_val = np.nanmean(corrected['VV_correction'])
    vv_corr_mean = np.nanmean(corrected['VV_corrected'])
    vv_raw_err = (vv_raw_mean - true_vv) / true_vv * 100 if true_vv != 0 else 0
    vv_corr_err = (vv_corr_mean - true_vv) / true_vv * 100 if true_vv != 0 else 0
    print(f"  {'VV':<12} {vv_raw_mean:<12.4f} {vv_corr_val:<12.4f} {vv_corr_mean:<12.4f} {true_vv:<12.4f} {vv_raw_err:<12.1f} {vv_corr_err:<12.1f}")

    return {
        'name': name,
        'uu_raw': uu_raw_mean,
        'uu_corrected': uu_corr_mean,
        'uu_correction': uu_corr_val,
        'uu_raw_err': uu_raw_err,
        'uu_corr_err': uu_corr_err,
        'vv_raw': vv_raw_mean,
        'vv_corrected': vv_corr_mean,
        'vv_correction': vv_corr_val,
        'vv_raw_err': vv_raw_err,
        'vv_corr_err': vv_corr_err,
        'dU_dy_max': np.nanmax(np.abs(grads['dU_dy'])),
        'dV_dx_max': np.nanmax(np.abs(grads['dV_dx'])),
    }

def apply_and_save_corrections(result_path: Path, coord_path: Path,
                                output_path: Path = None, backup: bool = True) -> dict:
    """
    Apply gradient corrections and save back to .mat file.

    Args:
        result_path: Path to ensemble_result.mat
        coord_path: Path to coordinates.mat
        output_path: Where to save (default: overwrite result_path)
        backup: If True and overwriting, create .mat.bak backup first

    Returns:
        Dictionary with correction statistics
    """
    if output_path is None:
        output_path = result_path

    # Create backup if overwriting
    if backup and output_path == result_path:
        backup_path = Path(str(result_path) + '.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy(result_path, backup_path)
            print(f"Created backup: {backup_path}")

    # Load the raw .mat data (not our parsed version)
    mat_data = loadmat(str(result_path))
    coord_data = loadmat(str(coord_path))

    ensemble = mat_data['ensemble_result']
    num_passes = ensemble.shape[1]

    print(f"\nApplying gradient corrections to {result_path}")
    print(f"  Processing {num_passes} passes...")

    stats = []

    # Process each pass
    for pass_idx in range(num_passes):
        result = ensemble[0, pass_idx]
        coord = coord_data['coordinates'][0, pass_idx]

        # Extract arrays
        U = np.squeeze(result['ux'])
        V = np.squeeze(result['uy'])
        X = np.squeeze(coord['x'])
        Y = np.squeeze(coord['y'])
        sig_A_x = np.squeeze(result['sig_A_x'])
        sig_A_y = np.squeeze(result['sig_A_y'])
        UU_stress = np.squeeze(result['UU_stress'])
        VV_stress = np.squeeze(result['VV_stress'])

        # Check if UV_stress exists
        has_uv = 'UV_stress' in result.dtype.names
        if has_uv:
            UV_stress = np.squeeze(result['UV_stress'])

        # Compute gradients with SIGNED spacing
        if X.ndim == 2:
            dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
            dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
        else:
            dx = X[1] - X[0] if len(X) > 1 else 1.0
            dy = Y[1] - Y[0] if len(Y) > 1 else 1.0

        dU_dy = np.gradient(U, dy, axis=0)
        dV_dx = np.gradient(V, dx, axis=1)

        # Compute corrections
        UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)
        VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)

        # Apply corrections
        UU_corrected = UU_stress - UU_correction
        VV_corrected = VV_stress - VV_correction

        # Store back into the struct (direct field assignment)
        result['UU_stress'][:] = UU_corrected
        result['VV_stress'][:] = VV_corrected

        if has_uv:
            UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)
            UV_corrected = UV_stress - UV_correction
            result['UV_stress'][:] = UV_corrected

            stats.append({
                'pass': pass_idx + 1,
                'dy': dy,
                'UU_mean_correction': np.nanmean(UU_correction),
                'VV_mean_correction': np.nanmean(VV_correction),
                'UV_mean_correction': np.nanmean(UV_correction),
            })
        else:
            stats.append({
                'pass': pass_idx + 1,
                'dy': dy,
                'UU_mean_correction': np.nanmean(UU_correction),
                'VV_mean_correction': np.nanmean(VV_correction),
            })

        print(f"  Pass {pass_idx + 1}: dy={dy:.1f}, "
              f"mean UU_corr={np.nanmean(UU_correction):.6f}, "
              f"mean VV_corr={np.nanmean(VV_correction):.6f}" +
              (f", mean UV_corr={np.nanmean(UV_correction):.6f}" if has_uv else ""))

    # Save the modified data
    savemat(str(output_path), mat_data)
    print(f"\nSaved corrected data to: {output_path}")

    return {'stats': stats, 'output_path': output_path}


def apply_corrections_to_directory(base_dir: Path, backup: bool = True):
    """
    Apply gradient corrections to all ensemble results in a directory tree.

    Searches for ensemble_result.mat files in uncalibrated_piv directories only.
    Calibrated data is always skipped (doesn't contain sig_A_x field needed for correction).
    """
    base_dir = Path(base_dir)

    # Find all ensemble_result.mat files
    result_files = list(base_dir.rglob("ensemble_result.mat"))

    if not result_files:
        print(f"No ensemble_result.mat files found in {base_dir}")
        return

    # Filter to only uncalibrated_piv paths
    uncalibrated_files = [f for f in result_files if "uncalibrated_piv" in str(f)]
    calibrated_files = [f for f in result_files if "calibrated_piv" in str(f)]

    if calibrated_files:
        print(f"Skipping {len(calibrated_files)} calibrated_piv files (correction only applies to uncalibrated)")

    if not uncalibrated_files:
        print(f"No uncalibrated_piv ensemble_result.mat files found in {base_dir}")
        return

    print(f"Found {len(uncalibrated_files)} uncalibrated ensemble result files")

    for result_path in uncalibrated_files:
        coord_path = result_path.parent / "coordinates.mat"

        if coord_path.exists():
            try:
                apply_and_save_corrections(result_path, coord_path, backup=backup)
            except Exception as e:
                print(f"Error processing {result_path}: {e}")
        else:
            print(f"Skipping {result_path}: coordinates.mat not found")


def main():
    base_dir = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/tests/rs_particle_d3")

    # True values for 3px particles (from image generation)
    # UU_true = VV_true = 0.1 (displacement std = 0.316 px/frame)
    true_uu = 0.1
    true_vv = 0.1

    print("="*70)
    print("GRADIENT CORRECTION ANALYSIS FOR REYNOLDS STRESS")
    print("="*70)
    print(f"\nTrue values: UU = {true_uu}, VV = {true_vv}")
    print("(Note: These are VARIANCES of displacement fluctuations)")

    # Test cases to analyze
    test_cases = [
        ("3-pass STD (64→32→16)", "output_3pass"),
        ("3-pass SINGLE sum64 (64→32→16)", "output_single_sumwin64"),
        ("1-pass SINGLE 32×32 sum64", "output_1pass_32_sum64"),
        ("1-pass SINGLE 16×16 sum64", "output_1pass_16_sum64"),
    ]

    results = []
    for name, folder in test_cases:
        result_path = base_dir / folder / "uncalibrated_piv" / "500" / "Cam1" / "ensemble" / "ensemble_result.mat"
        coord_path = base_dir / folder / "uncalibrated_piv" / "500" / "Cam1" / "ensemble" / "coordinates.mat"

        if result_path.exists() and coord_path.exists():
            r = analyze_result(name, result_path, coord_path, true_uu, true_vv)
            results.append(r)
        else:
            print(f"\nSkipping {name}: files not found")

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - UU Stress")
    print("="*100)
    print(f"\n{'Test Case':<35} {'UU Raw':<10} {'Corr':<10} {'UU Final':<10} {'True':<10} {'Raw Err%':<10} {'Corr Err%':<10}")
    print("-"*95)
    for r in results:
        print(f"{r['name']:<35} {r['uu_raw']:<10.4f} {r['uu_correction']:<10.4f} {r['uu_corrected']:<10.4f} {true_uu:<10.4f} {r['uu_raw_err']:<10.1f} {r['uu_corr_err']:<10.1f}")

    print("\n" + "="*100)
    print("SUMMARY TABLE - VV Stress")
    print("="*100)
    print(f"\n{'Test Case':<35} {'VV Raw':<10} {'Corr':<10} {'VV Final':<10} {'True':<10} {'Raw Err%':<10} {'Corr Err%':<10}")
    print("-"*95)
    for r in results:
        print(f"{r['name']:<35} {r['vv_raw']:<10.4f} {r['vv_correction']:<10.4f} {r['vv_corrected']:<10.4f} {true_vv:<10.4f} {r['vv_raw_err']:<10.1f} {r['vv_corr_err']:<10.1f}")

    print(f"\nTrue values: UU = {true_uu}, VV = {true_vv}")

if __name__ == "__main__":
    main()
