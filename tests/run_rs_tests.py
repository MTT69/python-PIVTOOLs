"""
Reynolds Stress Test Runner

Generates synthetic images with known Reynolds stress, runs ensemble PIV,
and analyzes results per-pass to diagnose why RS might drop with each pass.

Usage:
    python run_rs_tests.py [test1|test2|test3|all]
"""
import sys
import os
import yaml
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_rs_images import (
    generate_rs_test_images,
    generate_spatially_varying_rs_images,
)


def create_test_config(
    test_name: str,
    image_dir: Path,
    output_dir: Path,
    num_images: int = 100,
    window_sizes: list = None,
    overlaps: list = None,
    fit_method: str = 'gaussian',  # 'gaussian' or 'kspace'
) -> Path:
    """Create a YAML config file for the test."""
    if window_sizes is None:
        # 256x256 images: use smaller windows
        window_sizes = [
            [64, 64],
            [32, 32],
            [16, 16],
        ]
    if overlaps is None:
        overlaps = [50, 50, 50]

    config = {
        'paths': {
            'base_paths': [str(output_dir)],
            'source_paths': [str(image_dir.parent)],  # Parent of Cam1
            'active_paths': [0],
            'camera_numbers': [1],
            'camera_count': 1,
            'camera_subfolders': ['Cam1'],  # Explicit subfolder for single camera
        },
        'images': {
            'num_images': num_images,
            'image_format': ['B%05d_A.tif', 'B%05d_B.tif'],
            'vector_format': ['%05d.mat'],
            'time_resolved': False,
            'dtype': 'float32',
            'zero_based_indexing': False,
            'pairing_mode': 'sequential',
            'pairing_skip': 0,
            'num_frame_pairs': num_images,
            'image_type': 'standard',
            'use_camera_subfolders': True,  # Images are in Cam1/ subfolder
        },
        'batches': {
            'size': 10,
        },
        'logging': {
            'file': f'{test_name}.log',
            'level': 'INFO',
            'console': True,
        },
        'processing': {
            'backend': 'cpu',
            'debug': False,
            'auto_compute_params': False,
            'omp_threads': 4,
            'dask_workers_per_node': 2,
            'dask_threads_per_worker': 1,
            'dask_memory_limit': '2GB',
            'always_batch': True,
            'instantaneous': False,
            'ensemble': True,
        },
        'outlier_detection': {
            'enabled': False,  # Disable for clean test
            'methods': [],
        },
        'infilling': {
            'mid_pass': {
                'method': 'biharmonic',
                'parameters': {},
            },
            'final_pass': {
                'enabled': False,
                'method': 'biharmonic',
                'parameters': {},
            },
        },
        'ensemble_outlier_detection': {
            'enabled': False,  # Disable for clean test
            'methods': [],
        },
        'ensemble_infilling': {
            'mid_pass': {
                'method': 'biharmonic',
                'parameters': {},
            },
            'final_pass': {
                'enabled': False,
                'method': 'biharmonic',
                'parameters': {},
            },
        },
        'ensemble_piv': {
            'fit_offset': False,
            'fit_method': fit_method,  # 'gaussian' or 'kspace'
            'kspace_snr_threshold': 3.0,
            'window_size': window_sizes,
            'overlap': overlaps,
            'type': ['std'] * len(window_sizes),
            'runs': list(range(1, len(window_sizes) + 1)),
            'store_planes': True,
            'save_diagnostics': True,
            'sum_window': [16, 16],
            'resume_from_pass': 0,
            'window_type': 'square',
        },
        'masking': {
            'enabled': False,
        },
        'filters': [],
    }

    config_path = output_dir / f'{test_name}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_ensemble_piv(config_path: Path) -> bool:
    """Run ensemble PIV using the config file."""
    # The ensemble module looks for config.yaml in the working directory
    # We'll run from the config's directory or copy the config

    pivtools_dir = Path(__file__).parent.parent.parent
    config_dir = config_path.parent

    # Copy config to pivtools_dir as config.yaml temporarily
    temp_config = pivtools_dir / 'config.yaml'
    original_config = None

    # Backup existing config if present
    if temp_config.exists():
        original_config = temp_config.read_text()

    try:
        # Copy our test config
        shutil.copy2(config_path, temp_config)

        cmd = [sys.executable, '-m', 'pivtools_core.ensemble']

        print(f"\nRunning: {' '.join(cmd)}")
        print(f"Config: {config_path}")
        print(f"Working dir: {pivtools_dir}")

        result = subprocess.run(
            cmd,
            cwd=str(pivtools_dir),
            capture_output=False,  # Show output in real-time
        )

        return result.returncode == 0

    finally:
        # Restore original config
        if original_config is not None:
            temp_config.write_text(original_config)
        elif temp_config.exists():
            temp_config.unlink()


def load_ensemble_results(output_dir: Path, num_images: int = 100):
    """Load ensemble results from .mat file."""
    import scipy.io as sio

    # Try multiple possible paths
    possible_paths = [
        output_dir / 'Cam1' / 'uncalibrated' / 'ensemble_result.mat',
        output_dir / f'uncalibrated_piv/{num_images}/Cam1/ensemble/ensemble_result.mat',
        output_dir / 'uncalibrated_piv' / str(num_images) / 'Cam1' / 'ensemble' / 'ensemble_result.mat',
    ]

    result_path = None
    for path in possible_paths:
        if path.exists():
            result_path = path
            break

    # Also search recursively for ensemble_result.mat
    if result_path is None:
        for path in output_dir.rglob('ensemble_result.mat'):
            result_path = path
            break

    if result_path is None:
        print(f"Result file not found in {output_dir}")
        print(f"Searched paths: {[str(p) for p in possible_paths]}")
        return None

    print(f"Loading results from: {result_path}")

    mat_data = sio.loadmat(str(result_path), struct_as_record=True)

    # The data is stored as ensemble_result struct array
    # Each pass is in ensemble_result[0][pass_idx]
    if 'ensemble_result' not in mat_data:
        print(f"No 'ensemble_result' key in mat file. Keys: {list(mat_data.keys())}")
        return None

    ensemble_data = mat_data['ensemble_result']

    # Reshape to access passes - it's typically (1, n_passes) struct array
    # Each entry has fields: ux, uy, UU_stress, VV_stress, sig_AB_x, etc.
    n_passes = ensemble_data.shape[1] if len(ensemble_data.shape) > 1 else ensemble_data.size

    results = {}
    for pass_idx in range(n_passes):
        pass_data = {}
        struct = ensemble_data[0, pass_idx] if len(ensemble_data.shape) > 1 else ensemble_data.flat[pass_idx]

        # Extract fields from the struct
        for field in ['ux', 'uy', 'UU_stress', 'VV_stress', 'UV_stress',
                      'sig_AB_x', 'sig_AB_y', 'sig_AB_xy',
                      'sig_A_x', 'sig_A_y', 'sig_A_xy',
                      'peakheight', 'nan_reason', 'window_size']:
            try:
                val = struct[field]
                if val is not None and hasattr(val, 'shape'):
                    # Unwrap nested arrays from MATLAB
                    while val.ndim > 0 and val.shape[0] == 1 and val.ndim > 2:
                        val = val[0]
                    pass_data[field] = val
                else:
                    pass_data[field] = val
            except (ValueError, IndexError, KeyError):
                pass_data[field] = None

        results[f'pass_{pass_idx + 1}'] = pass_data

    return results


def trim_edges(arr: np.ndarray, edge_pixels: int = 1) -> np.ndarray:
    """Remove edge pixels from a 2D array to avoid boundary effects."""
    if arr is None or arr.ndim != 2:
        return arr
    if arr.shape[0] <= 2 * edge_pixels or arr.shape[1] <= 2 * edge_pixels:
        return arr  # Array too small to trim
    return arr[edge_pixels:-edge_pixels, edge_pixels:-edge_pixels]


def analyze_per_pass_results(data: dict, test_name: str, expected_rs: dict, edge_trim: int = 2):
    """Analyze Reynolds stress per pass and compare to expected values.

    Args:
        data: Dictionary of pass results
        test_name: Name for display
        expected_rs: Expected Reynolds stress values
        edge_trim: Number of edge pixels to exclude from averages (default 1)
    """
    print("\n" + "=" * 70)
    print(f"RESULTS: {test_name}")
    print("=" * 70)
    if edge_trim > 0:
        print(f"(Excluding {edge_trim} edge pixel(s) from averages)")

    # Find all passes
    pass_keys = sorted([k for k in data.keys() if k.startswith('pass_')],
                       key=lambda x: int(x.split('_')[1]))

    results_table = []

    for pass_key in pass_keys:
        pass_data = data[pass_key]
        pass_num = int(pass_key.split('_')[1])

        # Extract all sigma fields (these are standard deviations, not variances)
        sig_ab_x_raw = pass_data.get('sig_AB_x', None)
        sig_ab_y_raw = pass_data.get('sig_AB_y', None)
        sig_a_x_raw = pass_data.get('sig_A_x', None)
        sig_a_y_raw = pass_data.get('sig_A_y', None)

        # Reynolds stress (variance = sig_AB² - sig_A²)
        uu_stress = pass_data.get('UU_stress', None)
        vv_stress = pass_data.get('VV_stress', None)

        # Get mean velocities
        ux = pass_data.get('ux', None)
        uy = pass_data.get('uy', None)

        # Window size info
        win_size = pass_data.get('window_size', None)

        # Convert to arrays and trim edges
        def process_field(field):
            if field is None:
                return None
            arr = np.array(field).squeeze()
            if edge_trim > 0 and arr.ndim == 2:
                arr = trim_edges(arr, edge_trim)
            return arr

        sig_ab_x = process_field(sig_ab_x_raw)
        sig_ab_y = process_field(sig_ab_y_raw)
        sig_a_x = process_field(sig_a_x_raw)
        sig_a_y = process_field(sig_a_y_raw)
        uu_arr = process_field(uu_stress)
        vv_arr = process_field(vv_stress)
        ux_arr = process_field(ux)
        uy_arr = process_field(uy)

        # Compute statistics for each field
        def field_stats(arr):
            if arr is None:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            return {
                'mean': np.mean(valid),
                'std': np.std(valid),
                'min': np.min(valid),
                'max': np.max(valid),
            }

        results_table.append({
            'pass': pass_num,
            'win_size': win_size,
            # Sigma values (standard deviations)
            'sig_A_x': field_stats(sig_a_x),
            'sig_A_y': field_stats(sig_a_y),
            'sig_AB_x': field_stats(sig_ab_x),
            'sig_AB_y': field_stats(sig_ab_y),
            # Reynolds stress (variances)
            'UU': field_stats(uu_arr),
            'VV': field_stats(vv_arr),
            # Mean velocities
            'ux': field_stats(ux_arr),
            'uy': field_stats(uy_arr),
        })

    if not results_table:
        print("No results to analyze!")
        return []

    # Expected values
    exp_ux = expected_rs.get('mean_ux', 0)
    exp_uy = expected_rs.get('mean_uy', 0)
    exp_uu = expected_rs.get('UU', 0)
    exp_vv = expected_rs.get('VV', 0)
    exp_std_x = expected_rs.get('std_x', np.sqrt(exp_uu))  # displacement std
    exp_std_y = expected_rs.get('std_y', np.sqrt(exp_vv))
    particle_d = expected_rs.get('particle_d', 2.0)

    # Theoretical particle sigma (autocorrelation of Gaussian particle images)
    # For a Gaussian particle with diameter d_p, the autocorrelation width is:
    # sig_A = d_p / (2 * sqrt(2)) ≈ 0.354 * d_p for e^-2 diameter definition
    # But PIV fitting often gives sig_A ≈ d_p / sqrt(8) to d_p / 2
    # Empirically, for 2px particles, sig_A is typically ~1.0-1.5 px
    theoretical_sig_A = particle_d / np.sqrt(8)  # ≈ 0.71 for 2px

    print(f"\n{'='*80}")
    print("EXPECTED VALUES (Ground Truth)")
    print(f"{'='*80}")
    print(f"  Mean displacement:     ux = {exp_ux:.3f} px,  uy = {exp_uy:.3f} px")
    print(f"  Displacement std:      σx = {exp_std_x:.3f} px,  σy = {exp_std_y:.3f} px")
    print(f"  Reynolds stress:       UU = σx² = {exp_uu:.3f} px²,  VV = σy² = {exp_vv:.3f} px²")
    print(f"  Particle diameter:     d_p = {particle_d:.1f} px")
    print(f"  Theoretical sig_A:     {theoretical_sig_A:.3f} px (= d_p / sqrt(8))")
    print(f"  Note: Measured sig_A is typically ~2x larger due to finite sampling/interpolation")

    print(f"\n{'='*80}")
    print("MEASURED VALUES (Per Pass)")
    print(f"{'='*80}")

    for r in results_table:
        print(f"\n--- Pass {r['pass']} (Window: {r['win_size']}) ---")

        # Sigma A (autocorrelation - particle contribution) - READ DIRECTLY
        sig_a_x = r['sig_A_x']
        sig_a_y = r['sig_A_y']
        print(f"\n  [READ] sig_A (Autocorrelation peak width):")
        print(f"    sig_A_x:  mean={sig_a_x['mean']:.4f} px  std={sig_a_x['std']:.4f}  range=[{sig_a_x['min']:.3f}, {sig_a_x['max']:.3f}]")
        print(f"    sig_A_y:  mean={sig_a_y['mean']:.4f} px  std={sig_a_y['std']:.4f}  range=[{sig_a_y['min']:.3f}, {sig_a_y['max']:.3f}]")
        print(f"    sig_A²:   x={sig_a_x['mean']**2:.4f} px²  y={sig_a_y['mean']**2:.4f} px²")

        # Reynolds stress - READ DIRECTLY (UU = sig_AB = PDF variance)
        uu = r['UU']
        vv = r['VV']
        print(f"\n  [READ] UU/VV Reynolds Stress (= sig_AB = PDF variance):")
        print(f"    UU: mean={uu['mean']:.4f} px²  std={uu['std']:.4f}  (expected: {exp_uu:.3f})")
        print(f"    VV: mean={vv['mean']:.4f} px²  std={vv['std']:.4f}  (expected: {exp_vv:.3f})")

        # Displacement std (sqrt of Reynolds stress)
        sig_disp_x = np.sqrt(max(0, uu['mean']))
        sig_disp_y = np.sqrt(max(0, vv['mean']))
        print(f"\n  σ_disp = sqrt(UU/VV) (displacement standard deviation):")
        print(f"    σ_disp_x = {sig_disp_x:.4f} px  (expected: {exp_std_x:.3f})")
        print(f"    σ_disp_y = {sig_disp_y:.4f} px  (expected: {exp_std_y:.3f})")

        # Mean velocity - READ DIRECTLY
        ux = r['ux']
        uy = r['uy']
        print(f"\n  [READ] Mean Velocity:")
        print(f"    ux: mean={ux['mean']:.4f} px  std={ux['std']:.4f}  error={ux['mean']-exp_ux:+.4f} px  (expected: {exp_ux:.3f})")
        print(f"    uy: mean={uy['mean']:.4f} px  std={uy['std']:.4f}  error={uy['mean']-exp_uy:+.4f} px  (expected: {exp_uy:.3f})")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Pass':<6} {'sig_A_x':<10} {'sig_A_y':<10} {'UU':<10} {'VV':<10} {'σ_disp_x':<10} {'σ_disp_y':<10} {'ux':<10} {'uy':<10}")
    print("-" * 90)
    for r in results_table:
        sig_disp_x = np.sqrt(max(0, r['UU']['mean']))
        sig_disp_y = np.sqrt(max(0, r['VV']['mean']))
        print(f"{r['pass']:<6} {r['sig_A_x']['mean']:<10.4f} {r['sig_A_y']['mean']:<10.4f} "
              f"{r['UU']['mean']:<10.4f} {r['VV']['mean']:<10.4f} "
              f"{sig_disp_x:<10.4f} {sig_disp_y:<10.4f} "
              f"{r['ux']['mean']:<10.4f} {r['uy']['mean']:<10.4f}")

    # Per-pass change analysis
    if len(results_table) > 1:
        print(f"\n{'='*80}")
        print("PER-PASS CHANGES")
        print(f"{'='*80}")
        for i in range(1, len(results_table)):
            r_prev = results_table[i - 1]
            r_curr = results_table[i]

            def pct_change(curr, prev):
                if prev == 0:
                    return 0
                return 100 * (curr - prev) / prev

            print(f"\n  Pass {r_prev['pass']} -> {r_curr['pass']}:")
            print(f"    sig_A_x:  {r_prev['sig_A_x']['mean']:.4f} -> {r_curr['sig_A_x']['mean']:.4f} px  ({pct_change(r_curr['sig_A_x']['mean'], r_prev['sig_A_x']['mean']):+.1f}%)")
            print(f"    sig_A_y:  {r_prev['sig_A_y']['mean']:.4f} -> {r_curr['sig_A_y']['mean']:.4f} px  ({pct_change(r_curr['sig_A_y']['mean'], r_prev['sig_A_y']['mean']):+.1f}%)")
            print(f"    UU:       {r_prev['UU']['mean']:.4f} -> {r_curr['UU']['mean']:.4f} px²  ({pct_change(r_curr['UU']['mean'], r_prev['UU']['mean']):+.1f}%)")
            print(f"    VV:       {r_prev['VV']['mean']:.4f} -> {r_curr['VV']['mean']:.4f} px²  ({pct_change(r_curr['VV']['mean'], r_prev['VV']['mean']):+.1f}%)")

    return results_table


def run_test1():
    """Test 1: 4 mean groups with combined mean=0, RS=1,2"""
    print("\n" + "=" * 70)
    print("TEST 1: Four mean groups, combined mean=0, target RS UU=1, VV=2")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_test1'
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / 'output'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    print("\nGenerating 500 images...")
    group_means = [
        (-1.0, -1.0),  # pairs 1-125
        (1.0, 1.0),    # pairs 126-250
        (1.0, 0.0),    # pairs 251-375
        (-1.0, 0.0),   # pairs 376-500
    ]

    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=500,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=1.0,
        std_dy=np.sqrt(2.0),
        group_means=group_means,
        seed=42,
    )

    # Create config
    config_path = create_test_config(
        test_name='test1',
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=500,
    )

    # Run PIV
    print("\nRunning ensemble PIV...")
    success = run_ensemble_piv(config_path)

    if not success:
        print("PIV processing failed!")
        return None

    # Analyze results
    data = load_ensemble_results(output_dir, num_images=500)
    if data is not None:
        expected_rs = {
            'UU': 1.0,
            'VV': 2.0,
            'mean_ux': 0.0,
            'mean_uy': 0.0,
        }
        return analyze_per_pass_results(data, "Test 1: Four groups", expected_rs)

    return None


def run_test2():
    """Test 2: Zero mean, RS=2,3"""
    print("\n" + "=" * 70)
    print("TEST 2: Zero mean, target RS UU=2, VV=3")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_test2'
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / 'output'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    print("\nGenerating 500 images...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=500,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),
        std_dy=np.sqrt(3.0),
        seed=123,
    )

    # Create config
    config_path = create_test_config(
        test_name='test2',
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=500,
    )

    # Run PIV
    print("\nRunning ensemble PIV...")
    success = run_ensemble_piv(config_path)

    if not success:
        print("PIV processing failed!")
        return None

    # Analyze results
    data = load_ensemble_results(output_dir, num_images=500)
    if data is not None:
        expected_rs = {
            'UU': 2.0,
            'VV': 3.0,
            'mean_ux': 0.0,
            'mean_uy': 0.0,
            'std_x': np.sqrt(2.0),  # ~1.414 px
            'std_y': np.sqrt(3.0),  # ~1.732 px
            'particle_d': 2.0,
        }
        return analyze_per_pass_results(data, "Test 2: Zero mean", expected_rs)

    return None


def run_test3():
    """Test 3: 5-pass diagnostic (64→32→32→32→16) to isolate warping vs window size effects.

    By using multiple passes at the same window size (32→32→32), we can determine:
    - If RS drops between same-size passes → warping removes variance
    - If RS stays constant between same-size passes → window size is the factor
    """
    print("\n" + "=" * 70)
    print("TEST 3: 5-Pass Diagnostic (64→32→32→32→16)")
    print("Purpose: Isolate warping effect from window size effect")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_test3_5pass'
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / 'output'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images - same as test2 (zero mean, RS=2,3)
    num_images = 2000  # Configurable number of images
    print(f"\nGenerating {num_images} images (zero mean, UU=2, VV=3)...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=num_images,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),  # UU = 2.0
        std_dy=np.sqrt(3.0),  # VV = 3.0
        seed=789,  # Different seed from test2
    )

    # Create config with 5 passes: 64→32→32→32→16
    config_path = create_test_config(
        test_name='test3_5pass',
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=num_images,
        window_sizes=[
            [64, 64],
            [32, 32],
            [32, 32],
            [32, 32],
            [16, 16],
        ],
        overlaps=[50, 50, 50, 50, 50],
    )

    # Run PIV
    print(f"\nRunning ensemble PIV (5 passes, {num_images} images)...")
    success = run_ensemble_piv(config_path)

    if not success:
        print("PIV processing failed!")
        return None

    # Analyze results
    data = load_ensemble_results(output_dir, num_images=num_images)
    if data is not None:
        expected_rs = {
            'UU': 2.0,
            'VV': 3.0,
            'mean_ux': 0.0,
            'mean_uy': 0.0,
            'std_x': np.sqrt(2.0),
            'std_y': np.sqrt(3.0),
            'particle_d': 2.0,
        }
        results = analyze_per_pass_results(data, "Test 3: 5-Pass (64→32→32→32→16)", expected_rs)

        # Special analysis for same-size passes
        if len(results) >= 4:
            print("\n" + "=" * 80)
            print("SAME-SIZE PASS ANALYSIS (32×32 passes)")
            print("=" * 80)
            print("\nIf RS drops between passes 2→3→4 (all 32×32), warping removes variance.")
            print("If RS stays constant, only window size matters.\n")

            # Passes 2, 3, 4 are all 32x32
            for i in [1, 2, 3]:  # indices for passes 2, 3, 4
                if i < len(results):
                    r = results[i]
                    print(f"  Pass {r['pass']} (32×32): sig_A_x={r['sig_A_x']['mean']:.4f}  UU={r['UU']['mean']:.4f}  VV={r['VV']['mean']:.4f}")

            # Calculate drops between same-size passes
            if len(results) >= 4:
                print("\n  Changes between 32×32 passes:")
                for i in range(1, 3):  # Compare pass 2→3 and 3→4
                    r_prev = results[i]
                    r_curr = results[i + 1]
                    uu_drop = 100 * (r_curr['UU']['mean'] - r_prev['UU']['mean']) / r_prev['UU']['mean']
                    vv_drop = 100 * (r_curr['VV']['mean'] - r_prev['VV']['mean']) / r_prev['VV']['mean']
                    sig_a_drop = 100 * (r_curr['sig_A_x']['mean'] - r_prev['sig_A_x']['mean']) / r_prev['sig_A_x']['mean']
                    print(f"    Pass {r_prev['pass']}→{r_curr['pass']}: sig_A_x {sig_a_drop:+.2f}%  UU {uu_drop:+.2f}%  VV {vv_drop:+.2f}%")

        return results

    return None


def run_quick_test():
    """Quick test: 20 images, single pass for fast validation."""
    print("\n" + "=" * 70)
    print("QUICK TEST: 20 images, single pass, target RS UU=2, VV=3")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_quick_test'
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / 'output'

    if test_dir.exists():
        shutil.rmtree(test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating images...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=20,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),
        std_dy=np.sqrt(3.0),
        seed=999,
    )

    # Single pass config
    config_path = create_test_config(
        test_name='quick',
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=20,
        window_sizes=[[64, 64]],  # Single pass
        overlaps=[50],
    )

    print("\nRunning ensemble PIV...")
    success = run_ensemble_piv(config_path)

    if not success:
        print("PIV processing failed!")
        return None

    data = load_ensemble_results(output_dir, num_images=20)
    if data is not None:
        expected_rs = {'UU': 2.0, 'VV': 3.0, 'mean_ux': 0.0, 'mean_uy': 0.0}
        return analyze_per_pass_results(data, "Quick Test", expected_rs)

    return None


def run_single_pass_comparison():
    """Run single-pass tests at different window sizes to establish baseline.

    This tests each window size WITHOUT multi-pass processing to determine
    if the RS drop is inherent to smaller windows or caused by multi-pass.
    """
    print("\n" + "=" * 70)
    print("SINGLE-PASS COMPARISON: Each window size independently")
    print("Purpose: Establish baseline RS at each window size without multi-pass")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_single_pass_comparison'
    image_dir = test_dir / 'Cam1'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Generate images once (same as test2/test3 - zero mean, UU=2, VV=3)
    image_dir.mkdir(parents=True, exist_ok=True)
    num_images = 2000  # Configurable number of images
    print(f"\nGenerating {num_images} images (zero mean, UU=2, VV=3)...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=num_images,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),  # UU = 2.0
        std_dy=np.sqrt(3.0),  # VV = 3.0
        seed=456,  # Different seed
    )

    # Test each window size independently
    window_sizes_to_test = [64, 32, 16]
    all_results = []

    for win_size in window_sizes_to_test:
        print(f"\n{'='*60}")
        print(f"Running single-pass {win_size}x{win_size}")
        print(f"{'='*60}")

        output_dir = test_dir / f'output_{win_size}'
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = create_test_config(
            test_name=f'single_{win_size}',
            image_dir=image_dir,
            output_dir=output_dir,
            num_images=num_images,
            window_sizes=[[win_size, win_size]],
            overlaps=[50],
        )

        success = run_ensemble_piv(config_path)
        if not success:
            print(f"PIV processing failed for {win_size}x{win_size}!")
            continue

        data = load_ensemble_results(output_dir, num_images=num_images)
        if data is not None:
            expected_rs = {
                'UU': 2.0,
                'VV': 3.0,
                'mean_ux': 0.0,
                'mean_uy': 0.0,
                'std_x': np.sqrt(2.0),
                'std_y': np.sqrt(3.0),
                'particle_d': 2.0,
            }
            results = analyze_per_pass_results(
                data, f"Single-Pass {win_size}x{win_size}", expected_rs
            )
            if results:
                all_results.append({
                    'window': win_size,
                    'results': results[0],  # Only one pass
                })

    # Summary comparison
    if all_results:
        print("\n" + "=" * 80)
        print("SINGLE-PASS COMPARISON SUMMARY")
        print("=" * 80)
        print("\nBaseline RS at each window size (NO multi-pass, NO warping):\n")
        print(f"{'Window':<10} {'sig_A_x':<12} {'sig_A_y':<12} {'UU':<12} {'VV':<12} {'UU Error':<12} {'VV Error':<12}")
        print("-" * 82)

        for r in all_results:
            win = r['window']
            res = r['results']
            uu_err = 100 * (res['UU']['mean'] - 2.0) / 2.0
            vv_err = 100 * (res['VV']['mean'] - 3.0) / 3.0
            print(f"{win}x{win:<7} {res['sig_A_x']['mean']:<12.4f} {res['sig_A_y']['mean']:<12.4f} "
                  f"{res['UU']['mean']:<12.4f} {res['VV']['mean']:<12.4f} "
                  f"{uu_err:+.1f}%{'':<6} {vv_err:+.1f}%")

        # Compare to multi-pass final values
        print("\n" + "-" * 82)
        print("Compare: Multi-pass 64→32→16 final pass typically shows UU~1.85, error~-8%")
        print("If single-pass 16x16 shows similar error → inherent to small windows")
        print("If single-pass 16x16 is accurate → multi-pass introduces the bias")

    return all_results


def run_test_mean_only():
    """Test with ZERO Reynolds stress - only mean displacement.

    This isolates whether the RS measurement comes from mean variation
    across the field vs actual displacement variance.
    """
    print("\n" + "=" * 70)
    print("MEAN-ONLY TEST: Non-zero mean, zero RS (all pairs same displacement)")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_mean_only'
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / 'output'

    if test_dir.exists():
        shutil.rmtree(test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating 500 images with uniform displacement (no variance)...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=500,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=2.0,  # Non-zero mean
        mean_dy=3.0,
        std_dx=0.001,  # Near-zero variance
        std_dy=0.001,
        seed=777,
    )

    config_path = create_test_config(
        test_name='mean_only',
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=500,
    )

    print("\nRunning ensemble PIV...")
    success = run_ensemble_piv(config_path)

    if not success:
        print("PIV processing failed!")
        return None

    data = load_ensemble_results(output_dir, num_images=500)
    if data is not None:
        expected_rs = {
            'UU': 0.0,
            'VV': 0.0,
            'mean_ux': 2.0,
            'mean_uy': 3.0,
            'std_x': 0.001,  # Near-zero displacement variance
            'std_y': 0.001,
            'particle_d': 2.0,
        }
        return analyze_per_pass_results(data, "Mean-Only Test (expect RS~0)", expected_rs)

    return None


def run_kspace_comparison():
    """Compare Gaussian vs K-space fitting methods.

    Runs the same test with both fitting methods to compare accuracy.
    """
    print("\n" + "=" * 70)
    print("K-SPACE vs GAUSSIAN COMPARISON TEST")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_kspace_compare'
    image_dir = test_dir / 'Cam1'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Generate images once
    image_dir.mkdir(parents=True, exist_ok=True)
    num_images = 100  # Use fewer images for quick comparison
    print(f"\nGenerating {num_images} images (zero mean, UU=2, VV=3)...")
    stats = generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=num_images,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),  # UU = 2.0
        std_dy=np.sqrt(3.0),  # VV = 3.0
        seed=12345,
    )

    all_results = {}

    for method in ['gaussian', 'kspace']:
        print(f"\n{'='*60}")
        print(f"Running with fit_method = '{method}'")
        print(f"{'='*60}")

        output_dir = test_dir / f'output_{method}'
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = create_test_config(
            test_name=f'kspace_compare_{method}',
            image_dir=image_dir,
            output_dir=output_dir,
            num_images=num_images,
            window_sizes=[[64, 64], [32, 32]],  # 2 passes
            overlaps=[50, 50],
            fit_method=method,
        )

        success = run_ensemble_piv(config_path)
        if not success:
            print(f"PIV processing failed for {method}!")
            continue

        data = load_ensemble_results(output_dir, num_images=num_images)
        if data is not None:
            expected_rs = {
                'UU': 2.0,
                'VV': 3.0,
                'mean_ux': 0.0,
                'mean_uy': 0.0,
                'std_x': np.sqrt(2.0),
                'std_y': np.sqrt(3.0),
                'particle_d': 2.0,
            }
            results = analyze_per_pass_results(
                data, f"{method.upper()} fitting", expected_rs, edge_trim=1
            )
            all_results[method] = results

    # Summary comparison
    if len(all_results) == 2:
        print("\n" + "=" * 80)
        print("GAUSSIAN vs K-SPACE COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\n{'Method':<12} {'Pass':<6} {'UU':<12} {'VV':<12} {'UU Error':<12} {'VV Error':<12}")
        print("-" * 66)

        for method, results in all_results.items():
            for r in results:
                uu_err = 100 * (r['UU']['mean'] - 2.0) / 2.0
                vv_err = 100 * (r['VV']['mean'] - 3.0) / 3.0
                print(f"{method:<12} {r['pass']:<6} {r['UU']['mean']:<12.4f} {r['VV']['mean']:<12.4f} "
                      f"{uu_err:+.1f}%{'':<6} {vv_err:+.1f}%")

    return all_results


def run_particle_diameter_sweep():
    """Test RS accuracy across particle diameters 2, 3, 4, 5 px.

    For each particle size, runs:
    1. Single-pass test (16×16) - baseline without warping (matches 3-pass final window)
    2. 3-pass test (64→32→16) - with predictor warping

    Uses 500 images per test for good statistical accuracy.

    Comparison Logic:
    - If single-pass 16×16 gives accurate RS but 3-pass 16×16 doesn't → multi-pass warping is the problem
    - If both give same (wrong) RS → window size or fitting is the problem
    """
    print("\n" + "=" * 70)
    print("PARTICLE DIAMETER SWEEP TEST")
    print("Testing RS accuracy with particle diameters: 2, 3, 4, 5 px")
    print("=" * 70)

    particle_diameters = [2, 3, 4, 5]
    num_images = 500
    all_results = {}

    for particle_d in particle_diameters:
        print(f"\n{'='*70}")
        print(f"PARTICLE DIAMETER: {particle_d} px")
        print(f"{'='*70}")

        test_dir = Path(__file__).parent / f'rs_particle_d{particle_d}'
        image_dir = test_dir / 'Cam1'

        # Clean previous run
        if test_dir.exists():
            shutil.rmtree(test_dir)

        # Generate images once with specific particle_diameter
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {num_images} images (zero mean, UU=2, VV=3, particle_d={particle_d}px)...")

        stats = generate_rs_test_images(
            output_dir=image_dir,
            num_pairs=num_images,
            image_shape=(256, 256),
            particle_diameter=float(particle_d),
            mean_dx=0.0,
            mean_dy=0.0,
            std_dx=np.sqrt(2.0),  # UU = 2.0
            std_dy=np.sqrt(3.0),  # VV = 3.0
            seed=42 + particle_d,  # Different seed per particle size
        )

        expected_rs = {
            'UU': 2.0,
            'VV': 3.0,
            'mean_ux': 0.0,
            'mean_uy': 0.0,
            'std_x': np.sqrt(2.0),
            'std_y': np.sqrt(3.0),
            'particle_d': float(particle_d),
        }

        particle_results = {'particle_d': particle_d}

        # --- Single-pass test (16×16) - baseline without multi-pass warping ---
        print(f"\n--- Single-pass 16×16 (baseline, no warping) ---")
        output_dir_single = test_dir / 'output_single'
        output_dir_single.mkdir(parents=True, exist_ok=True)

        config_path_single = create_test_config(
            test_name=f'particle{particle_d}_single',
            image_dir=image_dir,
            output_dir=output_dir_single,
            num_images=num_images,
            window_sizes=[[16, 16]],  # Match 3-pass final window
            overlaps=[50],
        )

        success = run_ensemble_piv(config_path_single)
        if success:
            data = load_ensemble_results(output_dir_single, num_images=num_images)
            if data:
                results = analyze_per_pass_results(
                    data, f"Particle {particle_d}px - Single-pass 16×16", expected_rs
                )
                particle_results['single_pass'] = results

        # --- 3-pass test (64→32→16) - with predictor warping ---
        print(f"\n--- 3-pass 64→32→16 (with warping) ---")
        output_dir_3pass = test_dir / 'output_3pass'
        output_dir_3pass.mkdir(parents=True, exist_ok=True)

        config_path_3pass = create_test_config(
            test_name=f'particle{particle_d}_3pass',
            image_dir=image_dir,
            output_dir=output_dir_3pass,
            num_images=num_images,
            window_sizes=[[64, 64], [32, 32], [16, 16]],
            overlaps=[50, 50, 50],
        )

        success = run_ensemble_piv(config_path_3pass)
        if success:
            data = load_ensemble_results(output_dir_3pass, num_images=num_images)
            if data:
                results = analyze_per_pass_results(
                    data, f"Particle {particle_d}px - 3-pass 64→32→16", expected_rs
                )
                particle_results['three_pass'] = results

        all_results[particle_d] = particle_results

    # --- Summary comparison table ---
    print("\n" + "=" * 90)
    print("PARTICLE DIAMETER SWEEP SUMMARY")
    print("=" * 90)
    print("\nKey comparison: Single-pass 16×16 vs 3-pass final 16×16")
    print("If both match expected RS → window size/fitting OK, multi-pass OK")
    print("If single-pass OK but 3-pass wrong → multi-pass warping causes RS loss")
    print("If both wrong similarly → inherent to 16×16 window or fitting\n")

    print(f"{'Particle':<10} {'Config':<15} {'Pass':<6} {'Window':<10} {'sig_A_x':<10} {'UU':<10} {'VV':<10} {'UU Err':<10} {'VV Err':<10}")
    print("-" * 95)

    for particle_d, results in all_results.items():
        # Single-pass results
        if 'single_pass' in results and results['single_pass']:
            r = results['single_pass'][0]  # First (only) pass
            uu_err = 100 * (r['UU']['mean'] - 2.0) / 2.0
            vv_err = 100 * (r['VV']['mean'] - 3.0) / 3.0
            print(f"{particle_d}px{'':<6} {'Single 16×16':<15} {1:<6} {'16×16':<10} "
                  f"{r['sig_A_x']['mean']:<10.4f} {r['UU']['mean']:<10.4f} {r['VV']['mean']:<10.4f} "
                  f"{uu_err:+.1f}%{'':<4} {vv_err:+.1f}%")

        # 3-pass results (show final pass for direct comparison)
        if 'three_pass' in results and results['three_pass']:
            for r in results['three_pass']:
                uu_err = 100 * (r['UU']['mean'] - 2.0) / 2.0
                vv_err = 100 * (r['VV']['mean'] - 3.0) / 3.0
                win_size = r.get('win_size')
                if win_size is not None and hasattr(win_size, '__len__'):
                    win_str = f"{win_size[0]}×{win_size[1]}"
                else:
                    win_str = "?×?"
                print(f"{particle_d}px{'':<6} {'3-pass':<15} {r['pass']:<6} {win_str:<10} "
                      f"{r['sig_A_x']['mean']:<10.4f} {r['UU']['mean']:<10.4f} {r['VV']['mean']:<10.4f} "
                      f"{uu_err:+.1f}%{'':<4} {vv_err:+.1f}%")

    # Key insight comparison
    print("\n" + "-" * 95)
    print("SINGLE vs 3-PASS COMPARISON (final 16×16 window):")
    print("-" * 95)
    for particle_d, results in all_results.items():
        single_uu = results.get('single_pass', [{}])[0].get('UU', {}).get('mean', np.nan)
        # Get final (3rd) pass from 3-pass results
        three_pass_final_uu = np.nan
        if 'three_pass' in results and len(results['three_pass']) >= 3:
            three_pass_final_uu = results['three_pass'][2].get('UU', {}).get('mean', np.nan)

        if not np.isnan(single_uu) and not np.isnan(three_pass_final_uu):
            diff = three_pass_final_uu - single_uu
            pct_diff = 100 * diff / single_uu if single_uu > 0 else 0
            print(f"  {particle_d}px: Single UU={single_uu:.4f}, 3-pass UU={three_pass_final_uu:.4f}, "
                  f"Diff={diff:+.4f} ({pct_diff:+.1f}%)")

    return all_results


def main():
    """Run all tests or specific test."""
    print("=" * 70)
    print(f"Reynolds Stress Diagnostic Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nAvailable tests:")
    print("  mean     - Mean-only test (zero RS, non-zero mean)")
    print("  zero     - Zero mean test (RS=2,3, 3 passes)")
    print("  5pass    - 5-pass diagnostic (64→32→32→32→16) to isolate warping effect")
    print("  single   - Single-pass comparison (64, 32, 16 independently)")
    print("  kspace   - Compare Gaussian vs K-space fitting methods")
    print("  particle - Particle diameter sweep (2, 3, 4, 5 px)")
    print("  all      - Run all main tests (mean, zero, 5pass)")
    print("  full     - Run ALL tests including single-pass comparison")

    if len(sys.argv) > 1:
        test = sys.argv[1].lower()
        if test == 'mean':
            run_test_mean_only()
        elif test == 'zero' or test == 'test2' or test == '2':
            run_test2()
        elif test == '5pass' or test == 'test3' or test == '3':
            run_test3()
        elif test == 'single':
            run_single_pass_comparison()
        elif test == 'kspace':
            run_kspace_comparison()
        elif test == 'particle':
            run_particle_diameter_sweep()
        elif test == 'all':
            run_test_mean_only()
            run_test2()
            run_test3()
        elif test == 'full':
            run_test_mean_only()
            run_test2()
            run_test3()
            run_single_pass_comparison()
            run_kspace_comparison()
            run_particle_diameter_sweep()
        else:
            print(f"\nUnknown test: {test}")
            print("Usage: python run_rs_tests.py [mean|zero|5pass|single|kspace|particle|all|full]")
    else:
        # Default: run main tests
        print("\nNo test specified, running main tests...")
        run_test_mean_only()
        run_test2()
        run_test3()

    print("\n" + "=" * 70)
    print("Tests complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
