"""
Batch Size Consistency Test

Verifies that ensemble PIV produces identical results regardless of batch size.
This tests that the accumulation/reduction logic works correctly.

Tests:
- batch_size=20 (5 batches for 100 images)
- batch_size=100 (1 batch for 100 images)

Results should be IDENTICAL (within floating-point tolerance).
"""
import sys
import os
import yaml
import shutil
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.reynolds_stress_test.generate_rs_images import generate_rs_test_images


def create_config(test_dir: Path, batch_size: int, num_images: int = 100):
    """Create config with specified batch size."""
    image_dir = test_dir / 'Cam1'
    output_dir = test_dir / f'output_batch{batch_size}'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'paths': {
            'base_paths': [str(output_dir)],
            'source_paths': [str(test_dir)],
            'active_paths': [0],
            'camera_numbers': [1],
            'camera_count': 1,
            'camera_subfolders': ['Cam1'],
        },
        'images': {
            'num_images': num_images,
            'image_format': ['B%05d_A.tif', 'B%05d_B.tif'],
            'vector_format': ['%05d.mat'],
            'dtype': 'float32',
            'start_index': 1,
            'frame_stride': 0,
            'pair_stride': 1,
            'pairing_preset': 'ab_format',
            'image_type': 'standard',
            'use_camera_subfolders': True,
        },
        'batches': {
            'size': batch_size,  # THIS IS THE KEY VARIABLE
        },
        'logging': {
            'file': f'batch{batch_size}.log',
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
            'enabled': False,
            'methods': [],
        },
        'infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {}},
            'final_pass': {'enabled': False, 'method': 'biharmonic', 'parameters': {}},
        },
        'ensemble_outlier_detection': {
            'enabled': False,
            'methods': [],
        },
        'ensemble_infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {}},
            'final_pass': {'enabled': False, 'method': 'biharmonic', 'parameters': {}},
        },
        'ensemble_piv': {
            'fit_offset': False,
            'window_size': [[64, 64], [32, 32]],  # 2 passes
            'overlap': [50, 50],
            'type': ['std', 'std'],
            'runs': [1, 2],
            'store_planes': False,
            'save_diagnostics': False,
            'sum_window': [16, 16],
            'resume_from_pass': 0,
            'window_type': 'square',
        },
        'masking': {'enabled': False},
        'filters': [],
    }

    config_path = output_dir / f'config_batch{batch_size}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path, output_dir


def run_piv(config_path: Path):
    """Run ensemble PIV."""
    pivtools_dir = Path(__file__).parent.parent.parent
    temp_config = pivtools_dir / 'config.yaml'
    original_config = None

    if temp_config.exists():
        original_config = temp_config.read_text()

    try:
        shutil.copy2(config_path, temp_config)
        result = subprocess.run(
            [sys.executable, '-m', 'pivtools_core.ensemble'],
            cwd=str(pivtools_dir),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    finally:
        if original_config is not None:
            temp_config.write_text(original_config)
        elif temp_config.exists():
            temp_config.unlink()


def load_results(output_dir: Path, num_images: int):
    """Load results from .mat file."""
    import scipy.io as sio

    for path in output_dir.rglob('ensemble_result.mat'):
        mat_data = sio.loadmat(str(path), struct_as_record=True)
        if 'ensemble_result' in mat_data:
            return mat_data['ensemble_result']
    return None


def compare_results(results1, results2, label1: str, label2: str):
    """Compare two result sets and report differences."""
    print(f"\n{'='*70}")
    print(f"COMPARING: {label1} vs {label2}")
    print('='*70)

    n_passes = results1.shape[1]
    all_match = True

    for pass_idx in range(n_passes):
        print(f"\n--- Pass {pass_idx + 1} ---")

        s1 = results1[0, pass_idx]
        s2 = results2[0, pass_idx]

        for field in ['ux', 'uy', 'UU_stress', 'VV_stress', 'sig_AB_x', 'sig_AB_y']:
            try:
                v1 = np.array(s1[field]).squeeze()
                v2 = np.array(s2[field]).squeeze()

                if v1.shape != v2.shape:
                    print(f"  {field}: SHAPE MISMATCH {v1.shape} vs {v2.shape}")
                    all_match = False
                    continue

                # Compare values
                valid_mask = np.isfinite(v1) & np.isfinite(v2)
                if not valid_mask.any():
                    print(f"  {field}: No valid values to compare")
                    continue

                diff = np.abs(v1[valid_mask] - v2[valid_mask])
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                # Relative difference
                scale = np.maximum(np.abs(v1[valid_mask]), np.abs(v2[valid_mask]))
                scale = np.maximum(scale, 1e-10)  # Avoid div by zero
                rel_diff = diff / scale
                max_rel_diff = np.max(rel_diff)

                # Values
                mean1 = np.mean(v1[valid_mask])
                mean2 = np.mean(v2[valid_mask])

                # Tolerance check (allow tiny floating-point differences from warping)
                # 1e-4 pixels is ~0.01% of typical displacement - effectively identical
                is_match = max_diff < 1e-4 or max_rel_diff < 1e-4
                status = "✅ MATCH" if is_match else "❌ DIFFER"

                if not is_match:
                    all_match = False

                print(f"  {field}: {status}")
                print(f"    {label1} mean: {mean1:.6f}")
                print(f"    {label2} mean: {mean2:.6f}")
                print(f"    Max abs diff: {max_diff:.2e}, Max rel diff: {max_rel_diff:.2e}")

            except (ValueError, KeyError, IndexError) as e:
                print(f"  {field}: Error - {e}")
                all_match = False

    return all_match


def main():
    print("="*70)
    print("BATCH SIZE CONSISTENCY TEST")
    print("="*70)
    print("\nThis test verifies that batch_size=20 and batch_size=100")
    print("produce IDENTICAL results for the same input images.")
    print("="*70)

    test_dir = Path(__file__).parent / 'batch_consistency_test'

    # Clean previous run
    if test_dir.exists():
        shutil.rmtree(test_dir)

    image_dir = test_dir / 'Cam1'

    # Generate images ONCE (same images for both tests)
    print("\n1. Generating 100 synthetic images...")
    generate_rs_test_images(
        output_dir=image_dir,
        num_pairs=100,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=1.5,
        std_dy=2.0,
        seed=12345,  # Fixed seed for reproducibility
    )

    # Run with batch_size=20
    print("\n2. Running with batch_size=20...")
    config20, output20 = create_config(test_dir, batch_size=20, num_images=100)
    success20 = run_piv(config20)
    if not success20:
        print("ERROR: batch_size=20 failed!")
        return

    # Run with batch_size=100
    print("\n3. Running with batch_size=100...")
    config100, output100 = create_config(test_dir, batch_size=100, num_images=100)
    success100 = run_piv(config100)
    if not success100:
        print("ERROR: batch_size=100 failed!")
        return

    # Load and compare results
    print("\n4. Loading results...")
    results20 = load_results(output20, 100)
    results100 = load_results(output100, 100)

    if results20 is None or results100 is None:
        print("ERROR: Could not load results!")
        return

    # Compare
    all_match = compare_results(results20, results100, "batch=20", "batch=100")

    print("\n" + "="*70)
    if all_match:
        print("✅ SUCCESS: Results are IDENTICAL regardless of batch size")
    else:
        print("❌ FAILURE: Results DIFFER between batch sizes!")
    print("="*70)

    return all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
