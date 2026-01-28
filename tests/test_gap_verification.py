"""
Gap Verification Tests for PIV Coordinate System

These tests verify that the unmeasured region (gap) is correctly placed at the
TOP of the image (high physical y values) for both instantaneous and ensemble
PIV processing modes.

The tests use synthetic images with distinct velocity regions:
- GAP region (pixel y < 60, TOP of image): particles move DOWN (uy = -10)
- MEASURED region (pixel y >= 60): particles move UP (uy = +5)

If the gap is correctly at TOP:
- All measured uy values should be ~+5 (positive)
- NO uy values should be ~-10 (negative)
- y[0,:] should have HIGH physical y (near TOP)
- y[-1,:] should have LOW physical y (BOTTOM)

If the gap were incorrectly at BOTTOM:
- Top rows would show uy ~ -10 (WRONG!)
"""

import numpy as np
import pytest
import tempfile
import shutil
import yaml
from pathlib import Path
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import tifffile


# ============================================================================
# Synthetic Image Generation
# ============================================================================

def generate_gap_verification_images(
    output_dir: Path,
    num_pairs: int = 5,
    image_shape: tuple = (500, 500),
    num_particles: int = 3000,
    particle_diameter: float = 4.0,
    seed: int = 42,
):
    """
    Generate images with distinct velocity regions to verify gap placement.

    For 500x500 image with 128x128 window, 50% overlap:
    - Window centers at pixel y: [115.5, 179.5, 243.5, 307.5, 371.5, 435.5]
    - GAP region: pixel y from 0 to ~52 (TOP of image, high physical y)
    - First measured window: centered at pixel y=115.5
    - Last measured window: centered at pixel y=435.5 (BOTTOM)

    Velocity scheme:
    - GAP region (pixel y < 60): uy = -10 (DOWNWARD, should NOT appear in results)
    - Measured region (pixel y >= 60): uy = +5 (UPWARD, should appear in results)
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355

    # Gap boundary (with some margin)
    gap_boundary_pixel = 60

    # Velocities (in pixel displacement)
    # +10 means moving DOWN in image = negative uy in physical coords
    # -5 means moving UP in image = positive uy in physical coords
    gap_velocity_pixel = +10
    measured_velocity_pixel = -5

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx in range(1, num_pairs + 1):
        # Random particle positions
        x_pos = rng.uniform(20, W - 20, num_particles)
        y_pos_pixel = rng.uniform(20, H - 20, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A: particles at original positions
        img_a = np.zeros(image_shape, dtype=np.float32)
        for x, y, intensity in zip(x_pos, y_pos_pixel, intensities):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < H and 0 <= xi < W:
                img_a[yi, xi] += intensity
        img_a = gaussian_filter(img_a, sigma)

        # Frame B: particles displaced based on region
        img_b = np.zeros(image_shape, dtype=np.float32)
        for x, y_pix, intensity in zip(x_pos, y_pos_pixel, intensities):
            if y_pix < gap_boundary_pixel:
                dy_pixel = gap_velocity_pixel  # GAP: move DOWN
            else:
                dy_pixel = measured_velocity_pixel  # MEASURED: move UP

            new_y_pix = y_pix + dy_pixel
            yi, xi = int(round(new_y_pix)), int(round(x))
            if 0 <= yi < H and 0 <= xi < W:
                img_b[yi, xi] += intensity
        img_b = gaussian_filter(img_b, sigma)

        # Normalize to 16-bit range
        img_a = (img_a / img_a.max() * 65535).astype(np.uint16)
        img_b = (img_b / img_b.max() * 65535).astype(np.uint16)

        # Save as TIFF pairs
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif", img_a)
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif", img_b)

    return {
        'gap_velocity': -gap_velocity_pixel,  # Physical uy
        'measured_velocity': -measured_velocity_pixel,  # Physical uy
        'gap_boundary': gap_boundary_pixel,
        'image_shape': image_shape,
    }


def create_test_config(
    source_path: Path,
    base_path: Path,
    mode: str = 'instantaneous',
    num_images: int = 5,
) -> dict:
    """Create a minimal config for testing."""
    config = {
        'paths': {
            'base_paths': [str(base_path)],
            'source_paths': [str(source_path)],
            'active_paths': [0],
            'camera_numbers': [1],
            'camera_count': 1,
            'camera_subfolders': ['Cam1'],
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
            'use_camera_subfolders': True,
        },
        'batches': {'size': num_images},
        'logging': {
            'file': 'pypiv.log',
            'level': 'WARNING',  # Reduce noise in tests
            'console': False,
        },
        'processing': {
            'backend': 'cpu',
            'debug': False,
            'auto_compute_params': False,
            'omp_threads': 2,
            'dask_workers_per_node': 2,
            'dask_threads_per_worker': 1,
            'dask_memory_limit': '2GB',
            'always_batch': True,
            'instantaneous': mode == 'instantaneous',
            'ensemble': mode == 'ensemble',
        },
        'outlier_detection': {
            'enabled': True,
            'methods': [
                {'type': 'peak_mag', 'threshold': 0.2},
            ],
        },
        'infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {'ksize': 3}},
            'final_pass': {'enabled': True, 'method': 'biharmonic', 'parameters': {'ksize': 3}},
        },
        'ensemble_outlier_detection': {
            'enabled': True,
            'methods': [{'type': 'median_2d', 'epsilon': 0.2, 'threshold': 2}],
        },
        'ensemble_infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {'ksize': 3}},
            'final_pass': {'enabled': True, 'method': 'biharmonic', 'parameters': {'ksize': 3}},
        },
        'instantaneous_piv': {
            'window_size': [[128, 128]],
            'overlap': [50],
            'runs': [1],  # 1-indexed
            'time_resolved': False,
            'window_type': 'gaussian',
            'num_peaks': 1,
            'peak_finder': 'gauss6',
            'secondary_peak': False,
        },
        'ensemble_piv': {
            'window_size': [[128, 128], [64, 64]],
            'overlap': [50, 50],
            'type': ['std', 'std'],
            'runs': [1, 2],  # 1-indexed
            'store_planes': False,
            'save_diagnostics': False,
            'sum_window': [16, 16],
            'resume_from_pass': 0,
        },
        'filters': [],
        'masking': {'enabled': False},
    }
    return config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix='piv_gap_test_')
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def gap_test_images(temp_test_dir):
    """Generate gap verification test images."""
    source_path = temp_test_dir / 'source'
    image_dir = source_path / 'Cam1'

    params = generate_gap_verification_images(
        output_dir=image_dir,
        num_pairs=5,
        image_shape=(500, 500),
    )

    return {
        'source_path': source_path,
        'base_path': temp_test_dir / 'output',
        'params': params,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def run_piv_cli(config_path: Path, mode: str):
    """Run PIV processing via CLI."""
    import subprocess
    import sys

    python_exe = sys.executable
    cmd = [python_exe, '-m', 'pivtools_cli.cli', mode]

    result = subprocess.run(
        cmd,
        cwd=config_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"PIV processing failed:\n{result.stderr}")

    return result


def verify_gap_results(
    coords_path: Path,
    result_path: Path,
    expected_measured_velocity: float,
    expected_gap_velocity: float,
    result_key: str = 'piv_result',
    pass_idx: int = 0,
) -> dict:
    """
    Verify that gap is correctly at TOP of image.

    Returns dict with verification results.
    """
    coords = sio.loadmat(str(coords_path))
    result = sio.loadmat(str(result_path))

    # Get coordinates and velocities
    coord_struct = coords['coordinates']
    res_struct = result[result_key]

    # Handle different structures (instantaneous vs ensemble)
    y = coord_struct['y'][0, pass_idx].astype(float)
    uy = res_struct['uy'][0, pass_idx].astype(float)

    # Verification checks
    results = {}

    # 1. Coordinate ordering (should be descending: y[0,0] > y[-1,0])
    results['y_descending'] = y[0, 0] > y[-1, 0]
    results['y_first_row'] = float(y[0, 0])
    results['y_last_row'] = float(y[-1, 0])

    # 2. Gap exclusion check
    all_uy = uy.flatten()
    valid_uy = all_uy[~np.isnan(all_uy)]

    # Count vectors near gap velocity (should be 0)
    tolerance = 3.0
    near_gap_count = np.sum(np.abs(valid_uy - expected_gap_velocity) < tolerance)
    results['gap_velocity_count'] = int(near_gap_count)

    # Count vectors near measured velocity (should be most/all)
    near_measured_count = np.sum(np.abs(valid_uy - expected_measured_velocity) < tolerance)
    results['measured_velocity_count'] = int(near_measured_count)
    results['total_vectors'] = len(valid_uy)

    # 3. Positive velocity check (measured region has positive uy)
    positive_count = np.sum(valid_uy > 0)
    results['positive_uy_count'] = int(positive_count)
    results['positive_uy_fraction'] = float(positive_count / len(valid_uy))

    # 4. Mean velocity
    results['mean_uy'] = float(np.nanmean(uy))

    return results


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.slow
class TestGapVerificationInstantaneous:
    """Test that instantaneous PIV places gap at TOP of image."""

    def test_gap_at_top(self, gap_test_images, temp_test_dir):
        """
        Verify gap is at TOP for instantaneous PIV.

        Expected behavior:
        - y[0,:] should have HIGH physical y (near 384)
        - y[-1,:] should have LOW physical y (near 64)
        - All uy should be ~+5 (measured region)
        - No uy should be ~-10 (gap region excluded)
        """
        # Create config
        config = create_test_config(
            source_path=gap_test_images['source_path'],
            base_path=gap_test_images['base_path'],
            mode='instantaneous',
            num_images=5,
        )

        config_path = temp_test_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run PIV
        run_piv_cli(config_path, 'instantaneous')

        # Locate output files
        output_dir = (
            gap_test_images['base_path'] /
            'uncalibrated_piv' / '5' / 'Cam1' / 'instantaneous'
        )
        coords_path = output_dir / 'coordinates.mat'
        result_path = output_dir / '00001.mat'

        assert coords_path.exists(), f"Coordinates not found: {coords_path}"
        assert result_path.exists(), f"Result not found: {result_path}"

        # Verify results
        params = gap_test_images['params']
        results = verify_gap_results(
            coords_path=coords_path,
            result_path=result_path,
            expected_measured_velocity=params['measured_velocity'],  # +5
            expected_gap_velocity=params['gap_velocity'],  # -10
            result_key='piv_result',
            pass_idx=0,
        )

        # Assertions
        assert results['y_descending'], (
            f"y should be descending but y[0,0]={results['y_first_row']:.1f}, "
            f"y[-1,0]={results['y_last_row']:.1f}"
        )

        assert results['y_first_row'] > 300, (
            f"First row should have high y (near TOP), got {results['y_first_row']:.1f}"
        )

        assert results['y_last_row'] < 150, (
            f"Last row should have low y (BOTTOM), got {results['y_last_row']:.1f}"
        )

        assert results['gap_velocity_count'] == 0, (
            f"Gap region velocity (-10) should not appear in results, "
            f"but found {results['gap_velocity_count']} vectors"
        )

        assert results['positive_uy_fraction'] > 0.9, (
            f"Most uy should be positive (measured region), "
            f"but only {results['positive_uy_fraction']*100:.1f}% are positive"
        )

        assert results['measured_velocity_count'] > results['total_vectors'] * 0.8, (
            f"Most vectors should be near measured velocity (+5), "
            f"but only {results['measured_velocity_count']}/{results['total_vectors']}"
        )


@pytest.mark.slow
class TestGapVerificationEnsemble:
    """Test that ensemble PIV places gap at TOP of image."""

    def test_gap_at_top(self, gap_test_images, temp_test_dir):
        """
        Verify gap is at TOP for ensemble PIV.

        Expected behavior:
        - y[0,:] should have HIGH physical y
        - y[-1,:] should have LOW physical y
        - All uy should be ~+5 (measured region)
        - No uy should be ~-10 (gap region excluded)
        """
        # Create config
        config = create_test_config(
            source_path=gap_test_images['source_path'],
            base_path=gap_test_images['base_path'],
            mode='ensemble',
            num_images=5,
        )

        config_path = temp_test_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run PIV
        run_piv_cli(config_path, 'ensemble')

        # Locate output files
        output_dir = (
            gap_test_images['base_path'] /
            'uncalibrated_piv' / '5' / 'Cam1' / 'ensemble'
        )
        coords_path = output_dir / 'coordinates.mat'
        result_path = output_dir / 'ensemble_result.mat'

        assert coords_path.exists(), f"Coordinates not found: {coords_path}"
        assert result_path.exists(), f"Result not found: {result_path}"

        # Verify results (use pass 1 = 64x64 window)
        params = gap_test_images['params']
        results = verify_gap_results(
            coords_path=coords_path,
            result_path=result_path,
            expected_measured_velocity=params['measured_velocity'],  # +5
            expected_gap_velocity=params['gap_velocity'],  # -10
            result_key='ensemble_result',
            pass_idx=1,  # Second pass (64x64)
        )

        # Assertions
        assert results['y_descending'], (
            f"y should be descending but y[0,0]={results['y_first_row']:.1f}, "
            f"y[-1,0]={results['y_last_row']:.1f}"
        )

        assert results['y_first_row'] > 350, (
            f"First row should have high y (near TOP), got {results['y_first_row']:.1f}"
        )

        assert results['y_last_row'] < 100, (
            f"Last row should have low y (BOTTOM), got {results['y_last_row']:.1f}"
        )

        assert results['gap_velocity_count'] == 0, (
            f"Gap region velocity (-10) should not appear in results, "
            f"but found {results['gap_velocity_count']} vectors"
        )

        assert results['positive_uy_fraction'] > 0.9, (
            f"Most uy should be positive (measured region), "
            f"but only {results['positive_uy_fraction']*100:.1f}% are positive"
        )


# ============================================================================
# Quick Sanity Check (faster, for CI)
# ============================================================================

class TestCoordinateConvention:
    """Quick tests for coordinate convention without full PIV run."""

    def test_window_centers_ascending(self):
        """Verify window centers are computed in ascending order."""
        from pivtools_core.window_utils import compute_window_centers

        result = compute_window_centers(
            image_shape=(500, 500),
            window_size=(128, 128),
            overlap=50,
        )

        # win_ctrs_y should be ascending
        assert result.win_ctrs_y[0] < result.win_ctrs_y[-1], (
            f"win_ctrs_y should be ascending, got {result.win_ctrs_y}"
        )

        # Gap should be at TOP (low pixel y = high physical y)
        # First center should be > half window from top edge
        half_window = 64
        assert result.win_ctrs_y[0] > half_window, (
            f"First window center {result.win_ctrs_y[0]} should be > {half_window} "
            f"(gap at top)"
        )

    def test_physical_y_conversion(self):
        """Test pixel to physical y conversion."""
        H = 500

        # Pixel y = 0 (top of image) should give high physical y
        pixel_y_top = 0
        physical_y_top = (H - 1) - pixel_y_top
        assert physical_y_top == 499, f"Top pixel should give physical y=499, got {physical_y_top}"

        # Pixel y = 499 (bottom of image) should give low physical y
        pixel_y_bottom = 499
        physical_y_bottom = (H - 1) - pixel_y_bottom
        assert physical_y_bottom == 0, f"Bottom pixel should give physical y=0, got {physical_y_bottom}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
