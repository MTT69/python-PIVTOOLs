"""
Test Interpolation Method Effects on Reynolds Stress

This script tests how different cv2 interpolation methods affect Reynolds stress
measurements in ensemble PIV. The concern is that cubic interpolation may smooth
particle images, affecting the measured PSF and turbulence.

Interpolation methods tested:
- cv2.INTER_NEAREST (fastest, lowest quality, no smoothing)
- cv2.INTER_LINEAR (bilinear, moderate smoothing)
- cv2.INTER_CUBIC (bicubic, current default, most smoothing)

Two interpolation locations are tested:
1. Image warping: fused_symmetric_warp_batch (libfusedwarp C kernel, bicubic/Lanczos-3)
2. Predictor field: cv2.remap in _get_im_mesh() (predictor → current pass grid)

Usage:
    python test_interpolation_rs.py [quick|full|matrix]

    quick  - Run single interpolation comparison (NEAREST vs CUBIC)
    full   - Run all 9 combinations with 500 images
    matrix - Run focused 3x1 test (vary image warp only, fix predictor)
"""
import sys
import os
import yaml
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_rs_images import generate_rs_test_images


@dataclass
class PassMetrics:
    """Metrics captured for each pass."""
    pass_idx: int
    window_size: Tuple[int, int]

    # Autocorrelation (particle contribution)
    sig_A_x_mean: float
    sig_A_x_std: float
    sig_A_y_mean: float
    sig_A_y_std: float

    # Cross-correlation (includes RS)
    sig_AB_x_mean: float
    sig_AB_x_std: float
    sig_AB_y_mean: float
    sig_AB_y_std: float

    # Reynolds stress
    UU_mean: float
    UU_std: float
    VV_mean: float
    VV_std: float

    # Mean velocity
    ux_mean: float
    uy_mean: float


@dataclass
class TestResult:
    """Result of a single interpolation test configuration."""
    image_warp_interp: str
    predictor_interp: str
    passes: List[PassMetrics]
    expected_UU: float
    expected_VV: float


def create_test_config(
    test_name: str,
    image_dir: Path,
    output_dir: Path,
    num_images: int = 100,
    window_sizes: list = None,
    overlaps: list = None,
    image_warp_interp: str = 'cubic',
    predictor_interp: str = 'cubic',
) -> Path:
    """Create a YAML config file for the test."""
    if window_sizes is None:
        window_sizes = [[64, 64], [32, 32], [16, 16]]
    if overlaps is None:
        # Match number of overlaps to window sizes
        overlaps = [50] * len(window_sizes)

    config = {
        'paths': {
            'base_paths': [str(output_dir)],
            'source_paths': [str(image_dir.parent)],
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
        'batches': {'size': 10},
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
        'outlier_detection': {'enabled': False, 'methods': []},
        'infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {}},
            'final_pass': {'enabled': False, 'method': 'biharmonic', 'parameters': {}},
        },
        'ensemble_outlier_detection': {'enabled': False, 'methods': []},
        'ensemble_infilling': {
            'mid_pass': {'method': 'biharmonic', 'parameters': {}},
            'final_pass': {'enabled': False, 'method': 'biharmonic', 'parameters': {}},
        },
        'ensemble_piv': {
            'fit_offset': False,
            'fit_method': 'gaussian',
            'kspace_snr_threshold': 3.0,
            'window_size': window_sizes,
            'overlap': overlaps,
            'type': ['std'] * len(window_sizes),
            'runs': list(range(1, len(window_sizes) + 1)),
            'store_planes': False,  # Save space
            'save_diagnostics': False,
            'sum_window': [16, 16],
            'resume_from_pass': 0,
            'window_type': 'square',
            # Interpolation settings for image warping and predictor field
            'image_warp_interpolation': image_warp_interp,
            'predictor_interpolation': predictor_interp,
        },
        'masking': {'enabled': False},
        'filters': [],
    }

    config_path = output_dir / f'{test_name}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_ensemble_piv(
    config_path: Path,
    image_warp_interp: str,
    predictor_interp: str,
) -> bool:
    """Run ensemble PIV with config-based interpolation settings."""
    pivtools_dir = Path(__file__).parent.parent.parent

    # Copy config to pivtools_dir as config.yaml
    temp_config = pivtools_dir / 'config.yaml'
    original_config = None
    if temp_config.exists():
        original_config = temp_config.read_text()

    try:
        shutil.copy2(config_path, temp_config)

        # Run ensemble PIV
        cmd = [sys.executable, '-m', 'pivtools_core.ensemble']

        print(f"\n  Running PIV: image_warp={image_warp_interp}, predictor={predictor_interp}")

        result = subprocess.run(
            cmd,
            cwd=str(pivtools_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Always show some output for debugging
        if result.stdout:
            stdout_lines = result.stdout.strip().split('\n')
            # Show first and last few lines
            if len(stdout_lines) > 40:
                print(f"  [stdout first 20 lines]")
                for line in stdout_lines[:20]:
                    print(f"    {line}")
                print(f"  [stdout last 20 lines]")
                for line in stdout_lines[-20:]:
                    print(f"    {line}")
            else:
                for line in stdout_lines:
                    print(f"  {line}")

        # Show stderr too
        if result.stderr:
            print(f"  [stderr]")
            for line in result.stderr.strip().split('\n')[-20:]:
                print(f"    {line}")

        if result.returncode != 0:
            print(f"  ERROR (code {result.returncode})")
            return False

        return True

    finally:
        if original_config is not None:
            temp_config.write_text(original_config)
        elif temp_config.exists():
            temp_config.unlink()


def load_ensemble_results(output_dir: Path, num_images: int = 100) -> Optional[dict]:
    """Load ensemble results from .mat file."""
    import scipy.io as sio

    # Search for result file
    result_path = None
    for path in output_dir.rglob('ensemble_result.mat'):
        result_path = path
        break

    if result_path is None:
        print(f"  Result file not found in {output_dir}")
        return None

    mat_data = sio.loadmat(str(result_path), struct_as_record=True)

    if 'ensemble_result' not in mat_data:
        print(f"  No 'ensemble_result' key in mat file")
        return None

    ensemble_data = mat_data['ensemble_result']
    n_passes = ensemble_data.shape[1] if len(ensemble_data.shape) > 1 else ensemble_data.size

    results = {}
    for pass_idx in range(n_passes):
        pass_data = {}
        struct = ensemble_data[0, pass_idx] if len(ensemble_data.shape) > 1 else ensemble_data.flat[pass_idx]

        for field in ['ux', 'uy', 'UU_stress', 'VV_stress', 'UV_stress',
                      'sig_AB_x', 'sig_AB_y', 'sig_AB_xy',
                      'sig_A_x', 'sig_A_y', 'sig_A_xy', 'window_size']:
            try:
                val = struct[field]
                if val is not None and hasattr(val, 'shape'):
                    while val.ndim > 0 and val.shape[0] == 1 and val.ndim > 2:
                        val = val[0]
                    pass_data[field] = val
                else:
                    pass_data[field] = val
            except (ValueError, IndexError, KeyError):
                pass_data[field] = None

        results[f'pass_{pass_idx + 1}'] = pass_data

    return results


def extract_metrics(data: dict, expected_UU: float, expected_VV: float, edge_trim: int = 2) -> List[PassMetrics]:
    """Extract metrics from result data."""
    metrics_list = []

    pass_keys = sorted([k for k in data.keys() if k.startswith('pass_')],
                       key=lambda x: int(x.split('_')[1]))

    for pass_key in pass_keys:
        pass_data = data[pass_key]
        pass_num = int(pass_key.split('_')[1])

        def process_field(field):
            if field is None:
                return None
            arr = np.array(field).squeeze()
            if edge_trim > 0 and arr.ndim == 2:
                if arr.shape[0] > 2 * edge_trim and arr.shape[1] > 2 * edge_trim:
                    arr = arr[edge_trim:-edge_trim, edge_trim:-edge_trim]
            return arr

        def field_mean_std(arr):
            if arr is None:
                return np.nan, np.nan
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return np.nan, np.nan
            return np.mean(valid), np.std(valid)

        sig_A_x = process_field(pass_data.get('sig_A_x'))
        sig_A_y = process_field(pass_data.get('sig_A_y'))
        sig_AB_x = process_field(pass_data.get('sig_AB_x'))
        sig_AB_y = process_field(pass_data.get('sig_AB_y'))
        uu = process_field(pass_data.get('UU_stress'))
        vv = process_field(pass_data.get('VV_stress'))
        ux = process_field(pass_data.get('ux'))
        uy = process_field(pass_data.get('uy'))

        win_size = pass_data.get('window_size')
        if win_size is not None:
            win_size = tuple(np.array(win_size).flatten()[:2].astype(int))
        else:
            win_size = (0, 0)

        sig_A_x_mean, sig_A_x_std = field_mean_std(sig_A_x)
        sig_A_y_mean, sig_A_y_std = field_mean_std(sig_A_y)
        sig_AB_x_mean, sig_AB_x_std = field_mean_std(sig_AB_x)
        sig_AB_y_mean, sig_AB_y_std = field_mean_std(sig_AB_y)
        UU_mean, UU_std = field_mean_std(uu)
        VV_mean, VV_std = field_mean_std(vv)
        ux_mean, _ = field_mean_std(ux)
        uy_mean, _ = field_mean_std(uy)

        metrics_list.append(PassMetrics(
            pass_idx=pass_num,
            window_size=win_size,
            sig_A_x_mean=sig_A_x_mean,
            sig_A_x_std=sig_A_x_std,
            sig_A_y_mean=sig_A_y_mean,
            sig_A_y_std=sig_A_y_std,
            sig_AB_x_mean=sig_AB_x_mean,
            sig_AB_x_std=sig_AB_x_std,
            sig_AB_y_mean=sig_AB_y_mean,
            sig_AB_y_std=sig_AB_y_std,
            UU_mean=UU_mean,
            UU_std=UU_std,
            VV_mean=VV_mean,
            VV_std=VV_std,
            ux_mean=ux_mean,
            uy_mean=uy_mean,
        ))

    return metrics_list


def run_interpolation_test(
    test_dir: Path,
    num_images: int = 500,
    image_warp_interps: List[str] = None,
    predictor_interps: List[str] = None,
    window_sizes: list = None,
    expected_UU: float = 2.0,
    expected_VV: float = 3.0,
    seed: int = 42,
) -> Dict[str, TestResult]:
    """
    Run ensemble PIV with all interpolation combinations.

    Returns dict mapping config key to TestResult.
    """
    if image_warp_interps is None:
        image_warp_interps = ['nearest', 'linear', 'cubic']
    if predictor_interps is None:
        predictor_interps = ['cubic']  # Fix predictor, vary image warp
    if window_sizes is None:
        window_sizes = [[64, 64], [32, 32], [16, 16]]

    image_dir = test_dir / 'Cam1'

    # Generate images once (shared across all configs)
    if not image_dir.exists() or len(list(image_dir.glob('*.tif'))) < num_images * 2:
        print(f"\nGenerating {num_images} synthetic image pairs...")
        image_dir.mkdir(parents=True, exist_ok=True)

        stats = generate_rs_test_images(
            output_dir=image_dir,
            num_pairs=num_images,
            image_shape=(256, 256),
            particle_diameter=2.0,
            mean_dx=0.0,
            mean_dy=0.0,
            std_dx=np.sqrt(expected_UU),
            std_dy=np.sqrt(expected_VV),
            seed=seed,
        )
        print(f"  Generated images with target RS: UU={expected_UU}, VV={expected_VV}")
    else:
        print(f"\nUsing existing images in {image_dir}")

    results = {}

    for img_interp in image_warp_interps:
        for pred_interp in predictor_interps:
            key = f"img_{img_interp}_pred_{pred_interp}"
            print(f"\n{'='*60}")
            print(f"Testing: {key}")
            print(f"{'='*60}")

            output_dir = test_dir / f'output_{key}'
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create config with interpolation settings baked in
            config_path = create_test_config(
                test_name=f'interp_{key}',
                image_dir=image_dir,
                output_dir=output_dir,
                num_images=num_images,
                window_sizes=window_sizes,
                image_warp_interp=img_interp,
                predictor_interp=pred_interp,
            )

            # Run PIV (uses config-based interpolation settings)
            success = run_ensemble_piv(
                config_path, img_interp, pred_interp
            )

            if not success:
                print(f"  FAILED: {key}")
                continue

            # Load and analyze results
            data = load_ensemble_results(output_dir, num_images)
            if data is None:
                print(f"  No results for {key}")
                continue

            metrics = extract_metrics(data, expected_UU, expected_VV)

            results[key] = TestResult(
                image_warp_interp=img_interp,
                predictor_interp=pred_interp,
                passes=metrics,
                expected_UU=expected_UU,
                expected_VV=expected_VV,
            )

            # Print quick summary
            for m in metrics:
                uu_err = 100 * (m.UU_mean - expected_UU) / expected_UU if expected_UU > 0 else 0
                vv_err = 100 * (m.VV_mean - expected_VV) / expected_VV if expected_VV > 0 else 0
                print(f"  Pass {m.pass_idx} ({m.window_size}): "
                      f"UU={m.UU_mean:.3f} ({uu_err:+.1f}%), "
                      f"VV={m.VV_mean:.3f} ({vv_err:+.1f}%)")

    return results


def generate_comparison_report(results: Dict[str, TestResult], output_path: Path):
    """Generate markdown comparison report."""
    if not results:
        print("No results to report!")
        return

    expected_UU = list(results.values())[0].expected_UU
    expected_VV = list(results.values())[0].expected_VV

    lines = [
        "# Interpolation Method Comparison Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nExpected RS: UU={expected_UU}, VV={expected_VV}",
        "",
        "## Summary Table",
        "",
        "| Config | Pass | Window | sig_A_x | sig_A_y | UU | VV | UU Error | VV Error |",
        "|--------|------|--------|---------|---------|-----|-----|----------|----------|",
    ]

    for key, result in sorted(results.items()):
        for m in result.passes:
            uu_err = 100 * (m.UU_mean - expected_UU) / expected_UU if expected_UU > 0 else 0
            vv_err = 100 * (m.VV_mean - expected_VV) / expected_VV if expected_VV > 0 else 0
            win_str = f"{m.window_size[0]}x{m.window_size[1]}"
            lines.append(
                f"| {key} | {m.pass_idx} | {win_str} | "
                f"{m.sig_A_x_mean:.4f} | {m.sig_A_y_mean:.4f} | "
                f"{m.UU_mean:.4f} | {m.VV_mean:.4f} | "
                f"{uu_err:+.1f}% | {vv_err:+.1f}% |"
            )

    # Per-pass comparison across methods
    lines.extend([
        "",
        "## Per-Pass Comparison",
        "",
    ])

    # Group by pass
    all_passes = set()
    for result in results.values():
        for m in result.passes:
            all_passes.add(m.pass_idx)

    for pass_idx in sorted(all_passes):
        lines.append(f"### Pass {pass_idx}")
        lines.append("")
        lines.append("| Image Warp | Predictor | sig_A_x | UU | UU Error | VV | VV Error |")
        lines.append("|------------|-----------|---------|-----|----------|-----|----------|")

        for key, result in sorted(results.items()):
            for m in result.passes:
                if m.pass_idx == pass_idx:
                    uu_err = 100 * (m.UU_mean - expected_UU) / expected_UU if expected_UU > 0 else 0
                    vv_err = 100 * (m.VV_mean - expected_VV) / expected_VV if expected_VV > 0 else 0
                    lines.append(
                        f"| {result.image_warp_interp} | {result.predictor_interp} | "
                        f"{m.sig_A_x_mean:.4f} | {m.UU_mean:.4f} | {uu_err:+.1f}% | "
                        f"{m.VV_mean:.4f} | {vv_err:+.1f}% |"
                    )
        lines.append("")

    # Key findings
    lines.extend([
        "## Key Observations",
        "",
        "### sig_A (Particle Size) Changes",
        "",
    ])

    # Check if sig_A differs between methods
    for pass_idx in sorted(all_passes):
        sig_a_values = {}
        for key, result in results.items():
            for m in result.passes:
                if m.pass_idx == pass_idx:
                    sig_a_values[result.image_warp_interp] = m.sig_A_x_mean

        if len(sig_a_values) > 1:
            min_k = min(sig_a_values, key=sig_a_values.get)
            max_k = max(sig_a_values, key=sig_a_values.get)
            diff = sig_a_values[max_k] - sig_a_values[min_k]
            pct_diff = 100 * diff / sig_a_values[min_k] if sig_a_values[min_k] > 0 else 0
            lines.append(f"- Pass {pass_idx}: sig_A_x ranges from {sig_a_values[min_k]:.4f} ({min_k}) "
                        f"to {sig_a_values[max_k]:.4f} ({max_k}), diff={pct_diff:+.1f}%")

    lines.extend([
        "",
        "### Reynolds Stress Changes",
        "",
    ])

    for pass_idx in sorted(all_passes):
        uu_values = {}
        for key, result in results.items():
            for m in result.passes:
                if m.pass_idx == pass_idx:
                    uu_values[result.image_warp_interp] = m.UU_mean

        if len(uu_values) > 1:
            min_k = min(uu_values, key=uu_values.get)
            max_k = max(uu_values, key=uu_values.get)
            diff = uu_values[max_k] - uu_values[min_k]
            pct_diff = 100 * diff / expected_UU if expected_UU > 0 else 0
            lines.append(f"- Pass {pass_idx}: UU ranges from {uu_values[min_k]:.4f} ({min_k}) "
                        f"to {uu_values[max_k]:.4f} ({max_k}), diff={pct_diff:+.1f}% of expected")

    # Write report
    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    print(f"\nReport saved to: {output_path}")

    # Also print to console
    print("\n" + "=" * 80)
    print(report_text)


def run_quick_test():
    """Quick test: Compare NEAREST vs CUBIC with 100 images."""
    print("\n" + "=" * 70)
    print("QUICK TEST: NEAREST vs CUBIC comparison (100 images)")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_interp_quick'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    results = run_interpolation_test(
        test_dir=test_dir,
        num_images=100,
        image_warp_interps=['nearest', 'cubic'],
        predictor_interps=['cubic'],  # Fix predictor
        window_sizes=[[64, 64], [32, 32]],  # 2 passes only
        expected_UU=2.0,
        expected_VV=3.0,
        seed=42,
    )

    report_path = test_dir / 'INTERPOLATION_COMPARISON.md'
    generate_comparison_report(results, report_path)

    return results


def run_full_test():
    """Full test: All 9 combinations with 500 images."""
    print("\n" + "=" * 70)
    print("FULL TEST: All 9 interpolation combinations (500 images)")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_interp_full'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    results = run_interpolation_test(
        test_dir=test_dir,
        num_images=500,
        image_warp_interps=['nearest', 'linear', 'cubic'],
        predictor_interps=['nearest', 'linear', 'cubic'],
        window_sizes=[[64, 64], [32, 32], [16, 16]],
        expected_UU=2.0,
        expected_VV=3.0,
        seed=42,
    )

    report_path = test_dir / 'INTERPOLATION_COMPARISON.md'
    generate_comparison_report(results, report_path)

    return results


def run_matrix_test():
    """Matrix test: Vary image warp only, fix predictor to cubic."""
    print("\n" + "=" * 70)
    print("MATRIX TEST: Vary image warp (NEAREST/LINEAR/CUBIC), fix predictor=CUBIC")
    print("=" * 70)

    test_dir = Path(__file__).parent / 'rs_interp_matrix'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    results = run_interpolation_test(
        test_dir=test_dir,
        num_images=500,
        image_warp_interps=['nearest', 'linear', 'cubic'],
        predictor_interps=['cubic'],  # Fix predictor
        window_sizes=[[64, 64], [32, 32], [16, 16]],
        expected_UU=2.0,
        expected_VV=3.0,
        seed=42,
    )

    report_path = test_dir / 'INTERPOLATION_COMPARISON.md'
    generate_comparison_report(results, report_path)

    return results


def main():
    print("=" * 70)
    print(f"Interpolation RS Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nAvailable tests:")
    print("  quick  - NEAREST vs CUBIC, 100 images, 2 passes")
    print("  matrix - All 3 image interps, predictor=cubic, 500 images")
    print("  full   - All 9 combinations, 500 images")

    if len(sys.argv) > 1:
        test = sys.argv[1].lower()
        if test == 'quick':
            run_quick_test()
        elif test == 'matrix':
            run_matrix_test()
        elif test == 'full':
            run_full_test()
        else:
            print(f"\nUnknown test: {test}")
            print("Usage: python test_interpolation_rs.py [quick|matrix|full]")
    else:
        # Default: run quick test
        print("\nNo test specified, running quick test...")
        run_quick_test()

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
