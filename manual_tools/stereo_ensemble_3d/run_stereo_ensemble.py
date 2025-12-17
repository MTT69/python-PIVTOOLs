#!/usr/bin/env python3
"""
End-to-End Stereo Ensemble PIV Pipeline
========================================

This script demonstrates the complete workflow for extracting Reynolds stresses
from synthetic stereo PIV images:

1. Generate synthetic stereo image pairs with known Reynolds stress tensor
2. Reconstruct 3D volumes using MLOS
3. Accumulate ensemble correlations in frequency domain
4. Fit stacked 3D Gaussian to extract Reynolds stress

Usage:
    python run_stereo_ensemble.py --num-pairs 1000 --particles 500
    python run_stereo_ensemble.py --quick  # Fast test with fewer pairs
"""

import argparse
import time
import numpy as np
from typing import Tuple

# Local imports
from stereo_ensemble_generator import StereoEnsembleConfig, StereoEnsembleGenerator
from correlation_3d import StereoMLOSReconstructor, EnsembleAccumulator3D
from gaussian_fit_stacked_3d import fit_stacked_gaussian_3d, StackedGaussianResult3D
from debug_visualization import run_debug_visualization


def format_matrix(mat: np.ndarray, precision: int = 4) -> str:
    """Format a matrix for display."""
    lines = []
    for row in mat:
        row_str = " ".join(f"{val:8.{precision}f}" for val in row)
        lines.append(f"  [{row_str}]")
    return "\n".join(lines)


def run_pipeline(
    config: StereoEnsembleConfig,
    verbose: bool = True,
    images_dir: str = None,
    image_format: str = 'tiff'
) -> Tuple[StackedGaussianResult3D, float]:
    """
    Run the complete stereo ensemble PIV pipeline.

    Parameters
    ----------
    config : StereoEnsembleConfig
        Configuration parameters
    verbose : bool
        Print progress information
    images_dir : str, optional
        Directory for images. If exists, loads from it and generates any
        missing pairs. If doesn't exist, creates it and saves all images.
    image_format : str
        Format for saving images ('npz' or 'tiff')

    Returns
    -------
    result : StackedGaussianResult3D
        Fitted parameters including Reynolds stress tensor
    elapsed_time : float
        Total elapsed time in seconds
    """
    from pathlib import Path

    start_time = time.time()

    # Initialize components
    if verbose:
        print("Initializing generator and reconstructor...")

    generator = StereoEnsembleGenerator(config)
    reconstructor = StereoMLOSReconstructor(
        generator.cameras,
        config.volume_size,
        config.scale_px_per_mm
    )
    accumulator = EnsembleAccumulator3D(config.volume_size)

    num_pairs = config.num_image_pairs

    # Determine image source
    if images_dir:
        images_path = Path(images_dir)

        # Count existing pairs
        if image_format == 'tiff':
            existing = sorted(images_path.glob('pair_*_cam1_A.tiff'))
        else:
            existing = sorted(images_path.glob('pair_*.npz'))
        num_existing = len(existing)

        if num_existing >= num_pairs:
            # All images exist, just load
            if verbose:
                print(f"Loading {num_pairs} images from {images_dir}...")
            image_iterator = StereoEnsembleGenerator.load_images(images_dir)
        elif num_existing > 0:
            # Some exist, need to generate more
            if verbose:
                print(f"Found {num_existing} pairs, generating "
                      f"{num_pairs - num_existing} more...")
            # Generate missing pairs
            generator.save_images_range(
                images_path, num_existing, num_pairs, format=image_format
            )
            image_iterator = StereoEnsembleGenerator.load_images(images_dir)
        else:
            # None exist, generate all
            if verbose:
                print(f"Generating {num_pairs} images to {images_dir}...")
            generator.save_images(images_dir, format=image_format)
            image_iterator = StereoEnsembleGenerator.load_images(images_dir)
    else:
        # No caching, generate on the fly
        image_iterator = generator.generate_all()

    # Process ensemble
    if verbose:
        print(f"Processing {num_pairs} image pairs...")

    # Optional: tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(
            image_iterator,
            total=num_pairs,
            desc="Processing"
        )
    except ImportError:
        # Fallback without tqdm
        iterator = image_iterator
        if verbose:
            print("(Install tqdm for progress bar: pip install tqdm)")

    count = 0
    for images in iterator:
        # Reconstruct volumes from stereo images
        vol_A = reconstructor.reconstruct([images['cam1_A'], images['cam2_A']])
        vol_B = reconstructor.reconstruct([images['cam1_B'], images['cam2_B']])

        # Accumulate correlations
        accumulator.accumulate(vol_A, vol_B)
        count += 1

        # Progress update for non-tqdm case
        if verbose and count % 100 == 0 and 'tqdm' not in str(type(iterator)):
            print(f"  Processed {count}/{num_pairs} pairs...")

    # Finalize correlation volumes
    if verbose:
        print("Finalizing ensemble correlations...")

    map_auto, map_cross = accumulator.finalize()

    # Fit stacked Gaussian
    if verbose:
        print("Fitting stacked 3D Gaussian...")

    result = fit_stacked_gaussian_3d(
        map_auto,
        map_cross,
        roi_size=min(config.volume_size) // 2 - 2
    )

    elapsed_time = time.time() - start_time

    return result, elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description='Stereo Ensemble PIV for Reynolds Stress Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Ensemble parameters
    parser.add_argument(
        '--num-pairs', type=int, default=500,
        help='Number of image pairs to process'
    )
    parser.add_argument(
        '--particles', type=int, default=300,
        help='Number of particles per image'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    # Quick test mode
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test with 50 pairs and 100 particles'
    )

    # Volume parameters
    parser.add_argument(
        '--volume', type=int, nargs=3, default=[64, 64, 16],
        help='Volume size (X Y Z)'
    )

    # Reynolds stress specification
    parser.add_argument(
        '--stress-uu', type=float, default=1.0,
        help="Reynolds stress <u'u'> component"
    )
    parser.add_argument(
        '--stress-vv', type=float, default=0.8,
        help="Reynolds stress <v'v'> component"
    )
    parser.add_argument(
        '--stress-ww', type=float, default=0.5,
        help="Reynolds stress <w'w'> component"
    )
    parser.add_argument(
        '--stress-uv', type=float, default=0.3,
        help="Reynolds stress <u'v'> component"
    )
    parser.add_argument(
        '--stress-uw', type=float, default=0.1,
        help="Reynolds stress <u'w'> component"
    )
    parser.add_argument(
        '--stress-vw', type=float, default=0.2,
        help="Reynolds stress <v'w'> component"
    )

    # Output
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )

    # Image caching
    parser.add_argument(
        '--images', type=str, default=None,
        help='Image directory: load if exists, generate missing pairs as needed'
    )
    parser.add_argument(
        '--format', type=str, default='tiff', choices=['npz', 'tiff'],
        help='Image format for saving (npz or tiff)'
    )

    # Debug visualization
    parser.add_argument(
        '--debug-viz', type=str, default=None, metavar='DIR',
        help='Generate debug visualizations (camera geometry, MinLOS, ghost analysis) to DIR'
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_pairs = 50
        args.particles = 100
        args.volume = [32, 32, 8]

    # Build Reynolds stress tensor
    R_true = np.array([
        [args.stress_uu, args.stress_uv, args.stress_uw],
        [args.stress_uv, args.stress_vv, args.stress_vw],
        [args.stress_uw, args.stress_vw, args.stress_ww]
    ])

    # Validate positive semi-definiteness
    eigvals = np.linalg.eigvalsh(R_true)
    if np.any(eigvals < -1e-10):
        print("ERROR: Specified Reynolds stress tensor is not positive "
              "semi-definite!")
        print(f"Eigenvalues: {eigvals}")
        return 1

    # Mean displacement (zero for pure turbulence test)
    mean_disp = np.array([0.0, 0.0, 0.0])

    # Create configuration
    config = StereoEnsembleConfig(
        volume_size=tuple(args.volume),
        num_particles=args.particles,
        reynolds_stress=R_true,
        mean_displacement=mean_disp,
        num_image_pairs=args.num_pairs,
        seed=args.seed,
        # Camera/rendering defaults
        particle_diameter_px=3.0,
        camera_angles_deg=(-45.0, 45.0),
        working_distance_mm=100.0,
        image_size_px=max(args.volume[0], args.volume[1]),  # Match image to volume
        scale_px_per_mm=15.0
    )

    # Print configuration
    verbose = not args.quiet
    if verbose:
        print("=" * 60)
        print("STEREO ENSEMBLE PIV - REYNOLDS STRESS EXTRACTION")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Volume size: {config.volume_size}")
        print(f"  Particles: {config.num_particles}")
        print(f"  Image pairs: {config.num_image_pairs}")
        print(f"  Camera angles: {config.camera_angles_deg}")
        print(f"\nTarget Reynolds stress tensor:")
        print(format_matrix(R_true))
        print(f"\nTarget mean displacement: {mean_disp}")
        print()

    # Debug visualization (if requested)
    if args.debug_viz:
        print("=" * 60)
        print("DEBUG VISUALIZATION")
        print("=" * 60)
        # Create generator and reconstructor for debug viz
        debug_generator = StereoEnsembleGenerator(config)
        debug_reconstructor = StereoMLOSReconstructor(
            debug_generator.cameras,
            config.volume_size,
            config.scale_px_per_mm
        )
        run_debug_visualization(
            debug_generator,
            debug_reconstructor,
            output_dir=args.debug_viz,
            show=True
        )
        print()

    # Run pipeline
    result, elapsed_time = run_pipeline(
        config,
        verbose=verbose,
        images_dir=args.images,
        image_format=args.format
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFit quality:")
    print(f"  Success: {result.success}")
    print(f"  Cost: {result.cost:.4f}")
    print(f"  Message: {result.message}")

    print(f"\nDisplacement:")
    print(f"  Fitted:   [{result.displacement[0]:6.3f}, "
          f"{result.displacement[1]:6.3f}, {result.displacement[2]:6.3f}]")
    print(f"  Expected: [{mean_disp[0]:6.3f}, {mean_disp[1]:6.3f}, "
          f"{mean_disp[2]:6.3f}]")
    disp_error = np.linalg.norm(result.displacement - mean_disp)
    print(f"  Error: {disp_error:.4f} voxels")

    print(f"\nExtracted Reynolds stress tensor:")
    print(format_matrix(result.sigma_turb))

    print(f"\nTrue Reynolds stress tensor:")
    print(format_matrix(R_true))

    print(f"\nAbsolute error:")
    print(format_matrix(np.abs(result.sigma_turb - R_true)))

    # Error metrics
    rms_error = np.sqrt(np.mean((result.sigma_turb - R_true)**2))
    max_error = np.max(np.abs(result.sigma_turb - R_true))
    rel_errors = np.abs(result.sigma_turb - R_true) / (np.abs(R_true) + 1e-10)
    mean_rel_error = np.mean(rel_errors) * 100

    print(f"\nError metrics:")
    print(f"  RMS error: {rms_error:.4f}")
    print(f"  Max error: {max_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.1f}%")

    print(f"\nTiming:")
    print(f"  Total time: {elapsed_time:.1f}s")
    print(f"  Per pair: {elapsed_time/config.num_image_pairs*1000:.1f}ms")

    print("\n" + "=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
