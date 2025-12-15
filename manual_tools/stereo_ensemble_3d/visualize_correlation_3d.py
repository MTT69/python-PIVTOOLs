#!/usr/bin/env python3
"""
3D Plotly Visualization for Stereo Ensemble PIV Correlation Diagnostics
========================================================================

Generates interactive HTML visualizations of:
1. Auto-correlation volumes (isosurface)
2. Cross-correlation volumes (isosurface)
3. Data vs Fitted Gaussian comparison
4. Parameter comparison tables (expected vs actual)

Usage:
    python visualize_correlation_3d.py
    python visualize_correlation_3d.py --pairs 100 200 500
"""

import argparse
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stereo_ensemble_generator import StereoEnsembleConfig, StereoEnsembleGenerator
from correlation_3d import StereoMLOSReconstructor, EnsembleAccumulator3D
from gaussian_fit_stacked_3d import (
    fit_stacked_gaussian_3d, gaussian_3d, StackedGaussianResult3D
)


def create_isosurface_figure(
    volume: np.ndarray,
    title: str,
    num_surfaces: int = 5,
    opacity: float = 0.6
) -> go.Figure:
    """
    Create Plotly isosurface visualization of 3D volume.

    Parameters
    ----------
    volume : ndarray, shape (nx, ny, nz)
        3D volume data
    title : str
        Figure title
    num_surfaces : int
        Number of isosurface levels
    opacity : float
        Surface opacity

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    nx, ny, nz = volume.shape
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    z = np.arange(nz) - nz // 2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    vmin, vmax = volume.min(), volume.max()

    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=vmin + 0.1 * (vmax - vmin),
        isomax=vmax,
        opacity=opacity,
        surface_count=num_surfaces,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)',
            aspectmode='data'
        ),
        width=900, height=700,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def create_slice_figure(
    volume: np.ndarray,
    title: str,
    slice_axis: int = 2
) -> go.Figure:
    """
    Create slice views of 3D volume (XY, XZ, YZ central slices).

    Parameters
    ----------
    volume : ndarray, shape (nx, ny, nz)
        3D volume data
    title : str
        Figure title
    slice_axis : int
        Primary axis for central slice

    Returns
    -------
    fig : go.Figure
        Plotly figure with subplots
    """
    nx, ny, nz = volume.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['XY slice (z=0)', 'XZ slice (y=0)', 'YZ slice (x=0)'],
        horizontal_spacing=0.08
    )

    # XY slice
    xy_slice = volume[:, :, cz]
    fig.add_trace(
        go.Heatmap(z=xy_slice.T, colorscale='Viridis', showscale=False),
        row=1, col=1
    )

    # XZ slice
    xz_slice = volume[:, cy, :]
    fig.add_trace(
        go.Heatmap(z=xz_slice.T, colorscale='Viridis', showscale=False),
        row=1, col=2
    )

    # YZ slice
    yz_slice = volume[cx, :, :]
    fig.add_trace(
        go.Heatmap(z=yz_slice.T, colorscale='Viridis', showscale=True),
        row=1, col=3
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=1200, height=400,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def generate_fitted_volume(
    shape: tuple,
    amp: float,
    bg: float,
    center: np.ndarray,
    cov: np.ndarray
) -> np.ndarray:
    """Generate a 3D Gaussian volume from fitted parameters."""
    roi_center = np.array(shape) / 2
    x = np.arange(shape[0]) - roi_center[0]
    y = np.arange(shape[1]) - roi_center[1]
    z = np.arange(shape[2]) - roi_center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return gaussian_3d(coords, amp, bg, center, cov).reshape(shape)


def create_comparison_figure(
    data_auto: np.ndarray,
    data_cross: np.ndarray,
    fit_result: StackedGaussianResult3D,
    title: str
) -> go.Figure:
    """
    Create side-by-side comparison of data vs fitted Gaussians.
    Shows central XY slices.
    """
    shape = data_auto.shape
    cz = shape[2] // 2

    # Generate fitted volumes
    fitted_auto = generate_fitted_volume(
        shape,
        fit_result.amp_auto,
        fit_result.bg_auto,
        fit_result.center_auto,
        fit_result.sigma_geo
    )

    fitted_cross = generate_fitted_volume(
        shape,
        fit_result.amp_cross,
        fit_result.bg_cross,
        fit_result.center_cross,
        fit_result.sigma_cross
    )

    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Auto-Corr (Data)', 'Auto-Corr (Fitted)',
            'Cross-Corr (Data)', 'Cross-Corr (Fitted)'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # Data and fitted slices
    slices = [
        (data_auto[:, :, cz], 1, 1),
        (fitted_auto[:, :, cz], 1, 2),
        (data_cross[:, :, cz], 2, 1),
        (fitted_cross[:, :, cz], 2, 2),
    ]

    for slice_data, row, col in slices:
        fig.add_trace(
            go.Heatmap(
                z=slice_data.T,
                colorscale='Viridis',
                showscale=(col == 2)
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=900, height=800,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def create_3d_comparison_figure(
    data_auto: np.ndarray,
    data_cross: np.ndarray,
    fit_result: StackedGaussianResult3D,
    title: str
) -> go.Figure:
    """Create 3D isosurface comparison of data vs fitted."""
    shape = data_auto.shape
    nx, ny, nz = shape

    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    z = np.arange(nz) - nz // 2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Generate fitted volumes
    fitted_auto = generate_fitted_volume(
        shape, fit_result.amp_auto, fit_result.bg_auto,
        fit_result.center_auto, fit_result.sigma_geo
    )
    fitted_cross = generate_fitted_volume(
        shape, fit_result.amp_cross, fit_result.bg_cross,
        fit_result.center_cross, fit_result.sigma_cross
    )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scene'}, {'type': 'scene'}],
            [{'type': 'scene'}, {'type': 'scene'}]
        ],
        subplot_titles=[
            'Auto-Corr (Data)', 'Auto-Corr (Fitted)',
            'Cross-Corr (Data)', 'Cross-Corr (Fitted)'
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.02
    )

    volumes = [data_auto, fitted_auto, data_cross, fitted_cross]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    scene_names = ['scene', 'scene2', 'scene3', 'scene4']

    for vol, (row, col), scene in zip(volumes, positions, scene_names):
        vmax = vol.max()
        vmin = vol.min()
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=vol.flatten(),
                isomin=vmin + 0.3 * (vmax - vmin),
                isomax=vmax,
                opacity=0.5,
                surface_count=3,
                colorscale='Viridis',
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False
            ),
            row=row, col=col
        )

    # Update all scenes
    for scene in scene_names:
        fig.update_layout(**{
            scene: dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            )
        })

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=1000, width=1100,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def generate_parameter_table(
    fit_result: StackedGaussianResult3D,
    true_reynolds: np.ndarray,
    true_displacement: np.ndarray
) -> str:
    """Generate text table of expected vs actual parameters."""
    lines = []
    lines.append("=" * 80)
    lines.append("PARAMETER COMPARISON: Expected vs Fitted")
    lines.append("=" * 80)
    lines.append(f"{'Parameter':<25} {'Expected':>12} {'Fitted':>12} {'Error':>12} {'Rel%':>8}")
    lines.append("-" * 80)

    # Displacement
    for i, name in enumerate(['dx', 'dy', 'dz']):
        exp = true_displacement[i]
        fit = fit_result.displacement[i]
        err = abs(fit - exp)
        rel = 100 * err / (abs(exp) + 1e-10) if abs(exp) > 1e-10 else 0.0
        lines.append(f"{name:<25} {exp:>12.4f} {fit:>12.4f} {err:>12.4f} {rel:>7.1f}%")

    lines.append("-" * 80)

    # Reynolds stress components
    labels = [
        ('R_uu (sigma_turb_xx)', 0, 0),
        ('R_vv (sigma_turb_yy)', 1, 1),
        ('R_ww (sigma_turb_zz)', 2, 2),
        ('R_uv (sigma_turb_xy)', 0, 1),
        ('R_uw (sigma_turb_xz)', 0, 2),
        ('R_vw (sigma_turb_yz)', 1, 2),
    ]

    for name, i, j in labels:
        exp = true_reynolds[i, j]
        fit = fit_result.sigma_turb[i, j]
        err = abs(fit - exp)
        rel = 100 * err / (abs(exp) + 1e-10) if abs(exp) > 0.01 else 0.0
        lines.append(f"{name:<25} {exp:>12.4f} {fit:>12.4f} {err:>12.4f} {rel:>7.1f}%")

    lines.append("-" * 80)

    # Summary metrics
    diag_errors = [abs(fit_result.sigma_turb[i, i] - true_reynolds[i, i]) for i in range(3)]
    offdiag_errors = [
        abs(fit_result.sigma_turb[0, 1] - true_reynolds[0, 1]),
        abs(fit_result.sigma_turb[0, 2] - true_reynolds[0, 2]),
        abs(fit_result.sigma_turb[1, 2] - true_reynolds[1, 2]),
    ]

    rms_all = np.sqrt(np.mean((fit_result.sigma_turb - true_reynolds)**2))
    max_diag = max(diag_errors)
    max_offdiag = max(offdiag_errors)

    lines.append(f"\nSummary Metrics:")
    lines.append(f"  RMS error (all components): {rms_all:.4f}")
    lines.append(f"  Max diagonal error: {max_diag:.4f}")
    lines.append(f"  Max off-diagonal error: {max_offdiag:.4f}")

    # Geometric covariance
    lines.append(f"\nFitted Geometric Covariance (sigma_geo):")
    lines.append(f"  sigma_geo_xx = {fit_result.sigma_geo[0,0]:.4f}")
    lines.append(f"  sigma_geo_yy = {fit_result.sigma_geo[1,1]:.4f}")
    lines.append(f"  sigma_geo_zz = {fit_result.sigma_geo[2,2]:.4f}")
    lines.append(f"  sigma_geo_xy = {fit_result.sigma_geo[0,1]:.4f}")
    lines.append(f"  sigma_geo_xz = {fit_result.sigma_geo[0,2]:.4f}")
    lines.append(f"  sigma_geo_yz = {fit_result.sigma_geo[1,2]:.4f}")

    lines.append(f"\nFit Quality:")
    lines.append(f"  Success: {fit_result.success}")
    lines.append(f"  Cost: {fit_result.cost:.6f}")
    lines.append(f"  Message: {fit_result.message}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def run_visualization(
    num_pairs: int,
    output_dir: str,
    config_overrides: dict = None
) -> StackedGaussianResult3D:
    """
    Run pipeline and generate visualizations for given number of pairs.

    Parameters
    ----------
    num_pairs : int
        Number of image pairs to process
    output_dir : str
        Directory for output files
    config_overrides : dict, optional
        Override configuration parameters

    Returns
    -------
    result : StackedGaussianResult3D
        Fitting results
    """
    print(f"\n{'='*60}")
    print(f"Running with {num_pairs} image pairs")
    print(f"{'='*60}")

    # True parameters
    R_true = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 0.8, 0.2],
        [0.1, 0.2, 0.5]
    ])
    mean_disp = np.array([0.0, 0.0, 0.0])

    # Configuration
    cfg_params = dict(
        volume_size=(64, 64, 16),
        num_particles=200,
        reynolds_stress=R_true,
        mean_displacement=mean_disp,
        num_image_pairs=num_pairs,
        seed=42,
        particle_diameter_px=3.0,
        camera_angles_deg=(-45.0, 45.0),
        working_distance_mm=100.0,
        image_size_px=64,  # Match volume size
        scale_px_per_mm=15.0
    )

    if config_overrides:
        cfg_params.update(config_overrides)

    config = StereoEnsembleConfig(**cfg_params)

    # Run pipeline
    print(f"Initializing generator and reconstructor...")
    generator = StereoEnsembleGenerator(config)
    reconstructor = StereoMLOSReconstructor(
        generator.cameras, config.volume_size, config.scale_px_per_mm
    )
    accumulator = EnsembleAccumulator3D(config.volume_size)

    print(f"Processing {num_pairs} image pairs...")
    for i, images in enumerate(generator.generate_all()):
        vol_A = reconstructor.reconstruct([images['cam1_A'], images['cam2_A']])
        vol_B = reconstructor.reconstruct([images['cam1_B'], images['cam2_B']])
        accumulator.accumulate(vol_A, vol_B)
        if (i + 1) % 50 == 0 or i == num_pairs - 1:
            print(f"  Processed {i+1}/{num_pairs}")

    print("Finalizing correlations...")
    map_auto, map_cross = accumulator.finalize()

    # Fit
    print("Fitting stacked Gaussian...")
    result = fit_stacked_gaussian_3d(map_auto, map_cross, roi_size=20)

    # Generate visualizations
    print("Generating visualizations...")

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"N{num_pairs}"

    # 1. Auto-correlation isosurface
    print(f"  Creating {prefix}_auto_correlation.html...")
    fig_auto = create_isosurface_figure(
        map_auto, f"Auto-Correlation (N={num_pairs} pairs)"
    )
    fig_auto.write_html(f"{output_dir}/{prefix}_auto_correlation.html")

    # 2. Cross-correlation isosurface
    print(f"  Creating {prefix}_cross_correlation.html...")
    fig_cross = create_isosurface_figure(
        map_cross, f"Cross-Correlation (N={num_pairs} pairs)"
    )
    fig_cross.write_html(f"{output_dir}/{prefix}_cross_correlation.html")

    # 3. Slice views
    print(f"  Creating {prefix}_auto_slices.html...")
    fig_auto_slices = create_slice_figure(
        map_auto, f"Auto-Correlation Slices (N={num_pairs} pairs)"
    )
    fig_auto_slices.write_html(f"{output_dir}/{prefix}_auto_slices.html")

    print(f"  Creating {prefix}_cross_slices.html...")
    fig_cross_slices = create_slice_figure(
        map_cross, f"Cross-Correlation Slices (N={num_pairs} pairs)"
    )
    fig_cross_slices.write_html(f"{output_dir}/{prefix}_cross_slices.html")

    # 4. 2D slice comparison (data vs fitted)
    print(f"  Creating {prefix}_fitted_comparison_2d.html...")
    fig_compare_2d = create_comparison_figure(
        map_auto, map_cross, result,
        f"Data vs Fitted - Central XY Slices (N={num_pairs} pairs)"
    )
    fig_compare_2d.write_html(f"{output_dir}/{prefix}_fitted_comparison_2d.html")

    # 5. 3D comparison (data vs fitted)
    print(f"  Creating {prefix}_fitted_comparison_3d.html...")
    fig_compare_3d = create_3d_comparison_figure(
        map_auto, map_cross, result,
        f"Data vs Fitted - 3D Isosurfaces (N={num_pairs} pairs)"
    )
    fig_compare_3d.write_html(f"{output_dir}/{prefix}_fitted_comparison_3d.html")

    # 6. Parameter table
    print(f"  Creating {prefix}_parameters.txt...")
    table = generate_parameter_table(result, R_true, mean_disp)
    with open(f"{output_dir}/{prefix}_parameters.txt", 'w') as f:
        f.write(table)

    print("\n" + table)

    print(f"\nSaved {prefix}_*.html and {prefix}_parameters.txt to {output_dir}/")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='3D Visualization for Stereo Ensemble PIV Correlation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--pairs', type=int, nargs='+', default=[50, 200],
        help='Number of image pairs to process (can specify multiple)'
    )
    parser.add_argument(
        '--output', type=str, default='correlation_visualizations',
        help='Output directory for HTML files'
    )

    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output
    )

    results = {}
    for n in args.pairs:
        results[n] = run_visualization(n, output_dir)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}/")
    print("\nGenerated files:")
    for n in args.pairs:
        print(f"  N={n}:")
        print(f"    - N{n}_auto_correlation.html")
        print(f"    - N{n}_cross_correlation.html")
        print(f"    - N{n}_auto_slices.html")
        print(f"    - N{n}_cross_slices.html")
        print(f"    - N{n}_fitted_comparison_2d.html")
        print(f"    - N{n}_fitted_comparison_3d.html")
        print(f"    - N{n}_parameters.txt")


if __name__ == '__main__':
    main()
