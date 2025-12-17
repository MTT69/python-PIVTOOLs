#!/usr/bin/env python3
"""
Complete Pipeline Visualization for Stereo Ensemble PIV
========================================================

Generates Plotly visualizations for each step:
1. Line-of-sight intersections in 3D (per image pair)
2. MinLOS reconstruction volumes (per image pair)
3. Auto and cross correlation maps (per image pair)
4. Ensemble-averaged correlation maps

Usage:
    python pipeline_visualization.py
    python pipeline_visualization.py --num-pairs 10 --output ./viz_output
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from PIL import Image

from stereo_ensemble_generator import StereoEnsembleConfig, StereoEnsembleGenerator
from correlation_3d import StereoMLOSReconstructor, EnsembleAccumulator3D, correlate_3d
from gaussian_fit_stacked_3d import fit_stacked_gaussian_3d, StackedGaussianResult3D


def get_particle_positions(generator: StereoEnsembleGenerator, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recreate particle positions for a given pair index."""
    cfg = generator.config
    base_seed = cfg.seed + pair_idx * 3

    pos_A_mm = generator.generate_particles(seed=base_seed)
    displacements_px = generator.sample_displacements(len(pos_A_mm), seed=base_seed + 2)
    displacements_mm = displacements_px / cfg.scale_px_per_mm
    pos_B_mm = pos_A_mm + displacements_mm

    return pos_A_mm, pos_B_mm


def create_line_of_sight_figure(
    generator: StereoEnsembleGenerator,
    pos_A_mm: np.ndarray,
    pair_idx: int,
    max_particles: int = 30
) -> go.Figure:
    """
    Create 3D visualization showing camera sight lines through particles.

    Shows how lines from each camera intersect at particle locations.
    """
    cfg = generator.config
    cameras = generator.cameras
    vol_half = generator.volume_half_mm

    fig = go.Figure()

    # Volume bounding box
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]) * vol_half

    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Camera positions
    cam_colors = ['#FF6B6B', '#4ECDC4']  # Red-ish, Teal
    for cam_idx, cam in enumerate(cameras):
        fig.add_trace(go.Scatter3d(
            x=[cam.position[0]],
            y=[cam.position[1]],
            z=[cam.position[2]],
            mode='markers+text',
            marker=dict(size=12, color=cam_colors[cam_idx], symbol='diamond'),
            text=[f'Cam{cam_idx+1} ({cam.angle_deg:.0f}°)'],
            textposition='top center',
            name=f'Camera {cam_idx+1}',
            showlegend=True
        ))

    # Particles (all of them)
    fig.add_trace(go.Scatter3d(
        x=pos_A_mm[:, 0],
        y=pos_A_mm[:, 1],
        z=pos_A_mm[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.6),
        name='Particles',
        hovertemplate='(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # Sight lines for subset of particles
    n_show = min(max_particles, len(pos_A_mm))
    indices = np.linspace(0, len(pos_A_mm)-1, n_show, dtype=int)

    for p_idx in indices:
        particle_pos = pos_A_mm[p_idx]

        for cam_idx, cam in enumerate(cameras):
            # Line from camera through particle, extended
            direction = particle_pos - cam.position
            t_end = np.linalg.norm(direction) * 1.3
            direction = direction / np.linalg.norm(direction)

            line_end = cam.position + direction * t_end

            fig.add_trace(go.Scatter3d(
                x=[cam.position[0], line_end[0]],
                y=[cam.position[1], line_end[1]],
                z=[cam.position[2], line_end[2]],
                mode='lines',
                line=dict(color=cam_colors[cam_idx], width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

    # Highlight intersection points (particles)
    fig.add_trace(go.Scatter3d(
        x=pos_A_mm[indices, 0],
        y=pos_A_mm[indices, 1],
        z=pos_A_mm[indices, 2],
        mode='markers',
        marker=dict(size=6, color='yellow', symbol='circle',
                   line=dict(color='black', width=1)),
        name='LOS Intersections',
        hovertemplate='Intersection: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f'Line-of-Sight Intersections - Pair {pair_idx}', font=dict(size=16)),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=50, b=0),
        width=900, height=700
    )

    return fig


def find_ghost_particles(
    volume: np.ndarray,
    true_pos_mm: np.ndarray,
    scale_px_per_mm: float,
    intensity_threshold_fraction: float = 0.3,
    true_particle_radius_mm: float = 0.3,
    min_peak_intensity_fraction: float = 0.4
) -> np.ndarray:
    """
    Find ghost particles in MinLOS reconstruction.

    Ghost particles are high-intensity regions that don't correspond to true particles.

    Parameters
    ----------
    volume : ndarray
        MinLOS reconstructed volume
    true_pos_mm : ndarray (N, 3)
        True particle positions in mm
    scale_px_per_mm : float
        Scale factor
    intensity_threshold_fraction : float
        Fraction of max intensity for binary threshold (0-1)
    true_particle_radius_mm : float
        Radius around true particles to exclude from ghost detection
    min_peak_intensity_fraction : float
        Minimum peak intensity (as fraction of volume max) for a region to be considered

    Returns
    -------
    ghost_pos_mm : ndarray (M, 3)
        Ghost particle positions in mm
    """
    from scipy.ndimage import label, center_of_mass, maximum

    nx, ny, nz = volume.shape
    vmax = volume.max()

    # Threshold the volume to find bright regions (use fraction of max, not percentile)
    threshold = vmax * intensity_threshold_fraction
    binary = volume > threshold

    # Label connected components
    labeled, num_features = label(binary)

    if num_features == 0:
        return np.array([]).reshape(0, 3)

    # Find centers of mass and peak intensity for each component
    centers_voxel = center_of_mass(volume, labeled, range(1, num_features + 1))
    centers_voxel = np.array(centers_voxel)

    # Get peak intensity of each labeled region
    peak_intensities = maximum(volume, labeled, range(1, num_features + 1))
    peak_intensities = np.array(peak_intensities)

    if len(centers_voxel) == 0:
        return np.array([]).reshape(0, 3)

    # Convert voxel coordinates to mm
    centers_mm = np.zeros_like(centers_voxel)
    centers_mm[:, 0] = (centers_voxel[:, 0] - nx/2 + 0.5) / scale_px_per_mm
    centers_mm[:, 1] = (centers_voxel[:, 1] - ny/2 + 0.5) / scale_px_per_mm
    centers_mm[:, 2] = (centers_voxel[:, 2] - nz/2 + 0.5) / scale_px_per_mm

    # Filter: keep only regions that are bright enough AND not near true particles
    ghost_mask = np.ones(len(centers_mm), dtype=bool)

    for i, center in enumerate(centers_mm):
        # Check if this region is bright enough
        if peak_intensities[i] < vmax * min_peak_intensity_fraction:
            ghost_mask[i] = False
            continue

        # Check distance to all true particles
        distances = np.linalg.norm(true_pos_mm - center, axis=1)
        if np.min(distances) < true_particle_radius_mm:
            ghost_mask[i] = False  # This is a true particle, not a ghost

    ghost_pos_mm = centers_mm[ghost_mask]

    return ghost_pos_mm


def create_minlos_figure(
    volume: np.ndarray,
    generator: StereoEnsembleGenerator,
    pos_mm: np.ndarray,
    pair_idx: int,
    frame: str = 'A'
) -> go.Figure:
    """Create 3D isosurface of MinLOS reconstruction with true particle positions and ghosts."""
    cfg = generator.config
    nx, ny, nz = cfg.volume_size

    # Voxel coordinates in mm
    x_mm = (np.arange(nx) - nx/2 + 0.5) / cfg.scale_px_per_mm
    y_mm = (np.arange(ny) - ny/2 + 0.5) / cfg.scale_px_per_mm
    z_mm = (np.arange(nz) - nz/2 + 0.5) / cfg.scale_px_per_mm

    X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')

    # Threshold for visualization
    vmax = volume.max()
    vmin = volume.min()
    threshold = vmin + 0.3 * (vmax - vmin)

    # Find ghost particles (only bright ones that match the isosurface threshold)
    ghost_pos_mm = find_ghost_particles(
        volume, pos_mm, cfg.scale_px_per_mm,
        intensity_threshold_fraction=0.3,  # Same as isosurface threshold
        true_particle_radius_mm=0.4,
        min_peak_intensity_fraction=0.4  # Must be at least 40% of max intensity
    )

    num_ghosts = len(ghost_pos_mm)
    num_true = len(pos_mm)

    fig = go.Figure()

    # MinLOS isosurface
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=vmax,
        surface_count=4,
        colorscale='Viridis',
        cmin=0,  # Start colormap from zero
        cmax=vmax,  # End at max value
        opacity=0.4,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='MinLOS Volume',
        showscale=True,
        colorbar=dict(title='Intensity', x=1.02, len=0.7)
    ))

    # True particle positions (green circles)
    fig.add_trace(go.Scatter3d(
        x=pos_mm[:, 0],
        y=pos_mm[:, 1],
        z=pos_mm[:, 2],
        mode='markers',
        marker=dict(size=6, color='lime', symbol='circle',
                   line=dict(color='darkgreen', width=2)),
        name=f'True Particles ({num_true})',
        hovertemplate='True: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # Ghost particle positions (red X markers)
    if num_ghosts > 0:
        fig.add_trace(go.Scatter3d(
            x=ghost_pos_mm[:, 0],
            y=ghost_pos_mm[:, 1],
            z=ghost_pos_mm[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='x',
                       line=dict(color='darkred', width=2)),
            name=f'Ghost Particles ({num_ghosts})',
            hovertemplate='Ghost: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=f'MinLOS Reconstruction - Pair {pair_idx} Frame {frame}<br>'
                 f'<sub>True: {num_true} (green) | Ghosts: {num_ghosts} (red)</sub>',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=70, b=0),
        width=900, height=700
    )

    return fig


def create_correlation_figure(
    auto_corr: np.ndarray,
    cross_corr: np.ndarray,
    pair_idx: int
) -> go.Figure:
    """Create 2x3 subplot showing XY, XZ, YZ slices of auto and cross correlation."""
    shape = auto_corr.shape
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Auto-Corr XY (z=0)', 'Auto-Corr XZ (y=0)', 'Auto-Corr YZ (x=0)',
            'Cross-Corr XY (z=0)', 'Cross-Corr XZ (y=0)', 'Cross-Corr YZ (x=0)'
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # Auto-correlation slices (zmin=0)
    fig.add_trace(go.Heatmap(z=auto_corr[:, :, cz].T, colorscale='Viridis', zmin=0, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=auto_corr[:, cy, :].T, colorscale='Viridis', zmin=0, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=auto_corr[cx, :, :].T, colorscale='Viridis', zmin=0, showscale=True,
                             colorbar=dict(title='Auto', x=1.02, y=0.75, len=0.4)), row=1, col=3)

    # Cross-correlation slices (zmin=0)
    fig.add_trace(go.Heatmap(z=cross_corr[:, :, cz].T, colorscale='Viridis', zmin=0, showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(z=cross_corr[:, cy, :].T, colorscale='Viridis', zmin=0, showscale=False), row=2, col=2)
    fig.add_trace(go.Heatmap(z=cross_corr[cx, :, :].T, colorscale='Viridis', zmin=0, showscale=True,
                             colorbar=dict(title='Cross', x=1.02, y=0.25, len=0.4)), row=2, col=3)

    fig.update_layout(
        title=dict(text=f'Correlation Maps - Pair {pair_idx}', font=dict(size=16)),
        height=600, width=1000,
        margin=dict(l=10, r=80, t=80, b=10)
    )

    return fig


def create_correlation_3d_figure(
    auto_corr: np.ndarray,
    cross_corr: np.ndarray,
    pair_idx: int
) -> go.Figure:
    """Create side-by-side 3D isosurface of auto and cross correlation."""
    shape = auto_corr.shape

    # Coordinates centered at zero
    x = np.arange(shape[0]) - shape[0]//2
    y = np.arange(shape[1]) - shape[1]//2
    z = np.arange(shape[2]) - shape[2]//2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Auto-Correlation', 'Cross-Correlation'],
        horizontal_spacing=0.05
    )

    for col, (corr, name) in enumerate([(auto_corr, 'Auto'), (cross_corr, 'Cross')], 1):
        vmax = corr.max()

        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=corr.flatten(),
            isomin=vmax * 0.3,
            isomax=vmax,
            surface_count=4,
            colorscale='Viridis',
            cmin=0,  # Start colormap from zero
            cmax=vmax,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=(col == 2),
            colorbar=dict(title='Corr', x=1.02) if col == 2 else None
        ), row=1, col=col)

    for scene_name in ['scene', 'scene2']:
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='ΔX (voxels)',
                yaxis_title='ΔY (voxels)',
                zaxis_title='ΔZ (voxels)',
                aspectmode='data'
            )
        })

    fig.update_layout(
        title=dict(text=f'3D Correlation Volumes - Pair {pair_idx}', font=dict(size=16)),
        height=600, width=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def create_ensemble_average_figure(
    avg_auto: np.ndarray,
    avg_cross: np.ndarray,
    num_pairs: int
) -> go.Figure:
    """Create visualization of ensemble-averaged correlations."""
    shape = avg_auto.shape
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f'Avg Auto-Corr XY (N={num_pairs})', 'Avg Auto-Corr XZ', 'Avg Auto-Corr YZ',
            f'Avg Cross-Corr XY (N={num_pairs})', 'Avg Cross-Corr XZ', 'Avg Cross-Corr YZ'
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # Auto-correlation slices (zmin=0)
    fig.add_trace(go.Heatmap(z=avg_auto[:, :, cz].T, colorscale='Viridis', zmin=0, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=avg_auto[:, cy, :].T, colorscale='Viridis', zmin=0, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=avg_auto[cx, :, :].T, colorscale='Viridis', zmin=0, showscale=True,
                             colorbar=dict(title='Auto', x=1.02, y=0.75, len=0.4)), row=1, col=3)

    # Cross-correlation slices (zmin=0)
    fig.add_trace(go.Heatmap(z=avg_cross[:, :, cz].T, colorscale='Viridis', zmin=0, showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(z=avg_cross[:, cy, :].T, colorscale='Viridis', zmin=0, showscale=False), row=2, col=2)
    fig.add_trace(go.Heatmap(z=avg_cross[cx, :, :].T, colorscale='Viridis', zmin=0, showscale=True,
                             colorbar=dict(title='Cross', x=1.02, y=0.25, len=0.4)), row=2, col=3)

    fig.update_layout(
        title=dict(text=f'Ensemble-Averaged Correlation Maps (N={num_pairs} pairs)', font=dict(size=16)),
        height=600, width=1000,
        margin=dict(l=10, r=80, t=80, b=10)
    )

    return fig


def create_intensity_distribution_figure(
    avg_auto: np.ndarray,
    avg_cross: np.ndarray,
    num_pairs: int
) -> go.Figure:
    """
    Create sorted intensity plot to visualize signal vs noise distribution.

    This plot helps identify:
    - Where the noise floor is (flat region)
    - What fraction of voxels contain signal (steep rise)
    - Good threshold values for filtering
    """
    # Flatten and sort
    auto_flat = avg_auto.flatten()
    cross_flat = avg_cross.flatten()

    auto_sorted = np.sort(auto_flat)[::-1]  # Descending
    cross_sorted = np.sort(cross_flat)[::-1]

    n_voxels = len(auto_flat)
    percentile = np.linspace(0, 100, n_voxels)

    # Find some key thresholds
    auto_max = auto_sorted[0]
    cross_max = cross_sorted[0]

    # Find where intensity drops to various fractions of peak
    def find_percentile_at_threshold(sorted_vals, threshold):
        idx = np.searchsorted(-sorted_vals, -threshold)  # searchsorted on descending
        return 100 * idx / len(sorted_vals)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Sorted Intensity (Linear)',
            'Sorted Intensity (Log)',
            'Cumulative Distribution',
            'Intensity Statistics'
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ]
    )

    # Subsample for plotting (too many points otherwise)
    step = max(1, n_voxels // 2000)
    idx_sub = np.arange(0, n_voxels, step)

    # Top-left: Sorted intensity (linear)
    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=auto_sorted[idx_sub],
        mode='lines', name='Auto-Corr',
        line=dict(color='#2ecc71', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=cross_sorted[idx_sub],
        mode='lines', name='Cross-Corr',
        line=dict(color='#3498db', width=2)
    ), row=1, col=1)

    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Top-right: Log scale (absolute value for negatives)
    auto_abs = np.abs(auto_sorted) + 1e-10
    cross_abs = np.abs(cross_sorted) + 1e-10

    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=auto_abs[idx_sub],
        mode='lines', name='Auto-Corr',
        line=dict(color='#2ecc71', width=2),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=cross_abs[idx_sub],
        mode='lines', name='Cross-Corr',
        line=dict(color='#3498db', width=2),
        showlegend=False
    ), row=1, col=2)

    # Bottom-left: CDF (fraction of total signal below threshold)
    auto_cumsum = np.cumsum(auto_sorted) / np.sum(auto_sorted[auto_sorted > 0])
    cross_cumsum = np.cumsum(cross_sorted) / np.sum(cross_sorted[cross_sorted > 0])

    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=auto_cumsum[idx_sub] * 100,
        mode='lines', name='Auto-Corr',
        line=dict(color='#2ecc71', width=2),
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=percentile[idx_sub], y=cross_cumsum[idx_sub] * 100,
        mode='lines', name='Cross-Corr',
        line=dict(color='#3498db', width=2),
        showlegend=False
    ), row=2, col=1)

    # Bottom-right: Statistics table
    # Calculate useful stats
    auto_pos = auto_flat[auto_flat > 0]
    cross_pos = cross_flat[cross_flat > 0]
    auto_neg = auto_flat[auto_flat < 0]
    cross_neg = cross_flat[cross_flat < 0]

    # Find threshold where 90%, 95%, 99% of signal is captured
    def percentile_for_signal_fraction(sorted_vals, fraction):
        cumsum = np.cumsum(sorted_vals)
        total_pos = np.sum(sorted_vals[sorted_vals > 0])
        idx = np.searchsorted(cumsum, total_pos * fraction)
        return 100 * idx / len(sorted_vals), sorted_vals[min(idx, len(sorted_vals)-1)]

    p90_auto, v90_auto = percentile_for_signal_fraction(auto_sorted, 0.90)
    p95_auto, v95_auto = percentile_for_signal_fraction(auto_sorted, 0.95)
    p90_cross, v90_cross = percentile_for_signal_fraction(cross_sorted, 0.90)
    p95_cross, v95_cross = percentile_for_signal_fraction(cross_sorted, 0.95)

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Statistic</b>', '<b>Auto-Corr</b>', '<b>Cross-Corr</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                ['Peak value', 'Mean (all)', 'Std (all)',
                 'Positive voxels', 'Negative voxels',
                 '90% signal in top', '95% signal in top',
                 'Value at 90%', 'Value at 95%'],
                [f'{auto_max:.2f}', f'{auto_flat.mean():.2f}', f'{auto_flat.std():.2f}',
                 f'{len(auto_pos)} ({100*len(auto_pos)/n_voxels:.1f}%)',
                 f'{len(auto_neg)} ({100*len(auto_neg)/n_voxels:.1f}%)',
                 f'{p90_auto:.1f}%', f'{p95_auto:.1f}%',
                 f'{v90_auto:.2f}', f'{v95_auto:.2f}'],
                [f'{cross_max:.2f}', f'{cross_flat.mean():.2f}', f'{cross_flat.std():.2f}',
                 f'{len(cross_pos)} ({100*len(cross_pos)/n_voxels:.1f}%)',
                 f'{len(cross_neg)} ({100*len(cross_neg)/n_voxels:.1f}%)',
                 f'{p90_cross:.1f}%', f'{p95_cross:.1f}%',
                 f'{v90_cross:.2f}', f'{v95_cross:.2f}']
            ],
            fill_color=[['#ecf0f1']*9, ['#e8f6e8']*9, ['#e8f0f6']*9],
            align='left',
            font=dict(size=11)
        )
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text='Percentile of Voxels (sorted by intensity)', row=1, col=1)
    fig.update_xaxes(title_text='Percentile of Voxels', row=1, col=2)
    fig.update_xaxes(title_text='Percentile of Voxels', row=2, col=1)

    fig.update_yaxes(title_text='Correlation Intensity', row=1, col=1)
    fig.update_yaxes(title_text='|Intensity| (log)', type='log', row=1, col=2)
    fig.update_yaxes(title_text='Cumulative Signal (%)', row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f'Correlation Intensity Distribution Analysis (N={num_pairs} pairs, {n_voxels:,} voxels)',
            font=dict(size=16)
        ),
        height=700, width=1100,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def create_radial_intensity_figure(
    avg_auto: np.ndarray,
    avg_cross: np.ndarray,
    num_pairs: int
) -> go.Figure:
    """
    Create plot showing intensity vs distance from correlation peak.

    This reveals the shape/width of the correlation peaks.
    Only includes voxels with positive intensity (above noise floor).
    """
    shape = np.array(avg_auto.shape)
    center = shape // 2

    # Create coordinate grids
    x = np.arange(shape[0]) - center[0]
    y = np.arange(shape[1]) - center[1]
    z = np.arange(shape[2]) - center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Find peaks
    auto_peak_idx = np.unravel_index(np.argmax(avg_auto), shape)
    cross_peak_idx = np.unravel_index(np.argmax(avg_cross), shape)

    # Compute distances from peaks
    auto_dist = np.sqrt(
        (X - (auto_peak_idx[0] - center[0]))**2 +
        (Y - (auto_peak_idx[1] - center[1]))**2 +
        (Z - (auto_peak_idx[2] - center[2]))**2
    )
    cross_dist = np.sqrt(
        (X - (cross_peak_idx[0] - center[0]))**2 +
        (Y - (cross_peak_idx[1] - center[1]))**2 +
        (Z - (cross_peak_idx[2] - center[2]))**2
    )

    # Flatten
    auto_dist_flat = auto_dist.flatten()
    auto_int_flat = avg_auto.flatten()
    cross_dist_flat = cross_dist.flatten()
    cross_int_flat = avg_cross.flatten()

    # Define noise threshold - only include voxels above this
    # Use a small fraction of the peak value
    auto_threshold = avg_auto.max() * 0.01  # 1% of peak
    cross_threshold = avg_cross.max() * 0.01

    # Filter to positive values above threshold
    auto_positive_mask = auto_int_flat > auto_threshold
    cross_positive_mask = cross_int_flat > cross_threshold

    # Bin the data radially for cleaner visualization
    max_dist = min(shape) // 2
    n_bins = 30
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    auto_binned_mean = np.full(n_bins, np.nan)
    auto_binned_std = np.full(n_bins, np.nan)
    auto_binned_count = np.zeros(n_bins)
    cross_binned_mean = np.full(n_bins, np.nan)
    cross_binned_std = np.full(n_bins, np.nan)
    cross_binned_count = np.zeros(n_bins)

    for i in range(n_bins):
        # Auto: only positive values
        mask_auto = (auto_dist_flat >= bin_edges[i]) & (auto_dist_flat < bin_edges[i+1]) & auto_positive_mask
        if np.sum(mask_auto) > 0:
            auto_binned_mean[i] = np.mean(auto_int_flat[mask_auto])
            auto_binned_std[i] = np.std(auto_int_flat[mask_auto])
            auto_binned_count[i] = np.sum(mask_auto)

        # Cross: only positive values
        mask_cross = (cross_dist_flat >= bin_edges[i]) & (cross_dist_flat < bin_edges[i+1]) & cross_positive_mask
        if np.sum(mask_cross) > 0:
            cross_binned_mean[i] = np.mean(cross_int_flat[mask_cross])
            cross_binned_std[i] = np.std(cross_int_flat[mask_cross])
            cross_binned_count[i] = np.sum(mask_cross)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Radial Profile (positive values only)',
            'Voxel Scatter (subsampled)'
        ],
        horizontal_spacing=0.12
    )

    # Left: Binned radial profile
    # Only plot bins that have data
    valid_auto = ~np.isnan(auto_binned_mean)
    valid_cross = ~np.isnan(cross_binned_mean)

    fig.add_trace(go.Scatter(
        x=bin_centers[valid_auto], y=auto_binned_mean[valid_auto],
        mode='lines+markers',
        name='Auto-Corr',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=auto_binned_std[valid_auto], visible=True, thickness=1, width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bin_centers[valid_cross], y=cross_binned_mean[valid_cross],
        mode='lines+markers',
        name='Cross-Corr',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=cross_binned_std[valid_cross], visible=True, thickness=1, width=3)
    ), row=1, col=1)

    # Right: Scatter plot of individual voxels (subsampled for performance)
    # Show all positive voxels as scatter points
    max_points = 2000

    # Subsample if needed
    if np.sum(auto_positive_mask) > max_points:
        auto_indices = np.random.choice(np.where(auto_positive_mask)[0], max_points, replace=False)
    else:
        auto_indices = np.where(auto_positive_mask)[0]

    if np.sum(cross_positive_mask) > max_points:
        cross_indices = np.random.choice(np.where(cross_positive_mask)[0], max_points, replace=False)
    else:
        cross_indices = np.where(cross_positive_mask)[0]

    fig.add_trace(go.Scatter(
        x=auto_dist_flat[auto_indices],
        y=auto_int_flat[auto_indices],
        mode='markers',
        name='Auto voxels',
        marker=dict(size=3, color='#2ecc71', opacity=0.4),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=cross_dist_flat[cross_indices],
        y=cross_int_flat[cross_indices],
        mode='markers',
        name='Cross voxels',
        marker=dict(size=3, color='#3498db', opacity=0.4),
        showlegend=False
    ), row=1, col=2)

    # Add annotations for peak info
    auto_peak_val = avg_auto.max()
    cross_peak_val = avg_cross.max()
    cross_peak_offset = np.sqrt(
        (cross_peak_idx[0] - center[0])**2 +
        (cross_peak_idx[1] - center[1])**2 +
        (cross_peak_idx[2] - center[2])**2
    )

    # Count how many positive voxels
    n_auto_pos = np.sum(auto_positive_mask)
    n_cross_pos = np.sum(cross_positive_mask)
    n_total = len(auto_int_flat)

    fig.update_xaxes(title_text='Distance from Peak (voxels)', row=1, col=1)
    fig.update_xaxes(title_text='Distance from Peak (voxels)', row=1, col=2)
    fig.update_yaxes(title_text='Mean Correlation Intensity', row=1, col=1)
    fig.update_yaxes(title_text='Correlation Intensity', row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f'Radial Intensity Profile from Correlation Peaks (N={num_pairs})<br>'
                 f'<sub>Auto peak: {auto_peak_val:.2f} ({n_auto_pos}/{n_total} positive voxels) | '
                 f'Cross peak: {cross_peak_val:.2f} ({n_cross_pos}/{n_total} positive) | '
                 f'Offset: {cross_peak_offset:.2f} voxels</sub>',
            font=dict(size=16)
        ),
        height=500, width=1100,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=40, t=100, b=60)
    )

    return fig


def create_fitted_correlation_figure(
    avg_auto: np.ndarray,
    avg_cross: np.ndarray,
    result: StackedGaussianResult3D,
    num_pairs: int
) -> go.Figure:
    """
    Create figure comparing actual correlation data vs fitted Gaussian model.

    Shows XY and XZ slices through the peak for:
    - Actual auto-correlation vs fitted auto model
    - Actual cross-correlation vs fitted cross model
    """
    from gaussian_fit_stacked_3d import gaussian_3d

    shape = np.array(avg_auto.shape)
    center = shape // 2

    # Create coordinate grid for the full volume
    x = np.arange(shape[0]) - center[0]
    y = np.arange(shape[1]) - center[1]
    z = np.arange(shape[2]) - center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    # Generate fitted models
    sigma_geo = result.sigma_geo
    sigma_cross = result.sigma_geo + result.sigma_turb

    model_auto = gaussian_3d(
        coords, result.amp_auto, result.bg_auto,
        result.center_auto, sigma_geo
    ).reshape(shape)

    model_cross = gaussian_3d(
        coords, result.amp_cross, result.bg_cross,
        result.center_cross, sigma_cross
    ).reshape(shape)

    # Slice indices
    cx, cy, cz = center

    # Compute residuals
    residual_auto = avg_auto - model_auto
    residual_cross = avg_cross - model_cross

    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[
            'Auto Data (XY)', 'Auto Fit (XY)', 'Auto Residual (XY)', 'Auto X-Profile',
            'Auto Data (XZ)', 'Auto Fit (XZ)', 'Auto Residual (XZ)', 'Auto Z-Profile',
            'Cross Data (XY)', 'Cross Fit (XY)', 'Cross Residual (XY)', 'Cross X-Profile',
            'Cross Data (XZ)', 'Cross Fit (XZ)', 'Cross Residual (XZ)', 'Cross Z-Profile'
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08
    )

    # Color scale limits
    auto_vmax = max(avg_auto.max(), model_auto.max())
    cross_vmax = max(avg_cross.max(), model_cross.max())
    res_vmax = max(abs(residual_auto).max(), abs(residual_cross).max())

    # Row 1: Auto-correlation XY slice
    fig.add_trace(go.Heatmap(z=avg_auto[:, :, cz].T, colorscale='Viridis',
                             zmin=0, zmax=auto_vmax, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=model_auto[:, :, cz].T, colorscale='Viridis',
                             zmin=0, zmax=auto_vmax, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=residual_auto[:, :, cz].T, colorscale='RdBu_r',
                             zmid=0, zmin=-res_vmax, zmax=res_vmax, showscale=False), row=1, col=3)

    # Auto X-profile through center
    fig.add_trace(go.Scatter(x=x, y=avg_auto[:, cy, cz], mode='lines', name='Data',
                             line=dict(color='blue', width=2)), row=1, col=4)
    fig.add_trace(go.Scatter(x=x, y=model_auto[:, cy, cz], mode='lines', name='Fit',
                             line=dict(color='red', width=2, dash='dash')), row=1, col=4)

    # Row 2: Auto-correlation XZ slice
    fig.add_trace(go.Heatmap(z=avg_auto[:, cy, :].T, colorscale='Viridis',
                             zmin=0, zmax=auto_vmax, showscale=False), row=2, col=1)
    fig.add_trace(go.Heatmap(z=model_auto[:, cy, :].T, colorscale='Viridis',
                             zmin=0, zmax=auto_vmax, showscale=False), row=2, col=2)
    fig.add_trace(go.Heatmap(z=residual_auto[:, cy, :].T, colorscale='RdBu_r',
                             zmid=0, zmin=-res_vmax, zmax=res_vmax, showscale=False), row=2, col=3)

    # Auto Z-profile through center
    fig.add_trace(go.Scatter(x=z, y=avg_auto[cx, cy, :], mode='lines', name='Data',
                             line=dict(color='blue', width=2), showlegend=False), row=2, col=4)
    fig.add_trace(go.Scatter(x=z, y=model_auto[cx, cy, :], mode='lines', name='Fit',
                             line=dict(color='red', width=2, dash='dash'), showlegend=False), row=2, col=4)

    # Row 3: Cross-correlation XY slice
    fig.add_trace(go.Heatmap(z=avg_cross[:, :, cz].T, colorscale='Viridis',
                             zmin=0, zmax=cross_vmax, showscale=False), row=3, col=1)
    fig.add_trace(go.Heatmap(z=model_cross[:, :, cz].T, colorscale='Viridis',
                             zmin=0, zmax=cross_vmax, showscale=False), row=3, col=2)
    fig.add_trace(go.Heatmap(z=residual_cross[:, :, cz].T, colorscale='RdBu_r',
                             zmid=0, zmin=-res_vmax, zmax=res_vmax, showscale=False), row=3, col=3)

    # Cross X-profile through center
    fig.add_trace(go.Scatter(x=x, y=avg_cross[:, cy, cz], mode='lines', name='Data',
                             line=dict(color='blue', width=2), showlegend=False), row=3, col=4)
    fig.add_trace(go.Scatter(x=x, y=model_cross[:, cy, cz], mode='lines', name='Fit',
                             line=dict(color='red', width=2, dash='dash'), showlegend=False), row=3, col=4)

    # Row 4: Cross-correlation XZ slice
    fig.add_trace(go.Heatmap(z=avg_cross[:, cy, :].T, colorscale='Viridis',
                             zmin=0, zmax=cross_vmax, showscale=False), row=4, col=1)
    fig.add_trace(go.Heatmap(z=model_cross[:, cy, :].T, colorscale='Viridis',
                             zmin=0, zmax=cross_vmax, showscale=False), row=4, col=2)
    fig.add_trace(go.Heatmap(z=residual_cross[:, cy, :].T, colorscale='RdBu_r',
                             zmid=0, zmin=-res_vmax, zmax=res_vmax, showscale=False), row=4, col=3)

    # Cross Z-profile through center
    fig.add_trace(go.Scatter(x=z, y=avg_cross[cx, cy, :], mode='lines', name='Data',
                             line=dict(color='blue', width=2), showlegend=False), row=4, col=4)
    fig.add_trace(go.Scatter(x=z, y=model_cross[cx, cy, :], mode='lines', name='Fit',
                             line=dict(color='red', width=2, dash='dash'), showlegend=False), row=4, col=4)

    # Update axes labels
    for row in [1, 3]:
        fig.update_xaxes(title_text='X', row=row, col=4)
    for row in [2, 4]:
        fig.update_xaxes(title_text='Z', row=row, col=4)
    for row in [1, 2, 3, 4]:
        fig.update_yaxes(title_text='Intensity', row=row, col=4)

    # Compute fit quality metrics
    ss_res_auto = np.sum(residual_auto**2)
    ss_tot_auto = np.sum((avg_auto - avg_auto.mean())**2)
    r2_auto = 1 - ss_res_auto / ss_tot_auto

    ss_res_cross = np.sum(residual_cross**2)
    ss_tot_cross = np.sum((avg_cross - avg_cross.mean())**2)
    r2_cross = 1 - ss_res_cross / ss_tot_cross

    fig.update_layout(
        title=dict(
            text=f'Fitted vs Actual Correlation - XY and XZ Planes (N={num_pairs})<br>'
                 f'<sub>Auto R²={r2_auto:.4f} | Cross R²={r2_cross:.4f}</sub>',
            font=dict(size=16)
        ),
        height=900, width=1200,
        legend=dict(x=0.92, y=0.98),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


def create_reynolds_stress_figure(
    result: StackedGaussianResult3D,
    R_true: np.ndarray,
    num_pairs: int
) -> go.Figure:
    """
    Create visualization comparing fitted vs true Reynolds stress tensor.

    Parameters
    ----------
    result : StackedGaussianResult3D
        Fitting result containing sigma_turb (Reynolds stress)
    R_true : ndarray (3, 3)
        True Reynolds stress tensor used in simulation
    num_pairs : int
        Number of image pairs in ensemble
    """
    R_fitted = result.sigma_turb

    # Component labels and indices
    components = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]

    # Extract values
    true_vals = [R_true[i,j] for i,j in indices]
    fitted_vals = [R_fitted[i,j] for i,j in indices]
    abs_errors = [abs(f - t) for f, t in zip(fitted_vals, true_vals)]
    rel_errors = [100 * abs(f - t) / (abs(t) + 1e-10) for f, t in zip(fitted_vals, true_vals)]

    # Error metrics
    rms_error = np.sqrt(np.mean((R_fitted - R_true)**2))
    max_error = np.max(np.abs(R_fitted - R_true))
    mean_rel_error = np.mean(rel_errors)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "table", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        subplot_titles=[
            'Reynolds Stress Comparison',
            'Fitted vs True',
            'Fit Summary'
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15
    )

    # Main comparison table
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Component</b>', '<b>True</b>', '<b>Fitted</b>',
                   '<b>Abs Error</b>', '<b>Rel Error (%)</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=13),
            align='center',
            height=30
        ),
        cells=dict(
            values=[
                [f"<b>R<sub>{c}</sub></b>" for c in components],
                [f'{v:.4f}' for v in true_vals],
                [f'{v:.4f}' for v in fitted_vals],
                [f'{e:.4f}' for e in abs_errors],
                [f'{e:.1f}%' for e in rel_errors]
            ],
            fill_color=[
                ['#ecf0f1']*6,
                ['#e8f6e8']*6,
                ['#e8f0f6']*6,
                ['#ffeaa7' if e > 0.1 else '#e8f6e8' for e in abs_errors],
                ['#fab1a0' if e > 20 else '#e8f6e8' for e in rel_errors]
            ],
            align='center',
            font=dict(size=12),
            height=28
        )
    ), row=1, col=1)

    # Scatter plot: Fitted vs True
    fig.add_trace(go.Scatter(
        x=true_vals,
        y=fitted_vals,
        mode='markers+text',
        marker=dict(size=15, color='#3498db', line=dict(color='#2c3e50', width=2)),
        text=components,
        textposition='top center',
        name='Components',
        showlegend=False
    ), row=2, col=1)

    # Perfect fit line
    line_range = [min(min(true_vals), min(fitted_vals)) - 0.1,
                  max(max(true_vals), max(fitted_vals)) + 0.1]
    fig.add_trace(go.Scatter(
        x=line_range,
        y=line_range,
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect fit',
        showlegend=False
    ), row=2, col=1)

    # Summary statistics table
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                ['Ensemble size', 'Fit success', 'Fit cost',
                 'RMS error', 'Max error', 'Mean rel. error',
                 'Displacement (voxels)'],
                [f'{num_pairs} pairs',
                 'Yes' if result.success else 'No',
                 f'{result.cost:.4f}',
                 f'{rms_error:.4f}',
                 f'{max_error:.4f}',
                 f'{mean_rel_error:.1f}%',
                 f'[{result.displacement[0]:.2f}, {result.displacement[1]:.2f}, {result.displacement[2]:.2f}]']
            ],
            fill_color=[['#ecf0f1']*7, ['#f8f9fa']*7],
            align='left',
            font=dict(size=11)
        )
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text='True Value', row=2, col=1)
    fig.update_yaxes(title_text='Fitted Value', row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f'Reynolds Stress Extraction Results (N={num_pairs})<br>'
                 f'<sub>RMS Error: {rms_error:.4f} | Mean Relative Error: {mean_rel_error:.1f}%</sub>',
            font=dict(size=18)
        ),
        height=700,
        width=1000,
        margin=dict(l=60, r=40, t=100, b=40)
    )

    return fig


def create_ensemble_average_3d_figure(
    avg_auto: np.ndarray,
    avg_cross: np.ndarray,
    num_pairs: int
) -> go.Figure:
    """Create 3D isosurface of ensemble-averaged correlations."""
    shape = avg_auto.shape

    x = np.arange(shape[0]) - shape[0]//2
    y = np.arange(shape[1]) - shape[1]//2
    z = np.arange(shape[2]) - shape[2]//2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[f'Avg Auto-Corr (N={num_pairs})', f'Avg Cross-Corr (N={num_pairs})'],
        horizontal_spacing=0.05
    )

    for col, (corr, name) in enumerate([(avg_auto, 'Auto'), (avg_cross, 'Cross')], 1):
        vmax = corr.max()

        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=corr.flatten(),
            isomin=vmax * 0.2,
            isomax=vmax,
            surface_count=5,
            colorscale='Viridis',
            cmin=0,  # Start colormap from zero
            cmax=vmax,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=(col == 2),
            colorbar=dict(title='Corr', x=1.02) if col == 2 else None
        ), row=1, col=col)

    for scene_name in ['scene', 'scene2']:
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='ΔX (voxels)',
                yaxis_title='ΔY (voxels)',
                zaxis_title='ΔZ (voxels)',
                aspectmode='data'
            )
        })

    fig.update_layout(
        title=dict(text=f'Ensemble-Averaged 3D Correlation (N={num_pairs} pairs)', font=dict(size=16)),
        height=600, width=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def run_pipeline_visualization(
    num_pairs: int = 10,
    output_dir: str = './pipeline_viz',
    volume_size: Tuple[int, int, int] = (64, 64, 16),
    num_particles: int = 100,
    save_images: bool = False,
    save_every: int = 0
):
    """
    Run complete pipeline visualization.

    Parameters
    ----------
    num_pairs : int
        Number of image pairs to process
    output_dir : str
        Output directory
    volume_size : tuple
        Voxel dimensions (nx, ny, nz)
    num_particles : int
        Number of particles per image
    save_images : bool
        If True, save all generated camera images as .npy files
    save_every : int
        Save detailed visualizations every N pairs (0 = ensemble only)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEREO ENSEMBLE PIV - PIPELINE VISUALIZATION")
    print("=" * 60)

    # Configuration - Reynolds stress tensor (3x larger for better signal)
    R_true = np.array([
        [3.0, 0.9, 0.3],
        [0.9, 2.4, 0.6],
        [0.3, 0.6, 1.5]
    ])

    config = StereoEnsembleConfig(
        volume_size=volume_size,
        num_particles=num_particles,
        reynolds_stress=R_true,
        mean_displacement=np.array([0.0, 0.0, 0.0]),
        num_image_pairs=num_pairs,
        seed=42,
        particle_diameter_px=3.0,
        camera_angles_deg=(-45.0, 45.0),
        working_distance_mm=100.0,
        image_size_px=max(volume_size[0], volume_size[1]),
        scale_px_per_mm=15.0
    )

    print(f"\nConfiguration:")
    print(f"  Volume size: {config.volume_size}")
    print(f"  Image size: {config.image_size_px}x{config.image_size_px}")
    print(f"  Particles: {config.num_particles}")
    print(f"  Pairs: {num_pairs}")
    print(f"  Output: {output_path}")
    print(f"  Save images: {save_images}")
    print(f"  Save detailed viz every: {save_every if save_every > 0 else 'ensemble only'}")

    # Create subdirectories
    if save_images:
        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)

    # Initialize
    generator = StereoEnsembleGenerator(config)
    reconstructor = StereoMLOSReconstructor(
        generator.cameras, config.volume_size, config.scale_px_per_mm
    )

    # Storage for ensemble averaging (using proper accumulator with background subtraction)
    accumulator = EnsembleAccumulator3D(config.volume_size)

    print(f"\nProcessing {num_pairs} image pairs...")

    for pair_idx in range(num_pairs):
        # Progress indicator (less verbose for large runs)
        if num_pairs > 20:
            if pair_idx % 50 == 0 or pair_idx == num_pairs - 1:
                print(f"  Processing pair {pair_idx}/{num_pairs}...")
        else:
            print(f"\n--- Pair {pair_idx} ---")

        # Get particle positions
        pos_A_mm, pos_B_mm = get_particle_positions(generator, pair_idx)

        # Generate images
        images = generator.generate_image_pair(pair_idx)

        # Save images if requested (as 16-bit TIFF)
        if save_images:
            for key, img in images.items():
                # Normalize to 16-bit range and save as TIFF
                img_normalized = img / img.max() if img.max() > 0 else img
                img_16bit = (img_normalized * 65535).astype(np.uint16)
                tiff_img = Image.fromarray(img_16bit, mode='I;16')
                tiff_img.save(images_dir / f'pair_{pair_idx:04d}_{key}.tiff')

        # MinLOS reconstruction
        vol_A = reconstructor.reconstruct([images['cam1_A'], images['cam2_A']])
        vol_B = reconstructor.reconstruct([images['cam1_B'], images['cam2_B']])

        # Correlation maps (per-image)
        auto_corr = correlate_3d(vol_A, vol_A)
        cross_corr = correlate_3d(vol_A, vol_B)

        # Accumulate for ensemble (with proper background subtraction)
        accumulator.accumulate(vol_A, vol_B)

        # Save detailed visualizations every N pairs (if requested)
        should_save_detailed = (save_every > 0) and (pair_idx % save_every == 0)

        if should_save_detailed:
            print(f"  Saving detailed viz for pair {pair_idx}...")

            # Line-of-sight
            fig_los = create_line_of_sight_figure(generator, pos_A_mm, pair_idx)
            fig_los.write_html(output_path / f'pair_{pair_idx:04d}_1_line_of_sight.html')

            # MinLOS figures
            fig_minlos_A = create_minlos_figure(vol_A, generator, pos_A_mm, pair_idx, 'A')
            fig_minlos_A.write_html(output_path / f'pair_{pair_idx:04d}_2_minlos_A.html')

            fig_minlos_B = create_minlos_figure(vol_B, generator, pos_B_mm, pair_idx, 'B')
            fig_minlos_B.write_html(output_path / f'pair_{pair_idx:04d}_2_minlos_B.html')

            # Correlation figures
            fig_corr_3d = create_correlation_3d_figure(auto_corr, cross_corr, pair_idx)
            fig_corr_3d.write_html(output_path / f'pair_{pair_idx:04d}_3_correlation_3d.html')

    # 4. Ensemble averages WITH background subtraction
    # This subtracts <A>⋆<B> from <A⋆B> to remove DC offset from ghost correlations
    print(f"\n--- Ensemble Average (N={num_pairs}) ---")
    print(f"  Finalizing with background subtraction: <A⋆B> - <A>⋆<B>")
    avg_auto, avg_cross = accumulator.finalize()

    # Save correlation volumes for later analysis
    print(f"  Saving correlation volumes...")
    np.save(output_path / 'avg_auto_correlation.npy', avg_auto)
    np.save(output_path / 'avg_cross_correlation.npy', avg_cross)
    # Also save config info
    np.savez(output_path / 'correlation_metadata.npz',
             num_pairs=num_pairs,
             volume_size=np.array(config.volume_size),
             reynolds_stress_true=R_true,
             num_particles=config.num_particles)

    print(f"  Creating ensemble average figures...")
    fig_avg_slices = create_ensemble_average_figure(avg_auto, avg_cross, num_pairs)
    fig_avg_slices.write_html(output_path / f'ensemble_average_slices.html')

    fig_avg_3d = create_ensemble_average_3d_figure(avg_auto, avg_cross, num_pairs)
    fig_avg_3d.write_html(output_path / f'ensemble_average_3d.html')

    # 5. Distribution analysis and radial intensity plots
    print(f"  Creating intensity distribution analysis...")
    fig_distribution = create_intensity_distribution_figure(avg_auto, avg_cross, num_pairs)
    fig_distribution.write_html(output_path / f'ensemble_intensity_distribution.html')

    print(f"  Creating radial intensity profile...")
    fig_radial = create_radial_intensity_figure(avg_auto, avg_cross, num_pairs)
    fig_radial.write_html(output_path / f'ensemble_radial_profile.html')

    # 6. Reynolds stress extraction via 3D Gaussian fitting
    print(f"\n--- Reynolds Stress Extraction ---")
    print(f"  Fitting 22-parameter stacked 3D Gaussian...")
    roi_size = min(config.volume_size) // 2 - 2
    fit_result = fit_stacked_gaussian_3d(avg_auto, avg_cross, roi_size=roi_size)

    print(f"  Fit success: {fit_result.success}")
    print(f"  Fit cost: {fit_result.cost:.4f}")

    # Print Reynolds stress comparison
    print(f"\n  True Reynolds stress tensor:")
    for row in R_true:
        print(f"    [{row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}]")

    print(f"\n  Fitted Reynolds stress tensor:")
    for row in fit_result.sigma_turb:
        print(f"    [{row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}]")

    # Error metrics
    rms_error = np.sqrt(np.mean((fit_result.sigma_turb - R_true)**2))
    max_error = np.max(np.abs(fit_result.sigma_turb - R_true))
    abs_errors = np.abs(fit_result.sigma_turb - R_true)
    rel_errors = abs_errors / (np.abs(R_true) + 1e-10) * 100

    print(f"\n  Absolute error:")
    for row in abs_errors:
        print(f"    [{row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}]")

    print(f"\n  Error metrics:")
    print(f"    RMS error: {rms_error:.4f}")
    print(f"    Max error: {max_error:.4f}")
    print(f"    Mean rel. error: {np.mean(rel_errors):.1f}%")

    print(f"\n  Displacement: [{fit_result.displacement[0]:.3f}, {fit_result.displacement[1]:.3f}, {fit_result.displacement[2]:.3f}] voxels")

    # Create Reynolds stress figure
    print(f"  Creating Reynolds stress comparison figure...")
    fig_reynolds = create_reynolds_stress_figure(fit_result, R_true, num_pairs)
    fig_reynolds.write_html(output_path / f'ensemble_reynolds_stress.html')

    # Create fitted vs actual correlation figure
    print(f"  Creating fitted correlation comparison figure...")
    fig_fitted_corr = create_fitted_correlation_figure(avg_auto, avg_cross, fit_result, num_pairs)
    fig_fitted_corr.write_html(output_path / f'ensemble_fitted_correlation.html')

    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}/")
    print(f"\nGenerated files:")

    if save_images:
        print(f"\n  Images ({num_pairs * 4} files):")
        print(f"    images/pair_XXXX_cam[1,2]_[A,B].tiff")

    if save_every > 0:
        n_detailed = len([i for i in range(num_pairs) if i % save_every == 0])
        print(f"\n  Detailed viz ({n_detailed} pairs):")
        print(f"    pair_XXXX_1_line_of_sight.html")
        print(f"    pair_XXXX_2_minlos_A.html")
        print(f"    pair_XXXX_2_minlos_B.html")
        print(f"    pair_XXXX_3_correlation_3d.html")

    print(f"\n  Ensemble:")
    print(f"    - ensemble_average_slices.html")
    print(f"    - ensemble_average_3d.html")
    print(f"    - ensemble_intensity_distribution.html")
    print(f"    - ensemble_radial_profile.html")
    print(f"    - ensemble_reynolds_stress.html")
    print(f"    - ensemble_fitted_correlation.html")
    print(f"\n  Data files (for reloading):")
    print(f"    - avg_auto_correlation.npy")
    print(f"    - avg_cross_correlation.npy")
    print(f"    - correlation_metadata.npz")

    return avg_auto, avg_cross, fit_result


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline visualization for stereo ensemble PIV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-pairs', type=int, default=10, help='Number of image pairs')
    parser.add_argument('--output', type=str, default='./pipeline_viz', help='Output directory')
    parser.add_argument('--volume', type=int, nargs=3, default=[64, 64, 16], help='Volume size')
    parser.add_argument('--particles', type=int, default=100, help='Number of particles')
    parser.add_argument('--save-images', action='store_true', help='Save generated camera images')
    parser.add_argument('--save-every', type=int, default=0, help='Save detailed viz every N pairs (0=none)')

    args = parser.parse_args()

    run_pipeline_visualization(
        num_pairs=args.num_pairs,
        output_dir=args.output,
        volume_size=tuple(args.volume),
        num_particles=args.particles,
        save_images=args.save_images,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
