"""
Debug Visualization for Stereo Ensemble PIV
============================================

Creates detailed visualizations to debug the MinLOS reconstruction:
1. True particle positions with camera geometry and sight lines
2. MinLOS reconstruction with true particle positions overlaid
3. Camera images (raw stereo pairs)

This helps identify ghost artifacts and reconstruction quality.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Tuple, Optional, Dict

from stereo_ensemble_generator import StereoEnsembleGenerator, StereoEnsembleConfig
from correlation_3d import StereoMLOSReconstructor


def get_particle_positions_for_pair(
    generator: StereoEnsembleGenerator,
    pair_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recreate the exact particle positions for a given pair index.

    Uses the same seeding as generate_image_pair() to get reproducible positions.

    Returns
    -------
    pos_A_mm : ndarray (N, 3)
        Particle positions at time A in mm
    pos_B_mm : ndarray (N, 3)
        Particle positions at time B in mm
    displacements_px : ndarray (N, 3)
        Displacement vectors in pixels
    """
    cfg = generator.config
    base_seed = cfg.seed + pair_idx * 3

    # Recreate positions using same seeds as generate_image_pair()
    pos_A_mm = generator.generate_particles(seed=base_seed)
    displacements_px = generator.sample_displacements(len(pos_A_mm), seed=base_seed + 2)
    displacements_mm = displacements_px / cfg.scale_px_per_mm
    pos_B_mm = pos_A_mm + displacements_mm

    return pos_A_mm, pos_B_mm, displacements_px


def create_camera_geometry_figure(
    generator: StereoEnsembleGenerator,
    pos_A_mm: np.ndarray,
    pos_B_mm: np.ndarray,
    show_all_sightlines: bool = False,
    num_sightline_particles: int = 5
) -> go.Figure:
    """
    Create 3D visualization of camera geometry with particles and sight lines.

    Parameters
    ----------
    generator : StereoEnsembleGenerator
        Generator with camera models
    pos_A_mm : ndarray (N, 3)
        Particle positions at time A
    pos_B_mm : ndarray (N, 3)
        Particle positions at time B
    show_all_sightlines : bool
        If True, show sight lines for all particles (can be cluttered)
    num_sightline_particles : int
        Number of particles to show sight lines for (if not show_all)

    Returns
    -------
    fig : plotly Figure
    """
    cfg = generator.config
    cameras = generator.cameras

    # Volume bounds in mm
    vol_half = generator.volume_half_mm

    fig = go.Figure()

    # --- Volume bounding box ---
    # Create wireframe edges
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top
    ]) * vol_half

    edges = [
        (0,1), (1,2), (2,3), (3,0),  # bottom
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # verticals
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

    # --- Particles at time A (blue) ---
    fig.add_trace(go.Scatter3d(
        x=pos_A_mm[:, 0],
        y=pos_A_mm[:, 1],
        z=pos_A_mm[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.7),
        name='Particles (t=A)',
        hovertemplate='A: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # --- Particles at time B (red) ---
    fig.add_trace(go.Scatter3d(
        x=pos_B_mm[:, 0],
        y=pos_B_mm[:, 1],
        z=pos_B_mm[:, 2],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.7),
        name='Particles (t=B)',
        hovertemplate='B: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # --- Displacement vectors (arrows) for subset ---
    n_arrows = min(20, len(pos_A_mm))
    for i in range(n_arrows):
        fig.add_trace(go.Scatter3d(
            x=[pos_A_mm[i, 0], pos_B_mm[i, 0]],
            y=[pos_A_mm[i, 1], pos_B_mm[i, 1]],
            z=[pos_A_mm[i, 2], pos_B_mm[i, 2]],
            mode='lines',
            line=dict(color='green', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # --- Camera positions and orientations ---
    cam_colors = ['orange', 'purple']
    for cam_idx, cam in enumerate(cameras):
        # Camera position
        fig.add_trace(go.Scatter3d(
            x=[cam.position[0]],
            y=[cam.position[1]],
            z=[cam.position[2]],
            mode='markers+text',
            marker=dict(size=10, color=cam_colors[cam_idx], symbol='diamond'),
            text=[cam.name],
            textposition='top center',
            name=f'{cam.name} ({cam.angle_deg:.0f}°)',
            hovertemplate=f'{cam.name}<br>Pos: ({cam.position[0]:.1f}, {cam.position[1]:.1f}, {cam.position[2]:.1f})<extra></extra>'
        ))

        # View direction cone (simplified as line to origin)
        fig.add_trace(go.Scatter3d(
            x=[cam.position[0], 0],
            y=[cam.position[1], 0],
            z=[cam.position[2], 0],
            mode='lines',
            line=dict(color=cam_colors[cam_idx], width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # --- Sight lines through selected particles ---
    if show_all_sightlines:
        particles_for_sightlines = range(len(pos_A_mm))
    else:
        # Select particles spread across the volume
        indices = np.linspace(0, len(pos_A_mm)-1, num_sightline_particles, dtype=int)
        particles_for_sightlines = indices

    for p_idx in particles_for_sightlines:
        particle_pos = pos_A_mm[p_idx]

        for cam_idx, cam in enumerate(cameras):
            # Direction from camera to particle
            direction = particle_pos - cam.position
            direction = direction / np.linalg.norm(direction)

            # Extend line through particle (from camera, past particle)
            t_start = 0  # at camera
            t_end = np.linalg.norm(particle_pos - cam.position) * 1.5  # past particle

            line_start = cam.position
            line_end = cam.position + direction * t_end

            fig.add_trace(go.Scatter3d(
                x=[line_start[0], line_end[0]],
                y=[line_start[1], line_end[1]],
                z=[line_start[2], line_end[2]],
                mode='lines',
                line=dict(color=cam_colors[cam_idx], width=1, dash='dot'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

    # --- Layout ---
    max_range = max(np.max(np.abs(cameras[0].position)),
                    np.max(np.abs(cameras[1].position)),
                    np.max(vol_half)) * 1.2

    fig.update_layout(
        title=dict(
            text='Camera Geometry & Particle Positions',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
        width=900,
        height=700
    )

    return fig


def create_minlos_debug_figure(
    generator: StereoEnsembleGenerator,
    reconstructor: StereoMLOSReconstructor,
    images: Dict[str, np.ndarray],
    pos_A_mm: np.ndarray,
    pos_B_mm: np.ndarray,
    frame: str = 'A',
    threshold_percentile: float = 90
) -> go.Figure:
    """
    Create visualization of MinLOS reconstruction with true particle positions.

    Parameters
    ----------
    generator : StereoEnsembleGenerator
        Generator with config
    reconstructor : StereoMLOSReconstructor
        MinLOS reconstructor
    images : dict
        Image dictionary from generate_image_pair()
    pos_A_mm, pos_B_mm : ndarray
        True particle positions
    frame : str
        'A' or 'B' - which frame to visualize
    threshold_percentile : float
        Percentile threshold for volume visualization

    Returns
    -------
    fig : plotly Figure
    """
    cfg = generator.config

    # Select images for this frame
    if frame == 'A':
        img_list = [images['cam1_A'], images['cam2_A']]
        pos_mm = pos_A_mm
        frame_name = 'Frame A'
    else:
        img_list = [images['cam1_B'], images['cam2_B']]
        pos_mm = pos_B_mm
        frame_name = 'Frame B'

    # Perform MinLOS reconstruction
    volume = reconstructor.reconstruct(img_list)

    # Create voxel grid coordinates in mm
    nx, ny, nz = cfg.volume_size
    x_mm = (np.arange(nx) - nx/2 + 0.5) / cfg.scale_px_per_mm
    y_mm = (np.arange(ny) - ny/2 + 0.5) / cfg.scale_px_per_mm
    z_mm = (np.arange(nz) - nz/2 + 0.5) / cfg.scale_px_per_mm

    # Get threshold for visualization
    threshold = np.percentile(volume[volume > 0], threshold_percentile)

    fig = go.Figure()

    # --- MinLOS volume as isosurface ---
    X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')

    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=volume.max(),
        surface_count=3,
        colorscale='Viridis',
        opacity=0.3,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='MinLOS Volume',
        showscale=True,
        colorbar=dict(title='Intensity', x=1.02)
    ))

    # --- True particle positions (larger markers) ---
    fig.add_trace(go.Scatter3d(
        x=pos_mm[:, 0],
        y=pos_mm[:, 1],
        z=pos_mm[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='cross',
            line=dict(color='white', width=1)
        ),
        name=f'True Particles ({frame_name})',
        hovertemplate='True: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

    # --- Volume bounding box ---
    vol_half = generator.volume_half_mm
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

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f'MinLOS Reconstruction - {frame_name}<br><sub>Red crosses = true particle positions</sub>',
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
        margin=dict(l=0, r=0, t=60, b=0),
        width=900,
        height=700
    )

    return fig


def create_minlos_slices_figure(
    generator: StereoEnsembleGenerator,
    reconstructor: StereoMLOSReconstructor,
    images: Dict[str, np.ndarray],
    pos_A_mm: np.ndarray,
    frame: str = 'A'
) -> go.Figure:
    """
    Create XY, XZ, YZ slice views through MinLOS volume at particle locations.

    Shows 2D slices through the 3D reconstruction to visualize ghost structure.
    """
    cfg = generator.config

    if frame == 'A':
        img_list = [images['cam1_A'], images['cam2_A']]
        pos_mm = pos_A_mm
    else:
        img_list = [images['cam1_B'], images['cam2_B']]
        pos_mm = pos_A_mm  # Use A positions for slice locations

    volume = reconstructor.reconstruct(img_list)
    nx, ny, nz = cfg.volume_size

    # Coordinates in mm
    x_mm = (np.arange(nx) - nx/2 + 0.5) / cfg.scale_px_per_mm
    y_mm = (np.arange(ny) - ny/2 + 0.5) / cfg.scale_px_per_mm
    z_mm = (np.arange(nz) - nz/2 + 0.5) / cfg.scale_px_per_mm

    # Find a particle near center for slicing
    center_dists = np.linalg.norm(pos_mm, axis=1)
    center_particle_idx = np.argmin(center_dists)
    slice_pos_mm = pos_mm[center_particle_idx]

    # Convert to voxel indices
    slice_x = int((slice_pos_mm[0] * cfg.scale_px_per_mm) + nx/2)
    slice_y = int((slice_pos_mm[1] * cfg.scale_px_per_mm) + ny/2)
    slice_z = int((slice_pos_mm[2] * cfg.scale_px_per_mm) + nz/2)

    # Clamp to valid range
    slice_x = np.clip(slice_x, 0, nx-1)
    slice_y = np.clip(slice_y, 0, ny-1)
    slice_z = np.clip(slice_z, 0, nz-1)

    # Extract slices
    xy_slice = volume[:, :, slice_z]  # XY plane at Z
    xz_slice = volume[:, slice_y, :]  # XZ plane at Y
    yz_slice = volume[slice_x, :, :]  # YZ plane at X

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'XY Slice (Z={z_mm[slice_z]:.2f}mm)',
            f'XZ Slice (Y={y_mm[slice_y]:.2f}mm)',
            f'YZ Slice (X={x_mm[slice_x]:.2f}mm)'
        ],
        horizontal_spacing=0.08
    )

    # XY slice
    fig.add_trace(go.Heatmap(
        z=xy_slice.T,
        x=x_mm, y=y_mm,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='X=%{x:.2f}mm<br>Y=%{y:.2f}mm<br>I=%{z:.2f}<extra></extra>'
    ), row=1, col=1)

    # Mark particles in this slice (within 1 voxel of Z)
    z_tol = 1.0 / cfg.scale_px_per_mm
    in_slice = np.abs(pos_mm[:, 2] - z_mm[slice_z]) < z_tol
    fig.add_trace(go.Scatter(
        x=pos_mm[in_slice, 0],
        y=pos_mm[in_slice, 1],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        showlegend=False,
        hovertemplate='True particle<extra></extra>'
    ), row=1, col=1)

    # XZ slice
    fig.add_trace(go.Heatmap(
        z=xz_slice.T,
        x=x_mm, y=z_mm,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='X=%{x:.2f}mm<br>Z=%{y:.2f}mm<br>I=%{z:.2f}<extra></extra>'
    ), row=1, col=2)

    # Mark particles in XZ slice
    y_tol = 1.0 / cfg.scale_px_per_mm
    in_slice_xz = np.abs(pos_mm[:, 1] - y_mm[slice_y]) < y_tol
    fig.add_trace(go.Scatter(
        x=pos_mm[in_slice_xz, 0],
        y=pos_mm[in_slice_xz, 2],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        showlegend=False
    ), row=1, col=2)

    # YZ slice
    fig.add_trace(go.Heatmap(
        z=yz_slice.T,
        x=y_mm, y=z_mm,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Intensity', x=1.02),
        hovertemplate='Y=%{x:.2f}mm<br>Z=%{y:.2f}mm<br>I=%{z:.2f}<extra></extra>'
    ), row=1, col=3)

    # Mark particles in YZ slice
    x_tol = 1.0 / cfg.scale_px_per_mm
    in_slice_yz = np.abs(pos_mm[:, 0] - x_mm[slice_x]) < x_tol
    fig.add_trace(go.Scatter(
        x=pos_mm[in_slice_yz, 1],
        y=pos_mm[in_slice_yz, 2],
        mode='markers',
        marker=dict(size=8, color='red', symbol='cross'),
        showlegend=False
    ), row=1, col=3)

    fig.update_layout(
        title=dict(
            text=f'MinLOS Slices Through Particle at ({slice_pos_mm[0]:.2f}, {slice_pos_mm[1]:.2f}, {slice_pos_mm[2]:.2f})mm<br><sub>Red crosses = true particle positions in slice</sub>',
            font=dict(size=14)
        ),
        height=400,
        width=1200,
        margin=dict(t=80)
    )

    # Update axes labels
    fig.update_xaxes(title_text='X (mm)', row=1, col=1)
    fig.update_yaxes(title_text='Y (mm)', row=1, col=1)
    fig.update_xaxes(title_text='X (mm)', row=1, col=2)
    fig.update_yaxes(title_text='Z (mm)', row=1, col=2)
    fig.update_xaxes(title_text='Y (mm)', row=1, col=3)
    fig.update_yaxes(title_text='Z (mm)', row=1, col=3)

    return fig


def create_camera_images_figure(images: Dict[str, np.ndarray]) -> go.Figure:
    """
    Create figure showing all 4 camera images (2 cameras x 2 frames).
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Camera 1 - Frame A', 'Camera 2 - Frame A',
                       'Camera 1 - Frame B', 'Camera 2 - Frame B'],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    img_keys = [('cam1_A', 1, 1), ('cam2_A', 1, 2),
                ('cam1_B', 2, 1), ('cam2_B', 2, 2)]

    for key, row, col in img_keys:
        fig.add_trace(go.Heatmap(
            z=images[key],
            colorscale='gray',
            showscale=False,
            hovertemplate='u=%{x}<br>v=%{y}<br>I=%{z:.2f}<extra></extra>'
        ), row=row, col=col)

    fig.update_layout(
        title='Raw Camera Images',
        height=700,
        width=800,
        margin=dict(t=60)
    )

    # Make axes equal
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(scaleanchor=f'y{i}{j}' if i > 1 or j > 1 else 'y',
                           scaleratio=1, row=i, col=j)

    return fig


def create_ghost_analysis_figure(
    generator: StereoEnsembleGenerator,
    reconstructor: StereoMLOSReconstructor,
    images: Dict[str, np.ndarray],
    pos_A_mm: np.ndarray
) -> go.Figure:
    """
    Create detailed analysis of ghost formation for a single particle.

    Shows:
    - XZ slice through the particle (camera plane)
    - Expected ghost arm directions based on camera angles
    """
    cfg = generator.config
    cameras = generator.cameras

    volume = reconstructor.reconstruct([images['cam1_A'], images['cam2_A']])
    nx, ny, nz = cfg.volume_size

    x_mm = (np.arange(nx) - nx/2 + 0.5) / cfg.scale_px_per_mm
    z_mm = (np.arange(nz) - nz/2 + 0.5) / cfg.scale_px_per_mm

    # Find brightest particle (highest local intensity)
    # Use particle closest to center for cleaner visualization
    center_dists = np.linalg.norm(pos_A_mm, axis=1)
    particle_idx = np.argmin(center_dists)
    particle_pos = pos_A_mm[particle_idx]

    # Get Y slice through this particle
    slice_y = int((particle_pos[1] * cfg.scale_px_per_mm) + ny/2)
    slice_y = np.clip(slice_y, 0, ny-1)
    xz_slice = volume[:, slice_y, :]

    fig = go.Figure()

    # XZ heatmap
    fig.add_trace(go.Heatmap(
        z=xz_slice.T,
        x=x_mm, y=z_mm,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Intensity'),
        hovertemplate='X=%{x:.2f}mm<br>Z=%{y:.2f}mm<br>I=%{z:.2f}<extra></extra>'
    ))

    # Mark true particle position
    fig.add_trace(go.Scatter(
        x=[particle_pos[0]],
        y=[particle_pos[2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='cross', line=dict(width=2)),
        name='True Position'
    ))

    # Draw expected ghost arm directions from camera angles
    arm_length = 3.0  # mm
    for cam_idx, cam in enumerate(cameras):
        # Direction from camera to particle (projected to XZ plane)
        direction = particle_pos - cam.position
        direction[1] = 0  # Project to XZ plane
        direction = direction / np.linalg.norm(direction)

        # Ghost extends along this direction from particle
        ghost_start = particle_pos[[0, 2]] - direction[[0, 2]] * arm_length
        ghost_end = particle_pos[[0, 2]] + direction[[0, 2]] * arm_length

        color = 'orange' if cam_idx == 0 else 'purple'
        fig.add_trace(go.Scatter(
            x=[ghost_start[0], ghost_end[0]],
            y=[ghost_start[1], ghost_end[1]],
            mode='lines',
            line=dict(color=color, width=3, dash='dash'),
            name=f'{cam.name} sight line ({cam.angle_deg:.0f}°)'
        ))

    fig.update_layout(
        title=dict(
            text=f'Ghost Analysis - XZ Slice at Y={particle_pos[1]:.2f}mm<br><sub>Dashed lines show camera sight line directions (ghost arm axes)</sub>',
            font=dict(size=14)
        ),
        xaxis_title='X (mm)',
        yaxis_title='Z (mm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        height=600,
        width=800,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def run_debug_visualization(
    generator: StereoEnsembleGenerator,
    reconstructor: StereoMLOSReconstructor,
    output_dir: Optional[str] = None,
    show: bool = True
) -> Dict[str, go.Figure]:
    """
    Run all debug visualizations for the first image pair.

    Parameters
    ----------
    generator : StereoEnsembleGenerator
        Configured generator
    reconstructor : StereoMLOSReconstructor
        MinLOS reconstructor
    output_dir : str, optional
        Directory to save HTML files
    show : bool
        Whether to open figures in browser

    Returns
    -------
    figures : dict
        Dictionary of figure name -> plotly Figure
    """
    print("Generating debug visualizations for pair 0...")

    # Get positions for first pair
    pos_A_mm, pos_B_mm, displacements_px = get_particle_positions_for_pair(generator, pair_idx=0)

    # Generate images for first pair
    images = generator.generate_image_pair(0)

    figures = {}

    # 1. Camera geometry with particles
    print("  Creating camera geometry figure...")
    figures['camera_geometry'] = create_camera_geometry_figure(
        generator, pos_A_mm, pos_B_mm, num_sightline_particles=5
    )

    # 2. MinLOS reconstruction with true positions
    print("  Creating MinLOS 3D figure...")
    figures['minlos_3d'] = create_minlos_debug_figure(
        generator, reconstructor, images, pos_A_mm, pos_B_mm, frame='A'
    )

    # 3. MinLOS slices
    print("  Creating MinLOS slices figure...")
    figures['minlos_slices'] = create_minlos_slices_figure(
        generator, reconstructor, images, pos_A_mm, frame='A'
    )

    # 4. Raw camera images
    print("  Creating camera images figure...")
    figures['camera_images'] = create_camera_images_figure(images)

    # 5. Ghost analysis
    print("  Creating ghost analysis figure...")
    figures['ghost_analysis'] = create_ghost_analysis_figure(
        generator, reconstructor, images, pos_A_mm
    )

    # Save/show
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            filepath = output_path / f'debug_{name}.html'
            fig.write_html(str(filepath))
            print(f"  Saved: {filepath}")

    if show:
        # Show the most informative figures
        figures['ghost_analysis'].show()
        figures['minlos_3d'].show()

    return figures


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debug visualization for stereo ensemble PIV')
    parser.add_argument('--particles', type=int, default=50, help='Number of particles')
    parser.add_argument('--volume', type=int, nargs=3, default=[64, 64, 16], help='Volume size')
    parser.add_argument('--output', type=str, default='./debug_output', help='Output directory')
    parser.add_argument('--no-show', action='store_true', help='Do not open in browser')

    args = parser.parse_args()

    # Create config
    config = StereoEnsembleConfig(
        volume_size=tuple(args.volume),
        num_particles=args.particles,
        num_image_pairs=1,
        seed=42,
        particle_diameter_px=3.0,
        camera_angles_deg=(-45.0, 45.0),
        working_distance_mm=100.0,
        image_size_px=128,
        scale_px_per_mm=15.0
    )

    generator = StereoEnsembleGenerator(config)
    reconstructor = StereoMLOSReconstructor(
        generator.cameras,
        config.volume_size,
        config.scale_px_per_mm
    )

    run_debug_visualization(
        generator, reconstructor,
        output_dir=args.output,
        show=not args.no_show
    )
