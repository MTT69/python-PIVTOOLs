"""
Comprehensive debug script that runs PIV and visualizes vectors at each step.
This will help identify exactly where the indexing goes wrong.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import read_pair
from pypivtools.piv.piv_backend.factory import make_correlator_backend

def visualize_vectors(x_grid, y_grid, ux, uy, image, title, filename, subsample=1):
    """
    Visualize velocity vectors overlaid on image.
    
    Args:
        x_grid: (n_y, n_x) array of x-coordinates
        y_grid: (n_y, n_x) array of y-coordinates  
        ux: (n_y, n_x) array of x-velocity components
        uy: (n_y, n_x) array of y-velocity components
        image: Background image
        title: Plot title
        filename: Output filename
        subsample: Plot every Nth vector
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Show image
    ax.imshow(image, cmap='gray', origin='upper', alpha=0.7)
    
    # Subsample for clearer visualization
    x_sub = x_grid[::subsample, ::subsample]
    y_sub = y_grid[::subsample, ::subsample]
    ux_sub = ux[::subsample, ::subsample]
    uy_sub = uy[::subsample, ::subsample]
    
    # Compute magnitude for coloring
    mag = np.sqrt(ux_sub**2 + uy_sub**2)
    
    # Plot quiver
    Q = ax.quiver(x_sub, y_sub, ux_sub, uy_sub, mag,
                  cmap='jet', scale=500, width=0.003, alpha=0.9)
    
    plt.colorbar(Q, ax=ax, label='Velocity Magnitude (pixels)')
    
    # Add information
    info_text = f"Shape: ux={ux.shape}, uy={uy.shape}\n"
    info_text += f"Ux range: [{np.nanmin(ux):.3f}, {np.nanmax(ux):.3f}]\n"
    info_text += f"Uy range: [{np.nanmin(uy):.3f}, {np.nanmax(uy):.3f}]\n"
    info_text += f"Mean |U|: {np.nanmean(np.sqrt(ux**2 + uy**2)):.3f}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    # Load config and image
    config = Config()
    print("=" * 80)
    print("VECTOR VISUALIZATION DEBUG")
    print("=" * 80)
    print(f"Image shape: {config.image_shape} (H={config.image_shape[0]}, W={config.image_shape[1]})")
    print()
    
    # Load one image pair
    camera_num = config.camera_numbers[0]
    source_path = config.source_paths[0]
    camera_path = source_path / f"Cam{camera_num}"
    
    print(f"Loading image from: {camera_path}")
    image_pair = read_pair(1, camera_path, config)
    frame_a = image_pair[0]
    frame_b = image_pair[1]
    print(f"Image pair loaded: {image_pair.shape}")
    print()
    
    # Create correlator and run first pass only
    correlator = make_correlator_backend(config)
    
    print("=" * 80)
    print("RUNNING FIRST PIV PASS")
    print("=" * 80)
    
    # Add batch dimension for correlator
    images = image_pair[np.newaxis, ...]  # (1, 2, H, W)
    
    # Run correlation
    piv_result = correlator.correlate_batch(images, config=config, mask=None)
    
    # Extract results from first pass
    pass_result = piv_result.passes[0]
    ux_mat = pass_result.ux_mat
    uy_mat = pass_result.uy_mat
    
    print("Results from PIVPassResult:")
    print(f"  ux_mat shape: {ux_mat.shape}")
    print(f"  uy_mat shape: {uy_mat.shape}")
    print()
    print(f"  Ux range: [{np.nanmin(ux_mat):.3f}, {np.nanmax(ux_mat):.3f}]")
    print(f"  Uy range: [{np.nanmin(uy_mat):.3f}, {np.nanmax(uy_mat):.3f}]")
    print()
    print(f"  ux_mat[0,0:5] = {ux_mat[0,0:5]}")
    print(f"  uy_mat[0,0:5] = {uy_mat[0,0:5]}")
    print()
    
    # Get window centers for reference
    win_ctrs_x = correlator.win_ctrs_x[0]  # (n_x,)
    win_ctrs_y = correlator.win_ctrs_y[0]  # (n_y,)
    n_x = len(win_ctrs_x)
    n_y = len(win_ctrs_y)
    
    print("Window centers:")
    print(f"  n_x = {n_x} (along width)")
    print(f"  n_y = {n_y} (along height)")
    print(f"  win_ctrs_x range: [{win_ctrs_x[0]:.1f}, {win_ctrs_x[-1]:.1f}]")
    print(f"  win_ctrs_y range: [{win_ctrs_y[0]:.1f}, {win_ctrs_y[-1]:.1f}]")
    print()
    
    # Create meshgrid for visualization (this is the coordinate grid)
    # meshgrid with default indexing='xy' produces:
    # x_grid[i,j] has constant values along rows (i), varies along cols (j)
    # y_grid[i,j] has constant values along cols (j), varies along rows (i)
    x_grid_xy, y_grid_xy = np.meshgrid(win_ctrs_x, win_ctrs_y, indexing='xy')
    print("Meshgrid with indexing='xy' (default):")
    print(f"  x_grid shape: {x_grid_xy.shape} (n_y, n_x) = ({n_y}, {n_x})")
    print(f"  y_grid shape: {y_grid_xy.shape}")
    print()
    
    # Also create with 'ij' indexing which matches C library order
    x_grid_ij, y_grid_ij = np.meshgrid(win_ctrs_x, win_ctrs_y, indexing='ij')
    print("Meshgrid with indexing='ij':")
    print(f"  x_grid shape: {x_grid_ij.shape} (n_x, n_y) = ({n_x}, {n_y})")
    print(f"  y_grid shape: {y_grid_ij.shape}")
    print()
    
    # Verify the coordinate grids
    print("Coordinate grid verification:")
    print(f"  ux_mat.shape={ux_mat.shape}")
    if ux_mat.shape == x_grid_xy.shape:
        print(f"  ✓ Matches 'xy' indexing: {x_grid_xy.shape}")
        x_grid, y_grid = x_grid_xy, y_grid_xy
    elif ux_mat.shape == x_grid_ij.shape:
        print(f"  ✓ Matches 'ij' indexing: {x_grid_ij.shape}")
        x_grid, y_grid = x_grid_ij, y_grid_ij
    else:
        print(f"  ✗ No match! ux_mat={ux_mat.shape}, xy={x_grid_xy.shape}, ij={x_grid_ij.shape}")
        # Use xy as default
        x_grid, y_grid = x_grid_xy, y_grid_xy
    print()
    
    # Visualize the vectors
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Plot 1: Current output (as returned by correlator)
    visualize_vectors(
        x_grid, y_grid, ux_mat, uy_mat, frame_a,
        "Current PIV Output (First Pass)\nChannel flow should go left→right",
        "/tmp/piv_current_output.png",
        subsample=1
    )
    
    # Plot 2: What if we transpose ux and uy?
    visualize_vectors(
        x_grid, y_grid, ux_mat.T, uy_mat.T, frame_a,
        "TRANSPOSED ux/uy (Test)\nDoes this look more correct?",
        "/tmp/piv_transposed_velocities.png",
        subsample=1
    )
    
    # Plot 3: What if we swap ux and uy (without transpose)?
    visualize_vectors(
        x_grid, y_grid, uy_mat, ux_mat, frame_a,
        "SWAPPED ux↔uy (Test)\nDoes this look more correct?",
        "/tmp/piv_swapped_velocities.png",
        subsample=1
    )
    
    # Plot 4: What if we transpose AND swap?
    visualize_vectors(
        x_grid, y_grid, uy_mat.T, ux_mat.T, frame_a,
        "TRANSPOSED+SWAPPED ux↔uy (Test)\nDoes this look more correct?",
        "/tmp/piv_trans_swap_velocities.png",
        subsample=1
    )
    
    # Plot 5: Statistical comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Histogram of Ux
    ax = axes[0, 0]
    ax.hist(ux_mat[~np.isnan(ux_mat)].flatten(), bins=30, alpha=0.7, label='Current Ux')
    ax.hist(uy_mat[~np.isnan(uy_mat)].flatten(), bins=30, alpha=0.7, label='Current Uy')
    ax.set_xlabel('Velocity (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Current Output Histograms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spatial distribution - Ux
    ax = axes[0, 1]
    im = ax.imshow(ux_mat, cmap='RdBu_r', origin='upper')
    ax.set_title(f'Current Ux Spatial Distribution\nShape: {ux_mat.shape}')
    ax.set_xlabel('j (column index)')
    ax.set_ylabel('i (row index)')
    plt.colorbar(im, ax=ax)
    
    # Spatial distribution - Uy
    ax = axes[1, 0]
    im = ax.imshow(uy_mat, cmap='RdBu_r', origin='upper')
    ax.set_title(f'Current Uy Spatial Distribution\nShape: {uy_mat.shape}')
    ax.set_xlabel('j (column index)')
    ax.set_ylabel('i (row index)')
    plt.colorbar(im, ax=ax)
    
    # Profile comparison
    ax = axes[1, 1]
    # Take middle row and middle column
    mid_row = ux_mat.shape[0] // 2
    mid_col = ux_mat.shape[1] // 2
    ax.plot(ux_mat[mid_row, :], 'o-', label=f'Ux along row {mid_row}')
    ax.plot(ux_mat[:, mid_col], 's-', label=f'Ux along col {mid_col}')
    ax.plot(uy_mat[mid_row, :], '^-', label=f'Uy along row {mid_row}')
    ax.plot(uy_mat[:, mid_col], 'v-', label=f'Uy along col {mid_col}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Velocity (pixels)')
    ax.set_title('Velocity Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/piv_statistical_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/piv_statistical_analysis.png")
    plt.close()
    
    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("Expected for channel flow (left→right):")
    print("  • Ux should be LARGE (main flow)")
    print("  • Uy should be SMALL (cross-stream)")
    print("  • Vectors should point predominantly to the RIGHT")
    print()
    print("Current results:")
    print(f"  • Ux: mean={np.nanmean(ux_mat):.3f}, std={np.nanstd(ux_mat):.3f}, range=[{np.nanmin(ux_mat):.3f}, {np.nanmax(ux_mat):.3f}]")
    print(f"  • Uy: mean={np.nanmean(uy_mat):.3f}, std={np.nanstd(uy_mat):.3f}, range=[{np.nanmin(uy_mat):.3f}, {np.nanmax(uy_mat):.3f}]")
    print()
    print("Look at the generated images in /tmp/piv_*.png to see which transformation")
    print("produces the correct left→right flow!")
    print("=" * 80)

if __name__ == "__main__":
    main()
