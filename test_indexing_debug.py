"""
Debug script to visualize indexing at each step of PIV processing.
This helps identify where the row/column indexing gets mixed up.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import read_pair, load_mask_for_camera

# Load configuration
config = Config()
print("=" * 80)
print("IMAGE AND CONFIGURATION INFO")
print("=" * 80)
print(f"Image shape: {config.image_shape} = (height={config.image_shape[0]}, width={config.image_shape[1]})")
print("Expected: more vectors in y (height) than x (width)")
print()

# Load one image pair
camera = 1
source = config.source_paths[0]
camera_path = source / f"Cam{camera}"
image_pair = read_pair(1, camera_path, config)

print(f"Loaded image pair shape: {image_pair.shape}")
print(f"Frame A: {image_pair[0].shape}")
print(f"Frame B: {image_pair[1].shape}")
print()

# Create a simple test: compute window centers for pass 0
from pypivtools.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU

correlator = InstantaneousCorrelatorCPU(config)

# Get window info for first pass
pass_idx = 0
win_ctrs_x = correlator.win_ctrs_x[pass_idx]
win_ctrs_y = correlator.win_ctrs_y[pass_idx]
n_x = len(win_ctrs_x)
n_y = len(win_ctrs_y)

print("=" * 80)
print(f"PASS {pass_idx} - WINDOW CENTERS")
print("=" * 80)
print(f"win_ctrs_x: {n_x} points (along x-axis/width)")
print(f"  Range: [{win_ctrs_x[0]:.1f}, {win_ctrs_x[-1]:.1f}]")
print(f"  Spacing: {win_ctrs_x[1] - win_ctrs_x[0]:.1f} pixels")
print()
print(f"win_ctrs_y: {n_y} points (along y-axis/height)")
print(f"  Range: [{win_ctrs_y[0]:.1f}, {win_ctrs_y[-1]:.1f}]")
print(f"  Spacing: {win_ctrs_y[1] - win_ctrs_y[0]:.1f} pixels")
print()
print(f"Total windows: {n_x} × {n_y} = {n_x * n_y}")
print(f"Expected shape for vectors: (n_y={n_y}, n_x={n_x}) = ({n_y}, {n_x})")
print()

# Visualize the window grid
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Image with window centers
ax = axes[0]
ax.imshow(image_pair[0], cmap='gray', origin='upper')
win_x_grid, win_y_grid = np.meshgrid(win_ctrs_x, win_ctrs_y)
ax.plot(win_x_grid, win_y_grid, 'r+', markersize=8, markeredgewidth=2)
ax.set_title(f'Image with Window Centers\n{n_y} rows × {n_x} cols = {n_x*n_y} windows')
ax.set_xlabel('x (columns/width)')
ax.set_ylabel('y (rows/height)')
ax.grid(True, alpha=0.3)

# Add coordinate annotations
ax.text(0.02, 0.98, 'Origin (0,0)', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(0.98, 0.98, f'Top Right ({config.image_shape[1]-1},0)', transform=ax.transAxes,
        horizontalalignment='right', verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(0.02, 0.02, f'Bottom Left (0,{config.image_shape[0]-1})', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Plot 2: Expected flow direction
ax = axes[1]
ax.imshow(image_pair[0], cmap='gray', origin='upper', alpha=0.5)
# Draw arrows showing expected flow (left to right, horizontal)
y_positions = np.linspace(100, config.image_shape[0]-100, 5)
x_start = 100
x_end = config.image_shape[1] - 100
for y in y_positions:
    ax.arrow(x_start, y, x_end-x_start-50, 0, 
            head_width=30, head_length=30, fc='red', ec='red', linewidth=3)
ax.set_title('Expected Flow Direction\n(Channel flow: left → right)')
ax.set_xlabel('x (columns/width)')
ax.set_ylabel('y (rows/height)')
ax.text(0.5, 0.5, 'FLOW →', transform=ax.transAxes,
        fontsize=24, fontweight='bold', color='red',
        horizontalalignment='center', verticalalignment='center')

# Plot 3: Array indexing diagram
ax = axes[2]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.text(5, 9, 'Array Indexing Convention', fontsize=14, fontweight='bold',
        horizontalalignment='center')
ax.text(5, 8, f'Python shape: (n_y, n_x) = ({n_y}, {n_x})', fontsize=12,
        horizontalalignment='center', family='monospace')
ax.text(5, 7, f'arr[i, j] where i∈[0,{n_y-1}], j∈[0,{n_x-1}]', fontsize=11,
        horizontalalignment='center', family='monospace')
ax.text(5, 5.5, 'For vector at window (i, j):', fontsize=12, fontweight='bold',
        horizontalalignment='center')
ax.text(5, 4.5, '• i varies along rows (y-direction/vertical)', fontsize=11,
        horizontalalignment='center')
ax.text(5, 3.5, '• j varies along columns (x-direction/horizontal)', fontsize=11,
        horizontalalignment='center')
ax.text(5, 2, f'ux_mat[i,j] = horizontal velocity at (x={n_x} positions, y={n_y} positions)',
        fontsize=10, horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(5, 1, f'uy_mat[i,j] = vertical velocity at same location',
        fontsize=10, horizontalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('/tmp/piv_indexing_debug_setup.png', dpi=150, bbox_inches='tight')
print("=" * 80)
print("Saved visualization to: /tmp/piv_indexing_debug_setup.png")
print("=" * 80)
plt.close()

# Now let's trace through what the C library returns
print()
print("=" * 80)
print("C LIBRARY OUTPUT FORMAT")
print("=" * 80)
print("The C library uses Fortran ordering (column-major):")
print(f"• nWindows = [n_x={n_x}, n_y={n_y}]")
print(f"• Window indexing: iWindowIdx = ii + jj * n_x")
print(f"  where ii ∈ [0, {n_x-1}] (x-index), jj ∈ [0, {n_y-1}] (y-index)")
print()
print("Output arrays:")
print(f"• pk_loc_x: shape (n_peaks, n_x, n_y) = (1, {n_x}, {n_y})")
print(f"• pk_loc_y: shape (n_peaks, n_x, n_y) = (1, {n_x}, {n_y})")
print()
print("Question: What do pk_loc_x and pk_loc_y represent?")
print("• Displacement in x-direction (horizontal)?")
print("• Displacement in y-direction (vertical)?")
print("• Or are they indexed displacements along first/second dimensions?")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Run PIV on this image pair")
print("2. Visualize the raw output from C library (before transpose)")
print("3. Visualize after transpose")
print("4. Compare with expected flow pattern")
print("=" * 80)
