"""
Test meshgrid behavior to understand coordinate grid generation.
"""
import numpy as np

# Our window centers
win_ctrs_x = np.array([63.5, 127.5, 191.5, 255.5, 319.5])  # 5 points along width (x)
win_ctrs_y = np.array([63.5, 127.5, 191.5, 255.5, 319.5, 383.5, 447.5])  # 7 points along height (y)

n_x = len(win_ctrs_x)  # 5
n_y = len(win_ctrs_y)  # 7

print("="*80)
print("MESHGRID WITH DEFAULT indexing='xy'")
print("="*80)
print(f"win_ctrs_x: {n_x} points (x-direction/width/columns)")
print(f"win_ctrs_y: {n_y} points (y-direction/height/rows)")
print()

# Default meshgrid
x_grid, y_grid = np.meshgrid(win_ctrs_x, win_ctrs_y)

print(f"x_grid shape: {x_grid.shape}")  # Should be (n_y, n_x) = (7, 5)
print(f"y_grid shape: {y_grid.shape}")  # Should be (n_y, n_x) = (7, 5)
print()

print("x_grid (shows x-coordinates):")
print(x_grid)
print()

print("y_grid (shows y-coordinates):")
print(y_grid)
print()

print("="*80)
print("INTERPRETATION")
print("="*80)
print("For quiver plot: quiver(x_grid, y_grid, ux, uy)")
print("• x_grid[i, j] gives x-coordinate of vector at (row i, col j)")
print("• y_grid[i, j] gives y-coordinate of vector at (row i, col j)")
print("• ux[i, j] should be x-component (horizontal) of velocity at (row i, col j)")
print("• uy[i, j] should be y-component (vertical) of velocity at (row i, col j)")
print()
print(f"So velocity arrays should have shape ({n_y}, {n_x}) = (n_y, n_x)")
print()

print("="*80)
print("WHAT IF C LIBRARY GIVES US (n_x, n_y)?")
print("="*80)
print(f"C library shape: ({n_x}, {n_y})")
print(f"meshgrid shape: ({n_y}, {n_x})")
print("They DON'T match!")
print()
print("Options:")
print("  1. Transpose C library output: (n_x, n_y) → (n_y, n_x) [CURRENT]")
print("  2. Change meshgrid to give (n_x, n_y) [ALTERNATIVE]")
print("  3. Use different indexing in meshgrid")
print()

print("="*80)
print("MESHGRID WITH indexing='ij'")
print("="*80)
x_grid_ij, y_grid_ij = np.meshgrid(win_ctrs_x, win_ctrs_y, indexing='ij')
print(f"x_grid_ij shape: {x_grid_ij.shape}")  # Should be (n_x, n_y) = (5, 7)
print(f"y_grid_ij shape: {y_grid_ij.shape}")  # Should be (n_x, n_y) = (5, 7)
print()
print("x_grid_ij (shows x-coordinates):")
print(x_grid_ij)
print()
print("With indexing='ij', meshgrid gives (n_x, n_y) which MATCHES C library!")
