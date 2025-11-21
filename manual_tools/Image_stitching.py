import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivtools_gui"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivtools_core"))

from paths import get_data_paths
from vector_loading import (
    load_coords_from_directory,
    load_vectors_from_directory,
)


class Config:
    def __init__(self, num_images=1000, vector_format="B%05d.mat", piv_chunk_size=100):
        self.num_images = num_images
        self.vector_format = vector_format
        self.piv_chunk_size = piv_chunk_size


def find_non_empty_runs(
    data_dir: Path, vector_format: str = "B%05d.mat", max_runs: int = 20
):
    print(f"[DEBUG] Checking for non-empty runs in: {data_dir}")
    if not data_dir.exists():
        print("[DEBUG] Data directory does not exist.")
        return []

    fmt = vector_format
    first_file = data_dir / (fmt % 1)
    print(f"[DEBUG] Checking first vector file: {first_file}")
    if not first_file.exists():
        print("[DEBUG] First vector file does not exist.")
        return []

    try:
        import scipy.io
        mat = scipy.io.loadmat(str(first_file), struct_as_record=False, squeeze_me=True)
        print(f"[DEBUG] Loaded .mat file: {first_file}")
        piv_result = mat["piv_result"]
        print(f"[DEBUG] piv_result type: {type(piv_result)}")
        
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            total_runs = piv_result.size
            print(f"[DEBUG] Multiple runs detected: {total_runs}")
            valid_runs = []

            for run_idx in range(min(total_runs, max_runs)):
                pr = piv_result[run_idx]
                ux = np.asarray(pr.ux)
                uy = np.asarray(pr.uy)
                print(f"[DEBUG] Run {run_idx+1}: ux.size={ux.size}, uy.size={uy.size}")
                if ux.size > 0 and uy.size > 0:
                    valid_runs.append(run_idx + 1)

            print(f"[DEBUG] Valid runs: {valid_runs}")
            return valid_runs
        else:
            print("[DEBUG] Single run file detected.")
            pr = piv_result
            ux = np.asarray(pr.ux)
            uy = np.asarray(pr.uy)
            print(f"[DEBUG] Single run: ux.size={ux.size}, uy.size={uy.size}")
            if ux.size > 0 and uy.size > 0:
                return [1]
            else:
                return []
    except Exception as e:
        print(f"[DEBUG] Exception in find_non_empty_runs: {e}")
        warnings.warn(f"Error checking runs in {first_file}: {e}")
        return []


def plot_scalar_field(ax, field, x, y, title):
    """Plot a scalar field (ux or uy) using contourf."""
    im = ax.contourf(x, y, field, levels=100, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_distance_weights(x, y, x_bounds, y_bounds, blend_width_fraction=0.2):
    """
    Create distance-based weights for blending with smooth falloff.
    Uses cosine taper for smoother transitions to avoid grid artifacts.
    
    Args:
        blend_width_fraction: Fraction of domain to use for blending (0.2 = 20% on each edge)
    """
    x_norm = (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    y_norm = (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
    
    # Use cosine taper instead of sine for smoother falloff
    # This reduces artifacts at blend boundaries
    def smooth_taper(dist_norm, width):
        """Smooth cosine taper from 1 (center) to 0 (edge)"""
        taper = np.ones_like(dist_norm)
        # Left/bottom edge
        mask_low = dist_norm < width
        taper[mask_low] = 0.5 * (1 + np.cos(np.pi * (width - dist_norm[mask_low]) / width))
        # Right/top edge
        mask_high = dist_norm > (1 - width)
        taper[mask_high] = 0.5 * (1 + np.cos(np.pi * (dist_norm[mask_high] - (1 - width)) / width))
        return taper
    
    x_weight = smooth_taper(x_norm, blend_width_fraction)
    y_weight = smooth_taper(y_norm, blend_width_fraction)
    
    # Combine x and y weights
    weights = x_weight * y_weight
    
    return weights


def merge_vector_fields(
    x1, y1, ux1, uy1, mask1, x2, y2, ux2, uy2, mask2, grid_spacing=None
):
    """
    Merge two vector fields with smart overlap handling.
    ARTIFACT-FREE VERSION: Uses cubic interpolation and improved weighting
    to eliminate grid patterns in RMS calculations.
    
    Args:
        x1, y1, ux1, uy1, mask1: Camera 1 coordinates, vectors, and mask (True = masked/invalid)
        x2, y2, ux2, uy2, mask2: Camera 2 coordinates, vectors, and mask (True = masked/invalid)
        grid_spacing: Target grid spacing for merged field
        
    Returns:
        x_merged, y_merged, ux_merged, uy_merged: Merged field
    """
    print("[DEBUG] Starting artifact-free vector field merging...")
    
    # Get full extent of both cameras (no cropping)
    y1_min, y1_max = np.nanmin(y1), np.nanmax(y1)
    y2_min, y2_max = np.nanmin(y2), np.nanmax(y2)
    
    # Find overlapping y-range (for info only)
    y_overlap_min = max(y1_min, y2_min)
    y_overlap_max = min(y1_max, y2_max)
    
    print(f"[DEBUG] Camera 1 y-range: [{y1_min:.2f}, {y1_max:.2f}]")
    print(f"[DEBUG] Camera 2 y-range: [{y2_min:.2f}, {y2_max:.2f}]")
    print(f"[DEBUG] Overlapping y-range: [{y_overlap_min:.2f}, {y_overlap_max:.2f}]")
    
    # Use FULL y-range from both cameras (no cropping)
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)
    
    # Full x-range from both cameras
    x_min = min(np.nanmin(x1), np.nanmin(x2))
    x_max = max(np.nanmax(x1), np.nanmax(x2))
    
    print(f"[DEBUG] Merged bounds (FULL extent): x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    # Auto-determine grid spacing if not provided
    if grid_spacing is None:
        dx1 = np.median(np.diff(np.unique(x1)))
        dy1 = np.median(np.diff(np.unique(y1)))
        dx2 = np.median(np.diff(np.unique(x2)))
        dy2 = np.median(np.diff(np.unique(y2)))
        # Use the finer grid spacing to preserve detail
        grid_spacing = min(dx1, dy1, dx2, dy2)
        print(f"[DEBUG] Auto grid spacing: {grid_spacing:.3f}")
    
    # Create merged grid
    x_merged = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_merged = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X_merged, Y_merged = np.meshgrid(x_merged, y_merged)
    
    print(f"[DEBUG] Merged grid shape: {X_merged.shape}")
    
    # Flatten query points once for all interpolations
    query_points = np.column_stack([X_merged.ravel(), Y_merged.ravel()])
    
    # Initialize merged arrays with NaN (not zeros!)
    ux_merged = np.full(X_merged.size, np.nan)
    uy_merged = np.full(X_merged.size, np.nan)
    weight_sum = np.zeros(X_merged.size)
    
    # Process each camera
    for cam_idx, (x_cam, y_cam, ux_cam, uy_cam, mask_cam) in enumerate(
        [(x1, y1, ux1, uy1, mask1), (x2, y2, ux2, uy2, mask2)]
    ):
        print(f"[DEBUG] Processing camera {cam_idx + 1}...")
        
        # Apply mask: set masked values to NaN
        ux_cam_masked = np.where(mask_cam, np.nan, ux_cam)
        uy_cam_masked = np.where(mask_cam, np.nan, uy_cam)
        
        # Check if we have any valid data
        if np.all(np.isnan(ux_cam_masked)) or np.all(np.isnan(uy_cam_masked)):
            print(f"[DEBUG] No valid data for camera {cam_idx + 1}")
            continue

        # Extract unique x and y coordinates
        x_coords_1d = x_cam[0, :]
        y_coords_1d = y_cam[:, 0]
        
        print(f"[DEBUG] Camera {cam_idx + 1}: grid shape {x_cam.shape}, x range [{x_coords_1d[0]:.2f}, {x_coords_1d[-1]:.2f}], y range [{y_coords_1d[0]:.2f}, {y_coords_1d[-1]:.2f}]")

        # Use cubic interpolation for smoother results (reduces grid artifacts)
        try:
            ux_interp = interpn(
                (y_coords_1d, x_coords_1d),
                ux_cam_masked,
                (Y_merged, X_merged),
                method='cubic',  # Changed from 'linear' to 'cubic'
                bounds_error=False,
                fill_value=np.nan
            )
            
            uy_interp = interpn(
                (y_coords_1d, x_coords_1d),
                uy_cam_masked,
                (Y_merged, X_merged),
                method='cubic',  # Changed from 'linear' to 'cubic'
                bounds_error=False,
                fill_value=np.nan
            )
        except Exception as e:
            print(f"[DEBUG] Cubic interpolation failed for camera {cam_idx + 1}, falling back to linear: {e}")
            # Fallback to linear if cubic fails
            ux_interp = interpn(
                (y_coords_1d, x_coords_1d),
                ux_cam_masked,
                (Y_merged, X_merged),
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            
            uy_interp = interpn(
                (y_coords_1d, x_coords_1d),
                uy_cam_masked,
                (Y_merged, X_merged),
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        
        ux_interp_flat = ux_interp.ravel()
        uy_interp_flat = uy_interp.ravel()
        
        # Create weights based on distance from edges (with improved taper)
        x_bounds = [np.nanmin(x_cam), np.nanmax(x_cam)]
        y_bounds = [np.nanmin(y_cam), np.nanmax(y_cam)]
        weights = create_distance_weights(X_merged, Y_merged, x_bounds, y_bounds)
        
        # Flatten weights and identify valid interpolated data
        weights_flat = weights.ravel()
        valid_interp = ~(np.isnan(ux_interp_flat) | np.isnan(uy_interp_flat))
        
        # Apply weights only where we have valid data
        weights_flat = np.where(valid_interp, weights_flat, 0)
        
        # Accumulate weighted values (proper NaN handling)
        # Use np.nansum behavior: only accumulate valid values
        ux_contribution = np.where(valid_interp, ux_interp_flat * weights_flat, 0)
        uy_contribution = np.where(valid_interp, uy_interp_flat * weights_flat, 0)
        
        # Track first contribution to initialize output arrays
        first_contribution = np.isnan(ux_merged) & valid_interp
        ux_merged = np.where(first_contribution, 0, ux_merged)
        uy_merged = np.where(first_contribution, 0, uy_merged)
        
        ux_merged = np.where(valid_interp, ux_merged + ux_contribution, ux_merged)
        uy_merged = np.where(valid_interp, uy_merged + uy_contribution, uy_merged)
        weight_sum += weights_flat
    
    # Normalize by total weights (avoid division by zero)
    valid_weights = weight_sum > 1e-10
    ux_merged = np.where(valid_weights, ux_merged / weight_sum, np.nan)
    uy_merged = np.where(valid_weights, uy_merged / weight_sum, np.nan)
    
    # Reshape back to 2D
    ux_merged = ux_merged.reshape(X_merged.shape)
    uy_merged = uy_merged.reshape(X_merged.shape)
    
    print(f"[DEBUG] Merged field has {np.sum(valid_weights)} valid points")
    print(f"[DEBUG] Merged field has {np.sum(~valid_weights.reshape(X_merged.shape))} NaN points (unknown regions)")
    
    # MATLAB convention: lowest y-coordinate at bottom (first row)
    # NumPy meshgrid creates arrays where row 0 is at top, so flip vertically
    X_merged = np.flipud(X_merged)
    Y_merged = np.flipud(Y_merged)
    ux_merged = np.flipud(ux_merged)
    uy_merged = np.flipud(uy_merged)
    
    return X_merged, Y_merged, ux_merged, uy_merged


def plot_merged_comparison(
    x1, y1, ux1, uy1, x2, y2, ux2, uy2, x_merged, y_merged, ux_merged, uy_merged
):
    """Plot comparison of individual camera fields and merged result."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Camera 1
    plot_scalar_field(axes[0, 0], ux1, x1, y1, "Camera 1 - ux")
    plot_scalar_field(axes[0, 1], uy1, x1, y1, "Camera 1 - uy")
    
    # Camera 2
    plot_scalar_field(axes[1, 0], ux2, x2, y2, "Camera 2 - ux")
    plot_scalar_field(axes[1, 1], uy2, x2, y2, "Camera 2 - uy")
    
    # Merged
    plot_scalar_field(axes[2, 0], ux_merged, x_merged, y_merged, "Merged - ux")
    plot_scalar_field(axes[2, 1], uy_merged, x_merged, y_merged, "Merged - uy")
    
    plt.tight_layout()
    return fig


def crop_to_rectangular_boundary(X, Y, ux, uy):
    """
    Crop to the largest rectangle where every row and column has at least one valid point.
    Much simpler and faster than finding solid rectangles.
    """
    valid_mask = ~(np.isnan(ux) | np.isnan(uy))
    
    if not np.any(valid_mask):
        return X, Y, ux, uy
    
    # Find rows with any valid data
    valid_rows = np.any(valid_mask, axis=1)
    # Find columns with any valid data
    valid_cols = np.any(valid_mask, axis=0)
    
    # Find first and last valid row/col
    valid_row_indices = np.where(valid_rows)[0]
    valid_col_indices = np.where(valid_cols)[0]
    
    if len(valid_row_indices) == 0 or len(valid_col_indices) == 0:
        return X, Y, ux, uy
    
    top_row = valid_row_indices[0]
    bottom_row = valid_row_indices[-1]
    left_col = valid_col_indices[0]
    right_col = valid_col_indices[-1]
    
    print(f"[DEBUG] Cropping to rows [{top_row}:{bottom_row+1}], cols [{left_col}:{right_col+1}]")
    
    return (
        X[top_row:bottom_row+1, left_col:right_col+1],
        Y[top_row:bottom_row+1, left_col:right_col+1],
        ux[top_row:bottom_row+1, left_col:right_col+1],
        uy[top_row:bottom_row+1, left_col:right_col+1],
    )


def load_and_plot_cameras(
    base_path: str,
    run_number: int = None,
    type_name: str = "instantaneous",
    endpoint: str = "",
    num_images: int = 1000,
    vector_format: str = "B%05d.mat",
    piv_chunk_size: int = 100,
    show_merged: bool = True,
):
    print(f"[DEBUG] load_and_plot_cameras called with show_merged={show_merged}")
    
    base_dir = Path(base_path).expanduser()
    print(f"[DEBUG] Expanded base_dir: {base_dir}")
    if not base_dir.exists():
        print(f"[DEBUG] Base directory does not exist: {base_dir}")
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    config = Config(
        num_images=num_images,
        vector_format=vector_format,
        piv_chunk_size=piv_chunk_size,
    )
    cameras = [1, 2]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    print(f"[DEBUG] Created figure and axes for cameras: {cameras}")
    
    # Store data for merging
    camera_data = {}
    
    for i, cam in enumerate(cameras):
        print(f"\n[DEBUG] Processing Camera {cam}...")
        
        # Get data paths
        paths = get_data_paths(
            base_dir=base_dir,
            num_frame_pairs=num_images - 1,  # Assuming time-resolved
            cam=cam,
            type_name=type_name,
            endpoint=endpoint,
            use_merged=False,
        )
        print(f"[DEBUG] get_data_paths returned: {paths}")
        data_dir = paths["data_dir"]
        print(f"[DEBUG] Camera {cam} data_dir: {data_dir}")
        
        if not data_dir.exists():
            print(f"[DEBUG] Data directory not found for Camera {cam}: {data_dir}")
            axes[0, i].text(
                0.5, 0.5, f"Camera {cam}\nNo data found",
                ha="center", va="center", transform=axes[0, i].transAxes,
            )
            axes[0, i].set_title(f"Camera {cam} - No Data")
            axes[1, i].axis("off")
            camera_data[cam] = None
            continue
        
        # Find valid runs if run_number not specified
        if run_number is None:
            print(f"[DEBUG] Finding valid runs for Camera {cam}...")
            valid_runs = find_non_empty_runs(data_dir, vector_format)
            print(f"[DEBUG] Valid runs for Camera {cam}: {valid_runs}")
            if not valid_runs:
                print(f"  No valid runs found for Camera {cam}")
                axes[0, i].text(
                    0.5, 0.5, f"Camera {cam}\nNo valid runs",
                    ha="center", va="center", transform=axes[0, i].transAxes,
                )
                axes[0, i].set_title(f"Camera {cam} - No Valid Runs")
                axes[1, i].axis("off")
                camera_data[cam] = None
                continue
            selected_run = valid_runs[0]
            print(f"  Auto-selected run: {selected_run} (from {valid_runs})")
        else:
            selected_run = run_number
            print(f"[DEBUG] Using specified run: {selected_run}")
        
        try:
            print(f"[DEBUG] Loading coordinates for Camera {cam}, run {selected_run}...")
            # Load coordinates
            x_list, y_list = load_coords_from_directory(data_dir, runs=[selected_run])
            print(f"[DEBUG] x_list: {type(x_list)}, y_list: {type(y_list)}")
            if not x_list or not y_list:
                print(f"[DEBUG] No coordinates found for run {selected_run}")
                axes[0, i].text(
                    0.5, 0.5, f"Camera {cam}\nNo coordinates",
                    ha="center", va="center", transform=axes[0, i].transAxes,
                )
                axes[0, i].set_title(f"Camera {cam} - No Coordinates")
                axes[1, i].axis("off")
                camera_data[cam] = None
                continue
            
            x_coords, y_coords = x_list[0], y_list[0]
            print(f"[DEBUG] Loaded coordinates shapes: x={x_coords.shape}, y={y_coords.shape}")
            print(f"[DEBUG] Loading vectors for Camera {cam}, run {selected_run}...")
            
            # Load vector data for the first vector file only
            vectors = load_vectors_from_directory(data_dir, config, runs=[selected_run])
            print(f"[DEBUG] Vectors loaded: shape={getattr(vectors, 'shape', None)}")
            first_frame_vectors = vectors[0, 0].compute()
            print(f"[DEBUG] First frame vectors shape: {first_frame_vectors.shape}")
            
            ux = first_frame_vectors[0]
            uy = first_frame_vectors[1]
            mask = first_frame_vectors[2].astype(bool)
            ux_masked = np.where(mask, np.nan, ux)
            uy_masked = np.where(mask, np.nan, uy)
            
            # Store for merging
            camera_data[cam] = {
                "x": x_coords,
                "y": y_coords,
                "ux": ux_masked,
                "uy": uy_masked,
                "mask": mask,  # Store the mask
            }
            
            # Plot scalar fields
            plot_scalar_field(
                axes[0, i], ux_masked, x_coords, y_coords,
                f"Camera {cam} - Run {selected_run} - ux",
            )
            plot_scalar_field(
                axes[1, i], uy_masked, x_coords, y_coords,
                f"Camera {cam} - Run {selected_run} - uy",
            )
            print(f"[DEBUG] Successfully plotted Camera {cam}")
            
        except Exception as e:
            axes[0, i].text(
                0.5, 0.5, f"Camera {cam}\nError: {str(e)[:50]}...",
                ha="center", va="center", transform=axes[0, i].transAxes,
            )
            axes[0, i].set_title(f"Camera {cam} - Error")
            axes[1, i].axis("off")
            camera_data[cam] = None
            print(f"[DEBUG] Error loading Camera {cam}: {e}")
    
    print("[DEBUG] Finished plotting. Showing figure...")
    plt.tight_layout()
    plt.show()
    
    # Create merged field if we have data from both cameras
    if (
        show_merged
        and len(camera_data) == 2
        and all(data is not None for data in camera_data.values())
    ):
        print("[DEBUG] Creating merged vector field...")
        cam1_data = camera_data[1]
        cam2_data = camera_data[2]
        
        try:
            x_merged, y_merged, ux_merged, uy_merged = merge_vector_fields(
                cam1_data["x"], cam1_data["y"], cam1_data["ux"], cam1_data["uy"], cam1_data["mask"],
                cam2_data["x"], cam2_data["y"], cam2_data["ux"], cam2_data["uy"], cam2_data["mask"],
            )
            
            # No cropping - keep full extent with zeros in unknown regions
            print("[DEBUG] Skipping crop - keeping full extent with zero-filled unknown regions")
            
            # Plot comparison
            merged_fig = plot_merged_comparison(
                cam1_data["x"], cam1_data["y"], cam1_data["ux"], cam1_data["uy"],
                cam2_data["x"], cam2_data["y"], cam2_data["ux"], cam2_data["uy"],
                x_merged, y_merged, ux_merged, uy_merged,
            )
            plt.show()
            
            return fig, merged_fig
        except Exception as e:
            print(f"[DEBUG] Error creating merged field: {e}")
            return fig, None
    
    return fig


def main():
    print("[DEBUG] Entered main()")
    print("=== PIV Vector Field Viewer ===")
    print("This script loads and displays vector fields from both cameras side by side.\n")
    
    # --- Paste your values below ---
    base_path = r"D://Processed_Wake_PIV//Grid_Case_A_Run_2//"
    type_name = "instantaneous"
    endpoint = ""
    num_images = 30
    run_number = 4  # None = auto-select first valid run, or specify run number (1-based)
    vector_format = "%05d.mat"
    piv_chunk_size = 30
    show_merged = True
    # ------------------------------
    
    base_dir = Path(base_path).expanduser()
    print(f"[DEBUG] base_dir: {base_dir}")
    if not base_dir.exists():
        print(f"[DEBUG] Directory does not exist: {base_dir}")
        return
    
    print(f"[DEBUG] Loading vector fields from: {base_dir}")
    print(f"[DEBUG] Type: {type_name}, Endpoint: {endpoint or 'None'}")
    print(f"[DEBUG] Run: {run_number or 'Auto-select first valid'}")
    print("-" * 50)
    
    try:
        print("[DEBUG] Calling load_and_plot_cameras...")
        result = load_and_plot_cameras(
            base_path=str(base_dir),
            run_number=run_number,
            type_name=type_name,
            endpoint=endpoint,
            num_images=num_images,
            vector_format=vector_format,
            piv_chunk_size=piv_chunk_size,
            show_merged=show_merged,
        )
        if isinstance(result, tuple):
            fig, merged_fig = result
            print("[DEBUG] Individual and merged plots displayed.")
        else:
            print("[DEBUG] Individual plots displayed.")
    except Exception as e:
        print(f"[DEBUG] Exception in main: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()