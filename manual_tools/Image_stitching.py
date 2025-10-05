import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))


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
    """
    Find which runs have non-empty vector data by checking the first vector file.
    Returns list of 1-based run numbers that contain valid data.
    """
    if not data_dir.exists():
        print("[DEBUG] Data directory does not exist.")
        return []

    # Get first vector file to check run structure
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
        # Check if multiple runs
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
                    valid_runs.append(run_idx + 1)  # Convert to 1-based

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
    """
    Plot a scalar field (ux or uy) using imshow.
    """
    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=[np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)],
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_distance_weights(x, y, x_bounds, y_bounds):
    """
    Create distance-based weights for blending.
    Higher weights near center, lower weights near edges.
    """
    # Normalize coordinates to [0, 1] within bounds
    x_norm = (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    y_norm = (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])

    # Distance from edges (0 at edge, 0.5 at center)
    x_dist = np.minimum(x_norm, 1 - x_norm)
    y_dist = np.minimum(y_norm, 1 - y_norm)

    # Combined distance weight (Hanning-like)
    weights = np.sin(np.pi * x_dist) * np.sin(np.pi * y_dist)
    return weights


def merge_vector_fields(
    x1, y1, ux1, uy1, x2, y2, ux2, uy2, grid_spacing=None, overlap_weight=0.5
):
    """
    Merge two vector fields with smart overlap handling.

    Args:
        x1, y1, ux1, uy1: Camera 1 coordinates and vectors
        x2, y2, ux2, uy2: Camera 2 coordinates and vectors
        grid_spacing: Target grid spacing for merged field
        overlap_weight: Weight for overlap blending

    Returns:
        x_merged, y_merged, ux_merged, uy_merged: Merged field
    """
    print("[DEBUG] Starting vector field merging...")

    # Determine merged coordinate bounds
    x_min = min(np.nanmin(x1), np.nanmin(x2))
    x_max = max(np.nanmax(x1), np.nanmax(x2))
    y_min = min(np.nanmin(y1), np.nanmin(y2))
    y_max = max(np.nanmax(y1), np.nanmax(y2))

    print(
        f"[DEBUG] Merged bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]"
    )

    # Auto-determine grid spacing if not provided
    if grid_spacing is None:
        dx1 = np.median(np.diff(np.unique(x1)))
        dy1 = np.median(np.diff(np.unique(y1)))
        dx2 = np.median(np.diff(np.unique(x2)))
        dy2 = np.median(np.diff(np.unique(y2)))
        grid_spacing = min(dx1, dy1, dx2, dy2)
        print(f"[DEBUG] Auto grid spacing: {grid_spacing:.3f}")

    # Create merged grid
    x_merged = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_merged = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X_merged, Y_merged = np.meshgrid(x_merged, y_merged)

    print(f"[DEBUG] Merged grid shape: {X_merged.shape}")

    # Initialize merged arrays
    ux_merged = np.full_like(X_merged, np.nan)
    uy_merged = np.full_like(X_merged, np.nan)
    weight_sum = np.zeros_like(X_merged)

    # Process each camera
    for cam_idx, (x_cam, y_cam, ux_cam, uy_cam) in enumerate(
        [(x1, y1, ux1, uy1), (x2, y2, ux2, uy2)]
    ):
        print(f"[DEBUG] Processing camera {cam_idx + 1}...")

        # Create interpolation functions for this camera
        from scipy.interpolate import griddata

        # Get valid (non-NaN) points
        valid_mask = ~(np.isnan(ux_cam) | np.isnan(uy_cam))
        if not np.any(valid_mask):
            print(f"[DEBUG] No valid data for camera {cam_idx + 1}")
            continue

        x_valid = x_cam[valid_mask]
        y_valid = y_cam[valid_mask]
        ux_valid = ux_cam[valid_mask]
        uy_valid = uy_cam[valid_mask]

        print(f"[DEBUG] Camera {cam_idx + 1}: {len(x_valid)} valid points")

        # Interpolate to merged grid
        ux_interp = griddata(
            (x_valid, y_valid),
            ux_valid,
            (X_merged, Y_merged),
            method="linear",
            fill_value=np.nan,
        )
        uy_interp = griddata(
            (x_valid, y_valid),
            uy_valid,
            (X_merged, Y_merged),
            method="linear",
            fill_value=np.nan,
        )

        # Create weights based on distance from edges of this camera's domain
        x_bounds = [np.nanmin(x_cam), np.nanmax(x_cam)]
        y_bounds = [np.nanmin(y_cam), np.nanmax(y_cam)]
        weights = create_distance_weights(X_merged, Y_merged, x_bounds, y_bounds)

        # Only apply weights where data is valid
        valid_interp = ~(np.isnan(ux_interp) | np.isnan(uy_interp))
        weights = np.where(valid_interp, weights, 0)

        # Accumulate weighted values
        ux_merged = np.where(
            weight_sum == 0,
            np.where(valid_interp, ux_interp, np.nan),
            np.where(
                valid_interp,
                (ux_merged * weight_sum + ux_interp * weights) / (weight_sum + weights),
                ux_merged,
            ),
        )
        uy_merged = np.where(
            weight_sum == 0,
            np.where(valid_interp, uy_interp, np.nan),
            np.where(
                valid_interp,
                (uy_merged * weight_sum + uy_interp * weights) / (weight_sum + weights),
                uy_merged,
            ),
        )
        weight_sum += weights

    print(f"[DEBUG] Merged field has {np.sum(~np.isnan(ux_merged))} valid points")
    return X_merged, Y_merged, ux_merged, uy_merged


def plot_merged_comparison(
    x1, y1, ux1, uy1, x2, y2, ux2, uy2, x_merged, y_merged, ux_merged, uy_merged
):
    """
    Plot comparison of individual camera fields and merged result.
    """
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
    """
    Load vector fields from both cameras and plot them side by side.
    """
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
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 12)
    )  # 2 rows (ux, uy) x 2 columns (cameras)
    print(f"[DEBUG] Created figure and axes for cameras: {cameras}")
    # Store data for merging
    camera_data = {}

    for i, cam in enumerate(cameras):
        print(f"\n[DEBUG] Processing Camera {cam}...")

        # Get data paths
        paths = get_data_paths(
            base_dir=base_dir,
            num_images=num_images,
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
                0.5,
                0.5,
                f"Camera {cam}\nNo data found",
                ha="center",
                va="center",
                transform=axes[0, i].transAxes,
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
                    0.5,
                    0.5,
                    f"Camera {cam}\nNo valid runs",
                    ha="center",
                    va="center",
                    transform=axes[0, i].transAxes,
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
            print(
                f"[DEBUG] Loading coordinates for Camera {cam}, run {selected_run}..."
            )
            # Load coordinates
            x_list, y_list = load_coords_from_directory(data_dir, runs=[selected_run])
            print(f"[DEBUG] x_list: {type(x_list)}, y_list: {type(y_list)}")
            if not x_list or not y_list:
                print(f"[DEBUG] No coordinates found for run {selected_run}")
                axes[0, i].text(
                    0.5,
                    0.5,
                    f"Camera {cam}\nNo coordinates",
                    ha="center",
                    va="center",
                    transform=axes[0, i].transAxes,
                )
                axes[0, i].set_title(f"Camera {cam} - No Coordinates")
                axes[1, i].axis("off")
                camera_data[cam] = None
                continue

            x_coords, y_coords = x_list[0], y_list[0]
            print(
                f"[DEBUG] Loaded coordinates shapes: x={x_coords.shape}, y={y_coords.shape}"
            )
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
            }

            # Plot scalar fields
            plot_scalar_field(
                axes[0, i],
                ux_masked,
                x_coords,
                y_coords,
                f"Camera {cam} - Run {selected_run} - ux",
            )
            plot_scalar_field(
                axes[1, i],
                uy_masked,
                x_coords,
                y_coords,
                f"Camera {cam} - Run {selected_run} - uy",
            )
            print(f"[DEBUG] Successfully plotted Camera {cam}")
        except Exception as e:
            axes[0, i].text(
                0.5,
                0.5,
                f"Camera {cam}\nError: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=axes[0, i].transAxes,
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
                cam1_data["x"],
                cam1_data["y"],
                cam1_data["ux"],
                cam1_data["uy"],
                cam2_data["x"],
                cam2_data["y"],
                cam2_data["ux"],
                cam2_data["uy"],
            )

            # Crop to solid boundary
            x_merged_c, y_merged_c, ux_merged_c, uy_merged_c = crop_to_solid_boundary(
                x_merged, y_merged, ux_merged, uy_merged
            )

            # Plot comparison
            merged_fig = plot_merged_comparison(
                cam1_data["x"],
                cam1_data["y"],
                cam1_data["ux"],
                cam1_data["uy"],
                cam2_data["x"],
                cam2_data["y"],
                cam2_data["ux"],
                cam2_data["uy"],
                x_merged_c,
                y_merged_c,
                ux_merged_c,
                uy_merged_c,
            )
            plt.show()

            return fig, merged_fig
        except Exception as e:
            print(f"[DEBUG] Error creating merged field: {e}")
            return fig, None

    return fig


def crop_to_solid_boundary(X, Y, ux, uy):
    """
    Crop to the largest solid rectangle with no NaNs anywhere inside.
    Uses a dynamic programming approach to find the largest rectangle of valid data.
    """
    valid_mask = ~(np.isnan(ux) | np.isnan(uy))

    if not np.any(valid_mask):
        return X, Y, ux, uy

    # Convert boolean mask to integer for histogram calculation
    heights = valid_mask.astype(int)

    # For each row, calculate cumulative heights (consecutive 1s going up)
    for i in range(1, heights.shape[0]):
        heights[i] = np.where(valid_mask[i], heights[i - 1] + 1, 0)

    # Find largest rectangle in histogram for each row
    max_area = 0
    best_coords = None

    for row in range(heights.shape[0]):
        area, coords = largest_rectangle_in_histogram(heights[row])
        if area > max_area:
            max_area = area
            # coords is (left_col, right_col, height)
            left_col, right_col, height = coords
            bottom_row = row
            top_row = row - height + 1
            best_coords = (top_row, bottom_row, left_col, right_col)

    if best_coords is None:
        return X, Y, ux, uy

    top_row, bottom_row, left_col, right_col = best_coords

    return (
        X[top_row : bottom_row + 1, left_col : right_col + 1],
        Y[top_row : bottom_row + 1, left_col : right_col + 1],
        ux[top_row : bottom_row + 1, left_col : right_col + 1],
        uy[top_row : bottom_row + 1, left_col : right_col + 1],
    )


def largest_rectangle_in_histogram(heights):
    """
    Find the largest rectangle in a histogram using stack-based algorithm.
    Returns (area, (left_col, right_col, height))
    """
    stack = []
    max_area = 0
    best_coords = (0, 0, 0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            left = 0 if not stack else stack[-1] + 1
            area = height * width
            if area > max_area:
                max_area = area
                best_coords = (left, left + width - 1, height)
        stack.append(i)

    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        left = 0 if not stack else stack[-1] + 1
        area = height * width
        if area > max_area:
            max_area = area
            best_coords = (left, left + width - 1, height)

    return max_area, best_coords


def main():
    print("[DEBUG] Entered main()")
    """
    Interactive main function to get user input and plot vector fields.
    """
    print("=== PIV Vector Field Viewer ===")
    print(
        "This script loads and displays vector fields from both cameras side by side.\n"
    )

    # --- Paste your values below ---
    base_path = "/Users/morgan/Downloads/bailey"
    type_name = "instantaneous"
    endpoint = ""
    num_images = 500
    run_number = (
        3  # None = auto-select first valid run, or specify run number (1-based)
    )
    vector_format = "%05d.mat"
    piv_chunk_size = 100
    show_merged = True  # Add this parameter
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
