import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Any

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))  # Add root for pivtools_core

from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory, load_vectors_from_directory


class PIVConfig:
    def __init__(self, num_images=1000, vector_format="B%05d.mat", piv_chunk_size=100):
        self.num_images = num_images
        self.vector_format = vector_format
        self.piv_chunk_size = piv_chunk_size


def find_non_empty_runs(data_dir: Path, vector_format: str = "B%05d.mat", max_runs: int = 20):
    """Return list of 1-based run numbers that contain valid vector data."""
    if not data_dir.exists():
        return []

    fmt = vector_format
    first_file = data_dir / (fmt % 1)
    if not first_file.exists():
        return []

    try:
        import scipy.io
        import numpy as _np

        mat = scipy.io.loadmat(str(first_file), struct_as_record=False, squeeze_me=True)
        piv_result = mat.get("piv_result")
        if piv_result is None:
            return []

        if isinstance(piv_result, _np.ndarray) and piv_result.dtype == object:
            total_runs = piv_result.size
            valid_runs = []
            for run_idx in range(min(total_runs, max_runs)):
                pr = piv_result[run_idx]
                ux = _np.asarray(pr.ux)
                uy = _np.asarray(pr.uy)
                if ux.size > 0 and uy.size > 0:
                    valid_runs.append(run_idx + 1)
            return valid_runs
        else:
            pr = piv_result
            ux = _np.asarray(pr.ux)
            uy = _np.asarray(pr.uy)
            if ux.size > 0 and uy.size > 0:
                return [1]
            return []
    except Exception:
        warnings.warn("Unable to read piv_result from file to detect runs")
        return []


def load_camera_data(base_dir: Path, cam: int, run_number: int | None, config: PIVConfig,
                     type_name: str = "instantaneous", endpoint: str = ""):
    """Load coordinates (x,y) and vector fields (ux,uy) and mask for given camera and run."""
    try:
        paths = get_data_paths(
            base_dir=base_dir,
            num_frame_pairs=config.num_frame_pairs,
            cam=cam,
            type_name=type_name,
            endpoint=endpoint,
            use_merged=False,
        )
        data_dir = paths["data_dir"]
        if not data_dir.exists():
            print(f"[DEBUG] Camera {cam} data_dir does not exist: {data_dir}")
            return None

        if run_number is None:
            valid_runs = find_non_empty_runs(data_dir, config.vector_format)
            if not valid_runs:
                print(f"[DEBUG] No valid runs found for Camera {cam} in {data_dir}")
                return None
            selected_run = valid_runs[0]
        else:
            selected_run = run_number

        x_list, y_list = load_coords_from_directory(data_dir, runs=[selected_run])
        if not x_list or not y_list:
            print(f"[DEBUG] No coordinates found for Camera {cam}, run {selected_run}")
            return None
        x_coords, y_coords = x_list[0], y_list[0]

        vectors = load_vectors_from_directory(data_dir, config, runs=[selected_run])
        first_frame_vectors = vectors[0, 0].compute()
        ux = first_frame_vectors[0]
        uy = first_frame_vectors[1]
        mask = first_frame_vectors[2].astype(bool)
        ux_masked = np.where(mask, np.nan, ux)
        uy_masked = np.where(mask, np.nan, uy)

        return {
            "x": x_coords,
            "y": y_coords,
            "ux": ux_masked,
            "uy": uy_masked,
            "mask": mask,
            "run": selected_run,
        }
    except Exception as e:
        print(f"[DEBUG] Error loading camera {cam}: {e}")
        return None


def plot_n_cameras(
    base_path: str,
    run_number: int | None = None,
    type_name: str = "instantaneous",
    endpoint: str = "",
    num_images: int = 1000,
    vector_format: str = "B%05d.mat",
    piv_chunk_size: int = 100,
    max_cameras: int = 10
):
    """Load and plot n camera fields with automatic merging."""
    base_dir = Path(base_path).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    config = PIVConfig(num_images=num_images, vector_format=vector_format, piv_chunk_size=piv_chunk_size)

    # Load data from all available cameras
    print("Loading camera data...")
    cameras = {}
    for cam_idx in range(1, max_cameras + 1):
        cam_data = load_camera_data(base_dir, cam_idx, run_number, config, type_name, endpoint)
        if cam_data is not None:
            cameras[cam_idx] = cam_data
            print(f"  Loaded camera {cam_idx}")

    if not cameras:
        print("[DEBUG] No data loaded for any camera")
        return None

    print(f"\nTotal cameras loaded: {len(cameras)}")

    # Determine color limits using first available camera for consistency
    first_cam = cameras[min(cameras.keys())]
    print(f"Computing color limits from Camera {min(cameras.keys())}...")
    ux_values = []
    uy_values = []
    ux_valid = first_cam['ux'][~np.isnan(first_cam['ux'])]
    uy_valid = first_cam['uy'][~np.isnan(first_cam['uy'])]
    if len(ux_valid) > 0:
        ux_values.append(ux_valid)
    if len(uy_valid) > 0:
        uy_values.append(uy_valid)

    if ux_values:
        ux_all = np.concatenate(ux_values)
        ux_min, ux_max = np.percentile(ux_all, [1, 99])
    else:
        ux_min, ux_max = -1, 1

    if uy_values:
        uy_all = np.concatenate(uy_values)
        uy_min, uy_max = np.percentile(uy_all, [1, 99])
    else:
        uy_min, uy_max = -1, 1

    print(f"  ux limits: [{ux_min:.3f}, {ux_max:.3f}]")
    print(f"  uy limits: [{uy_min:.3f}, {uy_max:.3f}]")

    # Helper to plot scalar field with consistent colormap
    # For Cartesian PIV data: lowest y at bottom, so we need origin='lower'
    def plot_field(ax, field, x, y, title, vmin, vmax, cmap='RdBu_r'):
        field = np.asarray(field)
        x = np.asarray(x)
        y = np.asarray(y)

        # Create proper 2D coordinate grids
        if x.ndim == 1 and y.ndim == 1:
            xg, yg = np.meshgrid(x, y, indexing='xy')  # Critical: y first
        elif x.ndim == 2 and y.ndim == 2:
            xg, yg = x, y
        else:
            raise ValueError("x and y must both be 1D or both 2D")

        # Reshape field if needed
        expected_shape = (yg.shape[0], xg.shape[1])
        if field.shape != expected_shape:
            field = field.reshape(expected_shape)

        im = ax.contourf(xg, yg, field, levels=100, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X [mm]", fontsize=9)
        ax.set_ylabel("Y [mm]", fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        return im

    # Create unified grid and merge all cameras
    print("\nCreating unified grid and merging cameras...")

    # Compute grid spacing from first camera
    x_first = np.asarray(first_cam['x'])
    y_first = np.asarray(first_cam['y'])
    if x_first.ndim == 1:
        x_first_vec = x_first
        y_first_vec = y_first
    else:
        x_first_vec = x_first[0, :]
        y_first_vec = y_first[:, 0]
    dx = abs(np.median(np.diff(x_first_vec)))
    dy = abs(np.median(np.diff(y_first_vec)))

    # Combined bounds from all cameras
    x_min = min(cam_data['x'].min() for cam_data in cameras.values())
    x_max = max(cam_data['x'].max() for cam_data in cameras.values())
    y_min = min(cam_data['y'].min() for cam_data in cameras.values())
    y_max = max(cam_data['y'].max() for cam_data in cameras.values())

    # Create extended grid based on first camera spacing
    nx = int(np.round((x_max - x_min) / dx)) + 1
    ny = int(np.round((y_max - y_min) / dy)) + 1
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)  # Ascending for RegularGridInterpolator
    xg, yg = np.meshgrid(x_grid, y_grid, indexing='xy')

    print(f"  Unified grid size: {nx} x {ny}")
    print(f"  X range: [{x_min:.2f}, {x_max:.2f}] mm")
    print(f"  Y range: [{y_min:.2f}, {y_max:.2f}] mm")

    # Interpolate all cameras to the unified grid
    from scipy.interpolate import RegularGridInterpolator

    points = np.stack([yg.ravel(), xg.ravel()], axis=-1)
    camera_interp = {}

    for cam_idx, cam_data in cameras.items():
        print(f"  Interpolating camera {cam_idx}...")

        # Get camera coordinates and data
        x_cam = np.asarray(cam_data['x'])
        y_cam = np.asarray(cam_data['y'])
        ux_cam = np.asarray(cam_data['ux'])
        uy_cam = np.asarray(cam_data['uy'])
        mask_cam = np.asarray(cam_data['mask'])

        # Extract vectors for 1D coords
        if x_cam.ndim == 1:
            x_vec, y_vec = x_cam, y_cam
        else:
            x_vec = x_cam[0, :]
            y_vec = y_cam[:, 0]

        # Reshape data if needed
        if ux_cam.ndim == 1:
            ny_cam, nx_cam = len(y_vec), len(x_vec)
            ux_cam = ux_cam.reshape(ny_cam, nx_cam)
            uy_cam = uy_cam.reshape(ny_cam, nx_cam)
            mask_cam = mask_cam.reshape(ny_cam, nx_cam)

        # Ensure y_vec is ascending for RegularGridInterpolator
        if y_vec[1] < y_vec[0]:
            y_vec = y_vec[::-1]
            ux_cam = np.flipud(ux_cam)
            uy_cam = np.flipud(uy_cam)
            mask_cam = np.flipud(mask_cam)

        # Create interpolators (replace NaN with 0 for interpolation)
        valid_ux = np.where(np.isnan(ux_cam), 0, ux_cam)
        valid_uy = np.where(np.isnan(uy_cam), 0, uy_cam)
        interp_ux = RegularGridInterpolator((y_vec, x_vec), valid_ux, method='nearest',
                                           bounds_error=False, fill_value=np.nan)
        interp_uy = RegularGridInterpolator((y_vec, x_vec), valid_uy, method='nearest',
                                           bounds_error=False, fill_value=np.nan)
        interp_mask = RegularGridInterpolator((y_vec, x_vec), mask_cam.astype(float), method='nearest',
                                             bounds_error=False, fill_value=1.0)

        # Interpolate to unified grid
        ux_interp = interp_ux(points).reshape(yg.shape)
        uy_interp = interp_uy(points).reshape(yg.shape)
        mask_interp = interp_mask(points).reshape(yg.shape) > 0.5

        # Store interpolated data and valid region
        camera_interp[cam_idx] = {
            'ux': ux_interp,
            'uy': uy_interp,
            'mask': mask_interp,
            'valid': ~np.isnan(ux_interp) & ~mask_interp,
            'x_center': np.mean(x_cam),
            'y_center': np.mean(y_cam)
        }

    # Determine stacking direction (horizontal or vertical)
    cam_centers = [(camera_interp[idx]['x_center'], camera_interp[idx]['y_center'])
                   for idx in sorted(cameras.keys())]
    x_spread = max(c[0] for c in cam_centers) - min(c[0] for c in cam_centers)
    y_spread = max(c[1] for c in cam_centers) - min(c[1] for c in cam_centers)

    if x_spread >= y_spread:
        stack_direction = 'horizontal'
        print(f"  Detected horizontal stacking (x_spread={x_spread:.2f} mm)")
    else:
        stack_direction = 'vertical'
        print(f"  Detected vertical stacking (y_spread={y_spread:.2f} mm)")

    # Create weight maps for each camera using distance-based Hanning blend
    print("  Computing blend weights...")
    camera_weights = {}

    for cam_idx in cameras.keys():
        # Initialize weight to 1 where this camera is valid
        weight = np.where(camera_interp[cam_idx]['valid'], 1.0, 0.0)

        # For overlap regions, use distance-based weighting
        if stack_direction == 'horizontal':
            # Weight based on distance from camera center in x-direction
            cam_x_center = camera_interp[cam_idx]['x_center']
            for other_idx in cameras.keys():
                if other_idx == cam_idx:
                    continue

                # Find overlap region
                valid_this = camera_interp[cam_idx]['valid']
                valid_other = camera_interp[other_idx]['valid']
                overlap = valid_this & valid_other

                if np.any(overlap):
                    # Get x-coordinates of overlap
                    x_overlap = xg[overlap]
                    x_min_overlap = x_overlap.min()
                    x_max_overlap = x_overlap.max()

                    # Determine which camera is left/right
                    other_x_center = camera_interp[other_idx]['x_center']
                    if cam_x_center < other_x_center:
                        # This camera is on the left
                        # Weight: high on left, low on right
                        x_norm = (x_overlap - x_min_overlap) / (x_max_overlap - x_min_overlap)
                        overlap_weight = 0.5 * (1 + np.cos(np.pi * x_norm))
                    else:
                        # This camera is on the right
                        # Weight: low on left, high on right
                        x_norm = (x_overlap - x_min_overlap) / (x_max_overlap - x_min_overlap)
                        overlap_weight = 0.5 * (1 - np.cos(np.pi * x_norm))

                    weight[overlap] = overlap_weight
        else:  # vertical
            # Weight based on distance from camera center in y-direction
            cam_y_center = camera_interp[cam_idx]['y_center']
            for other_idx in cameras.keys():
                if other_idx == cam_idx:
                    continue

                # Find overlap region
                valid_this = camera_interp[cam_idx]['valid']
                valid_other = camera_interp[other_idx]['valid']
                overlap = valid_this & valid_other

                if np.any(overlap):
                    # Get y-coordinates of overlap
                    y_overlap = yg[overlap]
                    y_min_overlap = y_overlap.min()
                    y_max_overlap = y_overlap.max()

                    # Determine which camera is top/bottom
                    other_y_center = camera_interp[other_idx]['y_center']
                    if cam_y_center < other_y_center:
                        # This camera is on the bottom
                        # Weight: high on bottom, low on top
                        y_norm = (y_overlap - y_min_overlap) / (y_max_overlap - y_min_overlap)
                        overlap_weight = 0.5 * (1 + np.cos(np.pi * y_norm))
                    else:
                        # This camera is on the top
                        # Weight: low on bottom, high on top
                        y_norm = (y_overlap - y_min_overlap) / (y_max_overlap - y_min_overlap)
                        overlap_weight = 0.5 * (1 - np.cos(np.pi * y_norm))

                    weight[overlap] = overlap_weight

        camera_weights[cam_idx] = weight

    # Normalize weights so they sum to 1 at each point
    total_weight = np.zeros_like(xg)
    for cam_idx in cameras.keys():
        total_weight += camera_weights[cam_idx]

    for cam_idx in cameras.keys():
        # Avoid division by zero
        valid_total = total_weight > 0
        camera_weights[cam_idx] = np.where(valid_total,
                                          camera_weights[cam_idx] / total_weight,
                                          0)

    # Create merged fields by weighted sum
    print("  Blending cameras...")
    ux_merged = np.zeros_like(xg)
    uy_merged = np.zeros_like(yg)

    for cam_idx in cameras.keys():
        ux_merged += camera_weights[cam_idx] * np.nan_to_num(camera_interp[cam_idx]['ux'], nan=0.0)
        uy_merged += camera_weights[cam_idx] * np.nan_to_num(camera_interp[cam_idx]['uy'], nan=0.0)

    # Set to NaN where no camera has valid data
    no_data = total_weight == 0
    ux_merged[no_data] = np.nan
    uy_merged[no_data] = np.nan

    # Plot the merged continuous grid
    print("\nCreating plots...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # UX merged
    im_ux = axes1[0].contourf(xg, yg, ux_merged, levels=100, cmap='RdBu_r',
                              vmin=ux_min, vmax=ux_max, origin='lower')
    axes1[0].set_title(f"Merged UX ({len(cameras)} cameras)", fontsize=12)
    axes1[0].set_xlabel("X [mm]", fontsize=10)
    axes1[0].set_ylabel("Y [mm]", fontsize=10)
    axes1[0].set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im_ux, ax=axes1[0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # UY merged
    im_uy = axes1[1].contourf(xg, yg, uy_merged, levels=100, cmap='RdBu_r',
                              vmin=uy_min, vmax=uy_max, origin='lower')
    axes1[1].set_title(f"Merged UY ({len(cameras)} cameras)", fontsize=12)
    axes1[1].set_xlabel("X [mm]", fontsize=10)
    axes1[1].set_ylabel("Y [mm]", fontsize=10)
    axes1[1].set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im_uy, ax=axes1[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    # Figure 2: Camera weights (show up to 6 cameras)
    num_cameras = len(cameras)
    if num_cameras <= 3:
        ncols = num_cameras
        nrows = 1
    elif num_cameras <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 3
        nrows = (num_cameras + 2) // 3

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten() if isinstance(axes2, np.ndarray) else [axes2]

    for idx, (cam_idx, weight) in enumerate(camera_weights.items()):
        if idx < len(axes2):
            ax = axes2[idx]
            im_weight = ax.contourf(xg, yg, weight, levels=50, cmap='viridis',
                                   origin='lower', alpha=0.8, vmin=0, vmax=1)
            ax.set_title(f"Camera {cam_idx} Weight", fontsize=12)
            ax.set_xlabel("X [mm]", fontsize=10)
            ax.set_ylabel("Y [mm]", fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            cbar = plt.colorbar(im_weight, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Weight', fontsize=10)

    # Hide unused subplots
    for idx in range(len(camera_weights), len(axes2)):
        axes2[idx].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    base_path = r"/Users/morgan/Downloads"
    run_number = 4

    print("="*60)
    print("PIV N-Camera Merge Tool")
    print("="*60)

    plot_n_cameras(
        base_path=base_path,
        run_number=run_number,
        num_images=30,
        piv_chunk_size=1,
        vector_format="%05d.mat",
        max_cameras=10
    )


if __name__ == "__main__":
    main()