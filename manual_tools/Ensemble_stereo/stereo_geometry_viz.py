"""
Stereo PIV Geometry Visualization
=================================
Two views:
1. Top-down (XZ plane): Camera geometry and ghost formation (matplotlib)
2. 3D voxel view: Interactive Plotly visualization of the illuminated volume

Coordinate System:
- X: Horizontal (flow direction, left-to-right in image)
- Y: Vertical (up-down in image)
- Z: Depth (out-of-plane, laser sheet thickness)

Volume: 128 x 128 x 32 pixels (laser sheet configuration)
Cameras are at ±45° in the XZ plane (Y=0), looking toward the measurement volume.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from scipy.ndimage import map_coordinates, gaussian_filter
from gaussian_fit_3d import fit_gaussian_3d, get_ellipsoid_surface, GaussianFitResult

# =============================================================================
# PARAMETERS
# =============================================================================
IMAGE_SIZE = 128          # pixels (X and Y dimensions)
VOLUME_DEPTH = 32         # pixels (Z direction - laser sheet thickness)
SCALE = 10                # pixels per mm
NUM_PARTICLES = 32
PARTICLE_SEED = 42

# Camera parameters
WORKING_DISTANCE = 500    # mm from volume center to cameras
CAMERA_ANGLE = 45         # degrees from Z-axis (perpendicular to laser sheet)

# Displacement (in pixels) - applied to create frame B
DISPLACEMENT = np.array([5.0, 4.0, 3.0])  # dx, dy, dz

# Voxel parameters - UNIFORM 0.1mm in all directions
VOXEL_SIZE_MM = 0.1       # mm - uniform in all directions
VOXEL_SIZE_PX = VOXEL_SIZE_MM * SCALE  # = 1 pixel

# Particle rendering
PARTICLE_DIAMETER = 3     # pixels (in image space)
FOCAL_LENGTH = 100        # mm

# Derived
VOLUME_SIZE_MM = np.array([IMAGE_SIZE, IMAGE_SIZE, VOLUME_DEPTH]) / SCALE
VOLUME_SIZE_PX = np.array([IMAGE_SIZE, IMAGE_SIZE, VOLUME_DEPTH])

# Number of voxels in each dimension (uniform 0.5mm voxels)
N_VOXELS_X = int(np.ceil(VOLUME_SIZE_MM[0] / VOXEL_SIZE_MM))
N_VOXELS_Y = int(np.ceil(VOLUME_SIZE_MM[1] / VOXEL_SIZE_MM))
N_VOXELS_Z = int(np.ceil(VOLUME_SIZE_MM[2] / VOXEL_SIZE_MM))
N_VOXELS = np.array([N_VOXELS_X, N_VOXELS_Y, N_VOXELS_Z])


# =============================================================================
# CAMERA MODEL
# =============================================================================
class StereoCamera:
    """Pinhole camera model for stereo PIV at ±angle from Z-axis."""

    def __init__(self, angle_deg, working_distance_mm, name="Camera",
                 focal_length_mm=FOCAL_LENGTH, image_size_px=IMAGE_SIZE,
                 scale_px_per_mm=SCALE):
        self.name = name
        self.angle_rad = np.deg2rad(angle_deg)
        self.angle_deg = angle_deg
        self.d = working_distance_mm
        self.f = focal_length_mm
        self.image_size = image_size_px
        self.scale = scale_px_per_mm

        # Camera position: at angle from Z-axis in the XZ plane (Y=0)
        self.position = np.array([
            self.d * np.sin(self.angle_rad),  # X
            0.0,                               # Y (cameras in horizontal plane)
            self.d * np.cos(self.angle_rad)   # Z
        ])

        # Viewing direction: from camera toward origin
        self.view_dir = -self.position / np.linalg.norm(self.position)

        # Camera coordinate system
        self.z_cam = self.view_dir
        self.y_cam = np.array([0, 1, 0])
        self.x_cam = np.cross(self.y_cam, self.z_cam)
        self.x_cam = self.x_cam / np.linalg.norm(self.x_cam)

        # Rotation matrix: world to camera coordinates
        self.R = np.array([self.x_cam, self.y_cam, self.z_cam])

    def get_ray_direction(self, point_mm):
        """Get unit vector from camera to a 3D point."""
        ray = point_mm - self.position
        return ray / np.linalg.norm(ray)

    def world_to_camera(self, points_world_mm):
        """Transform points from world coordinates to camera coordinates."""
        translated = points_world_mm - self.position
        return np.dot(translated, self.R.T)

    def project(self, points_world_mm):
        """
        Project 3D world points to 2D image coordinates.

        Parameters
        ----------
        points_world_mm : ndarray, shape (N, 3) or (3,)
            3D points in world coordinates (mm)

        Returns
        -------
        pixels : ndarray, shape (N, 2) or (2,)
            2D pixel coordinates (u, v)
        valid : ndarray, shape (N,) or bool
            True if point is in front of camera
        """
        points_world_mm = np.atleast_2d(points_world_mm)

        # Transform to camera coordinates
        p_cam = self.world_to_camera(points_world_mm)

        # Check if points are in front of camera
        valid = p_cam[:, 2] > 0

        with np.errstate(divide='ignore', invalid='ignore'):
            # Perspective projection scaled so at z=d we get 'scale' px/mm
            u = p_cam[:, 0] * (self.scale * self.d / p_cam[:, 2]) + self.image_size / 2
            v = p_cam[:, 1] * (self.scale * self.d / p_cam[:, 2]) + self.image_size / 2

        pixels = np.column_stack([u, v])

        if len(pixels) == 1:
            return pixels[0], valid[0]
        return pixels, valid


# =============================================================================
# PARTICLE GENERATION
# =============================================================================
def generate_particles(num_particles, volume_size_px, scale, seed=42):
    """Generate random particles centered at origin (in mm)."""
    np.random.seed(seed)

    positions_px = np.random.rand(num_particles, 3)
    positions_px[:, 0] = positions_px[:, 0] * volume_size_px[0] - volume_size_px[0] / 2
    positions_px[:, 1] = positions_px[:, 1] * volume_size_px[1] - volume_size_px[1] / 2
    positions_px[:, 2] = positions_px[:, 2] * volume_size_px[2] - volume_size_px[2] / 2

    return positions_px / scale, positions_px


def displace_particles(positions_px, displacement_px, scale):
    """Apply displacement to particles."""
    new_positions_px = positions_px + displacement_px
    return new_positions_px / scale, new_positions_px


# =============================================================================
# GHOST PARTICLE COMPUTATION
# =============================================================================
def find_line_intersections(cam1, cam2, particles_mm, tolerance_mm=0.5):
    """Find where lines of sight from two cameras intersect."""
    n = len(particles_mm)
    ghost_positions = []
    ghost_info = []
    real_matches = set()

    for i in range(n):
        for j in range(n):
            p1 = cam1.position
            d1 = cam1.get_ray_direction(particles_mm[i])
            p2 = cam2.position
            d2 = cam2.get_ray_direction(particles_mm[j])

            cross = np.cross(d1, d2)
            cross_norm = np.linalg.norm(cross)

            if cross_norm < 1e-10:
                continue

            w0 = p1 - p2
            a = np.dot(d1, d1)
            b = np.dot(d1, d2)
            c = np.dot(d2, d2)
            d = np.dot(d1, w0)
            e = np.dot(d2, w0)

            denom = a * c - b * b
            if abs(denom) < 1e-10:
                continue

            t = (b * e - c * d) / denom
            s = (a * e - b * d) / denom

            point1 = p1 + t * d1
            point2 = p2 + s * d2
            midpoint = (point1 + point2) / 2
            distance = np.linalg.norm(point1 - point2)

            if t > 0 and s > 0 and distance < tolerance_mm:
                vol_half = VOLUME_SIZE_MM / 2 * 1.5

                if (abs(midpoint[0]) < vol_half[0] and
                    abs(midpoint[1]) < vol_half[1] and
                    abs(midpoint[2]) < vol_half[2]):

                    if i == j:
                        real_matches.add(i)
                    else:
                        ghost_positions.append(midpoint)
                        ghost_info.append((i, j))

    return list(real_matches), np.array(ghost_positions) if ghost_positions else np.array([]).reshape(0, 3), ghost_info


# =============================================================================
# IMAGE RENDERING (FOR MLOS)
# =============================================================================
def render_image(camera, particle_positions_mm, image_size=IMAGE_SIZE,
                 particle_diameter=PARTICLE_DIAMETER):
    """
    Render synthetic camera image from 3D particle positions.

    Parameters
    ----------
    camera : StereoCamera
        Camera model with project() method
    particle_positions_mm : ndarray, shape (N, 3)
        Particle positions in mm (world coordinates, centered at origin)
    image_size : int
        Image size in pixels
    particle_diameter : float
        Particle diameter in pixels (image space)

    Returns
    -------
    image : ndarray, shape (image_size, image_size)
        Rendered image with Gaussian particle blobs
    """
    image = np.zeros((image_size, image_size), dtype=np.float64)
    sigma = particle_diameter / (2 * 2.355)  # FWHM to sigma

    for pos_mm in particle_positions_mm:
        pixel_pos, valid = camera.project(pos_mm)

        if not valid:
            continue

        u, v = pixel_pos

        # Skip if outside image bounds
        margin = 3 * particle_diameter
        if u < -margin or u >= image_size + margin:
            continue
        if v < -margin or v >= image_size + margin:
            continue

        # Render Gaussian blob
        u_int, v_int = int(round(u)), int(round(v))
        half_size = int(3 * particle_diameter)

        for di in range(-half_size, half_size + 1):
            for dj in range(-half_size, half_size + 1):
                ii, jj = v_int + di, u_int + dj
                if 0 <= ii < image_size and 0 <= jj < image_size:
                    dist_sq = (jj - u)**2 + (ii - v)**2
                    image[ii, jj] += np.exp(-dist_sq / (2 * sigma**2))

    return image


# =============================================================================
# MLOS RECONSTRUCTION
# =============================================================================
def create_voxel_grid(volume_size_px):
    """
    Create coordinate arrays for the voxel grid (centered at origin).

    Returns
    -------
    x, y, z : arrays
        Coordinate arrays centered so (0,0,0) is volume center
    """
    nx, ny, nz = volume_size_px
    x = np.arange(nx) - nx / 2 + 0.5
    y = np.arange(ny) - ny / 2 + 0.5
    z = np.arange(nz) - nz / 2 + 0.5
    return x, y, z


def precompute_projections(cameras, voxel_coords_px, scale):
    """
    Precompute where each voxel projects to on each camera.

    Parameters
    ----------
    cameras : list of StereoCamera
    voxel_coords_px : tuple of arrays (x, y, z)
    scale : float
        Pixels per mm

    Returns
    -------
    projection_maps : list of tuples
        For each camera: (u_coords, v_coords) arrays of shape (nx, ny, nz)
    """
    x, y, z = voxel_coords_px
    nx, ny, nz = len(x), len(y), len(z)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten for batch projection
    points_px = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    points_mm = points_px / scale

    projection_maps = []

    for camera in cameras:
        pixels, valid = camera.project(points_mm)
        u = pixels[:, 0].reshape(nx, ny, nz)
        v = pixels[:, 1].reshape(nx, ny, nz)
        projection_maps.append((u, v))

    return projection_maps


def mlos_reconstruct(images, projection_maps):
    """
    Perform MLOS (Multiplicative Line-of-Sight) reconstruction.

    For each voxel:
    1. Project to each camera image
    2. Sample the intensity at that pixel location
    3. Multiply intensities from all cameras

    Real particles: High intensity in ALL cameras → high product
    Empty space: Low in at least one camera → low product
    Ghosts: Can have moderate intensity where lines cross

    Parameters
    ----------
    images : list of ndarray
        Camera images
    projection_maps : list of tuples
        Precomputed projection coordinates

    Returns
    -------
    volume : ndarray
        3D reconstructed intensity volume
    """
    u0, v0 = projection_maps[0]
    volume_shape = u0.shape

    # Initialize with ones (for multiplication)
    volume = np.ones(volume_shape, dtype=np.float64)

    for image, (u_map, v_map) in zip(images, projection_maps):
        # Sample image at projected coordinates (bilinear interpolation)
        # map_coordinates expects (row, col) = (v, u)
        coords = np.array([v_map.ravel(), u_map.ravel()])
        sampled = map_coordinates(image, coords, order=1, mode='constant', cval=0)
        sampled = sampled.reshape(volume_shape)

        # Multiply into volume
        volume *= sampled

    return volume


def find_peaks_in_volume(volume, threshold_fraction=0.1):
    """
    Find local maxima (particle candidates) in reconstructed volume.

    Returns
    -------
    coords_centered : ndarray, shape (n_peaks, 3)
        Peak coordinates centered at volume origin
    intensities : ndarray
        Intensity values at peaks
    """
    from scipy.ndimage import maximum_filter

    threshold = volume.max() * threshold_fraction

    # Find local maxima
    local_max = maximum_filter(volume, size=3)
    peaks = (volume == local_max) & (volume > threshold)

    # Get coordinates
    coords = np.array(np.where(peaks)).T  # (n_peaks, 3)

    # Convert to centered coordinates
    nx, ny, nz = volume.shape
    coords_centered = coords.astype(float)
    coords_centered[:, 0] -= nx / 2
    coords_centered[:, 1] -= ny / 2
    coords_centered[:, 2] -= nz / 2

    intensities = volume[peaks]

    return coords_centered, intensities


# =============================================================================
# 3D CROSS-CORRELATION
# =============================================================================
def correlate_3d(vol_a, vol_b, normalize=True):
    """
    3D FFT-based cross-correlation.

    Computes the cross-correlation of two 3D volumes using FFT:
        Corr = IFFT( FFT(A) * conj(FFT(B)) )

    The result is fftshift'd so the zero-lag (no displacement) is at the center.

    Parameters
    ----------
    vol_a : ndarray, shape (nx, ny, nz)
        First volume (reference, time t)
    vol_b : ndarray, shape (nx, ny, nz)
        Second volume (displaced, time t+dt)
    normalize : bool
        If True, subtract mean to remove DC bias

    Returns
    -------
    corr : ndarray, shape (nx, ny, nz)
        Cross-correlation volume, peak indicates displacement
    """
    # Normalize by removing mean (removes DC bias)
    if normalize:
        vol_a = vol_a - np.mean(vol_a)
        vol_b = vol_b - np.mean(vol_b)

    # FFT of both volumes
    F_A = np.fft.fftn(vol_a)
    F_B = np.fft.fftn(vol_b)

    # Cross-correlation in frequency domain: F(A) * conj(F(B))
    corr_fft = F_A * np.conj(F_B)

    # Inverse FFT to get correlation in spatial domain
    corr = np.fft.ifftn(corr_fft)

    # Shift so zero-lag is at center
    corr = np.fft.fftshift(np.real(corr))

    return corr


def find_displacement_3d(corr_volume, subpixel=True):
    """
    Find displacement from 3D correlation volume.

    The cross-correlation of vol_a and vol_b gives a peak at the lag where
    they best align. If vol_b is displaced by +d from vol_a, the correlation
    peak appears at -d (correlation convention). We negate to get the
    physical displacement (PIV convention).

    Parameters
    ----------
    corr_volume : ndarray, shape (nx, ny, nz)
        Cross-correlation volume (fftshift'd, zero-lag at center)
    subpixel : bool
        If True, use parabolic fit for sub-voxel accuracy

    Returns
    -------
    displacement : ndarray, shape (3,)
        Displacement vector (dx, dy, dz) in pixels/voxels
    peak_value : float
        Correlation value at peak
    """
    nx, ny, nz = corr_volume.shape
    center = np.array([nx // 2, ny // 2, nz // 2])

    # Find integer peak location
    peak_idx = np.argmax(corr_volume)
    peak_loc = np.array(np.unravel_index(peak_idx, corr_volume.shape))
    peak_value = corr_volume[tuple(peak_loc)]

    # Integer displacement (relative to center)
    # Negate to convert from correlation-lag to physical-displacement
    displacement = -(peak_loc - center)

    if subpixel:
        # Sub-voxel refinement using 3-point parabolic fit in each dimension
        x, y, z = peak_loc

        # Ensure we're not at edges
        if 1 <= x < nx-1 and 1 <= y < ny-1 and 1 <= z < nz-1:
            # X direction (note: sign flip in parabolic fit is handled by negation above)
            dx_sub = _parabolic_peak(
                corr_volume[x-1, y, z],
                corr_volume[x, y, z],
                corr_volume[x+1, y, z]
            )
            # Y direction
            dy_sub = _parabolic_peak(
                corr_volume[x, y-1, z],
                corr_volume[x, y, z],
                corr_volume[x, y+1, z]
            )
            # Z direction
            dz_sub = _parabolic_peak(
                corr_volume[x, y, z-1],
                corr_volume[x, y, z],
                corr_volume[x, y, z+1]
            )

            displacement = displacement.astype(float)
            # Sub-pixel corrections also need to be negated
            displacement[0] -= dx_sub
            displacement[1] -= dy_sub
            displacement[2] -= dz_sub

    return displacement.astype(float), peak_value


def _parabolic_peak(y_minus, y_center, y_plus):
    """
    Sub-pixel peak refinement using 3-point parabolic fit.

    Given three consecutive correlation values, finds the sub-pixel
    offset of the peak from the center point.

    Returns offset in range [-0.5, 0.5]
    """
    denom = 2 * (2 * y_center - y_minus - y_plus)
    if abs(denom) < 1e-10:
        return 0.0
    return (y_minus - y_plus) / denom


# =============================================================================
# HELPER: DRAW VOXEL GRID (MATPLOTLIB)
# =============================================================================
def draw_voxel_grid(ax, extent_min, extent_max, n_divisions, axis_pair='xy',
                    color='gray', linestyle=':', linewidth=0.5, alpha=0.5):
    """Draw voxel grid lines on a matplotlib axis."""
    if axis_pair == 'xz':
        x_lines = np.linspace(extent_min[0], extent_max[0], n_divisions[0] + 1)
        z_lines = np.linspace(extent_min[1], extent_max[1], n_divisions[1] + 1)

        for x in x_lines:
            ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        for z in z_lines:
            ax.axhline(z, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)


# =============================================================================
# VISUALIZATION 1: TOP-DOWN VIEW (MATPLOTLIB)
# =============================================================================
def make_top_down_figure(cameras, particles_mm, ghost_positions):
    """
    TOP-DOWN VIEW (XZ Plane) - Looking down Y-axis
    Shows camera positions, angles, sight lines, and ghost formation.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    half = VOLUME_SIZE_MM / 2
    cam_colors = ['red', 'blue']

    # Volume rectangle (laser sheet cross-section)
    vol_rect = Rectangle((-half[0], -half[2]), 2*half[0], 2*half[2],
                        facecolor='lightgreen', edgecolor='green',
                        linewidth=3, alpha=0.3, label='Laser Sheet Volume')
    ax.add_patch(vol_rect)

    # Voxel grid (uniform 0.5mm)
    draw_voxel_grid(ax,
                    extent_min=(-half[0], -half[2]),
                    extent_max=(half[0], half[2]),
                    n_divisions=(N_VOXELS_X, N_VOXELS_Z),
                    axis_pair='xz',
                    color='darkgreen', linestyle=':', linewidth=0.3, alpha=0.4)

    # Sight lines from cameras through particles
    for i, cam in enumerate(cameras):
        for p in particles_mm:
            direction = np.array([p[0] - cam.position[0], p[2] - cam.position[2]])
            direction = direction / np.linalg.norm(direction)
            end_point = np.array([p[0], p[2]]) + direction * 40
            ax.plot([cam.position[0], end_point[0]],
                   [cam.position[2], end_point[1]],
                   color=cam_colors[i], linewidth=0.5, alpha=0.3)

    # Camera positions
    for i, cam in enumerate(cameras):
        ax.plot(cam.position[0], cam.position[2], 's',
               color=cam_colors[i], markersize=18,
               markeredgecolor='black', markeredgewidth=2,
               label=cam.name, zorder=10)
        arrow_len = 35
        ax.annotate('', xy=(cam.position[0] + cam.view_dir[0]*arrow_len,
                            cam.position[2] + cam.view_dir[2]*arrow_len),
                   xytext=(cam.position[0], cam.position[2]),
                   arrowprops=dict(arrowstyle='->', color=cam_colors[i], lw=3))

    # Real particles
    ax.scatter(particles_mm[:, 0], particles_mm[:, 2],
              s=100, c='lime', edgecolors='darkgreen', linewidths=2,
              label=f'Real Particles (N={len(particles_mm)})', zorder=5)

    # Ghost particles
    if len(ghost_positions) > 0:
        ax.scatter(ghost_positions[:, 0], ghost_positions[:, 2],
                  s=120, c='magenta', marker='x', linewidths=2.5,
                  label=f'Ghost Particles (N={len(ghost_positions)})', zorder=6)

    # Grid legend
    legend_elements = [
        Line2D([0], [0], color='darkgreen', linestyle=':', linewidth=1,
               label=f'Voxels ({VOXEL_SIZE_MM}mm uniform)'),
    ]

    ax.set_xlabel('X (mm) - Flow Direction', fontsize=12)
    ax.set_ylabel('Z (mm) - Depth (Out-of-Plane)', fontsize=12)
    ax.set_title('TOP-DOWN VIEW (XZ Plane) - Looking down Y-axis\n'
                f'Cameras at ±{CAMERA_ANGLE}° | Volume: {VOLUME_SIZE_MM[0]:.1f}×{VOLUME_SIZE_MM[1]:.1f}×{VOLUME_SIZE_MM[2]:.1f} mm',
                fontsize=14, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(legend_elements)
    ax.legend(handles=handles, loc='upper left', fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(-half[0]*1.2, half[0]*1.2)
    ax.set_ylim(-half[2]*3, half[2]*3)
    ax.grid(False)

    # Stats text box
    n_ghost = len(ghost_positions)
    n_real = len(particles_mm)
    total_voxels = N_VOXELS_X * N_VOXELS_Y * N_VOXELS_Z
    stats_text = (f"Ghost-to-Real Ratio: {n_ghost/max(1,n_real):.2f}x\n"
                  f"Voxel Grid: {N_VOXELS_X}×{N_VOXELS_Y}×{N_VOXELS_Z} = {total_voxels:,} voxels\n"
                  f"Voxel Size: {VOXEL_SIZE_MM}mm uniform")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    return fig


# =============================================================================
# VISUALIZATION 2: 3D VOXEL VIEW (PLOTLY)
# =============================================================================
def make_3d_voxel_figure(cameras, particles_mm, ghost_positions):
    """
    Interactive 3D Plotly visualization showing:
    - Illuminated volume with voxel grid
    - Particles and ghost particles
    - Camera positions and sight lines
    """
    fig = go.Figure()

    half = VOLUME_SIZE_MM / 2
    cam_colors = ['red', 'blue']

    # =========================================================================
    # VOLUME BOUNDARY (wireframe box)
    # =========================================================================
    # 12 edges of a box
    edges_x = []
    edges_y = []
    edges_z = []

    # Bottom face (z = -half[2])
    for x1, x2 in [(-half[0], half[0]), (half[0], half[0]), (half[0], -half[0]), (-half[0], -half[0])]:
        for y1, y2 in [(-half[1], -half[1]), (-half[1], half[1]), (half[1], half[1]), (half[1], -half[1])]:
            pass  # Build edges properly

    # Simpler approach: define all 12 edges explicitly
    box_edges = [
        # Bottom face
        [[-half[0], -half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [-half[2], -half[2]]],
        # Top face
        [[-half[0], -half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [half[2], half[2]]],
        # Vertical edges
        [[-half[0], -half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[-half[0], -half[0]], [half[1], half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [half[1], half[1]], [-half[2], half[2]]],
    ]

    for edge in box_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2],
            mode='lines',
            line=dict(color='green', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add one trace for legend
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='green', width=4),
        name=f'Volume ({VOLUME_SIZE_MM[0]:.1f}×{VOLUME_SIZE_MM[1]:.1f}×{VOLUME_SIZE_MM[2]:.1f} mm)'
    ))

    # =========================================================================
    # VOXEL GRID LINES
    # =========================================================================
    # Create grid lines for voxel visualization
    voxel_lines_x = []
    voxel_lines_y = []
    voxel_lines_z = []

    # Grid positions
    x_positions = np.linspace(-half[0], half[0], N_VOXELS_X + 1)
    y_positions = np.linspace(-half[1], half[1], N_VOXELS_Y + 1)
    z_positions = np.linspace(-half[2], half[2], N_VOXELS_Z + 1)

    # Determine step size for visibility (show every Nth line)
    # For 0.1mm voxels we have many lines, so show every 10th (= 1mm spacing)
    step = max(1, N_VOXELS_X // 13)  # ~13 lines visible per dimension

    # Lines parallel to X (on YZ faces) - draw on all Z levels
    for y in y_positions[::step]:
        for z in z_positions:
            voxel_lines_x.extend([-half[0], half[0], None])
            voxel_lines_y.extend([y, y, None])
            voxel_lines_z.extend([z, z, None])

    # Lines parallel to Y (on XZ faces) - draw on all Z levels
    for x in x_positions[::step]:
        for z in z_positions:
            voxel_lines_x.extend([x, x, None])
            voxel_lines_y.extend([-half[1], half[1], None])
            voxel_lines_z.extend([z, z, None])

    # Lines parallel to Z (vertical lines through volume)
    for x in x_positions[::step]:
        for y in y_positions[::step]:
            voxel_lines_x.extend([x, x, None])
            voxel_lines_y.extend([y, y, None])
            voxel_lines_z.extend([-half[2], half[2], None])

    fig.add_trace(go.Scatter3d(
        x=voxel_lines_x, y=voxel_lines_y, z=voxel_lines_z,
        mode='lines',
        line=dict(color='rgba(50, 50, 50, 0.6)', width=2),
        name=f'Voxel Grid ({VOXEL_SIZE_MM}mm)',
        hoverinfo='skip'
    ))

    # =========================================================================
    # SIGHT LINES FROM CAMERAS
    # =========================================================================
    for i, cam in enumerate(cameras):
        sight_x = []
        sight_y = []
        sight_z = []

        for p in particles_mm:
            direction = cam.get_ray_direction(p)
            # Line from near camera to past particle
            start = cam.position + direction * (cam.d - 100)
            end = p + direction * 10

            sight_x.extend([start[0], end[0], None])
            sight_y.extend([start[1], end[1], None])
            sight_z.extend([start[2], end[2], None])

        fig.add_trace(go.Scatter3d(
            x=sight_x, y=sight_y, z=sight_z,
            mode='lines',
            line=dict(color=cam_colors[i], width=4),
            opacity=0.5,
            name=f'{cam.name} Sight Lines'
        ))

    # =========================================================================
    # CAMERA POSITIONS
    # =========================================================================
    for i, cam in enumerate(cameras):
        # Camera marker (scaled position for visibility)
        cam_display = cam.position * 0.15  # Scale down for display

        fig.add_trace(go.Scatter3d(
            x=[cam_display[0]], y=[cam_display[1]], z=[cam_display[2]],
            mode='markers+text',
            marker=dict(size=12, color=cam_colors[i], symbol='diamond'),
            text=[cam.name],
            textposition='top center',
            name=cam.name
        ))

        # Direction arrow
        arrow_end = cam_display + cam.view_dir * 20
        fig.add_trace(go.Scatter3d(
            x=[cam_display[0], arrow_end[0]],
            y=[cam_display[1], arrow_end[1]],
            z=[cam_display[2], arrow_end[2]],
            mode='lines',
            line=dict(color=cam_colors[i], width=4),
            showlegend=False
        ))

    # =========================================================================
    # REAL PARTICLES
    # =========================================================================
    fig.add_trace(go.Scatter3d(
        x=particles_mm[:, 0],
        y=particles_mm[:, 1],
        z=particles_mm[:, 2],
        mode='markers',
        marker=dict(size=6, color='lime', line=dict(color='darkgreen', width=1)),
        name=f'Real Particles (N={len(particles_mm)})',
        hovertemplate='Real<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<extra></extra>'
    ))

    # =========================================================================
    # GHOST PARTICLES
    # =========================================================================
    if len(ghost_positions) > 0:
        fig.add_trace(go.Scatter3d(
            x=ghost_positions[:, 0],
            y=ghost_positions[:, 1],
            z=ghost_positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='magenta', symbol='x'),
            name=f'Ghost Particles (N={len(ghost_positions)})',
            opacity=0.7,
            hovertemplate='Ghost<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<extra></extra>'
        ))

    # =========================================================================
    # LAYOUT
    # =========================================================================
    n_ghost = len(ghost_positions)
    n_real = len(particles_mm)
    total_voxels = N_VOXELS_X * N_VOXELS_Y * N_VOXELS_Z

    fig.update_layout(
        title=dict(
            text=f'<b>3D Voxel View - Stereo PIV Illuminated Region</b><br>'
                 f'<span style="font-size:12px">Volume: {VOLUME_SIZE_MM[0]:.1f}×{VOLUME_SIZE_MM[1]:.1f}×{VOLUME_SIZE_MM[2]:.1f} mm | '
                 f'Voxels: {N_VOXELS_X}×{N_VOXELS_Y}×{N_VOXELS_Z} ({VOXEL_SIZE_MM}mm uniform) = {total_voxels:,} total | '
                 f'Real: {n_real} | Ghosts: {n_ghost} ({n_ghost/max(1,n_real):.1f}x)</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm) - Flow', backgroundcolor='rgba(230,230,230,0.5)'),
            yaxis=dict(title='Y (mm) - Vertical', backgroundcolor='rgba(230,230,230,0.5)'),
            zaxis=dict(title='Z (mm) - Depth', backgroundcolor='rgba(230,230,230,0.5)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0)
            )
        ),
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


# =============================================================================
# VISUALIZATION 3: MLOS 3D RESULTS (PLOTLY)
# =============================================================================
def make_mlos_3d_figure(particles_mm, detected_peaks_mm, volume, scale):
    """
    Interactive 3D Plotly figure showing MLOS reconstruction results.

    Shows:
    - True particle positions
    - MLOS detected peaks (includes ghosts)
    - Volume intensity as isosurface or scatter
    """
    fig = go.Figure()

    half = VOLUME_SIZE_MM / 2

    # Volume boundary box
    box_edges = [
        [[-half[0], -half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], -half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [half[2], half[2]]],
        [[-half[0], -half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[-half[0], -half[0]], [half[1], half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [half[1], half[1]], [-half[2], half[2]]],
    ]

    for edge in box_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2],
            mode='lines',
            line=dict(color='green', width=3),
            showlegend=False, hoverinfo='skip'
        ))

    # True particles (larger, green)
    fig.add_trace(go.Scatter3d(
        x=particles_mm[:, 0],
        y=particles_mm[:, 1],
        z=particles_mm[:, 2],
        mode='markers',
        marker=dict(size=8, color='lime', line=dict(color='darkgreen', width=2)),
        name=f'True Particles (N={len(particles_mm)})',
        hovertemplate='TRUE<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<extra></extra>'
    ))

    # MLOS detected peaks
    if len(detected_peaks_mm) > 0:
        # Classify as real or ghost based on proximity to true particles
        is_real = []
        for peak in detected_peaks_mm:
            dists = np.linalg.norm(particles_mm - peak, axis=1)
            is_real.append(np.min(dists) < 0.5)  # 0.5mm tolerance

        is_real = np.array(is_real)
        n_real_detected = np.sum(is_real)
        n_ghost_detected = len(detected_peaks_mm) - n_real_detected

        # Plot real detections
        if n_real_detected > 0:
            real_peaks = detected_peaks_mm[is_real]
            fig.add_trace(go.Scatter3d(
                x=real_peaks[:, 0],
                y=real_peaks[:, 1],
                z=real_peaks[:, 2],
                mode='markers',
                marker=dict(size=6, color='cyan', symbol='diamond'),
                name=f'MLOS Real Detections (N={n_real_detected})',
                hovertemplate='DETECTED (real)<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<extra></extra>'
            ))

        # Plot ghost detections
        if n_ghost_detected > 0:
            ghost_peaks = detected_peaks_mm[~is_real]
            fig.add_trace(go.Scatter3d(
                x=ghost_peaks[:, 0],
                y=ghost_peaks[:, 1],
                z=ghost_peaks[:, 2],
                mode='markers',
                marker=dict(size=5, color='magenta', symbol='x'),
                opacity=0.7,
                name=f'MLOS Ghost Detections (N={n_ghost_detected})',
                hovertemplate='DETECTED (ghost)<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<extra></extra>'
            ))

    # Add high-intensity voxels as scatter (subsample for performance)
    threshold = volume.max() * 0.3
    high_idx = np.where(volume > threshold)

    if len(high_idx[0]) > 0:
        # Convert indices to mm coordinates
        nx, ny, nz = volume.shape
        x_coords = (high_idx[0] - nx/2 + 0.5) / scale
        y_coords = (high_idx[1] - ny/2 + 0.5) / scale
        z_coords = (high_idx[2] - nz/2 + 0.5) / scale
        intensities = volume[high_idx]

        # Subsample if too many points
        max_points = 5000
        if len(x_coords) > max_points:
            idx = np.random.choice(len(x_coords), max_points, replace=False)
            x_coords = x_coords[idx]
            y_coords = y_coords[idx]
            z_coords = z_coords[idx]
            intensities = intensities[idx]

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=2,
                color=intensities,
                colorscale='Hot',
                opacity=0.3,
                colorbar=dict(title='MLOS<br>Intensity', x=1.02)
            ),
            name='MLOS Volume (I > 30%)',
            hovertemplate='MLOS<br>X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Z: %{z:.2f}mm<br>I: %{marker.color:.2f}<extra></extra>'
        ))

    n_true = len(particles_mm)
    n_detected = len(detected_peaks_mm)

    fig.update_layout(
        title=dict(
            text=f'<b>MLOS Reconstruction Results</b><br>'
                 f'<span style="font-size:12px">True: {n_true} | Detected: {n_detected} '
                 f'(Real: {n_real_detected if len(detected_peaks_mm) > 0 else 0}, '
                 f'Ghosts: {n_ghost_detected if len(detected_peaks_mm) > 0 else 0})</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='rgba(230,230,230,0.5)'),
            yaxis=dict(title='Y (mm)', backgroundcolor='rgba(230,230,230,0.5)'),
            zaxis=dict(title='Z (mm)', backgroundcolor='rgba(230,230,230,0.5)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


# =============================================================================
# VISUALIZATION 4: 3D CORRELATION (PLOTLY)
# =============================================================================
def make_correlation_3d_figure(corr_volume, measured_displacement, true_displacement):
    """
    Interactive 3D Plotly figure showing the cross-correlation result.

    Shows:
    - Correlation volume (high values as scatter/isosurface)
    - Zero-lag marker at center
    - Displacement vector from center to peak
    - Comparison with true displacement
    """
    fig = go.Figure()

    nx, ny, nz = corr_volume.shape

    # Coordinate grids centered at zero (lag space)
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    z = np.arange(nz) - nz // 2

    # Threshold for visualization (top 20% of correlation values)
    threshold = corr_volume.max() * 0.5
    high_idx = np.where(corr_volume > threshold)

    if len(high_idx[0]) > 0:
        x_coords = x[high_idx[0]]
        y_coords = y[high_idx[1]]
        z_coords = z[high_idx[2]]
        intensities = corr_volume[high_idx]

        # Subsample if too many points
        max_points = 3000
        if len(x_coords) > max_points:
            idx = np.argsort(intensities)[-max_points:]  # Keep highest
            x_coords = x_coords[idx]
            y_coords = y_coords[idx]
            z_coords = z_coords[idx]
            intensities = intensities[idx]

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=intensities,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='Correlation', x=1.02)
            ),
            name='Correlation Peak',
            hovertemplate='Lag: (%{x}, %{y}, %{z})<br>Corr: %{marker.color:.2e}<extra></extra>'
        ))

    # Zero-lag marker (center = no displacement)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='Zero Lag (0,0,0)'
    ))

    # Measured displacement vector
    dx, dy, dz = measured_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dy], z=[0, dz],
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=6, color='red'),
        name=f'Measured: ({dx:.2f}, {dy:.2f}, {dz:.2f}) px'
    ))

    # True displacement vector
    tx, ty, tz = true_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, tx], y=[0, ty], z=[0, tz],
        mode='lines+markers',
        line=dict(color='green', width=4, dash='dash'),
        marker=dict(size=5, color='green', symbol='x'),
        name=f'True: ({tx:.1f}, {ty:.1f}, {tz:.1f}) px'
    ))

    # Error calculation
    error = np.array(measured_displacement) - np.array(true_displacement)
    error_mag = np.linalg.norm(error)

    fig.update_layout(
        title=dict(
            text=f'<b>3D Cross-Correlation Result (Lag Space)</b><br>'
                 f'<span style="font-size:12px">Measured: ({dx:.2f}, {dy:.2f}, {dz:.2f}) px | '
                 f'True: ({tx:.1f}, {ty:.1f}, {tz:.1f}) px | '
                 f'Error: {error_mag:.3f} px</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='Lag X (pixels)', backgroundcolor='rgba(230,230,230,0.5)'),
            yaxis=dict(title='Lag Y (pixels)', backgroundcolor='rgba(230,230,230,0.5)'),
            zaxis=dict(title='Lag Z (pixels)', backgroundcolor='rgba(230,230,230,0.5)'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


# =============================================================================
# VISUALIZATION 5: CORRELATION MAP VOLUME (PLOTLY)
# =============================================================================
def make_correlation_volume_figure(corr_volume, measured_displacement, true_displacement):
    """
    Interactive Plotly figure showing the full 3D correlation volume.

    Uses volume rendering to show the correlation intensity distribution,
    allowing inspection of the peak structure.
    """
    from plotly.subplots import make_subplots

    nx, ny, nz = corr_volume.shape

    # Coordinate arrays (centered at zero = zero-lag)
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    z = np.arange(nz) - nz // 2

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Normalize correlation for visualization
    corr_norm = corr_volume / corr_volume.max()

    fig = go.Figure()

    # Add volume rendering of correlation
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=corr_norm.flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis',
        colorbar=dict(title='Normalized<br>Correlation', x=1.02),
        name='Correlation Volume',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # Zero-lag marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='Zero Lag (0,0,0)'
    ))

    # Measured displacement vector
    dx, dy, dz = measured_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dy], z=[0, dz],
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(size=8, color='red'),
        name=f'Measured: ({dx:.2f}, {dy:.2f}, {dz:.2f}) px'
    ))

    # True displacement
    tx, ty, tz = true_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, tx], y=[0, ty], z=[0, tz],
        mode='lines+markers',
        line=dict(color='lime', width=6, dash='dash'),
        marker=dict(size=6, color='lime', symbol='x'),
        name=f'True: ({tx:.1f}, {ty:.1f}, {tz:.1f}) px'
    ))

    # Add axis lines through origin for reference
    axis_range = 20
    # X axis
    fig.add_trace(go.Scatter3d(
        x=[-axis_range, axis_range], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='gray', width=2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    # Y axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_range, axis_range], z=[0, 0],
        mode='lines', line=dict(color='gray', width=2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    # Z axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_range, axis_range],
        mode='lines', line=dict(color='gray', width=2, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    error = np.linalg.norm(np.array(measured_displacement) - np.array(true_displacement))

    fig.update_layout(
        title=dict(
            text=f'<b>3D Correlation Volume (Interactive)</b><br>'
                 f'<span style="font-size:12px">Measured: ({dx:.2f}, {dy:.2f}, {dz:.2f}) px | '
                 f'True: ({tx:.1f}, {ty:.1f}, {tz:.1f}) px | '
                 f'Error: {error:.3f} px</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='Lag X (pixels)', range=[-30, 30]),
            yaxis=dict(title='Lag Y (pixels)', range=[-30, 30]),
            zaxis=dict(title='Lag Z (pixels)', range=[-16, 16]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


# =============================================================================
# VISUALIZATION 6: GAUSSIAN FIT (PLOTLY)
# =============================================================================
def make_gaussian_fit_figure(corr_volume: np.ndarray,
                              fit_result: GaussianFitResult,
                              parabolic_displacement: np.ndarray,
                              true_displacement: np.ndarray) -> go.Figure:
    """
    Interactive Plotly figure showing the 3D Gaussian fit to the correlation peak.

    Shows:
    - Fitted Gaussian ellipsoid (1σ and 2σ contours)
    - Correlation data as scatter
    - Displacement vectors (fitted, parabolic, true)
    - Principal axes of the Gaussian
    - Fit parameters and error metrics
    """
    fig = go.Figure()

    nx, ny, nz = corr_volume.shape
    center_idx = np.array([nx // 2, ny // 2, nz // 2])

    # Convert fit center from ROI-relative to volume-relative coordinates
    # fit_result.center is relative to roi_center
    # fit_result.roi_center is in volume index space
    # For correlation volume, center (nx//2, ny//2, nz//2) = zero-lag
    fitted_center_vol = fit_result.roi_center + fit_result.center
    fitted_displacement = fitted_center_vol - center_idx

    # Negate to convert from correlation-lag to physical-displacement (same as parabolic)
    gaussian_displacement = -fitted_displacement

    # =========================================================================
    # CORRELATION DATA (high intensity voxels)
    # =========================================================================
    threshold = corr_volume.max() * 0.3
    high_idx = np.where(corr_volume > threshold)

    if len(high_idx[0]) > 0:
        # Convert to centered coordinates (zero-lag at origin)
        x_coords = high_idx[0] - center_idx[0]
        y_coords = high_idx[1] - center_idx[1]
        z_coords = high_idx[2] - center_idx[2]
        intensities = corr_volume[high_idx]

        # Subsample if needed
        max_points = 2000
        if len(x_coords) > max_points:
            idx = np.argsort(intensities)[-max_points:]
            x_coords = x_coords[idx]
            y_coords = y_coords[idx]
            z_coords = z_coords[idx]
            intensities = intensities[idx]

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=intensities,
                colorscale='Viridis',
                opacity=0.4,
                colorbar=dict(title='Correlation', x=1.02, len=0.5, y=0.75)
            ),
            name='Correlation Data',
            hovertemplate='Lag: (%{x}, %{y}, %{z})<br>Corr: %{marker.color:.2e}<extra></extra>'
        ))

    # =========================================================================
    # FITTED GAUSSIAN ELLIPSOIDS (1σ and 2σ)
    # =========================================================================
    # Note: center is in ROI-relative coords, need to convert to lag space
    # lag space center = -(fitted_displacement in volume coords relative to center)
    ellipsoid_center = -gaussian_displacement  # In lag space

    for n_sigma, opacity, name in [(1.0, 0.4, '1σ Ellipsoid'), (2.0, 0.2, '2σ Ellipsoid')]:
        X, Y, Z = get_ellipsoid_surface(ellipsoid_center, fit_result.covariance, n_sigma=n_sigma)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'rgb(255, 150, 50)'], [1, 'rgb(255, 150, 50)']],
            opacity=opacity,
            showscale=False,
            name=name,
            hoverinfo='skip'
        ))

    # =========================================================================
    # PRINCIPAL AXES
    # =========================================================================
    colors = ['red', 'green', 'blue']
    labels = ['Axis 1', 'Axis 2', 'Axis 3']
    for i in range(3):
        axis_dir = fit_result.principal_axes[:, i]
        axis_len = 2 * fit_result.principal_sigmas[i]  # 2σ length

        fig.add_trace(go.Scatter3d(
            x=[ellipsoid_center[0] - axis_dir[0]*axis_len, ellipsoid_center[0] + axis_dir[0]*axis_len],
            y=[ellipsoid_center[1] - axis_dir[1]*axis_len, ellipsoid_center[1] + axis_dir[1]*axis_len],
            z=[ellipsoid_center[2] - axis_dir[2]*axis_len, ellipsoid_center[2] + axis_dir[2]*axis_len],
            mode='lines',
            line=dict(color=colors[i], width=4),
            name=f'{labels[i]} (σ={fit_result.principal_sigmas[i]:.2f})',
            showlegend=True
        ))

    # =========================================================================
    # DISPLACEMENT VECTORS
    # =========================================================================
    # Zero-lag marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='black', symbol='diamond'),
        name='Zero Lag'
    ))

    # True displacement
    tx, ty, tz = true_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, tx], y=[0, ty], z=[0, tz],
        mode='lines+markers',
        line=dict(color='lime', width=8, dash='dash'),
        marker=dict(size=6, color='lime', symbol='x'),
        name=f'True: ({tx:.1f}, {ty:.1f}, {tz:.1f})'
    ))

    # Parabolic displacement
    px, py, pz = parabolic_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, px], y=[0, py], z=[0, pz],
        mode='lines+markers',
        line=dict(color='cyan', width=5),
        marker=dict(size=5, color='cyan'),
        name=f'Parabolic: ({px:.2f}, {py:.2f}, {pz:.2f})'
    ))

    # Gaussian-fitted displacement
    gx, gy, gz = gaussian_displacement
    fig.add_trace(go.Scatter3d(
        x=[0, gx], y=[0, gy], z=[0, gz],
        mode='lines+markers',
        line=dict(color='orange', width=6),
        marker=dict(size=7, color='orange'),
        name=f'Gaussian: ({gx:.3f}, {gy:.3f}, {gz:.3f})'
    ))

    # =========================================================================
    # CALCULATE ERRORS
    # =========================================================================
    true_disp = np.array(true_displacement)
    para_error = np.linalg.norm(parabolic_displacement - true_disp)
    gauss_error = np.linalg.norm(gaussian_displacement - true_disp)

    # =========================================================================
    # ANNOTATIONS
    # =========================================================================
    annotation_text = (
        f"<b>Gaussian Fit Parameters</b><br>"
        f"Amplitude: {fit_result.amplitude:.4f}<br>"
        f"Background: {fit_result.background:.4f}<br>"
        f"<br>"
        f"<b>Fitted Center (lag space)</b><br>"
        f"({ellipsoid_center[0]:.3f}, {ellipsoid_center[1]:.3f}, {ellipsoid_center[2]:.3f})<br>"
        f"<br>"
        f"<b>Covariance Matrix</b><br>"
        f"σxx={fit_result.covariance[0,0]:.2f}  σxy={fit_result.covariance[0,1]:.2f}  σxz={fit_result.covariance[0,2]:.2f}<br>"
        f"σyy={fit_result.covariance[1,1]:.2f}  σyz={fit_result.covariance[1,2]:.2f}<br>"
        f"σzz={fit_result.covariance[2,2]:.2f}<br>"
        f"<br>"
        f"<b>Principal Sigmas</b><br>"
        f"σ1={fit_result.principal_sigmas[0]:.2f}, σ2={fit_result.principal_sigmas[1]:.2f}, σ3={fit_result.principal_sigmas[2]:.2f}<br>"
        f"<br>"
        f"<b>Fit Quality</b><br>"
        f"R² = {fit_result.r_squared:.4f}<br>"
        f"RMS Residual = {fit_result.residual_rms:.4e}"
    )

    error_text = (
        f"<b>Displacement Comparison</b><br>"
        f"<br>"
        f"True:      ({tx:.1f}, {ty:.1f}, {tz:.1f}) px<br>"
        f"<br>"
        f"Parabolic: ({px:.3f}, {py:.3f}, {pz:.3f}) px<br>"
        f"Error:     {para_error:.4f} px<br>"
        f"<br>"
        f"Gaussian:  ({gx:.3f}, {gy:.3f}, {gz:.3f}) px<br>"
        f"Error:     {gauss_error:.4f} px<br>"
        f"<br>"
        f"<b>Improvement: {(para_error - gauss_error):.4f} px</b>"
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=f'<b>3D Gaussian Fit to Correlation Peak</b><br>'
                 f'<span style="font-size:12px">Gaussian Error: {gauss_error:.4f} px | '
                 f'Parabolic Error: {para_error:.4f} px | '
                 f'R² = {fit_result.r_squared:.4f}</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='Lag X (px)', range=[-15, 15]),
            yaxis=dict(title='Lag Y (px)', range=[-15, 15]),
            zaxis=dict(title='Lag Z (px)', range=[-10, 10]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        width=1400,
        height=900,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
        annotations=[
            dict(
                x=0.99, y=0.98,
                xref='paper', yref='paper',
                text=annotation_text,
                showarrow=False,
                font=dict(size=10, family='monospace'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                align='left',
                xanchor='right',
                yanchor='top'
            ),
            dict(
                x=0.99, y=0.45,
                xref='paper', yref='paper',
                text=error_text,
                showarrow=False,
                font=dict(size=11, family='monospace'),
                bgcolor='rgba(255,255,200,0.95)',
                bordercolor='orange',
                borderwidth=2,
                align='left',
                xanchor='right',
                yanchor='top'
            )
        ]
    )

    return fig


# =============================================================================
# MECHANICS EXPLAINER: PROJECTION (Step 5)
# =============================================================================
def make_projection_explainer_figure(camera, particle_mm):
    """
    Visualize how a 3D particle projects to a 2D sensor pixel.

    Shows:
    - 3D particle in world space
    - Camera position (pinhole)
    - Camera sensor plane
    - Light ray from particle through pinhole to sensor
    - Resulting pixel location
    """
    fig = go.Figure()

    # Camera properties
    cam_pos = camera.position
    view_dir = camera.view_dir

    # Create sensor plane corners (perpendicular to view direction)
    # Simple orthonormal basis from view direction
    up = np.array([0, 1, 0])
    if abs(np.dot(view_dir, up)) > 0.9:
        up = np.array([0, 0, 1])
    right = np.cross(view_dir, up)
    right = right / np.linalg.norm(right)
    local_up = np.cross(right, view_dir)
    local_up = local_up / np.linalg.norm(local_up)

    # Sensor plane is behind the pinhole (between camera and scene)
    sensor_distance = 50  # mm behind pinhole for visualization
    sensor_center = cam_pos + view_dir * sensor_distance
    sensor_size = 30  # mm (visual size)

    # Sensor corners
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]:
        corner = sensor_center + right * (sx * sensor_size) + local_up * (sy * sensor_size)
        corners.append(corner)
    corners = np.array(corners)

    # Project particle to sensor
    # Ray: cam_pos + t * (particle - cam_pos)
    # The projected point on sensor
    ray_dir = particle_mm - cam_pos
    ray_dir_norm = ray_dir / np.linalg.norm(ray_dir)

    # Find intersection with sensor plane
    # Plane: (P - sensor_center) . view_dir = 0
    # Line: P = cam_pos + t * ray_dir
    # t = (sensor_center - cam_pos) . view_dir / (ray_dir . view_dir)
    denom = np.dot(ray_dir_norm, view_dir)
    if abs(denom) > 1e-6:
        t = np.dot(sensor_center - cam_pos, view_dir) / denom
        sensor_hit = cam_pos + t * ray_dir_norm
    else:
        sensor_hit = sensor_center  # fallback

    # =========================================================================
    # TRACES
    # =========================================================================

    # 1. Volume boundary box (for context)
    half = VOLUME_SIZE_MM / 2
    box_edges = [
        [[-half[0], -half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], -half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [half[2], half[2]]],
        [[-half[0], -half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[-half[0], -half[0]], [half[1], half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [half[1], half[1]], [-half[2], half[2]]],
    ]
    for edge in box_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2],
            mode='lines', line=dict(color='lightgreen', width=2),
            showlegend=False, hoverinfo='skip'
        ))

    # 2. The 3D Particle
    fig.add_trace(go.Scatter3d(
        x=[particle_mm[0]], y=[particle_mm[1]], z=[particle_mm[2]],
        mode='markers+text',
        marker=dict(size=12, color='lime', line=dict(color='darkgreen', width=2)),
        text=['<b>3D Particle</b>'],
        textposition='top center',
        textfont=dict(size=12, color='darkgreen'),
        name='3D Particle'
    ))

    # 3. Camera Pinhole (scaled for visibility)
    cam_display = cam_pos * 0.15
    fig.add_trace(go.Scatter3d(
        x=[cam_display[0]], y=[cam_display[1]], z=[cam_display[2]],
        mode='markers+text',
        marker=dict(size=14, color='black', symbol='diamond'),
        text=[f'<b>{camera.name}<br>Pinhole</b>'],
        textposition='top center',
        textfont=dict(size=10),
        name='Camera Pinhole'
    ))

    # 4. Sensor Plane (scaled)
    corners_scaled = sensor_center * 0.15 + (corners - sensor_center) * 0.3
    fig.add_trace(go.Scatter3d(
        x=corners_scaled[:, 0], y=corners_scaled[:, 1], z=corners_scaled[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Sensor Plane'
    ))

    # Fill sensor plane
    fig.add_trace(go.Mesh3d(
        x=corners_scaled[:-1, 0], y=corners_scaled[:-1, 1], z=corners_scaled[:-1, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='lightblue', opacity=0.3,
        name='Sensor Surface',
        showlegend=False
    ))

    # 5. Light Ray (Particle -> Pinhole -> Sensor)
    sensor_hit_scaled = sensor_center * 0.15 + (sensor_hit - sensor_center) * 0.3
    fig.add_trace(go.Scatter3d(
        x=[particle_mm[0], cam_display[0], sensor_hit_scaled[0]],
        y=[particle_mm[1], cam_display[1], sensor_hit_scaled[1]],
        z=[particle_mm[2], cam_display[2], sensor_hit_scaled[2]],
        mode='lines+markers',
        line=dict(color='red', width=4, dash='dash'),
        marker=dict(size=4, color='red'),
        name='Light Ray'
    ))

    # 6. Pixel Hit Point
    fig.add_trace(go.Scatter3d(
        x=[sensor_hit_scaled[0]], y=[sensor_hit_scaled[1]], z=[sensor_hit_scaled[2]],
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='square'),
        text=['<b>Pixel (u,v)</b>'],
        textposition='bottom center',
        textfont=dict(size=11, color='red'),
        name='Projected Pixel'
    ))

    # Get actual pixel coordinates
    pixel_uv, _ = camera.project(particle_mm)

    fig.update_layout(
        title=dict(
            text=f'<b>Step 5: Projection (Forward Model)</b><br>'
                 f'<span style="font-size:12px">How a 3D particle at ({particle_mm[0]:.1f}, {particle_mm[1]:.1f}, {particle_mm[2]:.1f}) mm '
                 f'becomes pixel ({pixel_uv[0]:.1f}, {pixel_uv[1]:.1f})</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=-1.2, z=0.8))
        ),
        width=1100,
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
        annotations=[
            dict(
                x=0.02, y=0.02,
                xref='paper', yref='paper',
                text=(
                    '<b>Projection Equation:</b><br>'
                    'u = f * (x_cam / z_cam) + cx<br>'
                    'v = f * (y_cam / z_cam) + cy<br><br>'
                    'Light travels from particle through pinhole<br>'
                    'to form an inverted image on the sensor.'
                ),
                showarrow=False,
                font=dict(size=11, family='monospace'),
                bgcolor='rgba(255,255,200,0.95)',
                bordercolor='orange',
                borderwidth=1,
                align='left'
            )
        ]
    )

    return fig


# =============================================================================
# MECHANICS EXPLAINER: MLOS (Step 6)
# =============================================================================
def make_mlos_explainer_figure(cameras, voxel_center_mm):
    """
    Visualize how MLOS reconstruction works.

    Shows:
    - Two cameras with their positions
    - Lines of sight from each camera through the target voxel
    - The voxel where rays intersect
    - Explanation of the multiplication principle
    """
    fig = go.Figure()

    cam1, cam2 = cameras
    cam1_pos = cam1.position
    cam2_pos = cam2.position

    # =========================================================================
    # TRACES
    # =========================================================================

    # 1. Volume boundary box
    half = VOLUME_SIZE_MM / 2
    box_edges = [
        [[-half[0], -half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [-half[2], -half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [-half[2], -half[2]]],
        [[-half[0], -half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[half[0], half[0]], [-half[1], half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [-half[1], -half[1]], [half[2], half[2]]],
        [[-half[0], half[0]], [half[1], half[1]], [half[2], half[2]]],
        [[-half[0], -half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [-half[1], -half[1]], [-half[2], half[2]]],
        [[-half[0], -half[0]], [half[1], half[1]], [-half[2], half[2]]],
        [[half[0], half[0]], [half[1], half[1]], [-half[2], half[2]]],
    ]
    for edge in box_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2],
            mode='lines', line=dict(color='lightgreen', width=2),
            showlegend=False, hoverinfo='skip'
        ))

    # 2. Camera positions (scaled for visibility)
    cam_colors = ['red', 'blue']
    for i, cam in enumerate(cameras):
        cam_display = cam.position * 0.12
        fig.add_trace(go.Scatter3d(
            x=[cam_display[0]], y=[cam_display[1]], z=[cam_display[2]],
            mode='markers+text',
            marker=dict(size=14, color='black', symbol='diamond'),
            text=[f'<b>{cam.name}</b>'],
            textposition='top center',
            textfont=dict(size=11),
            name=cam.name
        ))

    # 3. The Target Voxel (cube)
    vx, vy, vz = voxel_center_mm
    voxel_size = 0.3  # mm
    # Create cube vertices
    cube_x = [vx-voxel_size, vx+voxel_size, vx+voxel_size, vx-voxel_size,
              vx-voxel_size, vx+voxel_size, vx+voxel_size, vx-voxel_size]
    cube_y = [vy-voxel_size, vy-voxel_size, vy+voxel_size, vy+voxel_size,
              vy-voxel_size, vy-voxel_size, vy+voxel_size, vy+voxel_size]
    cube_z = [vz-voxel_size, vz-voxel_size, vz-voxel_size, vz-voxel_size,
              vz+voxel_size, vz+voxel_size, vz+voxel_size, vz+voxel_size]

    fig.add_trace(go.Mesh3d(
        x=cube_x, y=cube_y, z=cube_z,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color='cyan', opacity=0.5,
        name='Reconstruction Voxel'
    ))

    # 4. Lines of Sight from each camera through the voxel
    for i, cam in enumerate(cameras):
        cam_display = cam.position * 0.12

        # Direction from camera to voxel
        ray_dir = voxel_center_mm - cam.position
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Extend ray beyond voxel
        ray_end = voxel_center_mm + ray_dir * 8  # mm beyond

        fig.add_trace(go.Scatter3d(
            x=[cam_display[0], voxel_center_mm[0], ray_end[0]],
            y=[cam_display[1], voxel_center_mm[1], ray_end[1]],
            z=[cam_display[2], voxel_center_mm[2], ray_end[2]],
            mode='lines',
            line=dict(color=cam_colors[i], width=6),
            name=f'Line of Sight {i+1}'
        ))

    # 5. Intersection marker
    fig.add_trace(go.Scatter3d(
        x=[vx], y=[vy], z=[vz],
        mode='markers',
        marker=dict(size=8, color='yellow', symbol='diamond',
                    line=dict(color='black', width=2)),
        name='Intersection Point'
    ))

    fig.update_layout(
        title=dict(
            text='<b>Step 6: MLOS Reconstruction</b><br>'
                 '<span style="font-size:12px">Lines of sight from multiple cameras intersect to locate particles</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        width=1100,
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
        annotations=[
            dict(
                x=0.02, y=0.02,
                xref='paper', yref='paper',
                text=(
                    '<b>MLOS Principle:</b><br>'
                    'I_voxel = I_cam1 * I_cam2<br><br>'
                    'At the TRUE particle location:<br>'
                    '  Both cameras see bright -> 1.0 * 1.0 = 1.0 (HIGH)<br><br>'
                    'At GHOST locations:<br>'
                    '  One camera sees dark -> 1.0 * 0.0 = 0.0 (LOW)<br><br>'
                    'Multiplication suppresses false intersections!'
                ),
                showarrow=False,
                font=dict(size=11, family='monospace'),
                bgcolor='rgba(200,255,200,0.95)',
                bordercolor='green',
                borderwidth=2,
                align='left'
            )
        ]
    )

    return fig


# =============================================================================
# MECHANICS EXPLAINER: CORRELATION OVERLAP
# =============================================================================
def make_correlation_explainer_figure(true_displacement):
    """
    Visualize how 3D correlation works as a "sliding alignment" problem.

    Shows:
    - Volume A (reference) as fixed cubes
    - Volume B (displaced) as semi-transparent cubes
    - The "winning shift" where they align
    - Displacement vector
    """
    fig = go.Figure()

    # Create synthetic particle positions for Volume A
    np.random.seed(123)
    n_demo_particles = 8
    vol_A_pts = np.random.rand(n_demo_particles, 3) * 8 - 4  # In range [-4, 4]

    # Volume B is shifted by the true displacement
    shift = np.array(true_displacement) / SCALE  # Convert px to mm
    vol_B_pts = vol_A_pts + shift

    # =========================================================================
    # TRACES
    # =========================================================================

    # 1. Volume A particles (Red - Fixed Reference)
    fig.add_trace(go.Scatter3d(
        x=vol_A_pts[:, 0], y=vol_A_pts[:, 1], z=vol_A_pts[:, 2],
        mode='markers',
        marker=dict(size=12, color='red', opacity=0.6,
                    line=dict(color='darkred', width=2)),
        name='Volume A (t=0) - Reference'
    ))

    # 2. Volume B particles (Blue - Displaced, semi-transparent)
    fig.add_trace(go.Scatter3d(
        x=vol_B_pts[:, 0], y=vol_B_pts[:, 1], z=vol_B_pts[:, 2],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.3,
                    line=dict(color='darkblue', width=1)),
        name='Volume B (t=1) - Displaced'
    ))

    # 3. Volume B shifted back to align with A (Green - The "winner")
    shifted_back = vol_B_pts - shift
    fig.add_trace(go.Scatter3d(
        x=shifted_back[:, 0], y=shifted_back[:, 1], z=shifted_back[:, 2],
        mode='markers',
        marker=dict(size=14, color='lime', opacity=0.8, symbol='diamond',
                    line=dict(color='darkgreen', width=2)),
        name='B Shifted Back (Perfect Alignment)'
    ))

    # 4. Displacement vectors for each particle
    for i in range(n_demo_particles):
        fig.add_trace(go.Scatter3d(
            x=[vol_A_pts[i, 0], vol_B_pts[i, 0]],
            y=[vol_A_pts[i, 1], vol_B_pts[i, 1]],
            z=[vol_A_pts[i, 2], vol_B_pts[i, 2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # 5. Main displacement vector (from center)
    center = np.mean(vol_A_pts, axis=0)
    fig.add_trace(go.Scatter3d(
        x=[center[0], center[0] + shift[0]],
        y=[center[1], center[1] + shift[1]],
        z=[center[2], center[2] + shift[2]],
        mode='lines+markers',
        line=dict(color='black', width=8),
        marker=dict(size=6, color='black'),
        name=f'Displacement ({true_displacement[0]:.0f}, {true_displacement[1]:.0f}, {true_displacement[2]:.0f}) px'
    ))

    # 6. Bounding boxes for volumes
    a_min, a_max = vol_A_pts.min(axis=0) - 1, vol_A_pts.max(axis=0) + 1
    b_min, b_max = vol_B_pts.min(axis=0) - 1, vol_B_pts.max(axis=0) + 1

    # Volume A box (red)
    for start, end in [
        ([a_min[0], a_min[1], a_min[2]], [a_max[0], a_min[1], a_min[2]]),
        ([a_min[0], a_max[1], a_min[2]], [a_max[0], a_max[1], a_min[2]]),
        ([a_min[0], a_min[1], a_max[2]], [a_max[0], a_min[1], a_max[2]]),
        ([a_min[0], a_max[1], a_max[2]], [a_max[0], a_max[1], a_max[2]]),
        ([a_min[0], a_min[1], a_min[2]], [a_min[0], a_max[1], a_min[2]]),
        ([a_max[0], a_min[1], a_min[2]], [a_max[0], a_max[1], a_min[2]]),
        ([a_min[0], a_min[1], a_max[2]], [a_min[0], a_max[1], a_max[2]]),
        ([a_max[0], a_min[1], a_max[2]], [a_max[0], a_max[1], a_max[2]]),
        ([a_min[0], a_min[1], a_min[2]], [a_min[0], a_min[1], a_max[2]]),
        ([a_max[0], a_min[1], a_min[2]], [a_max[0], a_min[1], a_max[2]]),
        ([a_min[0], a_max[1], a_min[2]], [a_min[0], a_max[1], a_max[2]]),
        ([a_max[0], a_max[1], a_min[2]], [a_max[0], a_max[1], a_max[2]]),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines', line=dict(color='red', width=2),
            showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(
            text='<b>Correlation Mechanism: Finding the Best Alignment</b><br>'
                 f'<span style="font-size:12px">True displacement: ({true_displacement[0]:.0f}, {true_displacement[1]:.0f}, {true_displacement[2]:.0f}) pixels</span>',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        width=1100,
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
        annotations=[
            dict(
                x=0.02, y=0.02,
                xref='paper', yref='paper',
                text=(
                    '<b>Correlation = Sliding Overlap Search</b><br><br>'
                    '1. Take Volume A (Red) as reference<br>'
                    '2. Slide Volume B (Blue) through all possible shifts<br>'
                    '3. At each shift, compute overlap score:<br>'
                    '   Score = sum( A[i,j,k] * B[i+dx, j+dy, k+dz] )<br><br>'
                    '4. The PEAK in the correlation map is where<br>'
                    '   B aligns perfectly with A (Green diamonds)<br><br>'
                    '<b>FFT makes this O(N log N) instead of O(N^2)!</b>'
                ),
                showarrow=False,
                font=dict(size=11, family='monospace'),
                bgcolor='rgba(255,255,200,0.95)',
                bordercolor='orange',
                borderwidth=2,
                align='left'
            )
        ]
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Stereo PIV Geometry Visualization")
    print("=" * 60)
    print(f"Volume: {VOLUME_SIZE_MM[0]:.1f} × {VOLUME_SIZE_MM[1]:.1f} × {VOLUME_SIZE_MM[2]:.1f} mm")
    print(f"Voxel size: {VOXEL_SIZE_MM} mm (uniform in all directions)")
    print(f"Voxel grid: {N_VOXELS_X} × {N_VOXELS_Y} × {N_VOXELS_Z} = {N_VOXELS_X*N_VOXELS_Y*N_VOXELS_Z:,} voxels")
    print(f"Displacement: {DISPLACEMENT} px = {DISPLACEMENT/SCALE} mm")

    # 1. Create cameras
    cam1 = StereoCamera(+CAMERA_ANGLE, WORKING_DISTANCE, name="Cam1 (+45°)")
    cam2 = StereoCamera(-CAMERA_ANGLE, WORKING_DISTANCE, name="Cam2 (-45°)")
    cameras = [cam1, cam2]

    print(f"\nCamera positions:")
    print(f"  Cam1: ({cam1.position[0]:.1f}, {cam1.position[1]:.1f}, {cam1.position[2]:.1f}) mm")
    print(f"  Cam2: ({cam2.position[0]:.1f}, {cam2.position[1]:.1f}, {cam2.position[2]:.1f}) mm")

    # 2. Generate particles
    particles_a_mm, particles_a_px = generate_particles(
        NUM_PARTICLES,
        [IMAGE_SIZE, IMAGE_SIZE, VOLUME_DEPTH],
        SCALE,
        seed=PARTICLE_SEED
    )

    particles_b_mm, particles_b_px = displace_particles(
        particles_a_px, DISPLACEMENT, SCALE
    )

    print(f"\nGenerated {NUM_PARTICLES} particles")

    # 3. Find ghost particles
    print("Finding ghost particle intersections...")
    real_indices, ghost_positions, ghost_info = find_line_intersections(
        cam1, cam2, particles_a_mm, tolerance_mm=0.5
    )

    print(f"  Real particles matched: {len(real_indices)}")
    print(f"  Ghost particles found: {len(ghost_positions)}")
    print(f"  Ghost-to-real ratio: {len(ghost_positions)/len(particles_a_mm):.2f}x")

    # 4. MLOS RECONSTRUCTION
    print("\n" + "=" * 60)
    print("MLOS RECONSTRUCTION")
    print("=" * 60)

    # Step 1: Render camera images
    print("\nStep 1: Rendering camera images...")
    image1 = render_image(cam1, particles_a_mm)
    image2 = render_image(cam2, particles_a_mm)
    images = [image1, image2]
    print(f"  Camera 1 image: {image1.shape}, max intensity: {image1.max():.3f}")
    print(f"  Camera 2 image: {image2.shape}, max intensity: {image2.max():.3f}")

    # Step 2: Create voxel grid and precompute projections
    print("\nStep 2: Precomputing voxel projections...")
    voxel_coords = create_voxel_grid([IMAGE_SIZE, IMAGE_SIZE, VOLUME_DEPTH])
    projection_maps = precompute_projections(cameras, voxel_coords, SCALE)
    print(f"  Voxel grid: {IMAGE_SIZE}×{IMAGE_SIZE}×{VOLUME_DEPTH} = {IMAGE_SIZE*IMAGE_SIZE*VOLUME_DEPTH:,} voxels")

    # Step 3: MLOS reconstruction
    print("\nStep 3: MLOS reconstruction (multiplying camera intensities)...")
    mlos_volume = mlos_reconstruct(images, projection_maps)
    print(f"  Volume shape: {mlos_volume.shape}")
    print(f"  Max MLOS intensity: {mlos_volume.max():.6f}")
    print(f"  Non-zero voxels: {np.sum(mlos_volume > 0):,}")

    # Step 4: Find peaks in reconstructed volume
    print("\nStep 4: Finding peaks in reconstructed volume...")
    detected_peaks_px, peak_intensities = find_peaks_in_volume(mlos_volume, threshold_fraction=0.1)
    detected_peaks_mm = detected_peaks_px / SCALE
    print(f"  Detected peaks: {len(detected_peaks_px)}")

    # Classify detected peaks as real or ghost
    n_real_detected = 0
    n_ghost_detected = 0
    for peak in detected_peaks_mm:
        dists = np.linalg.norm(particles_a_mm - peak, axis=1)
        if np.min(dists) < 0.5:  # 0.5mm tolerance
            n_real_detected += 1
        else:
            n_ghost_detected += 1

    print(f"  → Real particles detected: {n_real_detected}/{len(particles_a_mm)}")
    print(f"  → Ghost particles detected: {n_ghost_detected}")

    # 5. PROCESS TIME B (DISPLACED PARTICLES)
    print("\n" + "=" * 60)
    print("PROCESSING TIME B (DISPLACED)")
    print("=" * 60)

    # Render images for time B
    print("\nStep 5: Rendering Time B images...")
    image1_b = render_image(cam1, particles_b_mm)
    image2_b = render_image(cam2, particles_b_mm)
    images_b = [image1_b, image2_b]
    print(f"  Camera 1 image B: {image1_b.shape}, max intensity: {image1_b.max():.3f}")
    print(f"  Camera 2 image B: {image2_b.shape}, max intensity: {image2_b.max():.3f}")

    # MLOS reconstruction for time B (reuse projection_maps - cameras haven't moved!)
    print("\nStep 6: MLOS reconstruction for Time B...")
    mlos_volume_b = mlos_reconstruct(images_b, projection_maps)
    print(f"  Volume B shape: {mlos_volume_b.shape}")
    print(f"  Max MLOS intensity: {mlos_volume_b.max():.6f}")

    # 6. 3D CROSS-CORRELATION
    print("\n" + "=" * 60)
    print("3D CROSS-CORRELATION")
    print("=" * 60)

    print("\nStep 7: Computing 3D FFT cross-correlation...")
    corr_volume = correlate_3d(mlos_volume, mlos_volume_b)
    print(f"  Correlation volume shape: {corr_volume.shape}")
    print(f"  Max correlation: {corr_volume.max():.6e}")

    print("\nStep 8: Finding displacement peak...")
    measured_displacement, peak_value = find_displacement_3d(corr_volume, subpixel=True)
    true_displacement = DISPLACEMENT  # In pixels

    # Calculate error
    error = measured_displacement - true_displacement
    error_magnitude = np.linalg.norm(error)

    print(f"\n  {'='*50}")
    print(f"  DISPLACEMENT RESULTS")
    print(f"  {'='*50}")
    print(f"  True displacement (px):     ({true_displacement[0]:.1f}, {true_displacement[1]:.1f}, {true_displacement[2]:.1f})")
    print(f"  Measured displacement (px): ({measured_displacement[0]:.2f}, {measured_displacement[1]:.2f}, {measured_displacement[2]:.2f})")
    print(f"  Error (px):                 ({error[0]:.3f}, {error[1]:.3f}, {error[2]:.3f})")
    print(f"  Error magnitude:            {error_magnitude:.4f} px")
    print(f"  Peak correlation value:     {peak_value:.6e}")
    print(f"  {'='*50}")

    # 9. GAUSSIAN FIT TO CORRELATION PEAK
    print("\n" + "=" * 60)
    print("3D GAUSSIAN FIT TO CORRELATION PEAK")
    print("=" * 60)

    print("\nStep 9: Fitting 3D Gaussian to correlation peak...")

    # Find peak location for ROI
    peak_idx = np.argmax(corr_volume)
    peak_loc = np.array(np.unravel_index(peak_idx, corr_volume.shape))

    # Fit Gaussian
    gauss_fit = fit_gaussian_3d(corr_volume, roi_center=peak_loc, roi_size=10)

    # Convert to displacement (same sign convention as parabolic)
    nx, ny, nz = corr_volume.shape
    center_idx = np.array([nx // 2, ny // 2, nz // 2])
    fitted_center_vol = gauss_fit.roi_center + gauss_fit.center
    gaussian_displacement = -(fitted_center_vol - center_idx)

    gauss_error = np.linalg.norm(gaussian_displacement - true_displacement)
    para_error = np.linalg.norm(measured_displacement - true_displacement)

    print(f"\n  {'='*50}")
    print(f"  GAUSSIAN FIT RESULTS")
    print(f"  {'='*50}")
    print(f"  Amplitude: {gauss_fit.amplitude:.6f}")
    print(f"  Background: {gauss_fit.background:.6f}")
    print(f"  R-squared: {gauss_fit.r_squared:.4f}")
    print(f"  RMS Residual: {gauss_fit.residual_rms:.6e}")
    print(f"  ")
    print(f"  Covariance Matrix:")
    print(f"    σxx={gauss_fit.covariance[0,0]:.3f}  σxy={gauss_fit.covariance[0,1]:.3f}  σxz={gauss_fit.covariance[0,2]:.3f}")
    print(f"    σyy={gauss_fit.covariance[1,1]:.3f}  σyz={gauss_fit.covariance[1,2]:.3f}")
    print(f"    σzz={gauss_fit.covariance[2,2]:.3f}")
    print(f"  ")
    print(f"  Principal sigmas: σ1={gauss_fit.principal_sigmas[0]:.2f}, σ2={gauss_fit.principal_sigmas[1]:.2f}, σ3={gauss_fit.principal_sigmas[2]:.2f}")
    print(f"  ")
    print(f"  {'='*50}")
    print(f"  DISPLACEMENT COMPARISON")
    print(f"  {'='*50}")
    print(f"  True displacement (px):      ({true_displacement[0]:.1f}, {true_displacement[1]:.1f}, {true_displacement[2]:.1f})")
    print(f"  Parabolic displacement (px): ({measured_displacement[0]:.3f}, {measured_displacement[1]:.3f}, {measured_displacement[2]:.3f})")
    print(f"  Parabolic error:             {para_error:.4f} px")
    print(f"  Gaussian displacement (px):  ({gaussian_displacement[0]:.3f}, {gaussian_displacement[1]:.3f}, {gaussian_displacement[2]:.3f})")
    print(f"  Gaussian error:              {gauss_error:.4f} px")
    print(f"  Improvement:                 {para_error - gauss_error:.4f} px")
    print(f"  {'='*50}")

    # 10. Create figures
    base_path = "/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/manual_tools/Ensemble_stereo"

    print("\n" + "=" * 60)
    print("Creating figures...")
    print("=" * 60)

    # Figure 1: Top-down view (matplotlib)
    print("\n1. Top-down view (XZ plane)...")
    fig1 = make_top_down_figure(cameras, particles_a_mm, ghost_positions)
    fig1.savefig(f"{base_path}/view_1_top_down_XZ.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   Saved: view_1_top_down_XZ.png")

    # Figure 2: 3D voxel view (Plotly)
    print("\n2. 3D voxel view (interactive)...")
    fig2 = make_3d_voxel_figure(cameras, particles_a_mm, ghost_positions)
    fig2.write_html(f"{base_path}/view_2_3d_voxels.html")
    print(f"   Saved: view_2_3d_voxels.html")

    # Figure 3: MLOS 3D results (Plotly)
    print("\n3. MLOS 3D results (interactive)...")
    fig3 = make_mlos_3d_figure(particles_a_mm, detected_peaks_mm, mlos_volume, SCALE)
    fig3.write_html(f"{base_path}/view_3_mlos_3d.html")
    print(f"   Saved: view_3_mlos_3d.html")

    # Figure 4: 3D Cross-correlation results (Plotly)
    print("\n4. 3D Cross-correlation results (interactive)...")
    fig4 = make_correlation_3d_figure(corr_volume, measured_displacement, true_displacement)
    fig4.write_html(f"{base_path}/view_4_correlation_3d.html")
    print(f"   Saved: view_4_correlation_3d.html")

    # Figure 5: Correlation volume (Plotly interactive)
    print("\n5. Correlation volume (interactive)...")
    fig5 = make_correlation_volume_figure(corr_volume, measured_displacement, true_displacement)
    fig5.write_html(f"{base_path}/view_5_correlation_volume.html")
    print(f"   Saved: view_5_correlation_volume.html")

    # Figure 6: Gaussian fit (Plotly interactive)
    print("\n6. Gaussian fit to correlation peak (interactive)...")
    fig6 = make_gaussian_fit_figure(corr_volume, gauss_fit, measured_displacement, true_displacement)
    fig6.write_html(f"{base_path}/view_6_gaussian_fit.html")
    print(f"   Saved: view_6_gaussian_fit.html")

    # =========================================================================
    # MECHANICS EXPLAINER FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("MECHANICS EXPLAINER FIGURES")
    print("=" * 60)

    # Figure 7: Projection Explainer
    print("\n7. Projection mechanics explainer...")
    # Use first particle and first camera
    demo_particle = particles_a_mm[0]
    fig7 = make_projection_explainer_figure(cam1, demo_particle)
    fig7.write_html(f"{base_path}/explainer_1_projection.html")
    print(f"   Saved: explainer_1_projection.html")

    # Figure 8: MLOS Explainer
    print("\n8. MLOS mechanics explainer...")
    # Use first particle position as the voxel we're reconstructing
    fig8 = make_mlos_explainer_figure(cameras, particles_a_mm[0])
    fig8.write_html(f"{base_path}/explainer_2_mlos.html")
    print(f"   Saved: explainer_2_mlos.html")

    # Figure 9: Correlation Explainer
    print("\n9. Correlation mechanics explainer...")
    fig9 = make_correlation_explainer_figure(true_displacement)
    fig9.write_html(f"{base_path}/explainer_3_correlation.html")
    print(f"   Saved: explainer_3_correlation.html")

    plt.show()

    print("\n" + "=" * 60)
    print("Done! Generated 9 figures.")
    print("=" * 60)
    print("\nMain Pipeline Figures:")
    print(f"  1. view_1_top_down_XZ.png (matplotlib) - Camera geometry")
    print(f"  2. view_2_3d_voxels.html (Plotly) - 3D voxel structure")
    print(f"  3. view_3_mlos_3d.html (Plotly) - MLOS 3D results")
    print(f"  4. view_4_correlation_3d.html (Plotly) - 3D correlation peak")
    print(f"  5. view_5_correlation_volume.html (Plotly) - Full correlation volume")
    print(f"  6. view_6_gaussian_fit.html (Plotly) - 3D Gaussian fit with error")
    print("\nMechanics Explainer Figures:")
    print(f"  7. explainer_1_projection.html - How 3D particles project to 2D pixels")
    print(f"  8. explainer_2_mlos.html - How MLOS reconstruction works")
    print(f"  9. explainer_3_correlation.html - How correlation finds displacement")
    print("=" * 60)
