"""
Stereo Ensemble Image Generator
===============================

Generates synthetic stereo image pairs with known Reynolds stress tensor
for validation of the 3D ensemble PIV system.

The key physics:
- Particles are displaced according to: dx ~ N(mean_displacement, reynolds_stress)
- The Reynolds stress tensor R_ij = <u'_i u'_j> defines the covariance of velocity fluctuations
- By fitting the correlation peak width, we can extract this tensor
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Iterator, List, Optional, Union
import os


# =============================================================================
# STEREO CAMERA MODEL (adapted from stereo_geometry_viz.py)
# =============================================================================
class StereoCamera:
    """Pinhole camera model for stereo PIV at a given angle from Z-axis."""

    def __init__(
        self,
        angle_deg: float,
        working_distance_mm: float,
        name: str = "Camera",
        focal_length_mm: float = 100.0,
        image_size_px: int = 512,
        scale_px_per_mm: float = 10.0
    ):
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

    def world_to_camera(self, points_world_mm: np.ndarray) -> np.ndarray:
        """Transform points from world coordinates to camera coordinates."""
        translated = points_world_mm - self.position
        return np.dot(translated, self.R.T)

    def project(self, points_world_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
# IMAGE RENDERING
# =============================================================================
def render_image(
    camera: StereoCamera,
    particle_positions_mm: np.ndarray,
    image_size: int,
    particle_diameter: float,
    particle_intensities: Optional[np.ndarray] = None
) -> np.ndarray:
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
    particle_intensities : ndarray, shape (N,), optional
        Intensity of each particle (default: all 1.0)

    Returns
    -------
    image : ndarray, shape (image_size, image_size)
        Rendered image with Gaussian particle blobs
    """
    image = np.zeros((image_size, image_size), dtype=np.float64)
    sigma = particle_diameter / (2 * 2.355)  # FWHM to sigma

    if particle_intensities is None:
        particle_intensities = np.ones(len(particle_positions_mm))

    for idx, pos_mm in enumerate(particle_positions_mm):
        pixel_pos, valid = camera.project(pos_mm)

        if not valid:
            continue

        u, v = pixel_pos
        intensity = particle_intensities[idx]

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
                    image[ii, jj] += intensity * np.exp(-dist_sq / (2 * sigma**2))

    return image


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class StereoEnsembleConfig:
    """Configuration for stereo ensemble generation."""

    # Volume dimensions (pixels)
    volume_size: Tuple[int, int, int] = (128, 128, 32)  # X, Y, Z
    scale_px_per_mm: float = 10.0

    # Particle parameters
    num_particles: int = 500
    particle_diameter_px: float = 3.0
    particle_intensity_mean: float = 1.0
    particle_intensity_std: float = 0.1

    # Camera parameters
    camera_angles_deg: Tuple[float, float] = (-45.0, 45.0)
    working_distance_mm: float = 500.0
    focal_length_mm: float = 100.0
    image_size_px: int = 128  # Should match max(volume_size[:2]) for particles to fill image

    # Physics: Reynolds stress tensor (3x3 symmetric positive semi-definite)
    # R_ij = <u'_i u'_j> in (pixels/frame)^2
    # Default: some turbulence with shear stress
    reynolds_stress: np.ndarray = field(default_factory=lambda: np.array([
        [1.0, 0.3, 0.1],
        [0.3, 0.8, 0.2],
        [0.1, 0.2, 0.5]
    ]))

    # Mean displacement (pixels/frame)
    mean_displacement: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # Ensemble parameters
    num_image_pairs: int = 1000
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        self.reynolds_stress = np.asarray(self.reynolds_stress)
        self.mean_displacement = np.asarray(self.mean_displacement)

        # Check Reynolds stress is symmetric
        if not np.allclose(self.reynolds_stress, self.reynolds_stress.T):
            raise ValueError("Reynolds stress tensor must be symmetric")

        # Check positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.reynolds_stress)
        if np.any(eigvals < -1e-10):
            raise ValueError("Reynolds stress tensor must be positive semi-definite")


# =============================================================================
# GENERATOR CLASS
# =============================================================================
class StereoEnsembleGenerator:
    """Generate synthetic stereo image pairs with known Reynolds stress."""

    def __init__(self, config: StereoEnsembleConfig):
        """
        Initialize the generator.

        Parameters
        ----------
        config : StereoEnsembleConfig
            Configuration parameters
        """
        self.config = config
        self.cameras: List[StereoCamera] = []
        self._setup_cameras()

        # Particle volume bounds in mm (derived from volume_size and scale)
        # For particles to fill the image, image_size_px should equal max(volume_size[:2])
        vol_px = np.array(config.volume_size)
        self.volume_half_mm = vol_px / (2 * config.scale_px_per_mm)

    def _setup_cameras(self) -> None:
        """Initialize stereo camera pair."""
        cfg = self.config
        for i, angle in enumerate(cfg.camera_angles_deg):
            cam = StereoCamera(
                angle_deg=angle,
                working_distance_mm=cfg.working_distance_mm,
                name=f"Camera{i+1}",
                focal_length_mm=cfg.focal_length_mm,
                image_size_px=cfg.image_size_px,
                scale_px_per_mm=cfg.scale_px_per_mm
            )
            self.cameras.append(cam)

    def generate_particles(self, seed: int) -> np.ndarray:
        """
        Generate random 3D particle positions uniformly in volume.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility

        Returns
        -------
        positions_mm : ndarray, shape (N, 3)
            Particle positions in world coordinates (mm)
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        # Generate uniform positions in [-half, +half] for each dimension
        positions_mm = rng.uniform(
            low=-self.volume_half_mm,
            high=self.volume_half_mm,
            size=(cfg.num_particles, 3)
        )

        return positions_mm

    def generate_particle_intensities(self, num_particles: int, seed: int) -> np.ndarray:
        """
        Generate random particle intensities.

        Parameters
        ----------
        num_particles : int
            Number of particles
        seed : int
            Random seed

        Returns
        -------
        intensities : ndarray, shape (N,)
            Intensity values (always positive)
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        intensities = rng.normal(
            loc=cfg.particle_intensity_mean,
            scale=cfg.particle_intensity_std,
            size=num_particles
        )
        # Ensure positive
        intensities = np.maximum(intensities, 0.1)
        return intensities

    def sample_displacements(self, num_particles: int, seed: int) -> np.ndarray:
        """
        Sample particle displacements from multivariate normal.

        The covariance matrix equals the Reynolds stress tensor:
        cov(u', v', w') = R_ij

        Parameters
        ----------
        num_particles : int
            Number of particles
        seed : int
            Random seed for reproducibility

        Returns
        -------
        displacements_px : ndarray, shape (N, 3)
            Particle displacement vectors in pixels
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        displacements_px = rng.multivariate_normal(
            mean=cfg.mean_displacement,
            cov=cfg.reynolds_stress,
            size=num_particles
        )

        return displacements_px

    def render_frame(
        self,
        positions_mm: np.ndarray,
        intensities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render stereo image pair for given particle positions.

        Parameters
        ----------
        positions_mm : ndarray, shape (N, 3)
            Particle positions in world coordinates (mm)
        intensities : ndarray, shape (N,), optional
            Particle intensities

        Returns
        -------
        image1, image2 : ndarray
            Images from camera 1 and camera 2
        """
        cfg = self.config

        image1 = render_image(
            self.cameras[0],
            positions_mm,
            cfg.image_size_px,
            cfg.particle_diameter_px,
            intensities
        )
        image2 = render_image(
            self.cameras[1],
            positions_mm,
            cfg.image_size_px,
            cfg.particle_diameter_px,
            intensities
        )

        return image1, image2

    def generate_image_pair(self, pair_idx: int) -> dict:
        """
        Generate one stereo image pair with particle displacement.

        Parameters
        ----------
        pair_idx : int
            Index of the image pair (used for seeding)

        Returns
        -------
        images : dict
            {
                'cam1_A': ndarray - Camera 1, time A
                'cam2_A': ndarray - Camera 2, time A
                'cam1_B': ndarray - Camera 1, time B
                'cam2_B': ndarray - Camera 2, time B
            }
        """
        cfg = self.config
        base_seed = cfg.seed + pair_idx * 3

        # Generate particles at time A
        pos_A_mm = self.generate_particles(seed=base_seed)
        intensities = self.generate_particle_intensities(len(pos_A_mm), seed=base_seed + 1)

        # Sample displacements
        displacements_px = self.sample_displacements(len(pos_A_mm), seed=base_seed + 2)

        # Convert displacement from pixels to mm
        displacements_mm = displacements_px / cfg.scale_px_per_mm

        # Particles at time B
        pos_B_mm = pos_A_mm + displacements_mm

        # Render images
        cam1_A, cam2_A = self.render_frame(pos_A_mm, intensities)
        cam1_B, cam2_B = self.render_frame(pos_B_mm, intensities)

        return {
            'cam1_A': cam1_A,
            'cam2_A': cam2_A,
            'cam1_B': cam1_B,
            'cam2_B': cam2_B
        }

    def generate_all(self) -> Iterator[dict]:
        """
        Yield all image pairs as a generator (memory efficient).

        Yields
        ------
        images : dict
            Image pair dictionary from generate_image_pair()
        """
        for i in range(self.config.num_image_pairs):
            yield self.generate_image_pair(i)

    def save_images(
        self,
        output_dir: Union[str, Path],
        format: str = 'npz'
    ) -> Path:
        """
        Save all generated images to disk.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save images
        format : str
            'npz' for compressed numpy arrays, 'tiff' for TIFF images

        Returns
        -------
        output_dir : Path
            Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config metadata (always as npz)
        config_data = {
            'volume_size': self.config.volume_size,
            'num_particles': self.config.num_particles,
            'particle_diameter_px': self.config.particle_diameter_px,
            'camera_angles_deg': self.config.camera_angles_deg,
            'working_distance_mm': self.config.working_distance_mm,
            'image_size_px': self.config.image_size_px,
            'scale_px_per_mm': self.config.scale_px_per_mm,
            'reynolds_stress': self.config.reynolds_stress,
            'mean_displacement': self.config.mean_displacement,
            'num_image_pairs': self.config.num_image_pairs,
            'seed': self.config.seed,
            'format': format,
        }
        np.savez(output_dir / 'config.npz', **config_data)

        # Save each image pair
        for i in range(self.config.num_image_pairs):
            images = self.generate_image_pair(i)

            if format == 'tiff':
                import tifffile
                # Save as 16-bit TIFF (normalize to 0-65535)
                for key in ['cam1_A', 'cam2_A', 'cam1_B', 'cam2_B']:
                    img = images[key]
                    # Normalize to 16-bit range
                    img_norm = img / (img.max() + 1e-10) * 65535
                    img_16bit = img_norm.astype(np.uint16)
                    tifffile.imwrite(
                        output_dir / f'pair_{i:05d}_{key}.tiff',
                        img_16bit
                    )
            else:
                # NPZ format
                np.savez_compressed(
                    output_dir / f'pair_{i:05d}.npz',
                    cam1_A=images['cam1_A'].astype(np.float32),
                    cam2_A=images['cam2_A'].astype(np.float32),
                    cam1_B=images['cam1_B'].astype(np.float32),
                    cam2_B=images['cam2_B'].astype(np.float32),
                )

            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1}/{self.config.num_image_pairs} pairs")

        print(f"Saved {self.config.num_image_pairs} image pairs to {output_dir} ({format})")
        return output_dir

    def save_images_range(
        self,
        output_dir: Union[str, Path],
        start_idx: int,
        end_idx: int,
        format: str = 'npz'
    ) -> Path:
        """
        Save a range of image pairs to disk (for generating missing pairs).

        Parameters
        ----------
        output_dir : str or Path
            Directory to save images
        start_idx : int
            First pair index to generate (inclusive)
        end_idx : int
            Last pair index to generate (exclusive)
        format : str
            'npz' for compressed numpy arrays, 'tiff' for TIFF images

        Returns
        -------
        output_dir : Path
            Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Update config metadata if it exists, otherwise create it
        config_data = {
            'volume_size': self.config.volume_size,
            'num_particles': self.config.num_particles,
            'particle_diameter_px': self.config.particle_diameter_px,
            'camera_angles_deg': self.config.camera_angles_deg,
            'working_distance_mm': self.config.working_distance_mm,
            'image_size_px': self.config.image_size_px,
            'scale_px_per_mm': self.config.scale_px_per_mm,
            'reynolds_stress': self.config.reynolds_stress,
            'mean_displacement': self.config.mean_displacement,
            'num_image_pairs': end_idx,  # Update to new total
            'seed': self.config.seed,
            'format': format,
        }
        np.savez(output_dir / 'config.npz', **config_data)

        # Generate and save image pairs in the specified range
        num_to_generate = end_idx - start_idx
        for i in range(start_idx, end_idx):
            images = self.generate_image_pair(i)

            if format == 'tiff':
                import tifffile
                for key in ['cam1_A', 'cam2_A', 'cam1_B', 'cam2_B']:
                    img = images[key]
                    img_norm = img / (img.max() + 1e-10) * 65535
                    img_16bit = img_norm.astype(np.uint16)
                    tifffile.imwrite(
                        output_dir / f'pair_{i:05d}_{key}.tiff',
                        img_16bit
                    )
            else:
                np.savez_compressed(
                    output_dir / f'pair_{i:05d}.npz',
                    cam1_A=images['cam1_A'].astype(np.float32),
                    cam2_A=images['cam2_A'].astype(np.float32),
                    cam1_B=images['cam1_B'].astype(np.float32),
                    cam2_B=images['cam2_B'].astype(np.float32),
                )

            if (i - start_idx + 1) % 100 == 0:
                print(f"  Generated {i - start_idx + 1}/{num_to_generate} new pairs")

        print(f"Generated pairs {start_idx}-{end_idx-1} in {output_dir} ({format})")
        return output_dir

    @staticmethod
    def load_images(input_dir: Union[str, Path]) -> Iterator[dict]:
        """
        Load saved images from disk.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing saved images

        Yields
        ------
        images : dict
            Image pair dictionary with cam1_A, cam2_A, cam1_B, cam2_B
        """
        input_dir = Path(input_dir)

        # Check format from config
        config_file = input_dir / 'config.npz'
        if config_file.exists():
            config = np.load(config_file, allow_pickle=True)
            fmt = str(config.get('format', 'npz'))
        else:
            fmt = 'npz'

        if fmt == 'tiff':
            import tifffile
            # Find all cam1_A tiff files to get pair indices
            cam1_files = sorted(input_dir.glob('pair_*_cam1_A.tiff'))
            if not cam1_files:
                raise FileNotFoundError(f"No TIFF pairs found in {input_dir}")

            for cam1_file in cam1_files:
                prefix = cam1_file.name.replace('_cam1_A.tiff', '')
                yield {
                    'cam1_A': tifffile.imread(
                        input_dir / f'{prefix}_cam1_A.tiff'
                    ).astype(np.float64),
                    'cam2_A': tifffile.imread(
                        input_dir / f'{prefix}_cam2_A.tiff'
                    ).astype(np.float64),
                    'cam1_B': tifffile.imread(
                        input_dir / f'{prefix}_cam1_B.tiff'
                    ).astype(np.float64),
                    'cam2_B': tifffile.imread(
                        input_dir / f'{prefix}_cam2_B.tiff'
                    ).astype(np.float64),
                }
        else:
            # NPZ format
            pair_files = sorted(input_dir.glob('pair_*.npz'))
            if not pair_files:
                raise FileNotFoundError(f"No NPZ pairs found in {input_dir}")

            for pair_file in pair_files:
                data = np.load(pair_file)
                yield {
                    'cam1_A': data['cam1_A'],
                    'cam2_A': data['cam2_A'],
                    'cam1_B': data['cam1_B'],
                    'cam2_B': data['cam2_B'],
                }

    @staticmethod
    def load_config(input_dir: Union[str, Path]) -> dict:
        """
        Load saved configuration from disk.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing saved config

        Returns
        -------
        config : dict
            Configuration dictionary
        """
        input_dir = Path(input_dir)
        config_file = input_dir / 'config.npz'

        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        data = np.load(config_file, allow_pickle=True)
        return {key: data[key] for key in data.files}


# =============================================================================
# TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    print("Testing Stereo Ensemble Generator")
    print("=" * 50)

    # Create config with known Reynolds stress
    R_true = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 0.8, 0.2],
        [0.1, 0.2, 0.5]
    ])

    config = StereoEnsembleConfig(
        reynolds_stress=R_true,
        mean_displacement=np.array([0.0, 0.0, 0.0]),
        num_image_pairs=10,
        num_particles=100,
        seed=42
    )

    print(f"Volume size: {config.volume_size}")
    print(f"Num particles: {config.num_particles}")
    print(f"Camera angles: {config.camera_angles_deg}")
    print(f"\nReynolds stress tensor:")
    print(R_true)

    generator = StereoEnsembleGenerator(config)

    print(f"\nGenerating {config.num_image_pairs} image pairs...")
    for i, images in enumerate(generator.generate_all()):
        if i == 0:
            print(f"  Image shape: {images['cam1_A'].shape}")
            print(f"  cam1_A max: {images['cam1_A'].max():.3f}")
            print(f"  cam2_A max: {images['cam2_A'].max():.3f}")

    print("\nDone!")
