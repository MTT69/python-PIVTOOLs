"""
Consolidated synthetic PIV image generator.

All test particle image generators consolidated into one module using
local-patch Gaussian rendering: evaluate exp() only within ±3σ of each
particle center. This is ~2600× faster than full-image analytic rendering
while being sub-pixel accurate (unlike stamp+blur which rounds to nearest
pixel before blurring).

Usage:
    from synthetic_piv import generate_particle_image, generate_displaced_pair
    from synthetic_piv import generate_rs_test_images, generate_gap_verification_images
"""
import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, Tuple, List


def render_particles(shape, x_pos, y_pos, intensities, sigma,
                     dtype=np.float32, cutoff=3.0):
    """
    Render particles as Gaussians using local-patch evaluation.

    For each particle, evaluates exp(-r²/2σ²) only within ±cutoff*σ of
    the particle center. For typical σ=0.85 and cutoff=3.0, the patch
    is ~5×5 pixels. At 3σ the Gaussian value is exp(-4.5) ≈ 0.011.

    Parameters
    ----------
    shape : tuple
        (height, width) of the output image.
    x_pos, y_pos : array-like
        Particle center coordinates (sub-pixel float).
    intensities : array-like
        Peak intensity per particle.
    sigma : float
        Gaussian standard deviation in pixels.
    dtype : numpy dtype
        Output array dtype.
    cutoff : float
        Evaluate Gaussian within ±cutoff*sigma of center.

    Returns
    -------
    image : ndarray of shape `shape`
    """
    h, w = shape
    image = np.zeros(shape, dtype=np.float64)
    radius = int(np.ceil(cutoff * sigma))
    two_sigma2 = 2.0 * sigma * sigma

    for x, y, intensity in zip(x_pos, y_pos, intensities):
        # Bounding box clipped to image
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        # Local coordinate grids (sparse)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        image[y0:y1, x0:x1] += intensity * np.exp(-r2 / two_sigma2)

    return image.astype(dtype)


# ============================================================================
# In-memory generators (return ndarray, no disk I/O)
# ============================================================================

def generate_particle_image(shape, num_particles, particle_diameter, seed=None,
                            dtype=np.float32):
    """
    Generate a synthetic particle image.

    Parameters
    ----------
    shape : tuple
        (height, width) of image.
    num_particles : int
        Number of particles.
    particle_diameter : float
        Gaussian particle diameter (4*sigma convention).
    seed : int, optional
        Random seed.
    dtype : numpy dtype
        Output dtype (default float32).

    Returns
    -------
    image : ndarray
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    sigma = particle_diameter / 4.0

    x_pos = rng.uniform(0, w, num_particles)
    y_pos = rng.uniform(0, h, num_particles)
    intensities = rng.uniform(0.8, 1.0, num_particles)

    return render_particles(shape, x_pos, y_pos, intensities, sigma, dtype=dtype)


def generate_displaced_pair(shape, num_particles, particle_diameter, dx, dy,
                            seed=None, dtype=np.float32):
    """
    Generate a pair of particle images with known displacement.

    Parameters
    ----------
    shape : tuple
        (height, width).
    num_particles : int
        Number of particles.
    particle_diameter : float
        Gaussian particle diameter (4*sigma convention).
    dx, dy : float
        Displacement in pixels (positive dy = downward in image).
    seed : int, optional
        Random seed.
    dtype : numpy dtype
        Output dtype (default float32).

    Returns
    -------
    img_a, img_b : tuple of ndarray
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    sigma = particle_diameter / 4.0

    x_pos = rng.uniform(particle_diameter, w - particle_diameter, num_particles)
    y_pos = rng.uniform(particle_diameter, h - particle_diameter, num_particles)
    intensities = rng.uniform(0.8, 1.0, num_particles)

    img_a = render_particles(shape, x_pos, y_pos, intensities, sigma, dtype=dtype)
    img_b = render_particles(shape, x_pos + dx, y_pos + dy, intensities, sigma,
                             dtype=dtype)

    return img_a, img_b


def generate_particles(size, n_particles=200, sigma=1.5):
    """
    Generate a particle image for background method testing.

    Parameters
    ----------
    size : int
        Square image dimension (size × size).
    n_particles : int
        Number of particles.
    sigma : float
        Gaussian standard deviation in pixels.

    Returns
    -------
    image : ndarray (float64)
    """
    rng = np.random.default_rng()
    xy = rng.random((n_particles, 2)) * size
    x_pos = xy[:, 0]
    y_pos = xy[:, 1]
    intensities = np.ones(n_particles)

    return render_particles((size, size), x_pos, y_pos, intensities, sigma,
                            dtype=np.float64)


# ============================================================================
# Disk-writing generators (write TIFF uint16, return stats dict)
# ============================================================================

def _normalize_to_uint16(img):
    """Normalize image to uint16 range."""
    if img.max() > 0:
        return (img / img.max() * 65535).astype(np.uint16)
    return img.astype(np.uint16)


def generate_rs_test_images(
    output_dir: Path,
    num_pairs: int = 100,
    image_shape: Tuple[int, int] = (256, 256),
    num_particles: int = 2000,
    particle_diameter: float = 2.0,
    mean_dx: float = 0.0,
    mean_dy: float = 0.0,
    std_dx: float = 1.0,
    std_dy: float = 1.0,
    seed: int = 42,
    group_means: Optional[List[Tuple[float, float]]] = None,
    verbose: bool = True,
):
    """
    Generate image pairs with specified mean displacement and displacement variance.

    The displacement variance across pairs creates the Reynolds stress signal.
    Each pair has particles that ALL move by the SAME random displacement (drawn
    from N(mean, std)), so the cross-correlation peak for that pair is sharp,
    but the ENSEMBLE average has a broad peak due to pair-to-pair variation.

    Uses FWHM sigma convention: sigma = particle_diameter / 2.355.

    Parameters
    ----------
    output_dir : Path
        Directory to save images.
    num_pairs : int
        Number of image pairs.
    image_shape : tuple
        (height, width) in pixels.
    num_particles : int
        Particles per image.
    particle_diameter : float
        Particle size in pixels (FWHM).
    mean_dx, mean_dy : float
        Mean displacement (positive dy = upward in physical coords).
    std_dx, std_dy : float
        Std dev of displacement (creates Reynolds stress).
    seed : int
        Random seed.
    group_means : list of (dx, dy) tuples, optional
        Per-group mean displacements. Pairs divided equally among groups.
    verbose : bool
        Print progress.

    Returns
    -------
    stats : dict
        Actual displacement statistics.
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355  # FWHM to Gaussian sigma

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dx = []
    all_dy = []

    for pair_idx in range(1, num_pairs + 1):
        # Determine mean for this pair (handle group means)
        if group_means is not None:
            num_groups = len(group_means)
            pairs_per_group = num_pairs // num_groups
            group_idx = (pair_idx - 1) // pairs_per_group
            group_idx = min(group_idx, num_groups - 1)
            pair_mean_dx, pair_mean_dy = group_means[group_idx]
        else:
            pair_mean_dx, pair_mean_dy = mean_dx, mean_dy

        dx = rng.normal(pair_mean_dx, std_dx)
        dy = rng.normal(pair_mean_dy, std_dy)
        all_dx.append(dx)
        all_dy.append(dy)

        # Random particle positions with margin
        margin = max(20, abs(dy) + 3 * std_dy, abs(dx) + 3 * std_dx)
        x_pos = rng.uniform(margin, W - margin, num_particles)
        y_pos = rng.uniform(margin, H - margin, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A: original positions
        img_a = render_particles(image_shape, x_pos, y_pos, intensities, sigma)

        # Frame B: displaced (positive dy = upward = lower row index)
        img_b = render_particles(image_shape, x_pos + dx, y_pos - dy,
                                 intensities, sigma)

        # Save as uint16 TIFF
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif",
                         _normalize_to_uint16(img_a))
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif",
                         _normalize_to_uint16(img_b))

        if verbose and pair_idx % 25 == 0:
            print(f"  Generated {pair_idx}/{num_pairs} pairs")

    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)

    stats = {
        'num_pairs': num_pairs,
        'actual_mean_dx': np.mean(all_dx),
        'actual_mean_dy': np.mean(all_dy),
        'actual_std_dx': np.std(all_dx),
        'actual_std_dy': np.std(all_dy),
        'actual_var_dx': np.var(all_dx),
        'actual_var_dy': np.var(all_dy),
        'target_mean_dx': mean_dx,
        'target_mean_dy': mean_dy,
        'target_std_dx': std_dx,
        'target_std_dy': std_dy,
        'particle_diameter': particle_diameter,
        'image_shape': image_shape,
    }

    if verbose:
        print(f"\nGenerated {num_pairs} image pairs in {output_dir}")
        print(f"  Image size: {H}x{W}, Particle diameter: {particle_diameter}px")
        print(f"  Target mean: dx={mean_dx:.2f}, dy={mean_dy:.2f}")
        print(f"  Actual mean: dx={stats['actual_mean_dx']:.3f}, "
              f"dy={stats['actual_mean_dy']:.3f}")
        print(f"  Target std:  dx={std_dx:.2f}, dy={std_dy:.2f}")
        print(f"  Actual std:  dx={stats['actual_std_dx']:.3f}, "
              f"dy={stats['actual_std_dy']:.3f}")
        print(f"  Expected RS: UU~{std_dx**2:.2f}, VV~{std_dy**2:.2f}")

    return stats


def generate_spatially_varying_rs_images(
    output_dir: Path,
    num_pairs: int = 100,
    image_shape: Tuple[int, int] = (256, 256),
    num_particles: int = 2000,
    particle_diameter: float = 2.0,
    mean_dx: float = 0.0,
    mean_dy: float = 0.0,
    base_std_dx: float = 1.0,
    base_std_dy: float = 2.0,
    gradient_scale: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Generate images with SPATIALLY VARYING Reynolds stress.

    Displacement variance increases from bottom to top of the image.
    At bottom (y=0): RS = base_std^2. At top (y=H-1): RS = (gradient_scale * base_std)^2.

    Uses per-particle displacement (not per-pair), so each particle gets
    its own random displacement based on its local RS.

    Uses FWHM sigma convention: sigma = particle_diameter / 2.355.
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_dx, top_dy = [], []
    bot_dx, bot_dy = [], []

    for pair_idx in range(1, num_pairs + 1):
        margin = 20
        x_pos = rng.uniform(margin, W - margin, num_particles)
        y_pos = rng.uniform(margin, H - margin, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A
        img_a = render_particles(image_shape, x_pos, y_pos, intensities, sigma)

        # Frame B: spatially varying displacement per particle
        new_x = np.empty_like(x_pos)
        new_y = np.empty_like(y_pos)
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            physical_y = (H - 1) - y
            frac = physical_y / (H - 1)
            local_scale = 1.0 + frac * (gradient_scale - 1.0)
            local_std_dx = base_std_dx * local_scale
            local_std_dy = base_std_dy * local_scale

            dx = rng.normal(mean_dx, local_std_dx)
            dy = rng.normal(mean_dy, local_std_dy)

            if frac > 0.7:
                top_dx.append(dx)
                top_dy.append(dy)
            elif frac < 0.3:
                bot_dx.append(dx)
                bot_dy.append(dy)

            new_x[i] = x + dx
            new_y[i] = y - dy

        img_b = render_particles(image_shape, new_x, new_y, intensities, sigma)

        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif",
                         _normalize_to_uint16(img_a))
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif",
                         _normalize_to_uint16(img_b))

        if verbose and pair_idx % 25 == 0:
            print(f"  Generated {pair_idx}/{num_pairs} pairs")

    stats = {
        'num_pairs': num_pairs,
        'top_var_dx': np.var(top_dx) if top_dx else 0,
        'top_var_dy': np.var(top_dy) if top_dy else 0,
        'bot_var_dx': np.var(bot_dx) if bot_dx else 0,
        'bot_var_dy': np.var(bot_dy) if bot_dy else 0,
        'expected_bot_var_dx': base_std_dx**2,
        'expected_bot_var_dy': base_std_dy**2,
        'expected_top_var_dx': (base_std_dx * gradient_scale)**2,
        'expected_top_var_dy': (base_std_dy * gradient_scale)**2,
    }

    if verbose:
        print(f"\nGenerated {num_pairs} spatially varying RS images in {output_dir}")
        print(f"  Bottom RS (expected): UU={stats['expected_bot_var_dx']:.2f}, "
              f"VV={stats['expected_bot_var_dy']:.2f}")
        print(f"  Top RS (expected):    UU={stats['expected_top_var_dx']:.2f}, "
              f"VV={stats['expected_top_var_dy']:.2f}")
        print(f"  Bottom RS (actual):   UU={stats['bot_var_dx']:.2f}, "
              f"VV={stats['bot_var_dy']:.2f}")
        print(f"  Top RS (actual):      UU={stats['top_var_dx']:.2f}, "
              f"VV={stats['top_var_dy']:.2f}")

    return stats


def generate_gap_verification_images(
    output_dir: Path,
    num_pairs: int = 5,
    image_shape: tuple = (500, 500),
    num_particles: int = 3000,
    particle_diameter: float = 4.0,
    seed: int = 42,
):
    """
    Generate images with distinct velocity regions to verify gap placement.

    Velocity scheme:
    - GAP region (pixel y < 60): particles move DOWN (uy = -10 physical)
    - Measured region (pixel y >= 60): particles move UP (uy = +5 physical)

    Uses FWHM sigma convention: sigma = particle_diameter / 2.355.

    Returns
    -------
    stats : dict
        With gap_velocity, measured_velocity, gap_boundary, image_shape.
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355

    gap_boundary_pixel = 60
    gap_velocity_pixel = +10       # DOWN in image
    measured_velocity_pixel = -5   # UP in image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx in range(1, num_pairs + 1):
        x_pos = rng.uniform(20, W - 20, num_particles)
        y_pos_pixel = rng.uniform(20, H - 20, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A
        img_a = render_particles(image_shape, x_pos, y_pos_pixel, intensities,
                                 sigma)

        # Frame B: region-dependent displacement
        dy_pixel = np.where(y_pos_pixel < gap_boundary_pixel,
                            gap_velocity_pixel, measured_velocity_pixel)
        img_b = render_particles(image_shape, x_pos, y_pos_pixel + dy_pixel,
                                 intensities, sigma)

        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif",
                         _normalize_to_uint16(img_a))
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif",
                         _normalize_to_uint16(img_b))

    return {
        'gap_velocity': -gap_velocity_pixel,       # Physical uy
        'measured_velocity': -measured_velocity_pixel,  # Physical uy
        'gap_boundary': gap_boundary_pixel,
        'image_shape': image_shape,
    }


if __name__ == "__main__":
    # Quick smoke test
    print("Testing render_particles...")
    img = render_particles((64, 64),
                           np.array([32.5]), np.array([32.5]),
                           np.array([1.0]), sigma=0.85)
    print(f"  Peak value: {img.max():.4f} (expected ~1.0)")
    print(f"  Sum: {img.sum():.4f} (expected ~{2*np.pi*0.85**2:.4f})")

    print("\nTesting generate_particle_image...")
    img = generate_particle_image((128, 128), 100, 2.0, seed=42)
    print(f"  Shape: {img.shape}, dtype: {img.dtype}, max: {img.max():.3f}")

    print("\nTesting generate_displaced_pair...")
    a, b = generate_displaced_pair((128, 128), 100, 2.0, 3.0, 1.5, seed=42)
    print(f"  Shapes: {a.shape}, {b.shape}")

    print("\nAll tests passed.")
