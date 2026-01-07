"""
Generate synthetic PIV images with controlled Reynolds stress (displacement variance).

Reynolds stress in ensemble PIV is measured from the WIDTH of the cross-correlation peak.
When particles have random displacement fluctuations around a mean, the correlation peak
broadens. The Gaussian fit extracts this broadening as sig_AB_x, sig_AB_y.

Theory:
- Mean displacement: determines peak POSITION
- Displacement variance: determines peak WIDTH (Reynolds stress)
- sig_AB^2 = sig_particle^2 + displacement_variance
- RS = sig_AB^2 - sig_A^2 (subtract particle contribution)
"""
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List


def generate_rs_test_images(
    output_dir: Path,
    num_pairs: int = 100,
    image_shape: Tuple[int, int] = (256, 256),
    num_particles: int = 2000,
    particle_diameter: float = 2.0,
    mean_dx: float = 0.0,
    mean_dy: float = 0.0,
    std_dx: float = 1.0,  # Reynolds stress in x (pixels)
    std_dy: float = 1.0,  # Reynolds stress in y (pixels)
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

    Args:
        output_dir: Directory to save images
        num_pairs: Number of image pairs
        image_shape: (height, width) in pixels
        num_particles: Particles per image
        particle_diameter: Particle size in pixels (FWHM)
        mean_dx: Mean x displacement (pixels)
        mean_dy: Mean y displacement (pixels, positive = upward in physical coords)
        std_dx: Std dev of x displacement (creates UU Reynolds stress)
        std_dy: Std dev of y displacement (creates VV Reynolds stress)
        seed: Random seed
        group_means: If provided, list of (mean_dx, mean_dy) for each group.
                     Pairs are divided equally among groups.
                     Overall mean should be (mean_dx, mean_dy).
        verbose: Print progress

    Returns:
        dict with actual statistics of generated displacements
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355  # FWHM to Gaussian sigma

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track actual displacements for verification
    all_dx = []
    all_dy = []

    for pair_idx in range(1, num_pairs + 1):
        # Determine mean for this pair (handle group means)
        if group_means is not None:
            num_groups = len(group_means)
            pairs_per_group = num_pairs // num_groups
            group_idx = (pair_idx - 1) // pairs_per_group
            group_idx = min(group_idx, num_groups - 1)  # Handle remainder
            pair_mean_dx, pair_mean_dy = group_means[group_idx]
        else:
            pair_mean_dx, pair_mean_dy = mean_dx, mean_dy

        # Draw displacement for this pair from N(mean, std)
        # This creates the Reynolds stress signal in the ensemble
        dx = rng.normal(pair_mean_dx, std_dx)
        dy = rng.normal(pair_mean_dy, std_dy)

        all_dx.append(dx)
        all_dy.append(dy)

        # Random particle positions (in image coordinates)
        # Leave margin for displacement
        margin = max(20, abs(dy) + 3 * std_dy, abs(dx) + 3 * std_dx)
        x_pos = rng.uniform(margin, W - margin, num_particles)
        y_pos = rng.uniform(margin, H - margin, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A: particles at original positions
        img_a = np.zeros(image_shape, dtype=np.float32)
        for x, y, intensity in zip(x_pos, y_pos, intensities):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < H and 0 <= xi < W:
                img_a[yi, xi] += intensity
        img_a = gaussian_filter(img_a, sigma)

        # Frame B: particles displaced
        # dy in image coords: negative = upward (lower row index)
        # Convention: positive dy = upward in physical = negative row change
        img_b = np.zeros(image_shape, dtype=np.float32)
        for x, y, intensity in zip(x_pos, y_pos, intensities):
            new_x = x + dx
            new_y = y - dy  # Upward motion = lower row index
            yi, xi = int(round(new_y)), int(round(new_x))
            if 0 <= yi < H and 0 <= xi < W:
                img_b[yi, xi] += intensity
        img_b = gaussian_filter(img_b, sigma)

        # Normalize to 16-bit
        if img_a.max() > 0:
            img_a = (img_a / img_a.max() * 65535).astype(np.uint16)
        else:
            img_a = img_a.astype(np.uint16)
        if img_b.max() > 0:
            img_b = (img_b / img_b.max() * 65535).astype(np.uint16)
        else:
            img_b = img_b.astype(np.uint16)

        # Save
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif", img_a)
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif", img_b)

        if verbose and pair_idx % 25 == 0:
            print(f"  Generated {pair_idx}/{num_pairs} pairs")

    # Compute actual statistics
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
        print(f"  Actual mean: dx={stats['actual_mean_dx']:.3f}, dy={stats['actual_mean_dy']:.3f}")
        print(f"  Target std:  dx={std_dx:.2f}, dy={std_dy:.2f}")
        print(f"  Actual std:  dx={stats['actual_std_dx']:.3f}, dy={stats['actual_std_dy']:.3f}")
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
    gradient_scale: float = 2.0,  # RS increases by this factor from bottom to top
    seed: int = 42,
    verbose: bool = True,
):
    """
    Generate images with SPATIALLY VARYING Reynolds stress.

    The displacement variance increases from bottom to top of the image.
    This tests whether the per-pass drop is related to spatial smoothing.

    At bottom (y=0): RS = base_std^2
    At top (y=H-1): RS = (gradient_scale * base_std)^2

    Args:
        gradient_scale: Factor by which RS increases from bottom to top
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    sigma = particle_diameter / 2.355

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track displacements by region
    top_dx, top_dy = [], []
    bot_dx, bot_dy = [], []

    for pair_idx in range(1, num_pairs + 1):
        # Random particle positions
        margin = 20
        x_pos = rng.uniform(margin, W - margin, num_particles)
        y_pos = rng.uniform(margin, H - margin, num_particles)
        intensities = rng.uniform(200, 255, num_particles)

        # Frame A
        img_a = np.zeros(image_shape, dtype=np.float32)
        for x, y, intensity in zip(x_pos, y_pos, intensities):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < H and 0 <= xi < W:
                img_a[yi, xi] += intensity
        img_a = gaussian_filter(img_a, sigma)

        # Frame B: spatially varying displacement variance
        img_b = np.zeros(image_shape, dtype=np.float32)
        for x, y, intensity in zip(x_pos, y_pos, intensities):
            # Physical y: 0 at bottom, H-1 at top
            physical_y = (H - 1) - y
            frac = physical_y / (H - 1)  # 0 at bottom, 1 at top

            # Local std scales with position
            local_scale = 1.0 + frac * (gradient_scale - 1.0)
            local_std_dx = base_std_dx * local_scale
            local_std_dy = base_std_dy * local_scale

            # Draw displacement for this particle
            dx = rng.normal(mean_dx, local_std_dx)
            dy = rng.normal(mean_dy, local_std_dy)

            # Track by region
            if frac > 0.7:
                top_dx.append(dx)
                top_dy.append(dy)
            elif frac < 0.3:
                bot_dx.append(dx)
                bot_dy.append(dy)

            new_x = x + dx
            new_y = y - dy
            yi, xi = int(round(new_y)), int(round(new_x))
            if 0 <= yi < H and 0 <= xi < W:
                img_b[yi, xi] += intensity
        img_b = gaussian_filter(img_b, sigma)

        # Normalize
        if img_a.max() > 0:
            img_a = (img_a / img_a.max() * 65535).astype(np.uint16)
        else:
            img_a = img_a.astype(np.uint16)
        if img_b.max() > 0:
            img_b = (img_b / img_b.max() * 65535).astype(np.uint16)
        else:
            img_b = img_b.astype(np.uint16)

        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_A.tif", img_a)
        tifffile.imwrite(output_dir / f"B{pair_idx:05d}_B.tif", img_b)

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
        print(f"  Bottom RS (expected): UU={stats['expected_bot_var_dx']:.2f}, VV={stats['expected_bot_var_dy']:.2f}")
        print(f"  Top RS (expected):    UU={stats['expected_top_var_dx']:.2f}, VV={stats['expected_top_var_dy']:.2f}")
        print(f"  Bottom RS (actual):   UU={stats['bot_var_dx']:.2f}, VV={stats['bot_var_dy']:.2f}")
        print(f"  Top RS (actual):      UU={stats['top_var_dx']:.2f}, VV={stats['top_var_dy']:.2f}")

    return stats


if __name__ == "__main__":
    # Test 1: 4 groups with different means, combined mean=0
    print("=" * 60)
    print("TEST 1: Four mean groups, RS=1,2")
    print("=" * 60)
    group_means = [
        (-1.0, -1.0),  # pairs 1-25
        (1.0, 1.0),    # pairs 26-50
        (1.0, 0.0),    # pairs 51-75
        (-1.0, 0.0),   # pairs 76-100
    ]
    # Combined mean: (-1+1+1-1)/4=0, (-1+1+0+0)/4=0
    generate_rs_test_images(
        output_dir=Path("rs_test1/Cam1"),
        num_pairs=100,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=1.0,  # Target RS UU = 1
        std_dy=np.sqrt(2.0),  # Target RS VV = 2
        group_means=group_means,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("TEST 2: Zero mean, RS=2,3")
    print("=" * 60)
    generate_rs_test_images(
        output_dir=Path("rs_test2/Cam1"),
        num_pairs=100,
        image_shape=(256, 256),
        particle_diameter=2.0,
        mean_dx=0.0,
        mean_dy=0.0,
        std_dx=np.sqrt(2.0),  # Target RS UU = 2
        std_dy=np.sqrt(3.0),  # Target RS VV = 3
        seed=123,
    )

    print("\n" + "=" * 60)
    print("TEST 3: Spatially varying RS")
    print("=" * 60)
    generate_spatially_varying_rs_images(
        output_dir=Path("rs_test3_spatial/Cam1"),
        num_pairs=100,
        image_shape=(256, 256),
        particle_diameter=2.0,
        base_std_dx=1.0,
        base_std_dy=np.sqrt(2.0),
        gradient_scale=2.0,  # Top has 4x the RS of bottom
        seed=456,
    )
