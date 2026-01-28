"""
Test FFT correlation edge effects on sigma estimation.

Pipeline:
1. Generate synthetic particle images (known particle diameter)
2. Extract windows of different sizes from SAME image
3. Compute FFT autocorrelation (AA plane)
4. Fit Gaussian to get sig_A
5. Compare sig_A across window sizes

If sig_A varies with window size → FFT/edge effects cause bias
If sig_A is constant → bias comes from elsewhere

Usage:
    python test_fft_edge_effects.py
"""
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pivtools_cli.piv.piv_backend.gaussian_fitting import (
    _load_marquadt_lib,
    set_offset_fitting,
)


def generate_particle_image(shape, num_particles, particle_diameter, seed=None):
    """
    Generate a synthetic particle image.

    Parameters
    ----------
    shape : tuple
        (height, width) of image
    num_particles : int
        Number of particles to generate
    particle_diameter : float
        Gaussian particle diameter (4*sigma)
    seed : int, optional
        Random seed

    Returns
    -------
    image : np.ndarray
        Particle image (float64)
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    image = np.zeros((h, w), dtype=np.float64)

    # Particle sigma from diameter (d = 4*sigma convention)
    sigma = particle_diameter / 4.0

    # Generate random particle positions
    x_pos = np.random.uniform(0, w, num_particles)
    y_pos = np.random.uniform(0, h, num_particles)
    intensities = np.random.uniform(0.8, 1.0, num_particles)

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[0:h, 0:w]

    # Add each particle as a Gaussian
    for x, y, intensity in zip(x_pos, y_pos, intensities):
        # Compute Gaussian contribution
        r2 = (x_grid - x)**2 + (y_grid - y)**2
        particle = intensity * np.exp(-r2 / (2 * sigma**2))
        image += particle

    return image


def compute_autocorrelation_fft(window):
    """
    Compute autocorrelation using FFT.

    AC(window) = IFFT(|FFT(window)|^2)

    Parameters
    ----------
    window : np.ndarray
        Image window

    Returns
    -------
    ac : np.ndarray
        Autocorrelation plane (centered)
    """
    # Zero-pad to avoid circular correlation artifacts
    h, w = window.shape
    padded = np.zeros((2*h, 2*w), dtype=np.float64)
    padded[:h, :w] = window

    # FFT-based autocorrelation
    F = fft2(padded)
    ac_full = np.real(ifft2(F * np.conj(F)))

    # Extract and center
    ac = fftshift(ac_full[:h, :w])

    return ac


def compute_crosscorrelation_fft(window_a, window_b):
    """
    Compute cross-correlation using FFT.

    CC(A,B) = IFFT(FFT(A) * conj(FFT(B)))
    """
    h, w = window_a.shape
    padded_a = np.zeros((2*h, 2*w), dtype=np.float64)
    padded_b = np.zeros((2*h, 2*w), dtype=np.float64)
    padded_a[:h, :w] = window_a
    padded_b[:h, :w] = window_b

    F_a = fft2(padded_a)
    F_b = fft2(padded_b)
    cc_full = np.real(ifft2(F_a * np.conj(F_b)))

    cc = fftshift(cc_full[:h, :w])

    return cc


def fit_gaussian_2d_simple(plane):
    """
    Simple 2D Gaussian fit using least squares on log of data.

    Returns sigma_x, sigma_y, x0, y0
    """
    h, w = plane.shape

    # Find peak
    peak_idx = np.unravel_index(np.argmax(plane), plane.shape)
    y0, x0 = peak_idx

    # Extract region around peak
    margin = min(8, h//2-1, w//2-1)
    y_slice = slice(max(0, y0-margin), min(h, y0+margin+1))
    x_slice = slice(max(0, x0-margin), min(w, x0+margin+1))

    region = plane[y_slice, x_slice]

    # Create coordinate grids
    y_coords = np.arange(y_slice.start, y_slice.stop)
    x_coords = np.arange(x_slice.start, x_slice.stop)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Take log for linear fit (Gaussian -> parabola in log space)
    # Only use points above threshold
    threshold = 0.1 * np.max(region)
    mask = region > threshold

    if np.sum(mask) < 10:
        return None, None, None, None

    log_region = np.log(np.maximum(region[mask], 1e-10))
    X_flat = X[mask]
    Y_flat = Y[mask]

    # Fit: log(G) = a - b*(x-x0)^2 - c*(y-y0)^2
    # Use iterative refinement

    # Weighted centroid for center
    weights = region[mask]
    x0_fit = np.sum(X_flat * weights) / np.sum(weights)
    y0_fit = np.sum(Y_flat * weights) / np.sum(weights)

    # Compute second moments for sigma
    dx = X_flat - x0_fit
    dy = Y_flat - y0_fit

    sigma_x = np.sqrt(np.sum(weights * dx**2) / np.sum(weights))
    sigma_y = np.sqrt(np.sum(weights * dy**2) / np.sum(weights))

    return sigma_x, sigma_y, x0_fit, y0_fit


def test_autocorrelation_window_size():
    """Test autocorrelation sigma vs window size."""
    print("\n" + "="*80)
    print("AUTOCORRELATION SIGMA vs WINDOW SIZE")
    print("="*80)
    print("\nGenerating large particle image, extracting different window sizes...")
    print("Computing FFT autocorrelation and measuring sig_A")

    # Generate large particle image
    full_size = (512, 512)
    particle_diameter = 2.0  # Same as RS tests
    particles_per_pixel = 0.05  # ~5% fill
    num_particles = int(full_size[0] * full_size[1] * particles_per_pixel)

    print(f"\nImage size: {full_size[0]}x{full_size[1]}")
    print(f"Particle diameter: {particle_diameter} px")
    print(f"Number of particles: {num_particles}")
    print(f"Theoretical sig_A = d_p / sqrt(8) = {particle_diameter / np.sqrt(8):.4f} px")

    image = generate_particle_image(full_size, num_particles, particle_diameter, seed=42)

    # Test different window sizes
    window_sizes = [16, 32, 64, 128, 256]

    print(f"\n{'Window':<10} {'sig_A_x':<12} {'sig_A_y':<12} {'sig_A_mean':<12} {'Theory Ratio':<12}")
    print("-"*60)

    theory_sig_A = particle_diameter / np.sqrt(8)

    results = []
    for win_size in window_sizes:
        # Extract center window
        offset = (full_size[0] - win_size) // 2
        window = image[offset:offset+win_size, offset:offset+win_size]

        # Compute autocorrelation
        ac = compute_autocorrelation_fft(window)

        # Fit Gaussian
        sig_x, sig_y, x0, y0 = fit_gaussian_2d_simple(ac)

        if sig_x is None:
            print(f"{win_size}x{win_size:<7} FAILED")
            continue

        sig_mean = (sig_x + sig_y) / 2
        ratio = sig_mean / theory_sig_A

        print(f"{win_size}x{win_size:<7} {sig_x:<12.4f} {sig_y:<12.4f} {sig_mean:<12.4f} {ratio:<12.2f}x")

        results.append({
            'window': win_size,
            'sig_x': sig_x,
            'sig_y': sig_y,
            'sig_mean': sig_mean,
            'ratio': ratio,
        })

    return results


def test_many_windows_statistics():
    """Test autocorrelation sigma across many random windows."""
    print("\n" + "="*80)
    print("STATISTICAL TEST: Many Random Windows")
    print("="*80)
    print("\nExtracting 100 random windows at each size, computing mean sig_A")

    full_size = (512, 512)
    particle_diameter = 2.0
    num_particles = int(full_size[0] * full_size[1] * 0.05)

    image = generate_particle_image(full_size, num_particles, particle_diameter, seed=42)

    window_sizes = [16, 32, 64]
    n_samples = 100

    np.random.seed(123)

    print(f"\n{'Window':<10} {'Mean sig_A':<12} {'Std sig_A':<12} {'CV%':<10}")
    print("-"*50)

    theory_sig_A = particle_diameter / np.sqrt(8)

    for win_size in window_sizes:
        sig_values = []

        max_offset = full_size[0] - win_size - 1

        for _ in range(n_samples):
            # Random window position
            y_off = np.random.randint(0, max_offset)
            x_off = np.random.randint(0, max_offset)

            window = image[y_off:y_off+win_size, x_off:x_off+win_size]
            ac = compute_autocorrelation_fft(window)

            sig_x, sig_y, _, _ = fit_gaussian_2d_simple(ac)

            if sig_x is not None and sig_y is not None:
                sig_values.append((sig_x + sig_y) / 2)

        if len(sig_values) > 0:
            mean_sig = np.mean(sig_values)
            std_sig = np.std(sig_values)
            cv = 100 * std_sig / mean_sig

            print(f"{win_size}x{win_size:<7} {mean_sig:<12.4f} {std_sig:<12.4f} {cv:<10.2f}")


def test_ensemble_correlation():
    """
    Test ensemble (summed) correlation like real PIV.

    Sum many correlation planes before fitting - this is what ensemble PIV does.
    """
    print("\n" + "="*80)
    print("ENSEMBLE CORRELATION TEST")
    print("="*80)
    print("\nSimulating ensemble PIV: sum N correlation planes, then fit")
    print("This tests whether summing reduces/increases the bias")

    full_size = (256, 256)
    particle_diameter = 2.0
    num_particles = int(full_size[0] * full_size[1] * 0.05)

    window_sizes = [16, 32, 64]
    n_images = 100

    theory_sig_A = particle_diameter / np.sqrt(8)

    print(f"\nTheoretical sig_A = {theory_sig_A:.4f} px")
    print(f"Number of images to sum: {n_images}")

    print(f"\n{'Window':<10} {'Ensemble sig_A':<15} {'Theory Ratio':<12}")
    print("-"*45)

    for win_size in window_sizes:
        # Sum correlation planes from many images
        ac_sum = np.zeros((win_size, win_size), dtype=np.float64)

        for i in range(n_images):
            # Generate new particle image each time
            image = generate_particle_image(full_size, num_particles,
                                           particle_diameter, seed=42+i)

            # Extract center window
            offset = (full_size[0] - win_size) // 2
            window = image[offset:offset+win_size, offset:offset+win_size]

            # Compute and sum autocorrelation
            ac = compute_autocorrelation_fft(window)
            ac_sum += ac

        # Fit summed correlation
        sig_x, sig_y, _, _ = fit_gaussian_2d_simple(ac_sum)

        if sig_x is not None:
            sig_mean = (sig_x + sig_y) / 2
            ratio = sig_mean / theory_sig_A
            print(f"{win_size}x{win_size:<7} {sig_mean:<15.4f} {ratio:<12.2f}x")
        else:
            print(f"{win_size}x{win_size:<7} FAILED")


def test_edge_particle_truncation():
    """
    Test effect of particle truncation at window edges.

    Place a single particle at different positions and measure how
    autocorrelation width changes as particle approaches edge.
    """
    print("\n" + "="*80)
    print("PARTICLE TRUNCATION TEST")
    print("="*80)
    print("\nPlacing single particle at varying distances from window edge")
    print("Measuring autocorrelation width to quantify truncation effect")

    win_size = 32
    particle_diameter = 2.0
    sigma = particle_diameter / 4.0

    # Particle positions relative to center (0 = center, positive = toward edge)
    center = win_size / 2
    offsets = [0, 2, 4, 6, 8, 10, 12, 14]  # pixels from center

    print(f"\nWindow: {win_size}x{win_size}, particle d={particle_diameter} px")
    print(f"Particle sigma = {sigma:.3f} px")

    print(f"\n{'Offset from center':<20} {'Distance to edge':<18} {'Measured sig_A':<15} {'Change%':<10}")
    print("-"*70)

    baseline_sig = None

    for offset in offsets:
        # Create single-particle image
        image = np.zeros((win_size, win_size), dtype=np.float64)

        # Place particle
        x_pos = center + offset
        y_pos = center

        y_grid, x_grid = np.ogrid[0:win_size, 0:win_size]
        r2 = (x_grid - x_pos)**2 + (y_grid - y_pos)**2
        image = np.exp(-r2 / (2 * sigma**2))

        # Compute autocorrelation
        ac = compute_autocorrelation_fft(image)

        sig_x, sig_y, _, _ = fit_gaussian_2d_simple(ac)

        if sig_x is not None:
            sig_mean = (sig_x + sig_y) / 2

            if baseline_sig is None:
                baseline_sig = sig_mean
                change = 0.0
            else:
                change = 100 * (sig_mean - baseline_sig) / baseline_sig

            dist_to_edge = win_size/2 - offset
            print(f"{offset:<20} {dist_to_edge:<18.1f} {sig_mean:<15.4f} {change:+<10.2f}")


def main():
    """Run all FFT edge effect tests."""
    print("="*80)
    print("FFT CORRELATION EDGE EFFECTS TEST")
    print("="*80)
    print("\nThis tests whether FFT-based correlation introduces")
    print("window-size-dependent bias in sigma estimation.")

    # Test 1: Basic window size effect
    results = test_autocorrelation_window_size()

    # Test 2: Statistical test
    test_many_windows_statistics()

    # Test 3: Ensemble correlation
    test_ensemble_correlation()

    # Test 4: Edge truncation
    test_edge_particle_truncation()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        # Check if sig_A varies with window size
        sigs = [r['sig_mean'] for r in results]
        sig_range = max(sigs) - min(sigs)
        sig_mean = np.mean(sigs)
        variation_pct = 100 * sig_range / sig_mean

        print(f"\nsig_A variation across window sizes: {variation_pct:.1f}%")

        if variation_pct > 5:
            print("→ FFT correlation DOES show window-size-dependent sigma")
            print("→ This likely contributes to the ensemble PIV bias")
        else:
            print("→ FFT correlation shows minimal window-size effect")
            print("→ Bias may come from other sources (particle count, etc.)")

    return 0


if __name__ == "__main__":
    exit(main())
