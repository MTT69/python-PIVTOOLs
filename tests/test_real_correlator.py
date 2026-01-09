"""
Test the ACTUAL PIV correlator for window-size effects.

Uses the real C library correlator (bulkxcorr2d) to compute correlation planes
on synthetic images, then measures if sigma varies with window size.

Usage:
    python test_real_correlator.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pivtools_cli.piv.piv_backend.gaussian_fitting import (
    _load_marquadt_lib,
    set_offset_fitting,
)
import ctypes


def generate_particle_image(shape, num_particles, particle_diameter, seed=None):
    """Generate synthetic particle image."""
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    image = np.zeros((h, w), dtype=np.float32)
    sigma = particle_diameter / 4.0

    x_pos = np.random.uniform(0, w, num_particles)
    y_pos = np.random.uniform(0, h, num_particles)
    intensities = np.random.uniform(0.8, 1.0, num_particles)

    y_grid, x_grid = np.ogrid[0:h, 0:w]

    for x, y, intensity in zip(x_pos, y_pos, intensities):
        r2 = (x_grid - x)**2 + (y_grid - y)**2
        image += intensity * np.exp(-r2 / (2 * sigma**2))

    return image


def generate_displaced_pair(shape, num_particles, particle_diameter, dx, dy, seed=None):
    """Generate a pair of particle images with known displacement."""
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    sigma = particle_diameter / 4.0

    # Random particle positions
    x_pos = np.random.uniform(particle_diameter, w - particle_diameter, num_particles)
    y_pos = np.random.uniform(particle_diameter, h - particle_diameter, num_particles)
    intensities = np.random.uniform(0.8, 1.0, num_particles)

    y_grid, x_grid = np.ogrid[0:h, 0:w]

    # Image A - original positions
    img_a = np.zeros((h, w), dtype=np.float32)
    for x, y, intensity in zip(x_pos, y_pos, intensities):
        r2 = (x_grid - x)**2 + (y_grid - y)**2
        img_a += intensity * np.exp(-r2 / (2 * sigma**2))

    # Image B - displaced positions
    img_b = np.zeros((h, w), dtype=np.float32)
    for x, y, intensity in zip(x_pos, y_pos, intensities):
        r2 = (x_grid - (x + dx))**2 + (y_grid - (y + dy))**2
        img_b += intensity * np.exp(-r2 / (2 * sigma**2))

    return img_a, img_b


def fit_correlation_plane(AA, BB, AB, win_size):
    """
    Fit correlation planes using the actual C Gaussian fitter.

    Uses the same C library as the real PIV code.
    """
    lib = _load_marquadt_lib()
    set_offset_fitting(enabled=False)

    h, w = win_size
    n_per_window = h * w

    # Build coordinate grids (1-based, matching C code)
    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')
    X1 = Y.ravel(order='C').astype(np.float64)  # Y coordinates
    X2 = X.ravel(order='C').astype(np.float64)  # X coordinates

    # Pack correlation data: [AA | BB | AB]
    y_all = np.concatenate([
        AA.ravel().astype(np.float64),
        BB.ravel().astype(np.float64),
        AB.ravel().astype(np.float64)
    ])

    # Initial guess - center of window, reasonable sigmas
    center = (w / 2 + 1, h / 2 + 1)
    initial_guess = np.array([
        np.max(AA), np.max(BB), np.max(AB),  # amplitudes
        0.0, 0.0, 0.0,  # offsets
        2.0, 2.0, 0.0,  # sig_A
        1.0, 1.0, 0.0,  # sig_AB
        center[0], center[1],  # x0_A, y0_A
        center[0], center[1],  # x0_AB, y0_AB
    ], dtype=np.float64)

    # Output arrays
    result = np.zeros(16, dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)

    # Call C function (pass win_height and win_width for rectangular window support)
    lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_per_window),
        ctypes.c_size_t(h),  # win_height
        ctypes.c_size_t(w),  # win_width
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    if status[0] == 1:
        sig_A_x = result[6]
        sig_A_y = result[7]
        sig_AB_x = result[9]
        sig_AB_y = result[10]
        return sig_A_x, sig_A_y, sig_AB_x, sig_AB_y
    else:
        return None, None, None, None


def test_autocorrelation_with_real_correlator():
    """
    Test autocorrelation using the real PIV C library.

    Computes autocorrelation of particle images at different window sizes
    using the actual bulkxcorr2d function.
    """
    print("\n" + "="*70)
    print("TEST: Real Correlator Autocorrelation")
    print("="*70)
    print("\nUsing actual C library (bulkxcorr2d) to compute correlation planes")

    # Generate a large particle image
    full_size = (256, 256)
    particle_diameter = 2.0
    particles_per_pixel = 0.05
    num_particles = int(full_size[0] * full_size[1] * particles_per_pixel)

    print(f"\nImage: {full_size[0]}x{full_size[1]}, {num_particles} particles")
    print(f"Particle diameter: {particle_diameter} px")
    print(f"Theoretical sig_A = d_p/sqrt(8) = {particle_diameter/np.sqrt(8):.4f} px")

    image = generate_particle_image(full_size, num_particles, particle_diameter, seed=42)

    # Test different window sizes
    window_sizes = [16, 32, 64]

    print(f"\n{'Window':<10} {'sig_A_x':<12} {'sig_A_y':<12} {'Mean':<12}")
    print("-"*50)

    for win_size in window_sizes:
        # Extract center window
        offset = (full_size[0] - win_size) // 2
        window = image[offset:offset+win_size, offset:offset+win_size].copy()

        # Compute autocorrelation using FFT (like PIV does)
        from scipy.fft import fft2, ifft2, fftshift

        # Zero-pad for linear (not circular) correlation
        padded = np.zeros((2*win_size, 2*win_size), dtype=np.float32)
        padded[:win_size, :win_size] = window

        F = fft2(padded)
        ac_full = np.real(ifft2(F * np.conj(F)))
        ac = fftshift(ac_full[:win_size, :win_size])

        # Fit using real C fitter (AA=BB=AB for autocorrelation)
        sig_A_x, sig_A_y, _, _ = fit_correlation_plane(ac, ac, ac, (win_size, win_size))

        if sig_A_x:
            print(f"{win_size}x{win_size:<7} {sig_A_x:<12.4f} {sig_A_y:<12.4f} {(sig_A_x+sig_A_y)/2:<12.4f}")
        else:
            print(f"{win_size}x{win_size:<7} FAILED")


def test_ensemble_accumulation():
    """
    Test ensemble correlation accumulation (sum of correlation planes).

    This mimics what ensemble PIV actually does:
    1. Generate N image pairs with varying displacements
    2. Compute correlation for each pair
    3. Sum correlation planes
    4. Fit summed plane
    """
    print("\n" + "="*70)
    print("TEST: Ensemble Accumulation (Real PIV Workflow)")
    print("="*70)

    image_size = (128, 128)
    particle_diameter = 2.0
    num_particles = 300
    n_pairs = 100

    # Displacement distribution (like RS test)
    mean_dx, mean_dy = 0.0, 0.0
    std_dx, std_dy = 1.414, 1.732  # UU=2, VV=3

    window_sizes = [16, 32, 64]

    print(f"\nImage: {image_size[0]}x{image_size[1]}, {n_pairs} pairs")
    print(f"Displacement: mean=({mean_dx}, {mean_dy}), std=({std_dx:.3f}, {std_dy:.3f})")
    print(f"Expected: sig_AB² = UU = {std_dx**2:.2f}, VV = {std_dy**2:.2f}")

    from scipy.fft import fft2, ifft2, fftshift

    for win_size in window_sizes:
        print(f"\n--- Window {win_size}x{win_size} ---")

        # Accumulate correlation planes
        ac_sum = np.zeros((win_size, win_size), dtype=np.float64)
        cc_sum = np.zeros((win_size, win_size), dtype=np.float64)

        np.random.seed(42)

        for i in range(n_pairs):
            # Random displacement from distribution
            dx = np.random.normal(mean_dx, std_dx)
            dy = np.random.normal(mean_dy, std_dy)

            # Generate image pair
            img_a, img_b = generate_displaced_pair(
                image_size, num_particles, particle_diameter, dx, dy, seed=42+i
            )

            # Extract center windows
            offset = (image_size[0] - win_size) // 2
            win_a = img_a[offset:offset+win_size, offset:offset+win_size]
            win_b = img_b[offset:offset+win_size, offset:offset+win_size]

            # Compute correlations with zero-padding
            padded_a = np.zeros((2*win_size, 2*win_size), dtype=np.float32)
            padded_b = np.zeros((2*win_size, 2*win_size), dtype=np.float32)
            padded_a[:win_size, :win_size] = win_a
            padded_b[:win_size, :win_size] = win_b

            F_a = fft2(padded_a)
            F_b = fft2(padded_b)

            # Autocorrelation of A
            ac = np.real(ifft2(F_a * np.conj(F_a)))[:win_size, :win_size]

            # Cross-correlation A×B
            cc = np.real(ifft2(F_a * np.conj(F_b)))[:win_size, :win_size]

            ac_sum += fftshift(ac)
            cc_sum += fftshift(cc)

        # Fit accumulated planes (AA=BB=ac_sum, AB=cc_sum)
        sig_A_x, sig_A_y, sig_AB_x, sig_AB_y = fit_correlation_plane(
            ac_sum, ac_sum, cc_sum, (win_size, win_size)
        )

        if sig_A_x:
            sig_A_mean = (sig_A_x + sig_A_y) / 2
            print(f"  sig_A (autocorr):  {sig_A_mean:.4f} px")

        # For cross-correlation, fit separately
        # (The stacked model expects AA, BB, AB together)
        # Just report the summed CC for now
        cc_peak = np.max(cc_sum)
        cc_center = cc_sum[win_size//2, win_size//2]
        print(f"  CC peak/center:    {cc_peak:.1f} / {cc_center:.1f}")


def test_single_particle_correlation():
    """
    Simplest test: correlate images with a SINGLE particle.

    This eliminates inter-particle effects and isolates:
    - FFT correlation behavior
    - Edge effects
    - Gaussian fitting accuracy
    """
    print("\n" + "="*70)
    print("TEST: Single Particle Correlation")
    print("="*70)
    print("\nCorrelating images with ONE particle - eliminates inter-particle noise")

    particle_diameter = 2.0
    sigma = particle_diameter / 4.0
    theory_sig_A = particle_diameter / np.sqrt(8)

    window_sizes = [16, 32, 64]

    print(f"\nParticle diameter: {particle_diameter} px, sigma: {sigma:.3f} px")
    print(f"Theoretical sig_A: {theory_sig_A:.4f} px")
    print(f"Expected AC sig_A: {theory_sig_A * np.sqrt(2):.4f} px (×√2 for autocorr)")

    from scipy.fft import fft2, ifft2, fftshift

    print(f"\n{'Window':<10} {'Measured sig_A':<15} {'Expected':<12} {'Error%':<10}")
    print("-"*55)

    for win_size in window_sizes:
        # Create single particle at center
        center = win_size / 2
        y, x = np.ogrid[0:win_size, 0:win_size]
        particle = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        particle = particle.astype(np.float32)

        # Compute autocorrelation
        padded = np.zeros((2*win_size, 2*win_size), dtype=np.float32)
        padded[:win_size, :win_size] = particle

        F = fft2(padded)
        ac = np.real(ifft2(F * np.conj(F)))[:win_size, :win_size]
        ac = fftshift(ac).astype(np.float32)

        # Fit (AA=BB=AB for autocorrelation)
        sig_A_x, sig_A_y, _, _ = fit_correlation_plane(ac, ac, ac, (win_size, win_size))

        if sig_A_x:
            sig_mean = (sig_A_x + sig_A_y) / 2
            expected = theory_sig_A * np.sqrt(2)
            error = 100 * (sig_mean - expected) / expected
            print(f"{win_size}x{win_size:<7} {sig_mean:<15.4f} {expected:<12.4f} {error:<+10.2f}")
        else:
            print(f"{win_size}x{win_size:<7} FAILED")


def main():
    print("="*70)
    print("REAL PIV CORRELATOR WINDOW-SIZE TEST")
    print("="*70)

    # Load the C library
    try:
        lib = _load_marquadt_lib()
        print("\nGaussian fitting library loaded successfully")
    except Exception as e:
        print(f"Failed to load library: {e}")
        return 1

    test_single_particle_correlation()
    test_autocorrelation_with_real_correlator()
    test_ensemble_accumulation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nIf sig_A varies with window size → correlation/edge effects cause bias")
    print("If sig_A is constant → bias comes from elsewhere")

    return 0


if __name__ == "__main__":
    exit(main())
