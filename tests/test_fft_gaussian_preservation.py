"""
Test if FFT operations preserve Gaussian widths.

This is the correct test for FFT edge effects:
1. Create ideal Gaussian with known sigma
2. Apply FFT → IFFT (circular convolution)
3. Check if sigma is preserved at different window sizes

If sigma changes with window size → FFT introduces bias
If sigma is preserved → FFT is not the source of bias

Usage:
    python test_fft_gaussian_preservation.py
"""
import numpy as np
from scipy.fft import fft2, ifft2, fftshift


def create_centered_gaussian(shape, sigma_x, sigma_y, amplitude=1.0):
    """Create a 2D Gaussian centered in the window."""
    h, w = shape
    y = np.arange(h) - h/2
    x = np.arange(w) - w/2
    X, Y = np.meshgrid(x, y)

    gaussian = amplitude * np.exp(-0.5 * (X**2/sigma_x**2 + Y**2/sigma_y**2))
    return gaussian


def fit_gaussian_width(plane):
    """
    Fit Gaussian width using second moments (robust method).

    For a Gaussian: sigma^2 = sum(x^2 * I(x)) / sum(I(x))
    """
    h, w = plane.shape
    y = np.arange(h) - h/2
    x = np.arange(w) - w/2
    X, Y = np.meshgrid(x, y)

    # Use only positive values above threshold
    threshold = 0.01 * np.max(plane)
    mask = plane > threshold

    if np.sum(mask) < 10:
        return None, None

    weights = plane[mask]
    total = np.sum(weights)

    # Second moments
    sigma_x = np.sqrt(np.sum(weights * X[mask]**2) / total)
    sigma_y = np.sqrt(np.sum(weights * Y[mask]**2) / total)

    return sigma_x, sigma_y


def test_fft_roundtrip():
    """Test if FFT→IFFT preserves Gaussian width."""
    print("\n" + "="*70)
    print("TEST 1: FFT Round-Trip (FFT → IFFT)")
    print("="*70)
    print("\nDoes FFT→IFFT change Gaussian width at different window sizes?")

    sigma_true = 1.4  # Same as PIV tests
    window_sizes = [16, 32, 64, 128]

    print(f"\nTrue sigma: {sigma_true}")
    print(f"\n{'Window':<10} {'Before FFT':<15} {'After FFT':<15} {'Error%':<10}")
    print("-"*55)

    for win in window_sizes:
        # Create Gaussian
        g = create_centered_gaussian((win, win), sigma_true, sigma_true)

        # FFT round-trip
        G = fft2(g)
        g_recovered = np.real(ifft2(G))

        # Measure widths
        sig_before, _ = fit_gaussian_width(fftshift(g))
        sig_after, _ = fit_gaussian_width(fftshift(g_recovered))

        if sig_before and sig_after:
            error = 100 * (sig_after - sig_before) / sig_before
            print(f"{win}x{win:<7} {sig_before:<15.6f} {sig_after:<15.6f} {error:<10.4f}")


def test_fft_autocorrelation():
    """Test if FFT-based autocorrelation preserves Gaussian width."""
    print("\n" + "="*70)
    print("TEST 2: FFT Autocorrelation")
    print("="*70)
    print("\nAutocorrelation of Gaussian with sigma → Gaussian with sigma*sqrt(2)")
    print("Testing: AC(g) = IFFT(|FFT(g)|²)")

    sigma_true = 1.4
    sigma_ac_theory = sigma_true * np.sqrt(2)  # Autocorrelation widens by sqrt(2)

    window_sizes = [16, 32, 64, 128]

    print(f"\nInput sigma: {sigma_true}")
    print(f"Expected AC sigma: {sigma_ac_theory:.4f} (×√2)")

    print(f"\n{'Window':<10} {'Measured AC sig':<18} {'Expected':<15} {'Error%':<10}")
    print("-"*60)

    for win in window_sizes:
        # Create Gaussian
        g = create_centered_gaussian((win, win), sigma_true, sigma_true)

        # FFT autocorrelation (with zero-padding to avoid circular effects)
        g_padded = np.zeros((2*win, 2*win))
        g_padded[:win, :win] = g

        G = fft2(g_padded)
        ac_full = np.real(ifft2(G * np.conj(G)))

        # Extract center
        ac = ac_full[:win, :win]

        # Measure width
        sig_x, sig_y = fit_gaussian_width(fftshift(ac))

        if sig_x:
            sig_mean = (sig_x + sig_y) / 2
            error = 100 * (sig_mean - sigma_ac_theory) / sigma_ac_theory
            print(f"{win}x{win:<7} {sig_mean:<18.6f} {sigma_ac_theory:<15.4f} {error:<10.4f}")


def test_fft_crosscorrelation():
    """Test FFT cross-correlation with displaced Gaussian."""
    print("\n" + "="*70)
    print("TEST 3: FFT Cross-Correlation (simulating PIV)")
    print("="*70)
    print("\nCross-correlate two Gaussians with known displacement")
    print("Peak position should equal displacement, width should be sigma*sqrt(2)")

    sigma_true = 1.4
    displacement = (2.5, 1.5)  # pixels
    sigma_cc_theory = sigma_true * np.sqrt(2)

    window_sizes = [16, 32, 64, 128]

    print(f"\nInput sigma: {sigma_true}, displacement: {displacement}")
    print(f"Expected CC sigma: {sigma_cc_theory:.4f}")

    print(f"\n{'Window':<10} {'Measured sig':<15} {'Error%':<10} {'Peak pos':<20} {'Pos Error':<15}")
    print("-"*75)

    for win in window_sizes:
        # Create two Gaussians with displacement
        g1 = create_centered_gaussian((win, win), sigma_true, sigma_true)

        # Shift g2 by displacement
        h, w = win, win
        y = np.arange(h) - h/2 + displacement[1]
        x = np.arange(w) - w/2 + displacement[0]
        X, Y = np.meshgrid(x, y)
        g2 = np.exp(-0.5 * (X**2/sigma_true**2 + Y**2/sigma_true**2))

        # Zero-padded cross-correlation
        g1_pad = np.zeros((2*win, 2*win))
        g2_pad = np.zeros((2*win, 2*win))
        g1_pad[:win, :win] = g1
        g2_pad[:win, :win] = g2

        G1 = fft2(g1_pad)
        G2 = fft2(g2_pad)
        cc_full = np.real(ifft2(G1 * np.conj(G2)))

        cc = fftshift(cc_full)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
        peak_y = peak_idx[0] - cc.shape[0]/2
        peak_x = peak_idx[1] - cc.shape[1]/2

        # Measure width around peak
        sig_x, sig_y = fit_gaussian_width(cc)

        if sig_x:
            sig_mean = (sig_x + sig_y) / 2
            error = 100 * (sig_mean - sigma_cc_theory) / sigma_cc_theory
            pos_error = np.sqrt((peak_x - displacement[0])**2 + (peak_y - displacement[1])**2)
            print(f"{win}x{win:<7} {sig_mean:<15.4f} {error:<10.2f} ({peak_x:.1f}, {peak_y:.1f}){'':<8} {pos_error:<15.2f}")


def test_edge_truncation_effect():
    """Test what happens when Gaussian is truncated at edges."""
    print("\n" + "="*70)
    print("TEST 4: Edge Truncation Effect")
    print("="*70)
    print("\nWhat if Gaussian extends beyond window? (simulates large particles)")

    window_sizes = [16, 32, 64]
    sigma_values = [1.0, 2.0, 4.0, 8.0]  # Different particle sizes

    print(f"\nMeasuring sigma error when Gaussian is truncated at edges")
    print(f"\n{'Window':<10}", end="")
    for sig in sigma_values:
        print(f"{'sig='+str(sig):<12}", end="")
    print()
    print("-"*60)

    for win in window_sizes:
        print(f"{win}x{win:<7}", end="")
        for sigma_true in sigma_values:
            # Create Gaussian that may extend beyond window
            g = create_centered_gaussian((win, win), sigma_true, sigma_true)

            # Measure what we actually get
            sig_x, sig_y = fit_gaussian_width(fftshift(g))

            if sig_x:
                sig_mean = (sig_x + sig_y) / 2
                error = 100 * (sig_mean - sigma_true) / sigma_true
                print(f"{error:+.1f}%{'':<6}", end="")
            else:
                print(f"{'FAIL':<12}", end="")
        print()


def test_particle_count_effect():
    """Test autocorrelation with different numbers of particles."""
    print("\n" + "="*70)
    print("TEST 5: Particle Count Effect (the real issue?)")
    print("="*70)
    print("\nSimulating particle images with different particle counts")

    window_size = 32
    particle_sigma = 0.5  # particle radius
    particle_counts = [5, 10, 20, 50, 100, 200]
    n_trials = 20

    print(f"\nWindow: {window_size}x{window_size}, particle sigma: {particle_sigma}")
    print(f"Theoretical autocorr sigma: {particle_sigma * np.sqrt(2):.4f}")

    print(f"\n{'N particles':<15} {'Mean AC sig':<15} {'Std':<10} {'Error%':<10}")
    print("-"*55)

    np.random.seed(42)
    theory_ac_sig = particle_sigma * np.sqrt(2)

    for n_particles in particle_counts:
        ac_sigs = []

        for _ in range(n_trials):
            # Generate random particle image
            image = np.zeros((window_size, window_size))

            x_pos = np.random.uniform(0, window_size, n_particles)
            y_pos = np.random.uniform(0, window_size, n_particles)

            Y, X = np.ogrid[0:window_size, 0:window_size]

            for x, y in zip(x_pos, y_pos):
                r2 = (X - x)**2 + (Y - y)**2
                image += np.exp(-r2 / (2 * particle_sigma**2))

            # Compute autocorrelation
            img_pad = np.zeros((2*window_size, 2*window_size))
            img_pad[:window_size, :window_size] = image

            F = fft2(img_pad)
            ac = np.real(ifft2(F * np.conj(F)))[:window_size, :window_size]

            # Fit
            sig_x, sig_y = fit_gaussian_width(fftshift(ac))
            if sig_x:
                ac_sigs.append((sig_x + sig_y) / 2)

        if ac_sigs:
            mean_sig = np.mean(ac_sigs)
            std_sig = np.std(ac_sigs)
            error = 100 * (mean_sig - theory_ac_sig) / theory_ac_sig
            print(f"{n_particles:<15} {mean_sig:<15.4f} {std_sig:<10.4f} {error:<+10.1f}")


def main():
    print("="*70)
    print("FFT GAUSSIAN PRESERVATION TEST")
    print("="*70)
    print("\nTesting whether FFT operations preserve Gaussian widths")
    print("This isolates FFT effects from particle/statistical effects")

    test_fft_roundtrip()
    test_fft_autocorrelation()
    test_fft_crosscorrelation()
    test_edge_truncation_effect()
    test_particle_count_effect()

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)


if __name__ == "__main__":
    main()
