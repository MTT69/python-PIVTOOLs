"""
Synthetic correlation plane generator for k-space fitter unit tests.

Creates Gaussian correlation planes from first principles:
- Auto-correlations R_AA = R_BB: centered Gaussian, width = 2·sigma_particle
  (convolution of particle image with itself doubles the variance)
- Cross-correlation R_AB: shifted Gaussian, total covariance = 2·Sigma_particle + Sigma_stress
  (particle image convolution + velocity PDF convolution)

The k-space fitter extracts ONLY Sigma_stress by dividing out the particle
shape via F_ref = sqrt(|F_AA|·|F_BB|).
"""

import numpy as np


def generate_2d_gaussian(shape, cov_matrix, center, amplitude=1.0):
    """Generate a 2D Gaussian with full covariance matrix.

    Parameters
    ----------
    shape : tuple of int
        (height, width) of the output array.
    cov_matrix : ndarray, shape (2, 2)
        Covariance matrix [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
    center : tuple of float
        (center_y, center_x) — peak location.
    amplitude : float
        Peak value.

    Returns
    -------
    ndarray
        2D Gaussian, shape ``shape``.
    """
    h, w = shape
    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    cy, cx = center
    dx = X - cx
    dy = Y - cy

    det = cov_matrix[0, 0] * cov_matrix[1, 1] - cov_matrix[0, 1] ** 2
    if det < 1e-20:
        return np.zeros(shape)

    inv_cov = np.array([
        [cov_matrix[1, 1], -cov_matrix[0, 1]],
        [-cov_matrix[1, 0], cov_matrix[0, 0]],
    ]) / det

    # Quadratic form: (dx, dy) · inv_cov · (dx, dy)^T
    quad = (inv_cov[0, 0] * dx ** 2
            + (inv_cov[0, 1] + inv_cov[1, 0]) * dx * dy
            + inv_cov[1, 1] * dy ** 2)

    return amplitude * np.exp(-0.5 * quad)


def generate_autocorrelation(
    shape,
    sigma_x=2.0,
    sigma_y=2.0,
    amplitude=1.0,
    noise_std=0.0,
    rng=None,
    offset=0.0,
):
    """Generate an auto-correlation plane R_AA (or R_BB).

    The auto-correlation of a Gaussian particle image with width sigma
    is a Gaussian with width 2·sigma (convolution doubles variance).

    Parameters
    ----------
    shape : tuple of int
        (height, width).
    sigma_x, sigma_y : float
        Particle image standard deviations (pixels).
    amplitude : float
        Peak value.
    noise_std : float
        Additive Gaussian noise standard deviation.
    rng : np.random.Generator or None
        Random number generator.
    offset : float
        Flat constant added to the Gaussian (DC noise floor).

    Returns
    -------
    ndarray
        Auto-correlation plane.
    """
    # Convolution doubles the variance: cov = 2·diag(sigma^2)
    cov = np.array([
        [2.0 * sigma_x ** 2, 0.0],
        [0.0, 2.0 * sigma_y ** 2],
    ])
    center = (shape[0] / 2.0, shape[1] / 2.0)
    R = generate_2d_gaussian(shape, cov, center, amplitude) + offset

    if noise_std > 0 and rng is not None:
        R = R + rng.normal(0, noise_std, shape)

    return R


def generate_crosscorrelation(
    shape,
    sigma_particle_x=2.0,
    sigma_particle_y=2.0,
    sigma_stress_xx=0.0,
    sigma_stress_yy=0.0,
    sigma_stress_xy=0.0,
    mu_x=0.0,
    mu_y=0.0,
    amplitude=1.0,
    noise_std=0.0,
    rng=None,
    offset=0.0,
):
    """Generate a cross-correlation plane R_AB.

    The cross-correlation has:
    - Center shifted by (mu_x, mu_y) — mean displacement
    - Total covariance = 2·Sigma_particle + Sigma_stress

    Parameters
    ----------
    shape : tuple of int
        (height, width).
    sigma_particle_x, sigma_particle_y : float
        Particle image standard deviations (pixels).
    sigma_stress_xx, sigma_stress_yy, sigma_stress_xy : float
        Reynolds stress tensor components (pixels^2).
    mu_x, mu_y : float
        Mean displacement (pixels).
    amplitude : float
        Peak value.
    noise_std : float
        Additive noise.
    rng : np.random.Generator or None
        Random number generator.
    offset : float
        Flat constant added to the Gaussian (DC noise floor).

    Returns
    -------
    ndarray
        Cross-correlation plane.
    """
    # Total covariance = 2·Sigma_particle + Sigma_stress
    cov = np.array([
        [2.0 * sigma_particle_x ** 2 + sigma_stress_xx, sigma_stress_xy],
        [sigma_stress_xy, 2.0 * sigma_particle_y ** 2 + sigma_stress_yy],
    ])
    center = (shape[0] / 2.0 + mu_y, shape[1] / 2.0 + mu_x)
    R = generate_2d_gaussian(shape, cov, center, amplitude) + offset

    if noise_std > 0 and rng is not None:
        R = R + rng.normal(0, noise_std, shape)

    return R


def generate_correlation_triplet(
    shape,
    sigma_particle_x=2.0,
    sigma_particle_y=2.0,
    sigma_stress_xx=0.0,
    sigma_stress_yy=0.0,
    sigma_stress_xy=0.0,
    mu_x=0.0,
    mu_y=0.0,
    amplitude=1.0,
    noise_std=0.0,
    seed=42,
    offset_A=0.0,
    offset_B=0.0,
    offset_AB=0.0,
):
    """Generate a matched (R_AA, R_BB, R_AB) triplet.

    R_AA and R_BB are identical (same particle field convolved with itself).

    Returns
    -------
    tuple of ndarray
        (R_AA, R_BB, R_AB)
    """
    rng = np.random.default_rng(seed)

    R_AA = generate_autocorrelation(
        shape, sigma_particle_x, sigma_particle_y,
        amplitude, noise_std, rng, offset=offset_A,
    )
    R_BB = generate_autocorrelation(
        shape, sigma_particle_x, sigma_particle_y,
        amplitude, noise_std, rng, offset=offset_B,
    )
    R_AB = generate_crosscorrelation(
        shape, sigma_particle_x, sigma_particle_y,
        sigma_stress_xx, sigma_stress_yy, sigma_stress_xy,
        mu_x, mu_y, amplitude, noise_std, rng, offset=offset_AB,
    )
    return R_AA, R_BB, R_AB


def flatten_for_kspace(R_AA, R_BB, R_AB, n_windows=1):
    """Pack correlation planes into the flat format expected by fit_windows_kspace().

    Parameters
    ----------
    R_AA, R_BB, R_AB : ndarray
        2D correlation planes, all shape (h, w).
    n_windows : int
        Number of windows. If > 1, the same planes are tiled.

    Returns
    -------
    tuple
        (R_AA_flat, R_BB_flat, R_AB_flat, mask_flat, corr_size)
        Ready to pass to ``fit_windows_kspace()``.
    """
    h, w = R_AA.shape
    corr_size = (h, w)

    R_AA_flat = np.tile(R_AA.ravel(), n_windows)
    R_BB_flat = np.tile(R_BB.ravel(), n_windows)
    R_AB_flat = np.tile(R_AB.ravel(), n_windows)
    mask_flat = np.zeros(n_windows, dtype=bool)

    return R_AA_flat, R_BB_flat, R_AB_flat, mask_flat, corr_size


# Alias: same flat packing format works for both k-space and Gaussian fitters
flatten_for_gaussian = flatten_for_kspace


class _MockConfig:
    """Minimal mock config for fit_windows_kspace() and fit_windows_openmp()."""

    def __init__(
        self,
        fit_method='kspace',
        gradient_correction=False,
        ensemble_window_sizes=None,
        ensemble_type=None,
        ensemble_sum_window=None,
        ensemble_sum_fitting_window=None,
        ensemble_fit_offset=True,
        ensemble_mask_center_pixel=True,
    ):
        self.ensemble_fit_method = fit_method
        self.ensemble_gradient_correction = gradient_correction
        self.ensemble_window_sizes = ensemble_window_sizes or [[32, 32]]
        self.ensemble_type = ensemble_type or ['std']
        self.ensemble_sum_window = ensemble_sum_window or [64, 64]
        self.ensemble_sum_fitting_window = ensemble_sum_fitting_window
        self.ensemble_fit_offset = ensemble_fit_offset
        self.ensemble_mask_center_pixel = ensemble_mask_center_pixel


def make_mock_config(
    fit_method='kspace',
    gradient_correction=False,
    ensemble_window_sizes=None,
    ensemble_type=None,
    ensemble_sum_window=None,
    ensemble_sum_fitting_window=None,
    ensemble_fit_offset=True,
    ensemble_mask_center_pixel=True,
):
    """Create a minimal mock config object."""
    return _MockConfig(
        fit_method, gradient_correction,
        ensemble_window_sizes, ensemble_type,
        ensemble_sum_window, ensemble_sum_fitting_window,
        ensemble_fit_offset, ensemble_mask_center_pixel,
    )
