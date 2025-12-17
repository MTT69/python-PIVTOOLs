"""
Stacked 3D Gaussian Fitting for Reynolds Stress Extraction
==========================================================

Fits a 22-parameter model to simultaneously estimate:
- Particle image shape (geometric covariance)
- Reynolds stress tensor (turbulent covariance)
- 3D displacement

The key insight from ensemble PIV:
- Auto-correlation width = geometric shape only
- Cross-correlation width = geometric shape + turbulent broadening

By fitting both simultaneously with coupled constraints:
    Sigma_auto = Sigma_geo
    Sigma_cross = Sigma_geo + Sigma_turb

We can extract Sigma_turb = Reynolds stress tensor.

22-Parameter Model:
    [0-1]   Amplitudes:    A_auto, A_cross
    [2-3]   Backgrounds:   B_auto, B_cross
    [4-9]   Sigma_geo:     sxx, syy, szz, sxy, sxz, syz (6 unique)
    [10-15] Sigma_turb:    txx, tyy, tzz, txy, txz, tyz (6 unique)
    [16-18] Center_auto:   x0, y0, z0
    [19-21] Center_cross:  x0, y0, z0 (displacement from auto center)
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


# =============================================================================
# RESULT CONTAINER
# =============================================================================
@dataclass
class StackedGaussianResult3D:
    """Results from 3D stacked Gaussian fitting."""

    # Amplitudes and backgrounds
    amp_auto: float
    amp_cross: float
    bg_auto: float
    bg_cross: float

    # Covariances (3x3 symmetric matrices)
    sigma_geo: np.ndarray    # Geometric shape (particle image spread)
    sigma_turb: np.ndarray   # Reynolds stress tensor

    # Peak positions
    center_auto: np.ndarray   # (3,) auto-correlation center
    center_cross: np.ndarray  # (3,) cross-correlation center

    # Fit quality
    cost: float
    success: bool
    message: str

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector (cross center - auto center)."""
        return self.center_cross - self.center_auto

    @property
    def sigma_cross(self) -> np.ndarray:
        """Cross-correlation covariance = geo + turb."""
        return self.sigma_geo + self.sigma_turb

    def reynolds_stress_components(self) -> dict:
        """Return Reynolds stress components as a dictionary."""
        return {
            'uu': self.sigma_turb[0, 0],
            'vv': self.sigma_turb[1, 1],
            'ww': self.sigma_turb[2, 2],
            'uv': self.sigma_turb[0, 1],
            'uw': self.sigma_turb[0, 2],
            'vw': self.sigma_turb[1, 2],
        }


# =============================================================================
# COVARIANCE MATRIX UTILITIES
# =============================================================================
def cov_params_to_matrix(params: np.ndarray) -> np.ndarray:
    """
    Convert 6 covariance parameters to 3x3 symmetric matrix.

    Parameters
    ----------
    params : ndarray, shape (6,)
        [sxx, syy, szz, sxy, sxz, syz]

    Returns
    -------
    cov : ndarray, shape (3, 3)
        Symmetric covariance matrix
    """
    sxx, syy, szz, sxy, sxz, syz = params
    return np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ])


def matrix_to_cov_params(cov: np.ndarray) -> np.ndarray:
    """
    Extract 6 unique parameters from 3x3 symmetric covariance matrix.

    Returns
    -------
    params : ndarray, shape (6,)
        [sxx, syy, szz, sxy, sxz, syz]
    """
    return np.array([
        cov[0, 0],  # sxx
        cov[1, 1],  # syy
        cov[2, 2],  # szz
        cov[0, 1],  # sxy
        cov[0, 2],  # sxz
        cov[1, 2],  # syz
    ])


def is_positive_definite(cov: np.ndarray) -> bool:
    """Check if matrix is positive definite via Cholesky."""
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


# =============================================================================
# 3D GAUSSIAN MODEL
# =============================================================================
def gaussian_3d(
    coords: np.ndarray,
    A: float,
    B: float,
    center: np.ndarray,
    cov: np.ndarray
) -> np.ndarray:
    """
    Evaluate 3D Gaussian with full covariance matrix.

    Model: G(r) = A * exp(-0.5 * (r-c)^T @ Sigma^{-1} @ (r-c)) + B

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Coordinate points [x, y, z]
    A : float
        Amplitude
    B : float
        Background offset
    center : ndarray, shape (3,)
        Center position [x0, y0, z0]
    cov : ndarray, shape (3, 3)
        Covariance matrix

    Returns
    -------
    values : ndarray, shape (N,)
        Gaussian values at each point
    """
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.full(len(coords), np.nan)

    # Displacement from center
    diff = coords - center  # (N, 3)

    # Mahalanobis distance squared: (r-c)^T @ Sigma^{-1} @ (r-c)
    # Vectorized: sum over axis 1 of (diff @ cov_inv) * diff
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)

    return A * np.exp(-0.5 * mahal_sq) + B


# =============================================================================
# PARAMETER PACKING/UNPACKING
# =============================================================================
def pack_params(
    A_auto: float, A_cross: float,
    B_auto: float, B_cross: float,
    sigma_geo: np.ndarray,
    sigma_turb: np.ndarray,
    center_auto: np.ndarray,
    center_cross: np.ndarray
) -> np.ndarray:
    """Pack all parameters into a single 22-element array."""
    return np.concatenate([
        [A_auto, A_cross, B_auto, B_cross],
        matrix_to_cov_params(sigma_geo),    # 6 params
        matrix_to_cov_params(sigma_turb),   # 6 params
        center_auto,                         # 3 params
        center_cross,                        # 3 params
    ])


def unpack_params(params: np.ndarray) -> dict:
    """Unpack 22-element array into named components."""
    return {
        'A_auto': params[0],
        'A_cross': params[1],
        'B_auto': params[2],
        'B_cross': params[3],
        'sigma_geo': cov_params_to_matrix(params[4:10]),
        'sigma_turb': cov_params_to_matrix(params[10:16]),
        'center_auto': params[16:19].copy(),
        'center_cross': params[19:22].copy(),
    }


# =============================================================================
# RESIDUAL FUNCTION
# =============================================================================
def residuals(
    params: np.ndarray,
    coords: np.ndarray,
    auto_data: np.ndarray,
    cross_data: np.ndarray,
    weights_auto: Optional[np.ndarray] = None,
    weights_cross: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute stacked residuals for auto + cross correlation fitting.

    The coupled constraint is enforced:
        Sigma_auto = Sigma_geo
        Sigma_cross = Sigma_geo + Sigma_turb

    Parameters
    ----------
    params : ndarray, shape (22,)
        Parameter vector
    coords : ndarray, shape (N, 3)
        Coordinate grid points
    auto_data : ndarray, shape (N,)
        Flattened auto-correlation data
    cross_data : ndarray, shape (N,)
        Flattened cross-correlation data
    weights_auto : ndarray, shape (N,), optional
        Weights for auto-correlation residuals
    weights_cross : ndarray, shape (N,), optional
        Weights for cross-correlation residuals

    Returns
    -------
    residuals : ndarray, shape (2*N,)
        Stacked residuals [auto_residuals, cross_residuals]
    """
    p = unpack_params(params)

    sigma_geo = p['sigma_geo']
    sigma_turb = p['sigma_turb']
    sigma_cross = sigma_geo + sigma_turb

    # Check positive definiteness of both covariances
    if not is_positive_definite(sigma_geo):
        return np.full(len(auto_data) + len(cross_data), 1e10)

    if not is_positive_definite(sigma_cross):
        return np.full(len(auto_data) + len(cross_data), 1e10)

    # Auto-correlation model (uses sigma_geo only)
    model_auto = gaussian_3d(
        coords, p['A_auto'], p['B_auto'],
        p['center_auto'], sigma_geo
    )

    # Cross-correlation model (uses sigma_geo + sigma_turb)
    model_cross = gaussian_3d(
        coords, p['A_cross'], p['B_cross'],
        p['center_cross'], sigma_cross
    )

    # Handle NaN from invalid covariances
    if np.any(np.isnan(model_auto)) or np.any(np.isnan(model_cross)):
        return np.full(len(auto_data) + len(cross_data), 1e10)

    # Stack residuals with optional weighting
    res_auto = model_auto - auto_data
    res_cross = model_cross - cross_data

    if weights_auto is not None:
        res_auto = res_auto * weights_auto
    if weights_cross is not None:
        res_cross = res_cross * weights_cross

    return np.concatenate([res_auto, res_cross])


# =============================================================================
# INITIAL GUESS COMPUTATION
# =============================================================================
def estimate_covariance_from_moments(
    volume: np.ndarray,
    center: np.ndarray,
    threshold_fraction: float = 0.1
) -> np.ndarray:
    """
    Estimate covariance matrix from intensity-weighted second moments.

    This provides a much better initial guess than isotropic assumption.

    Parameters
    ----------
    volume : ndarray
        Correlation volume
    center : ndarray (3,)
        Center position (relative to volume center)
    threshold_fraction : float
        Only use voxels above this fraction of peak

    Returns
    -------
    cov : ndarray (3, 3)
        Estimated covariance matrix
    """
    shape = np.array(volume.shape)
    vol_center = shape // 2

    # Create coordinate grid relative to peak
    x = np.arange(shape[0]) - vol_center[0] - center[0]
    y = np.arange(shape[1]) - vol_center[1] - center[1]
    z = np.arange(shape[2]) - vol_center[2] - center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Threshold to focus on peak region
    threshold = volume.max() * threshold_fraction
    mask = volume > threshold
    weights = np.where(mask, volume - threshold, 0)
    total_weight = weights.sum()

    if total_weight < 1e-10:
        # Fallback to isotropic
        sigma = min(shape) / 6.0
        return np.diag([sigma**2, sigma**2, sigma**2])

    # Compute weighted second moments
    cov = np.zeros((3, 3))
    coords = [X, Y, Z]

    for i in range(3):
        for j in range(i, 3):
            moment = np.sum(weights * coords[i] * coords[j]) / total_weight
            cov[i, j] = moment
            cov[j, i] = moment

    # Ensure positive definiteness with minimum eigenvalue
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.5)  # Minimum variance of 0.5
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov


def compute_initial_guess(
    auto_volume: np.ndarray,
    cross_volume: np.ndarray,
    roi_center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute initial 22-parameter guess from correlation volumes.

    Uses moment-based covariance estimation for better initial guess.

    Parameters
    ----------
    auto_volume : ndarray, shape (nx, ny, nz)
        Auto-correlation volume (peak at center)
    cross_volume : ndarray, shape (nx, ny, nz)
        Cross-correlation volume
    roi_center : ndarray, shape (3,), optional
        Center of ROI in volume coordinates

    Returns
    -------
    initial_guess : ndarray, shape (22,)
        Initial parameter values
    """
    shape = np.array(auto_volume.shape)

    if roi_center is None:
        roi_center = shape / 2

    # Find peaks relative to roi_center
    auto_peak_idx = np.unravel_index(np.argmax(auto_volume), shape)
    cross_peak_idx = np.unravel_index(np.argmax(cross_volume), shape)

    # Initial estimates for amplitudes and backgrounds
    A_auto = float(auto_volume.max() - np.percentile(auto_volume, 5))
    A_cross = float(cross_volume.max() - np.percentile(cross_volume, 5))
    B_auto = float(np.percentile(auto_volume, 5))
    B_cross = float(np.percentile(cross_volume, 5))

    # Centers relative to roi_center
    center_auto = np.array(auto_peak_idx, dtype=float) - roi_center
    center_cross = np.array(cross_peak_idx, dtype=float) - roi_center

    # Moment-based covariance estimation (much better than isotropic!)
    sigma_auto_est = estimate_covariance_from_moments(auto_volume, center_auto)
    sigma_cross_est = estimate_covariance_from_moments(cross_volume, center_cross)

    # sigma_geo = auto covariance (particle shape only)
    sigma_geo = sigma_auto_est

    # sigma_turb = cross - auto (turbulent broadening)
    # But ensure it's positive semi-definite
    sigma_turb_raw = sigma_cross_est - sigma_auto_est
    eigvals, eigvecs = np.linalg.eigh(sigma_turb_raw)
    eigvals = np.maximum(eigvals, 0.1)  # Ensure positive
    sigma_turb = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return pack_params(
        A_auto, A_cross, B_auto, B_cross,
        sigma_geo, sigma_turb,
        center_auto, center_cross
    )


# =============================================================================
# MAIN FITTING FUNCTION
# =============================================================================
def fit_stacked_gaussian_3d(
    auto_volume: np.ndarray,
    cross_volume: np.ndarray,
    roi_size: int = 15,
    initial_guess: Optional[np.ndarray] = None,
    use_weights: bool = True,
    verbose: bool = False
) -> StackedGaussianResult3D:
    """
    Fit 22-parameter stacked 3D Gaussian to auto and cross correlation volumes.

    Extracts:
    - Geometric covariance (particle shape)
    - Reynolds stress tensor (turbulent covariance)
    - 3D displacement vector

    Parameters
    ----------
    auto_volume : ndarray, shape (nx, ny, nz)
        Ensemble-averaged auto-correlation (fftshift'd, peak at center)
    cross_volume : ndarray, shape (nx, ny, nz)
        Ensemble-averaged cross-correlation
    roi_size : int
        Half-size of ROI around peak for fitting
    initial_guess : ndarray, shape (22,), optional
        Initial parameter guess. If None, computed automatically.
    use_weights : bool
        If True, weight residuals by intensity (higher weight near peak)
    verbose : bool
        Print fitting progress

    Returns
    -------
    result : StackedGaussianResult3D
        Fitted parameters including Reynolds stress tensor
    """
    shape = np.array(auto_volume.shape)
    center = shape // 2

    # Extract ROI around center
    slices = tuple(
        slice(max(0, c - roi_size), min(s, c + roi_size + 1))
        for c, s in zip(center, shape)
    )

    auto_roi = auto_volume[slices]
    cross_roi = cross_volume[slices]
    roi_shape = np.array(auto_roi.shape)
    roi_center = roi_shape / 2

    # Create coordinate grid (relative to ROI center)
    x = np.arange(roi_shape[0]) - roi_center[0]
    y = np.arange(roi_shape[1]) - roi_center[1]
    z = np.arange(roi_shape[2]) - roi_center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    # Flatten data
    auto_data = auto_roi.ravel().astype(np.float64)
    cross_data = cross_roi.ravel().astype(np.float64)

    # Compute weights (intensity-based, higher near peak)
    if use_weights:
        # Weight by normalized intensity (gives more importance to peak region)
        auto_positive = np.maximum(auto_data, 0)
        cross_positive = np.maximum(cross_data, 0)
        weights_auto = np.sqrt(auto_positive / (auto_positive.max() + 1e-10) + 0.1)
        weights_cross = np.sqrt(cross_positive / (cross_positive.max() + 1e-10) + 0.1)
    else:
        weights_auto = None
        weights_cross = None

    # Initial guess
    if initial_guess is None:
        initial_guess = compute_initial_guess(auto_roi, cross_roi, roi_center)

    # Parameter bounds
    # Amplitudes > 0, backgrounds unconstrained
    # Variances > 0.1 to prevent singularities
    # Off-diagonal covariances bounded
    # Centers within ROI
    lower = np.array([
        0.0, 0.0, -np.inf, -np.inf,          # A_auto, A_cross, B_auto, B_cross
        0.1, 0.1, 0.1, -100, -100, -100,     # sigma_geo: sxx, syy, szz, sxy, sxz, syz
        -50, -50, -50, -50, -50, -50,        # sigma_turb (can be small/negative initially)
        -roi_size, -roi_size, -roi_size,     # center_auto
        -roi_size, -roi_size, -roi_size,     # center_cross
    ])
    upper = np.array([
        np.inf, np.inf, np.inf, np.inf,      # A_auto, A_cross, B_auto, B_cross
        1000, 1000, 1000, 100, 100, 100,     # sigma_geo
        100, 100, 100, 50, 50, 50,           # sigma_turb
        roi_size, roi_size, roi_size,        # center_auto
        roi_size, roi_size, roi_size,        # center_cross
    ])

    # Fit using scipy least_squares (trust-region reflective)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = least_squares(
            residuals,
            initial_guess,
            args=(coords, auto_data, cross_data, weights_auto, weights_cross),
            bounds=(lower, upper),
            method='trf',
            max_nfev=5000,
            verbose=2 if verbose else 0
        )

    # Unpack results
    p = unpack_params(result.x)

    return StackedGaussianResult3D(
        amp_auto=p['A_auto'],
        amp_cross=p['A_cross'],
        bg_auto=p['B_auto'],
        bg_cross=p['B_cross'],
        sigma_geo=p['sigma_geo'],
        sigma_turb=p['sigma_turb'],
        center_auto=p['center_auto'],
        center_cross=p['center_cross'],
        cost=float(result.cost),
        success=result.success,
        message=result.message
    )


# =============================================================================
# TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    print("Testing Stacked 3D Gaussian Fitting")
    print("=" * 50)

    # Create synthetic test data
    shape = (64, 64, 16)
    roi_center = np.array(shape) / 2

    # Coordinate grid
    x = np.arange(shape[0]) - roi_center[0]
    y = np.arange(shape[1]) - roi_center[1]
    z = np.arange(shape[2]) - roi_center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # True parameters
    true_geo = np.array([
        [4.0, 0.5, 0.2],
        [0.5, 3.0, 0.3],
        [0.2, 0.3, 2.0]
    ])
    true_turb = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 0.8, 0.2],
        [0.1, 0.2, 0.5]
    ])
    true_cross_cov = true_geo + true_turb
    true_displacement = np.array([2.0, 1.5, 0.5])

    print("True geometric covariance:")
    print(true_geo)
    print("\nTrue Reynolds stress (turbulence):")
    print(true_turb)
    print(f"\nTrue displacement: {true_displacement}")

    # Generate synthetic volumes
    auto_vol = gaussian_3d(
        coords, A=1.0, B=0.1,
        center=np.array([0.0, 0.0, 0.0]),
        cov=true_geo
    ).reshape(shape)

    cross_vol = gaussian_3d(
        coords, A=0.8, B=0.1,
        center=true_displacement,
        cov=true_cross_cov
    ).reshape(shape)

    # Add noise
    auto_vol += 0.02 * np.random.randn(*shape)
    cross_vol += 0.02 * np.random.randn(*shape)

    print("\nFitting...")
    result = fit_stacked_gaussian_3d(auto_vol, cross_vol, roi_size=20)

    print(f"\nFit success: {result.success}")
    print(f"Fit cost: {result.cost:.4f}")

    print(f"\nFitted displacement: {result.displacement}")
    print(f"True displacement: {true_displacement}")
    print(f"Displacement error: {np.linalg.norm(result.displacement - true_displacement):.4f}")

    print("\nFitted Reynolds stress:")
    print(result.sigma_turb)
    print("\nTrue Reynolds stress:")
    print(true_turb)
    print(f"\nReynolds stress RMS error: {np.sqrt(np.mean((result.sigma_turb - true_turb)**2)):.4f}")

    print("\nDone!")
