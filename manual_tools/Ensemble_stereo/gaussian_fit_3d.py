"""
3D Gaussian Fitting for Cross-Correlation Peak
===============================================

Full degree-of-freedom 3D Gaussian fitting with:
- Amplitude and background offset
- 3D center position (sub-voxel)
- Full 3x3 covariance matrix (6 unique parameters)

Model:
    G(x,y,z) = A * exp(-0.5 * r^T @ Sigma^-1 @ r) + B
    where r = [x-x0, y-y0, z-z0]^T

Parameters (10 DOF):
    A     - Amplitude
    B     - Background offset
    x0    - Center X
    y0    - Center Y
    z0    - Center Z
    sxx   - Variance in X (sigma_xx)
    syy   - Variance in Y (sigma_yy)
    szz   - Variance in Z (sigma_zz)
    sxy   - Covariance XY (sigma_xy)
    sxz   - Covariance XZ (sigma_xz)
    syz   - Covariance YZ (sigma_yz)
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GaussianFitResult:
    """Container for 3D Gaussian fit results."""
    # Fitted parameters
    amplitude: float
    background: float
    center: np.ndarray  # [x0, y0, z0]
    covariance: np.ndarray  # 3x3 covariance matrix

    # Derived quantities
    principal_axes: np.ndarray  # 3x3 eigenvectors (columns)
    principal_sigmas: np.ndarray  # sqrt of eigenvalues (standard deviations along axes)

    # Fit quality
    residual_rms: float
    r_squared: float

    # Original data for visualization
    roi_center: np.ndarray  # Center of ROI in original volume coords
    roi_size: int

    def displacement_from_roi_center(self) -> np.ndarray:
        """Get displacement relative to ROI center."""
        return self.center

    def sigma_x(self) -> float:
        return np.sqrt(self.covariance[0, 0])

    def sigma_y(self) -> float:
        return np.sqrt(self.covariance[1, 1])

    def sigma_z(self) -> float:
        return np.sqrt(self.covariance[2, 2])


def cov_params_to_matrix(cov_params: np.ndarray) -> np.ndarray:
    """
    Convert 6 covariance parameters to 3x3 symmetric matrix.

    Parameters
    ----------
    cov_params : array of shape (6,)
        [sxx, syy, szz, sxy, sxz, syz]

    Returns
    -------
    cov : array of shape (3, 3)
        Symmetric covariance matrix
    """
    sxx, syy, szz, sxy, sxz, syz = cov_params
    return np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ])


def matrix_to_cov_params(cov: np.ndarray) -> np.ndarray:
    """
    Extract 6 unique parameters from symmetric 3x3 covariance matrix.

    Returns
    -------
    cov_params : array of shape (6,)
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


def gaussian_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                A: float, B: float,
                x0: float, y0: float, z0: float,
                cov: np.ndarray) -> np.ndarray:
    """
    Evaluate 3D Gaussian with full covariance matrix.

    Parameters
    ----------
    x, y, z : arrays
        Coordinate arrays (can be meshgrid outputs)
    A : float
        Amplitude
    B : float
        Background offset
    x0, y0, z0 : float
        Center position
    cov : array of shape (3, 3)
        Covariance matrix

    Returns
    -------
    values : array
        Gaussian values at each point
    """
    # Compute inverse covariance
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If singular, return zeros
        return np.zeros_like(x)

    # Flatten for vectorized computation
    shape = x.shape
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()

    # Displacement from center
    dx = x_flat - x0
    dy = y_flat - y0
    dz = z_flat - z0

    # Mahalanobis distance: r^T @ cov_inv @ r
    # For each point: [dx, dy, dz] @ cov_inv @ [dx, dy, dz]^T
    r = np.column_stack([dx, dy, dz])  # (N, 3)

    # Suppress warnings during optimization exploration of invalid covariances
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        mahal_sq = np.sum(r @ cov_inv * r, axis=1)  # (N,)

        # Handle any NaN/inf values from invalid covariances
        if np.any(~np.isfinite(mahal_sq)):
            return np.full(shape, np.nan)

        # Gaussian: A * exp(-0.5 * mahal_sq) + B
        values = A * np.exp(-0.5 * mahal_sq) + B

    return values.reshape(shape)


def _pack_params(A, B, x0, y0, z0, cov_params):
    """Pack all parameters into single array."""
    return np.concatenate([[A, B, x0, y0, z0], cov_params])


def _unpack_params(params):
    """Unpack parameter array."""
    A = params[0]
    B = params[1]
    x0 = params[2]
    y0 = params[3]
    z0 = params[4]
    cov_params = params[5:11]
    return A, B, x0, y0, z0, cov_params


def _residuals(params, x, y, z, data):
    """Compute residuals for least squares fitting."""
    A, B, x0, y0, z0, cov_params = _unpack_params(params)
    cov = cov_params_to_matrix(cov_params)

    # Check positive definiteness via Cholesky
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Not positive definite - return large residual
        return np.full_like(data.ravel(), 1e10)

    model = gaussian_3d(x, y, z, A, B, x0, y0, z0, cov)
    return (model - data).ravel()


def fit_gaussian_3d(volume: np.ndarray,
                    roi_center: Optional[np.ndarray] = None,
                    roi_size: int = 15) -> GaussianFitResult:
    """
    Fit a 3D Gaussian to a correlation volume.

    Parameters
    ----------
    volume : ndarray, shape (nx, ny, nz)
        3D correlation volume (already fftshift'd, zero-lag at center)
    roi_center : array of shape (3,), optional
        Center of ROI for fitting (x, y, z indices).
        If None, uses the location of the maximum value.
    roi_size : int
        Half-size of ROI in each dimension

    Returns
    -------
    result : GaussianFitResult
        Fitted parameters and quality metrics
    """
    nx, ny, nz = volume.shape

    # Find peak if roi_center not specified
    if roi_center is None:
        peak_idx = np.argmax(volume)
        roi_center = np.array(np.unravel_index(peak_idx, volume.shape))
    else:
        roi_center = np.asarray(roi_center)

    # Extract ROI around peak
    x_min = max(0, int(roi_center[0] - roi_size))
    x_max = min(nx, int(roi_center[0] + roi_size + 1))
    y_min = max(0, int(roi_center[1] - roi_size))
    y_max = min(ny, int(roi_center[1] + roi_size + 1))
    z_min = max(0, int(roi_center[2] - roi_size))
    z_max = min(nz, int(roi_center[2] + roi_size + 1))

    roi = volume[x_min:x_max, y_min:y_max, z_min:z_max]

    # Create coordinate grids (relative to ROI center)
    rx = np.arange(x_min, x_max) - roi_center[0]
    ry = np.arange(y_min, y_max) - roi_center[1]
    rz = np.arange(z_min, z_max) - roi_center[2]
    X, Y, Z = np.meshgrid(rx, ry, rz, indexing='ij')

    # Initial guess
    A_init = roi.max() - roi.min()
    B_init = roi.min()
    x0_init = 0.0  # Relative to roi_center
    y0_init = 0.0
    z0_init = 0.0

    # Initial covariance: isotropic, based on data extent
    sigma_init = roi_size / 3.0
    cov_init = np.array([
        sigma_init**2, sigma_init**2, sigma_init**2,  # diagonal
        0.0, 0.0, 0.0  # off-diagonal
    ])

    params_init = _pack_params(A_init, B_init, x0_init, y0_init, z0_init, cov_init)

    # Bounds
    # A > 0, B can be negative, center within ROI, variances > 0
    lower_bounds = [0, -np.inf, -roi_size, -roi_size, -roi_size,
                    0.1, 0.1, 0.1, -100, -100, -100]
    upper_bounds = [np.inf, np.inf, roi_size, roi_size, roi_size,
                    1000, 1000, 1000, 100, 100, 100]

    # Fit using least squares
    result = least_squares(
        _residuals,
        params_init,
        args=(X, Y, Z, roi),
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        max_nfev=1000
    )

    # Extract fitted parameters
    A, B, x0, y0, z0, cov_params = _unpack_params(result.x)
    cov = cov_params_to_matrix(cov_params)

    # Compute principal axes via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ensure positive (numerical issues can make them slightly negative)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    principal_sigmas = np.sqrt(eigenvalues)

    # Compute fit quality
    model = gaussian_3d(X, Y, Z, A, B, x0, y0, z0, cov)
    residuals = roi - model
    residual_rms = np.sqrt(np.mean(residuals**2))

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((roi - np.mean(roi))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Create result
    fit_result = GaussianFitResult(
        amplitude=A,
        background=B,
        center=np.array([x0, y0, z0]),  # Relative to roi_center
        covariance=cov,
        principal_axes=eigenvectors,
        principal_sigmas=principal_sigmas,
        residual_rms=residual_rms,
        r_squared=r_squared,
        roi_center=roi_center,
        roi_size=roi_size
    )

    return fit_result


def get_ellipsoid_surface(center: np.ndarray,
                          covariance: np.ndarray,
                          n_sigma: float = 2.0,
                          n_points: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate surface points for a 3D ellipsoid from covariance matrix.

    The ellipsoid represents the n_sigma contour of the Gaussian.

    Parameters
    ----------
    center : array of shape (3,)
        Ellipsoid center [x0, y0, z0]
    covariance : array of shape (3, 3)
        Covariance matrix
    n_sigma : float
        Number of standard deviations for the contour
    n_points : int
        Number of points in each angular direction

    Returns
    -------
    X, Y, Z : arrays of shape (n_points, n_points)
        Surface coordinates for plotting
    """
    # Eigendecomposition (suppress warnings for edge cases)
    with np.errstate(divide='ignore', invalid='ignore'):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Radii along principal axes
    radii = n_sigma * np.sqrt(eigenvalues)

    # Generate unit sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    U, V = np.meshgrid(u, v)

    # Sphere coordinates
    x_sphere = np.sin(V) * np.cos(U)
    y_sphere = np.sin(V) * np.sin(U)
    z_sphere = np.cos(V)

    # Scale by radii
    x_scaled = radii[0] * x_sphere
    y_scaled = radii[1] * y_sphere
    z_scaled = radii[2] * z_sphere

    # Stack and rotate
    points = np.stack([x_scaled.ravel(), y_scaled.ravel(), z_scaled.ravel()], axis=0)
    rotated = eigenvectors @ points  # (3, N)

    # Translate to center
    X = rotated[0].reshape(n_points, n_points) + center[0]
    Y = rotated[1].reshape(n_points, n_points) + center[1]
    Z = rotated[2].reshape(n_points, n_points) + center[2]

    return X, Y, Z


def evaluate_gaussian_on_grid(fit_result: GaussianFitResult,
                              grid_extent: int = 15,
                              grid_points: int = 31) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate fitted Gaussian on a regular grid for visualization.

    Parameters
    ----------
    fit_result : GaussianFitResult
        Fitted Gaussian parameters
    grid_extent : int
        Half-extent of grid in each dimension
    grid_points : int
        Number of points along each dimension

    Returns
    -------
    X, Y, Z : arrays of shape (grid_points, grid_points, grid_points)
        Coordinate grids
    values : array of same shape
        Gaussian values
    """
    x = np.linspace(-grid_extent, grid_extent, grid_points)
    y = np.linspace(-grid_extent, grid_extent, grid_points)
    z = np.linspace(-grid_extent, grid_extent, grid_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    values = gaussian_3d(
        X, Y, Z,
        fit_result.amplitude,
        fit_result.background,
        fit_result.center[0],
        fit_result.center[1],
        fit_result.center[2],
        fit_result.covariance
    )

    return X, Y, Z, values


# =============================================================================
# TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    print("Testing 3D Gaussian Fitting")
    print("=" * 50)

    # Create synthetic Gaussian data
    np.random.seed(42)

    # True parameters
    true_A = 1.0
    true_B = 0.1
    true_center = np.array([2.3, -1.7, 0.8])  # Sub-voxel
    true_cov = np.array([
        [4.0, 0.5, 0.2],
        [0.5, 3.0, 0.3],
        [0.2, 0.3, 2.0]
    ])

    # Generate grid
    x = np.arange(-10, 11)
    y = np.arange(-10, 11)
    z = np.arange(-10, 11)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Generate data with noise
    data = gaussian_3d(X, Y, Z, true_A, true_B,
                       true_center[0], true_center[1], true_center[2],
                       true_cov)
    noise = 0.02 * np.random.randn(*data.shape)
    data_noisy = data + noise

    print(f"True center: {true_center}")
    print(f"True amplitude: {true_A}, background: {true_B}")
    print(f"True covariance diagonal: [{true_cov[0,0]:.2f}, {true_cov[1,1]:.2f}, {true_cov[2,2]:.2f}]")

    # Fit
    print("\nFitting...")
    result = fit_gaussian_3d(data_noisy, roi_center=np.array([10, 10, 10]), roi_size=10)

    print(f"\nFitted center: [{result.center[0]:.3f}, {result.center[1]:.3f}, {result.center[2]:.3f}]")
    print(f"Center error: {np.linalg.norm(result.center - true_center):.4f}")
    print(f"Fitted amplitude: {result.amplitude:.3f}, background: {result.background:.3f}")
    print(f"Fitted covariance diagonal: [{result.covariance[0,0]:.2f}, {result.covariance[1,1]:.2f}, {result.covariance[2,2]:.2f}]")
    print(f"Principal sigmas: {result.principal_sigmas}")
    print(f"R-squared: {result.r_squared:.4f}")
    print(f"Residual RMS: {result.residual_rms:.4f}")
