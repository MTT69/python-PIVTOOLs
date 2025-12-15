"""
3D Correlation Module for Stereo Ensemble PIV
==============================================

Implements:
1. MLOS (Multiplicative Line-of-Sight) volume reconstruction from stereo images
2. 3D FFT-based ensemble correlation accumulation

The key concept:
- Auto-correlation: R_AA = <A * A> captures the particle shape (geometric spread)
- Cross-correlation: R_AB = <A * B> captures shape + displacement + turbulent broadening
- The difference in widths gives the Reynolds stress tensor
"""

import numpy as np
from scipy.ndimage import map_coordinates
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .stereo_ensemble_generator import StereoCamera


# =============================================================================
# MLOS RECONSTRUCTION
# =============================================================================
class StereoMLOSReconstructor:
    """
    MLOS (Multiplicative Line-of-Sight) volume reconstruction from stereo images.

    For each voxel:
    1. Project to each camera image using pinhole model
    2. Sample the intensity at that pixel location (bilinear interpolation)
    3. Multiply intensities from all cameras

    Real particles: High intensity in ALL cameras -> high product
    Empty space: Low in at least one camera -> low product
    Ghosts: Can appear where lines of sight cross incorrectly
    """

    def __init__(
        self,
        cameras: List['StereoCamera'],
        volume_shape: Tuple[int, int, int],
        scale_px_per_mm: float
    ):
        """
        Initialize with precomputed projection maps.

        Parameters
        ----------
        cameras : list of StereoCamera
            Camera models with project() method
        volume_shape : tuple (nx, ny, nz)
            Voxel dimensions of reconstructed volume
        scale_px_per_mm : float
            Scale factor for coordinate conversion
        """
        self.cameras = cameras
        self.volume_shape = volume_shape
        self.scale = scale_px_per_mm

        # Precompute projection maps for efficiency
        self.projection_maps = self._precompute_projections()

    def _precompute_projections(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Precompute (u, v) pixel coordinates for each voxel in each camera.

        Returns
        -------
        projection_maps : list of (u_map, v_map) tuples
            Each map has shape (nx, ny, nz)
        """
        nx, ny, nz = self.volume_shape

        # Create voxel grid centered at origin (in mm)
        x_mm = (np.arange(nx) - nx / 2 + 0.5) / self.scale
        y_mm = (np.arange(ny) - ny / 2 + 0.5) / self.scale
        z_mm = (np.arange(nz) - nz / 2 + 0.5) / self.scale

        X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')
        voxel_coords_mm = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        projection_maps = []
        for cam in self.cameras:
            pixels, valid = cam.project(voxel_coords_mm)
            u_map = pixels[:, 0].reshape(self.volume_shape)
            v_map = pixels[:, 1].reshape(self.volume_shape)
            projection_maps.append((u_map, v_map))

        return projection_maps

    def reconstruct(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Perform MinLOS reconstruction from camera images.

        Uses MINIMUM instead of multiplication to suppress ghost artifacts.
        Ghost rays only appear bright in one camera, so min() selects
        the low/zero value from the other camera.

        Parameters
        ----------
        images : list of ndarray
            One image per camera, shape (H, W)

        Returns
        -------
        volume : ndarray, shape (nx, ny, nz)
            Reconstructed intensity volume
        """
        # Initialize with infinity (will be replaced by min)
        volume = np.full(self.volume_shape, np.inf, dtype=np.float64)

        for image, (u_map, v_map) in zip(images, self.projection_maps):
            # Bilinear interpolation using map_coordinates
            # Note: map_coordinates expects (row, col) = (v, u)
            coords = np.array([v_map.ravel(), u_map.ravel()])
            sampled = map_coordinates(
                image, coords,
                order=1,        # Bilinear interpolation
                mode='constant',
                cval=0.0        # Zero outside image bounds
            )
            sampled = sampled.reshape(self.volume_shape)

            # Take MINIMUM instead of multiply (MinLOS)
            # This suppresses ghost "arms" that only exist in one camera
            volume = np.minimum(volume, sampled)

        # Replace any remaining inf with 0 (shouldn't happen with 2+ cameras)
        volume = np.where(np.isinf(volume), 0.0, volume)

        return volume


# =============================================================================
# ENSEMBLE ACCUMULATOR
# =============================================================================
class EnsembleAccumulator3D:
    """
    Accumulates 3D correlation volumes across image pairs in frequency domain.

    For efficiency, we accumulate in frequency domain:
        Sum_Auto_FFT  += FFT(A) * conj(FFT(A))
        Sum_Cross_FFT += FFT(A) * conj(FFT(B))

    Then at the end:
        R_AA = IFFT(Sum_Auto_FFT) / N
        R_AB = IFFT(Sum_Cross_FFT) / N

    This is mathematically equivalent to averaging the correlation volumes,
    but more efficient since FFT is computed only once per volume.
    """

    def __init__(self, volume_shape: Tuple[int, int, int], normalize: bool = True):
        """
        Initialize accumulator.

        Parameters
        ----------
        volume_shape : tuple (nx, ny, nz)
            Shape of correlation volumes
        normalize : bool
            Whether to subtract mean before FFT (recommended for better SNR)
        """
        self.volume_shape = volume_shape
        self.normalize = normalize

        # FFT accumulators (complex-valued)
        self.sum_auto_fft = np.zeros(volume_shape, dtype=np.complex128)
        self.sum_cross_fft = np.zeros(volume_shape, dtype=np.complex128)

        # Volume accumulators for background subtraction (single-pass optimization)
        # These store the sum of raw volumes for computing mean images
        self.sum_vol_A = np.zeros(volume_shape, dtype=np.float64)
        self.sum_vol_B = np.zeros(volume_shape, dtype=np.float64)

        self.count = 0

    def reset(self) -> None:
        """Reset accumulators to zero."""
        self.sum_auto_fft.fill(0)
        self.sum_cross_fft.fill(0)
        self.sum_vol_A.fill(0)
        self.sum_vol_B.fill(0)
        self.count = 0

    def accumulate(self, vol_A: np.ndarray, vol_B: np.ndarray) -> None:
        """
        Add one volume pair to the ensemble.

        Parameters
        ----------
        vol_A : ndarray, shape (nx, ny, nz)
            Reference volume (time t)
        vol_B : ndarray, shape (nx, ny, nz)
            Displaced volume (time t+dt)
        """
        # Accumulate raw volumes for background subtraction
        # (must be done BEFORE mean subtraction)
        self.sum_vol_A += vol_A
        self.sum_vol_B += vol_B

        # Mean subtraction for better SNR (removes DC component)
        if self.normalize:
            A = vol_A - vol_A.mean()
            B = vol_B - vol_B.mean()
        else:
            A = vol_A
            B = vol_B

        # Compute FFTs
        F_A = np.fft.fftn(A)
        F_B = np.fft.fftn(B)

        # Accumulate auto-correlation: F_A * conj(F_A) = |F_A|^2
        self.sum_auto_fft += F_A * np.conj(F_A)

        # Accumulate cross-correlation: F_A * conj(F_B)
        self.sum_cross_fft += F_A * np.conj(F_B)

        self.count += 1

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute final ensemble-averaged correlation maps with background subtraction.

        Implements single-pass optimization formula:
            R_AA = <A⋆A> - <A>⋆<A>
            R_AB = <A⋆B> - <A>⋆<B>

        The background term <A>⋆<B> (correlation of mean volumes) captures
        ghost/spurious correlations from MLOS reconstruction artifacts.

        Returns
        -------
        map_auto : ndarray, shape (nx, ny, nz)
            Ensemble-averaged auto-correlation (peak at center)
        map_cross : ndarray, shape (nx, ny, nz)
            Ensemble-averaged cross-correlation (peak shifted by displacement)
        """
        if self.count == 0:
            raise RuntimeError("No volumes accumulated - call accumulate() first")

        # Step 1: Compute mean volumes
        vol_A_mean = self.sum_vol_A / self.count
        vol_B_mean = self.sum_vol_B / self.count

        # Step 2: Compute raw ensemble correlations (existing approach)
        avg_auto_fft = self.sum_auto_fft / self.count
        avg_cross_fft = self.sum_cross_fft / self.count

        # IFFT and take real part
        map_auto_raw = np.real(np.fft.ifftn(avg_auto_fft))
        map_cross_raw = np.real(np.fft.ifftn(avg_cross_fft))

        # fftshift to center the zero-lag
        map_auto_raw = np.fft.fftshift(map_auto_raw)
        map_cross_raw = np.fft.fftshift(map_cross_raw)

        # Step 3: Compute background correlations from mean volumes
        # This captures the ghost/artifact correlations that should be subtracted
        R_AA_bg = correlate_3d(vol_A_mean, vol_A_mean, normalize=self.normalize)
        R_AB_bg = correlate_3d(vol_A_mean, vol_B_mean, normalize=self.normalize)

        # Step 4: Apply background subtraction (SINGLE-PASS OPTIMIZATION)
        # R_ensemble = <A⋆B> - <A>⋆<B>
        map_auto = map_auto_raw - R_AA_bg
        map_cross = map_cross_raw - R_AB_bg

        return map_auto, map_cross


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def correlate_3d(vol_A: np.ndarray, vol_B: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute 3D cross-correlation between two volumes.

    Parameters
    ----------
    vol_A : ndarray, shape (nx, ny, nz)
        First volume
    vol_B : ndarray, shape (nx, ny, nz)
        Second volume
    normalize : bool
        Whether to subtract mean before correlation

    Returns
    -------
    corr : ndarray
        Cross-correlation volume (zero-lag at center)
    """
    if normalize:
        A = vol_A - vol_A.mean()
        B = vol_B - vol_B.mean()
    else:
        A = vol_A
        B = vol_B

    F_A = np.fft.fftn(A)
    F_B = np.fft.fftn(B)

    corr = np.fft.fftshift(np.real(np.fft.ifftn(F_A * np.conj(F_B))))

    return corr


def find_correlation_peak(
    corr_volume: np.ndarray,
    subpixel: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Find the peak location in a correlation volume.

    Parameters
    ----------
    corr_volume : ndarray
        Correlation volume (zero-lag at center)
    subpixel : bool
        Whether to use parabolic interpolation for sub-voxel accuracy

    Returns
    -------
    peak_location : ndarray, shape (3,)
        Peak location relative to volume center [dx, dy, dz]
    peak_value : float
        Value at peak
    """
    shape = np.array(corr_volume.shape)
    center = shape // 2

    # Find integer peak
    peak_idx = np.unravel_index(np.argmax(corr_volume), corr_volume.shape)
    peak_idx = np.array(peak_idx)
    peak_value = corr_volume[tuple(peak_idx)]

    if not subpixel:
        return peak_idx - center, peak_value

    # Sub-voxel refinement using 3-point parabolic interpolation
    displacement = np.zeros(3)
    for dim in range(3):
        idx = peak_idx.copy()

        # Check bounds
        if idx[dim] <= 0 or idx[dim] >= shape[dim] - 1:
            displacement[dim] = peak_idx[dim] - center[dim]
            continue

        # Get three points along this dimension
        idx_m = idx.copy()
        idx_m[dim] -= 1
        idx_p = idx.copy()
        idx_p[dim] += 1

        f_m = corr_volume[tuple(idx_m)]
        f_0 = corr_volume[tuple(idx)]
        f_p = corr_volume[tuple(idx_p)]

        # Parabolic interpolation
        denom = 2 * (f_m - 2 * f_0 + f_p)
        if abs(denom) > 1e-10:
            delta = (f_m - f_p) / denom
        else:
            delta = 0.0

        displacement[dim] = peak_idx[dim] + delta - center[dim]

    return displacement, peak_value


# =============================================================================
# TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    print("Testing 3D Correlation Module")
    print("=" * 50)

    # Create synthetic test volumes
    shape = (64, 64, 16)
    center = np.array(shape) // 2

    # Volume A: Gaussian at center
    x = np.arange(shape[0]) - center[0]
    y = np.arange(shape[1]) - center[1]
    z = np.arange(shape[2]) - center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    sigma = 3.0
    vol_A = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))

    # Volume B: Same Gaussian shifted by (2, 1, 0.5) voxels
    displacement_true = np.array([2.0, 1.0, 0.5])
    vol_B = np.exp(-((X - displacement_true[0])**2 +
                     (Y - displacement_true[1])**2 +
                     (Z - displacement_true[2])**2) / (2 * sigma**2))

    print(f"Volume shape: {shape}")
    print(f"True displacement: {displacement_true}")

    # Test single correlation
    corr = correlate_3d(vol_A, vol_B)
    peak_loc, peak_val = find_correlation_peak(corr)
    print(f"\nSingle correlation peak: {peak_loc}")
    print(f"Peak value: {peak_val:.4f}")

    # Test ensemble accumulator
    print("\nTesting ensemble accumulator...")
    accumulator = EnsembleAccumulator3D(shape)

    # Add some noise realizations
    for i in range(10):
        noise_A = vol_A + 0.1 * np.random.randn(*shape)
        noise_B = vol_B + 0.1 * np.random.randn(*shape)
        accumulator.accumulate(noise_A, noise_B)

    map_auto, map_cross = accumulator.finalize()
    print(f"Auto-correlation shape: {map_auto.shape}")
    print(f"Cross-correlation shape: {map_cross.shape}")

    peak_loc, peak_val = find_correlation_peak(map_cross)
    print(f"Ensemble cross-correlation peak: {peak_loc}")
    print(f"Error: {np.linalg.norm(peak_loc - displacement_true):.4f} voxels")

    print("\nDone!")
