#!/usr/bin/env python3
"""
POD Filter Test Script - Python Implementation
Load TIFFs directly, run POD, compare with MATLAB results

This is a standalone test that matches the MATLAB implementation exactly.

Usage:
    python test_pod_python.py /path/to/test/data [num_images]

    Default data path: /Users/morgan/Documents/PIV_test
    Default num_images: 50
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Optional: for MATLAB comparison
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not available - MATLAB comparison will be skipped")

# Optional: for figure generation
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - figures will be skipped")

# Configuration (can be overridden via command line)
DEFAULT_DATA_PATH = Path("/Users/morgan/Documents/PIV_test")
DEFAULT_NUM_IMAGES = 50

# POD parameters (must match MATLAB exactly)
EPS_AUTO_PSI = 0.01
EPS_AUTO_SIGMA = 0.01


def load_images(data_path: Path, num_images: int):
    """Load TIFF pairs directly from disk."""
    print(f"Loading {num_images} image pairs from {data_path}")

    # Get image size from first file
    first_img = np.array(Image.open(data_path / "B00001_A.tif"))
    H, W = first_img.shape
    print(f"Image size: {H} x {W}")
    print(f"First image dtype: {first_img.dtype}, range: [{first_img.min()}, {first_img.max()}]")

    # Preallocate as float32 (matches MATLAB 'single')
    M_A = np.zeros((num_images, H * W), dtype=np.float32)
    M_B = np.zeros((num_images, H * W), dtype=np.float32)

    # Load all images
    for i in range(1, num_images + 1):
        fname_A = data_path / f"B{i:05d}_A.tif"
        fname_B = data_path / f"B{i:05d}_B.tif"

        img_A = np.array(Image.open(fname_A), dtype=np.float32)
        img_B = np.array(Image.open(fname_B), dtype=np.float32)

        M_A[i-1, :] = img_A.ravel()
        M_B[i-1, :] = img_B.ravel()

    print(f"Loaded {num_images} pairs")
    print(f"Data range A: [{M_A.min():.2f}, {M_A.max():.2f}]")
    print(f"Data range B: [{M_B.min():.2f}, {M_B.max():.2f}]")
    return M_A, M_B, H, W


def find_auto_mode(PSI, eigvals, N, frame_name=""):
    """
    Automatic mode selection (matching MATLAB logic exactly).

    Returns the number of modes to remove (1-indexed count to match MATLAB).
    """
    # Use eigenvalue at middle index for normalization (matches MATLAB round(N/2))
    mid_idx = round(N / 2) - 1  # -1 for 0-indexing, round() matches MATLAB
    norm_factor = eigvals[mid_idx] if eigvals[mid_idx] > 1e-10 else 1.0
    threshold = EPS_AUTO_SIGMA * eigvals[0]

    print(f"\nMode selection (norm_factor={norm_factor:.2e}, threshold={threshold:.2e}):")

    for i in range(N - 1):
        mean_psi = np.abs(np.mean(PSI[:, i]))
        sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / norm_factor

        if mean_psi < EPS_AUTO_PSI and sig_diff < threshold:
            print(f"  Mode {i+1:2d}: Mean_PSI={mean_psi:.6f}, Sig_Diff={sig_diff:.4e} -> NOISE FLOOR DETECTED")
            return i + 1  # Return count (1-indexed to match MATLAB)
        else:
            if i < 10:  # Only print first 10 modes
                print(f"  Mode {i+1:2d}: Mean_PSI={mean_psi:.6f}, Sig_Diff={sig_diff:.4e}")

    return 0


def pod_filter_current_python_method(M, n_remove):
    """
    Current Python implementation: PSI @ PSI.T @ M projection.

    This matches the existing filters.py code exactly.
    """
    if n_remove == 0:
        return M.copy()

    N = M.shape[0]

    # Covariance (float32, matching filters.py line 138)
    Cov = M @ M.T

    # SVD
    PSI, S, _ = np.linalg.svd(Cov, full_matrices=False)

    # Project and subtract
    PSI_bad = PSI[:, :n_remove]
    t_coeffs = PSI_bad.T @ M
    bad_signal = PSI_bad @ t_coeffs

    return M - bad_signal


def pod_filter_matlab_method(M, PSI, n_remove):
    """
    MATLAB's method: explicit PHI/TCoeff computation.

    PHI = M.T @ PSI (normalized)
    TCoeff = M @ PHI
    Subtract: sum_i(TCoeff_i * PHI_i.T)
    """
    if n_remove == 0:
        return M.copy()

    N = M.shape[0]

    # Compute PHI and TCoeff like MATLAB
    PHI_list = []
    TCoeff_list = []

    for mod_i in range(n_remove):
        # PHI = M.T @ PSI[:, mod_i], then normalize
        phi = M.T @ PSI[:, mod_i]
        phi_norm = np.sqrt(np.sum(phi ** 2))
        if phi_norm > 1e-10:
            phi = phi / phi_norm
        PHI_list.append(phi)

        # TCoeff = M @ PHI
        tcoeff = M @ phi
        TCoeff_list.append(tcoeff)

    # Subtract modes (exactly like MATLAB loop)
    M_filtered = M.copy()
    for i in range(N):
        for mod_i in range(n_remove):
            M_filtered[i, :] -= TCoeff_list[mod_i][i] * PHI_list[mod_i]

    return M_filtered


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        DATA_PATH = Path(sys.argv[1])
    else:
        DATA_PATH = DEFAULT_DATA_PATH

    if len(sys.argv) > 2:
        NUM_IMAGES = int(sys.argv[2])
    else:
        NUM_IMAGES = DEFAULT_NUM_IMAGES

    print("=" * 60)
    print("POD Filter Test - Python Implementation")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Num images: {NUM_IMAGES}")

    # Load images
    M_A, M_B, H, W = load_images(DATA_PATH, NUM_IMAGES)
    N = NUM_IMAGES

    # ========== FRAME A ==========
    print("\n" + "=" * 50)
    print("Processing Frame A")
    print("=" * 50)

    # Covariance (matching filters.py - uses float32)
    print("\nComputing covariance (using float32, matching filters.py)...")
    Cov_A = M_A @ M_A.T
    print(f"Covariance matrix: min={Cov_A.min():.2e}, max={Cov_A.max():.2e}")
    print(f"Covariance has NaN: {np.isnan(Cov_A).any()}, Inf: {np.isinf(Cov_A).any()}")

    # SVD
    print("\nComputing SVD...")
    PSI_A, S_A, _ = np.linalg.svd(Cov_A, full_matrices=False)
    print(f"Eigenvalues: max={S_A.max():.2e}, min={S_A.min():.2e}")
    print(f"PSI has NaN: {np.isnan(PSI_A).any()}")

    # Auto mode selection
    N_auto_A = find_auto_mode(PSI_A, S_A, N, "A")
    print(f"Frame A: Removing {N_auto_A} modes")

    # ========== FRAME B ==========
    print("\n" + "=" * 50)
    print("Processing Frame B")
    print("=" * 50)

    Cov_B = M_B @ M_B.T
    print(f"Covariance matrix: min={Cov_B.min():.2e}, max={Cov_B.max():.2e}")
    print(f"Covariance has NaN: {np.isnan(Cov_B).any()}, Inf: {np.isinf(Cov_B).any()}")

    PSI_B, S_B, _ = np.linalg.svd(Cov_B, full_matrices=False)
    print(f"Eigenvalues: max={S_B.max():.2e}, min={S_B.min():.2e}")

    N_auto_B = find_auto_mode(PSI_B, S_B, N, "B")
    print(f"Frame B: Removing {N_auto_B} modes")

    # ========== APPLY FILTER (BOTH METHODS) ==========
    print("\n" + "=" * 50)
    print("Applying POD Filter")
    print("=" * 50)

    print("\nMethod 1: Current Python (PSI @ PSI.T @ M)...")
    M_A_filtered_py = pod_filter_current_python_method(M_A, N_auto_A)
    M_B_filtered_py = pod_filter_current_python_method(M_B, N_auto_B)

    print("Method 2: MATLAB-style (PHI/TCoeff)...")
    M_A_filtered_mat = pod_filter_matlab_method(M_A, PSI_A, N_auto_A)
    M_B_filtered_mat = pod_filter_matlab_method(M_B, PSI_B, N_auto_B)

    # ========== COMPARE METHODS ==========
    print("\n" + "=" * 50)
    print("Method Comparison (Python vs MATLAB-style in Python)")
    print("=" * 50)
    diff_A = np.abs(M_A_filtered_py - M_A_filtered_mat)
    diff_B = np.abs(M_B_filtered_py - M_B_filtered_mat)
    print(f"Frame A: max diff = {diff_A.max():.2e}, mean diff = {diff_A.mean():.2e}")
    print(f"Frame B: max diff = {diff_B.max():.2e}, mean diff = {diff_B.mean():.2e}")

    if diff_A.max() < 1e-3 and diff_B.max() < 1e-3:
        print("PASS: Both methods produce nearly identical results")
    else:
        print("WARNING: Methods produce different results!")

    # ========== SAVE RESULTS ==========
    output_path = DATA_PATH / "python_pod_results"
    output_path.mkdir(exist_ok=True)

    # Save filtered images (use MATLAB-style method for comparison)
    img_A_out = M_A_filtered_mat[0].reshape(H, W)
    img_B_out = M_B_filtered_mat[0].reshape(H, W)

    Image.fromarray(np.clip(img_A_out, 0, 65535).astype(np.uint16)).save(
        output_path / "filtered_A_001.tif"
    )
    Image.fromarray(np.clip(img_B_out, 0, 65535).astype(np.uint16)).save(
        output_path / "filtered_B_001.tif"
    )

    # Save numerical data
    np.savez(
        output_path / "pod_results.npz",
        N_auto_A=N_auto_A,
        N_auto_B=N_auto_B,
        eigVal_A=S_A,
        eigVal_B=S_B,
        M_A_filtered=M_A_filtered_mat,  # Save MATLAB-style for comparison
        M_B_filtered=M_B_filtered_mat,
        H=H,
        W=W,
        num_images=NUM_IMAGES,
    )

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Modes removed:  Frame A = {N_auto_A}, Frame B = {N_auto_B}")
    print(f"Output range A: [{M_A_filtered_mat.min():.2f}, {M_A_filtered_mat.max():.2f}]")
    print(f"Output range B: [{M_B_filtered_mat.min():.2f}, {M_B_filtered_mat.max():.2f}]")
    print(f"Has NaN A: {np.isnan(M_A_filtered_mat).any()}")
    print(f"Has NaN B: {np.isnan(M_B_filtered_mat).any()}")
    print(f"Has negative A: {(M_A_filtered_mat < 0).any()} (min={M_A_filtered_mat.min():.2f})")
    print(f"Has negative B: {(M_B_filtered_mat < 0).any()} (min={M_B_filtered_mat.min():.2f})")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    # ========== MATLAB COMPARISON ==========
    matlab_results = DATA_PATH / "matlab_pod_results" / "pod_results.mat"
    if HAS_SCIPY and matlab_results.exists():
        print("\n" + "=" * 60)
        print("MATLAB COMPARISON")
        print("=" * 60)

        mat = sio.loadmat(str(matlab_results))

        mat_N_auto_A = int(mat['N_auto_A'].squeeze())
        mat_N_auto_B = int(mat['N_auto_B'].squeeze())

        print(f"Mode counts:")
        print(f"  MATLAB:  Frame A = {mat_N_auto_A}, Frame B = {mat_N_auto_B}")
        print(f"  Python:  Frame A = {N_auto_A}, Frame B = {N_auto_B}")
        print(f"  Match:   A = {mat_N_auto_A == N_auto_A}, B = {mat_N_auto_B == N_auto_B}")

        # Compare eigenvalues
        mat_eigVal_A = mat['eigVal_A'].squeeze()
        mat_eigVal_B = mat['eigVal_B'].squeeze()

        eigval_diff_A = np.abs(S_A - mat_eigVal_A)
        eigval_diff_B = np.abs(S_B - mat_eigVal_B)
        print(f"\nEigenvalue comparison:")
        print(f"  Frame A: max diff = {eigval_diff_A.max():.2e}, relative = {(eigval_diff_A / mat_eigVal_A).max():.2e}")
        print(f"  Frame B: max diff = {eigval_diff_B.max():.2e}, relative = {(eigval_diff_B / mat_eigVal_B).max():.2e}")

        # Compare filtered output (use Fortran order for MATLAB data)
        # MATLAB is column-major, Python is row-major
        mat_A_img = mat['M_A_filtered'][0].reshape((H, W), order='F').astype(np.float32)
        mat_B_img = mat['M_B_filtered'][0].reshape((H, W), order='F').astype(np.float32)

        py_A_img = M_A_filtered_mat[0].reshape(H, W)
        py_B_img = M_B_filtered_mat[0].reshape(H, W)

        orig_A_img = M_A[0].reshape(H, W)
        orig_B_img = M_B[0].reshape(H, W)

        diff_A_img = py_A_img - mat_A_img
        diff_B_img = py_B_img - mat_B_img

        print(f"\nFiltered output comparison (with correct ordering):")
        print(f"  Frame A: max diff = {np.abs(diff_A_img).max():.2e}, mean diff = {np.abs(diff_A_img).mean():.2e}")
        print(f"  Frame B: max diff = {np.abs(diff_B_img).max():.2e}, mean diff = {np.abs(diff_B_img).mean():.2e}")
        print(f"  Frame A: correlation = {np.corrcoef(py_A_img.ravel(), mat_A_img.ravel())[0,1]:.6f}")
        print(f"  Frame B: correlation = {np.corrcoef(py_B_img.ravel(), mat_B_img.ravel())[0,1]:.6f}")

        if np.abs(diff_A_img).max() < 1.0 and np.abs(diff_B_img).max() < 1.0:
            print("\nPASS: Python and MATLAB produce nearly identical results")
        else:
            print("\nWARNING: Significant differences between Python and MATLAB!")

        print("=" * 60)

        # ========== SAVE COMPARISON FIGURE ==========
        if HAS_MATPLOTLIB:
            print("\nGenerating comparison figure...")

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # Frame A row
            # Original
            im0 = axes[0, 0].imshow(orig_A_img, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title(f'Original A\n[{orig_A_img.min():.0f}, {orig_A_img.max():.0f}]')
            axes[0, 0].axis('off')

            # Python filtered
            vmin_filt = min(py_A_img.min(), mat_A_img.min())
            vmax_filt = max(py_A_img.max(), mat_A_img.max())
            im1 = axes[0, 1].imshow(py_A_img, cmap='gray', vmin=vmin_filt, vmax=vmax_filt)
            axes[0, 1].set_title(f'Python Filtered A\n[{py_A_img.min():.1f}, {py_A_img.max():.1f}]')
            axes[0, 1].axis('off')

            # MATLAB filtered
            im2 = axes[0, 2].imshow(mat_A_img, cmap='gray', vmin=vmin_filt, vmax=vmax_filt)
            axes[0, 2].set_title(f'MATLAB Filtered A\n[{mat_A_img.min():.1f}, {mat_A_img.max():.1f}]')
            axes[0, 2].axis('off')

            # Difference
            diff_max = max(np.abs(diff_A_img).max(), 0.1)
            im3 = axes[0, 3].imshow(diff_A_img, cmap='RdBu', vmin=-diff_max, vmax=diff_max)
            axes[0, 3].set_title(f'Difference (Py - MAT)\nmax={np.abs(diff_A_img).max():.2f}')
            axes[0, 3].axis('off')
            plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

            # Frame B row
            # Original
            im4 = axes[1, 0].imshow(orig_B_img, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title(f'Original B\n[{orig_B_img.min():.0f}, {orig_B_img.max():.0f}]')
            axes[1, 0].axis('off')

            # Python filtered
            vmin_filt_B = min(py_B_img.min(), mat_B_img.min())
            vmax_filt_B = max(py_B_img.max(), mat_B_img.max())
            im5 = axes[1, 1].imshow(py_B_img, cmap='gray', vmin=vmin_filt_B, vmax=vmax_filt_B)
            axes[1, 1].set_title(f'Python Filtered B\n[{py_B_img.min():.1f}, {py_B_img.max():.1f}]')
            axes[1, 1].axis('off')

            # MATLAB filtered
            im6 = axes[1, 2].imshow(mat_B_img, cmap='gray', vmin=vmin_filt_B, vmax=vmax_filt_B)
            axes[1, 2].set_title(f'MATLAB Filtered B\n[{mat_B_img.min():.1f}, {mat_B_img.max():.1f}]')
            axes[1, 2].axis('off')

            # Difference
            diff_max_B = max(np.abs(diff_B_img).max(), 0.1)
            im7 = axes[1, 3].imshow(diff_B_img, cmap='RdBu', vmin=-diff_max_B, vmax=diff_max_B)
            axes[1, 3].set_title(f'Difference (Py - MAT)\nmax={np.abs(diff_B_img).max():.2f}')
            axes[1, 3].axis('off')
            plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)

            # Add summary text
            fig.suptitle(f'POD Filter Comparison: Python vs MATLAB\n'
                        f'Modes removed: A={N_auto_A}, B={N_auto_B} | '
                        f'Max diff: {max(np.abs(diff_A_img).max(), np.abs(diff_B_img).max()):.2f}',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()

            fig_path = output_path / "pod_comparison.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Comparison figure saved to: {fig_path}")
    elif not matlab_results.exists():
        print(f"\nNote: Run MATLAB script first to enable comparison")
        print(f"      Expected: {matlab_results}")


if __name__ == "__main__":
    main()
