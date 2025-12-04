import unittest

import numpy as np
import pandas as pd


def compare_matrices(matlab_uy, python_uy, precision=6):
    """
    Compare two UY matrices (MATLAB vs Python) and summarize differences.

    Parameters
    ----------
    matlab_uy : 2D array_like
        MATLAB matrix.
    python_uy : 2D array_like
        Python matrix (same shape as matlab_uy).
    precision : int
        Number of decimal places to print.

    Returns
    -------
    summary_df : pandas DataFrame
        A table with MATLAB, Python, Diff, and Relative Diff for each element.
    """
    if matlab_uy.shape != python_uy.shape:
        raise ValueError(
            f"Matrices must have the same shape: {matlab_uy.shape} != {python_uy.shape}"
        )

    # Flatten for element-wise comparison
    matlab_flat = matlab_uy.ravel()
    python_flat = python_uy.ravel()

    # Differences
    diff = python_flat - matlab_flat
    rel_diff = np.where(matlab_flat != 0, diff / matlab_flat, 0)

    # Prepare a DataFrame for easy inspection
    summary_df = pd.DataFrame(
        {
            "MATLAB": np.round(matlab_flat, precision),
            "Python": np.round(python_flat, precision),
            "Diff": np.round(diff, precision),
            "RelDiff (%)": np.round(100 * rel_diff, precision),
        }
    )

    # Summary stats
    stats = {
        "Mean Absolute Diff": np.mean(np.abs(diff)),
        "Max Absolute Diff": np.max(np.abs(diff)),
        "Min Absolute Diff": np.min(np.abs(diff)),
        "Mean Relative Diff (%)": np.mean(np.abs(rel_diff)) * 100,
        "Max Relative Diff (%)": np.max(np.abs(rel_diff)) * 100,
        "Min Relative Diff (%)": np.min(np.abs(rel_diff)) * 100,
    }

    print("=== Full Element-wise Comparison ===")
    pd.set_option("display.max_rows", None)

    pd.set_option("display.max_columns", None)
    print(summary_df)
    print("\n=== Summary Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    print()
    return summary_df, stats


def assert_arrays_close(
    testcase: unittest.TestCase, arr1: np.ndarray, arr2: np.ndarray, tol=1e-3
):
    """Assert that two arrays are close within a tolerance.

    :param testcase: The unittest.TestCase instance
    :type testcase: unittest.TestCase
    :param arr1: first array to compare
    :type arr1: np.ndarray
    :param arr2: second array to compare
    :type arr2: np.ndarray
    :param tol: tolerance for comparison, defaults to 1e-3
    :type tol: float, optional
    :param name: name of the arrays, defaults to "array"
    :type name: str, optional
    """

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    arr1[np.isnan(arr1)] = 0.0
    arr2[np.isnan(arr2)] = 0.0
    testcase.assertEqual(arr1.shape, arr2.shape)
    max_diff = np.max(np.abs(arr1 - arr2))
    if max_diff > tol:
        print(f"Max difference: {max_diff}")

    bad = ~np.isclose(arr1, arr2, atol=tol, rtol=0, equal_nan=True)
    if bad.any():
        print(f"Number of differing elements: {np.sum(bad)} out of {arr1.size}")

        for idx in np.argwhere(bad):
            print(
                f"Index {tuple(idx)}: arr1={arr1[tuple(idx)]}, arr2={arr2[tuple(idx)]}"
            )
    testcase.assertTrue(
        np.allclose(arr1, arr2, atol=tol, rtol=0, equal_nan=True),
    )
