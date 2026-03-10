"""
Regression tests for spatial filters in _apply_spatial_filters_numpy.

Tests each filter type (gaussian, median, norm, maxnorm, lmax) against
inline scipy reference implementations, plus pixel mask + spatial ordering
and the unified apply_all_filters_slim pipeline.
"""

import numpy as np
import pytest
from scipy.ndimage import (
    gaussian_filter as scipy_gaussian,
    median_filter as scipy_median,
    maximum_filter as scipy_maximum,
    minimum_filter as scipy_minimum,
    uniform_filter as scipy_uniform,
)

from pivtools_cli.processing.dask_pipeline import (
    _apply_spatial_filters_numpy,
    apply_all_filters_slim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_block():
    """Float32 block (4, 2, 64, 64) with fixed seed."""
    rng = np.random.RandomState(42)
    return rng.rand(4, 2, 64, 64).astype(np.float32) * 255.0


# ---------------------------------------------------------------------------
# Per-filter golden-value tests
# ---------------------------------------------------------------------------

class TestGaussianFilter:
    def test_matches_scipy(self, synthetic_block):
        spec = [{"type": "gaussian", "sigma": 1.5}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_gaussian(synthetic_block, sigma=(0, 0, 1.5, 1.5))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_shape_dtype(self, synthetic_block):
        spec = [{"type": "gaussian", "sigma": 1.0}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        assert result.shape == synthetic_block.shape
        assert result.dtype == synthetic_block.dtype


class TestMedianFilter:
    def test_matches_scipy(self, synthetic_block):
        spec = [{"type": "median", "size": [5, 5]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_median(synthetic_block, size=(1, 1, 5, 5))
        np.testing.assert_array_equal(result, expected)

    def test_even_size_made_odd(self, synthetic_block):
        """Even sizes are bumped to odd."""
        spec = [{"type": "median", "size": [4, 6]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_median(synthetic_block, size=(1, 1, 5, 7))
        np.testing.assert_array_equal(result, expected)


class TestNormFilter:
    def test_matches_reference(self, synthetic_block):
        size = (7, 7)
        max_gain = 2.0
        spec = [{"type": "norm", "size": list(size), "max_gain": max_gain}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        # Inline reference
        spatial_size = (1, 1) + size
        block_float = synthetic_block.astype(np.float32)
        local_min = scipy_minimum(block_float, size=spatial_size)
        local_max = scipy_maximum(block_float, size=spatial_size)
        denom = np.maximum(local_max - local_min, 1.0 / max_gain)
        expected = ((block_float - local_min) / denom).astype(synthetic_block.dtype)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_output_range(self, synthetic_block):
        spec = [{"type": "norm", "size": [7, 7], "max_gain": 1.0}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        assert result.min() >= -0.01  # should be ~0
        assert result.max() <= 1.01   # should be ~1


class TestMaxnormFilter:
    def test_matches_reference(self, synthetic_block):
        size = (7, 7)
        max_gain = 2.0
        spec = [{"type": "maxnorm", "size": list(size), "max_gain": max_gain}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        # Inline reference
        spatial_size = (1, 1) + size
        block_float = synthetic_block.astype(np.float32)
        local_max = scipy_maximum(block_float, size=spatial_size)
        smoothed_max = scipy_uniform(local_max, size=spatial_size)
        denom = np.maximum(smoothed_max, 1.0 / max_gain)
        expected = (np.maximum(block_float, 0) / denom).astype(synthetic_block.dtype)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestLmaxFilter:
    def test_matches_scipy(self, synthetic_block):
        spec = [{"type": "lmax", "size": [7, 7]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_maximum(synthetic_block, size=(1, 1, 7, 7))
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Pixel mask + spatial ordering
# ---------------------------------------------------------------------------

class TestIntegerDtypeConversion:
    def test_uint16_norm_no_underflow(self):
        """uint16 input is promoted to float32 before in-place subtraction."""
        rng = np.random.RandomState(99)
        block = (rng.rand(2, 2, 32, 32) * 1000).astype(np.uint16)
        spec = [{"type": "norm", "size": [5, 5], "max_gain": 1.0}]
        result = _apply_spatial_filters_numpy(block.copy(), spec)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert result.min() >= 0.0

    def test_uint16_maxnorm_no_underflow(self):
        """uint16 input is promoted to float32 for maxnorm."""
        rng = np.random.RandomState(99)
        block = (rng.rand(2, 2, 32, 32) * 1000).astype(np.uint16)
        spec = [{"type": "maxnorm", "size": [5, 5], "max_gain": 1.0}]
        result = _apply_spatial_filters_numpy(block.copy(), spec)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))


class TestPixelMaskThenSpatial:
    def test_mask_applied_before_spatial(self):
        """Pixel mask zeros should be present before spatial filter runs."""
        rng = np.random.RandomState(123)
        block = rng.rand(2, 2, 32, 32).astype(np.float32) * 100.0
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True  # mask center

        spatial_specs = [{"type": "gaussian", "sigma": 2.0}]
        result = apply_all_filters_slim(
            block, spatial_specs, temporal_specs=[], pixel_mask=mask,
        )

        # After masking + gaussian, the center of the masked region should
        # have much lower values than the unmasked original
        center_original = block[:, :, 14:16, 14:16].mean()
        center_filtered = result[:, :, 14:16, 14:16].mean()
        assert center_filtered < center_original * 0.5


# ---------------------------------------------------------------------------
# Unified pipeline consistency
# ---------------------------------------------------------------------------

class TestApplyAllFiltersSlim:
    def test_matches_manual_pieces(self, synthetic_block):
        """apply_all_filters_slim produces identical output to calling pieces."""
        spatial_specs = [
            {"type": "gaussian", "sigma": 1.0},
            {"type": "norm", "size": [5, 5], "max_gain": 1.0},
        ]

        # Unified path
        unified = apply_all_filters_slim(
            synthetic_block, spatial_specs, temporal_specs=[], pixel_mask=None,
        )

        # Manual path: copy + spatial
        manual = synthetic_block.copy()
        manual = _apply_spatial_filters_numpy(manual, spatial_specs)

        np.testing.assert_array_equal(unified, manual)

    def test_with_mask_matches_manual(self, synthetic_block):
        """Unified path with mask matches manual mask-then-filter."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[:5, :5] = True

        spatial_specs = [{"type": "gaussian", "sigma": 1.0}]

        unified = apply_all_filters_slim(
            synthetic_block, spatial_specs, temporal_specs=[], pixel_mask=mask,
        )

        manual = synthetic_block.copy()
        manual[:, :, mask] = 0
        manual = _apply_spatial_filters_numpy(manual, spatial_specs)

        np.testing.assert_array_equal(unified, manual)

    def test_no_filters_returns_unchanged(self, synthetic_block):
        """No filters and no mask returns input unchanged (zero-copy contract)."""
        result = apply_all_filters_slim(
            synthetic_block, [], temporal_specs=[], pixel_mask=None,
        )
        # Zero-copy: no filters means no allocation, returns same object
        assert result is synthetic_block
