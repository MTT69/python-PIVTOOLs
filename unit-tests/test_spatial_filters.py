"""
Regression tests for spatial filters in _apply_spatial_filters_numpy.

Tests each filter type (gaussian, median, norm, maxnorm, norm2, ssmin, lmax)
against inline scipy reference implementations, plus pixel mask + spatial
ordering and the unified apply_all_filters_slim pipeline. The production
backend is cv2 with scipy border semantics; min/max/median must stay
bit-exact vs scipy, box/gaussian are allowed float32 rounding differences.
"""

import numpy as np
import pytest
from scipy.ndimage import maximum_filter as scipy_maximum
from scipy.ndimage import median_filter as scipy_median
from scipy.ndimage import minimum_filter as scipy_minimum
from scipy.ndimage import uniform_filter as scipy_uniform

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
    def test_matches_fir_kernel(self, synthetic_block):
        """Gaussian filter uses explicit FIR kernel matching MATLAB fspecial."""
        from scipy.ndimage import correlate

        spec = [{"type": "gaussian", "sigma": 1.5, "size": [7, 7]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        # Build the same FIR kernel as the implementation
        from pivtools_cli.processing.dask_pipeline import _gaussian_kernel_1d

        ky = _gaussian_kernel_1d(7, 1.5)
        kx = _gaussian_kernel_1d(7, 1.5)
        kernel_2d = np.outer(ky, kx).astype(np.float32)
        expected = synthetic_block.astype(np.float32).copy()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                expected[i, j] = correlate(expected[i, j], kernel_2d, mode="constant")
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
        assert result.max() <= 1.01  # should be ~1


class TestMaxnormFilter:
    def test_matches_reference(self, synthetic_block):
        size = (7, 7)
        max_gain = 2.0
        spec = [{"type": "maxnorm", "size": list(size), "max_gain": max_gain}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        # Inline reference — matches MATLAB filter_maxnorm:
        # MATLAB minmaxfiltnd returns [minu, maxu]; single-output = minu.
        # mode='nearest' matches van Herk index clipping (minimum_filter).
        # mode='constant' matches convn(..., 'same') zero-pad (uniform_filter).
        spatial_size = (1, 1) + size
        block_float = synthetic_block.astype(np.float32)
        local_min = scipy_minimum(block_float, size=spatial_size, mode="nearest")
        smoothed_min = scipy_uniform(local_min, size=spatial_size, mode="constant")
        denom = np.maximum(smoothed_min, 1.0 / max_gain)
        expected = (np.maximum(block_float, 0) / denom).astype(synthetic_block.dtype)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestLmaxFilter:
    def test_matches_scipy(self, synthetic_block):
        spec = [{"type": "lmax", "size": [7, 7]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_maximum(synthetic_block, size=(1, 1, 7, 7))
        np.testing.assert_array_equal(result, expected)


class TestSsminFilter:
    def test_matches_reference(self, synthetic_block):
        """ssmin: 3x3 median -> sliding min -> box smooth -> subtract, clip.

        The median and min stages are bit-exact vs scipy (cv2 pad-trick /
        erode); the box smooth differs at float32 rounding level, hence
        allclose rather than array_equal for the full chain.
        """
        size = (7, 7)
        spec = [{"type": "ssmin", "size": list(size)}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        # Inline reference — matches MATLAB filter_ssmin
        spatial_size = (1, 1) + size
        block_float = synthetic_block.astype(np.float32)
        bg = scipy_median(block_float, size=(1, 1, 3, 3), mode="constant")
        bg = scipy_minimum(bg, size=spatial_size, mode="nearest")
        bg = scipy_uniform(bg, size=spatial_size, mode="constant")
        expected = np.maximum(block_float - bg, 0)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4)

    def test_median_min_stages_bit_exact(self, synthetic_block):
        """The rank stages of ssmin (median 3x3 constant, min nearest) are
        bit-exact vs scipy — the pad-trick border handling included."""
        from pivtools_cli.processing.dask_pipeline import _median2d, _min2d

        frame = synthetic_block[0, 0].copy()
        np.testing.assert_array_equal(
            _median2d(frame, (3, 3), mode="constant"),
            scipy_median(frame, size=(3, 3), mode="constant"),
        )
        np.testing.assert_array_equal(
            _min2d(frame, (7, 7), mode="nearest"),
            scipy_minimum(frame, size=(7, 7), mode="nearest"),
        )


class TestNorm2Filter:
    def test_matches_reference(self, synthetic_block):
        size = (7, 7)
        max_gain = 2.0
        spec = [{"type": "norm2", "size": list(size), "max_gain": max_gain}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        # Inline reference — matches MATLAB filter_norm2
        spatial_size = (1, 1) + size
        block_float = synthetic_block.astype(np.float32)
        local_min = scipy_minimum(block_float, size=spatial_size, mode="nearest")
        local_max = scipy_maximum(block_float, size=spatial_size, mode="nearest")
        local_min = scipy_uniform(local_min, size=spatial_size, mode="constant")
        local_max = scipy_uniform(local_max, size=spatial_size, mode="constant")
        denom = np.maximum(local_max - local_min, 1.0 / max_gain)
        expected = ((block_float - local_min) / denom).astype(synthetic_block.dtype)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class TestAnisotropicKernels:
    """Pin the cv2 kernel-size argument order.

    cv2 takes (width, height) where scipy takes (sy, sx); a swapped
    transposition passes every square-kernel test in this file. These use
    a (5, 9) kernel so any swap fails loudly.
    """

    SIZE = (5, 9)

    def test_min_max_bit_exact(self, synthetic_block):
        from pivtools_cli.processing.dask_pipeline import _max2d, _min2d

        frame = synthetic_block[0, 0].copy()
        np.testing.assert_array_equal(
            _min2d(frame, self.SIZE, mode="nearest"),
            scipy_minimum(frame, size=self.SIZE, mode="nearest"),
        )
        np.testing.assert_array_equal(
            _max2d(frame, self.SIZE, mode="reflect"),
            scipy_maximum(frame, size=self.SIZE, mode="reflect"),
        )

    def test_box_matches_scipy(self, synthetic_block):
        from pivtools_cli.processing.dask_pipeline import _box2d

        frame = synthetic_block[0, 0].copy()
        dst = np.empty_like(frame)
        _box2d(frame, self.SIZE, dst=dst)
        expected = scipy_uniform(frame, size=self.SIZE, mode="constant")
        np.testing.assert_allclose(dst, expected, rtol=1e-5, atol=1e-5)

    def test_gaussian_matches_2d_reference(self, synthetic_block):
        from scipy.ndimage import correlate

        from pivtools_cli.processing.dask_pipeline import _gaussian_kernel_1d

        spec = [{"type": "gaussian", "sigma": 1.5, "size": list(self.SIZE)}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)

        ky = _gaussian_kernel_1d(self.SIZE[0], 1.5)
        kx = _gaussian_kernel_1d(self.SIZE[1], 1.5)
        kernel_2d = np.outer(ky, kx).astype(np.float32)
        expected = synthetic_block.astype(np.float32).copy()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                expected[i, j] = correlate(expected[i, j], kernel_2d, mode="constant")
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestMedianScipyFallback:
    def test_large_square_kernel_matches_scipy(self, synthetic_block):
        """Sizes cv2.medianBlur cannot do on float32 (>5) use the scipy
        fallback and stay exactly equal to scipy."""
        spec = [{"type": "median", "size": [7, 7]}]
        result = _apply_spatial_filters_numpy(synthetic_block.copy(), spec)
        expected = scipy_median(synthetic_block, size=(1, 1, 7, 7))
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

        filter_specs = [{"type": "gaussian", "sigma": 2.0}]
        result = apply_all_filters_slim(
            block,
            filter_specs=filter_specs,
            pixel_mask=mask,
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
        filter_specs = [
            {"type": "gaussian", "sigma": 1.0},
            {"type": "norm", "size": [5, 5], "max_gain": 1.0},
        ]

        # Unified path
        unified = apply_all_filters_slim(
            synthetic_block,
            filter_specs=filter_specs,
            pixel_mask=None,
        )

        # Manual path: copy + spatial
        manual = synthetic_block.copy()
        manual = _apply_spatial_filters_numpy(manual, filter_specs)

        np.testing.assert_array_equal(unified, manual)

    def test_with_mask_matches_manual(self, synthetic_block):
        """Unified path with mask matches manual mask-then-filter."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[:5, :5] = True

        filter_specs = [{"type": "gaussian", "sigma": 1.0}]

        unified = apply_all_filters_slim(
            synthetic_block,
            filter_specs=filter_specs,
            pixel_mask=mask,
        )

        manual = synthetic_block.copy()
        manual[:, :, mask] = 0
        manual = _apply_spatial_filters_numpy(manual, filter_specs)

        np.testing.assert_array_equal(unified, manual)

    def test_no_filters_returns_unchanged(self, synthetic_block):
        """No filters and no mask returns input unchanged (zero-copy contract)."""
        result = apply_all_filters_slim(
            synthetic_block,
            filter_specs=[],
            pixel_mask=None,
        )
        # Zero-copy: no filters means no allocation, returns same object
        assert result is synthetic_block

    def test_user_defined_filter_order_preserved(self, synthetic_block):
        """Filters are applied in the exact order the user specified.

        Regression test: the old code split filters into spatial-then-temporal
        groups, destroying the user's interleaved ordering. The fix applies
        them in declared order.

        We verify this by checking that [gaussian, median] and
        [median, gaussian] produce DIFFERENT results (proving order matters
        and is respected).
        """
        specs_a = [
            {"type": "gaussian", "sigma": 2.0},
            {"type": "median", "size": [5, 5]},
        ]
        specs_b = [
            {"type": "median", "size": [5, 5]},
            {"type": "gaussian", "sigma": 2.0},
        ]

        result_a = apply_all_filters_slim(
            synthetic_block,
            filter_specs=specs_a,
            pixel_mask=None,
        )
        result_b = apply_all_filters_slim(
            synthetic_block,
            filter_specs=specs_b,
            pixel_mask=None,
        )

        # gaussian→median should differ from median→gaussian
        assert not np.array_equal(
            result_a, result_b
        ), "Different filter orderings should produce different results"
