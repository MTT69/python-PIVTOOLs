import unittest
import numpy as np
import dask.array as da
from pivtools_cli.preprocessing.filters import (
    filter_images,
    time_filter, pod_filter, clip_filter, invert_filter,
    levelize_filter, lmax_filter, maxnorm_filter,
)
from scipy.ndimage import maximum_filter

class FilterTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(loc=100.0, scale=20.0, size=(3, 2, 4, 4)).astype(np.float32)
        self.images = da.from_array(arr, chunks=(2, 2, 4, 4))
        white = rng.uniform(low=0.5, high=2.0, size=(4, 4)).astype(np.float32)
        self.white = da.from_array(white, chunks=(4, 4))

    def test_clip_with_explicit_threshold(self):
        out = clip_filter(self.images, threshold=(10, 20)).compute()
        self.assertTrue(out.min() >= 10)
        self.assertTrue(out.max() <= 20)

    def test_invert_with_offset(self):
        offset = 200.0
        out = invert_filter(self.images, offset=offset).compute()
        expected = offset - self.images.compute()
        np.testing.assert_allclose(out, expected)

    def test_levelize_with_white_image(self):
        out = levelize_filter(self.images, self.white).compute()
        expected = self.images.compute() / self.white.compute()
        np.testing.assert_allclose(out, expected)
    
    def test_lmax_filter(self):
        size = (3, 3)
        out = lmax_filter(self.images, size=size).compute()
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                local_max = maximum_filter(self.images[i, j].compute(), size=size)
                np.testing.assert_allclose(out[i, j], local_max)