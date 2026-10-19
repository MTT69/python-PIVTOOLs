"""Two Setup/calibration validation regressions found in the 2026-10-19 real-data pass.

1. A calibration source whose camera directory does not exist (wrong subfolder) must
   report "Camera path does not exist" with the subfolder suggestion, not the internal
   "expected_count is required for ... images".
2. ``read_pair`` returns a ``(2, H, W)`` stack, so a colour test on ``shape[-1] > 1``
   fired on every grayscale dataset ("Color images will be converted to grayscale").
"""

import numpy as np

from pivtools_core.calibration_settings import save_settings
from pivtools_core.image_handling import calibration_loader as CL
from pivtools_core.image_handling.load_images import is_colour_pair


class _FakeConfig:
    def __init__(self, source, camera_count=2):
        self._source = source
        self.camera_count = camera_count

    def get_calibration_source(self, idx):
        return self._source


def test_missing_camera_dir_names_the_path_not_expected_count(tmp_path):
    # Two cameras, standard format: the loader looks in <source>/Cam1, which is absent.
    (tmp_path / "camera1").mkdir()
    save_settings(
        tmp_path, {"image": {"image_format": "B%05d.tif", "image_type": "standard"}}
    )
    result = CL.validate_calibration_images(1, _FakeConfig(tmp_path), 0)
    assert result["valid"] is False
    assert "expected_count" not in (result["error"] or "")
    assert "Camera path does not exist" in result["error"]
    assert result["suggested_subfolder"] == "camera1"


def test_grayscale_pair_is_not_colour():
    assert not is_colour_pair(np.zeros((2, 8, 16), np.uint16))
    assert not is_colour_pair(np.zeros((2, 8, 1), np.uint16))


def test_colour_pair_is_colour():
    assert is_colour_pair(np.zeros((2, 8, 16, 3), np.uint8))
    assert is_colour_pair(np.zeros((2, 8, 16, 4), np.uint8))
