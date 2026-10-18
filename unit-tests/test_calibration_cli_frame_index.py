"""CLI calibration frames use the GUI's indexing: logical 1..N, mapped by zero_based_indexing.

There is no calibration ``start_index`` knob: the GUI never read or wrote one, so a CLI that
honoured it read different files from the GUI for the same sidecar.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import pivtools_cli.calibration_cli as cli
from pivtools_core import calibration_settings as cs

_FMT = "calib%05d.png"
_N_VIEWS = 3
_SHAPE = (8, 8)


def _write_frames(folder, first_file_number: int) -> None:
    """One flat image per file whose pixel value equals its file number."""
    for k in range(_N_VIEWS):
        n = first_file_number + k
        cv2.imwrite(str(folder / (_FMT % n)), np.full(_SHAPE, n, dtype=np.uint8))


@pytest.mark.parametrize("zero_based, first_file", [(False, 1), (True, 0)])
def test_load_views_reads_files_by_zero_based_flag(tmp_path, zero_based, first_file):
    _write_frames(tmp_path, first_file)
    images = cli._load_views(
        tmp_path,
        _FMT,
        _N_VIEWS,
        cli.FIRST_CALIBRATION_FRAME,
        camera=1,
        image_type="standard",
        use_camera_subfolders=False,
        zero_based=zero_based,
        num_cameras=1,
    )
    assert [int(im[0, 0]) for im in images] == list(
        range(first_file, first_file + _N_VIEWS)
    )


def test_settings_template_has_no_calibration_start_index():
    assert "start_index" not in cs.default_settings()["image"]
