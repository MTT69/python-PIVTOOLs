"""The joint bundle's analytic Jacobian is finite and raises no RuntimeWarning.

``_JointBA.jac`` chains ``cv2.projectPoints`` pose columns through ``cv2.composeRT``'s
Jacobians with two matmuls. NumPy 2.2.6 on arm64 Apple Accelerate sets spurious
divide/invalid/overflow FP flags on those finite products, so ``jac`` silences the flags
under ``np.errstate`` and then checks finiteness itself, raising ``FloatingPointError``.
These tests pin both halves: no warning escapes a real solve, and a genuinely non-finite
product still fails loudly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from test_calibration_global_grid import (  # noqa: E402  (sibling-test reuse, as elsewhere)
    SPACING,
    _make_dataset,
    _spec,
)

from pivtools_gui.calibration import joint as J
from pivtools_gui.calibration.global_grid import resolve_global_grid

# The synthetic projects around principal point (640, 512) with a 1280x1024-ish sensor.
_IMAGE_SIZE = (1280, 1024)


def _solve(**kwargs) -> J.JointResult:
    """Run the pinhole joint solve on the noise-free three-camera synthetic dataset."""
    detections, _truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    global_index = resolve_global_grid(detections, spec, spacing_mm=SPACING)
    return J.run_joint(
        detections,
        global_index,
        SPACING,
        datum_camera=1,
        datum_view=0,
        image_size_by_cam={c: _IMAGE_SIZE for c in detections},
        expected_cameras=sorted(detections),
        **kwargs,
    )


def test_joint_jacobian_finite_and_warning_free(monkeypatch):
    """Every Jacobian the bundle requests is finite, and no RuntimeWarning escapes."""
    jacobians = []
    orig_jac = J._JointBA.jac

    def recording_jac(self, x):
        out = orig_jac(self, x)
        jacobians.append(out)
        return out

    monkeypatch.setattr(J._JointBA, "jac", recording_jac)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _solve()

    assert jacobians, "the bundle never evaluated the analytic Jacobian"
    for jm in jacobians:
        assert np.isfinite(jm.data).all()
    assert np.isfinite(res.rms_px) and res.rms_px < 1.0


@pytest.mark.parametrize(
    ("which", "match"),
    [
        ("rig", r"non-finite rig Jacobian for camera \d+"),
        ("board", r"non-finite board Jacobian for view \(\d+, \d+\)"),
    ],
)
def test_joint_jacobian_nonfinite_product_raises(monkeypatch, which, match):
    """A NaN in a composeRT Jacobian surfaces as FloatingPointError, not a silent NaN column."""
    orig = J._JointBA._pose_with_jac

    def poisoned(self, x, k):
        r3, t3, J_rig, J_board = orig(self, x, k)
        if which == "rig":
            J_rig = J_rig.copy()
            J_rig[0, 0] = np.nan
        else:
            J_board = J_board.copy()
            J_board[0, 0] = np.nan
        return r3, t3, J_rig, J_board

    monkeypatch.setattr(J._JointBA, "_pose_with_jac", poisoned)
    with pytest.raises(FloatingPointError, match=match):
        _solve()
