"""Per-type model record filenames + the CLI stereo reuse guard (triage items 8, A7).

Records are saved as ``model_{model_type}.mat`` / ``stereo_model_{model_type}.mat``
so fitting a second model type never clobbers the first. Resolution rules under
test (``record.resolve_mono_path`` / ``resolve_stereo_path``):

- a requested type loads exactly that record (and a mismatching file raises);
- with no requested type, one record present -> it, several -> ValueError listing
  them (no silent pick).

A7 — the CLI stereo reuse path must not reuse a record fitted for a DIFFERENT
board: the stereo model dir is shared across board types (mono dirs embed the
board in their path), so ``_reusable_stereo`` checks the stored ``board_type``.
"""

import numpy as np
import pytest

from pivtools_cli.calibration_cli import _reusable_stereo
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.camera_model import (
    CameraModel,
    DistortionModel,
    PolynomialModel,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _pinhole() -> CameraModel:
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 512], [0, 0, 1]])
    return CameraModel(K=K, dist=np.zeros(5), R=np.eye(3), t=np.zeros((3, 1)),
                       image_size=(1024, 1024),
                       distortion_model=DistortionModel.STANDARD, rms=0.1)


def _polynomial() -> PolynomialModel:
    return PolynomialModel(coeffs_x=np.arange(10, dtype=float),
                           coeffs_y=np.arange(10, dtype=float) * 2,
                           x0=512.0, sx=512.0, y0=512.0, sy=512.0,
                           image_size=(1024, 1024), rms_x_mm=0.01, rms_y_mm=0.02)


def _mono_record(model, board="dotboard") -> rec.MonoRecord:
    return rec.MonoRecord(camera=1, board_type=board, camera_model=model,
                          per_view_rms=[0.1])


def _stereo_record(board="dotboard") -> rec.StereoRecord:
    return rec.StereoRecord(cam1=1, cam2=2, board_type=board,
                            model1=_pinhole(), model2=_pinhole(),
                            R_stereo=np.eye(3), T_stereo=np.array([[100.0], [0], [0]]),
                            per_view_rms1=[0.1], per_view_rms2=[0.2])


# ---------------------------------------------------------------------------
# Per-type filenames + coexistence
# ---------------------------------------------------------------------------


def test_save_uses_per_type_filenames(tmp_path):
    assert rec.save_mono(_mono_record(_pinhole()), tmp_path).name == "model_pinhole.mat"
    assert rec.save_mono(_mono_record(_polynomial()), tmp_path).name == "model_polynomial.mat"
    assert rec.save_stereo(_stereo_record(), tmp_path).name == "stereo_model_pinhole.mat"


def test_mono_types_coexist_and_load_by_type(tmp_path):
    rec.save_mono(_mono_record(_pinhole()), tmp_path)
    rec.save_mono(_mono_record(_polynomial()), tmp_path)
    assert isinstance(rec.load_mono(tmp_path, "pinhole").camera_model, CameraModel)
    assert isinstance(rec.load_mono(tmp_path, "polynomial").camera_model, PolynomialModel)


def test_load_mono_ambiguous_without_type(tmp_path):
    rec.save_mono(_mono_record(_pinhole()), tmp_path)
    rec.save_mono(_mono_record(_polynomial()), tmp_path)
    with pytest.raises(ValueError, match="pinhole.*polynomial|polynomial.*pinhole"):
        rec.load_mono(tmp_path)


def test_load_mono_single_type_needs_no_type(tmp_path):
    rec.save_mono(_mono_record(_polynomial()), tmp_path)
    assert isinstance(rec.load_mono(tmp_path).camera_model, PolynomialModel)


def test_resolve_missing_requested_type_names_what_exists(tmp_path):
    rec.save_mono(_mono_record(_polynomial()), tmp_path)
    with pytest.raises(FileNotFoundError, match="polynomial"):
        rec.resolve_mono_path(tmp_path, "pinhole")


def test_load_requested_type_mismatch_on_file_raises(tmp_path):
    path = rec.save_mono(_mono_record(_pinhole()), tmp_path)
    with pytest.raises(ValueError, match="pinhole"):
        rec.load_mono(path, "polynomial")


# ---------------------------------------------------------------------------
# Forward-looking: old single-name files are no longer resolved
# ---------------------------------------------------------------------------


def test_old_single_name_mono_file_not_resolved(tmp_path):
    # A pre-per-type ``model.mat`` is not a per-type file, so the dir reads empty.
    path = rec.save_mono(_mono_record(_pinhole()), tmp_path)
    path.rename(tmp_path / "model.mat")
    with pytest.raises(FileNotFoundError):
        rec.resolve_mono_path(tmp_path)


def test_old_single_name_stereo_file_not_resolved(tmp_path):
    path = rec.save_stereo(_stereo_record(), tmp_path)
    path.rename(tmp_path / "stereo_model.mat")
    with pytest.raises(FileNotFoundError):
        rec.resolve_stereo_path(tmp_path)


# ---------------------------------------------------------------------------
# A7 — stereo reuse guard (board_type must match)
# ---------------------------------------------------------------------------


def test_reuse_rejects_other_board_stereo_model(tmp_path, capsys):
    rec.save_stereo(_stereo_record(board="stepped"), tmp_path)
    assert _reusable_stereo(tmp_path, "dotboard", "pinhole", force=False) is None
    assert "recomputing" in capsys.readouterr().out


def test_reuse_accepts_matching_board_stereo_model(tmp_path):
    saved = rec.save_stereo(_stereo_record(board="dotboard"), tmp_path)
    reuse = _reusable_stereo(tmp_path, "dotboard", "pinhole", force=False)
    assert reuse is not None
    path, existing = reuse
    assert path == saved
    assert existing.board_type == "dotboard"


def test_reuse_force_recomputes(tmp_path):
    rec.save_stereo(_stereo_record(), tmp_path)
    assert _reusable_stereo(tmp_path, "dotboard", "pinhole", force=True) is None


def test_reuse_requested_type_absent_recomputes(tmp_path):
    # Only a pinhole record exists; a polynomial3d detect run must recompute.
    rec.save_stereo(_stereo_record(), tmp_path)
    assert _reusable_stereo(tmp_path, "dotboard", "polynomial3d", force=False) is None
