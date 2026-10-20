"""Forward-only calibration readers: an always-written field that is missing raises.

Every record (mono / stereo pinhole / stereo polynomial3d / joint) and the ``inputs.mat``
sidecar is written by the production writer, one field is deleted with ``scipy.io``
(``fixtures.stale``), and the loader must raise naming the file and the field. A missing
field means the file was written by an older build: the reader never fakes a default.

Also pins the read-error / parse-error split of ``inputs_store.try_load_inputs``: an
unreadable file is an ``OSError`` naming the read, only a parse failure is
``InputsCorrupt``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.io
from fixtures.stale import (
    WORLD_FRAME_FIELDS,
    real_record,
    real_sidecar,
    rewrite_mat,
    stale_record,
    stale_sidecar,
)

from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as REC

_LOADERS = {
    "mono": REC.load_mono,
    "stereo": REC.load_stereo,
    "stereo_poly": REC.load_stereo,
    "joint": REC.load_joint,
}

# (record kind, field path) for every always-written record field that used to be read
# with a default. World-frame fields are expanded per kind below.
_RECORD_FIELDS = [
    ("mono", "board_meta"),
    ("stereo", "board_meta"),
    ("stereo", "self_cal"),
    ("stereo", "R_stereo"),
    ("stereo", "T_stereo"),
    ("stereo_poly", "board_meta"),
    ("stereo_poly", "self_cal"),
    ("stereo_poly", "R_stereo"),
    ("stereo_poly", "T_stereo"),
    ("joint", "board_meta"),
    ("joint", "pose_diversity"),
    ("joint", ("camera_models", "cam_rms")),
] + [
    (kind, ("world_frame", name))
    for kind in ("mono", "stereo", "stereo_poly", "joint")
    for name in WORLD_FRAME_FIELDS
]


def _field_name(field) -> str:
    return field if isinstance(field, str) else field[-1]


@pytest.mark.parametrize(
    "kind, field",
    _RECORD_FIELDS,
    ids=[f"{k}-{_field_name(f)}" for k, f in _RECORD_FIELDS],
)
def test_record_missing_field_raises_naming_it(tmp_path, kind, field):
    path = stale_record(kind, tmp_path, field)
    with pytest.raises(ValueError) as info:
        _LOADERS[kind](path)
    msg = str(info.value)
    assert str(path) in msg
    assert repr(_field_name(field))[1:-1] in msg
    assert "regenerate" in msg


@pytest.mark.parametrize("kind", sorted(_LOADERS))
def test_record_rewrite_control_still_loads(tmp_path, kind):
    """The scipy rewrite itself (no field dropped) must not break the loader."""
    path = rewrite_mat(real_record(kind, tmp_path))
    _LOADERS[kind](path)


def test_joint_mono_view_reads_per_camera_rms_directly(tmp_path):
    """The per-camera MonoRecord built from a joint record carries that camera's rms."""
    root = tmp_path / "calibration"
    real_record("joint", root / "joint_dotboard" / "model")
    mono_dir = root / "Cam1" / "dotboard_planar" / "model"
    mono_dir.mkdir(parents=True)
    rec = REC.mono_record_for_camera(mono_dir, 1)
    assert rec.per_view_rms == [pytest.approx(0.18)]


def test_pinhole_stereo_with_empty_pose_raises(tmp_path):
    """A pinhole pair must carry R_stereo (9) / T_stereo (3); an empty pose is stale."""
    record = REC.load_stereo(real_record("stereo", tmp_path))
    record.R_stereo = None  # the writer stores None as an empty array
    path = REC.save_stereo(record, tmp_path)
    with pytest.raises(ValueError, match=r"R_stereo.*size 0"):
        REC.load_stereo(path)


def test_polynomial_stereo_with_a_pose_raises(tmp_path):
    """A polynomial3d pair has no extrinsic pose: a stored R_stereo is not ours."""
    record = REC.load_stereo(real_record("stereo_poly", tmp_path))
    record.R_stereo = np.eye(3)
    record.T_stereo = np.zeros((3, 1))
    path = REC.save_stereo(record, tmp_path)
    with pytest.raises(ValueError, match=r"R_stereo.*polynomial3d"):
        REC.load_stereo(path)


# ---------------------------------------------------------------------------
# inputs.mat sidecar
# ---------------------------------------------------------------------------

_SIDECAR_FIELDS = [
    "coords_json",
    "board_params_json",
    "det_count",
    "detections",
    "image_size_by_cam",
    "path_type",
    "board_type",
    "det_key",
    "detection_stamp",
]


@pytest.mark.parametrize("field", _SIDECAR_FIELDS)
def test_sidecar_missing_field_raises_naming_it(tmp_path, field):
    path = stale_sidecar(tmp_path, field)
    with pytest.raises(ValueError) as info:
        INP.load_inputs(tmp_path)
    assert str(path) in str(info.value)
    assert field in str(info.value)
    # try_load_inputs reports the same as a damaged sidecar, field name included.
    with pytest.raises(INP.InputsCorrupt, match=field):
        INP.try_load_inputs(tmp_path)


def test_sidecar_without_detections_needs_no_detections_field(tmp_path):
    """det_count == 0 legitimately writes no ``detections`` variable."""
    INP.save_inputs(tmp_path, path_type="mono", board_type="dotboard", coords=None)
    loaded = INP.load_inputs(tmp_path)
    assert loaded.detections is None
    assert loaded.coords is None
    assert loaded.det_key == "" and loaded.detection_stamp == ""
    assert loaded.image_size_by_cam == {}


def test_sidecar_non_string_det_key_raises(tmp_path):
    path = real_sidecar(tmp_path)
    mat = scipy.io.loadmat(str(path))
    mat["det_key"] = np.array([1, 2, 3], dtype=np.int64)
    scipy.io.savemat(str(path), {k: v for k, v in mat.items() if k[:2] != "__"})
    with pytest.raises(ValueError, match="det_key"):
        INP.load_inputs(tmp_path)


def test_sidecar_blank_coords_json_raises(tmp_path):
    """The writer encodes "no coords" as JSON ``null``; a blank field is damage."""
    path = real_sidecar(tmp_path)
    mat = scipy.io.loadmat(str(path))
    mat["coords_json"] = ""
    scipy.io.savemat(str(path), {k: v for k, v in mat.items() if k[:2] != "__"})
    with pytest.raises(ValueError, match="coords_json"):
        INP.load_inputs(tmp_path)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="chmod 000 does not block reads on Windows or for root",
)
def test_unreadable_sidecar_is_a_read_error_not_corrupt(tmp_path):
    path = real_sidecar(tmp_path)
    os.chmod(path, 0)
    try:
        with pytest.raises(OSError, match="could not read inputs.mat") as info:
            INP.try_load_inputs(tmp_path)
        assert not isinstance(info.value, INP.InputsCorrupt)
    finally:
        os.chmod(path, 0o644)


def test_truncated_sidecar_is_corrupt(tmp_path):
    """A parse failure (scipy raises OSError on a short read) is still InputsCorrupt."""
    path = real_sidecar(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(INP.InputsCorrupt):
        INP.try_load_inputs(tmp_path)
