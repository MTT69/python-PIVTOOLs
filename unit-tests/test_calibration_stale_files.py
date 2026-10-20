"""Forward-only readers: a sidecar or record missing an always-written field must raise.

Every case saves a REAL sidecar / record through the production writer, deletes one field
with ``scipy.io`` (``fixtures.stale``), and asserts the loader raises ``ValueError`` naming
the file and the field. A missing field means the file was written by an older build and
must be regenerated, never read with a faked default.

The control tests rewrite each file unchanged and assert it still loads, so a failure in
the parametrised cases is the missing field and not the rewrite.
"""

from __future__ import annotations

import pytest
from fixtures.stale import (
    RECORD_KINDS,
    WORLD_FRAME_FIELDS,
    real_record,
    real_sidecar,
    rewrite_mat,
    stale_record,
    stale_sidecar,
)

from pivtools_gui.calibration import inputs_store as INP

# Always-written sidecar fields (Design B). ``detections`` is conditional: it is only
# required while ``det_count > 0``, which the fixture sidecar satisfies.
SIDECAR_FIELDS = (
    "coords_json",
    "board_params_json",
    "det_count",
    "detections",
    "image_size_by_cam",
    "path_type",
    "board_type",
    "det_key",
    "detection_stamp",
)

_WORLD_FRAME = [("world_frame", f) for f in WORLD_FRAME_FIELDS]

# (record kind, field path) for every always-written record field.
RECORD_FIELDS = (
    [("mono", fp) for fp in _WORLD_FRAME]
    + [("mono", ("board_meta",))]
    + [("stereo", fp) for fp in _WORLD_FRAME]
    + [
        ("stereo", ("board_meta",)),
        ("stereo", ("self_cal",)),
        ("stereo", ("R_stereo",)),
        ("stereo", ("T_stereo",)),
        ("stereo_poly", ("R_stereo",)),
        ("stereo_poly", ("T_stereo",)),
        ("stereo_poly", ("board_meta",)),
        ("stereo_poly", ("self_cal",)),
    ]
    + [("joint", fp) for fp in _WORLD_FRAME]
    + [
        ("joint", ("board_meta",)),
        ("joint", ("pose_diversity",)),
        ("joint", ("camera_models", "cam_rms")),
    ]
)


def _id(field) -> str:
    return field if isinstance(field, str) else ".".join(field)


def _assert_names(exc: pytest.ExceptionInfo, path, field) -> None:
    msg = str(exc.value)
    leaf = field if isinstance(field, str) else field[-1]
    assert path.name in msg, f"error does not name the file {path.name!r}: {msg}"
    assert leaf in msg, f"error does not name the field {leaf!r}: {msg}"


# ---------------------------------------------------------------------------
# Controls: the rewrite alone must not break a load
# ---------------------------------------------------------------------------


def test_unchanged_sidecar_rewrite_loads(tmp_path):
    path = rewrite_mat(real_sidecar(tmp_path))
    rec = INP.load_inputs(tmp_path)
    assert path.is_file()
    assert sorted(rec.detections) == [1, 2]
    assert rec.det_key


@pytest.mark.parametrize("kind", sorted(RECORD_KINDS))
def test_unchanged_record_rewrite_loads(tmp_path, kind):
    path = rewrite_mat(real_record(kind, tmp_path))
    assert RECORD_KINDS[kind][1](path) is not None


# ---------------------------------------------------------------------------
# Each always-written field, removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", SIDECAR_FIELDS)
def test_sidecar_missing_field_raises(tmp_path, field):
    path = stale_sidecar(tmp_path, field)
    with pytest.raises(ValueError) as exc:
        INP.load_inputs(tmp_path)
    _assert_names(exc, path, field)


@pytest.mark.parametrize(
    "kind,field", RECORD_FIELDS, ids=[f"{k}-{_id(f)}" for k, f in RECORD_FIELDS]
)
def test_record_missing_field_raises(tmp_path, kind, field):
    path = stale_record(kind, tmp_path, field)
    with pytest.raises(ValueError) as exc:
        RECORD_KINDS[kind][1](path)
    _assert_names(exc, path, field)
