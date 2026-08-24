#!/usr/bin/env python3
"""
test_container_error_reporting.py

POST /backend/validate_files must report why a container would not decode.

Background: a user pointed the GUI at an intact two-camera .set whose pixel
encoding the reader did not support. The panel said, for both cameras:

    First frame not found. Container file: 10deg_….set. Found files: 10deg_….set

Not found, and found, in one sentence. The reader had raised a precise
``ValueError: Unsupported pixel decoder: 'raw-16-bit'`` and validate_images_generic
had wrapped it correctly, but app.py overwrote it. ``first_frame_error`` — the
variable whose whole purpose is to carry the real exception — was only ever
assigned in the standard-format branch, so the ``file_found_but_unreadable`` guard
was dead for every .set and .cine, and the message was synthesised from
``image_type`` alone. The "Found files" list came from a separate response field
that the frontend renders regardless of the message.

These tests drive the real route against a real Config and a real container, and
pin both halves of the fix.

Usage:
    pytest unit-tests/test_container_error_reporting.py -v
"""

import struct
import sys
from pathlib import Path

import pytest
import yaml
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.app as gui_app  # noqa: E402
from pivtools_core.config import Config  # noqa: E402

_TABLE_OFFSET = 256 + 768
_ENTRY_STRUCT = "<iqq"


def _write_container(tmp_path, decoder, width=4, height=2, n_entries=2):
    """Write a structurally valid .set whose pixel encoding is `decoder`.

    Two frame streams, so a single camera's pre-paired A/B read resolves.
    """
    set_file = tmp_path / "recording.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / "recording"
    set_dir.mkdir()

    payload = b"\x00" * (width * height * 2)
    for stream in range(2):
        index = bytearray(_TABLE_OFFSET)
        struct.pack_into("<i", index, 12, width)
        struct.pack_into("<i", index, 16, height)
        for entry in range(n_entries):
            index += struct.pack(_ENTRY_STRUCT, 0, entry * len(payload), len(payload))
        (set_dir / f"Frame{stream}-0.ims").write_bytes(bytes(index))
        (set_dir / f"Frame{stream}-1.ims").write_bytes(payload * n_entries)
        (set_dir / f"Frame{stream}-decoder.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f"<FrameDecoder>\n<id>{decoder}</id>\n</FrameDecoder>\n"
        )
    return set_file


def _make_config(tmp_path, set_file, num_images=2):
    """A real Config pointing one camera at the .set container."""
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "out")],
            "source_paths": [str(set_file)],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_type": "lavision_set",
            "image_format": [set_file.name],
            "num_images": num_images,
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "pre_paired",
            "num_loops": 1,
            "use_camera_subfolders": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return Config(path=str(config_path))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real api_bp route, with get_config bound to a tmp workspace."""

    def _build(decoder):
        set_file = _write_container(tmp_path, decoder)
        config = _make_config(tmp_path, set_file)
        monkeypatch.setattr(gui_app, "get_config", lambda: config)
        app = Flask(__name__)
        app.register_blueprint(gui_app.api_bp)
        return app.test_client()

    return _build


def _camera_result(client, decoder):
    response = client(decoder).post("/backend/validate_files", json={})
    assert response.status_code == 200
    return response.get_json()["details"]["camera_1"]


def test_unsupported_encoding_reports_the_real_reason(client):
    """The panel names the encoding, not a missing file.

    This is the exact regression: the container is present and its first frame
    read fails on the encoding.
    """
    result = _camera_result(client, "mono-12pmsb")

    assert result["status"] == "error"
    assert "mono-12pmsb" in result["error"]
    assert "First frame not found" not in result["error"]


def test_decode_failure_withholds_the_folder_listing(client):
    """No "Files found in folder" block for a file that was found.

    The frontend renders sample_files independently of the message
    (ValidationAlert.tsx), which is what produced "not found … Found files: <the
    file>" in one panel.
    """
    result = _camera_result(client, "mono-12pmsb")

    assert result["sample_files"] == []


def test_supported_encoding_still_validates(client):
    """Control: the same container in a supported encoding passes.

    Without this, a route that errored unconditionally would satisfy the two tests
    above.
    """
    result = _camera_result(client, "raw-16-bit")

    assert result["status"] == "ok", result["error"]
    assert result["first_frame"] == "exists"


def test_preview_render_failure_is_not_reported_as_a_missing_frame(client, monkeypatch):
    """A container that decodes but whose PNG preview fails is still readable.

    ``_image_to_base64`` runs inside a try/except that only warns, so a render
    failure leaves ``first_image_preview`` None while ``read_error`` stays None too.
    The container branch used to decide "does frame 1 exist" from the preview, which
    turned a cosmetic render failure into this module's original bug: "First frame
    not found" for a file that is present and decodes. ``image_size`` is taken
    straight from the decoded frame, so that is what the decision keys on.
    """
    import pivtools_core.image_handling.path_utils as path_utils

    def _render_fails(*_args, **_kwargs):
        raise RuntimeError("preview render failed")

    monkeypatch.setattr(path_utils, "_image_to_base64", _render_fails)

    result = _camera_result(client, "raw-16-bit")

    assert result["first_frame"] == "exists"
    assert result["status"] == "ok", result["error"]
    assert "First frame not found" not in (result["error"] or "")


def test_wrong_davis_node_reaches_the_user(client, tmp_path, monkeypatch):
    """A .set pointing at a DaVis calibration node names the .im7 reader.

    The container file exists, so the failure arrives as a FileNotFoundError from
    inside the reader. That must NOT be reclassified as "first frame not found" —
    the exclusion only applies to standard formats, where a missing numbered file
    is genuinely what happened.
    """
    set_file = tmp_path / "camera1.set"
    set_file.write_bytes(b"stub")
    companion = tmp_path / "camera1"
    companion.mkdir()
    (companion / "B00001.im7").write_bytes(b"stub")

    config = _make_config(tmp_path, set_file)
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)

    response = app.test_client().post("/backend/validate_files", json={})
    result = response.get_json()["details"]["camera_1"]

    assert result["status"] == "error"
    assert "lavision_im7" in result["error"]
    assert result["sample_files"] == []


def test_validation_dict_is_json_serializable(tmp_path):
    """validate_images_generic's result must survive jsonify on a read failure.

    The calibration validate route returns this dict straight to jsonify
    (pivtools_gui/calibration/app/views.py), outside its try/except. Holding the
    raw exception in ``read_error`` raised ``TypeError: Object of type ValueError
    is not JSON serializable`` there — turning the very failure the field exists
    to report into an HTTP 500 with no message. Caught in review, 2026-08-22.
    """
    import json

    from pivtools_core.image_handling.path_utils import validate_images_generic

    set_file = _write_container(tmp_path, "mono-12pmsb")

    def _read(_idx):
        from pivtools_core.image_handling.readers.set_reader import read_set_frame

        return read_set_frame(set_file, entry_no=1, frame_idx=0)

    result = validate_images_generic(
        camera_path=set_file,
        camera=1,
        image_format=set_file.name,
        image_type="lavision_set",
        expected_count=2,
        zero_based_indexing=False,
        read_frame_fn=_read,
    )

    assert result["valid"] is False
    assert "mono-12pmsb" in result["read_error"]
    # The assertion that matters: this must not raise.
    json.dumps(result)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
