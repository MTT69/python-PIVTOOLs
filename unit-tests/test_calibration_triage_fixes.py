"""Targeted tests for the 2026-06 triage fixes (dossier items A5, B3).

A5 — ``record`` must raise on an unrecognized ``model_type`` tag instead of
silently loading the record as pinhole (no-silent-fallback rule). Files merely
MISSING the tag (pre-tag era) now also raise — the legacy pinhole default was
removed, so a tagless file is not a valid per-type record.

B3 — the BFS grid walk's reciprocity check must be a real check: stepping back
from a candidate by the ideal lattice vector must land nearest the current
blob. The old form stepped back by the candidate's own actual vector, which
reconstructs the query point exactly (a tautology), letting off-lattice decoys
into the grid whenever the true dot was missing.
"""

import numpy as np
import pytest
from scipy.io import savemat
from scipy.spatial import cKDTree

from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.camera_model import CameraModel, DistortionModel
from pivtools_gui.calibration.detection.grid_detection import _bfs_grid_walk_dict

# ---------------------------------------------------------------------------
# A5 — unknown model_type tag
# ---------------------------------------------------------------------------


def _pinhole() -> CameraModel:
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 512], [0, 0, 1]])
    return CameraModel(
        K=K,
        dist=np.zeros(4),
        R=np.eye(3),
        t=np.zeros((3, 1)),
        image_size=(1024, 1024),
        distortion_model=DistortionModel.STANDARD,
        rms=0.1,
    )


def test_load_mono_raises_on_unknown_model_type(tmp_path):
    # The tag is dispatched before any other field is read, so a tag-only
    # file is enough to prove the unknown-tag path raises.
    path = tmp_path / "model.mat"
    savemat(str(path), {"model_type": "bogus_future_model"})
    with pytest.raises(ValueError, match="bogus_future_model"):
        rec.load_mono(path)


def test_load_stereo_raises_on_unknown_model_type(tmp_path):
    path = tmp_path / "stereo_model.mat"
    savemat(str(path), {"model_type": "bogus"})
    with pytest.raises(ValueError, match="bogus"):
        rec.load_stereo(path)


def test_load_mono_missing_tag_raises(tmp_path):
    # Forward-looking: a pre-tag-era file written WITHOUT model_type is no longer
    # silently assumed pinhole — it fails loudly as not-a-per-type-record.
    path = tmp_path / "model.mat"
    data = {
        "contract_version": 1,
        "board_type": "dotboard",
        "camera": 1,
        "world_frame": rec._world_frame_to_dict(rec.WorldFrame()),
        "per_view_rms": np.zeros((1, 1)),
        "board_meta": {"_empty": 0},
        "camera_model": rec._camera_to_dict(_pinhole()),
    }
    savemat(str(path), data, oned_as="row")
    with pytest.raises(ValueError, match="no model_type tag"):
        rec.load_mono(path)


# ---------------------------------------------------------------------------
# B3 — reciprocity check rejects an off-lattice decoy
# ---------------------------------------------------------------------------

SPACING = 100.0


def _lattice_with_decoy():
    """5x5 lattice with the dot right of centre missing and a decoy nearby.

    The decoy at centre + (130, 70) passes the distance band (40..160) and the
    30-degree cone toward +v1, and trivially passed the old tautological
    reciprocity check. Stepping BACK from it by the ideal v1 lands at
    centre + (30, 70), whose nearest blob is the dot ABOVE the centre — so the
    real reciprocity check rejects it.
    """
    pts = [
        np.array([i * SPACING, j * SPACING], float)
        for i in range(5)
        for j in range(5)
        if not (i == 3 and j == 2)
    ]  # hole right of the centre (2,2)
    decoy = np.array([2 * SPACING + 130.0, 2 * SPACING + 70.0])
    centers = np.vstack(pts + [decoy])
    return centers, len(centers) - 1


def test_bfs_rejects_off_lattice_decoy():
    centers, decoy_idx = _lattice_with_decoy()
    grid = _bfs_grid_walk_dict(
        centers, np.array([SPACING, 0.0]), np.array([0.0, SPACING]), cKDTree(centers)
    )
    assert decoy_idx not in grid.values(), "decoy blob entered the grid"
    # Every true lattice dot is still reachable (the walk routes around the hole).
    assert len(grid) == len(centers) - 1
    # Grid coords must be lattice-consistent: index deltas match position deltas.
    inv = {idx: coord for coord, idx in grid.items()}
    items = sorted(inv.items())
    (i0, (c0, r0)) = items[0]
    p0 = centers[i0]
    for idx, (c, r) in items[1:]:
        d = (centers[idx] - p0) / SPACING
        assert (c - c0, r - r0) == (round(d[0]), round(d[1]))
