"""Stereo reconstruction of ensemble (mean-flow) results.

The stereo calibration-apply step gains an ``ensemble_result.mat`` branch:
both cameras' ensemble mean-displacement fields are reconstructed to (U,V,W)
on the regular world grid, per saved pass. Mean flow only — the 4x3 Jacobian
solve is linear in the displacements, so the mean reconstructs exactly;
Reynolds stresses do NOT reconstruct linearly and must not appear in the
output. Pure synthetic pinhole pair, no C extension.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_gui.calibration.camera_model import CameraModel
from pivtools_gui.calibration.runio import (
    read_ensemble_vectors,
    reconstruct_stereo_run,
)


# ---------------------------------------------------------------------------
# Synthetic stereo pair (same factory as test_calibration_regular_grid.py)
# ---------------------------------------------------------------------------

K = np.array([[2000.0, 0.0, 800.0], [0.0, 2000.0, 600.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1600, 1200)


def _Ry(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _stereo_pair(angle_deg: float = 15.0):
    t = np.array([[0.0], [0.0], [600.0]])
    m1 = CameraModel(K, np.zeros(5), _Ry(+angle_deg), t.copy(), IMAGE_SIZE)
    m2 = CameraModel(K, np.zeros(5), _Ry(-angle_deg), t.copy(), IMAGE_SIZE)
    return m1, m2


def _piv_coords(nx=24, ny=18, x0=200.0, x1=1400.0, y0=150.0, y1=1050.0):
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    return np.stack([gx, gy], axis=-1)


def _uniform_disp(model: CameraModel, coords: np.ndarray, U_mm) -> tuple:
    flat = coords.reshape(-1, 2)
    world = model.back_project_to_plane(flat, 0.0, 0.0, 0.0)
    disp = model.project(world + np.asarray(U_mm, dtype=np.float64)) - flat
    H, W = coords.shape[:2]
    return disp[:, 0].reshape(H, W), disp[:, 1].reshape(H, W)


# ---------------------------------------------------------------------------
# Fabricate the on-disk uncalibrated ensemble layout
# ---------------------------------------------------------------------------

# Grids per populated pass (pass 0 left empty, mirroring runs=[2,3])
PASS_GRIDS = {1: dict(nx=24, ny=18), 2: dict(nx=48, ny=36)}
N_PASSES = 3
U_MM = np.array([0.4, -0.3, 0.25])   # uniform world displacement, mm/frame
DT = 0.002                            # s/frame


def _write_uncal_ensemble_dir(dirpath: Path, model: CameraModel,
                              passes=PASS_GRIDS, n_passes=N_PASSES,
                              with_stress_names=True, drop_pass=None):
    """Write coordinates.mat + ensemble_result.mat the way the pipeline does."""
    dirpath.mkdir(parents=True, exist_ok=True)
    empty = np.array([], dtype=np.float64)

    coords_st = np.empty((n_passes,), dtype=[("x", object), ("y", object)])
    fields = [("ux", object), ("uy", object), ("b_mask", object)]
    if with_stress_names:
        fields += [("UU_stress", object), ("VV_stress", object), ("UV_stress", object)]
    ens_st = np.empty((n_passes,), dtype=fields)
    for i in range(n_passes):
        coords_st["x"][i] = coords_st["y"][i] = empty
        for name, _ in fields:
            ens_st[name][i] = empty

    for p, g in passes.items():
        if drop_pass is not None and p == drop_pass:
            continue
        coords = _piv_coords(nx=g["nx"], ny=g["ny"])
        # coordinates.mat stores 1-based MATLAB coords
        coords_st["x"][p] = coords[..., 0] + 1.0
        coords_st["y"][p] = coords[..., 1] + 1.0
        ux, uy = _uniform_disp(model, coords, U_MM)
        ens_st["ux"][p] = ux
        ens_st["uy"][p] = uy
        ens_st["b_mask"][p] = np.zeros(ux.shape, dtype=bool)

    scipy.io.savemat(str(dirpath / "coordinates.mat"),
                     {"coordinates": coords_st}, oned_as="row", do_compression=True)
    scipy.io.savemat(str(dirpath / "ensemble_result.mat"),
                     {"ensemble_result": ens_st}, oned_as="row", do_compression=True)


@pytest.fixture()
def stereo_setup(tmp_path):
    m1, m2 = _stereo_pair()
    d1 = tmp_path / "Cam1" / "ensemble"
    d2 = tmp_path / "Cam2" / "ensemble"
    out = tmp_path / "stereo" / "ensemble"
    _write_uncal_ensemble_dir(d1, m1)
    _write_uncal_ensemble_dir(d2, m2)
    record = SimpleNamespace(model1=m1, model2=m2)
    return record, d1, d2, out


# ---------------------------------------------------------------------------
# read_ensemble_vectors
# ---------------------------------------------------------------------------

def test_read_ensemble_vectors_roundtrip(tmp_path, stereo_setup):
    record, d1, _, _ = stereo_setup
    vecs = read_ensemble_vectors(d1 / "ensemble_result.mat")
    assert sorted(vecs) == sorted(PASS_GRIDS)
    for p, g in PASS_GRIDS.items():
        ux, uy, b = vecs[p]
        assert ux.shape == (g["ny"], g["nx"])
        assert uy.shape == (g["ny"], g["nx"])


def test_read_ensemble_vectors_requires_b_mask(tmp_path):
    st = np.empty((1,), dtype=[("ux", object), ("uy", object)])
    st["ux"][0] = np.ones((3, 4))
    st["uy"][0] = np.ones((3, 4))
    path = tmp_path / "ensemble_result.mat"
    scipy.io.savemat(str(path), {"ensemble_result": st})
    with pytest.raises(ValueError, match="b_mask"):
        read_ensemble_vectors(path)


# ---------------------------------------------------------------------------
# reconstruct_stereo_run — ensemble branch
# ---------------------------------------------------------------------------

def test_ensemble_reconstructs_all_passes_ground_truth(stereo_setup):
    record, d1, d2, out = stereo_setup
    written = reconstruct_stereo_run(
        record, d1, d2, out, dt=DT, vector_glob="*.mat", interpolator="linear",
    )
    assert any(Path(w).name == "ensemble_result.mat" for w in written)

    er = scipy.io.loadmat(str(out / "ensemble_result.mat"), squeeze_me=True)["ensemble_result"]
    # Mean flow only: exactly these fields, no stress names.
    assert set(er.dtype.names) == {"ux", "uy", "uz", "b_mask"}

    coords = scipy.io.loadmat(str(out / "coordinates.mat"), squeeze_me=True)["coordinates"]

    expected = U_MM / 1000.0 / DT  # m/s
    for p in PASS_GRIDS:
        U = np.asarray(er["ux"][p].item() if np.asarray(er["ux"][p]).dtype == object else er["ux"][p])
        V = np.asarray(er["uy"][p].item() if np.asarray(er["uy"][p]).dtype == object else er["uy"][p])
        W = np.asarray(er["uz"][p].item() if np.asarray(er["uz"][p]).dtype == object else er["uz"][p])
        mask = np.asarray(er["b_mask"][p].item() if np.asarray(er["b_mask"][p]).dtype == object else er["b_mask"][p]).astype(bool)
        assert U.size, f"pass {p} empty"
        assert not mask.all()
        assert np.allclose(U[~mask], expected[0], rtol=1e-3)
        assert np.allclose(V[~mask], expected[1], rtol=1e-3)
        assert np.allclose(W[~mask], expected[2], rtol=1e-3)
        # Per-pass world grid present and matching the velocity field shape.
        gx = np.asarray(coords["x"][p].item() if np.asarray(coords["x"][p]).dtype == object else coords["x"][p])
        assert gx.shape == U.shape
    # Unpopulated pass stays empty.
    assert np.asarray(er["ux"][0].item() if np.asarray(er["ux"][0]).dtype == object else er["ux"][0]).size == 0

    # Pass 2 has double the pixel-grid resolution -> finer world grid than pass 1.
    g1 = np.asarray(coords["x"][1].item() if np.asarray(coords["x"][1]).dtype == object else coords["x"][1])
    g2 = np.asarray(coords["x"][2].item() if np.asarray(coords["x"][2]).dtype == object else coords["x"][2])
    sp1 = float(np.median(np.diff(g1[0, :])))
    sp2 = float(np.median(np.diff(g2[0, :])))
    assert sp2 < sp1


def test_ensemble_branch_fires_with_per_frame_glob(stereo_setup):
    """A per-frame vector_glob (B*.mat) must not silently skip the ensemble.

    The default glob derived from vector_format never matches
    ensemble_result.mat, so detection goes by the file on disk, not the glob.
    """
    record, d1, d2, out = stereo_setup
    written = reconstruct_stereo_run(
        record, d1, d2, out, dt=DT, vector_glob="B*.mat", interpolator="linear",
    )
    assert any(Path(w).name == "ensemble_result.mat" for w in written)


def test_pass_index_selects_single_pass(stereo_setup):
    record, d1, d2, out = stereo_setup
    reconstruct_stereo_run(
        record, d1, d2, out, dt=DT, pass_index=1,
        vector_glob="*.mat", interpolator="linear",
    )
    er = scipy.io.loadmat(str(out / "ensemble_result.mat"), squeeze_me=True)["ensemble_result"]
    assert np.asarray(er["ux"][1].item() if np.asarray(er["ux"][1]).dtype == object else er["ux"][1]).size
    assert np.asarray(er["ux"][2].item() if np.asarray(er["ux"][2]).dtype == object else er["ux"][2]).size == 0


def test_ensemble_in_one_camera_only_raises(stereo_setup):
    record, d1, d2, out = stereo_setup
    (d2 / "ensemble_result.mat").unlink()
    # Give cam2 a per-frame file instead so the dir is not empty.
    st = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    st["ux"][0] = st["uy"][0] = np.ones((3, 4))
    st["b_mask"][0] = np.zeros((3, 4), dtype=bool)
    scipy.io.savemat(str(d2 / "B00001.mat"), {"piv_result": st})
    with pytest.raises(RuntimeError, match="both cameras"):
        reconstruct_stereo_run(record, d1, d2, out, dt=DT, vector_glob="*.mat")


def test_mixed_ensemble_and_frames_raises(stereo_setup):
    record, d1, d2, out = stereo_setup
    st = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    st["ux"][0] = st["uy"][0] = np.ones((3, 4))
    st["b_mask"][0] = np.zeros((3, 4), dtype=bool)
    for d in (d1, d2):
        scipy.io.savemat(str(d / "B00001.mat"), {"piv_result": st})
    with pytest.raises(RuntimeError, match="per-frame"):
        reconstruct_stereo_run(record, d1, d2, out, dt=DT, vector_glob="*.mat")


def test_shape_mismatch_raises(stereo_setup, tmp_path):
    record, d1, d2, out = stereo_setup
    # Rewrite cam2 with a velocity grid that does not match its coordinates.
    m1, m2 = record.model1, record.model2
    _write_uncal_ensemble_dir(d2, m2, passes={1: dict(nx=10, ny=8)}, n_passes=N_PASSES)
    # Overwrite its coordinates with the standard (24x18) grid -> mismatch.
    coords = _piv_coords()
    coords_st = np.empty((N_PASSES,), dtype=[("x", object), ("y", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(N_PASSES):
        coords_st["x"][i] = coords_st["y"][i] = empty
    coords_st["x"][1] = coords[..., 0] + 1.0
    coords_st["y"][1] = coords[..., 1] + 1.0
    scipy.io.savemat(str(d2 / "coordinates.mat"), {"coordinates": coords_st},
                     oned_as="row", do_compression=True)
    with pytest.raises(ValueError, match="does not match coordinates"):
        reconstruct_stereo_run(record, d1, d2, out, dt=DT, vector_glob="*.mat")
