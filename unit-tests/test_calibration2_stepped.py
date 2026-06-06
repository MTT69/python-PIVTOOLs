"""S1 — stepped (dual-level) dotboard detector recovery tests.

Renders a synthetic stepped board: a peak grid at z=0 and an interleaved trough grid
offset by half a dot spacing in x,y and by -step_height in z, projected through a
real pinhole pose (genuine perspective + z-parallax between the levels). Asserts the
detector separates the two levels, stitches them into one consistent pose-local grid,
and emits board-local geometry with the correct two-Z + half-spacing interleave.

Run with --make-figures to drop a diagnostic PNG in PyPIVTools/figures/debug/.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from pivtools_cli.synthetic_calibration_common import make_camera_matrix
from pivtools_gui.calibration2.detection.stepped import SteppedDetector, SteppedParams

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "debug"

# Board geometry (the DaVis 309-15-3 plate, scaled-down grid for a fast test).
SPACING_MM = 15.0
STEP_MM = 3.0
OFFSET_MM = SPACING_MM / 2.0
PEAK_COLS, PEAK_ROWS = 9, 9          # 81 peak dots
TROUGH_COLS, TROUGH_ROWS = 8, 8      # 64 trough dots


def _board_points_m():
    """Peak + trough 3D points in metres, plus a level label per point."""
    sp, step, off = SPACING_MM / 1000.0, STEP_MM / 1000.0, OFFSET_MM / 1000.0
    peak, trough = [], []
    for r in range(PEAK_ROWS):
        for c in range(PEAK_COLS):
            peak.append([c * sp, r * sp, 0.0])
    for r in range(TROUGH_ROWS):
        for c in range(TROUGH_COLS):
            trough.append([c * sp + off, r * sp + off, -step])
    return np.array(peak, np.float64), np.array(trough, np.float64)


def _render_stepped(W=1200, H=1200, rvec=(0.16, 0.10, 0.02)):
    """Project both levels through one pinhole pose and draw filled dots."""
    cam = make_camera_matrix(W, H)
    fx = float(cam[0, 0])
    peak, trough = _board_points_m()
    allpts = np.vstack([peak, trough])
    centre = allpts.mean(axis=0)

    rvec = np.asarray(rvec, np.float64)
    R, _ = cv2.Rodrigues(rvec)
    board_w = (PEAK_COLS - 1) * SPACING_MM / 1000.0
    Z = fx * board_w / (0.70 * W)                 # ~70% frame fill
    tvec = np.array([0.0, 0.0, Z]) - R @ centre   # centre the board on the optical axis

    dist = np.zeros(5)
    proj, _ = cv2.projectPoints(allpts, rvec, tvec, cam, dist)
    px = proj.reshape(-1, 2)

    tree = cKDTree(px)
    nn = tree.query(px, k=2)[0][:, 1]
    radii = np.maximum(np.round(0.22 * nn).astype(int), 2)

    img = np.zeros((H, W), np.uint8)
    for (cx, cy), rad in zip(px, radii):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(img, (ix, iy), int(rad), 255, -1)

    n_peak = len(peak)
    return img, px, n_peak, (cam, rvec, tvec)


def _figpath(slug: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date(2026, 6, 6).isoformat()
    return FIG_DIR / f"{today}-calib2-{slug}.png"


@pytest.fixture(scope="module")
def stepped_render():
    return _render_stepped()


def test_stepped_detector_separates_and_stitches(stepped_render, request):
    img, px, n_peak, _pose = stepped_render
    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    res = det.detect(img)

    assert res.success, res.diagnostics.get("error")
    assert res.board_type == "stepped"

    # Both levels recovered. Edge dots are pruned by RANSAC + connected-component
    # filtering in grid assembly (normal, and harmless for calibration), so the
    # yield bar is "most dots", not "every dot" — the geometry checks below are the
    # real correctness test.
    na = res.diagnostics["level_a"]["n_points"]
    nb = res.diagnostics["level_b"]["n_points"]
    assert na >= 0.75 * (PEAK_COLS * PEAK_ROWS), f"level A short: {na}"
    assert nb >= 0.75 * (TROUGH_COLS * TROUGH_ROWS), f"level B short: {nb}"
    assert res.n >= 0.78 * (PEAK_COLS * PEAK_ROWS + TROUGH_COLS * TROUGH_ROWS), (
        f"too few points: {res.n} of {PEAK_COLS*PEAK_ROWS + TROUGH_COLS*TROUGH_ROWS}"
    )

    # Stitch produced a genuine two-level frame (not a degraded single level).
    meta = res.diagnostics["stitch"]
    assert not meta["degraded_single_level"]
    assert meta["consensus_pct"] > 80.0, f"low stitch consensus: {meta['consensus_pct']}"

    # board_local carries exactly the two physical Z planes (neutral 0 / -step).
    zs = np.unique(np.round(res.board_local_points[:, 2], 6))
    assert set(zs.tolist()) == {0.0, -STEP_MM}, f"unexpected z levels: {zs}"

    # The two levels are half-spacing interleaved in x,y: reference dots land on the
    # integer grid (x % spacing == 0), the other level on the +offset half-grid.
    ref = res.board_local_points[res.board_local_points[:, 2] == 0.0]
    oth = res.board_local_points[res.board_local_points[:, 2] == -STEP_MM]
    assert np.allclose(np.mod(ref[:, 0], SPACING_MM), 0.0, atol=1e-6)
    assert np.allclose(np.mod(oth[:, 0] - OFFSET_MM, SPACING_MM), 0.0, atol=1e-6)
    assert np.allclose(np.mod(oth[:, 1] - OFFSET_MM, SPACING_MM), 0.0, atol=1e-6)

    # image_points / grid_indices / board_local are aligned and image-down pixels.
    assert res.image_points.shape[0] == res.board_local_points.shape[0] == res.grid_indices.shape[0]
    assert res.image_points[:, 0].min() >= 0 and res.image_points[:, 1].min() >= 0

    if request.config.getoption("--make-figures", default=False):
        _make_figure(img, res)


def test_stepped_detector_fails_cleanly_on_blank():
    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    res = det.detect(np.zeros((400, 400), np.uint8))
    assert not res.success
    assert res.image_points.shape == (0, 2)
    assert "error" in res.diagnostics


def _make_figure(img, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray(res.diagnostics["level_labels"])
    ip = res.image_points
    bl = res.board_local_points

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(img, cmap="gray", origin="upper")
    a = labels == "A"
    ax[0].scatter(ip[a, 0], ip[a, 1], s=14, facecolors="none", edgecolors="tab:blue", label="level A")
    ax[0].scatter(ip[~a, 0], ip[~a, 1], s=14, facecolors="none", edgecolors="tab:red", label="level B")
    ax[0].set_title("stepped detection overlay (image-down px)")
    ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]"); ax[0].legend(loc="upper right")

    sc = ax[1].scatter(bl[:, 0], bl[:, 1], c=bl[:, 2], cmap="coolwarm", s=22)
    ax[1].set_title("board-local points (colour = z mm)")
    ax[1].set_xlabel("X [mm]"); ax[1].set_ylabel("Y [mm]"); ax[1].set_aspect("equal")
    ax[1].invert_yaxis()
    fig.colorbar(sc, ax=ax[1], label="z [mm]")
    fig.tight_layout()
    fig.savefig(_figpath("stepped-detect"), dpi=110)
    plt.close(fig)
