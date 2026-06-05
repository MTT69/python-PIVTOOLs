"""calibration2.figures — archival proof figures, written beside the model.

Generated at fit-time inside ``run_mono`` / ``run_stereo`` while the per-view poses,
the per-view detections, and the world frame are still in scope (the record stays
lean — per-view arrays are drawn here and never persisted). Every figure is wrapped
so a drawing failure is logged (visible, per CLAUDE.md "no silent fallbacks") but
never aborts the calibration.

All figures are static matplotlib PNGs (Agg) — no interactive HTML, no extra deps.
Ported from ``pivtools_gui/calibration/calibration_figures.py`` and re-bound to the
v2 dataclasses (``DetectionResult``, ``CameraModel``, ``WorldFrame``); v2 does not
import v1. Public entry points are the two orchestrators ``write_mono_figures`` /
``write_stereo_figures``; the per-figure functions are individually safe to call too.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib.gridspec import GridSpec  # noqa: E402

try:  # loguru is the project logger; fall back to stdlib if unavailable
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from scipy.spatial import cKDTree  # noqa: E402

if TYPE_CHECKING:  # annotations only — avoids a runtime import cycle
    from .camera_model import CameraModel
    from .detection.base import DetectionResult
    from .record import WorldFrame

# Camera colours, shared by the per-camera figures and the stereo 3D scene.
CAM_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _save(fig, output_path, dpi: int = 150) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Image -> uint8 grayscale for display (handles BGR, float, uint16)."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = np.squeeze(img)
    if img.dtype in (np.float32, np.float64):
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img


def _draw_grid_network(ax, centers, grid_indices, color="limegreen", lw=0.6, alpha=0.7) -> None:
    """Lines between grid-neighbouring points (col/row +1) on a mpl axis."""
    idx = {(int(g[0]), int(g[1])): i for i, g in enumerate(grid_indices)}
    for i, g in enumerate(grid_indices):
        c, r = int(g[0]), int(g[1])
        for dc, dr in ((1, 0), (0, 1)):
            j = idx.get((c + dc, r + dr))
            if j is not None:
                ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]],
                        color=color, linewidth=lw, alpha=alpha)


def _dot_at_grid(grid_indices: np.ndarray, image_points: np.ndarray, col, row):
    """Pixel of the detected dot nearest grid cell (col,row), or None."""
    gi = np.asarray(grid_indices, np.float64).reshape(-1, 2)
    d = np.abs(gi[:, 0] - col) + np.abs(gi[:, 1] - row)
    k = int(np.argmin(d))
    return None if d[k] > 0.5 else np.asarray(image_points, np.float64).reshape(-1, 2)[k]


# ---------------------------------------------------------------------------
# 1. Detection overlay (per view)
# ---------------------------------------------------------------------------

def write_detection_figure(image, detection: "DetectionResult", output_path, title=None) -> None:
    """Detected features + grid-index labels on the image, plus the grid network."""
    try:
        gray = _to_uint8_gray(image)
        pts = np.asarray(detection.image_points, np.float64).reshape(-1, 2)
        gi = None if detection.grid_indices is None else np.asarray(detection.grid_indices).reshape(-1, 2)
        ids = None if detection.point_ids is None else np.asarray(detection.point_ids).reshape(-1)
        n = len(pts)
        h, w = gray.shape[:2]
        scale = max(1, w // 1400)

        if gi is not None and n:
            ncols = int(gi[:, 0].max() - gi[:, 0].min() + 1)
            nrows = int(gi[:, 1].max() - gi[:, 1].min() + 1)
            grid_txt = f"grid {ncols}x{nrows}"
        else:
            grid_txt = "no grid"
        status = f"{n} features, {grid_txt}" if detection.success else "FAILED"

        fig = plt.figure(figsize=(18, 8))
        fig.suptitle(f"{title or 'Detection'} — {status}", fontsize=13,
                     color="darkgreen" if detection.success else "darkred")
        gs = GridSpec(1, 2, figure=fig, wspace=0.08)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(gray[::scale, ::scale], cmap="gray")
        ax1.scatter(pts[:, 0] / scale, pts[:, 1] / scale, s=10, facecolors="none",
                    edgecolors="lime", linewidths=0.6)
        if gi is not None:
            label_all = n <= 130
            for k in range(n):
                if label_all or (int(gi[k, 0]) % 2 == 0 and int(gi[k, 1]) % 2 == 0):
                    ax1.text(pts[k, 0] / scale + 3, pts[k, 1] / scale - 3,
                             f"{int(gi[k, 0])},{int(gi[k, 1])}", color="yellow", fontsize=5)
        ax1.set_title("Features + grid indices (col,row)", fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 1])
        if gi is not None and n:
            _draw_grid_network(ax2, pts, gi)
        if detection.board_type == "charuco" and ids is not None and n:
            ax2.scatter(pts[:, 0], pts[:, 1], c=plt.cm.hsv(ids / max(int(ids.max()), 1)),
                        s=16, zorder=5)
        elif n:
            ax2.scatter(pts[:, 0], pts[:, 1], c="limegreen", s=12, zorder=5)
        ax2.invert_yaxis(); ax2.set_aspect("equal", adjustable="datalim")
        ax2.set_title("Grid network", fontsize=10)
        ax2.set_xlabel("x (px)"); ax2.set_ylabel("y (px, image-down)")
        _save(fig, output_path)
    except Exception:
        logger.warning(f"detection figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 2. World-frame pick (datum view)
# ---------------------------------------------------------------------------

def write_world_frame_figure(image, detection: "DetectionResult", wf: "WorldFrame",
                             spacing, output_path, title=None) -> None:
    """Datum view: snapped origin ● + +X/+Y arrows to the neighbour dots.

    For ``clicks`` mode the raw clicks are drawn faint and the snapped dots get the
    arrows; for ``grid`` / ``default`` mode the origin + axis-neighbour dots are
    located directly from the grid indices.
    """
    try:
        gray = _to_uint8_gray(image)
        h, w = gray.shape[:2]
        scale = max(1, w // 1400)
        pts = np.asarray(detection.image_points, np.float64).reshape(-1, 2)
        gi = np.asarray(detection.grid_indices).reshape(-1, 2)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(gray[::scale, ::scale], cmap="gray")
        ax.scatter(pts[:, 0] / scale, pts[:, 1] / scale, s=6, facecolors="none",
                   edgecolors="deepskyblue", linewidths=0.4, alpha=0.6)

        def arrows(o, xa, ya):
            ax.annotate("", xy=xa / scale, xytext=o / scale,
                        arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.4))
            ax.annotate("", xy=ya / scale, xytext=o / scale,
                        arrowprops=dict(arrowstyle="-|>", color="magenta", lw=2.4))
            ax.scatter([o[0] / scale], [o[1] / scale], s=180, facecolors="none",
                       edgecolors="lime", linewidths=2.4, zorder=10)
            ax.scatter([o[0] / scale], [o[1] / scale], s=45, color="lime", marker="+", zorder=11)
            ax.text(o[0] / scale + 7, o[1] / scale - 7, "O", color="lime", fontsize=13, fontweight="bold")
            ax.text(xa[0] / scale + 7, xa[1] / scale - 7, "+X", color="yellow", fontsize=13, fontweight="bold")
            ax.text(ya[0] / scale + 7, ya[1] / scale - 7, "+Y", color="magenta", fontsize=13, fontweight="bold")

        sub = f"{wf.mode} frame"
        if wf.mode == "clicks" and wf.origin_px is not None:
            tree = cKDTree(pts)

            def snap(click):
                _, i = tree.query(np.asarray(click, np.float64).reshape(2))
                return pts[int(i)]

            o, xa, ya = snap(wf.origin_px), snap(wf.x_axis_px), snap(wf.y_axis_px)
            for c, col in ((wf.origin_px, "white"), (wf.x_axis_px, "yellow"), (wf.y_axis_px, "magenta")):
                c = np.asarray(c, np.float64).reshape(2)
                ax.plot(c[0] / scale, c[1] / scale, "x", color=col, ms=7, alpha=0.45)
            arrows(o, xa, ya)
            if wf.origin_grid is not None:
                sub = (f"clicks — origin dot grid (col,row)=("
                       f"{int(wf.origin_grid[0])},{int(wf.origin_grid[1])}), spacing={float(spacing):g} mm")
        elif wf.origin_grid is not None:
            og = np.asarray(wf.origin_grid, np.float64).reshape(2)
            # +X / +Y grid steps under the resolved frame.
            xstep = (wf.col_sign, 0) if not wf.swap_axes else (0, wf.col_sign)
            ystep = (0, wf.row_sign) if not wf.swap_axes else (wf.row_sign, 0)
            o = _dot_at_grid(gi, pts, og[0], og[1])
            xa = _dot_at_grid(gi, pts, og[0] + xstep[0], og[1] + xstep[1])
            ya = _dot_at_grid(gi, pts, og[0] + ystep[0], og[1] + ystep[1])
            if o is not None and xa is not None and ya is not None:
                arrows(o, xa, ya)
            elif o is not None:
                ax.scatter([o[0] / scale], [o[1] / scale], s=180, facecolors="none",
                           edgecolors="lime", linewidths=2.4, zorder=10)
                ax.text(o[0] / scale + 7, o[1] / scale - 7, "O", color="lime", fontsize=13, fontweight="bold")
            sub = (f"{wf.mode} — origin dot grid (col,row)=({int(og[0])},{int(og[1])}), "
                   f"spacing={float(spacing):g} mm")

        ax.set_title(f"{title or 'World frame'}\n{sub}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        _save(fig, output_path)
    except Exception:
        logger.warning(f"world-frame figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3. Reprojection residuals + per-view RMS
# ---------------------------------------------------------------------------

def write_reprojection_figure(object_points, image_points, K, dist, rvecs, tvecs,
                              per_view, rms, output_path, pose_indices=None, title=None) -> None:
    """(dx,dy) residual scatter coloured by view + RMS circle, and a per-view RMS bar chart."""
    try:
        n_views = len(object_points)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_views, 1)))
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(f"{title or 'Reprojection'} — {n_views} views, overall RMS={rms:.4f} px", fontsize=13)
        gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1.1, 1.0])

        ax = fig.add_subplot(gs[0, 0])
        all_r: List[np.ndarray] = []
        for i in range(n_views):
            proj, _ = cv2.projectPoints(
                np.asarray(object_points[i], np.float64).reshape(-1, 1, 3),
                np.asarray(rvecs[i], np.float64).reshape(3),
                np.asarray(tvecs[i], np.float64).reshape(3), K, dist)
            res = (np.asarray(image_points[i], np.float64).reshape(-1, 1, 2) - proj).reshape(-1, 2)
            all_r.append(res)
            ax.scatter(res[:, 0], res[:, 1], s=4, alpha=0.5, color=colors[i])
        allr = np.vstack(all_r) if all_r else np.zeros((0, 2))
        ax.add_patch(plt.Circle((0, 0), rms, fill=False, color="red", ls="--", lw=1.5,
                                label=f"RMS={rms:.3f} px"))
        mr = max(float(np.abs(allr).max()) * 1.15, rms * 1.3) if len(allr) else max(rms * 1.3, 1e-3)
        ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("residual x (px)"); ax.set_ylabel("residual y (px)")
        ax.set_title(f"Reprojection residuals ({len(allr)} pts)", fontsize=10)
        ax.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        labels = [str(p) for p in (pose_indices if pose_indices is not None else range(n_views))]
        ax2.bar(labels, per_view, color=colors[: len(per_view)])
        ax2.axhline(rms, color="red", ls="--", lw=1, label=f"overall {rms:.3f}")
        ax2.set_xlabel("view (pose index)"); ax2.set_ylabel("RMS (px)")
        ax2.set_title("Per-view RMS", fontsize=10); ax2.legend(fontsize=8)
        _save(fig, output_path)
    except Exception:
        logger.warning(f"reprojection figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 4. Distortion map
# ---------------------------------------------------------------------------

def write_distortion_map_figure(K, dist, image_size, output_path, title=None) -> None:
    """Sensor heatmap + quiver of the distortion displacement (distorted -> ideal)."""
    try:
        w, h = int(image_size[0]), int(image_size[1])
        K = np.asarray(K, np.float64)
        new_cam, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        gx = np.linspace(0, w - 1, 25)
        gy = np.linspace(0, h - 1, 25)
        GX, GY = np.meshgrid(gx, gy)
        grid = np.column_stack([GX.ravel(), GY.ravel()]).astype(np.float32).reshape(-1, 1, 2)
        und = cv2.undistortPoints(grid, K, dist, P=new_cam).reshape(-1, 2)
        orig = grid.reshape(-1, 2)
        disp = und - orig
        mag = np.linalg.norm(disp, axis=1)

        fig, ax = plt.subplots(figsize=(10, 9))
        sc = ax.scatter(orig[:, 0], orig[:, 1], c=mag, cmap="viridis", s=40)
        ax.quiver(orig[:, 0], orig[:, 1], disp[:, 0], disp[:, 1], angles="xy",
                  scale_units="xy", scale=1, color="white", width=0.002, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="displacement (px)")
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px, image-down)")
        kf = np.asarray(dist, np.float64).reshape(-1)
        ax.set_title(f"{title or 'Distortion map'} — k1={kf[0]:.4g} k2={kf[1]:.4g} "
                     f"p1={kf[2]:.4g} p2={kf[3]:.4g}", fontsize=11)
        _save(fig, output_path)
    except Exception:
        logger.warning(f"distortion-map figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3D scene helpers
# ---------------------------------------------------------------------------

def _frustum_faces(apex, fwd, right, up):
    """mpl Poly3DCollection faces for a pyramidal camera frustum."""
    base = apex + fwd
    c = [base + right + up, base - right + up, base - right - up, base + right - up]
    return [[apex, c[0], c[1]], [apex, c[1], c[2]], [apex, c[2], c[3]], [apex, c[3], c[0]],
            [c[0], c[1], c[2], c[3]]]


def _board_corners_cam(R, t, board_local):
    """4 corners of a board's extent, transformed board-local -> camera frame."""
    pl = np.asarray(board_local, np.float64).reshape(-1, 3)
    x0, x1 = float(pl[:, 0].min()), float(pl[:, 0].max())
    y0, y1 = float(pl[:, 1].min()), float(pl[:, 1].max())
    corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    return (R @ corners.T).T + np.asarray(t, np.float64).reshape(3)


# ---------------------------------------------------------------------------
# 6. Boards in physical space — all poses as stacked planes (DaVis-style), planar
# ---------------------------------------------------------------------------

def write_boards_planes_3d(cam: "CameraModel", board_local_points: Sequence[np.ndarray],
                           rvecs, tvecs, pose_indices, output_path, datum_pos=0, title=None) -> None:
    """Every detected board pose drawn as a tilted plane in the camera frame.

    DaVis-style: the planes stack in physical space around the camera, the datum pose
    is bold/orange, the others faint. The view auto-bounds to the plate cluster (the
    camera sits ~metres away at PIV magnification) and a ray annotates its direction.
    """
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        planes = []
        for rv, tv, pl in zip(rvecs, tvecs, board_local_points):
            R, _ = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3))
            planes.append(_board_corners_cam(R, tv, pl))
        if not planes:
            return
        allc = np.vstack(planes)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        for k, c in enumerate(planes):
            datum = (k == datum_pos)
            ax.add_collection3d(Poly3DCollection(
                [c], alpha=0.6 if datum else 0.15,
                facecolor="#ff7f0e" if datum else "#4a78b0",
                edgecolor="#c25e00" if datum else "#3b6ea5",
                linewidths=2.2 if datum else 0.7))
        ax.plot([], [], color="#ff7f0e", lw=3, label="datum plane")
        ax.plot([], [], color="#3b6ea5", lw=1.5, label="other poses")

        # Camera direction: planes sit ~|t| away; draw a short ray toward the origin.
        ctr = allc.mean(0)
        dist_mm = float(np.linalg.norm(ctr))
        span = float(np.max(allc.max(0) - allc.min(0))) or 1.0
        tip = ctr - (ctr / (dist_mm + 1e-9)) * 0.35 * span
        ax.plot([ctr[0], tip[0]], [ctr[1], tip[1]], [ctr[2], tip[2]], color="crimson", lw=1.5)
        ax.text(tip[0], tip[1], tip[2], f"→ camera (~{dist_mm:.0f} mm)", color="crimson", fontsize=10)

        ax.set_xlim(allc[:, 0].min(), allc[:, 0].max())
        ax.set_ylim(allc[:, 1].min(), allc[:, 1].max())
        ax.set_zlim(allc[:, 2].min(), allc[:, 2].max())
        ax.set_xlabel("X cam (mm)"); ax.set_ylabel("Y cam (mm)"); ax.set_zlabel("Z cam (mm)")
        ax.set_title(title or f"Boards in physical space — {len(planes)} poses (datum highlighted)",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.view_init(elev=18, azim=-60)
        _save(fig, output_path, dpi=150)
    except Exception:
        logger.warning(f"boards-planes figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 7. Cameras relative to the board (stereo) — datum plane only, PNG
# ---------------------------------------------------------------------------

def _camera_records(models, labels, board_world):
    """Camera world centres + board-pointing optical axes (handles SIG reflection)."""
    centroid = np.asarray(board_world, np.float64).reshape(-1, 3).mean(0)
    recs = []
    for m, lab, col in zip(models, labels, CAM_COLORS):
        R = np.asarray(m.R, np.float64)
        t = np.asarray(m.t, np.float64).reshape(3)
        pos = (-R.T @ t).ravel()
        direction = centroid - pos
        nrm = float(np.linalg.norm(direction))
        axis = direction / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        recs.append({"label": lab, "color": col, "pos": pos, "axis": axis,
                     "right": (R.T @ np.array([1.0, 0, 0])).ravel(),
                     "up": (R.T @ np.array([0, -1.0, 0])).ravel()})
    return recs


def write_cameras_3d(model1: "CameraModel", model2: "CameraModel", board_world,
                     R_stereo, T_stereo, output_path) -> None:
    """World origin triad + datum board plane + both camera frusta/axes (PNG)."""
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        bw = np.asarray(board_world, np.float64).reshape(-1, 3)
        recs = _camera_records([model1, model2], ["Cam1", "Cam2"], bw)
        baseline = float(np.linalg.norm(recs[0]["pos"] - recs[1]["pos"]))
        ang = float(np.degrees(np.arccos(np.clip(
            (np.trace(np.asarray(R_stereo, np.float64)) - 1) / 2, -1, 1))))
        Tn = float(np.linalg.norm(np.asarray(T_stereo, np.float64)))

        scene = np.vstack([bw] + [r["pos"].reshape(1, 3) for r in recs])
        span = float(np.max(scene.max(0) - scene.min(0))) or 1.0
        fr = 0.05 * span
        axis_len = 0.25 * span

        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection="3d")
        # datum board as a translucent plane + its dots
        x0, x1 = bw[:, 0].min(), bw[:, 0].max()
        y0, y1 = bw[:, 1].min(), bw[:, 1].max()
        z0 = float(bw[:, 2].mean())
        ax.add_collection3d(Poly3DCollection(
            [np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])],
            alpha=0.45, facecolor="#ff7f0e", edgecolor="#c25e00", linewidths=1.5))
        ax.scatter(bw[:, 0], bw[:, 1], bw[:, 2], c="#c25e00", s=8, alpha=0.7, label="datum board")
        for col, vec in (("r", [1, 0, 0]), ("g", [0, 1, 0]), ("b", [0, 0, 1])):
            v = np.array(vec, float) * 0.15 * span
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2)
        for r in recs:
            ax.add_collection3d(Poly3DCollection(
                _frustum_faces(r["pos"], r["axis"] * fr, r["right"] * fr * 0.7, r["up"] * fr * 0.5),
                alpha=0.35, facecolor=r["color"], edgecolor=r["color"]))
            end = r["pos"] + r["axis"] * axis_len
            ax.plot([r["pos"][0], end[0]], [r["pos"][1], end[1]], [r["pos"][2], end[2]],
                    color=r["color"], lw=1.5, alpha=0.7)
            ax.text(r["pos"][0], r["pos"][1], r["pos"][2] + fr * 1.5, r["label"],
                    color=r["color"], fontsize=11, fontweight="bold", ha="center")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.set_title(f"Cameras relative to board — baseline |T|={Tn:.1f} mm, "
                     f"stereo angle={ang:.2f} deg", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        _save(fig, output_path, dpi=150)
    except Exception:
        logger.warning(f"cameras-3d figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 8. Dewarp red-cyan anaglyph (stereo agreement in world space)
# ---------------------------------------------------------------------------

def write_dewarp_overlay(model1: "CameraModel", model2: "CameraModel", img1, img2,
                         board_world, spacing, output_path, title=None, mm_per_px=0.1) -> None:
    """Remap both images to the world Z=0 plane, overlay as red(cam1)/cyan(cam2)."""
    try:
        bw = np.asarray(board_world, np.float64).reshape(-1, 3)
        x_min, x_max = float(bw[:, 0].min()) - 5, float(bw[:, 0].max()) + 5
        y_min, y_max = float(bw[:, 1].min()) - 5, float(bw[:, 1].max()) + 5
        nx = max(int((x_max - x_min) / mm_per_px), 32)
        ny = max(int((y_max - y_min) / mm_per_px), 32)
        X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        world = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float64)

        def dewarp(model, img):
            g = _to_uint8_gray(img).astype(np.float64)
            proj, _ = cv2.projectPoints(world, model.rvec, np.asarray(model.t, np.float64).reshape(3),
                                        np.asarray(model.K, np.float64), np.asarray(model.dist, np.float64))
            proj = proj.reshape(-1, 2)
            mx = proj[:, 0].reshape(X.shape).astype(np.float32)
            my = proj[:, 1].reshape(X.shape).astype(np.float32)
            dw = cv2.remap(g, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = (mx >= 0) & (mx < g.shape[1]) & (my >= 0) & (my < g.shape[0])
            dw[~valid] = 0
            lo, hi = float(dw.min()), float(dw.max())
            out = ((dw - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(dw, np.uint8)
            out[~valid] = 0
            return out

        r = dewarp(model1, img1)
        c = dewarp(model2, img2)
        overlay = np.stack([r, c, c], axis=-1)

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin="lower")
        ax.axhline(0, color="lime", lw=1, alpha=0.6); ax.axvline(0, color="lime", lw=1, alpha=0.6)
        ax.scatter(0, 0, s=250, c="lime", marker="+", linewidths=3, zorder=20)
        sp = float(spacing)
        ax.set_xticks(np.arange(np.ceil(x_min / sp) * sp, x_max, sp))
        ax.set_yticks(np.arange(np.ceil(y_min / sp) * sp, y_max, sp))
        ax.grid(True, color="white", alpha=0.15, lw=0.5)
        ax.scatter(bw[:, 0], bw[:, 1], s=22, facecolors="none", edgecolors="white",
                   linewidths=0.6, marker="s", alpha=0.7, label="board dots")
        ax.set_xlabel("X world (mm)"); ax.set_ylabel("Y world (mm)")
        ax.set_title(title or "Dewarp overlay (Cam1=red, Cam2=cyan; sharp = agreement)", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, output_path, dpi=160)
    except Exception:
        logger.warning(f"dewarp overlay figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Orchestrators — called by the pipeline at fit time
# ---------------------------------------------------------------------------

def write_mono_figures(figure_dir, *, images, detections, used, K, dist, rvecs, tvecs,
                       per_view, rms, cam, wf, spacing, board_type, datum_index,
                       board_meta=None, prefix="") -> None:
    """All single-camera proof figures. ``used`` = list of (pose_index, DetectionResult).

    The boards-in-physical-space planes figure is drawn only for a true mono run
    (``prefix`` empty); stereo per-camera calls skip it (the stereo run gets one
    cameras-relative-to-board figure instead).
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    pose_indices = [i for i, _ in used]
    objs = [d.board_local_points for _, d in used]
    imgs = [d.image_points for _, d in used]
    datum_pos = pose_indices.index(datum_index) if datum_index in pose_indices else 0

    for i, d in used:
        write_detection_figure(images[i], d, figd / f"{prefix}detection_{i:02d}.png",
                               title=f"{prefix}pose {i}")
    write_world_frame_figure(images[datum_index], detections[datum_index], wf, spacing,
                             figd / f"{prefix}world_frame.png", title=f"{prefix}world frame")
    write_reprojection_figure(objs, imgs, K, dist, rvecs, tvecs, per_view, rms,
                              figd / f"{prefix}reprojection.png", pose_indices=pose_indices,
                              title=f"{prefix}reprojection")
    write_distortion_map_figure(K, dist, cam.image_size, figd / f"{prefix}distortion_map.png",
                                title=f"{prefix}distortion")
    if not prefix:
        write_boards_planes_3d(cam, objs, rvecs, tvecs, pose_indices,
                               figd / "boards_3d.png", datum_pos=datum_pos)


def write_stereo_figures(figure_dir, *, model1, model2, R_stereo, T_stereo, img1, img2,
                         datum_board_world, spacing) -> None:
    """Stereo-only proof figures: cameras relative to the datum board + the dewarp anaglyph."""
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    write_cameras_3d(model1, model2, datum_board_world, R_stereo, T_stereo, figd / "cameras_3d.png")
    write_dewarp_overlay(model1, model2, img1, img2, datum_board_world, spacing,
                         figd / "dewarp_overlay.png")
