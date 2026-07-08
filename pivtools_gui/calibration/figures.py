"""calibration.figures — archival proof figures, written beside the model.

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

import math
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import cv2
import matplotlib
import numpy as np

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
    from .camera_model import CameraModel, PolynomialModel
    from .detection.base import DetectionResult
    from .record import WorldFrame

# Camera colours, shared by the per-camera figures and the stereo / joint 3D scenes.
# (>=8 so a joint rig with many cameras is not silently truncated by ``_camera_records``,
# which zips models against this tuple.)
CAM_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _save(fig, output_path, dpi: int = 150) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _is_2d_image(img) -> bool:
    """True if ``img`` is a usable 2-D (gray) or 3-D (colour) image array.

    Guards the dewarp/detection writers against ``None`` / 0-D / 1-D arrays — a missing
    datum frame reaches the figure layer as ``np.asarray(None)`` (0-D), and a bare
    ``g.shape[1]`` then throws an opaque ``IndexError`` instead of a clear message.
    """
    if img is None:
        return False
    arr = np.asarray(img)
    return arr.ndim >= 2 and arr.size > 0


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Image -> uint8 grayscale for display (handles BGR, float, uint16)."""
    img = np.asarray(img)
    if img.ndim < 2 or img.size == 0:
        raise ValueError(f"image is not 2-D: shape={img.shape}")
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = np.squeeze(img)
    if img.dtype in (np.float32, np.float64, np.uint16):
        # Min-max stretch for float AND uint16: a fixed /256 on uint16 crushes
        # 12-bit-range data (0..4095 -> 0..15, near-black figures).
        lo, hi = float(img.min()), float(img.max())
        img = (
            ((img.astype(np.float64) - lo) / (hi - lo) * 255).astype(np.uint8)
            if hi > lo
            else np.zeros_like(img, np.uint8)
        )
    return img


def _draw_grid_network(
    ax, centers, grid_indices, color="limegreen", lw=0.6, alpha=0.7
) -> None:
    """Lines between grid-neighbouring points (col/row +1) on a mpl axis."""
    idx = {(int(g[0]), int(g[1])): i for i, g in enumerate(grid_indices)}
    for i, g in enumerate(grid_indices):
        c, r = int(g[0]), int(g[1])
        for dc, dr in ((1, 0), (0, 1)):
            j = idx.get((c + dc, r + dr))
            if j is not None:
                ax.plot(
                    [centers[i, 0], centers[j, 0]],
                    [centers[i, 1], centers[j, 1]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )


def _dot_at_grid(grid_indices: np.ndarray, image_points: np.ndarray, col, row):
    """Pixel of the detected dot nearest grid cell (col,row), or None."""
    gi = np.asarray(grid_indices, np.float64).reshape(-1, 2)
    d = np.abs(gi[:, 0] - col) + np.abs(gi[:, 1] - row)
    k = int(np.argmin(d))
    return (
        None if d[k] > 0.5 else np.asarray(image_points, np.float64).reshape(-1, 2)[k]
    )


# GUI overlay convention (StereoSteppedCalibration.tsx): peak face blue, trough face red.
_STEPPED_PEAK_COLOR = (80 / 255, 140 / 255, 1.0)     # GUI rgba(80,140,255)
_STEPPED_TROUGH_COLOR = (1.0, 120 / 255, 120 / 255)  # GUI rgba(255,120,120)


def _draw_stepped_grid_networks(ax, level_data, level_a_face) -> bool:
    """Draw one grid network per stepped face (peak=blue, trough=red), like the GUI.

    A stepped board carries two interleaved grids; a single network across both bridges
    peak<->trough dots with spurious diagonals (the artefact this fixes). Each level's own
    ``centers`` + ``grid_indices`` give a clean within-face network. ``level_a_face``
    ('peak'|'trough') labels which face level 'a' is for this pose (from ``pose_levels``);
    None defaults to 'peak' (matching the GUI's no-pose-level default). Returns False when
    there is no usable two-level data so the caller falls back to the single network.
    """
    if not level_data:
        return False
    a_is_peak = level_a_face != "trough"
    colors = {"a": _STEPPED_PEAK_COLOR, "b": _STEPPED_TROUGH_COLOR}
    faces = {"a": "peak", "b": "trough"}
    if not a_is_peak:  # level 'a' is the trough face -> swap, so peak stays blue
        colors = {"a": _STEPPED_TROUGH_COLOR, "b": _STEPPED_PEAK_COLOR}
        faces = {"a": "trough", "b": "peak"}
    drew = False
    for key in ("a", "b"):
        lv = level_data.get(key)
        if not lv:
            continue
        centers, gidx = lv.get("centers"), lv.get("grid_indices")
        if centers is None or gidx is None:
            continue
        c = np.asarray(centers, np.float64).reshape(-1, 2)
        g = np.asarray(gidx).reshape(-1, 2)
        if len(c) == 0 or len(c) != len(g):
            continue
        _draw_grid_network(ax, c, g, color=colors[key])
        ax.scatter(
            c[:, 0], c[:, 1], color=[colors[key]], s=12, zorder=5,
            label=f"{faces[key]} ({len(c)})",
        )
        drew = True
    if drew:
        ax.legend(fontsize=8, loc="upper right")
    return drew


# ---------------------------------------------------------------------------
# 1. Detection overlay (per view)
# ---------------------------------------------------------------------------


def write_detection_figure(
    image, detection: "DetectionResult", output_path, title=None, level_a_face=None
) -> None:
    """Detected features + grid-index labels on the image, plus the grid network.

    ``level_a_face`` ('peak'|'trough'|None): for a stepped board, the face of level 'a'
    in this pose, so the grid network is drawn as two separate per-face networks
    (peak=blue, trough=red) instead of one network bridging both faces.
    """
    try:
        gray = _to_uint8_gray(image)
        pts = np.asarray(detection.image_points, np.float64).reshape(-1, 2)
        gi = (
            None
            if detection.grid_indices is None
            else np.asarray(detection.grid_indices).reshape(-1, 2)
        )
        ids = (
            None
            if detection.point_ids is None
            else np.asarray(detection.point_ids).reshape(-1)
        )
        h, w = gray.shape[:2]
        scale = max(1, w // 1400)

        # Show only genuinely-detected, accepted dots. Synthetic (rescued/infilled) points are
        # dropped from the FIGURE (they are still stored in the record/sidecar); RANSAC-rejected
        # points are already absent from the detection result. dotboard-only — charuco has no mask.
        synth = detection.synthetic_mask
        if synth is not None:
            synth = np.asarray(synth, dtype=bool).reshape(-1)
            if len(synth) == len(pts) and synth.any():
                keep = ~synth
                pts = pts[keep]
                if gi is not None:
                    gi = gi[keep]
                if ids is not None:
                    ids = ids[keep]
        n = len(pts)

        if gi is not None and n:
            ncols = int(gi[:, 0].max() - gi[:, 0].min() + 1)
            nrows = int(gi[:, 1].max() - gi[:, 1].min() + 1)
            grid_txt = f"grid {ncols}x{nrows}"
        else:
            grid_txt = "no grid"
        status = f"{n} features, {grid_txt}" if detection.success else "FAILED"

        # Only the detector warning is surfaced. The synthetic / RANSAC-rejected counts are kept in
        # the record/sidecar diagnostics but intentionally NOT drawn here (the figure shows the
        # genuine accepted dots, nothing else).
        diag = detection.diagnostics or {}
        extras = []
        if diag.get("warning"):
            extras.append(str(diag["warning"]))
        subtitle = ("\n" + " | ".join(extras)) if extras else ""

        fig = plt.figure(figsize=(18, 8))
        fig.suptitle(
            f"{title or 'Detection'} — {status}{subtitle}",
            fontsize=13,
            color="darkgreen" if detection.success else "darkred",
        )
        gs = GridSpec(1, 2, figure=fig, wspace=0.08)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(gray[::scale, ::scale], cmap="gray")
        ax1.scatter(
            pts[:, 0] / scale,
            pts[:, 1] / scale,
            s=10,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.6,
        )
        if gi is not None:
            label_all = n <= 130
            for k in range(n):
                if label_all or (int(gi[k, 0]) % 2 == 0 and int(gi[k, 1]) % 2 == 0):
                    ax1.text(
                        pts[k, 0] / scale + 3,
                        pts[k, 1] / scale - 3,
                        f"{int(gi[k, 0])},{int(gi[k, 1])}",
                        color="yellow",
                        fontsize=5,
                    )
        ax1.set_title("Features + grid indices (col,row)", fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 1])
        # Stepped board -> two per-face networks (peak=blue, trough=red); else one network.
        drew_stepped = (
            _draw_stepped_grid_networks(ax2, detection.level_data, level_a_face)
            if n
            else False
        )
        if not drew_stepped:
            if gi is not None and n:
                _draw_grid_network(ax2, pts, gi)
            if detection.board_type == "charuco" and ids is not None and n:
                ax2.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=plt.cm.hsv(ids / max(int(ids.max()), 1)),
                    s=16,
                    zorder=5,
                )
            elif n:
                ax2.scatter(pts[:, 0], pts[:, 1], c="limegreen", s=12, zorder=5)
        ax2.invert_yaxis()
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.set_title("Grid network", fontsize=10)
        ax2.set_xlabel("x (px)")
        ax2.set_ylabel("y (px, image-down)")
        _save(fig, output_path)
    except Exception:
        logger.warning(f"detection figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 2. World-frame pick (datum view)
# ---------------------------------------------------------------------------


def write_world_frame_figure(
    image,
    detection: "DetectionResult",
    wf: "WorldFrame",
    spacing,
    output_path,
    title=None,
) -> None:
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
        ax.scatter(
            pts[:, 0] / scale,
            pts[:, 1] / scale,
            s=6,
            facecolors="none",
            edgecolors="deepskyblue",
            linewidths=0.4,
            alpha=0.6,
        )

        def arrows(o, xa, ya):
            ax.annotate(
                "",
                xy=xa / scale,
                xytext=o / scale,
                arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.4),
            )
            ax.annotate(
                "",
                xy=ya / scale,
                xytext=o / scale,
                arrowprops=dict(arrowstyle="-|>", color="magenta", lw=2.4),
            )
            ax.scatter(
                [o[0] / scale],
                [o[1] / scale],
                s=180,
                facecolors="none",
                edgecolors="lime",
                linewidths=2.4,
                zorder=10,
            )
            ax.scatter(
                [o[0] / scale],
                [o[1] / scale],
                s=45,
                color="lime",
                marker="+",
                zorder=11,
            )
            ax.text(
                o[0] / scale + 7,
                o[1] / scale - 7,
                "O",
                color="lime",
                fontsize=13,
                fontweight="bold",
            )
            ax.text(
                xa[0] / scale + 7,
                xa[1] / scale - 7,
                "+X",
                color="yellow",
                fontsize=13,
                fontweight="bold",
            )
            ax.text(
                ya[0] / scale + 7,
                ya[1] / scale - 7,
                "+Y",
                color="magenta",
                fontsize=13,
                fontweight="bold",
            )

        sub = f"{wf.mode} frame"
        if wf.mode == "clicks" and wf.origin_px is not None:
            tree = cKDTree(pts)

            def snap(click):
                _, i = tree.query(np.asarray(click, np.float64).reshape(2))
                return pts[int(i)]

            o, xa, ya = snap(wf.origin_px), snap(wf.x_axis_px), snap(wf.y_axis_px)
            for c, col in (
                (wf.origin_px, "white"),
                (wf.x_axis_px, "yellow"),
                (wf.y_axis_px, "magenta"),
            ):
                c = np.asarray(c, np.float64).reshape(2)
                ax.plot(c[0] / scale, c[1] / scale, "x", color=col, ms=7, alpha=0.45)
            arrows(o, xa, ya)
            if wf.origin_grid is not None:
                sub = (
                    f"clicks — origin dot grid (col,row)=("
                    f"{int(wf.origin_grid[0])},{int(wf.origin_grid[1])}), spacing={float(spacing):g} mm"
                )
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
                ax.scatter(
                    [o[0] / scale],
                    [o[1] / scale],
                    s=180,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=2.4,
                    zorder=10,
                )
                ax.text(
                    o[0] / scale + 7,
                    o[1] / scale - 7,
                    "O",
                    color="lime",
                    fontsize=13,
                    fontweight="bold",
                )
            sub = (
                f"{wf.mode} — origin dot grid (col,row)=({int(og[0])},{int(og[1])}), "
                f"spacing={float(spacing):g} mm"
            )

        ax.set_title(f"{title or 'World frame'}\n{sub}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _save(fig, output_path)
    except Exception:
        logger.warning(f"world-frame figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3. Reprojection residuals + per-view RMS
# ---------------------------------------------------------------------------


def write_reprojection_figure(
    per_view,
    rms,
    output_path,
    pose_indices=None,
    title=None,
) -> None:
    """Per-view RMS reprojection-error bar chart with the overall RMS line."""
    try:
        n_views = len(per_view)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_views, 1)))
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(
            f"{title or 'Reprojection'} — {n_views} views, overall RMS={rms:.4f} px",
            fontsize=13,
        )
        labels = [
            str(p)
            for p in (pose_indices if pose_indices is not None else range(n_views))
        ]
        ax.bar(labels, per_view, color=colors[: len(per_view)])
        ax.axhline(rms, color="red", ls="--", lw=1, label=f"overall {rms:.3f}")
        ax.set_xlabel("view (pose index)")
        ax.set_ylabel("RMS (px)")
        ax.set_title("Per-view RMS", fontsize=10)
        ax.legend(fontsize=8)
        _save(fig, output_path)
    except Exception:
        logger.warning(f"reprojection figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3D scene helpers
# ---------------------------------------------------------------------------


def _frustum_faces(apex, fwd, right, up):
    """mpl Poly3DCollection faces for a pyramidal camera frustum."""
    base = apex + fwd
    c = [base + right + up, base - right + up, base - right - up, base + right - up]
    return [
        [apex, c[0], c[1]],
        [apex, c[1], c[2]],
        [apex, c[2], c[3]],
        [apex, c[3], c[0]],
        [c[0], c[1], c[2], c[3]],
    ]


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


def write_boards_planes_3d(
    cam: "CameraModel",
    board_local_points: Sequence[np.ndarray],
    rvecs,
    tvecs,
    pose_indices,
    output_path,
    datum_pos=0,
    title=None,
) -> None:
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
            datum = k == datum_pos
            ax.add_collection3d(
                Poly3DCollection(
                    [c],
                    alpha=0.6 if datum else 0.15,
                    facecolor="#ff7f0e" if datum else "#4a78b0",
                    edgecolor="#c25e00" if datum else "#3b6ea5",
                    linewidths=2.2 if datum else 0.7,
                )
            )
        ax.plot([], [], color="#ff7f0e", lw=3, label="datum plane")
        ax.plot([], [], color="#3b6ea5", lw=1.5, label="other poses")

        # Camera direction: planes sit ~|t| away; draw a short ray toward the origin.
        ctr = allc.mean(0)
        dist_mm = float(np.linalg.norm(ctr))
        span = float(np.max(allc.max(0) - allc.min(0))) or 1.0
        tip = ctr - (ctr / (dist_mm + 1e-9)) * 0.35 * span
        ax.plot(
            [ctr[0], tip[0]],
            [ctr[1], tip[1]],
            [ctr[2], tip[2]],
            color="crimson",
            lw=1.5,
        )
        ax.text(
            tip[0],
            tip[1],
            tip[2],
            f"→ camera (~{dist_mm:.0f} mm)",
            color="crimson",
            fontsize=10,
        )

        ax.set_xlim(allc[:, 0].min(), allc[:, 0].max())
        ax.set_ylim(allc[:, 1].min(), allc[:, 1].max())
        ax.set_zlim(allc[:, 2].min(), allc[:, 2].max())
        ax.set_xlabel("X cam (mm)")
        ax.set_ylabel("Y cam (mm)")
        ax.set_zlabel("Z cam (mm)")
        ax.set_title(
            title
            or f"Boards in physical space — {len(planes)} poses (datum highlighted)",
            fontsize=12,
        )
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
        recs.append(
            {
                "label": lab,
                "color": col,
                "pos": pos,
                "axis": axis,
                "right": (R.T @ np.array([1.0, 0, 0])).ravel(),
                "up": (R.T @ np.array([0, -1.0, 0])).ravel(),
            }
        )
    return recs


def _draw_cameras_scene(ax, recs, bw, *, board_alpha, scatter_alpha, board_label) -> float:
    """Shared 3D rig scene: a translucent board plane + its dots, the world triad, and each
    camera's frustum + optical axis + label. The two callers differ only in the board/scatter
    alphas and the board legend label; the caller owns the title/legend/save. Returns the scene
    ``span`` (mm)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scene = np.vstack([bw] + [r["pos"].reshape(1, 3) for r in recs])
    span = float(np.max(scene.max(0) - scene.min(0))) or 1.0
    fr = 0.05 * span
    axis_len = 0.25 * span

    x0, x1 = bw[:, 0].min(), bw[:, 0].max()
    y0, y1 = bw[:, 1].min(), bw[:, 1].max()
    z0 = float(bw[:, 2].mean())
    ax.add_collection3d(
        Poly3DCollection(
            [np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])],
            alpha=board_alpha,
            facecolor="#ff7f0e",
            edgecolor="#c25e00",
            linewidths=1.5,
        )
    )
    ax.scatter(
        bw[:, 0],
        bw[:, 1],
        bw[:, 2],
        c="#c25e00",
        s=8,
        alpha=scatter_alpha,
        label=board_label,
    )
    for col, vec in (("r", [1, 0, 0]), ("g", [0, 1, 0]), ("b", [0, 0, 1])):
        v = np.array(vec, float) * 0.15 * span
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2)
    for r in recs:
        ax.add_collection3d(
            Poly3DCollection(
                _frustum_faces(
                    r["pos"],
                    r["axis"] * fr,
                    r["right"] * fr * 0.7,
                    r["up"] * fr * 0.5,
                ),
                alpha=0.35,
                facecolor=r["color"],
                edgecolor=r["color"],
            )
        )
        end = r["pos"] + r["axis"] * axis_len
        ax.plot(
            [r["pos"][0], end[0]],
            [r["pos"][1], end[1]],
            [r["pos"][2], end[2]],
            color=r["color"],
            lw=1.5,
            alpha=0.7,
        )
        ax.text(
            r["pos"][0],
            r["pos"][1],
            r["pos"][2] + fr * 1.5,
            r["label"],
            color=r["color"],
            fontsize=11,
            fontweight="bold",
            ha="center",
        )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    return span


def write_cameras_3d(
    model1: "CameraModel",
    model2: "CameraModel",
    board_world,
    R_stereo,
    T_stereo,
    output_path,
) -> None:
    """World origin triad + datum board plane + both camera frusta/axes (PNG)."""
    try:
        bw = np.asarray(board_world, np.float64).reshape(-1, 3)
        recs = _camera_records([model1, model2], ["Cam1", "Cam2"], bw)
        ang = float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(np.asarray(R_stereo, np.float64)) - 1) / 2, -1, 1)
                )
            )
        )
        Tn = float(np.linalg.norm(np.asarray(T_stereo, np.float64)))

        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw_cameras_scene(
            ax, recs, bw, board_alpha=0.45, scatter_alpha=0.7, board_label="datum board"
        )
        ax.set_title(
            f"Cameras relative to board — baseline |T|={Tn:.1f} mm, "
            f"stereo angle={ang:.2f} deg",
            fontsize=12,
        )
        ax.legend(fontsize=9, loc="upper left")
        _save(fig, output_path, dpi=150)
    except Exception:
        logger.warning(f"cameras-3d figure failed: {traceback.format_exc()}")


def write_cameras_planes_3d(
    models_by_cam, board_world, output_path, labels=None
) -> None:
    """All cameras' frusta + optical axes around the shared released board (N-camera, PNG).

    The joint analogue of ``write_cameras_3d`` (which is 2-camera). Built on the same
    ``_camera_records`` list-loop + ``_frustum_faces``; the board is drawn once as a
    translucent world-frame plane. Shows the rig geometry — every camera pointing at the
    one released board — which is request "orientation of the boards in space" for a joint solve.
    """
    try:
        cams = list(models_by_cam.keys())
        labels = labels or {c: f"Cam{c}" for c in cams}
        bw = np.asarray(board_world, np.float64).reshape(-1, 3)
        recs = _camera_records(
            [models_by_cam[c] for c in cams], [labels[c] for c in cams], bw
        )
        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw_cameras_scene(
            ax, recs, bw, board_alpha=0.4, scatter_alpha=0.6, board_label="released board"
        )
        ax.set_title(f"Cameras relative to board — {len(cams)} cameras", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        _save(fig, output_path, dpi=150)
    except Exception:
        logger.warning(f"cameras-planes-3d figure failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 8. Dewarp through the model onto the world plane (agreement proof)
#
# The image is pushed BACKWARDS through the calibrated model (intrinsics + distortion
# + board pose) onto the world Z=0 plane: a correct model rectifies the board to a
# regular mm lattice. One camera -> a single dewarped board; two -> a red/cyan
# anaglyph (sharp = the models agree); many -> a per-camera panel each plus a
# back-projected-dot scatter (markers coincide = agreement). NOT the detection overlay.
# ---------------------------------------------------------------------------


def _world_extent(board_world, pad: float = 5.0):
    """(x_min, x_max, y_min, y_max) of the board's XY extent, padded by ``pad`` mm."""
    bw = np.asarray(board_world, np.float64).reshape(-1, 3)
    return (
        float(bw[:, 0].min()) - pad,
        float(bw[:, 0].max()) + pad,
        float(bw[:, 1].min()) - pad,
        float(bw[:, 1].max()) + pad,
    )


# Absolute OOM backstop on the dewarp raster (only binds for pathologically large
# sensors); the normal ceiling is the source frame's own pixel count — see below.
_DEWARP_ABS_MAX_PIXELS = 40_000_000
# projectPoints is evaluated on at most this many cells; the smooth world->pixel map is
# then bilinear-resized to the full raster, so a fine dewarp stays cheap to build.
_DEWARP_MAP_MAX_PIXELS = 1_000_000


def _mm_per_source_pixel(model: "CameraModel", x_min, x_max, y_min, y_max) -> float:
    """Approx. world-plane mm spanned by one source pixel at the board.

    Projects the four world-plane corners of the dewarp extent through the model and
    divides the world diagonal by the mean image-space diagonal. Sampling the dewarp at
    this resolution matches the source pixel pitch — coarser blurs away the agreement
    (red/cyan) signal the figure exists to show; finer only wastes memory.
    """
    corners = np.array(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        np.float64,
    )
    px, _ = cv2.projectPoints(
        corners,
        model.rvec,
        np.asarray(model.t, np.float64).reshape(3),
        np.asarray(model.K, np.float64),
        np.asarray(model.dist, np.float64),
    )
    px = px.reshape(-1, 2)
    world_diag = math.hypot(x_max - x_min, y_max - y_min)
    px_diag = 0.5 * (
        float(np.linalg.norm(px[2] - px[0])) + float(np.linalg.norm(px[3] - px[1]))
    )
    return 0.1 if px_diag < 1e-6 else world_diag / px_diag


def _resolve_mm_per_px(
    model: "CameraModel", x_min, x_max, y_min, y_max, mm_per_px: float | None
) -> float:
    """Final dewarp resolution: an explicit value as-is, else magnification-matched.

    None -> the source pixel pitch at the board (`_mm_per_source_pixel`), clamped to
    [1e-3, 1.0] mm/px and coarsened only if the raster would exceed the source frame's own
    pixel count (you never need a dewarp finer than the image it came from) or the absolute
    OOM backstop. Resolving once and passing the concrete value lets the 2-camera overlay
    sample both cameras onto an identical raster (required for the red/cyan stack).
    """
    if mm_per_px is not None:
        return float(mm_per_px)
    est = float(
        np.clip(_mm_per_source_pixel(model, x_min, x_max, y_min, y_max), 1e-3, 1.0)
    )
    span_x, span_y = x_max - x_min, y_max - y_min
    w, h = model.image_size
    ceiling = min(int(w) * int(h), _DEWARP_ABS_MAX_PIXELS)
    if (span_x / est) * (span_y / est) > ceiling:
        est = math.sqrt(span_x * span_y / ceiling)
    return est


def _dewarp_image_to_world(
    model: "CameraModel",
    img,
    x_min,
    x_max,
    y_min,
    y_max,
    mm_per_px: float | None = None,
) -> np.ndarray:
    """Remap one image to the world Z=0 plane over [x_min,x_max]x[y_min,y_max] (uint8).

    ``mm_per_px`` is the output raster resolution; None matches it to the actual
    magnification (see ``_resolve_mm_per_px``) instead of a fixed 0.1 mm/px that
    under-samples high-magnification rigs and blurs the agreement signal.
    """
    mm_per_px = _resolve_mm_per_px(model, x_min, x_max, y_min, y_max, mm_per_px)
    nx = max(int((x_max - x_min) / mm_per_px), 32)
    ny = max(int((y_max - y_min) / mm_per_px), 32)
    g = _to_uint8_gray(img).astype(np.float64)
    # Build the world->pixel map on a coarse grid (the projective+distortion field is
    # smooth) and bilinear-resize it to the full output raster. projectPoints cost stays
    # bounded by _DEWARP_MAP_MAX_PIXELS while the dewarp keeps its fine, magnification-
    # matched resolution (remap still samples the full-res source, so dots stay sharp).
    scale = min(1.0, math.sqrt(_DEWARP_MAP_MAX_PIXELS / float(nx * ny)))
    cnx, cny = max(int(nx * scale), 2), max(int(ny * scale), 2)
    Xc, Yc = np.meshgrid(np.linspace(x_min, x_max, cnx), np.linspace(y_min, y_max, cny))
    world = np.column_stack([Xc.ravel(), Yc.ravel(), np.zeros(Xc.size)]).astype(
        np.float64
    )
    proj, _ = cv2.projectPoints(
        world,
        model.rvec,
        np.asarray(model.t, np.float64).reshape(3),
        np.asarray(model.K, np.float64),
        np.asarray(model.dist, np.float64),
    )
    proj = proj.reshape(-1, 2)
    mxc = proj[:, 0].reshape(cny, cnx).astype(np.float32)
    myc = proj[:, 1].reshape(cny, cnx).astype(np.float32)
    mx = cv2.resize(mxc, (nx, ny), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(myc, (nx, ny), interpolation=cv2.INTER_LINEAR)
    dw = cv2.remap(
        g, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    valid = (mx >= 0) & (mx < g.shape[1]) & (my >= 0) & (my < g.shape[0])
    dw[~valid] = 0
    lo, hi = float(dw.min()), float(dw.max())
    out = (
        ((dw - lo) / (hi - lo) * 255).astype(np.uint8)
        if hi > lo
        else np.zeros_like(dw, np.uint8)
    )
    out[~valid] = 0
    return out


def _dewarp_axes(ax, board_world, spacing, x_min, x_max, y_min, y_max) -> None:
    """Shared world-plane styling: origin cross, mm gridlines, board-dot squares."""
    bw = np.asarray(board_world, np.float64).reshape(-1, 3)
    ax.axhline(0, color="lime", lw=1, alpha=0.6)
    ax.axvline(0, color="lime", lw=1, alpha=0.6)
    ax.scatter(0, 0, s=250, c="lime", marker="+", linewidths=3, zorder=20)
    sp = float(spacing)
    ax.set_xticks(np.arange(np.ceil(x_min / sp) * sp, x_max, sp))
    ax.set_yticks(np.arange(np.ceil(y_min / sp) * sp, y_max, sp))
    ax.grid(True, color="white", alpha=0.15, lw=0.5)
    ax.scatter(
        bw[:, 0],
        bw[:, 1],
        s=22,
        facecolors="none",
        edgecolors="white",
        linewidths=0.6,
        marker="s",
        alpha=0.7,
        label="board dots",
    )
    ax.set_xlabel("X world (mm)")
    ax.set_ylabel("Y world (mm)")


def write_dewarp_overlay(
    model1: "CameraModel",
    model2: "CameraModel",
    img1,
    img2,
    board_world,
    spacing,
    output_path,
    title=None,
    mm_per_px=None,
) -> None:
    """Remap both images to the world Z=0 plane, overlay as red(cam1)/cyan(cam2)."""
    try:
        x_min, x_max, y_min, y_max = _world_extent(board_world)
        # Resolve once so both cameras dewarp onto the same raster (the red/cyan stack
        # below requires identical shapes).
        res = _resolve_mm_per_px(model1, x_min, x_max, y_min, y_max, mm_per_px)
        # A missing datum frame reaches here as None/0-D; dewarp what is available and
        # leave the other channel black rather than crashing the whole figure.
        if not _is_2d_image(img1):
            s1 = None if img1 is None else np.asarray(img1).shape
            logger.warning(f"dewarp overlay: cam1 image unavailable (shape={s1})")
        if not _is_2d_image(img2):
            s2 = None if img2 is None else np.asarray(img2).shape
            logger.warning(f"dewarp overlay: cam2 image unavailable (shape={s2})")
        r = (
            _dewarp_image_to_world(model1, img1, x_min, x_max, y_min, y_max, res)
            if _is_2d_image(img1)
            else None
        )
        c = (
            _dewarp_image_to_world(model2, img2, x_min, x_max, y_min, y_max, res)
            if _is_2d_image(img2)
            else None
        )
        shape = (r if r is not None else c)
        if shape is None:
            raise ValueError("both cam images unavailable")
        if r is None:
            r = np.zeros_like(shape)
        if c is None:
            c = np.zeros_like(shape)
        overlay = np.stack([r, c, c], axis=-1)
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin="lower")
        _dewarp_axes(ax, board_world, spacing, x_min, x_max, y_min, y_max)
        ax.set_title(
            title or "Dewarp overlay (Cam1=red, Cam2=cyan; sharp = agreement)",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, output_path, dpi=160)
    except Exception:
        logger.warning(f"dewarp overlay figure failed: {traceback.format_exc()}")


def _dewarp_panel(
    ax, model, img, board_world, spacing, x_min, x_max, y_min, y_max, cam_num, mm_per_px
) -> None:
    """One side-by-side panel: a camera's image dewarped onto the world (mm) plane.

    Draws a labelled placeholder instead of crashing when the datum image is missing
    (``None`` / 0-D), so the other camera's panel still renders.
    """
    if not _is_2d_image(img):
        shp = None if img is None else np.asarray(img).shape
        logger.warning(f"dewarp pair: cam{cam_num} image unavailable (shape={shp})")
        ax.text(
            0.5,
            0.5,
            f"Cam{cam_num}: image unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )
        ax.set_xlabel("X world (mm)")
        ax.set_title(f"Cam{cam_num} dewarped → world (mm)", fontsize=11)
        return
    dw = _dewarp_image_to_world(model, img, x_min, x_max, y_min, y_max, mm_per_px)
    ax.imshow(dw, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="gray")
    _dewarp_axes(ax, board_world, spacing, x_min, x_max, y_min, y_max)
    ax.set_title(f"Cam{cam_num} dewarped → world (mm)", fontsize=11)


def write_dewarp_pair(
    model1: "CameraModel",
    model2: "CameraModel",
    img1,
    img2,
    board_world,
    spacing,
    output_path,
    cam1_num: int = 1,
    cam2_num: int = 2,
    title=None,
    mm_per_px=None,
) -> None:
    """Side-by-side: each camera's board image dewarped onto the world (mm) plane.

    A correct model rectifies the board to a regular mm grid (dots land on the
    ``_dewarp_axes`` gridlines); a side-by-side pair lets each camera be judged
    independently. Each panel resolves its own resolution (separate axes, so the rasters
    need not match — unlike the red/cyan overlay).
    """
    try:
        x_min, x_max, y_min, y_max = _world_extent(board_world)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        if title:
            fig.suptitle(title, fontsize=13)
        _dewarp_panel(
            ax1, model1, img1, board_world, spacing,
            x_min, x_max, y_min, y_max, cam1_num, mm_per_px,
        )
        _dewarp_panel(
            ax2, model2, img2, board_world, spacing,
            x_min, x_max, y_min, y_max, cam2_num, mm_per_px,
        )
        _save(fig, output_path, dpi=160)
    except Exception:
        logger.warning(f"dewarp pair figure failed: {traceback.format_exc()}")


def _write_dewarp_single(
    dewarp_fn, model, img, board_world, spacing, output_path,
    title, default_title, mm_per_px, log_label,
) -> None:
    """Shared body for the pinhole/polynomial single-image dewarp figures — they differ only in
    the image-remap function (``dewarp_fn``), the default title, and the log label."""
    try:
        x_min, x_max, y_min, y_max = _world_extent(board_world)
        dw = dewarp_fn(model, img, x_min, x_max, y_min, y_max, mm_per_px)
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.imshow(dw, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="gray")
        _dewarp_axes(ax, board_world, spacing, x_min, x_max, y_min, y_max)
        ax.set_title(title or default_title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, output_path, dpi=160)
    except Exception:
        logger.warning(f"{log_label} figure failed: {traceback.format_exc()}")


def write_dewarp_single(
    model: "CameraModel",
    img,
    board_world,
    spacing,
    output_path,
    title=None,
    mm_per_px=None,
) -> None:
    """Remap one image to the world Z=0 plane; a correct model rectifies the board to a regular grid."""
    _write_dewarp_single(
        _dewarp_image_to_world, model, img, board_world, spacing, output_path,
        title, "Dewarped board (model -> world plane; dots on the grid = good model)",
        mm_per_px, "dewarp single",
    )


def _backproject_to_world_plane(model: "CameraModel", pixels) -> np.ndarray:
    """Detected pixels -> world XY on the Z=0 plane through ``model`` (undistort + ray/plane).

    Same world-ray convention as ``joint._pixel_rays_world``: direction = R^T d_cam, origin =
    -R^T t; the ray is intersected with Z=0 so each camera's detection lands in world mm.
    """
    K = np.asarray(model.K, np.float64)
    dist = np.asarray(model.dist, np.float64)
    R = np.asarray(model.R, np.float64)
    t = np.asarray(model.t, np.float64).reshape(3)
    und = cv2.undistortPoints(
        np.asarray(pixels, np.float64).reshape(-1, 1, 2), K, dist
    ).reshape(-1, 2)
    cam = np.column_stack([und, np.ones(len(und))])  # normalised camera rays
    o = -R.T @ t  # camera centre in world
    d = cam @ R  # ray directions in world
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    s = -o[2] / np.where(np.abs(d[:, 2]) < 1e-6, np.nan, d[:, 2])  # intersect Z=0
    world = o[None, :2] + s[:, None] * d[:, :2]
    # A ray nearly parallel to the board plane has no well-defined intersection; drop it (NaN)
    # rather than placing it near the camera centre. matplotlib omits NaN points from the scatter.
    return world


def _write_dewarp_dots(
    backproject_fn, models_by_cam, detect_pixels_by_cam, board_world, output_path,
    labels, title, default_title, board_label, log_label,
) -> None:
    """Shared body for the pinhole/polynomial dewarp-dots figures — they differ only in how each
    camera's pixels map to world XY (``backproject_fn``), the released-board legend label, the
    default title, and the log label."""
    try:
        bw = np.asarray(board_world, np.float64).reshape(-1, 3)
        cams = list(models_by_cam.keys())
        labels = labels or {c: f"Cam{c}" for c in cams}
        fig, ax = plt.subplots(figsize=(13, 11))
        ax.scatter(
            bw[:, 0],
            bw[:, 1],
            s=60,
            facecolors="none",
            edgecolors="0.6",
            linewidths=0.8,
            marker="s",
            label=board_label,
        )
        for i, c in enumerate(cams):
            px = detect_pixels_by_cam.get(c)
            if px is None or not len(px):
                continue
            w = backproject_fn(models_by_cam[c], px)
            ax.scatter(
                w[:, 0],
                w[:, 1],
                s=18,
                color=CAM_COLORS[i % len(CAM_COLORS)],
                alpha=0.7,
                label=labels[c],
            )
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X world (mm)")
        ax.set_ylabel("Y world (mm)")
        ax.set_title(title or default_title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, output_path, dpi=150)
    except Exception:
        logger.warning(f"{log_label} figure failed: {traceback.format_exc()}")


def write_dewarp_dots(
    models_by_cam,
    detect_pixels_by_cam,
    board_world,
    spacing,
    output_path,
    labels=None,
    title=None,
) -> None:
    """Each camera's detected dots back-projected to the world plane (coincident markers = agree)."""
    _write_dewarp_dots(
        _backproject_to_world_plane, models_by_cam, detect_pixels_by_cam, board_world,
        output_path, labels, title,
        "Detected dots dewarped to world (markers coincide = models agree)",
        "released board", "dewarp dots",
    )


# ---------------------------------------------------------------------------
# Polynomial dewarp — visualization-only inverse of the pixel->world cubic
# ---------------------------------------------------------------------------
# A planar PolynomialModel only maps pixel->world (``back_project_to_plane``); the pinhole dewarp
# above needs the opposite (world->pixel, via cv2.projectPoints with R,t,K,dist) which the
# polynomial has no parameters for. To rectify the image we invert the cubic NUMERICALLY, for the
# figure only: sample a coarse pixel lattice, push it through the polynomial to world mm, then
# interpolate the pixel coordinates back onto the regular world raster. No calibration-math change.


def _dewarp_image_to_world_poly(
    model: "PolynomialModel",
    img,
    x_min,
    x_max,
    y_min,
    y_max,
    mm_per_px: float | None = None,
) -> np.ndarray:
    """Remap one image to the world Z=0 plane through a planar polynomial (uint8).

    The world->pixel map is built by scattered inversion of ``back_project_to_plane`` (the
    polynomial is not analytically invertible). World-raster cells outside the sampled hull come
    back NaN from ``griddata`` and are masked to the (black) border.
    """
    from scipy.interpolate import griddata

    w, h = int(model.image_size[0]), int(model.image_size[1])
    span_x, span_y = float(x_max - x_min), float(y_max - y_min)

    # Coarse source-pixel lattice -> world mm (the map is smooth, so ~220x220 samples suffice).
    # The same lattice gives the true average magnification (world diagonal / image diagonal) used
    # to size the output raster — a geometric measure, not the sensor area-to-pixel-count ratio.
    npx = 220
    Px, Py = np.meshgrid(np.linspace(0, w - 1, npx), np.linspace(0, h - 1, npx))
    src = np.column_stack([Px.ravel(), Py.ravel()])
    world = model.back_project_to_plane(src)[:, :2]

    if mm_per_px is None:
        world_diag = math.hypot(
            float(world[:, 0].max() - world[:, 0].min()),
            float(world[:, 1].max() - world[:, 1].min()),
        )
        img_diag = math.hypot(w - 1, h - 1)
        mm_per_px = float(np.clip(world_diag / img_diag if img_diag else 0.1, 1e-3, 1.0))
        ceiling = min(w * h, _DEWARP_ABS_MAX_PIXELS)
        if (span_x / mm_per_px) * (span_y / mm_per_px) > ceiling:
            mm_per_px = math.sqrt(span_x * span_y / ceiling)
    nx = max(int(span_x / mm_per_px), 32)
    ny = max(int(span_y / mm_per_px), 32)

    Xc, Yc = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    target = np.column_stack([Xc.ravel(), Yc.ravel()])
    mx = griddata(world, src[:, 0], target, method="linear").reshape(ny, nx)
    my = griddata(world, src[:, 1], target, method="linear").reshape(ny, nx)

    g = _to_uint8_gray(img).astype(np.float64)
    valid = np.isfinite(mx) & np.isfinite(my)
    mxs = np.where(valid, mx, -1.0).astype(np.float32)
    mys = np.where(valid, my, -1.0).astype(np.float32)
    dw = cv2.remap(
        g, mxs, mys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    inb = valid & (mx >= 0) & (mx < g.shape[1]) & (my >= 0) & (my < g.shape[0])
    if inb.any():
        lo, hi = float(dw[inb].min()), float(dw[inb].max())
    else:
        lo, hi = 0.0, 0.0
    out = (
        ((dw - lo) / (hi - lo) * 255).astype(np.uint8)
        if hi > lo
        else np.zeros_like(dw, np.uint8)
    )
    out[~inb] = 0
    return out


def write_dewarp_single_poly(
    model: "PolynomialModel",
    img,
    board_world,
    spacing,
    output_path,
    title=None,
    mm_per_px=None,
) -> None:
    """Remap one image to the world plane through a planar polynomial; a correct fit rectifies the
    board to a regular grid (the polynomial analogue of ``write_dewarp_single``)."""
    _write_dewarp_single(
        _dewarp_image_to_world_poly, model, img, board_world, spacing, output_path,
        title, "Dewarped board (polynomial -> world plane; dots on the grid = good fit)",
        mm_per_px, "dewarp single (poly)",
    )


def write_dewarp_dots_poly(
    models_by_cam,
    detect_pixels_by_cam,
    board_world,
    spacing,
    output_path,
    labels=None,
    title=None,
) -> None:
    """Each camera's detected dots back-projected to the world plane via its polynomial (coincident
    markers = cameras agree). The polynomial maps pixel->world directly — no ray/plane cast."""

    def _poly_backproject(model, px):
        return model.back_project_to_plane(np.asarray(px, np.float64))[:, :2]

    _write_dewarp_dots(
        _poly_backproject, models_by_cam, detect_pixels_by_cam, board_world,
        output_path, labels, title,
        "Detected dots back-projected to world (coincident = cameras agree)",
        "grid target", "dewarp dots (poly)",
    )


# ---------------------------------------------------------------------------
# Orchestrators — called by the pipeline at fit time
# ---------------------------------------------------------------------------


def write_mono_figures(
    figure_dir,
    *,
    images,
    detections,
    used,
    K,
    dist,
    rvecs,
    tvecs,
    per_view,
    rms,
    cam,
    wf,
    spacing,
    board_type,
    datum_index,
    board_meta=None,
    prefix="",
    world_pts=None,
) -> None:
    """All single-camera proof figures. ``used`` = list of (pose_index, DetectionResult).

    The boards-in-physical-space planes figure and the dewarped-board figure are drawn only
    for a true mono run (``prefix`` empty); stereo per-camera calls skip them (the stereo run
    gets one cameras-relative-to-board figure + the red/cyan anaglyph instead). ``world_pts``
    are the datum view's detected dots in world mm (the same correspondence passed to
    ``fit_pose``); when given, the dewarp remaps the datum image onto that world plane.
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    pose_indices = [i for i, _ in used]
    objs = [d.board_local_points for _, d in used]
    imgs = [d.image_points for _, d in used]
    datum_pos = pose_indices.index(datum_index) if datum_index in pose_indices else 0

    for i, d in used:
        write_detection_figure(
            images[i],
            d,
            figd / f"{prefix}detection_{i:02d}.png",
            title=f"{prefix}pose {i}",
        )
    write_world_frame_figure(
        images[datum_index],
        detections[datum_index],
        wf,
        spacing,
        figd / f"{prefix}world_frame.png",
        title=f"{prefix}world frame",
    )
    write_reprojection_figure(
        per_view,
        rms,
        figd / f"{prefix}reprojection.png",
        pose_indices=pose_indices,
        title=f"{prefix}reprojection",
    )
    if not prefix:
        write_boards_planes_3d(
            cam,
            objs,
            rvecs,
            tvecs,
            pose_indices,
            figd / "boards_3d.png",
            datum_pos=datum_pos,
        )
        if world_pts is not None and images is not None:
            write_dewarp_single(
                cam,
                images[datum_index],
                world_pts,
                spacing,
                figd / "dewarp.png",
                title="dewarped board",
            )


def write_stepped_figures(
    figure_dir,
    *,
    images,
    used_detections,
    used_pose_indices,
    pose_obj_views,
    pose_img_views,
    rvecs,
    tvecs,
    per_view,
    rms,
    cam,
    wf,
    spacing,
    datum_index,
    datum_detection,
    pose_levels=None,
    prefix="",
) -> None:
    """All single-camera proof figures for a STEPPED (dual-level) fit.

    The generic ``write_mono_figures`` reprojects ``detection.board_local_points`` (a
    board-canonical neutral frame); a stepped fit instead runs on assembled, oriented,
    absolute-Z object points (``pose_obj_views``) whose ``rvecs``/``tvecs`` only match
    THOSE points. So the reprojection + boards-3D figures must use the fitted views,
    not the raw detections — that is the only reason this exists separately. The
    detection-overlay + world-frame figures still take the raw datum detection (they
    draw on the image, not the fit).

    ``used_pose_indices`` / ``used_detections`` / ``pose_obj_views`` / ``pose_img_views``
    / ``rvecs`` / ``tvecs`` / ``per_view`` are all position-aligned (datum first). Each
    sub-figure swallows its own errors, so a figure failure never aborts the fit.
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    datum_pos = (
        used_pose_indices.index(datum_index) if datum_index in used_pose_indices else 0
    )

    if images is not None:
        for pose_idx, det in zip(used_pose_indices, used_detections):
            # pose_levels labels level 'a's face per pose (position-aligned to the original
            # detections); pass it so the network is split peak(blue)/trough(red).
            level_a_face = (
                pose_levels[pose_idx]
                if pose_levels is not None and 0 <= pose_idx < len(pose_levels)
                else None
            )
            write_detection_figure(
                images[pose_idx],
                det,
                figd / f"{prefix}detection_{pose_idx:02d}.png",
                title=f"{prefix}pose {pose_idx}",
                level_a_face=level_a_face,
            )
        write_world_frame_figure(
            images[datum_index],
            datum_detection,
            wf,
            spacing,
            figd / f"{prefix}world_frame.png",
            title=f"{prefix}world frame",
        )
    write_reprojection_figure(
        per_view=per_view,
        rms=rms,
        output_path=figd / f"{prefix}reprojection.png",
        pose_indices=used_pose_indices,
        title=f"{prefix}reprojection",
    )
    if not prefix:
        write_boards_planes_3d(
            cam,
            pose_obj_views,
            rvecs,
            tvecs,
            used_pose_indices,
            figd / "boards_3d.png",
            datum_pos=datum_pos,
        )


def write_polynomial_figures(
    figure_dir,
    *,
    image,
    detection: "DetectionResult",
    world_pts,
    model,
    wf: "WorldFrame",
    spacing,
    prefix="",
) -> None:
    """Single-plane polynomial proof figures: detection, world frame, and the fit.

    The fit figure plots, in the resolved world frame, the detected dots' target
    world-mm positions, the polynomial's evaluation at the same pixels, the residual
    quiver, and a residual scatter with the per-axis RMS — the least-squares analogue
    of the pinhole reprojection figure.
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    write_detection_figure(
        image, detection, figd / f"{prefix}detection_datum.png", title=f"{prefix}datum"
    )
    write_world_frame_figure(
        image,
        detection,
        wf,
        spacing,
        figd / f"{prefix}world_frame.png",
        title=f"{prefix}world frame",
    )
    write_polynomial_fit_figure(
        detection,
        world_pts,
        model,
        figd / f"{prefix}polynomial_fit.png",
        title=f"{prefix}polynomial fit",
    )
    # Dewarped board (one board for a single-camera fit), matching the pinhole mono dewarp.
    write_dewarp_single_poly(
        model,
        image,
        world_pts,
        spacing,
        figd / f"{prefix}dewarp.png",
        title=f"{prefix}dewarped board",
    )


def write_polynomial_fit_figure(
    detection: "DetectionResult", world_pts, model, output_path, title=None
) -> None:
    """World-mm target vs polynomial evaluation: positions + quiver + residual scatter."""
    try:
        target = np.asarray(world_pts, np.float64).reshape(-1, 3)[:, :2]
        px = np.asarray(detection.image_points, np.float64).reshape(-1, 2)
        evald = model.back_project_to_plane(px)[:, :2]
        resid = evald - target
        rms_x = float(model.rms_x_mm)
        rms_y = float(model.rms_y_mm)
        rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1)))) if len(resid) else 0.0

        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(
            f"{title or 'Polynomial fit'} — {len(target)} dots, "
            f"RMS_x={rms_x:.4f} mm, RMS_y={rms_y:.4f} mm",
            fontsize=13,
        )
        gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1.1, 1.0])

        # Left: world positions + residual quiver (exaggerated so structure is visible).
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(
            target[:, 0],
            target[:, 1],
            s=14,
            facecolors="none",
            edgecolors="steelblue",
            linewidths=0.7,
            label="grid target",
        )
        span = float(max(np.ptp(target[:, 0]), np.ptp(target[:, 1]), 1.0))
        rmax = float(np.max(np.hypot(resid[:, 0], resid[:, 1]))) if len(resid) else 0.0
        scale = (0.08 * span / rmax) if rmax > 1e-12 else 1.0
        ax.quiver(
            target[:, 0],
            target[:, 1],
            resid[:, 0] * scale,
            resid[:, 1] * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="crimson",
            width=0.004,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"world frame — residual quiver (x{scale:.0f})", fontsize=10)
        ax.legend(fontsize=8, loc="best")

        # Right: residual (dx,dy) scatter with RMS circle.
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(resid[:, 0], resid[:, 1], s=8, alpha=0.6, color="crimson")
        ax2.add_patch(
            plt.Circle(
                (0, 0),
                rms,
                fill=False,
                color="red",
                ls="--",
                lw=1.5,
                label=f"RMS={rms:.4f} mm",
            )
        )
        mr = max(float(np.abs(resid).max()) * 1.15, rms * 1.3) if len(resid) else 1e-3
        ax2.set_xlim(-mr, mr)
        ax2.set_ylim(-mr, mr)
        ax2.set_aspect("equal")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axvline(0, color="gray", lw=0.5)
        ax2.set_xlabel("residual X (mm)")
        ax2.set_ylabel("residual Y (mm)")
        ax2.set_title("fit residuals", fontsize=10)
        ax2.legend(fontsize=8, loc="best")
        _save(fig, output_path)
    except Exception:
        logger.warning(f"polynomial fit figure failed: {traceback.format_exc()}")


def write_scale_factor_figure(
    figure_dir,
    *,
    image: np.ndarray,
    origin_px,
    col_sign: int,
    row_sign: int,
    swap_axes: bool,
    mm_per_pixel: float,
    dt: float,
    prefix: str = "",
    origin_mm=(0.0, 0.0),
) -> None:
    """Proof figure for a scale-factor model: frame + origin + +X/+Y arrows + scale.

    The arrows point along the chosen world axes in PIXEL space, so the user can
    eyeball that the origin and directions match what they picked. ``col_sign`` is
    the +X sign, ``row_sign`` the +Y sign, ``swap_axes`` selects which pixel delta
    feeds X — the same convention the model uses.
    """
    try:
        gray = _to_uint8_gray(image)
        h, w = gray.shape[:2]
        ox, oy = float(origin_px[0]), float(origin_px[1])
        # Pixel-space direction of each world axis (see ScaleFactorModel algebra).
        if not swap_axes:
            x_dir = (col_sign, 0)  # +X along the column axis
            y_dir = (0, row_sign)  # +Y along the row axis
        else:
            x_dir = (0, col_sign)  # +X along the row axis
            y_dir = (row_sign, 0)  # +Y along the column axis
        length = 0.12 * max(w, h)
        px_per_mm = (1.0 / mm_per_pixel) if mm_per_pixel else float("nan")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(gray, cmap="gray", origin="upper")
        ax.plot(ox, oy, "+", color="yellow", markersize=18, markeredgewidth=2.5)
        ax.annotate(
            "",
            xy=(ox + x_dir[0] * length, oy + x_dir[1] * length),
            xytext=(ox, oy),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5),
        )
        ax.annotate(
            "",
            xy=(ox + y_dir[0] * length, oy + y_dir[1] * length),
            xytext=(ox, oy),
            arrowprops=dict(arrowstyle="-|>", color="deepskyblue", lw=2.5),
        )
        ax.text(
            ox + x_dir[0] * length,
            oy + x_dir[1] * length,
            "  +X",
            color="red",
            fontsize=12,
            fontweight="bold",
            va="center",
        )
        ax.text(
            ox + y_dir[0] * length,
            oy + y_dir[1] * length,
            "  +Y",
            color="deepskyblue",
            fontsize=12,
            fontweight="bold",
            va="center",
        )
        om_x, om_y = float(origin_mm[0]), float(origin_mm[1])
        om_txt = f" = ({om_x:g}, {om_y:g}) mm" if (om_x or om_y) else ""
        ax.set_title(
            f"Scale-factor frame — origin ({ox:.1f}, {oy:.1f}) px{om_txt}, "
            f"{px_per_mm:.4f} px/mm ({mm_per_pixel:.5f} mm/px), dt={dt:g} s",
            fontsize=11,
        )
        ax.set_xlabel("pixel x (image-down)")
        ax.set_ylabel("pixel y (image-down)")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        _save(fig, Path(figure_dir) / f"{prefix}scale_factor.png")
    except Exception:
        logger.warning(f"scale-factor figure failed: {traceback.format_exc()}")


def write_stereo_figures(
    figure_dir,
    *,
    model1,
    model2,
    R_stereo,
    T_stereo,
    img1,
    img2,
    datum_board_world,
    spacing,
    cam1_num: int = 1,
    cam2_num: int = 2,
) -> None:
    """Stereo-only proof figures: cameras vs the datum board + side-by-side dewarped boards."""
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    write_cameras_3d(
        model1, model2, datum_board_world, R_stereo, T_stereo, figd / "cameras_3d.png"
    )
    write_dewarp_pair(
        model1,
        model2,
        img1,
        img2,
        datum_board_world,
        spacing,
        figd / "dewarp_pair.png",
        cam1_num=cam1_num,
        cam2_num=cam2_num,
    )


def write_joint_figures(
    figure_dir,
    *,
    result,
    detections_by_cam,
    global_index,
    spacing,
    board_type,
    datum_view,
    image_loader=None,
) -> None:
    """All multi-camera joint proof figures, written beside the joint record.

    The joint analogue of ``write_mono_figures``, drawn from the in-memory ``JointResult`` (the saved
    record does not keep per-view poses). Per camera: a detection overlay per view and one
    reprojection figure (world board points pushed through each view's pose). Once: the N-camera
    cameras-relative-to-board scene (``boards_3d.png``) and the dewarp agreement proof — a single
    dewarped board for 1 camera, a red/cyan anaglyph for 2, per-camera dewarped panels + a
    back-projected-dot scatter for >2.

    ``global_index[(cam,view)]`` rows correspond one-for-one with that detection's image points;
    ``result.board[(gx,gy)]`` gives the released world point. ``image_loader(cam, view) -> ndarray``
    supplies raw images for the image-based figures (detection overlays + dewarp); when it is None
    those are skipped and only the geometry figures (reprojection, 3D scene) are written. Every
    sub-figure swallows its own errors, so a figure failure never aborts the calibration.
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    cams = list(result.cameras)
    board_world = np.array(list(result.board.values()), dtype=np.float64).reshape(-1, 3)

    # Per-camera reprojection (world board points through each view's pose) + detection overlays.
    for cam in cams:
        model = result.models[cam]
        views = sorted(v for (c, v) in result.view_poses if c == cam)
        objs, imgs, rvecs, tvecs, per_view, used_views = [], [], [], [], [], []
        for v in views:
            key = (cam, v)
            det = detections_by_cam[cam][v]
            gi = global_index.get(key)
            if gi is None or not det.success:
                continue
            gi = np.asarray(gi, np.int64).reshape(-1, 2)
            ipts = np.asarray(det.image_points, np.float64).reshape(-1, 2)
            keep = [
                i for i, g in enumerate(gi) if (int(g[0]), int(g[1])) in result.board
            ]
            if not keep:
                continue
            world = np.array(
                [result.board[(int(gi[i, 0]), int(gi[i, 1]))] for i in keep], np.float64
            )
            ipts = ipts[keep]
            R, t = result.view_poses[key]
            rv, _ = cv2.Rodrigues(np.asarray(R, np.float64))
            tv = np.asarray(t, np.float64).reshape(3)
            proj, _ = cv2.projectPoints(
                world.reshape(-1, 1, 3), rv.reshape(3), tv, model.K, model.dist
            )
            res = ipts - proj.reshape(-1, 2)
            objs.append(world)
            imgs.append(ipts)
            rvecs.append(rv.reshape(3))
            tvecs.append(tv)
            per_view.append(float(np.sqrt(np.mean(np.sum(res**2, axis=1)))))
            used_views.append(v)
            if image_loader is not None:
                img = image_loader(cam, v)
                if img is not None:
                    write_detection_figure(
                        img,
                        det,
                        figd / f"detection_cam{cam}_{v:02d}.png",
                        title=f"cam{cam} view {v}",
                    )
        if objs:
            write_reprojection_figure(
                per_view,
                float(result.per_camera_rms.get(cam, float("nan"))),
                figd / f"reprojection_cam{cam}.png",
                pose_indices=used_views,
                title=f"cam{cam} reprojection",
            )

    # Orientation of the rig in space: every camera around the one shared board.
    write_cameras_planes_3d(
        {c: result.models[c] for c in cams}, board_world, figd / "boards_3d.png"
    )

    # Dewarp agreement proof (image-based; needs the datum-view images).
    if image_loader is not None:
        datum_imgs = {c: image_loader(c, datum_view) for c in cams}
        if len(cams) == 1:
            c = cams[0]
            if datum_imgs[c] is not None:
                write_dewarp_single(
                    result.models[c],
                    datum_imgs[c],
                    board_world,
                    spacing,
                    figd / "dewarp.png",
                    title=f"Cam{c} dewarped board",
                )
        elif len(cams) == 2:
            a, b = cams
            if datum_imgs[a] is not None and datum_imgs[b] is not None:
                write_dewarp_overlay(
                    result.models[a],
                    result.models[b],
                    datum_imgs[a],
                    datum_imgs[b],
                    board_world,
                    spacing,
                    figd / "dewarp_overlay.png",
                    title=f"Dewarp overlay (Cam{a}=red, Cam{b}=cyan; sharp = agreement)",
                )
        else:
            for c in cams:
                if datum_imgs[c] is not None:
                    write_dewarp_single(
                        result.models[c],
                        datum_imgs[c],
                        board_world,
                        spacing,
                        figd / f"dewarp_cam{c}.png",
                        title=f"Cam{c} dewarped board",
                    )

    # For >2 cameras, the shared world-frame agreement scatter (no raw image needed). Only the
    # dots the solver actually used (global index present in the released board) are plotted —
    # the same filter the reprojection loop applies — so the figure shows the solved dots, not
    # any stray detections.
    if len(cams) >= 3:
        detect_px = {}
        for c in cams:
            dets = detections_by_cam[c]
            gi = global_index.get((c, datum_view))
            if gi is None or datum_view >= len(dets) or not dets[datum_view].success:
                continue
            gi = np.asarray(gi, np.int64).reshape(-1, 2)
            ipts = np.asarray(dets[datum_view].image_points, np.float64).reshape(-1, 2)
            keep = [
                i for i, g in enumerate(gi) if (int(g[0]), int(g[1])) in result.board
            ]
            if keep:
                detect_px[c] = ipts[keep]
        if len(detect_px) >= 2:
            write_dewarp_dots(
                {c: result.models[c] for c in cams},
                detect_px,
                board_world,
                spacing,
                figd / "dewarp_dots.png",
            )


def write_joint_polynomial_figures(
    figure_dir,
    *,
    models_by_cam,
    detections_by_cam,
    global_index,
    spacing,
    origin_mm,
    datum_view,
    image_loader=None,
) -> None:
    """All joint per-camera polynomial proof figures, written beside the per-camera records.

    The polynomial analogue of ``write_joint_figures``. Each camera fitted a single-plane cubic on
    its ``datum_view`` detection with world targets from the SHARED global index
    (``gi*spacing + origin`` — identical to ``run_joint_polynomial``), so all cameras already live in
    one world frame. Per camera: detection overlay and the polynomial-fit residual (the reprojection
    analogue). For >=2 cameras the single dewarp proof is one shared back-projected-dots agreement
    scatter (both cameras overlaid in one world frame); a lone camera instead gets its own dewarped
    board, since there is no second camera to scatter against. ``image_loader(cam, view) -> ndarray``
    supplies raw images for the image-based figures (detection + dewarp); None skips those. Every
    sub-figure swallows its own errors, so a figure failure never aborts the calibration.
    """
    figd = Path(figure_dir)
    figd.mkdir(parents=True, exist_ok=True)
    ox, oy = float(origin_mm[0]), float(origin_mm[1])
    cams = sorted(models_by_cam)
    detect_px: dict = {}
    world_all = []
    for cam in cams:
        gi = global_index.get((int(cam), int(datum_view)))
        dets = detections_by_cam.get(cam) or []
        if gi is None or datum_view >= len(dets) or not dets[datum_view].success:
            continue
        det = dets[datum_view]
        gi = np.asarray(gi, np.float64).reshape(-1, 2)
        ipts = np.asarray(det.image_points, np.float64).reshape(-1, 2)
        if len(gi) != len(ipts):
            continue
        world = np.column_stack(
            [gi[:, 0] * spacing + ox, gi[:, 1] * spacing + oy, np.zeros(len(gi))]
        )
        world_all.append(world)
        detect_px[cam] = ipts
        model = models_by_cam[cam]
        write_polynomial_fit_figure(
            det,
            world,
            model,
            figd / f"polynomial_fit_cam{cam}.png",
            title=f"cam{cam} polynomial fit",
        )
        if image_loader is not None:
            img = image_loader(cam, datum_view)
            if img is not None:
                write_detection_figure(
                    img,
                    det,
                    figd / f"detection_cam{cam}_{datum_view:02d}.png",
                    title=f"cam{cam} datum",
                )
                # A lone camera has no agreement scatter, so it still gets its own dewarped-board
                # proof. For >=2 cameras the shared back-projected-dots figure below is the single
                # unification proof (both cameras overlaid in one world frame); the redundant
                # per-camera dewarp panels are intentionally not drawn.
                if len(cams) == 1:
                    write_dewarp_single_poly(
                        model,
                        img,
                        world,
                        spacing,
                        figd / f"dewarp_cam{cam}.png",
                        title=f"Cam{cam} dewarped board",
                    )
    if len(detect_px) >= 2 and world_all:
        # The figure dir is not cleared between runs, so drop any per-camera dewarp panels left by
        # an earlier run — for >=2 cameras the shared scatter below is the only dewarp proof.
        for stale in figd.glob("dewarp_cam*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
        board_world = np.unique(np.round(np.vstack(world_all), 3), axis=0)
        write_dewarp_dots_poly(
            models_by_cam, detect_px, board_world, spacing, figd / "dewarp_dots.png"
        )


# ---------------------------------------------------------------------------
# Self-calibration diagnostics (red-cyan dewarp overlay + the 6 figures)
#
# Ported verbatim in content from the v1
# ``calibration/services/self_calibration_service.py`` (only the output path and
# the cam-label parameters changed). They take already-bridged ``PinholeCamera``
# objects (from ``self_cal.pinhole_from_model``) so figures.py does not import
# ``self_cal`` (which would be a cycle); the dewarp primitives come from the
# algorithm core, imported lazily.
# ---------------------------------------------------------------------------


def _percentile_u8(img: np.ndarray) -> np.ndarray:
    """Percentile contrast-stretch a sparse-particle dewarp to uint8 (out-of-bounds = 0)."""
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo = float(np.percentile(pos, 1))
    hi = float(np.percentile(pos, 99.5))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def write_self_cal_figures(
    result,
    cam1,
    cam2,
    images_cam1,
    images_cam2,
    *,
    world_bounds,
    mm_per_pixel,
    figure_dir,
    cam1_num: int = 1,
    cam2_num: int = 2,
) -> Path:
    """Write the six self-calibration diagnostic PNGs + ``correlation_planes.mat``.

    ``cam1``/``cam2`` are ``PinholeCamera`` objects. Figures land in ``figure_dir``
    (inside the calibration source folder), so the existing figure-serving endpoints
    surface them and they travel with the dataset. Each figure swallows its own
    exception so a single failure never aborts the rest.
    """
    from pivtools_gui.stereo_reconstruction.self_calibration import (
        compute_dewarp_maps,
        dewarp_image,
    )

    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = result.history
    if not hist:
        logger.warning("self-cal: no iteration history — skipping figures")
        return out_dir

    # ----- fig1: convergence history -----
    iters = [h.iteration for h in hist]
    rms_vals = [h.rms_disparity for h in hist]
    z_vals = [h.cumulative_z for h in hist]
    tx_vals = [math.degrees(h.cumulative_tilt_x) for h in hist]
    ty_vals = [math.degrees(h.cumulative_tilt_y) for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Self-Calibration Convergence — cam{cam1_num} vs cam{cam2_num}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 0].semilogy(iters, rms_vals, "bo-", lw=2, ms=8)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("RMS disparity (px)")
    axes[0, 0].set_title("RMS Disparity Convergence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, z_vals, "rs-", lw=2, ms=8)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Z offset (mm)")
    axes[0, 1].set_title(f"Z Recovery (final: {result.z_offset:.3f} mm)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, tx_vals, "g^-", lw=2, ms=8, label="Tilt X")
    axes[1, 0].plot(iters, ty_vals, "mD-", lw=2, ms=8, label="Tilt Y")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Tilt (deg)")
    axes[1, 0].set_title("Tilt Recovery")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    rows = [
        ["Parameter", "Value"],
        ["Z offset", f"{result.z_offset:.4f} mm"],
        ["Tilt X", f"{math.degrees(result.tilt_x):.4f} deg"],
        ["Tilt Y", f"{math.degrees(result.tilt_y):.4f} deg"],
        ["Final RMS", f"{result.final_rms_disparity:.4f} px"],
        ["Iterations", f"{result.n_iterations}"],
        ["Converged", f"{result.converged}"],
        ["Initial RMS", f"{hist[0].rms_disparity:.2f} px"],
        [
            "Reduction",
            f"{hist[0].rms_disparity / max(result.final_rms_disparity, 0.001):.1f}x",
        ],
    ]
    table = axes[1, 1].table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colWidths=[0.45, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for j in range(2):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#d0d0d0")

    fig.tight_layout()
    fig.savefig(str(out_dir / "fig1_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- fig2 + fig3: disparity field before/after + histograms -----
    dx_b, dy_b = result.dx_before, result.dy_before
    dx_a, dy_a = result.dx_after, result.dy_after

    if dx_b is not None and dx_a is not None:
        mag_b = np.sqrt(
            np.where(np.isfinite(dx_b), dx_b, 0) ** 2
            + np.where(np.isfinite(dy_b), dy_b, 0) ** 2
        )
        mag_a = np.sqrt(
            np.where(np.isfinite(dx_a), dx_a, 0) ** 2
            + np.where(np.isfinite(dy_a), dy_a, 0) ** 2
        )
        vmax = float(np.nanpercentile(mag_b, 95)) if mag_b.size else 1.0
        rms_b = float(np.sqrt(np.nanmean(dx_b**2 + dy_b**2)))
        rms_a = float(np.sqrt(np.nanmean(dx_a**2 + dy_a**2)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Disparity Fields: Before vs After Correction",
            fontsize=14,
            fontweight="bold",
        )
        for row, (dx, dy, mag, rms, lbl) in enumerate(
            [
                (dx_b, dy_b, mag_b, rms_b, "BEFORE"),
                (dx_a, dy_a, mag_a, rms_a, "AFTER"),
            ]
        ):
            im = axes[row, 0].imshow(dx, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 0].set_title(f"dx {lbl}\nmean={np.nanmean(dx):.2f} px")
            plt.colorbar(im, ax=axes[row, 0], shrink=0.8)
            im = axes[row, 1].imshow(dy, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, 1].set_title(f"dy {lbl}\nmean={np.nanmean(dy):.2f} px")
            plt.colorbar(im, ax=axes[row, 1], shrink=0.8)
            im = axes[row, 2].imshow(mag, cmap="hot", vmin=0, vmax=vmax)
            axes[row, 2].set_title(f"|d| {lbl}\nRMS={rms:.2f} px")
            plt.colorbar(im, ax=axes[row, 2], shrink=0.8)
        for ax in axes.flat:
            ax.set_xlabel("Window X")
            ax.set_ylabel("Window Y")
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig2_disparity_before_after.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Mean/std of finite values — what self-cal minimises (mean → 0) and the
        # irreducible noise floor (std).
        dx_b_mean = float(np.nanmean(dx_b))
        dx_b_std = float(np.nanstd(dx_b))
        dx_a_mean = float(np.nanmean(dx_a))
        dx_a_std = float(np.nanstd(dx_a))
        dy_b_mean = float(np.nanmean(dy_b))
        dy_b_std = float(np.nanstd(dy_b))
        dy_a_mean = float(np.nanmean(dy_a))
        dy_a_std = float(np.nanstd(dy_a))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Disparity Distributions: Before vs After\n"
            "(vertical lines mark mean — self-cal's minimisation target)",
            fontsize=13,
            fontweight="bold",
        )
        bins = np.linspace(-vmax * 1.5, vmax * 1.5, 60)
        ax1.hist(
            dx_b[np.isfinite(dx_b)].ravel(),
            bins=bins,
            alpha=0.6,
            color="red",
            label=f"Before (μ={dx_b_mean:+.3f}, σ={dx_b_std:.2f})",
        )
        ax1.hist(
            dx_a[np.isfinite(dx_a)].ravel(),
            bins=bins,
            alpha=0.6,
            color="green",
            label=f"After (μ={dx_a_mean:+.3f}, σ={dx_a_std:.2f})",
        )
        ax1.axvline(dx_b_mean, color="darkred", lw=2, ls="--")
        ax1.axvline(dx_a_mean, color="darkgreen", lw=2, ls="--")
        ax1.set_xlabel("dx disparity (px)")
        ax1.set_ylabel("Count")
        ax1.set_title("dx Disparity")
        ax1.legend(fontsize=9)
        ax1.axvline(0, color="k", ls="-", alpha=0.3)

        ax2.hist(
            dy_b[np.isfinite(dy_b)].ravel(),
            bins=bins,
            alpha=0.6,
            color="red",
            label=f"Before (μ={dy_b_mean:+.3f}, σ={dy_b_std:.2f})",
        )
        ax2.hist(
            dy_a[np.isfinite(dy_a)].ravel(),
            bins=bins,
            alpha=0.6,
            color="green",
            label=f"After (μ={dy_a_mean:+.3f}, σ={dy_a_std:.2f})",
        )
        ax2.axvline(dy_b_mean, color="darkred", lw=2, ls="--")
        ax2.axvline(dy_a_mean, color="darkgreen", lw=2, ls="--")
        ax2.set_xlabel("dy disparity (px)")
        ax2.set_ylabel("Count")
        ax2.set_title("dy Disparity")
        ax2.legend(fontsize=9)
        ax2.axvline(0, color="k", ls="-", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig3_disparity_histograms.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ----- fig4: dewarped red/cyan overlay before/after -----
    if images_cam1 and images_cam2:
        img1 = images_cam1[0]
        img2 = images_cam2[0]
        m1b = compute_dewarp_maps(cam1, world_bounds, mm_per_pixel)
        m2b = compute_dewarp_maps(cam2, world_bounds, mm_per_pixel)
        dw1b = dewarp_image(img1, m1b[0], m1b[1])
        dw2b = dewarp_image(img2, m2b[0], m2b[1])
        m1a = compute_dewarp_maps(
            cam1,
            world_bounds,
            mm_per_pixel,
            result.z_offset,
            result.tilt_x,
            result.tilt_y,
        )
        m2a = compute_dewarp_maps(
            cam2,
            world_bounds,
            mm_per_pixel,
            result.z_offset,
            result.tilt_x,
            result.tilt_y,
        )
        dw1a = dewarp_image(img1, m1a[0], m1a[1])
        dw2a = dewarp_image(img2, m2a[0], m2a[1])

        def _rc(d1, d2):
            r = _percentile_u8(d1)
            c = _percentile_u8(d2)
            return np.stack([r, c, c], axis=-1)

        ov_before = _rc(dw1b, dw2b)
        ov_after = _rc(dw1a, dw2a)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Dewarped Particle Overlay: Before vs After Self-Cal",
            fontsize=13,
            fontweight="bold",
        )
        x_min, x_max, y_min, y_max = world_bounds
        extent = [x_min, x_max, y_min, y_max]
        ax1.imshow(ov_before, extent=extent, origin="lower", aspect="equal")
        ax1.set_title("BEFORE (Z=0, no tilt)")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax2.imshow(ov_after, extent=extent, origin="lower", aspect="equal")
        ax2.set_title(
            f"AFTER (Z={result.z_offset:.2f} mm, "
            f"tX={math.degrees(result.tilt_x):.2f}°)"
        )
        ax2.set_xlabel("X (mm)")
        fig.tight_layout()
        fig.savefig(
            str(out_dir / "fig4_overlay_before_after.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # ----- fig7: zoom windows so individual particles are resolvable -----
        # fig4 is too coarse to see particles on a large board. Crop the same dewarped
        # rasters at 5 world locations (corners + centre) and show each as a
        # before|after red/cyan pair so the alignment improvement is visible per spot.
        try:
            zoom_px = 100  # window size in dewarped-raster pixels (user-specified)
            half = zoom_px // 2
            mx = (x_max - x_min) * 0.15
            my = (y_max - y_min) * 0.15
            cx_mm, cy_mm = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
            locs_mm = [
                ("centre", cx_mm, cy_mm),
                ("lower-left", x_min + mx, y_min + my),
                ("lower-right", x_max - mx, y_min + my),
                ("upper-left", x_min + mx, y_max - my),
                ("upper-right", x_max - mx, y_max - my),
            ]
            ny_r, nx_r = dw1b.shape[:2]

            def _crop4(cx_px, cy_px):
                x0, y0 = int(round(cx_px)) - half, int(round(cy_px)) - half
                x1, y1 = x0 + zoom_px, y0 + zoom_px
                xa, xb = max(0, x0), min(nx_r, x1)
                ya, yb = max(0, y0), min(ny_r, y1)
                if xb - xa < 4 or yb - ya < 4:
                    return None
                sl = (slice(ya, yb), slice(xa, xb))
                return dw1b[sl], dw2b[sl], dw1a[sl], dw2a[sl]

            n = len(locs_mm)
            figz, axz = plt.subplots(n, 2, figsize=(7, 3.0 * n), squeeze=False)
            figz.suptitle(
                f"Self-cal zoom (red=cam{cam1_num}, cyan=cam{cam2_num}; "
                f"sharp overlap = aligned), {zoom_px}px windows",
                fontsize=12,
                fontweight="bold",
            )
            for row, (name, xw, yw) in enumerate(locs_mm):
                cx_px = (xw - x_min) / mm_per_pixel
                cy_px = (yw - y_min) / mm_per_pixel
                crops = _crop4(cx_px, cy_px)
                if crops is None:
                    for col in range(2):
                        axz[row][col].text(
                            0.5, 0.5, "out of bounds", ha="center", va="center",
                            transform=axz[row][col].transAxes, color="red", fontsize=9,
                        )
                        axz[row][col].set_xticks([])
                        axz[row][col].set_yticks([])
                    continue
                c1b, c2b, c1a, c2a = crops
                axz[row][0].imshow(_rc(c1b, c2b), origin="lower")
                axz[row][1].imshow(_rc(c1a, c2a), origin="lower")
                axz[row][0].set_ylabel(
                    f"{name}\n({xw:.0f}, {yw:.0f}) mm", fontsize=8
                )
                if row == 0:
                    axz[row][0].set_title("BEFORE", fontsize=10)
                    axz[row][1].set_title("AFTER", fontsize=10)
                for col in range(2):
                    axz[row][col].set_xticks([])
                    axz[row][col].set_yticks([])
            figz.tight_layout(rect=[0, 0, 1, 0.97])
            figz.savefig(
                str(out_dir / "fig7_zoom_before_after.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(figz)
        except Exception:
            logger.warning(f"self-cal zoom figure failed: {traceback.format_exc()}")

        # ----- fig5: BEFORE vs AFTER correlation planes at 6 world positions -----
        # Pulls correlation planes directly from the C-library output stored in the
        # result (corr_first_iter = iter 1 = before; corr_last_iter = final = after).
        # No Python recomputation — the planes shown are pixel-exact what self-cal
        # optimised on. Each probe snaps to the nearest C-library window centre.
        try:
            if (
                result.corr_first_iter is None
                or result.corr_last_iter is None
                or result.win_ctrs_x is None
                or result.win_ctrs_y is None
                or result.window_size_used is None
            ):
                raise RuntimeError(
                    "correlation planes missing from SelfCalibrationResult"
                )

            corr_before = result.corr_first_iter
            corr_after = result.corr_last_iter
            win_ctrs_x = np.asarray(result.win_ctrs_x, dtype=np.float32)
            win_ctrs_y = np.asarray(result.win_ctrs_y, dtype=np.float32)
            ws_full = int(result.window_size_used)

            crop = min(64, ws_full)
            c0 = (ws_full - crop) // 2
            c1 = c0 + crop

            x_min, x_max, y_min, y_max = world_bounds
            mx = (x_max - x_min) * 0.15
            my = (y_max - y_min) * 0.15
            xs_probe = np.linspace(x_min + mx, x_max - mx, 3)
            ys_probe = np.linspace(y_min + my, y_max - my, 2)

            img1_a = dewarp_image(images_cam1[0], m1a[0], m1a[1])
            img2_a = dewarp_image(images_cam2[0], m2a[0], m2a[1])
            out_h, out_w = img1_a.shape

            W_thumb = 64
            half_thumb = W_thumb // 2

            def _stretch(im, lo_pct=2.0, hi_pct=99.5):
                lo = float(np.percentile(im, lo_pct))
                hi = float(np.percentile(im, hi_pct))
                if hi - lo < 1e-6:
                    hi = lo + 1.0
                return np.clip((im - lo) / (hi - lo), 0, 1)

            n_iters = len(result.history)
            crop_label = f" (display crop: {crop}×{crop})" if crop < ws_full else ""

            fig, axes = plt.subplots(2, 15, figsize=(32, 7))
            fig.suptitle(
                f"Correlation planes BEFORE vs AFTER self-cal — "
                f"cam{cam1_num} vs cam{cam2_num} (z={result.z_offset:.4f} mm, "
                f"tx={math.degrees(result.tilt_x):+.4f}°, "
                f"ty={math.degrees(result.tilt_y):+.4f}°)\n"
                f"per position: cam{cam1_num} f0 | cam{cam2_num} f0 | "
                f"R/G overlay | corr BEFORE (iter 1) | "
                f"corr AFTER (iter {n_iters}) — "
                f"C library window: {ws_full}×{ws_full}{crop_label}",
                fontsize=11,
                fontweight="bold",
            )

            for row, y_t in enumerate(ys_probe[::-1]):
                for col, x_t in enumerate(xs_probe):
                    base_col = col * 5

                    probe_px_x = (x_t - x_min) / mm_per_pixel
                    probe_px_y = (y_t - y_min) / mm_per_pixel
                    ix = int(np.argmin(np.abs(win_ctrs_x - probe_px_x)))
                    iy = int(np.argmin(np.abs(win_ctrs_y - probe_px_y)))
                    snap_px_x = float(win_ctrs_x[ix])
                    snap_px_y = float(win_ctrs_y[iy])
                    snap_mm_x = x_min + snap_px_x * mm_per_pixel
                    snap_mm_y = y_min + snap_px_y * mm_per_pixel

                    cx_p = int(round(snap_px_x))
                    cy_p = int(round(snap_px_y))
                    x0 = cx_p - half_thumb
                    x1 = cx_p + half_thumb
                    y0 = cy_p - half_thumb
                    y1 = cy_p + half_thumb
                    if x0 < 0 or y0 < 0 or x1 > out_w or y1 > out_h:
                        for k in range(5):
                            axes[row, base_col + k].axis("off")
                        continue

                    s1 = _stretch(img1_a[y0:y1, x0:x1])
                    s2 = _stretch(img2_a[y0:y1, x0:x1])
                    axes[row, base_col + 0].imshow(
                        s1,
                        origin="lower",
                        cmap="inferno",
                        vmin=0,
                        vmax=1,
                    )
                    axes[row, base_col + 0].set_title(
                        f"({snap_mm_x:+.0f},{snap_mm_y:+.0f}) c{cam1_num} f0",
                        fontsize=9,
                    )
                    axes[row, base_col + 1].imshow(
                        s2,
                        origin="lower",
                        cmap="inferno",
                        vmin=0,
                        vmax=1,
                    )
                    axes[row, base_col + 1].set_title(f"c{cam2_num} f0", fontsize=9)
                    rgb = np.zeros((W_thumb, W_thumb, 3), dtype=np.float32)
                    rgb[..., 0] = s1
                    rgb[..., 1] = s2
                    axes[row, base_col + 2].imshow(rgb, origin="lower")
                    axes[row, base_col + 2].set_title(
                        f"R=c{cam1_num}, G=c{cam2_num}",
                        fontsize=9,
                    )

                    plane_before = corr_before[iy, ix, c0:c1, c0:c1]
                    plane_after = corr_after[iy, ix, c0:c1, c0:c1]
                    vmax_b = float(np.max(plane_before))
                    vmax_a = float(np.max(plane_after))

                    axes[row, base_col + 3].imshow(
                        plane_before,
                        origin="lower",
                        cmap="viridis",
                        vmin=float(np.min(plane_before)),
                        vmax=vmax_b,
                    )
                    axes[row, base_col + 3].set_title(
                        f"BEFORE iter 1\npeak={vmax_b:.3g}",
                        fontsize=8,
                    )
                    axes[row, base_col + 4].imshow(
                        plane_after,
                        origin="lower",
                        cmap="viridis",
                        vmin=float(np.min(plane_after)),
                        vmax=vmax_a,
                    )
                    axes[row, base_col + 4].set_title(
                        f"AFTER iter {n_iters}\npeak={vmax_a:.3g}",
                        fontsize=8,
                    )

                    for k in range(5):
                        axes[row, base_col + k].set_xticks([])
                        axes[row, base_col + k].set_yticks([])

            fig.tight_layout()
            fig.savefig(
                str(out_dir / "fig5_correlation_probes.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"self-cal fig5_correlation_probes failed: {e}")

    # ----- fig6: forward-model decomposition -----
    # Forward-predict the iter-1 BEFORE disparity field from iter-1's recovered
    # (delta_z, delta_tilt_x, delta_tilt_y) using the SAME linear model that
    # fit_disparity_plane inverts. Residual = observed − predicted = noise floor.
    if (
        result.dx_before is not None
        and result.dy_before is not None
        and result.grid_x_mm is not None
        and result.grid_y_mm is not None
        and result.disp_px_per_mm is not None
        and result.disp_direction is not None
        and result.history
    ):
        try:
            dx_o = result.dx_before
            dy_o = result.dy_before
            Xm = result.grid_x_mm
            Ym = result.grid_y_mm
            dpm = float(result.disp_px_per_mm)
            ddir = np.asarray(result.disp_direction, dtype=np.float64)

            h1 = result.history[0]
            z_i1 = float(h1.cumulative_z)
            tx_i1 = float(h1.cumulative_tilt_x)
            ty_i1 = float(h1.cumulative_tilt_y)

            disp_mag_pred = dpm * (z_i1 + math.tan(ty_i1) * Xm + math.tan(tx_i1) * Ym)
            dx_pred = disp_mag_pred * ddir[0]
            dy_pred = disp_mag_pred * ddir[1]
            dx_res = dx_o - dx_pred
            dy_res = dy_o - dy_pred

            finite_obs = dy_o[np.isfinite(dy_o)]
            v = (
                float(np.nanpercentile(np.abs(finite_obs), 98))
                if finite_obs.size
                else 1.0
            )
            if v < 0.1:
                v = 0.1

            def _stats(a):
                finite = a[np.isfinite(a)]
                if finite.size == 0:
                    return 0.0, 0.0
                return float(np.mean(finite)), float(np.std(finite))

            dy_o_m, dy_o_s = _stats(dy_o)
            dy_p_m, dy_p_s = _stats(dy_pred)
            dy_r_m, dy_r_s = _stats(dy_res)
            dx_o_m, dx_o_s = _stats(dx_o)
            dx_p_m, dx_p_s = _stats(dx_pred)
            dx_r_m, dx_r_s = _stats(dx_res)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(
                "Forward-Model Decomposition — "
                f"cam{cam1_num} vs cam{cam2_num}\n"
                f"Iter-1 fit: dz={z_i1:+.4f} mm, "
                f"dtx={math.degrees(tx_i1):+.4f}°, "
                f"dty={math.degrees(ty_i1):+.4f}° | "
                f"sensitivity={dpm:.2f} px/mm, "
                f"direction=({ddir[0]:+.3f}, {ddir[1]:+.3f}) | "
                f"Final converged: z={result.z_offset:+.4f} mm, "
                f"tx={math.degrees(result.tilt_x):+.4f}°, "
                f"ty={math.degrees(result.tilt_y):+.4f}°\n"
                "PREDICTED is the linear fit through iter-1 OBSERVED, "
                "so RESIDUAL = observation - fit = noise floor.",
                fontsize=11,
                fontweight="bold",
            )

            panels = [
                ("dy OBSERVED (iter 1)", dy_o, dy_o_m, dy_o_s),
                ("dy PREDICTED from (z,tx,ty)", dy_pred, dy_p_m, dy_p_s),
                ("dy RESIDUAL = obs − pred", dy_res, dy_r_m, dy_r_s),
            ]
            for i, (title, fld, mean_v, std_v) in enumerate(panels):
                im = axes[0, i].imshow(
                    fld, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower"
                )
                axes[0, i].set_title(
                    f"{title}\nμ={mean_v:+.3f} px, σ={std_v:.2f} px", fontsize=11
                )
                axes[0, i].set_xlabel("Window X")
                axes[0, i].set_ylabel("Window Y")
                plt.colorbar(im, ax=axes[0, i], shrink=0.8)

            panels_x = [
                ("dx OBSERVED (iter 1)", dx_o, dx_o_m, dx_o_s),
                ("dx PREDICTED", dx_pred, dx_p_m, dx_p_s),
                ("dx RESIDUAL", dx_res, dx_r_m, dx_r_s),
            ]
            for i, (title, fld, mean_v, std_v) in enumerate(panels_x):
                im = axes[1, i].imshow(
                    fld, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower"
                )
                axes[1, i].set_title(
                    f"{title}\nμ={mean_v:+.3f} px, σ={std_v:.2f} px", fontsize=11
                )
                axes[1, i].set_xlabel("Window X")
                axes[1, i].set_ylabel("Window Y")
                plt.colorbar(im, ax=axes[1, i], shrink=0.8)

            fig.tight_layout()
            fig.savefig(
                str(out_dir / "fig6_forward_model.png"), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"self-cal fig6_forward_model failed: {e}")

    # ----- correlation_planes.mat (first + last iteration C-correlator output) -----
    try:
        from scipy.io import savemat

        if result.corr_first_iter is not None and result.corr_last_iter is not None:
            mat_data = {
                "corr_first_iter": result.corr_first_iter.astype(np.float32),
                "corr_last_iter": result.corr_last_iter.astype(np.float32),
                "win_ctrs_x": np.asarray(result.win_ctrs_x, dtype=np.float32),
                "win_ctrs_y": np.asarray(result.win_ctrs_y, dtype=np.float32),
                "n_win_x": np.int32(result.n_win_x or 0),
                "n_win_y": np.int32(result.n_win_y or 0),
                "window_size": np.int32(result.window_size_used or 0),
                "mm_per_pixel": np.float64(result.mm_per_pixel or 0.0),
                "world_bounds": np.asarray(world_bounds, dtype=np.float64),
                "z_offset_final": np.float64(result.z_offset),
                "tilt_x_final": np.float64(result.tilt_x),
                "tilt_y_final": np.float64(result.tilt_y),
                "n_iterations": np.int32(result.n_iterations),
            }
            if result.grid_x_mm is not None:
                mat_data["grid_x_mm"] = np.asarray(result.grid_x_mm, dtype=np.float32)
            if result.grid_y_mm is not None:
                mat_data["grid_y_mm"] = np.asarray(result.grid_y_mm, dtype=np.float32)
            savemat(
                str(out_dir / "correlation_planes.mat"), mat_data, do_compression=True
            )
        else:
            logger.warning("self-cal: no correlation planes captured — skipping .mat")
    except Exception as e:
        logger.warning(f"self-cal: failed to save correlation_planes.mat: {e}")

    logger.info(f"self-cal: saved figures to {out_dir}")
    return out_dir
