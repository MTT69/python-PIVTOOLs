"""
Calibration Diagnostic Figures

Reusable figure generation for calibration pipelines. All functions are
headless-safe (Agg backend) and wrapped in try/except so that figure
failure never blocks calibration.

Used by:
  - planar_calibration_production.py (dotboard)
  - charuco_calibration_production.py
  - stereo_dotboard_calibration_production.py
  - stereo_charuco_calibration_production.py
  - stepped_calibration_production.py
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from loguru import logger


# ---------------------------------------------------------------------------
# Grid network drawing helpers
# ---------------------------------------------------------------------------

def _draw_grid_network(ax, centers, grid_indices, color='limegreen', lw=0.6, alpha=0.7):
    """Draw lines between grid-neighboring points on a matplotlib axis."""
    idx_map = {}
    for i, gi in enumerate(grid_indices):
        idx_map[(int(gi[0]), int(gi[1]))] = i
    for i, gi in enumerate(grid_indices):
        c, r = int(gi[0]), int(gi[1])
        for dc, dr in [(1, 0), (0, 1)]:
            nb = (c + dc, r + dr)
            if nb in idx_map:
                j = idx_map[nb]
                ax.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]],
                        color=color, linewidth=lw, alpha=alpha)


def _draw_grid_network_cv(img, centers, grid_indices, color=(0, 200, 0), thickness=2):
    """Draw grid network lines on an OpenCV image (in-place)."""
    idx_map = {}
    for i, gi in enumerate(grid_indices):
        idx_map[(int(gi[0]), int(gi[1]))] = i
    for i, gi in enumerate(grid_indices):
        c, r = int(gi[0]), int(gi[1])
        pt1 = (int(centers[i, 0]), int(centers[i, 1]))
        for dc, dr in [(1, 0), (0, 1)]:
            nb = (c + dc, r + dr)
            if nb in idx_map:
                j = idx_map[nb]
                pt2 = (int(centers[j, 0]), int(centers[j, 1]))
                cv2.line(img, pt1, pt2, color, thickness)


def _to_uint8_gray(img):
    """Ensure image is uint8 grayscale for display."""
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[0] == 1:
            img = img[0]
        elif img.shape[-1] == 1:
            img = img[:, :, 0]
        else:
            img = np.squeeze(img)
    if img.dtype in (np.float32, np.float64):
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# 1. Per-frame dotboard detection figure (6-panel)
# ---------------------------------------------------------------------------

def make_detection_figure(
    image: np.ndarray,
    success: bool,
    grid_data: Optional[Dict[str, Any]],
    info: Dict[str, Any],
    output_path,
    title: Optional[str] = None,
):
    """Generate and save a 3-panel detection diagnostic figure.

    Panels:
        1. Blob detection overlay (green=kept, red=noise-filtered)
        2. Final grid network
        3. Final grid overlaid on image

    Parameters
    ----------
    image : np.ndarray
        Raw input image.
    success : bool
        Whether detection succeeded.
    grid_data : dict or None
        Detection result with 'centers', 'grid_indices', 'n_cols', 'n_rows'.
    info : dict
        Diagnostic info from detect_grid_automatic().
    output_path : str or Path
        Where to save the figure.
    title : str, optional
        Title prefix for the figure.
    """
    try:
        from pivtools_gui.calibration.detection.grid_detection import to_grayscale_2d

        fig = plt.figure(figsize=(20, 7))
        status = (f'SUCCESS: {grid_data["n_cols"]}x{grid_data["n_rows"]} ({len(grid_data["centers"])} pts)'
                  if success and grid_data else f'FAILED: {info.get("error", "?")[:60]}')
        fig.suptitle(f'{title or "Detection"}  —  {status}', fontsize=13,
                     color='darkgreen' if success else 'darkred')
        gs = GridSpec(1, 3, figure=fig, wspace=0.10)

        clahe_img = info.get('gray_image')
        if clahe_img is None:
            clahe_img = to_grayscale_2d(_to_uint8_gray(image))
        scale = max(1, clahe_img.shape[1] // 1200)
        noise_thresh = info.get('noise_threshold', 0)

        # Panel 1: Blob detection
        ax1 = fig.add_subplot(gs[0, 0])
        disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        for (pt, sz) in info.get('all_keypoints', []):
            color = (255, 80, 80) if sz < noise_thresh else (0, 255, 0)
            cv2.circle(disp, (int(pt[0]), int(pt[1])), max(int(sz / 2), 3), color, 2)
        ax1.imshow(disp[::scale, ::scale])
        n_raw = len(info.get('all_keypoints', []))
        n_noise = info.get('noise_blobs_filtered', 0)
        n_kept = n_raw - n_noise
        ax1.set_title(f'Blob Detection: {n_kept} kept, {n_noise} filtered', fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Panel 2: Final grid network
        ax2 = fig.add_subplot(gs[0, 1])
        if success and grid_data:
            _draw_grid_network(ax2, grid_data['centers'], grid_data['grid_indices'])
            ax2.scatter(grid_data['centers'][:, 0], grid_data['centers'][:, 1],
                        c='limegreen', s=10, zorder=5,
                        label=f'{len(grid_data["centers"])} pts')
            ax2.legend(fontsize=8)
            ax2.invert_yaxis()
            ax2.set_title(f'Final Grid ({grid_data["n_cols"]}x{grid_data["n_rows"]})', fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Panel 3: Grid on image
        ax3 = fig.add_subplot(gs[0, 2])
        result_disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        if success and grid_data:
            _draw_grid_network_cv(result_disp, grid_data['centers'], grid_data['grid_indices'])
            for pt in grid_data['centers']:
                cv2.circle(result_disp, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 3)
        ax3.imshow(result_disp[::scale, ::scale])
        ax3.set_title('Final Grid on Image', fontsize=10)
        ax3.set_xticks([]); ax3.set_yticks([])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate detection figure: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 2. Per-frame ChArUco detection figure (3-panel)
# ---------------------------------------------------------------------------

def make_charuco_detection_figure(
    image: np.ndarray,
    corners: Optional[np.ndarray],
    ids: Optional[np.ndarray],
    board_params: Dict[str, Any],
    output_path,
    title: Optional[str] = None,
):
    """Generate and save a 3-panel ChArUco detection figure.

    Panels:
        (0) Image with detected corners color-coded by ID
        (1) Corner coverage map
        (2) Text summary
    """
    try:
        fig = plt.figure(figsize=(18, 6))
        n_corners = len(corners) if corners is not None else 0
        sq_h = board_params.get('squares_h', '?')
        sq_v = board_params.get('squares_v', '?')
        status = f'{n_corners} corners' if n_corners > 0 else 'FAILED'
        fig.suptitle(f'{title or "ChArUco Detection"}  —  {status}', fontsize=13,
                     color='darkgreen' if n_corners > 0 else 'darkred')

        img_disp = _to_uint8_gray(image)
        if img_disp.ndim == 2:
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
        h, w = img_disp.shape[:2]
        scale = max(1, w // 1200)

        # Panel 1: Corners on image
        ax1 = fig.add_subplot(1, 3, 1)
        disp = img_disp.copy()
        if corners is not None and len(corners) > 0:
            for i, pt in enumerate(corners):
                cid = int(ids[i]) if ids is not None and i < len(ids) else i
                color_rgb = plt.cm.hsv(cid / max(n_corners, 1))[:3]
                color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 8, color_bgr, 3)
        ax1.imshow(disp[::scale, ::scale])
        ax1.set_title('Detected Corners', fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Panel 2: Coverage map
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_xlim(0, w); ax2.set_ylim(h, 0)
        ax2.set_aspect('equal')
        ax2.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor='gray', linewidth=1))
        if corners is not None and len(corners) > 0:
            ax2.scatter(corners[:, 0], corners[:, 1], c='limegreen', s=8, zorder=5)
        ax2.set_title('Corner Coverage', fontsize=10)

        # Panel 3: Summary text
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis('off')
        total_possible = (int(sq_h) - 1) * (int(sq_v) - 1) if isinstance(sq_h, int) and isinstance(sq_v, int) else '?'
        text = (
            f'ChArUco Detection Summary\n\n'
            f'Board:  {sq_h} x {sq_v} squares\n'
            f'Square size: {board_params.get("square_size_mm", board_params.get("square_size", "?"))} mm\n'
            f'Marker ratio: {board_params.get("marker_ratio", "?")}\n\n'
            f'Corners detected: {n_corners}\n'
            f'Total possible: {total_possible}\n'
            f'Detection rate: {n_corners / total_possible * 100:.0f}%\n' if isinstance(total_possible, int) and total_possible > 0 else
            f'ChArUco Detection Summary\n\n'
            f'Board:  {sq_h} x {sq_v} squares\n'
            f'Corners detected: {n_corners}\n'
        )
        ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=11,
                 va='top', fontfamily='monospace', color='darkblue')
        ax3.set_title('Summary', fontsize=10)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate ChArUco detection figure: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3. Calibration model summary figure (6-panel)
# ---------------------------------------------------------------------------

def make_calibration_model_figure(
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: list,
    tvecs: list,
    rms: float,
    all_imgpoints: list,
    all_objpoints: list,
    image_size: Tuple[int, int],
    output_path,
    best_image: Optional[np.ndarray] = None,
):
    """Generate and save a 6-panel calibration model summary figure.

    Panels:
        (0,0) All detection points overlaid on best image
        (0,1) Per-view reprojection error bar chart
        (0,2) Camera intrinsics and distortion text
        (1,0) Distortion map (pixel displacement heatmap)
        (1,1) Reprojection residual scatter (dx vs dy)
        (1,2) Point coverage (convex hulls per view)

    Parameters
    ----------
    K : np.ndarray
        Camera matrix (3x3).
    dist : np.ndarray
        Distortion coefficients.
    rvecs, tvecs : list
        Per-view extrinsics.
    rms : float
        Overall RMS reprojection error.
    all_imgpoints : list
        Per-view image point arrays.
    all_objpoints : list
        Per-view object point arrays.
    image_size : tuple
        (width, height) of calibration images.
    output_path : str or Path
        Where to save the figure.
    best_image : np.ndarray, optional
        Background image for all-points panel.
    """
    try:
        n_views = len(all_objpoints)
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'Calibration Model — {n_views} views, RMS={rms:.3f} px', fontsize=14)
        gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.25)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_views, 1)))
        w, h = image_size

        # Panel 1: All points (on image if available)
        ax1 = fig.add_subplot(gs[0, 0])
        if best_image is not None:
            bg = _to_uint8_gray(best_image)
            clahe_bg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(bg)
            sc = max(1, clahe_bg.shape[1] // 1200)
            ax1.imshow(clahe_bg[::sc, ::sc], cmap='gray', alpha=0.5)
        else:
            sc = 1
        for idx in range(n_views):
            pts = np.array(all_imgpoints[idx]).reshape(-1, 2)
            ax1.scatter(pts[:, 0] / sc, pts[:, 1] / sc,
                        c=[colors[idx]], s=4, label=f'view {idx + 1} ({len(pts)})')
        if n_views <= 15:
            ax1.legend(fontsize=7, loc='lower left')
        ax1.set_title('All Points', fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Panel 2: Intrinsics text
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.axis('off')
        dist_flat = np.array(dist).flatten()
        text = (
            f'Camera Intrinsics\n\n'
            f'fx = {K[0, 0]:.1f} px\n'
            f'fy = {K[1, 1]:.1f} px\n'
            f'cx = {K[0, 2]:.1f} px  (centre: {w / 2:.0f})\n'
            f'cy = {K[1, 2]:.1f} px  (centre: {h / 2:.0f})\n\n'
            f'Distortion:\n'
            f'  k1 = {dist_flat[0]:.6f}\n'
            f'  k2 = {dist_flat[1]:.6f}\n'
            f'  p1 = {dist_flat[2]:.6f}\n'
            f'  p2 = {dist_flat[3]:.6f}\n\n'
            f'Image: {w} x {h}\n'
            f'Views: {n_views}\n'
            f'Total: {sum(len(np.array(o).reshape(-1, 3)) for o in all_objpoints)} pts\n'
            f'RMS: {rms:.4f} px'
        )
        ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=11,
                 va='top', fontfamily='monospace', color='darkblue')
        ax3.set_title('Calibration Parameters', fontsize=10)

        # Panel 4: Distortion map
        ax4 = fig.add_subplot(gs[1, 0])
        try:
            new_cam, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            grid_pts = np.array(
                [[gx, gy] for gy in range(0, h, max(h // 20, 1)) for gx in range(0, w, max(w // 20, 1))],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            undist_pts = cv2.undistortPoints(grid_pts, K, dist, P=new_cam)
            orig = grid_pts.reshape(-1, 2)
            undist = undist_pts.reshape(-1, 2)
            mag = np.sqrt(np.sum((undist - orig) ** 2, axis=1))
            sc4 = ax4.scatter(orig[:, 0], orig[:, 1], c=mag, cmap='hot', s=6, vmin=0)
            plt.colorbar(sc4, ax=ax4, label='px')
            ax4.set_xlim(0, w); ax4.set_ylim(h, 0)
            ax4.set_aspect('equal')
        except Exception:
            ax4.text(0.5, 0.5, 'Distortion map\nunavailable', ha='center', va='center',
                     transform=ax4.transAxes)
        ax4.set_title('Distortion Map', fontsize=10)

        # Panel 5: Residual scatter
        ax5 = fig.add_subplot(gs[1, 1])
        all_rx = []
        all_ry = []
        for i in range(n_views):
            proj, _ = cv2.projectPoints(all_objpoints[i], rvecs[i], tvecs[i], K, dist)
            res = (np.array(all_imgpoints[i]).reshape(-1, 1, 2) - proj).reshape(-1, 2)
            all_rx.extend(res[:, 0])
            all_ry.extend(res[:, 1])
        ax5.scatter(all_rx, all_ry, s=3, alpha=0.5, color='steelblue')
        ax5.axhline(0, color='gray', lw=0.5)
        ax5.axvline(0, color='gray', lw=0.5)
        if all_rx:
            mr = max(max(abs(np.array(all_rx))), max(abs(np.array(all_ry)))) * 1.2
            ax5.set_xlim(-mr, mr); ax5.set_ylim(-mr, mr)
        ax5.set_aspect('equal')
        ax5.set_xlabel('Res X (px)')
        ax5.set_ylabel('Res Y (px)')
        ax5.set_title(f'Residuals ({len(all_rx)} pts)', fontsize=10)

        # Panel 6: Coverage (convex hulls)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_xlim(0, w); ax6.set_ylim(h, 0)
        ax6.set_aspect('equal')
        for idx in range(n_views):
            pts = np.array(all_imgpoints[idx]).reshape(-1, 2).astype(np.float32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts).reshape(-1, 2)
                hull_closed = np.vstack([hull, hull[0]])
                ax6.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.15, color=colors[idx])
                ax6.plot(hull_closed[:, 0], hull_closed[:, 1], color=colors[idx], lw=1.5)
            ax6.scatter(pts[:, 0], pts[:, 1], c=[colors[idx]], s=2)
        ax6.set_title('Point Coverage', fontsize=10)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate calibration model figure: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 4. Camera placement figure (2-panel, for stereo)
# ---------------------------------------------------------------------------

def make_camera_placement_figure(
    cam_data_list: List[Dict[str, Any]],
    output_path,
    board_obj_points: Optional[np.ndarray] = None,
    all_pose_obj_points: Optional[List[np.ndarray]] = None,
):
    """Generate a 3D camera-placement figure with the calibration board
    visible in the world frame.

    Layout
    ------
    Four panels on one canvas:
      (1) 3D perspective showing cameras (as small pyramidal frustums),
          their optical axes, and the calibration board(s).
      (2) Top-down X-Z view.
      (3) Side X-Y view.
      (4) End-on Z-Y view.

    The 3D panel is the money shot — same-side stereo rigs look
    symmetric about the board normal, transmission rigs span the board
    with cameras on opposite sides, and operator mistakes (e.g. both
    cameras on the same side when transmission was expected) become
    visually obvious rather than hiding behind sub-pixel RMS numbers.

    Every camera's extrinsic rvec/tvec is interpreted in the WORLD frame
    of its OWN fiducial-anchored object points:
        pos_world = -R^T @ t
        optical_axis_world = R^T @ [0, 0, 1]
    where R = Rodrigues(rvec). Non-datum pose extrinsics from `rvecs_all`
    / `tvecs_all` are shown as faint additional board planes so the
    pose-diversity story is legible at a glance.

    Parameters
    ----------
    cam_data_list : list of dict
        Each dict has:
            'label'      str
            'rvec'       (3,) or (3,1) datum-pose rvec
            'tvec'       (3,) or (3,1) datum-pose tvec
            'color'      matplotlib colour
            'rvecs_all'  optional — list of (3,) rvecs, one per pose (incl. datum)
            'tvecs_all'  optional — list of (3,) tvecs, same length as rvecs_all
            'obj_points' optional — (N, 3) datum-pose 3D points in THIS camera's
                          fiducial-anchored object frame. Used to render the
                          actual dot layout on the board panel.
    board_obj_points : (N, 3) ndarray, optional
        Datum-pose object points to draw as the board. If not given, and
        no cam in cam_data_list has 'obj_points', falls back to a plain
        rectangle.
    all_pose_obj_points : list of (N, 3) ndarray, optional
        One entry per pose. Each is transformed by its pose's (R, t) and
        drawn as a faint plane on the 3D panel to show pose diversity.
    output_path : str or Path
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — side-effect import
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # ---- Extract camera world positions + axes ----
        # The "looking direction" of each camera is computed from the
        # camera centre to the board centroid instead of from
        # `R.T @ [0,0,1]`. Reason: SIG's `Projection.c` convention
        # makes the implicit world→camera map a reflection (det=-1 or
        # absorbed into `t.z < 0`), so OpenCV's recovered R aims the
        # camera's local +Z AWAY from the board. Using "to-centroid"
        # is geometrically unambiguous and works for any calibration
        # convention — OpenCV, SIG, or a real camera.
        cam_records = []
        for cam in cam_data_list:
            rvec = np.asarray(cam['rvec'], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(cam['tvec'], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            pos = (-R.T @ tvec).flatten()            # camera centre in world
            right = (R.T @ np.array([1.0, 0.0, 0.0])).flatten()
            up = (R.T @ np.array([0.0, -1.0, 0.0])).flatten()
            cam_records.append({
                'label': cam.get('label', 'Cam'),
                'color': cam.get('color', 'tab:blue'),
                'pos': pos,
                'axis': None,  # resolved below once we know the board centroid
                'up': up,
                'right': right,
                'rvecs_all': cam.get('rvecs_all'),
                'tvecs_all': cam.get('tvecs_all'),
                'obj_points': cam.get('obj_points'),
            })

        # ---- Board datum object points ----
        # Each per-camera fit has its OWN world frame (anchored at its
        # own fiducial click). For same-side stereo the two frames are
        # z-coincident and cam1's points alone tell the whole story.
        # For transmission stereo cam1's frame has dots at z={0, -h}
        # (front face) while cam2's has dots at z={-(T-h), -T} (back
        # face) — sharing xy but at different z. Showing BOTH per-
        # camera point clouds in one figure is what makes the trans-
        # mission geometry visually obvious: you see 4 z-levels
        # stacked 12 mm apart instead of just 2.
        per_cam_dots: List[Tuple[str, np.ndarray]] = []  # [(color, (N,3)), ...]
        for cr in cam_records:
            if cr['obj_points'] is not None:
                pts = np.asarray(cr['obj_points'], dtype=np.float64)
                per_cam_dots.append((cr['color'], pts, cr['label']))

        if board_obj_points is None and per_cam_dots:
            # Concatenate all per-camera dots so bounding box / axes
            # computations cover the full plate (both faces in tx).
            board_obj_points = np.vstack([pts for _, pts, _ in per_cam_dots])
        if board_obj_points is None:
            # Fallback rectangle — 150 × 60 mm at z=0
            board_obj_points = np.array([
                [0.0, 0.0, 0.0], [150.0, 0.0, 0.0],
                [150.0, 60.0, 0.0], [0.0, 60.0, 0.0],
            ])
        board_obj_points = np.asarray(board_obj_points, dtype=np.float64)

        # Board bounding rectangle at its most-populated z plane
        unique_z = np.unique(np.round(board_obj_points[:, 2], 1))
        z_primary = unique_z[np.argmin(np.abs(unique_z))]
        primary_mask = np.abs(board_obj_points[:, 2] - z_primary) < 0.5
        x_min, x_max = board_obj_points[primary_mask, 0].min(), board_obj_points[primary_mask, 0].max()
        y_min, y_max = board_obj_points[primary_mask, 1].min(), board_obj_points[primary_mask, 1].max()
        board_corners_local = np.array([
            [x_min, y_min, z_primary],
            [x_max, y_min, z_primary],
            [x_max, y_max, z_primary],
            [x_min, y_max, z_primary],
        ])

        # Resolve each camera's optical axis as the direction from its
        # world centre to the full board centroid. This is invariant to
        # whether OpenCV's (R, t) absorbed a reflection.
        board_centroid = board_obj_points.mean(axis=0)
        for cr in cam_records:
            direction = board_centroid - cr['pos']
            norm = float(np.linalg.norm(direction))
            cr['axis'] = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        # ---- Scene bounds for consistent axes ----
        all_pts = [board_obj_points]
        for cr in cam_records:
            all_pts.append(cr['pos'].reshape(1, 3))
        all_pts = np.vstack(all_pts)
        pad = 0.1 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1e-6)
        lo = all_pts.min(axis=0) - pad
        hi = all_pts.max(axis=0) + pad

        # ---- Figure layout ----
        # The geometry spans two very different scales — cameras live
        # thousands of mm from the board (~5 m standoff at PIV magnif-
        # ication), while the board itself is ~150 × 60 mm. Equal-aspect
        # panels squash the board into an invisible sliver, so we let
        # each 2D panel auto-scale independently and rely on two zoomed
        # inset axes to make the board legible at its own scale.
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle('Camera Placement — 3D scene + top-down view + board detail',
                     fontsize=14, y=0.98)
        gs = GridSpec(2, 3, figure=fig,
                      width_ratios=[1.5, 1.2, 1.0],
                      height_ratios=[1.5, 1.0],
                      hspace=0.30, wspace=0.32)

        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax_xz = fig.add_subplot(gs[0, 1])       # top-down (primary 2D)
        ax_legend = fig.add_subplot(gs[0, 2])
        ax_legend.axis('off')
        ax_board = fig.add_subplot(gs[1, 0])    # zoomed board plan view
        ax_zy = fig.add_subplot(gs[1, 1])       # side view
        ax_zoom3d = fig.add_subplot(gs[1, 2])   # zoomed board cross-section

        # ---- 3D panel ----
        # 1. Draw the board dots — coloured per camera so transmission
        #    (cam1 at z≈0, cam2 at z≈-(T-h)) is visually distinct from
        #    same-side (cam1 and cam2 z-coincident).
        if per_cam_dots:
            for color, pts, label in per_cam_dots:
                ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                              c=color, s=10, alpha=0.75,
                              label=f'{label} object frame',
                              edgecolors='black', linewidths=0.2)
        else:
            ax_3d.scatter(board_obj_points[:, 0],
                          board_obj_points[:, 1],
                          board_obj_points[:, 2],
                          c='black', s=8, alpha=0.9, label='Board dots')

        # 2. Draw the primary-face outline as a semi-transparent patch
        board_patch = Poly3DCollection([board_corners_local],
                                        alpha=0.10, facecolor='tab:orange',
                                        edgecolor='tab:orange', linewidth=1.2)
        ax_3d.add_collection3d(board_patch)

        # 3. If we have all-pose extrinsics, draw each non-datum pose
        #    as a faint board plane so the operator can see the pose
        #    diversity. The pose transform takes board-frame corners
        #    through the pose's R/t into THIS camera's view, but we
        #    want them back in the world frame — which is the same
        #    board frame, so the "pose plane" IS the local rectangle
        #    rotated about the world origin by the delta between
        #    non-datum and datum rvec/tvec. Simpler: each rvec/tvec
        #    maps board-frame → camera-frame; applying R_pose^T to the
        #    board corners (without t) gives them back in world-frame.
        #    But that's just the same rectangle. The pose diversity
        #    story the operator wants is "how does the board move
        #    between poses from the camera's POV" — already captured
        #    by detection figures. Skip drawing per-pose outlines;
        #    keep the 3D panel readable.
        _ = all_pose_obj_points  # reserved for future per-pose planes

        # 4. Draw each camera as a pyramidal frustum + optical axis
        axis_len = 0.4 * max(hi - lo)   # scale to scene
        frustum_scale = 0.06 * max(hi - lo)
        for cr in cam_records:
            pos = cr['pos']
            color = cr['color']

            # Pyramid apex at camera centre, base offset along optical axis
            apex = pos
            fwd = cr['axis'] * frustum_scale
            right = cr['right'] * frustum_scale * 0.7
            up = cr['up'] * frustum_scale * 0.5
            base_centre = apex + fwd
            corners = np.array([
                base_centre + right + up,
                base_centre - right + up,
                base_centre - right - up,
                base_centre + right - up,
            ])
            frustum_faces = [
                [apex, corners[0], corners[1]],
                [apex, corners[1], corners[2]],
                [apex, corners[2], corners[3]],
                [apex, corners[3], corners[0]],
                [corners[0], corners[1], corners[2], corners[3]],  # back face
            ]
            frustum_coll = Poly3DCollection(
                frustum_faces, alpha=0.35, facecolor=color, edgecolor=color, linewidth=0.8,
            )
            ax_3d.add_collection3d(frustum_coll)

            # Optical axis ray toward the scene
            axis_end = pos + cr['axis'] * axis_len
            ax_3d.plot([pos[0], axis_end[0]],
                       [pos[1], axis_end[1]],
                       [pos[2], axis_end[2]],
                       color=color, linewidth=1.5, alpha=0.7)
            ax_3d.text(pos[0], pos[1], pos[2] + frustum_scale * 1.5,
                       cr['label'], color=color, fontsize=11,
                       fontweight='bold', ha='center')

        ax_3d.set_xlabel('X (mm)')
        ax_3d.set_ylabel('Y (mm)')
        ax_3d.set_zlabel('Z (mm)')
        ax_3d.set_xlim(lo[0], hi[0])
        ax_3d.set_ylim(lo[1], hi[1])
        ax_3d.set_zlim(lo[2], hi[2])
        ax_3d.set_title('3D scene (board + cameras)', fontsize=11)
        # Do NOT force equal aspect — the camera-to-board scale ratio
        # (~50:1) would make the board invisible. Let matplotlib pick
        # per-axis scaling and show the zoomed board detail separately.

        # ---- 2D projections ----
        def _draw_projection(ax, ix, iy, xlabel, ylabel, title,
                             equal_aspect=False):
            # Per-camera board dots in camera colour
            if per_cam_dots:
                for color, pts, _ in per_cam_dots:
                    ax.scatter(pts[:, ix], pts[:, iy],
                               c=color, s=10, alpha=0.75,
                               edgecolors='black', linewidths=0.2)
            else:
                ax.scatter(board_obj_points[:, ix],
                           board_obj_points[:, iy],
                           c='black', s=8, alpha=0.8)
            # board outline
            bc = np.vstack([board_corners_local, board_corners_local[:1]])
            ax.plot(bc[:, ix], bc[:, iy], color='tab:orange',
                    linewidth=1.5, alpha=0.6)
            # cameras
            for cr in cam_records:
                pos = cr['pos']
                axis = cr['axis']
                ax.plot(pos[ix], pos[iy], 'o', color=cr['color'],
                        markersize=12, markeredgecolor='black', markeredgewidth=0.8)
                ax.annotate(cr['label'],
                            xy=(pos[ix], pos[iy]),
                            xytext=(10, 10), textcoords='offset points',
                            color=cr['color'], fontsize=11, fontweight='bold')
                # Optical axis ray
                span_x = ax.get_xlim()[1] - ax.get_xlim()[0]
                span_y = ax.get_ylim()[1] - ax.get_ylim()[0]
                arrow_len = 0.18 * max(abs(span_x), abs(span_y))
                ax.annotate(
                    '', xy=(pos[ix] + axis[ix] * arrow_len,
                            pos[iy] + axis[iy] * arrow_len),
                    xytext=(pos[ix], pos[iy]),
                    arrowprops=dict(arrowstyle='->', color=cr['color'],
                                    lw=1.8, alpha=0.8),
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            if equal_aspect:
                ax.set_aspect('equal', adjustable='box')

        # Primary 2D diagnostic: top-down world X–Z.
        _draw_projection(ax_xz, 0, 2, 'X (mm)', 'Z (mm)',
                         'Top-down (X–Z): cameras + board')
        # Side view (Z–Y): shows the board as a nearly-point plus
        # camera pair tilt above/below laser sheet.
        _draw_projection(ax_zy, 2, 1, 'Z (mm)', 'Y (mm)',
                         'Side view (Z–Y)')

        # ---- Zoomed "board detail" panels (cameras omitted) ----
        # Show the actual dot layout in full detail at board scale.
        # Per-camera dots are drawn in the camera's colour so that
        # transmission geometry (cam1 front face, cam2 back face at
        # different z) is visually distinct from same-side geometry
        # (both cameras' dots coincident).
        def _draw_board_detail(ax, ix, iy, xlabel, ylabel, title,
                               split_levels=False):
            if per_cam_dots:
                for color, pts, label in per_cam_dots:
                    z_levels = np.unique(np.round(pts[:, 2], 1))
                    for z in z_levels:
                        mask = np.abs(pts[:, 2] - z) < 0.5
                        if not mask.any():
                            continue
                        # Peaks (xy_offset=0) and troughs (xy_offset=7.5)
                        # are distinguished by shape: circle vs square.
                        # xy_offset 0 dots have col*spacing exactly (int),
                        # xy_offset dots have col*spacing + 7.5 (half).
                        x0 = pts[mask, ix]
                        frac = x0 - np.round(x0)
                        # any dot whose x is closer to an integer multiple
                        # of spacing ⇒ "peak" family; else "trough" family
                        # (crude but works for our board)
                        marker = 'o'  # default
                        ax.scatter(pts[mask, ix], pts[mask, iy],
                                   c=color, s=35, alpha=0.8,
                                   edgecolors='black', linewidths=0.4,
                                   marker=marker,
                                   label=f'{label} z={z:+.0f}')
            else:
                ax.scatter(board_obj_points[:, ix],
                           board_obj_points[:, iy],
                           c='black', s=35)
            bc = np.vstack([board_corners_local, board_corners_local[:1]])
            ax.plot(bc[:, ix], bc[:, iy], color='tab:orange',
                    linewidth=1.2, alpha=0.5)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

        _draw_board_detail(ax_board, 0, 1, 'X (mm)', 'Y (mm)',
                           'Board detail: plan view (X–Y)')
        _draw_board_detail(ax_zoom3d, 0, 2, 'X (mm)', 'Z (mm)',
                           'Board detail: cross-section (X–Z)')

        # ---- Legend / metadata panel ----
        legend_lines = []
        if len(cam_records) >= 2:
            p1 = cam_records[0]['pos']
            p2 = cam_records[1]['pos']
            baseline = float(np.linalg.norm(p1 - p2))
            a1 = cam_records[0]['axis']
            a2 = cam_records[1]['axis']
            ang = np.degrees(np.arccos(np.clip(np.dot(a1, a2), -1.0, 1.0)))
            legend_lines.append(f"Cam1–Cam2 world baseline : {baseline:.1f} mm")
            legend_lines.append(f"Optical-axis angle       : {ang:.2f}°")
            legend_lines.append('')
        for cr in cam_records:
            p = cr['pos']
            legend_lines.append(
                f"{cr['label']:8s}  pos = ({p[0]:+8.1f}, {p[1]:+8.1f}, {p[2]:+8.1f}) mm"
            )
            a = cr['axis']
            legend_lines.append(
                f"          axis = ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})"
            )
        legend_lines.append('')
        legend_lines.append('Axes: X = board right, Y = board up, Z = board normal')
        legend_lines.append('(world frame of the datum-pose fiducial click)')
        ax_legend.text(0.02, 0.98, '\n'.join(legend_lines),
                       family='monospace', fontsize=9,
                       va='top', ha='left', transform=ax_legend.transAxes)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=140, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate camera placement figure: {traceback.format_exc()}")


def make_camera_placement_html(
    cam_data_list: List[Dict[str, Any]],
    output_path,
    title: str = "Camera Placement (interactive)",
):
    """Write an interactive 3D camera-placement figure as standalone HTML.

    Uses plotly.js from CDN — no Python plotly dependency required.
    The HTML file opens in any browser with a fully rotatable, pannable,
    zoomable 3D scene: cameras as pyramidal frustums, their optical
    axes as rays, and each camera's board object frame as coloured
    dots (so transmission geometry — cam1 front face at z=0/-h, cam2
    back face at z=-(T-h)/-T — is visually distinct from same-side,
    where both cameras' dots overlap).

    The file is self-contained (other than the CDN plotly.js fetch),
    so it can be copied, archived, emailed, or committed to a ticket.

    Parameters
    ----------
    cam_data_list : list of dict
        Each dict has:
            'label'       str
            'rvec'        (3,) or (3,1) datum-pose rvec
            'tvec'        (3,) or (3,1) datum-pose tvec
            'color'       any CSS colour name or #rrggbb
            'obj_points'  optional (N, 3) datum-pose 3D points in THIS
                          camera's fiducial-anchored world frame
    output_path : str or Path
        Will be written as <name>.html.
    title : str
    """
    try:
        # ---- Derive camera geometry ----
        # Optical axis is computed below from cam→board-centroid, not
        # from R.T @ [0,0,1], so that SIG's reflection convention (and
        # any other frame where the extrinsic absorbs a sign flip) is
        # handled uniformly. See the PNG-generator rationale.
        cam_records = []
        all_obj_pts: List[np.ndarray] = []
        for cam in cam_data_list:
            rvec = np.asarray(cam['rvec'], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(cam['tvec'], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            pos = (-R.T @ tvec).flatten()
            right = (R.T @ np.array([1.0, 0.0, 0.0])).flatten()
            up = (R.T @ np.array([0.0, -1.0, 0.0])).flatten()
            obj_pts = cam.get('obj_points')
            if obj_pts is not None:
                obj_pts = np.asarray(obj_pts, dtype=np.float64)
                all_obj_pts.append(obj_pts)
            cam_records.append({
                'label': cam.get('label', 'Cam'),
                'color': cam.get('color', '#1f77b4'),
                'pos': pos,
                'axis': None,
                'up': up,
                'right': right,
                'obj_points': obj_pts,
            })

        # Board centroid across all cameras' object frames (in tx
        # mode this averages the front + back faces, giving a centroid
        # between the two levels — still on the line from each camera
        # to the plate, which is what we want for the axis ray).
        if all_obj_pts:
            stacked = np.vstack(all_obj_pts)
            board_centroid = stacked.mean(axis=0)
        else:
            board_centroid = np.zeros(3)
        for cr in cam_records:
            direction = board_centroid - cr['pos']
            norm = float(np.linalg.norm(direction))
            cr['axis'] = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        # ---- Scene bounds for frustum scaling ----
        all_pts = []
        for cr in cam_records:
            all_pts.append(cr['pos'].reshape(1, 3))
            if cr['obj_points'] is not None:
                all_pts.append(cr['obj_points'])
        scene_pts = np.vstack(all_pts) if all_pts else np.zeros((1, 3))
        scene_span = float(np.max(scene_pts.max(axis=0) - scene_pts.min(axis=0)))
        if scene_span <= 0:
            scene_span = 1.0
        frustum_scale = 0.04 * scene_span
        axis_len = 0.25 * scene_span

        # ---- Build plotly traces ----
        traces: List[Dict[str, Any]] = []

        # Board dots per camera
        for cr in cam_records:
            if cr['obj_points'] is None:
                continue
            pts = np.asarray(cr['obj_points'], dtype=np.float64)
            traces.append({
                'type': 'scatter3d',
                'mode': 'markers',
                'x': pts[:, 0].tolist(),
                'y': pts[:, 1].tolist(),
                'z': pts[:, 2].tolist(),
                'marker': {
                    'size': 3.5,
                    'color': cr['color'],
                    'line': {'width': 0.5, 'color': 'black'},
                    'opacity': 0.85,
                },
                'name': f"{cr['label']} board dots",
                'hovertemplate': (
                    '<b>%{fullData.name}</b><br>'
                    'x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm'
                    '<extra></extra>'
                ),
            })

        # Per-camera: frustum (as line segments) + optical axis + point
        for cr in cam_records:
            pos = cr['pos']
            fwd = cr['axis'] * frustum_scale
            right = cr['right'] * frustum_scale * 0.8
            up = cr['up'] * frustum_scale * 0.55
            base_centre = pos + fwd
            corners = np.array([
                base_centre + right + up,
                base_centre - right + up,
                base_centre - right - up,
                base_centre + right - up,
            ])

            # Frustum edges: 4 apex→corner lines + the square base loop.
            # Plotly draws multi-segment lines by inserting None/NaN
            # between disjoint polylines.
            fx: List[Any] = []
            fy: List[Any] = []
            fz: List[Any] = []
            for c in corners:
                fx += [pos[0], c[0], None]
                fy += [pos[1], c[1], None]
                fz += [pos[2], c[2], None]
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                fx += [a[0], b[0], None]
                fy += [a[1], b[1], None]
                fz += [a[2], b[2], None]
            traces.append({
                'type': 'scatter3d',
                'mode': 'lines',
                'x': fx,
                'y': fy,
                'z': fz,
                'line': {'color': cr['color'], 'width': 4},
                'name': f"{cr['label']} frustum",
                'hoverinfo': 'skip',
                'showlegend': False,
            })

            # Optical axis
            axis_end = pos + cr['axis'] * axis_len
            traces.append({
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [pos[0], axis_end[0]],
                'y': [pos[1], axis_end[1]],
                'z': [pos[2], axis_end[2]],
                'line': {'color': cr['color'], 'width': 3, 'dash': 'dash'},
                'name': f"{cr['label']} optical axis",
                'hoverinfo': 'skip',
                'showlegend': False,
            })

            # Camera position marker with label
            traces.append({
                'type': 'scatter3d',
                'mode': 'markers+text',
                'x': [pos[0]],
                'y': [pos[1]],
                'z': [pos[2]],
                'marker': {
                    'size': 9,
                    'color': cr['color'],
                    'line': {'width': 1.5, 'color': 'black'},
                    'symbol': 'diamond',
                },
                'text': [f"<b>{cr['label']}</b>"],
                'textposition': 'top center',
                'textfont': {'size': 14, 'color': cr['color']},
                'name': cr['label'],
                'hovertemplate': (
                    f"<b>{cr['label']}</b><br>"
                    f"pos = ({pos[0]:+.0f}, {pos[1]:+.0f}, {pos[2]:+.0f}) mm<br>"
                    f"axis = ({cr['axis'][0]:+.3f}, {cr['axis'][1]:+.3f}, {cr['axis'][2]:+.3f})"
                    '<extra></extra>'
                ),
            })

        # ---- Metadata banner ----
        header_bits = []
        if len(cam_records) >= 2:
            p1, p2 = cam_records[0]['pos'], cam_records[1]['pos']
            baseline = float(np.linalg.norm(p1 - p2))
            a1, a2 = cam_records[0]['axis'], cam_records[1]['axis']
            ang = float(np.degrees(np.arccos(np.clip(float(np.dot(a1, a2)), -1.0, 1.0))))
            header_bits.append(
                f"Baseline: {baseline:.1f} mm &nbsp;•&nbsp; "
                f"Optical-axis angle: {ang:.2f}°"
            )
        for cr in cam_records:
            p = cr['pos']
            header_bits.append(
                f"<span style='color:{cr['color']}'>"
                f"{cr['label']}</span>: pos=({p[0]:+.0f},&nbsp;{p[1]:+.0f},&nbsp;{p[2]:+.0f}) mm"
            )
        header_html = ' &nbsp;•&nbsp; '.join(header_bits)

        layout = {
            'title': {'text': title, 'x': 0.5, 'xanchor': 'center'},
            'scene': {
                'xaxis': {'title': 'X (mm)', 'backgroundcolor': 'rgb(240,240,240)'},
                'yaxis': {'title': 'Y (mm)', 'backgroundcolor': 'rgb(240,240,240)'},
                'zaxis': {'title': 'Z (mm)', 'backgroundcolor': 'rgb(240,240,240)'},
                'aspectmode': 'data',
                'camera': {
                    'eye': {'x': 1.4, 'y': -1.4, 'z': 1.1},
                    'up': {'x': 0, 'y': 1, 'z': 0},
                },
            },
            'legend': {'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(255,255,255,0.85)'},
            'margin': {'l': 0, 'r': 0, 't': 50, 'b': 0},
            'paper_bgcolor': 'white',
        }

        plot_json = json.dumps(
            {'data': traces, 'layout': layout},
            default=lambda o: (None if (isinstance(o, float) and np.isnan(o)) else o),
            allow_nan=False,
            separators=(',', ':'),
        )

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }}
  #banner {{ padding: 10px 18px; background: #f7f7f7; border-bottom: 1px solid #ddd;
             font-size: 13px; color: #333; }}
  #plot {{ width: 100vw; height: calc(100vh - 55px); }}
</style>
</head>
<body>
<div id="banner">{header_html}</div>
<div id="plot"></div>
<script>
  var figure = {plot_json};
  Plotly.newPlot('plot', figure.data, figure.layout,
                 {{responsive: true, displaylogo: false}});
  window.addEventListener('resize', function() {{ Plotly.Plots.resize('plot'); }});
</script>
</body>
</html>
"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), 'w', encoding='utf-8') as f:
            f.write(html_template)
    except Exception:
        logger.warning(f"Failed to generate camera placement HTML: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 5. Stepped board detection figure (2-panel)
# ---------------------------------------------------------------------------

def make_stepped_detection_figure(
    image: np.ndarray,
    level_a: Optional[Dict[str, Any]],
    level_b: Optional[Dict[str, Any]],
    output_path,
    title: Optional[str] = None,
    blob_info: Optional[Dict[str, Any]] = None,
    peak_level: Optional[str] = None,
    trough_level: Optional[str] = None,
    fiducials: Optional[Dict[str, Any]] = None,
):
    """Generate and save a 3-panel stepped board detection figure.

    Panels:
        1. Blob detection summary (kept blob centres overlaid on CLAHE
           image, plus fiducial crosshairs when the datum pose supplies
           them). The title reports the per-polarity blob counts from
           `detect_dotboard_blobs` so the operator can see whether the
           detector picked the right polarity.
        2. Final grid: Peak (blue) + Trough (red) with per-level grid
           network lines. Colours are assigned by *physical* peak/trough
           via `peak_level` / `trough_level`, not by the arbitrary Level
           A / Level B row-parity naming (which can flip per pose).
        3. Both levels overlaid on CLAHE image, same colour convention.

    Parameters
    ----------
    image : np.ndarray
        Raw input image.
    level_a : dict or None
        Level A detection: {'centers': (N,2), 'grid_indices': (N,2)}.
    level_b : dict or None
        Level B detection: {'centers': (N,2), 'grid_indices': (N,2)}.
    output_path : str or Path
    title : str, optional
    blob_info : dict, optional
        Diagnostic dict from `detect_dotboard_blobs`. Reads
        `n_blobs_detected`, `image_mode`, and `_polarity_results`.
    peak_level : {'A', 'B', None}
        Which of `level_a` / `level_b` is the physical PEAK. When None,
        the figure falls back to Level A / Level B naming (legacy).
    trough_level : {'A', 'B', None}
        The other level. Must be consistent with `peak_level`.
    fiducials : dict or None
        `{'origin': [x, y], 'x_axis': [x, y], 'y_axis': [x, y]}` drawn
        as crosshairs + axis arrows on Panel 1. Only supply for the
        datum pose — non-datum poses should pass None.
    """
    try:
        n_a = 0
        n_b = 0
        if level_a is not None and 'centers' in level_a:
            n_a = len(np.array(level_a['centers']))
        if level_b is not None and 'centers' in level_b:
            n_b = len(np.array(level_b['centers']))

        # Resolve peak/trough display mapping.
        if peak_level in ('A', 'B') and trough_level in ('A', 'B') and peak_level != trough_level:
            level_peak_data = level_a if peak_level == 'A' else level_b
            level_trough_data = level_a if trough_level == 'A' else level_b
            peak_label_text = 'Peak'
            trough_label_text = 'Trough'
        else:
            # Legacy fallback when caller doesn't know the resolution.
            level_peak_data = level_a
            level_trough_data = level_b
            peak_label_text = 'Level A'
            trough_label_text = 'Level B'

        n_peak = len(np.array(level_peak_data['centers'])) if (
            level_peak_data is not None and 'centers' in level_peak_data
        ) else 0
        n_trough = len(np.array(level_trough_data['centers'])) if (
            level_trough_data is not None and 'centers' in level_trough_data
        ) else 0

        fig = plt.figure(figsize=(20, 7))
        status = f'{n_a + n_b} points ({peak_label_text}={n_peak}, {trough_label_text}={n_trough})'
        fig.suptitle(f'{title or "Stepped Board Detection"}  —  {status}', fontsize=13,
                     color='darkgreen' if (n_a + n_b) > 0 else 'darkred')
        gs = GridSpec(1, 3, figure=fig, wspace=0.10)

        clahe_img = _to_uint8_gray(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(clahe_img)
        scale = max(1, clahe_img.shape[1] // 1200)

        # Real blob_info contract (from `detect_dotboard_blobs`):
        # n_blobs_detected, image_mode ('original'|'inverted'), and
        # _polarity_results = [{'centers', 'n_blobs', 'invert', ...}].
        # The legacy figure read `all_keypoints`/`noise_blobs_filtered`
        # which were never populated (reported "0 kept, 0 filtered").
        n_blobs = blob_info.get('n_blobs_detected', 0) if blob_info else 0
        polarity_mode = blob_info.get('image_mode', 'unknown') if blob_info else 'unknown'
        polarity_results = blob_info.get('_polarity_results', []) if blob_info else []
        candidate_bits = []
        for p in polarity_results:
            p_inv = p.get('invert', False)
            p_n = int(p.get('n_blobs', 0))
            candidate_bits.append(f"{'inverted' if p_inv else 'original'}={p_n}")
        candidates_str = ' | '.join(candidate_bits) if candidate_bits else 'n/a'

        # Panel 1: Blob detection summary — kept blob centers overlaid
        # on CLAHE gray, plus fiducial crosshairs for the datum pose.
        ax1 = fig.add_subplot(gs[0, 0])
        disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        # Overlay kept blob centers (everything that made it into
        # level_a OR level_b) as small green circles — shows the actual
        # detector output, not a stale pre-filter keypoint list.
        for level_data in (level_a, level_b):
            if level_data is None or 'centers' not in level_data:
                continue
            centers_np = np.array(level_data['centers'])
            for pt in centers_np:
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), 2)
        ax1.imshow(disp[::scale, ::scale])
        ax1.set_title(
            f'Blob Detection: {n_blobs} kept ({polarity_mode})  —  '
            f'polarity candidates: {candidates_str}',
            fontsize=9,
        )
        ax1.set_xticks([]); ax1.set_yticks([])

        # Fiducial overlay — only when the caller supplies fiducials
        # (i.e. the datum pose). Show origin, +X, +Y labels and the
        # direction arrows so the operator can confirm the board
        # orientation convention at a glance.
        if fiducials is not None:
            try:
                origin = np.asarray(fiducials.get('origin'), dtype=np.float64) / scale
                x_axis = np.asarray(fiducials.get('x_axis'), dtype=np.float64) / scale
                y_axis = np.asarray(fiducials.get('y_axis'), dtype=np.float64) / scale

                ax1.annotate(
                    '', xy=x_axis, xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', color='yellow', lw=2),
                )
                ax1.annotate(
                    '', xy=y_axis, xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', color='magenta', lw=2),
                )
                ax1.scatter(
                    [origin[0]], [origin[1]],
                    s=100, facecolor='none', edgecolor='lime', linewidths=2,
                    marker='o', zorder=10,
                )
                ax1.scatter(
                    [origin[0]], [origin[1]],
                    s=30, color='lime', marker='+', zorder=11,
                )
                ax1.text(origin[0] + 5, origin[1] - 5, 'O',
                         color='lime', fontsize=10, fontweight='bold')
                ax1.text(x_axis[0] + 5, x_axis[1] - 5, '+X',
                         color='yellow', fontsize=10, fontweight='bold')
                ax1.text(y_axis[0] + 5, y_axis[1] - 5, '+Y',
                         color='magenta', fontsize=10, fontweight='bold')
            except Exception:
                logger.debug(
                    f"Fiducial overlay failed: {traceback.format_exc()}"
                )

        # Panel 2: Final grid network — Peak=royalblue, Trough=crimson,
        # with each level drawn as its OWN network (no cross-level edges,
        # no (col,row) collisions — peaks and troughs live on distinct
        # sub-lattices).
        ax2 = fig.add_subplot(gs[0, 1])
        for level_data, color, lbl in [
            (level_peak_data, 'royalblue', peak_label_text),
            (level_trough_data, 'crimson', trough_label_text),
        ]:
            if level_data is None or 'centers' not in level_data:
                continue
            centers = np.array(level_data['centers'])
            if len(centers) == 0:
                continue
            ax2.scatter(centers[:, 0], centers[:, 1], c=color, s=10, zorder=5,
                        label=f'{lbl} ({len(centers)})')
            if 'grid_indices' in level_data:
                gi = np.array(level_data['grid_indices'])
                if len(gi) > 0:
                    _draw_grid_network(ax2, centers, gi, color=color, lw=0.6, alpha=0.7)
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.legend(fontsize=8)
        ax2.set_title('Final Grid (per-level networks)', fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Panel 3: Grid on image — same colour convention
        ax3 = fig.add_subplot(gs[0, 2])
        result_disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        # OpenCV is BGR: peak (royalblue-ish) = (225, 105, 65), trough
        # (crimson-ish) = (60, 20, 220). These are picked to match the
        # matplotlib colours above.
        for level_data, cv_color in [
            (level_peak_data, (225, 105, 65)),
            (level_trough_data, (60, 20, 220)),
        ]:
            if level_data is None or 'centers' not in level_data:
                continue
            centers = np.array(level_data['centers'])
            if len(centers) == 0:
                continue
            for pt in centers:
                cv2.circle(result_disp, (int(pt[0]), int(pt[1])), 8, cv_color, 3)
            if 'grid_indices' in level_data:
                gi = np.array(level_data['grid_indices'])
                if len(gi) > 0:
                    _draw_grid_network_cv(result_disp, centers, gi, color=cv_color, thickness=1)
        ax3.imshow(result_disp[::scale, ::scale])
        ax3.set_title('Grid on Image', fontsize=10)
        ax3.set_xticks([]); ax3.set_yticks([])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate stepped detection figure: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 6. Stepped board reprojection error figure (per-camera scatter with RMS circle)
# ---------------------------------------------------------------------------

def make_stepped_reprojection_figure(
    cam_results: Dict[int, Dict[str, Any]],
    output_path,
):
    """Generate per-camera reprojection error scatter plot.

    Supports both single-pose and multi-pose data. For multi-pose,
    errors are pooled across all poses and colored by pose index so
    one outlier pose is visible at a glance. For single-pose, errors
    are colored by Z-plane as before.

    Parameters
    ----------
    cam_results : dict
        {cam_num: {K, dist, rms, ...}} with either:
          Single-pose form:
            obj_points, img_points, rvec, tvec
          Multi-pose form:
            obj_views_per_pose: list of (M_i, 3) arrays
            img_views_per_pose: list of (M_i, 2) arrays
            rvecs_all: list of (3, 1) per pose, datum at index 0
            tvecs_all: list of (3, 1) per pose, datum at index 0
            pose_indices: list of int (0-based pose sequence IDs)
        If both forms are present, multi-pose is used.
    output_path : str or Path
    """
    try:
        n_cams = len(cam_results)
        fig, axes = plt.subplots(1, n_cams, figsize=(10 * n_cams, 8))
        if n_cams == 1:
            axes = [axes]

        for ax, (cam_num, r) in zip(axes, cam_results.items()):
            K = r['K']
            dist = r['dist']
            rms_val = r['rms']

            obj_views = r.get('obj_views_per_pose')
            img_views = r.get('img_views_per_pose')
            rvecs_all = r.get('rvecs_all')
            tvecs_all = r.get('tvecs_all')
            pose_indices = r.get('pose_indices')
            is_multi = (
                obj_views is not None and img_views is not None
                and rvecs_all is not None and tvecs_all is not None
                and len(obj_views) > 1
            )

            if not is_multi:
                # --- Single-pose path: original Z-plane coloring ---
                obj = r['obj_points']
                img = r['img_points']
                rvec = r['rvec']
                tvec = r['tvec']
                projected, _ = cv2.projectPoints(
                    obj.astype(np.float64),
                    rvec.astype(np.float64),
                    tvec.astype(np.float64),
                    K, dist,
                )
                errors = img.reshape(-1, 2) - projected.reshape(-1, 2)
                z_vals = obj[:, 2]
                unique_z = np.unique(np.round(z_vals, 2))
                for z in unique_z:
                    mask = np.abs(z_vals - z) < 0.5
                    ax.scatter(errors[mask, 0], errors[mask, 1], s=4, alpha=0.6,
                               label=f'Z={z:.1f}mm ({mask.sum()} pts)')
            else:
                # --- Multi-pose path: per-pose coloring ---
                n_poses = len(obj_views)
                if pose_indices is None or len(pose_indices) != n_poses:
                    pose_indices = list(range(n_poses))
                cmap = plt.get_cmap('tab10' if n_poses <= 10 else 'tab20')
                all_errs = []
                for i, (obj, img_pts, rv, tv, pose_id) in enumerate(zip(
                    obj_views, img_views, rvecs_all, tvecs_all, pose_indices,
                )):
                    obj = np.asarray(obj, dtype=np.float64)
                    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
                    projected, _ = cv2.projectPoints(
                        obj,
                        np.asarray(rv, dtype=np.float64),
                        np.asarray(tv, dtype=np.float64),
                        K, dist,
                    )
                    errors = img_pts - projected.reshape(-1, 2)
                    all_errs.append(errors)
                    pose_rms = float(np.sqrt(np.mean(errors ** 2)))
                    label = (f'pose {pose_id} (n={len(errors)}, '
                             f'rms={pose_rms:.3f}px)')
                    if i == 0:
                        label = 'DATUM ' + label
                    ax.scatter(
                        errors[:, 0], errors[:, 1],
                        s=8, alpha=0.7,
                        color=cmap(i % cmap.N),
                        label=label,
                    )
                if all_errs:
                    err_stack = np.concatenate(all_errs, axis=0)
                    # Axes span: 1.5x the 99th-percentile error magnitude,
                    # or 3x rms, whichever is larger — makes outliers visible
                    mags = np.linalg.norm(err_stack, axis=1)
                    span = max(3 * rms_val, 1.5 * float(np.percentile(mags, 99))) + 0.5

            if is_multi:
                # multi-pose span computed above
                pass
            else:
                span = 3 * rms_val + 0.5

            # RMS circle (full-fit rms across all poses)
            circle = plt.Circle((0, 0), rms_val, fill=False, color='red',
                                linewidth=1.5, linestyle='--',
                                label=f'total RMS={rms_val:.3f}px')
            ax.add_patch(circle)
            ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlim(-span, span)
            ax.set_ylim(-span, span)
            ax.set_aspect('equal')
            ax.set_xlabel('x error (px)')
            ax.set_ylabel('y error (px)')
            pose_suffix = f' ({len(obj_views)} poses)' if is_multi else ''
            ax.set_title(f'Cam {cam_num}: RMS={rms_val:.3f}px{pose_suffix}')
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle('Reprojection Errors (per calibration dot)\n'
                     'Red dashed circle = total fit RMS radius',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate stepped reprojection figure: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 7. Dewarp overlay pair figure (red-cyan stereo verification)
# ---------------------------------------------------------------------------

def make_dewarp_overlay_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    pr1: Dict[str, Any],
    pr2: Dict[str, Any],
    cam1_obj: np.ndarray,
    cam2_obj: np.ndarray,
    output_path,
    title: Optional[str] = None,
    mm_per_px: float = 0.1,
    z1: Optional[float] = None,
    z2: Optional[float] = None,
    cam1_level_obj: Optional[np.ndarray] = None,
    cam2_level_obj: Optional[np.ndarray] = None,
    dot_spacing_mm: float = 15.0,
    grid_offset: float = 0.0,
):
    """Generate a red-cyan dewarp overlay figure for stereo verification.

    Dewarps both camera images to a common world plane and overlays them
    as red-cyan. Matching the style in manual_tools/stepped_calibration_board/.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Grayscale images from camera 1 and 2.
    pr1, pr2 : dict
        Per-camera results: {K, dist, rvec, tvec}.
    cam1_obj, cam2_obj : np.ndarray
        3D world points for cam1 and cam2 (N,3). Used for world bounds.
    output_path : str or Path
    title : str, optional
    mm_per_px : float
        Resolution of the dewarped image (mm per pixel).
    z1, z2 : float, optional
        Per-camera Z-planes for dewarping. If None, uses average of all points.
    cam1_level_obj, cam2_level_obj : np.ndarray, optional
        Per-level object points for the specific Z being dewarped.
        If None, falls back to cam1_obj/cam2_obj for dot markers.
    dot_spacing_mm : float
        Dot grid spacing (mm) for axis ticks.
    grid_offset : float
        XY offset for grid tick lines (mm).
    """
    try:
        # Compute common world bounds from calibration points
        all_world = np.vstack([cam1_obj[:, :2], cam2_obj[:, :2]])
        x_min = float(all_world[:, 0].min()) - 5
        x_max = float(all_world[:, 0].max()) + 5
        y_min = float(all_world[:, 1].min()) - 5
        y_max = float(all_world[:, 1].max()) + 5

        nx = int((x_max - x_min) / mm_per_px)
        ny = int((y_max - y_min) / mm_per_px)
        x_1d = np.linspace(x_min, x_max, nx)
        y_1d = np.linspace(y_min, y_max, ny)
        X_grid, Y_grid = np.meshgrid(x_1d, y_1d)

        # Determine Z for each camera's dewarp plane
        if z1 is None:
            z1 = float(np.mean(np.concatenate([cam1_obj[:, 2], cam2_obj[:, 2]])))
        if z2 is None:
            z2 = z1

        def build_dewarp_maps(pr, z_mm):
            Z_grid = np.full_like(X_grid, z_mm)
            world_pts = np.column_stack([X_grid.ravel(), Y_grid.ravel(),
                                         Z_grid.ravel()]).astype(np.float64)
            projected, _ = cv2.projectPoints(world_pts, pr['rvec'], pr['tvec'],
                                              pr['K'], pr['dist'])
            projected = projected.reshape(-1, 2)
            map_x = projected[:, 0].reshape(X_grid.shape).astype(np.float32)
            map_y = projected[:, 1].reshape(Y_grid.shape).astype(np.float32)
            return map_x, map_y

        def normalize_u8(im):
            lo, hi = float(im.min()), float(im.max())
            if hi - lo < 1e-6:
                return np.zeros(im.shape, dtype=np.uint8)
            return ((im - lo) / (hi - lo) * 255).astype(np.uint8)

        img1_f = img1.astype(np.float64) if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        img2_f = img2.astype(np.float64) if img2.ndim == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

        map1_x, map1_y = build_dewarp_maps(pr1, z1)
        map2_x, map2_y = build_dewarp_maps(pr2, z2)

        dw1 = cv2.remap(img1_f, map1_x, map1_y, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dw2 = cv2.remap(img2_f, map2_x, map2_y, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        h_img1, w_img1 = img1_f.shape[:2]
        h_img2, w_img2 = img2_f.shape[:2]
        valid1 = (map1_x >= 0) & (map1_x < w_img1) & (map1_y >= 0) & (map1_y < h_img1)
        valid2 = (map2_x >= 0) & (map2_x < w_img2) & (map2_y >= 0) & (map2_y < h_img2)
        dw1[~valid1] = 0
        dw2[~valid2] = 0

        r_ch = normalize_u8(dw1)
        c_ch = normalize_u8(dw2)
        r_ch[~valid1] = 0
        c_ch[~valid2] = 0
        overlay = np.stack([r_ch, c_ch, c_ch], axis=-1)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin='lower')

        # Grid lines at dot spacing, offset for the level being displayed
        ax.axhline(0, color='lime', linewidth=1, alpha=0.6)
        ax.axvline(0, color='lime', linewidth=1, alpha=0.6)
        ax.scatter(0, 0, s=300, c='lime', marker='+', linewidths=3, zorder=20)
        ax.set_xticks(np.arange(
            np.ceil((x_min - grid_offset) / dot_spacing_mm) * dot_spacing_mm + grid_offset,
            x_max, dot_spacing_mm))
        ax.set_yticks(np.arange(
            np.ceil((y_min - grid_offset) / dot_spacing_mm) * dot_spacing_mm + grid_offset,
            y_max, dot_spacing_mm))
        ax.grid(True, color='white', alpha=0.15, linewidth=0.5)

        # Cal dot markers for the specific levels being dewarped
        c1_pts = cam1_level_obj if cam1_level_obj is not None else cam1_obj
        c2_pts = cam2_level_obj if cam2_level_obj is not None else cam2_obj
        ax.scatter(c1_pts[:, 0], c1_pts[:, 1], s=25, facecolors='none',
                   edgecolors='white', linewidths=0.6, marker='s', alpha=0.7,
                   zorder=6, label=f'Cam1 ({len(c1_pts)})')
        ax.scatter(c2_pts[:, 0], c2_pts[:, 1], s=25, facecolors='none',
                   edgecolors='lime', linewidths=0.6, marker='D', alpha=0.7,
                   zorder=6, label=f'Cam2 ({len(c2_pts)})')

        ax.set_xlabel('X world (mm)')
        ax.set_ylabel('Y world (mm)')
        ax.set_title(title or f'Dewarp Overlay (Cam1=red Z={z1:.1f}mm, Cam2=cyan Z={z2:.1f}mm)',
                     fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate dewarp overlay figure: {traceback.format_exc()}")
