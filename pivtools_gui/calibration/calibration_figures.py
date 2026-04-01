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
        from pivtools_gui.calibration.grid_detection import to_grayscale_2d

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
):
    """Generate and save a 2-panel camera placement figure.

    Parameters
    ----------
    cam_data_list : list of dict
        Each dict has 'label', 'rvec', 'tvec', 'color'.
        Camera world position = -R^T @ t.
    output_path : str or Path
    """
    try:
        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Camera Placement', fontsize=13)

        for cam in cam_data_list:
            R, _ = cv2.Rodrigues(np.array(cam['rvec']).reshape(3, 1))
            t = np.array(cam['tvec']).reshape(3, 1)
            pos = (-R.T @ t).flatten()  # World position
            fwd = R[2, :]  # Viewing direction (z-axis of camera)

            color = cam.get('color', 'blue')
            label = cam.get('label', '')
            arrow_len = 50  # mm

            # Top-down: X-Z plane
            ax_top.plot(pos[0], pos[2], 'o', color=color, markersize=10)
            ax_top.annotate(label, (pos[0], pos[2]), fontsize=9, ha='center', va='bottom')
            ax_top.arrow(pos[0], pos[2], fwd[0] * arrow_len, fwd[2] * arrow_len,
                         head_width=5, head_length=3, fc=color, ec=color, alpha=0.7)

            # Side: Y-Z plane
            ax_side.plot(pos[2], pos[1], 'o', color=color, markersize=10)
            ax_side.annotate(label, (pos[2], pos[1]), fontsize=9, ha='center', va='bottom')
            ax_side.arrow(pos[2], pos[1], fwd[2] * arrow_len, fwd[1] * arrow_len,
                          head_width=5, head_length=3, fc=color, ec=color, alpha=0.7)

        # Board at Z=0
        ax_top.axhline(0, color='gray', lw=2, alpha=0.5, label='Board (Z=0)')
        ax_side.axvline(0, color='gray', lw=2, alpha=0.5, label='Board (Z=0)')

        ax_top.set_xlabel('X (mm)'); ax_top.set_ylabel('Z (mm)')
        ax_top.set_title('Top-Down View (X-Z)', fontsize=10)
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_side.set_xlabel('Z (mm)'); ax_side.set_ylabel('Y (mm)')
        ax_side.set_title('Side View (Z-Y)', fontsize=10)
        ax_side.legend(fontsize=8)
        ax_side.grid(True, alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        logger.warning(f"Failed to generate camera placement figure: {traceback.format_exc()}")


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
):
    """Generate and save a 3-panel stepped board detection figure.

    Panels:
        1. Blob detection overlay (green=kept, red=filtered from blob_info)
        2. Final grid: Level A (blue) + Level B (red) with network lines
        3. Both levels overlaid on CLAHE image

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
        Diagnostic info from detect_dotboard_blobs() with 'all_keypoints',
        'noise_threshold', 'noise_blobs_filtered', 'gray_image'.
    """
    try:
        from pivtools_gui.calibration.grid_detection import to_grayscale_2d

        n_a = 0
        n_b = 0
        if level_a is not None and 'centers' in level_a:
            n_a = len(np.array(level_a['centers']))
        if level_b is not None and 'centers' in level_b:
            n_b = len(np.array(level_b['centers']))

        fig = plt.figure(figsize=(20, 7))
        status = f'{n_a + n_b} points (A={n_a}, B={n_b})'
        fig.suptitle(f'{title or "Stepped Board Detection"}  —  {status}', fontsize=13,
                     color='darkgreen' if (n_a + n_b) > 0 else 'darkred')
        gs = GridSpec(1, 3, figure=fig, wspace=0.10)

        clahe_img = None
        if blob_info is not None:
            clahe_img = blob_info.get('gray_image')
        if clahe_img is None:
            clahe_img = _to_uint8_gray(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(clahe_img)
        scale = max(1, clahe_img.shape[1] // 1200)
        noise_thresh = blob_info.get('noise_threshold', 0) if blob_info else 0

        # Panel 1: Blob detection (green=kept, red=filtered)
        ax1 = fig.add_subplot(gs[0, 0])
        disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        if blob_info is not None:
            for (pt, sz) in blob_info.get('all_keypoints', []):
                color = (255, 80, 80) if sz < noise_thresh else (0, 255, 0)
                cv2.circle(disp, (int(pt[0]), int(pt[1])), max(int(sz / 2), 3), color, 2)
        n_raw = len(blob_info.get('all_keypoints', [])) if blob_info else 0
        n_noise = blob_info.get('noise_blobs_filtered', 0) if blob_info else 0
        n_kept = n_raw - n_noise
        ax1.imshow(disp[::scale, ::scale])
        ax1.set_title(f'Blob Detection: {n_kept} kept, {n_noise} filtered', fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Panel 2: Final grid network (Level A=blue, Level B=red)
        ax2 = fig.add_subplot(gs[0, 1])
        for level_data, color, lbl in [
            (level_a, 'royalblue', 'Level A'),
            (level_b, 'crimson', 'Level B'),
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
        ax2.legend(fontsize=8)
        ax2.set_title('Final Grid', fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Panel 3: Grid on image
        ax3 = fig.add_subplot(gs[0, 2])
        result_disp = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        for level_data, cv_color in [
            (level_a, (255, 100, 100)),  # blue-ish for Level A
            (level_b, (100, 100, 255)),  # red-ish for Level B
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

    Matching the style in manual_tools/stepped_calibration_board/stepped_pinhole_calibration.py.
    Points colored by Z-plane, RMS circle overlay.

    Parameters
    ----------
    cam_results : dict
        {cam_num: {K, dist, rvec, tvec, rms, obj_points, img_points}}
    output_path : str or Path
    """
    try:
        n_cams = len(cam_results)
        fig, axes = plt.subplots(1, n_cams, figsize=(10 * n_cams, 8))
        if n_cams == 1:
            axes = [axes]

        for ax, (cam_num, r) in zip(axes, cam_results.items()):
            obj = r['obj_points']
            img = r['img_points']
            K = r['K']
            dist = r['dist']
            rvec = r['rvec']
            tvec = r['tvec']
            rms_val = r['rms']

            # Compute per-point reprojection errors
            projected, _ = cv2.projectPoints(
                obj.astype(np.float64),
                rvec.astype(np.float64),
                tvec.astype(np.float64),
                K, dist,
            )
            errors = (img.reshape(-1, 2) - projected.reshape(-1, 2))

            # Color by Z-plane
            z_vals = obj[:, 2]
            unique_z = np.unique(np.round(z_vals, 2))
            for z in unique_z:
                mask = np.abs(z_vals - z) < 0.5
                ax.scatter(errors[mask, 0], errors[mask, 1], s=4, alpha=0.6,
                           label=f'Z={z:.1f}mm ({mask.sum()} pts)')

            # RMS circle
            circle = plt.Circle((0, 0), rms_val, fill=False, color='red',
                                linewidth=1.5, linestyle='--', label=f'RMS={rms_val:.3f}px')
            ax.add_patch(circle)
            ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlim(-3 * rms_val - 0.5, 3 * rms_val + 0.5)
            ax.set_ylim(-3 * rms_val - 0.5, 3 * rms_val + 0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('x error (px)')
            ax.set_ylabel('y error (px)')
            ax.set_title(f'Cam {cam_num}: RMS={rms_val:.3f}px')
            ax.legend(fontsize=8)

        fig.suptitle('Reprojection Errors (per calibration dot)\n'
                     'Red dashed circle = RMS radius', fontsize=13, fontweight='bold')
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
