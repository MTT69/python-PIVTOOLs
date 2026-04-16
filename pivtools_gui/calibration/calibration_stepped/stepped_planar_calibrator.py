"""Stepped board as per-camera 3D calibration target.

This calibrator uses BOTH Z levels of a stepped-dot board to fit a
single-camera pinhole model. Each pose's dots at two Z planes provide
non-coplanar 3D object points, giving `cv2.calibrateCamera` real depth
information — this breaks the fx↔tz depth-focal ridge at PIV
magnification without needing a stereo pair.

The previous stepped-planar path threw away one level per frame to keep
the points coplanar, halving the data and surrendering the stepped
board's depth signal. This version keeps both levels and uses them as
genuine non-coplanar calibration points.

Architecture: subclass `SteppedCalibrator` and reuse its per-camera
pinhole fit helper (`_fit_per_camera_pinhole`) + save routine. The
stereo-pair state on the parent class (`camera_pair`) is unused in this
subclass — each `generate_camera_model` call handles ONE camera.

Output: `{base}/calibration/Cam{N}/stepped_planar/model/camera_model.mat`
with the same schema as `SteppedCalibrator` (shared by
`VectorCalibrator` and global alignment) — downstream consumers work
unchanged.
"""
from typing import Optional

from loguru import logger

from pivtools_gui.calibration.calibration_stepped.stepped_calibration_production import (
    SteppedCalibrator,
    assign_absolute_grid_indices,
    compute_z_and_offsets,
)


class SteppedPlanarCalibrator(SteppedCalibrator):
    """Per-camera 3D stepped board calibration (single camera).

    Inherits `__init__`, `detect_single_camera`, `_build_non_datum_pose_view`,
    `_fit_per_camera_pinhole`, and `_save_per_camera_model` from
    `SteppedCalibrator`. The parent's `camera_pair` attribute is unused
    — each `generate_camera_model` call takes the camera number as an
    argument.

    The sequence of detections is driven by the views layer, which
    loops over frames and calls `detect_single_camera(cam, frame)` to
    assemble the per-pose detection list. This mirrors how
    `stepped_board_views` drives sequence detection for the stereo
    calibrator.
    """

    _subdir_name = "stepped_planar"

    def generate_camera_model(
        self,
        cam_num: int,
        detections_per_pose: list,
        fiducials_for_camera: dict,
        clicked_level: str,
        frame_indices: list,
        pose_levels: dict,
        datum_pose_index: int = 0,
        progress_callback=None,
    ) -> dict:
        """Fit a per-camera pinhole model from multi-pose stepped
        board detections.

        Parameters
        ----------
        cam_num : int
            Camera number (1-based).
        detections_per_pose : list of dicts
            One entry per pose, each a dict keyed by `str(cam_num)` with
            the value returned by `detect_single_camera`. Matches the
            shape used by `SteppedCalibrator.generate_model`.
        fiducials_for_camera : dict
            `{'origin', 'x_axis', 'y_axis'}` with `snapped_px` fields.
            The operator clicks these on the datum pose of THIS camera.
        clicked_level : {'peak', 'trough'}
            Which physical face the origin click lands on. Declared by
            the operator via a radio button. This resolves the A/B →
            peak/trough mapping for the datum pose only.
        frame_indices : list[int]
            The 1-based frame number for each entry in
            `detections_per_pose`. Used to map `pose_levels` onto
            position indices. Length must match `detections_per_pose`.
        pose_levels : dict[int, str]
            Per-pose peak/trough label for THIS camera, keyed by
            frame_idx. Each value is 'peak' or 'trough' = the label
            for that pose's level A sub-lattice (level B gets the
            opposite). Required — no auto-detection fallback.
        datum_pose_index : int
            0-based index of the datum pose in `detections_per_pose`.
        progress_callback : callable, optional
            Receives `{'progress': pct, 'stage': str}`.

        Returns
        -------
        dict
            `{'success': True, 'rms', 'K', 'dist', 'num_poses',
              'pose_indices', 'model_path'}` on success, or
            `{'success': False, 'error': str}` on failure.
        """
        def _progress(pct, stage):
            if progress_callback:
                progress_callback({'progress': pct, 'stage': stage})

        if not detections_per_pose:
            return {'success': False, 'error': 'detections_per_pose is empty'}
        num_poses = len(detections_per_pose)
        if not (0 <= datum_pose_index < num_poses):
            return {
                'success': False,
                'error': (
                    f'datum_pose_index {datum_pose_index} out of range '
                    f'[0, {num_poses})'
                ),
            }
        if clicked_level not in ('peak', 'trough'):
            return {
                'success': False,
                'error': (
                    f"clicked_level must be 'peak' or 'trough', "
                    f"got {clicked_level!r}"
                ),
            }
        if len(frame_indices) != num_poses:
            return {
                'success': False,
                'error': (
                    f'frame_indices has length {len(frame_indices)} but '
                    f'detections_per_pose has {num_poses} entries'
                ),
            }
        if pose_levels is None:
            return {
                'success': False,
                'error': (
                    f'Camera {cam_num}: pose_levels is required. Each '
                    f"frame in frame_indices must have an explicit 'peak' "
                    f"or 'trough' label — no auto-detect fallback."
                ),
            }
        try:
            pose_levels_int = {int(k): v for k, v in pose_levels.items()}
        except (TypeError, ValueError) as exc:
            return {
                'success': False,
                'error': f'Camera {cam_num}: pose_levels has non-integer key: {exc}',
            }
        pose_labels_by_position = []
        for f in frame_indices:
            if f not in pose_levels_int:
                return {
                    'success': False,
                    'error': (
                        f'Camera {cam_num}: pose_levels missing frame_idx={f}. '
                        f"Every frame needs an explicit 'peak' or 'trough' label."
                    ),
                }
            val = pose_levels_int[f]
            if val not in ('peak', 'trough'):
                return {
                    'success': False,
                    'error': (
                        f'Camera {cam_num}: pose_levels[{f}]={val!r}, '
                        f"expected 'peak' or 'trough'."
                    ),
                }
            pose_labels_by_position.append(val)

        _progress(5, 'assigning_grid_indices')

        datum_pose = detections_per_pose[datum_pose_index]
        det = datum_pose.get(str(cam_num))
        if det is None:
            return {
                'success': False,
                'error': (
                    f'Camera {cam_num}: no detection for datum pose '
                    f'{datum_pose_index}'
                ),
            }
        level_A = det.get('_level_A_full')
        level_B = det.get('_level_B_full')
        if level_A is None and level_B is None:
            return {
                'success': False,
                'error': f'Camera {cam_num}: no grid detected on datum pose',
            }

        absolute_datum = assign_absolute_grid_indices(
            fiducials_for_camera, level_A, level_B, clicked_level, self.board,
        )

        _progress(15, 'computing_geometry')

        # Single-camera stepped planar: only `same_side` geometry is
        # meaningful. The clicked face becomes the reference Z (via
        # `compute_z_and_offsets`), the other face is offset by
        # step_height. Both cameras in the returned geo dict carry the
        # same Z information in `same_side` mode, so either works.
        other_level = 'trough' if clicked_level == 'peak' else 'peak'
        geo = compute_z_and_offsets(
            'same_side', clicked_level, other_level, self.board,
        )
        geo_cam = geo['Cam1']

        image_size = det['image_size']  # [H, W]

        _progress(30, f'fitting_cam{cam_num}')

        result = self._fit_per_camera_pinhole(
            cam_num=cam_num,
            detections_per_pose=detections_per_pose,
            absolute_datum=absolute_datum,
            geo_cam=geo_cam,
            image_size=image_size,
            clicked_level=clicked_level,
            datum_pose_index=datum_pose_index,
            pose_level_labels_by_position=pose_labels_by_position,
        )
        if not result['success']:
            return result

        _progress(85, 'saving_model')

        pr = {
            'rms': result['rms'],
            'K': result['K'],
            'dist': result['dist'],
            'rvec': result['rvec'],
            'tvec': result['tvec'],
            'rvecs_all': result['rvecs_all'],
            'tvecs_all': result['tvecs_all'],
            'pose_indices': result['pose_indices'],
            'image_size': result['image_size'],
            'obj_points': result['obj_points'],
        }
        model_path = self._save_per_camera_model(cam_num, pr)

        # Clear any stale self-calibration file that referenced a
        # previous stepped_planar model for this camera. Stepped planar
        # doesn't produce stereo output, so there's no stereo_cam{A}_cam{B}
        # directory to clean up here — unlike SteppedCalibrator.
        # (Self-calibration currently only runs on stereo models.)

        if self._config is not None:
            try:
                snapshot_path = self._config.save_calibration_snapshot(self.base_dir)
                logger.debug(f"Calibration snapshot saved: {snapshot_path}")
            except Exception as e:
                logger.warning(f"Failed to save calibration snapshot: {e}")

        _progress(100, 'complete')

        return {
            'success': True,
            'cam_num': int(cam_num),
            'rms': float(result['rms']),
            'K': result['K'].tolist(),
            'dist': result['dist'].flatten().tolist(),
            'num_poses': len(result['obj_views_per_pose']),
            'pose_indices': [int(i) for i in result['pose_indices']],
            'model_path': str(model_path),
        }
