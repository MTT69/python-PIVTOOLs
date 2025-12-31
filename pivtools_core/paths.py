from pathlib import Path
from typing import Optional, Tuple, Union


def get_data_paths(
    base_dir,
    num_frame_pairs,
    cam,
    type_name,
    endpoint="",
    use_merged=False,
    use_uncalibrated=False,
    use_stereo=False,
    stereo_camera_pair: Optional[Tuple[int, int]] = None,
    calibration=False,
):
    """
    Construct directories for data, statistics, and videos.

    Args:
        base_dir: Base directory path
        num_frame_pairs: Number of frame pairs
        cam: Camera number (ignored for stereo, use stereo_camera_pair instead)
        type_name: Type name (e.g., "instantaneous", "ensemble")
        endpoint: Optional subfolder ('' ignored)
        use_merged: If True, return paths for merged data
        use_uncalibrated: If True, return paths for uncalibrated data
        use_stereo: If True, return paths for stereo calibrated data
        stereo_camera_pair: Tuple of (cam1, cam2) for stereo paths (required if use_stereo=True)
        calibration: If True, return calibration directory
    """
    base_dir = Path(base_dir)
    num_str = str(num_frame_pairs)

    # Calibration data
    if calibration:
        cam_str = f"Cam{cam}"
        calib_dir = base_dir / "calibration" / cam_str
        if endpoint:
            calib_dir = calib_dir / endpoint
        return dict(calib_dir=calib_dir)

    # Stereo calibrated data - uses dedicated stereo path structure
    if use_stereo:
        if stereo_camera_pair is None:
            raise ValueError("stereo_camera_pair required when use_stereo=True")
        cam_pair_str = f"Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}"
        data_dir = base_dir / "stereo_calibrated" / num_str / cam_pair_str / type_name
        stats_dir = base_dir / "statistics" / num_str / "stereo" / cam_pair_str / type_name
        video_dir = base_dir / "videos" / num_str / "stereo" / cam_pair_str
    # Uncalibrated data
    elif use_uncalibrated:
        cam_str = f"Cam{cam}"
        data_dir = base_dir / "uncalibrated_piv" / num_str / cam_str / type_name
        stats_dir = (
            base_dir / "statistics" / "uncalibrated" /
            num_str / cam_str / type_name
        )
        video_dir = base_dir / "videos" / "uncalibrated" / num_str / cam_str
    # Merged data
    elif use_merged:
        cam_str = f"Cam{cam}"
        data_dir = base_dir / "calibrated_piv" / num_str / "Merged" / type_name
        stats_dir = base_dir / "statistics" / num_str / "Merged" / type_name
        video_dir = base_dir / "videos" / num_str / "merged"
    # Regular calibrated data
    else:
        cam_str = f"Cam{cam}"
        data_dir = base_dir / "calibrated_piv" / num_str / cam_str / type_name
        stats_dir = base_dir / "statistics" / num_str / cam_str / type_name
        video_dir = base_dir / "videos" / num_str / cam_str
    if endpoint:
        data_dir = data_dir / endpoint
        stats_dir = stats_dir / endpoint
        video_dir = video_dir / endpoint
    return dict(data_dir=data_dir, stats_dir=stats_dir, video_dir=video_dir)
