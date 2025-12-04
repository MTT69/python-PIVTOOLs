from pathlib import Path


def get_data_paths(
    base_dir,
    num_frame_pairs,
    cam,
    type_name,
    endpoint="",
    use_merged=False,
    use_uncalibrated=False,
    calibration=False,  # New argument
):
    """
    Construct directories for data, statistics, and videos.
    endpoint: optional subfolder ('' ignored).
    use_uncalibrated: if True, return paths for uncalibrated data
    calibration: if True, return calibration directory
    """
    base_dir = Path(base_dir)
    cam = f"Cam{cam}"
    # Calibration data
    if calibration:
        calib_dir = base_dir / "calibration" / cam
        if endpoint:
            calib_dir = calib_dir / endpoint
        return dict(calib_dir=calib_dir)
    # Uncalibrated data
    if use_uncalibrated:
        num_str = str(num_frame_pairs)
        data_dir = base_dir / "uncalibrated_piv" / num_str / cam / type_name
        stats_dir = (
            base_dir / "statistics" / "uncalibrated" /
            num_str / cam / type_name
        )
        video_dir = base_dir / "videos" / "uncalibrated" / num_str / cam
    # Merged data
    elif use_merged:
        data_dir = base_dir / "merged" / str(num_frame_pairs) / cam / type_name
        stats_dir = base_dir / "statistics" / "merged" / cam / type_name
        video_dir = base_dir / "videos" / "merged" / cam / type_name
    # Regular calibrated data
    else:
        num_str = str(num_frame_pairs)
        data_dir = base_dir / "calibrated_piv" / num_str / cam / type_name
        stats_dir = base_dir / "statistics" / num_str / cam / type_name
        video_dir = base_dir / "videos" / num_str / cam
    if endpoint:
        data_dir = data_dir / endpoint
        stats_dir = stats_dir / endpoint
        video_dir = video_dir / endpoint
    return dict(data_dir=data_dir, stats_dir=stats_dir, video_dir=video_dir)
