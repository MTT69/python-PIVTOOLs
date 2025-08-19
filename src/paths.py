from pathlib import Path


def get_data_paths(
    base_dir,
    num_images,
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
        data_dir = base_dir / "uncalibrated_piv" / str(num_images) / cam / type_name
        stats_dir = (
            base_dir / "statistics" / "uncalibrated" / str(num_images) / cam / type_name
        )
        video_dir = base_dir / "videos" / "uncalibrated" / str(num_images) / cam
    # Merged data
    elif use_merged:
        data_dir = base_dir / "merged" / str(num_images) / cam / type_name
        stats_dir = base_dir / "statistics" / "merged" / cam / type_name
        video_dir = base_dir / "videos" / "merged" / cam / type_name
    # Regular calibrated data
    else:
        data_dir = base_dir / "calibrated_piv" / str(num_images) / cam / type_name
        stats_dir = base_dir / "statistics" / str(num_images) / cam / type_name
        video_dir = base_dir / "videos" / str(num_images) / cam
    if endpoint:
        data_dir = data_dir / endpoint
        stats_dir = stats_dir / endpoint
        video_dir = video_dir / endpoint
    return dict(data_dir=data_dir, stats_dir=stats_dir, video_dir=video_dir)
