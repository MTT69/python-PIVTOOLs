from pathlib import Path


def get_data_paths(
    base_dir,
    num_images,
    cam_folder,
    type_name,
    endpoint="",
    use_merged=False,
    use_uncalibrated=False,
):
    """
    Construct directories for data, statistics, and videos.
    endpoint: optional subfolder ('' ignored).
    use_uncalibrated: if True, return paths for uncalibrated data
    """
    base_dir = Path(base_dir)
    # Uncalibrated data
    if use_uncalibrated:
        data_dir = (
            base_dir / "uncalibrated_piv" / str(num_images) / cam_folder / type_name
        )
        stats_dir = (
            base_dir
            / "statistics"
            / "uncalibrated"
            / str(num_images)
            / cam_folder
            / type_name
        )
        video_dir = base_dir / "videos" / "uncalibrated" / str(num_images) / cam_folder
    # Merged data
    elif use_merged:
        data_dir = base_dir / "merged" / str(num_images) / cam_folder / type_name
        stats_dir = base_dir / "statistics" / "merged" / cam_folder / type_name
        video_dir = base_dir / "videos" / "merged" / cam_folder / type_name
    # Regular calibrated data
    else:
        data_dir = (
            base_dir / "calibrated_piv" / str(num_images) / cam_folder / type_name
        )
        stats_dir = base_dir / "statistics" / str(num_images) / cam_folder / type_name
        video_dir = base_dir / "videos" / str(num_images) / cam_folder
    if endpoint:
        data_dir = data_dir / endpoint
        stats_dir = stats_dir / endpoint
        video_dir = video_dir / endpoint
    return dict(data_dir=data_dir, stats_dir=stats_dir, video_dir=video_dir)
