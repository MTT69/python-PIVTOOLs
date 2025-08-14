from pathlib import Path


def get_data_paths(
    base_dir, num_images, cam_folder, type_name, endpoint="", use_merged=False
):
    """
    Construct directories for data, statistics, and videos.
    endpoint: optional subfolder ('' ignored).
    """
    base_dir = Path(base_dir)
    # Merged data
    if use_merged:
        data_dir = base_dir / "merged" / type_name
        stats_dir = base_dir / "statistics" / "merged" / type_name
        video_dir = base_dir / "videos" / "merged" / type_name
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
