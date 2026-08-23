import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _expected_vector_names(vector_format: str, num_frame_pairs: int) -> dict:
    """Map every expected result file name to its 1-based frame index.

    The value is what restores frame order after a directory scan: ``os.scandir``
    yields entries in arbitrary order, and sorting by name is only correct while
    ``vector_format`` is zero-padded (``"B10.mat"`` sorts before ``"B2.mat"``).

    Names match EXACTLY, including case. The superseded ``Path.exists()`` probe was
    case-insensitive on Windows and NTFS, so a tree written as ``00001.MAT`` and read
    back with a ``"%05d.mat"`` config used to be found and now reads as empty. Writer
    and reader both build the name from the same ``vector_format``, so this only bites
    if that config value's case is edited between the run and the read -- fail loudly
    there rather than case-fold, which would make two files differing only in case
    collide on a case-sensitive filesystem.
    """
    return {vector_format % i: i for i in range(1, num_frame_pairs + 1)}


def list_vector_files(
    data_dir, vector_format: str, num_frame_pairs: int
) -> list[Path]:
    """Expected vector files present in ``data_dir``, in frame order.

    Uses a single :func:`os.scandir` instead of one ``Path.exists()`` per frame. On a
    3600-pair dataset on an external HDD that is 5 ms against 1037 ms -- the sweep was
    the dominant cost of the Results-tab mount.

    Parameters
    ----------
    data_dir : path-like
        Directory to scan. A missing directory yields ``[]`` rather than raising: a
        data source that was never produced simply has no output folder.
    vector_format : str
        printf pattern from config, e.g. ``"%05d.mat"``.
    num_frame_pairs : int
        Frame pairs the run is expected to have produced.

    Returns
    -------
    list of Path
        Sorted by frame index, so ``[0]`` is the lowest-numbered frame present.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    expected = _expected_vector_names(vector_format, num_frame_pairs)
    found = []
    try:
        with os.scandir(data_dir) as entries:
            for entry in entries:
                if entry.name in expected and entry.is_file():
                    found.append(Path(entry.path))
    except OSError as exc:
        logger.warning(f"Cannot scan {data_dir}: {exc}")
        return []

    found.sort(key=lambda p: expected[p.name])
    return found


def count_vector_files_by_name(data_dir, expected_names) -> int:
    """Number of ``expected_names`` present in ``data_dir``.

    The scan primitive. Takes a prebuilt name collection so a caller sweeping many
    datasets builds the expected set once instead of per directory -- see
    ``pivtools_gui.app._scan_dataset_progress``. Prefer :func:`count_vector_files`
    when you have the format and pair count to hand.

    Parameters
    ----------
    data_dir : path-like
        Directory to scan. A missing directory counts as zero rather than raising: a
        dataset that has not been started yet simply has no output folder.
    expected_names : set or dict
        Result file names this run is expected to produce.

    Returns
    -------
    int
        Files present, bounded above by ``len(expected_names)``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return 0

    count = 0
    try:
        with os.scandir(data_dir) as entries:
            for entry in entries:
                if entry.name in expected_names and entry.is_file():
                    count += 1
    except OSError as exc:
        logger.warning(f"Cannot scan {data_dir}: {exc}")
        return 0
    return count


def count_vector_files(data_dir, vector_format: str, num_frame_pairs: int) -> int:
    """Number of expected vector files present in ``data_dir``.

    Same scan as :func:`list_vector_files` without building the path list. Callers
    that need the files should call :func:`list_vector_files` once rather than
    counting and then listing.
    """
    return count_vector_files_by_name(
        data_dir, _expected_vector_names(vector_format, num_frame_pairs)
    )


def vector_glob_from_format(vector_format: str) -> str:
    """Glob matching PIV vector files from a printf ``vector_format``.

    ``"%05d.mat" -> "*.mat"``, ``"B%05d.mat" -> "B*.mat"``. The hardcoded ``"B*.mat"``
    default missed the standard ``"%05d.mat"`` naming, so callers must derive the glob
    from config rather than guessing. Non-vector ``.mat`` files (``coordinates.mat``,
    ``mask.mat``) are filtered downstream, so a bare ``"*.mat"`` is safe.
    """
    return re.sub(r"%[0-9]*d", "*", str(vector_format))


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
        stats_dir = (
            base_dir / "statistics" / num_str / "stereo" / cam_pair_str / type_name
        )
        video_dir = base_dir / "videos" / num_str / "stereo" / cam_pair_str
    # Uncalibrated data
    elif use_uncalibrated:
        cam_str = f"Cam{cam}"
        data_dir = base_dir / "uncalibrated_piv" / num_str / cam_str / type_name
        stats_dir = (
            base_dir / "statistics" / "uncalibrated" / num_str / cam_str / type_name
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
