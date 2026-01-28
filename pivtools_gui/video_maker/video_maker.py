import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_core.vector_loading import (
    find_valid_runs,
    get_highest_valid_run,
    read_mat_contents,
)

# Constants for optimization
DEFAULT_BATCH_SIZE = 10  # Files to preload for processing
LIMIT_SAMPLE_SIZE = 50  # Files for limit computation
LUT_SIZE = 4096  # High LUT resolution for smooth gradients and reduced banding
PERCENTILE_LOWER = 5
PERCENTILE_UPPER = 95

# ------------------------- Settings -------------------------


@dataclass
class PlotSettings:
    corners: tuple | None = None  # (x0, y0, x1, y1)

    variableName: str = ""
    variableUnits: str = ""
    length_units: str = "mm"
    title: str = ""

    save_name: str | None = None
    save_extension: str = ".png"
    save_varle: bool = False

    cmap: str | None = None
    levels: int | list = 500
    lower_limit: float | None = None
    upper_limit: float | None = None
    symmetric_around_zero: bool = True

    _xlabel: str = "x"
    _ylabel: str = "y"
    _fontsize: int = 12
    _title_fontsize: int = 14

    # New: optional coordinates
    coords_x: np.ndarray | None = None
    coords_y: np.ndarray | None = None

    # Video options
    fps: int = 30
    out_path: str = "field.mp4"
    mask_rgb: Tuple[int, int, int] = (200, 200, 200)  # RGB for masked pixels

    # Quality knobs - optimized for sharp, production-quality output
    use_ffmpeg: bool = True  # only ffmpeg supported
    crf: int = 8  # Maximum quality (lower = higher quality, range 0-51)
    codec: str = "libx264"  # ensure H.264 by default
    pix_fmt: str = "yuv420p"  # 4:2:0 for maximum compatibility (macOS, Windows, browsers)
    preset: str = "veryslow"  # Maximum quality encoding (slower but better compression)
    dither: bool = False  # Disabled by default to avoid graininess
    dither_strength: float = 0.0001  # Much lower strength when enabled
    upscale: Optional[float | Tuple[int, int]] = (
        None  # e.g. 2.0 or (H_out, W_out) or None (keep native)
    )

    # Extra ffmpeg args - optimized for scientific visualization with Mac/browser compatibility
    ffmpeg_extra_args: Tuple[str, ...] | List[str] = (
        "-profile:v", "high",  # H.264 High profile (compatible with macOS/QuickTime)
        "-level:v", "4.0",  # Level 4.0 for maximum compatibility (supports 1080p)
        "-tune", "stillimage",  # Optimize for still images/slow motion (scientific data)
        "-x264-params", "aq-mode=3:aq-strength=0.5:deblock=-1,-1",  # Reduced blocking artifacts
    )
    ffmpeg_loglevel: str = "warning"

    # For progress updates
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    # Test mode attributes
    test_mode: bool = False
    test_frames: Optional[int] = None

    @property
    def xlabel(self):
        if self.length_units:
            return f"{self._xlabel} ({self.length_units})"
        return self._xlabel

    @property
    def ylabel(self):
        if self.length_units:
            return f"{self._ylabel} ({self.length_units})"
        return self._ylabel


# ------------------------- Helpers -------------------------

_num_re = re.compile(r"(\d+)")


def _resolve_upscale(
    h: int, w: int, upscale: Optional[float | Tuple[int, int]]
) -> Tuple[int, int]:
    """Return (H_out, W_out). `upscale` can be None, a float factor, or (H, W)."""
    if upscale is None or upscale == 1.0:
        H = h
        W = w
    elif isinstance(upscale, (int, float)):
        H = int(round(h * float(upscale)))
        W = int(round(w * float(upscale)))
    else:  # assume (H, W) tuple
        target_h, target_w = upscale
        aspect_ratio = w / h
        # Fit to the largest possible size that matches the aspect ratio
        if target_w / target_h > aspect_ratio:
            H = target_h
            W = int(target_h * aspect_ratio)
        else:
            W = target_w
            H = int(target_w / aspect_ratio)
    # ensure even dims (important for yuv420p, many players/codecs)
    if H % 2:
        H += 1
    if W % 2:
        W += 1
    return H, W


def _natural_key(p: Path) -> List:
    s = str(p)
    parts = _num_re.split(s)
    parts[1::2] = [int(n) for n in parts[1::2]]
    return parts


def find_highest_valid_run_from_file(filepath: str, var: str) -> int:
    """
    Find the highest run index (0-based) that has valid non-empty data.
    Returns 0 if no valid runs found or if single run.

    Uses unified validation from pivtools_core.vector_loading.
    """
    try:
        result = get_highest_valid_run(filepath, one_based=False)
        return result if result is not None else 0
    except Exception as e:
        logger.error(f"Error finding highest valid run in {filepath}: {e}")
        return 0


def find_all_valid_runs_from_file(filepath: str, var: str) -> List[int]:
    """
    Find all run indices (0-based) that have valid non-empty data.
    Returns list of valid run indices sorted ascending.

    Uses unified validation from pivtools_core.vector_loading.
    """
    try:
        result = find_valid_runs(filepath, one_based=False)
        return result.valid_runs if result.valid_runs else [0]
    except Exception as e:
        logger.error(f"Error finding valid runs in {filepath}: {e}")
        return [0]


# Keep old function names as aliases for backward compatibility
def find_highest_valid_run(arrs: np.ndarray, filepath: str, var: str) -> int:
    """Deprecated: Use find_highest_valid_run_from_file instead."""
    return find_highest_valid_run_from_file(filepath, var)


def find_all_valid_runs(arrs: np.ndarray, var: str) -> List[int]:
    """Deprecated: This function requires a filepath. Use find_all_valid_runs_from_file instead."""
    # This is a fallback that won't work properly - callers should use find_all_valid_runs_from_file
    return [0]


def _select_variable_from_arrs(
    arrs: np.ndarray, filepath: str, var: str, run_index: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract variable and mask from arrays or MAT file, selecting the specified run index for multi-run data.

    Supports:
    - PIV variables: ux, uy, uz, mag
    - Instantaneous stats: u_prime, v_prime, w_prime, vorticity, divergence, gamma1, gamma2
    """

    # Debug: Check if var is actually a numpy array (which would be an error in calling code)
    if isinstance(var, np.ndarray):
        logger.error(f"ERROR: var parameter is a numpy array instead of string! var.shape={var.shape}, var.dtype={var.dtype}")
        logger.error(f"This suggests a bug in the calling code. Defaulting to 'ux'")
        var = "ux"  # Default to ux as a fallback
    elif not isinstance(var, (str, int)):
        logger.error(f"ERROR: var parameter has unexpected type {type(var)}: {var}")
        logger.error(f"Converting to string as fallback")
        var = str(var)

    # Define stats variables that are loaded from piv_result struct
    STATS_VARIABLES = {
        "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
        "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }
    
    # ndarray case (common path)
    if isinstance(arrs, np.ndarray):
        try:
            if arrs.ndim == 4:
                # Common layout: (R, N, H, W) with N>=3 (ux=0, uy=1, b_mask=2), R is runs
                # Validate run_index
                if not (0 <= run_index < arrs.shape[0]):
                    logger.warning(f"run_index {run_index} out of bounds for {filepath}, using 0")
                    run_index = 0
                var_idx = None
                if isinstance(var, str):
                    if var == "ux":
                        var_idx = 0
                    elif var == "uy":
                        var_idx = 1
                    elif var == "mag":  # Calculate magnitude for vector field
                        ux = arrs[run_index, 0]
                        uy = arrs[run_index, 1]
                        arr = np.sqrt(ux**2 + uy**2)
                        b_mask = arrs[run_index, 2] if arrs.shape[1] > 2 else None
                        return arr, (b_mask if b_mask is not None else None)
                    else:
                        # allow numeric string like "0"/"1"
                        try:
                            var_idx = int(var)
                        except Exception:
                            var_idx = None
                elif isinstance(var, int):
                    var_idx = var

                if var_idx is not None and 0 <= var_idx < arrs.shape[1]:
                    arr = arrs[run_index, var_idx]
                    b_mask = arrs[run_index, 2] if arrs.shape[1] > 2 else None
                    if arr.ndim != 2:
                        raise ValueError(f"Expected 2D array for {var} in {filepath} (run_index {run_index}), but got {arr.ndim}D with shape {arr.shape}. The MAT file may contain 1D data for this run; try a different run (e.g., run=1).")
                    return arr, (b_mask if b_mask is not None else None)
            elif arrs.ndim == 3:
                # Layout: (N, H, W) with N>=3 (ux=0, uy=1, b_mask=2) - single run already selected
                var_idx = None
                if isinstance(var, str):
                    if var == "ux":
                        var_idx = 0
                    elif var == "uy":
                        var_idx = 1
                    elif var == "mag":  # Calculate magnitude for vector field
                        ux = arrs[0]
                        uy = arrs[1]
                        arr = np.sqrt(ux**2 + uy**2)
                        b_mask = arrs[2] if arrs.shape[0] > 2 else None
                        return arr, (b_mask if b_mask is not None else None)
                    else:
                        # allow numeric string like "0"/"1"
                        try:
                            var_idx = int(var)
                        except Exception:
                            var_idx = None
                elif isinstance(var, int):
                    var_idx = var

                if var_idx is not None and 0 <= var_idx < arrs.shape[0]:
                    arr = arrs[var_idx]
                    b_mask = arrs[2] if arrs.shape[0] > 2 else None
                    if arr.ndim != 2:
                        raise ValueError(f"Expected 2D array for {var} in {filepath} (3D case), but got {arr.ndim}D with shape {arr.shape}. The MAT file may contain 1D data.")
                    return arr, (b_mask if b_mask is not None else None)
                else:
                    # If var_idx is invalid, default to first component (ux) for 3D arrays
                    logger.warning(f"Invalid variable '{var}' for 3D array in {filepath}, defaulting to index 0 (ux)")
                    arr = arrs[0]
                    b_mask = arrs[2] if arrs.shape[0] > 2 else None
                    if arr.ndim != 2:
                        raise ValueError(f"Expected 2D array for default variable (index 0) in {filepath} (3D case), but got {arr.ndim}D with shape {arr.shape}.")
                    # logger.debug(f"Returning default arr from 3D: arr.shape={arr.shape}, b_mask.shape={getattr(b_mask, 'shape', 'N/A')}")
                    return arr, (b_mask if b_mask is not None else None)

            # fallback: flatten first item (for non-3D/4D or invalid var_idx)
            # logger.debug(f"Fallback: arrs[0].shape={arrs[0].shape}")
            arr = arrs[0]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array for {var} in {filepath} (fallback), but got {arr.ndim}D with shape {arr.shape}. The MAT file may contain 1D data.")
            return arr, None
        except Exception as e:
            logger.error(f"Error in ndarray case for {filepath}: {e}")
            pass

    # dict-like or unknown: try loadmat to find a variable by name
    try:
        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        # Check for stats variables in piv_result struct (instantaneous stats files)
        if var in STATS_VARIABLES:
            piv_result = mat.get("piv_result")
            if piv_result is not None:
                # Handle object array wrapper - select correct run
                if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                    # piv_result is array of runs, select the requested run
                    # Find valid run with data (some runs may be empty)
                    pr = None
                    if 0 <= run_index < piv_result.size:
                        candidate = piv_result.flat[run_index]
                        if hasattr(candidate, var):
                            arr_check = getattr(candidate, var, None)
                            if arr_check is not None and np.asarray(arr_check).size > 0:
                                pr = candidate

                    # If requested run doesn't have data, find first valid run
                    if pr is None:
                        for i in range(piv_result.size):
                            candidate = piv_result.flat[i]
                            if hasattr(candidate, var):
                                arr_check = getattr(candidate, var, None)
                                if arr_check is not None and np.asarray(arr_check).size > 0:
                                    pr = candidate
                                    logger.debug(f"Stats var {var}: run {run_index} empty, using run {i}")
                                    break

                    if pr is None:
                        raise ValueError(f"No valid run found with variable '{var}' in {filepath}")
                else:
                    pr = piv_result

                if hasattr(pr, var):
                    arr = np.asarray(getattr(pr, var))
                    if arr.ndim != 2:
                        raise ValueError(f"Expected 2D array for {var} in {filepath}, got {arr.ndim}D with shape {arr.shape}")
                    # Try to get mask from piv_result
                    b_mask = None
                    for mask_attr in ("b_mask", "bmask", "mask", "valid_mask"):
                        if hasattr(pr, mask_attr):
                            b_mask = np.asarray(getattr(pr, mask_attr))
                            break
                    return arr, b_mask
                else:
                    raise ValueError(f"Stats variable '{var}' not found in piv_result struct in {filepath}")

        if var in mat:
            arr = np.asarray(mat[var])
            b_mask = None
            for key in ("b_mask", "bmask", "mask", "valid_mask"):
                if key in mat:
                    b_mask = np.asarray(mat[key])
                    break
            return arr, b_mask

        # Try to calculate magnitude if requested
        if var == "mag" and "ux" in mat and "uy" in mat:
            ux = np.asarray(mat["ux"])
            uy = np.asarray(mat["uy"])
            arr = np.sqrt(ux**2 + uy**2)
            b_mask = None
            for key in ("b_mask", "bmask", "mask", "valid_mask"):
                if key in mat:
                    b_mask = np.asarray(mat[key])
                    break
            return arr, b_mask
    except Exception as e:
        logger.error(f"Error loading MAT for {filepath}: {e}")
        pass

    # If arrs is dict-like, try to pull key directly
    try:
        if hasattr(arrs, "get") and not isinstance(arrs, np.ndarray):
            # Only proceed if it's actually dict-like and not a numpy array
            if var in arrs:
                arr = np.asarray(arrs[var])
                b_mask = arrs.get("b_mask", arrs.get("mask", None))
 
                return arr, (np.asarray(b_mask) if b_mask is not None else None)

            # Try to calculate magnitude if requested
            if var == "mag" and "ux" in arrs and "uy" in arrs:
                ux = np.asarray(arrs["ux"])
                uy = np.asarray(arrs["uy"])
                arr = np.sqrt(ux**2 + uy**2)
                b_mask = arrs.get("b_mask", arrs.get("mask", None))

                return arr, (np.asarray(b_mask) if b_mask is not None else None)
    except Exception as e:
        logger.error(f"Error in dict case for {filepath}: {e}")
        pass

    # give up with a clear error
    raise ValueError(f"Unable to extract variable '{var}' from {filepath}")


def _compute_global_limits_from_files(
    files: List[Path], var: str, settings: PlotSettings, run_index: int = 0
) -> Tuple[float, float, bool, float, float]:
    """Compute limits using parallel processing for efficiency."""
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin = float(settings.lower_limit)
        vmax = float(settings.upper_limit)
        use_two = settings.symmetric_around_zero and (vmin < 0 < vmax)
        return vmin, vmax, use_two, vmin, vmax

    files_to_check = (
        files[:LIMIT_SAMPLE_SIZE] if len(files) > LIMIT_SAMPLE_SIZE else files
    )
    all_values = []

    def process_file(f: Path) -> Optional[np.ndarray]:
        try:
            arrs = read_mat_contents(str(f), run_index=run_index)
            arr, b_mask = _select_variable_from_arrs(arrs, str(f), var, 0)  # Run already selected by read_mat_contents
            masked = np.ma.array(
                arr, mask=b_mask.astype(bool) if b_mask is not None else None
            )
            return masked.compressed() if masked.count() > 0 else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers = min(os.cpu_count(), 8)) as executor:
        futures = [executor.submit(process_file, f) for f in files_to_check]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_values.extend(result)

    if not all_values:
        actual_min = actual_max = 0.0
        vmin = -1.0
        vmax = 1.0
    else:
        all_values = np.array(all_values)
        actual_min = float(np.min(all_values))
        actual_max = float(np.max(all_values))
        vmin = (
            float(np.percentile(all_values, PERCENTILE_LOWER))
            if settings.lower_limit is None
            else float(settings.lower_limit)
        )
        vmax = (
            float(np.percentile(all_values, PERCENTILE_UPPER))
            if settings.upper_limit is None
            else float(settings.upper_limit)
        )

    use_two = False
    if settings.symmetric_around_zero and vmin < 0 < vmax:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
        use_two = True

    return vmin, vmax, use_two, actual_min, actual_max


def _make_lut(
    cmap_name: Optional[str], use_two_slope: bool, vmin: float, vmax: float
) -> np.ndarray:
    """Create LUT with caching for reuse."""
    # 1024-step LUT to reduce banding before codec quantization
    if cmap_name == "default":
        cmap_name = None
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
    else:
        if use_two_slope:
            cmap = plt.get_cmap("bwr")
        else:
            bwr = plt.get_cmap("bwr")
            if vmax <= 0:
                colors = bwr(np.linspace(0.0, 0.5, 256))
                cmap = mpl_colors.LinearSegmentedColormap.from_list("bwr_lower", colors)
            else:
                colors = bwr(np.linspace(0.5, 1.0, 256))
                cmap = mpl_colors.LinearSegmentedColormap.from_list("bwr_upper", colors)
    lut = (cmap(np.linspace(0, 1, LUT_SIZE))[:, :3] * 255).astype(
        np.uint8
    )  # (1024,3) RGB
    return lut


def _to_uint16_var(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Vectorized index computation."""
    norm = (frame - vmin) / (vmax - vmin)
    return np.clip((norm * (LUT_SIZE - 1)).round(), 0, LUT_SIZE - 1).astype(np.uint16)


# ------------------------- Writers (FFmpeg + fallback OpenCV) -------------------------


class FFmpegVideoWriter:
    def __init__(
        self,
        path,
        width,
        height,
        fps=30,
        crf=18,
        codec="libx264",
        pix_fmt="yuv420p",
        preset="slow",
        extra_args=None,
        loglevel="warning",
    ):
        # Use bundled FFmpeg from imageio-ffmpeg (always available via pip install)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        path = Path(path).resolve()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            loglevel,
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            codec,
            "-pix_fmt",
            pix_fmt,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ]
        # append any user-supplied extra args
        if extra_args:
            cmd += list(extra_args)
        cmd.append(str(path))

        # Capture stderr so the caller can see ffmpeg warnings and tuning info when we close
        # Annotate proc for type-checkers
        self.proc: subprocess.Popen = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.width, self.height = width, height
        self.path = str(path)

    def write(self, rgb_frame_uint8):
        # mypy/pylance treat proc.stdin as Optional; guard at runtime
        stdin = self.proc.stdin
        if stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        try:
            stdin.write(rgb_frame_uint8.tobytes())
        except BrokenPipeError:
            _, stderr = self.proc.communicate()
            if stderr:
                msg = stderr.decode(errors="replace").strip()
                print(f"ffmpeg stderr: {msg}")
            raise RuntimeError("ffmpeg process has exited (broken pipe)")

    def release(self):
        stdin = self.proc.stdin
        # Only close if not already closed
        if stdin is not None and not stdin.closed:
            stdin.close()
        # Only call communicate if stdin is not closed
        try:
            _, stderr = self.proc.communicate()
        except ValueError:
            # Already closed, ignore
            stderr = None
        if stderr:
            try:
                msg = stderr.decode(errors="replace").strip()
            except Exception:
                msg = str(stderr)
            if msg:
                print(f"ffmpeg stderr for {self.path}:\n", msg)


def verify_video_ready(video_path: str, timeout_sec: float = 5.0) -> bool:
    """
    Verify video file is complete by checking file exists and size is stable.

    Args:
        video_path: Path to the video file
        timeout_sec: Maximum time to wait for file to be ready

    Returns:
        True if video file exists and is ready, False if timeout reached
    """
    start_time = time.time()
    min_stable_checks = 2  # Require file size to be stable for 2 consecutive checks
    check_interval = 0.3  # Seconds between checks

    prev_size = -1
    stable_count = 0

    while (time.time() - start_time) < timeout_sec:
        try:
            path = Path(video_path)
            if not path.exists():
                time.sleep(check_interval)
                continue

            current_size = path.stat().st_size
            if current_size == 0:
                time.sleep(check_interval)
                continue

            # Track size stability (file might still be writing)
            if current_size == prev_size:
                stable_count += 1
            else:
                stable_count = 0
            prev_size = current_size

            # Consider file ready when size is stable
            if stable_count >= min_stable_checks:
                logger.debug(f"Video verified: {video_path} (size={current_size} bytes)")
                return True

            time.sleep(check_interval)
        except Exception as e:
            logger.debug(f"Video verification attempt failed: {e}")
            time.sleep(check_interval)

    # If file exists with size after timeout, still consider it ready
    path = Path(video_path)
    if path.exists() and path.stat().st_size > 0:
        logger.debug(f"Video ready after timeout: {video_path}")
        return True

    logger.warning(f"Video verification failed for {video_path}")
    return False


# ------------------------- Core: high-quality renderer -------------------------


def make_video_from_scalar(
    folder: str | Path,
    var: str = "uy",
    pattern: str = "[0-9]*.mat",
    settings: Optional[PlotSettings] = None,
    cancel_event=None,
    run_index: int = 0,
) -> dict:
    """
    Optimized video generation with batching and vectorization.
    Validates inputs, handles errors gracefully, and optimizes memory usage.
    run_index: int, default 0 - specifies which run (0-based index) to extract from multi-run .mat files (e.g., 4D arrays with shape (R, N, H, W)).
    """
    t0 = time.time()
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder}")
    
    files = sorted(
        [Path(p) for p in glob.glob(str(folder / pattern))], key=_natural_key
    )
    files = [f for f in files if "coordinate" not in f.name.lower()]
    if not files:
        raise FileNotFoundError(f"No MAT files found in {folder} matching '{pattern}'")

    if settings is None:
        settings = PlotSettings()
    if hasattr(settings, "test_mode") and getattr(settings, "test_mode", False):
        test_frames = getattr(settings, "test_frames", 50)
        files = files[:test_frames]

    # Validate that the run_index exists in the files, or fall back to highest valid run
    effective_run_index = run_index
    try:
        # Find all valid runs by reading the mat file directly (avoids inhomogeneous array issue)
        valid_runs = find_all_valid_runs_from_file(str(files[0]), var)

        if run_index in valid_runs:
            # Requested run is valid
            effective_run_index = run_index
        elif valid_runs:
            # Fall back to highest valid run (like vector viewer does)
            effective_run_index = max(valid_runs)
            logger.info(f"Run {run_index} has no data, falling back to highest valid run: {effective_run_index}")
        else:
            raise ValueError(f"No valid runs found in {files[0]} for variable {var}")

        # Verify we can read the effective run
        test_arrs = read_mat_contents(str(files[0]), run_index=effective_run_index)
        if isinstance(test_arrs, np.ndarray):
            if test_arrs.size == 0 or not np.any(test_arrs):
                raise ValueError(f"Run not found: run_index {effective_run_index} contains empty/zero data in {files[0]}")
        else:
            raise ValueError(f"Run not found: unexpected data type returned for run_index {effective_run_index}")
    except ValueError as e:
        # read_mat_contents already validates run_index and raises informative errors
        if "Invalid run_index" in str(e) or "No valid runs" in str(e) or "Run not found" in str(e):
            raise ValueError(f"Run not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to validate run_index {run_index} in {files[0]}: {e}")
        raise ValueError(f"Run not found: unable to load data with run_index {run_index}")

    # Use effective_run_index for all subsequent operations
    run_index = effective_run_index

    # Compute limits in parallel
    try:
        vmin, vmax, use_two, actual_min, actual_max = _compute_global_limits_from_files(
            files, var, settings, run_index
        )
    except Exception as e:
        logger.error(f"Failed to compute limits: {e}")
        raise

    lut = _make_lut(settings.cmap, use_two, vmin, vmax)

    # Get dimensions from first file
    try:
        arrs0 = read_mat_contents(str(files[0]), run_index=run_index)
        arr0, _ = _select_variable_from_arrs(arrs0, str(files[0]), var, 0)  # Run already selected by read_mat_contents
        logger.debug(f"First file arr0.shape={arr0.shape}, arr0.ndim={arr0.ndim}")
        if arr0.ndim != 2:
            raise ValueError(f"Expected 2D array for {var} in {files[0]}, but got {arr0.ndim}D with shape {arr0.shape}")
        H, W = arr0.shape
        if H == 0 or W == 0:
            raise ValueError(f"Invalid dimensions {H}x{W} in {files[0]}")
    except Exception as e:
        logger.error(f"Failed to read first file {files[0]}: {e}")
        raise

    Hout, Wout = _resolve_upscale(H, W, settings.upscale)

    try:
        writer = FFmpegVideoWriter(
            settings.out_path,
            Wout,
            Hout,
            fps=settings.fps,
            crf=settings.crf,
            codec=settings.codec,
            pix_fmt=settings.pix_fmt,
            preset=settings.preset,
            extra_args=settings.ffmpeg_extra_args,
            loglevel=settings.ffmpeg_loglevel,
        )
    except RuntimeError as e:
        logger.error(f"FFmpeg writer initialization failed: {e}")
        raise

    total_frames = len(files)
    for i in range(0, total_frames, DEFAULT_BATCH_SIZE):
        if cancel_event and cancel_event.is_set():
            logger.info("Video creation cancelled")
            break
        batch_files = files[i : i + DEFAULT_BATCH_SIZE]
        for j, f in enumerate(batch_files):
            try:
                arrs = read_mat_contents(str(f), run_index=run_index)
                field, b_mask = _select_variable_from_arrs(arrs, str(f), var, 0)  # Run already selected by read_mat_contents
                field_indices = _to_uint16_var(field, vmin, vmax)
                rgb = lut[field_indices]
                if Hout != H or Wout != W:
                    rgb = cv2.resize(rgb, (Wout, Hout), interpolation=cv2.INTER_LANCZOS4)
                    b_mask = (
                        cv2.resize(
                            b_mask.astype(np.uint8),
                            (Wout, Hout),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        if b_mask is not None
                        else None
                    )
                if b_mask is not None:
                    rgb[b_mask] = settings.mask_rgb
                writer.write(rgb)
                if settings.progress_callback:
                    settings.progress_callback(i + j + 1, total_frames)
            except Exception as e:
                logger.error(f"Error processing file {f}: {e}")
                continue  # Skip bad files but continue processing
        # Clear batch to free memory immediately
        del batch_files

    try:
        writer.release()
    except Exception as e:
        logger.error(f"Error releasing writer: {e}")

    t1 = time.time()
    return {
        "out_path": settings.out_path,
        "vmin": vmin,
        "vmax": vmax,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "use_two_slope": use_two,
        "fps": settings.fps,
        "frames": len(files),
        "shape": (H, W),
        "shape_out": (Hout, Wout),
        "variable": var,
        "cmap": settings.cmap,
        "elapsed_sec": round(t1 - t0, 3),
        "writer": "ffmpeg",
        "pix_fmt": getattr(settings, "pix_fmt", None),
        "crf": getattr(settings, "crf", None),
        "codec": getattr(settings, "codec", None),
        "effective_run": run_index + 1,  # Return 1-based run number that was actually used
    }


# ===================== VideoMaker CLASS =====================


class VideoMaker:
    """
    Production class for creating PIV visualization videos.

    Designed for:
    - GUI integration with progress callbacks
    - Command-line execution via __main__
    - Frame-level parallelism for color limit computation

    Pattern matches: planar_calibration_production.py

    Supported data sources:
    - calibrated: Calibrated instantaneous PIV frames
    - uncalibrated: Uncalibrated instantaneous PIV frames
    - merged: Merged stereo PIV frames
    - inst_stats: Per-frame instantaneous statistics (u_prime, vorticity, etc.)
    """

    # Variables that come from instantaneous stats files (not PIV files)
    STATS_VARIABLES = {
        "u_prime", "v_prime", "w_prime",  # Legacy fluctuations
        "uu_inst", "vv_inst", "ww_inst", "uv_inst", "uw_inst", "vw_inst",  # Stress tensor components
        "vorticity", "divergence", "gamma1", "gamma2",  # Derived stats
    }

    def __init__(
        self,
        base_dir: Path,
        camera: int = 1,
        config=None,
    ):
        """
        Initialize the video maker.

        Parameters
        ----------
        base_dir : Path
            Base directory for output (calibrated_piv, videos, statistics will be relative to this)
        camera : int
            Camera number (1-based)
        config : Config, optional
            Config object for additional settings
        """
        self.base_dir = Path(base_dir)
        self.camera = camera
        self._config = config

    def _get_data_dir(self, data_source: str, num_frame_pairs: int) -> Path:
        """
        Get the data directory for a given data source.

        Parameters
        ----------
        data_source : str
            One of: 'calibrated', 'uncalibrated', 'merged', 'stereo', 'inst_stats'
        num_frame_pairs : int
            Number of frame pairs (for path construction)

        Returns
        -------
        Path
            Directory containing the data files
        """
        num_str = str(num_frame_pairs)

        if data_source == "stereo":
            # Get stereo camera pair from config
            if self._config and self._config.stereo_pairs:
                cam_pair = self._config.stereo_pairs[0]
            else:
                cam_pair = (1, 2)
            cam_pair_str = f"Cam{cam_pair[0]}_Cam{cam_pair[1]}"
            return self.base_dir / "stereo_calibrated" / num_str / cam_pair_str / "instantaneous"
        elif data_source == "calibrated":
            return self.base_dir / "calibrated_piv" / num_str / f"Cam{self.camera}" / "instantaneous"
        elif data_source == "uncalibrated":
            return self.base_dir / "uncalibrated_piv" / num_str / f"Cam{self.camera}" / "instantaneous"
        elif data_source == "merged":
            return self.base_dir / "calibrated_piv" / num_str / "Merged" / "instantaneous"
        elif data_source == "inst_stats":
            return self.base_dir / "statistics" / num_str / f"Cam{self.camera}" / "instantaneous" / "instantaneous_stats"
        else:
            raise ValueError(f"Unknown data_source: {data_source}. Must be one of: calibrated, uncalibrated, merged, stereo, inst_stats")

    def _get_video_dir(self, data_source: str, num_frame_pairs: int) -> Path:
        """
        Get the video output directory for a given data source.

        Parameters
        ----------
        data_source : str
            One of: 'calibrated', 'uncalibrated', 'merged', 'stereo', 'inst_stats'
        num_frame_pairs : int
            Number of frame pairs (for path construction)

        Returns
        -------
        Path
            Directory for video output
        """
        num_str = str(num_frame_pairs)

        if data_source == "stereo":
            # Get stereo camera pair from config
            if self._config and self._config.stereo_pairs:
                cam_pair = self._config.stereo_pairs[0]
            else:
                cam_pair = (1, 2)
            cam_pair_str = f"Cam{cam_pair[0]}_Cam{cam_pair[1]}"
            return self.base_dir / "videos" / num_str / "stereo" / cam_pair_str
        elif data_source == "merged":
            return self.base_dir / "videos" / num_str / "merged"
        elif data_source == "inst_stats":
            return self.base_dir / "videos" / num_str / f"Cam{self.camera}" / "stats"
        elif data_source == "uncalibrated":
            return self.base_dir / "videos" / num_str / f"Cam{self.camera}" / "uncalibrated"
        else:
            return self.base_dir / "videos" / num_str / f"Cam{self.camera}"

    def process_video(
        self,
        variable: str = "ux",
        run: int = 1,
        data_source: str = "calibrated",
        fps: int = 30,
        crf: int = 15,
        resolution: Optional[Tuple[int, int]] = (1080, 1920),
        cmap: Optional[str] = None,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        test_mode: bool = False,
        test_frames: int = 50,
        out_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event=None,
    ) -> dict:
        """
        Process and create a video for the specified variable and data source.

        This is the main entry point for GUI integration, handling data source
        routing and progress callbacks compatible with job_manager.

        Parameters
        ----------
        variable : str
            Variable to visualize (ux, uy, uz, mag, u_prime, vorticity, etc.)
        run : int
            Run number (1-based)
        data_source : str
            Data source type: 'calibrated', 'uncalibrated', 'merged', 'inst_stats'
        fps : int
            Video frame rate
        crf : int
            FFmpeg CRF quality (lower = higher quality)
        resolution : tuple, optional
            Output resolution as (height, width)
        cmap : str, optional
            Matplotlib colormap name
        lower_limit : float, optional
            Lower color scale limit (None = auto)
        upper_limit : float, optional
            Upper color scale limit (None = auto)
        test_mode : bool
            If True, only process test_frames frames
        test_frames : int
            Number of frames for test mode
        out_name : str, optional
            Custom output filename
        progress_callback : callable, optional
            Function called with (current_frame, total_frames, message)
            Compatible with job_manager.update_job pattern
        cancel_event : threading.Event, optional
            Event to signal cancellation

        Returns
        -------
        dict
            success: bool
            out_path: str (path to created video)
            error: str (if failed)
            vmin, vmax: float (computed color limits)
            effective_run: int (run number actually used)
            data_source: str (data source used)
        """
        try:
            # Get num_frame_pairs from config
            num_frame_pairs = self._config.num_frame_pairs if self._config else 100

            # Validate data_source and variable compatibility
            if variable in self.STATS_VARIABLES and data_source != "inst_stats":
                # Auto-switch to inst_stats for stats variables
                logger.info(f"Variable '{variable}' requires inst_stats data source, switching from '{data_source}'")
                data_source = "inst_stats"

            # Get appropriate directories
            data_dir = self._get_data_dir(data_source, num_frame_pairs)
            video_dir = self._get_video_dir(data_source, num_frame_pairs)

            # Ensure video directory exists
            video_dir.mkdir(parents=True, exist_ok=True)

            # Validate data directory exists
            if not data_dir.exists():
                return {
                    "success": False,
                    "error": f"Data directory does not exist: {data_dir}",
                    "data_source": data_source,
                }

            # Determine output filename
            if out_name is None:
                source_suffix = f"_{data_source}" if data_source != "calibrated" else ""
                out_name = f"run{run}_Cam{self.camera}_{variable}{source_suffix}{'_test' if test_mode else ''}.mp4"

            out_path = str(video_dir / out_name)

            # Create settings
            settings = PlotSettings(
                fps=fps,
                crf=crf,
                upscale=resolution,
                cmap=cmap,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                out_path=out_path,
                progress_callback=progress_callback,
                test_mode=test_mode,
                test_frames=test_frames if test_mode else None,
            )

            logger.info(f"Creating video: var={variable}, run={run}, source={data_source}, data_dir={data_dir}")

            # Call core video generation
            result = make_video_from_scalar(
                folder=data_dir,
                var=variable,
                pattern="[0-9]*.mat",
                settings=settings,
                cancel_event=cancel_event,
                run_index=run - 1,  # Convert to 0-based
            )

            # Verify video is ready
            if not verify_video_ready(out_path, timeout_sec=30.0):
                logger.warning(f"Video verification timed out for {out_path}")

            return {
                "success": True,
                "out_path": result["out_path"],
                "vmin": result["vmin"],
                "vmax": result["vmax"],
                "actual_min": result["actual_min"],
                "actual_max": result["actual_max"],
                "effective_run": result["effective_run"],
                "frames": result["frames"],
                "elapsed_sec": result["elapsed_sec"],
                "data_source": data_source,
                "variable": variable,
            }

        except Exception as e:
            logger.error(f"Video creation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "data_source": data_source}

    def create_video(
        self,
        variable: str = "ux",
        run: int = 1,
        fps: int = 30,
        crf: int = 15,
        resolution: Optional[Tuple[int, int]] = (1080, 1920),
        cmap: Optional[str] = None,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        test_mode: bool = False,
        test_frames: int = 50,
        out_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event=None,
        data_source: str = "calibrated",
    ) -> dict:
        """
        Create a video for the specified variable and run.

        This is a convenience wrapper around process_video() for backward compatibility.
        For new code, prefer using process_video() directly.

        Parameters
        ----------
        variable : str
            Variable to visualize (ux, uy, uz, mag, u_prime, vorticity, etc.)
        run : int
            Run number (1-based)
        fps : int
            Video frame rate
        crf : int
            FFmpeg CRF quality (lower = higher quality)
        resolution : tuple, optional
            Output resolution as (height, width)
        cmap : str, optional
            Matplotlib colormap name
        lower_limit : float, optional
            Lower color scale limit (None = auto)
        upper_limit : float, optional
            Upper color scale limit (None = auto)
        test_mode : bool
            If True, only process test_frames frames
        test_frames : int
            Number of frames for test mode
        out_name : str, optional
            Custom output filename
        progress_callback : callable, optional
            Function called with (current_frame, total_frames, message)
        cancel_event : threading.Event, optional
            Event to signal cancellation
        data_source : str
            Data source type (default: 'calibrated')

        Returns
        -------
        dict
            success: bool
            out_path: str (path to created video)
            error: str (if failed)
            vmin, vmax: float (computed color limits)
            effective_run: int (run number actually used)
        """
        return self.process_video(
            variable=variable,
            run=run,
            data_source=data_source,
            fps=fps,
            crf=crf,
            resolution=resolution,
            cmap=cmap,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            test_mode=test_mode,
            test_frames=test_frames,
            out_name=out_name,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )


# ===================== PRODUCTION SCRIPT =====================

# Hardcoded configuration for standalone execution
# These are used when USE_CONFIG_DIRECTLY = False

# BASE_DIR: Base directory containing PIV data
_BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images"

# CAMERA_NUMS: List of camera numbers to process (1-based)
_CAMERA_NUMS = [1]

# VIDEO PARAMETERS
_VARIABLE = "ux"  # Variable to visualize: ux, uy, mag, u_prime, vorticity, etc.
_RUN = 1  # Run number (1-based)
_DATA_SOURCE = "calibrated"  # calibrated, uncalibrated, merged, inst_stats
_FPS = 30  # Video frame rate
_CRF = 15  # FFmpeg CRF quality (lower = higher quality)
_RESOLUTION = (1080, 1920)  # Output resolution (height, width) or None for native
_CMAP = None  # Colormap name or None for auto
_LOWER_LIMIT = None  # Lower color limit or None for auto
_UPPER_LIMIT = None  # Upper color limit or None for auto

# TEST MODE
_TEST_MODE = False  # If True, only process _TEST_FRAMES frames
_TEST_FRAMES = 50  # Number of frames for test mode

# USE_CONFIG_DIRECTLY: If True, load video settings from config.yaml
# If False, use hardcoded settings above and write them to config.yaml
USE_CONFIG_DIRECTLY = True


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the video maker uses the correct paths and settings.

    Returns
    -------
    Config
        The reloaded config object with updated settings
    """
    from pivtools_core.config import get_config, reload_config

    config = get_config()

    # Paths
    config.data["paths"]["base_paths"] = [_BASE_DIR]
    config.data["paths"]["camera_count"] = len(_CAMERA_NUMS)
    config.data["paths"]["camera_numbers"] = _CAMERA_NUMS

    # Video settings (using new single-dict format)
    if "video" not in config.data:
        config.data["video"] = {}

    config.data["video"]["base_path_idx"] = 0
    config.data["video"]["camera"] = _CAMERA_NUMS[0] if _CAMERA_NUMS else 1
    config.data["video"]["data_source"] = _DATA_SOURCE
    config.data["video"]["variable"] = _VARIABLE
    config.data["video"]["run"] = _RUN
    config.data["video"]["piv_type"] = "instantaneous"
    config.data["video"]["cmap"] = _CMAP if _CMAP else "default"
    config.data["video"]["lower"] = str(_LOWER_LIMIT) if _LOWER_LIMIT is not None else ""
    config.data["video"]["upper"] = str(_UPPER_LIMIT) if _UPPER_LIMIT is not None else ""
    config.data["video"]["fps"] = _FPS
    config.data["video"]["crf"] = _CRF
    if _RESOLUTION is not None:
        if _RESOLUTION[0] >= 2160:
            config.data["video"]["resolution"] = "4k"
        else:
            config.data["video"]["resolution"] = "1080p"
    else:
        config.data["video"]["resolution"] = "1080p"

    # Save to disk
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    return reload_config()


if __name__ == "__main__":
    from pivtools_core.config import get_config

    logger.info("=" * 60)
    logger.info("Video Maker - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Extract settings from config
        base_dir = Path(config.base_paths[config.video_base_path_idx])
        camera_nums = [config.video_camera]
        variable = config.video_variable
        run = config.video_run
        data_source = config.video_data_source
        fps = config.video_fps
        crf = config.video_crf
        resolution = config.video_resolution
        cmap = config.video_cmap if config.video_cmap != "default" else None
        lower_limit = config.video_lower_limit
        upper_limit = config.video_upper_limit
        test_mode = _TEST_MODE
        test_frames = _TEST_FRAMES
    else:
        # Apply CLI settings to config.yaml
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        base_dir = Path(_BASE_DIR)
        camera_nums = _CAMERA_NUMS
        variable = _VARIABLE
        run = _RUN
        data_source = _DATA_SOURCE
        fps = _FPS
        crf = _CRF
        resolution = _RESOLUTION
        cmap = _CMAP
        lower_limit = _LOWER_LIMIT
        upper_limit = _UPPER_LIMIT
        test_mode = _TEST_MODE
        test_frames = _TEST_FRAMES

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Cameras: {camera_nums}")
    logger.info(f"Variable: {variable}, Run: {run}, Data source: {data_source}")
    logger.info(f"FPS: {fps}, CRF: {crf}, Resolution: {resolution}")
    logger.info(f"Colormap: {cmap or 'auto'}, Limits: [{lower_limit or 'auto'}, {upper_limit or 'auto'}]")
    if test_mode:
        logger.info(f"TEST MODE: {test_frames} frames")

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"\nProcessing Camera {camera_num}...")

        try:
            maker = VideoMaker(
                base_dir=base_dir,
                camera=camera_num,
                config=config,
            )

            result = maker.process_video(
                variable=variable,
                run=run,
                data_source=data_source,
                fps=fps,
                crf=crf,
                resolution=resolution,
                cmap=cmap,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                test_mode=test_mode,
                test_frames=test_frames,
            )

            if result["success"]:
                logger.info(f"Video created: {result['out_path']}")
                logger.info(f"  Limits: {result['vmin']:.3f} to {result['vmax']:.3f}")
                logger.info(f"  Frames: {result['frames']}, Time: {result['elapsed_sec']:.1f}s")
                logger.info(f"  Effective run: {result['effective_run']}")
            else:
                logger.error(f"Failed: {result.get('error')}")
                failed_cameras.append(camera_num)

        except Exception as e:
            logger.error(f"Camera {camera_num} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_cameras.append(camera_num)

    logger.info("=" * 60)
    if failed_cameras:
        logger.error(f"Video creation failed for cameras: {failed_cameras}")
    else:
        logger.info("Video creation completed successfully for all cameras")
