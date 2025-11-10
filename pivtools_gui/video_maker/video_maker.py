import glob
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import os
import cv2
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..vector_loading import read_mat_contents

# Constants for optimization
DEFAULT_BATCH_SIZE = 10  # Files to preload for processing
LIMIT_SAMPLE_SIZE = 50  # Files for limit computation
LUT_SIZE = 1024  # LUT resolution for color mapping
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

    # Quality knobs
    use_ffmpeg: bool = True  # only ffmpeg supported
    crf: int = 18  # tuned for compatible H.264
    codec: str = "libx264"  # ensure H.264 by default
    pix_fmt: str = "yuv420p"  # ensure maximum compatibility (Windows players)
    preset: str = "slow"  # encoding speed/size tradeoff
    dither: bool = False  # Disabled by default to avoid graininess
    dither_strength: float = 0.0001  # Much lower strength when enabled
    upscale: Optional[float | Tuple[int, int]] = (
        None  # e.g. 2.0 or (H_out, W_out) or None (keep native)
    )

    # Extra ffmpeg args (appended to the ffmpeg command) - use this to tune quality further
    ffmpeg_extra_args: Tuple[str, ...] | List[str] = ()
    ffmpeg_loglevel: str = "warning"

    # For progress updates
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    # Test mode attributes
    test_mode: bool = False
    test_frames: Optional[int] = None

    # Noise reduction options
    apply_smoothing: bool = True  # Enable light smoothing by default
    smoothing_sigma: float = 0.8  # Gaussian smoothing strength
    median_filter_size: int = 3  # Median filter to remove salt-and-pepper noise

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


def _select_variable_from_arrs(
    arrs: np.ndarray, filepath: str, var: str, run_index: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract variable and mask from arrays or MAT file, selecting the specified run index for multi-run data."""
    
    # Debug: Check if var is actually a numpy array (which would be an error in calling code)
    if isinstance(var, np.ndarray):
        logger.error(f"ERROR: var parameter is a numpy array instead of string! var.shape={var.shape}, var.dtype={var.dtype}")
        logger.error(f"This suggests a bug in the calling code. Defaulting to 'ux'")
        var = "ux"  # Default to ux as a fallback
    elif not isinstance(var, (str, int)):
        logger.error(f"ERROR: var parameter has unexpected type {type(var)}: {var}")
        logger.error(f"Converting to string as fallback")
        var = str(var)
    
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


def _apply_noise_reduction(field: np.ndarray, settings: PlotSettings) -> np.ndarray:
    """Apply smoothing and filtering efficiently."""
    if not getattr(settings, "apply_smoothing", True):
        return field
    field_smooth = field.astype(np.float32)
    median_size = getattr(settings, "median_filter_size", 3)
    if median_size > 1:
        field_smooth = cv2.medianBlur(field_smooth, median_size)
    sigma = getattr(settings, "smoothing_sigma", 0.8)
    if sigma > 0:
        field_smooth = cv2.GaussianBlur(field_smooth, (0, 0), sigma)
    return field_smooth


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
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        path = Path(path).resolve()
        cmd = [
            "ffmpeg",
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

    # Validate that the run_index exists in the files
    # Note: read_mat_contents will raise ValueError if run_index is invalid
    try:
        test_arrs = read_mat_contents(str(files[0]), run_index=run_index)
        # Check if the returned data contains any non-zero elements (i.e., is not empty)
        if isinstance(test_arrs, np.ndarray):
            if test_arrs.size == 0 or not np.any(test_arrs):
                raise ValueError(f"Run not found: run_index {run_index} contains empty/zero data in {files[0]}")
        else:
            raise ValueError(f"Run not found: unexpected data type returned for run_index {run_index}")
    except ValueError as e:
        # read_mat_contents already validates run_index and raises informative errors
        if "Invalid run_index" in str(e) or "No valid runs" in str(e) or "Run not found" in str(e):
            raise ValueError(f"Run not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to validate run_index {run_index} in {files[0]}: {e}")
        raise ValueError(f"Run not found: unable to load data with run_index {run_index}")

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
                field = _apply_noise_reduction(field, settings)
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
    }
