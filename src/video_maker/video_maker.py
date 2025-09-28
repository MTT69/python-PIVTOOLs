import glob
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent.parent))

from post_processing.vector_loading import read_mat_contents

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
    save_pickle: bool = False

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
    mask_rgb: tuple = (200, 200, 200)  # RGB for masked pixels

    # Quality knobs
    use_ffmpeg: bool = True  # only ffmpeg supported
    crf: int = 18  # tuned for compatible H.264
    codec: str = "libx264"  # ensure H.264 by default
    pix_fmt: str = "yuv420p"  # ensure maximum compatibility (Windows players)
    preset: str = "slow"  # encoding speed/size tradeoff
    dither: bool = False  # Disabled by default to avoid graininess
    dither_strength: float = 0.0001  # Much lower strength when enabled
    upscale: float | tuple | None = (
        None  # e.g. 2.0 or (H_out, W_out) or None (keep native)
    )

    # Extra ffmpeg args (appended to the ffmpeg command) - use this to tune quality further
    ffmpeg_extra_args: tuple | list = ()
    ffmpeg_loglevel: str = "warning"

    # For progress updates
    progress_callback: callable = None

    # Test mode attributes
    test_mode: bool = False
    test_frames: int | None = None

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


def _resolve_upscale(h, w, upscale):
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


def _natural_key(p: Path):
    s = str(p)
    parts = _num_re.split(s)
    parts[1::2] = [int(n) for n in parts[1::2]]
    return parts


def _select_variable_from_arrs(arrs, filepath: str, pick: str):
    """
    Return (arr, b_mask) where `arr` is the requested variable (2D array) and b_mask is either
    a mask array or None. This function is tolerant: it will try ndarray indexing patterns,
    integer indices, and finally inspect the MAT file (loadmat) to find a variable named `pick`.
    """
    # ndarray case (common path)
    if isinstance(arrs, np.ndarray):
        try:
            if arrs.ndim == 4:
                # Common layout: (R, N, H, W) with N>=3 (ux=0, uy=1, b_mask=2)
                idx = None
                if isinstance(pick, str):
                    if pick == "ux":
                        idx = 0
                    elif pick == "uy":
                        idx = 1
                    elif pick == "mag":  # Calculate magnitude for vector field
                        ux = arrs[0, 0]
                        uy = arrs[0, 1]
                        arr = np.sqrt(ux**2 + uy**2)
                        b_mask = arrs[0, 2] if arrs.shape[1] > 2 else None
                        return arr, (b_mask if b_mask is not None else None)
                    else:
                        # allow numeric string like "0"/"1"
                        try:
                            idx = int(pick)
                        except Exception:
                            idx = None
                elif isinstance(pick, int):
                    idx = pick

                if idx is not None and 0 <= idx < arrs.shape[1]:
                    arr = arrs[0, idx]
                    b_mask = arrs[0, 2] if arrs.shape[1] > 2 else None
                    return arr, (b_mask if b_mask is not None else None)

            # fallback: flatten first item
            return arrs[0], None
        except Exception:
            pass

    # dict-like or unknown: try loadmat to find a variable by name
    try:
        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        if pick in mat:
            arr = np.asarray(mat[pick])
            b_mask = None
            for key in ("b_mask", "bmask", "mask", "valid_mask"):
                if key in mat:
                    b_mask = np.asarray(mat[key])
                    break
            return arr, b_mask

        # Try to calculate magnitude if requested
        if pick == "mag" and "ux" in mat and "uy" in mat:
            ux = np.asarray(mat["ux"])
            uy = np.asarray(mat["uy"])
            arr = np.sqrt(ux**2 + uy**2)
            b_mask = None
            for key in ("b_mask", "bmask", "mask", "valid_mask"):
                if key in mat:
                    b_mask = np.asarray(mat[key])
                    break
            return arr, b_mask
    except Exception:
        pass

    # If arrs is dict-like, try to pull key directly
    try:
        if hasattr(arrs, "get"):
            if pick in arrs:
                arr = np.asarray(arrs[pick])
                b_mask = arrs.get("b_mask", arrs.get("mask", None))
                return arr, (np.asarray(b_mask) if b_mask is not None else None)

            # Try to calculate magnitude if requested
            if pick == "mag" and "ux" in arrs and "uy" in arrs:
                ux = np.asarray(arrs["ux"])
                uy = np.asarray(arrs["uy"])
                arr = np.sqrt(ux**2 + uy**2)
                b_mask = arrs.get("b_mask", arrs.get("mask", None))
                return arr, (np.asarray(b_mask) if b_mask is not None else None)
    except Exception:
        pass

    # give up with a clear error
    raise ValueError(f"Unable to extract variable '{pick}' from {filepath}")


def _compute_global_limits_from_files(files, pick, settings: PlotSettings):
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin = float(settings.lower_limit)
        vmax = float(settings.upper_limit)
        use_two = settings.symmetric_around_zero and (vmin < 0 < vmax)
        return vmin, vmax, use_two, vmin, vmax

    # Use first 50 frames for automatic limit computation
    N = 50
    files_to_check = files[:N] if len(files) > N else files

    all_values = []
    for f in files_to_check:
        try:
            arrs = read_mat_contents(str(f))
            arr, b_mask = _select_variable_from_arrs(arrs, str(f), pick)
        except Exception:
            # skip files we can't read or that don't contain the requested variable
            continue

        masked = np.ma.array(
            arr, mask=b_mask.astype(bool) if b_mask is not None else None
        )
        if masked.count() == 0:
            continue

        # Collect valid values for percentile calculation
        valid_values = masked.compressed()
        if len(valid_values) > 0:
            all_values.extend(valid_values.flatten())

    if len(all_values) == 0:
        gmin, gmax = -1.0, 1.0
        actual_min, actual_max = gmin, gmax
    else:
        all_values = np.array(all_values)
        actual_min = float(np.min(all_values))
        actual_max = float(np.max(all_values))

        # Use explicit limits if provided, otherwise use percentiles
        if settings.lower_limit is not None:
            gmin = float(settings.lower_limit)
        else:
            gmin = float(np.percentile(all_values, 5))  # 5th percentile

        if settings.upper_limit is not None:
            gmax = float(settings.upper_limit)
        else:
            gmax = float(np.percentile(all_values, 95))  # 95th percentile

    use_two = False
    if settings.symmetric_around_zero and gmin < 0 < gmax:
        vabs = max(abs(gmin), abs(gmax))
        gmin, gmax = -vabs, vabs
        use_two = True

    if gmax == gmin:
        gmax = gmin + 1e-9

    return gmin, gmax, use_two, actual_min, actual_max


def _make_lut(cmap_name: str | None, use_two_slope: bool, vmin, vmax):
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
    lut = (cmap(np.linspace(0, 1, 1024))[:, :3] * 255).astype(np.uint8)  # (1024,3) RGB
    return lut


def _to_uint16_idx(frame, vmin, vmax):
    norm = (frame - vmin) / (vmax - vmin)
    return np.clip((norm * 1023.0).round(), 0, 1023).astype(np.uint16)


def _blue_noise(shape, strength=0.0001):
    """Generate blue noise dithering pattern with extremely low strength to minimize visible grain"""
    rng = np.random.default_rng()
    n = rng.random(shape, dtype=np.float32) - 0.5
    # light blur to decorrelate; sigma ~0.5 px
    return cv2.GaussianBlur(n, (0, 0), 0.5) * strength


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
    pick: str = "uy",
    pattern: str = "[0-9]*.mat",
    settings: PlotSettings | None = None,
    cancel_event=None,
) -> dict:
    """
    Reads a sequence of MAT files containing piv_result.{ux,uy,b_mask} and writes a high-quality MP4.
    Returns metadata dict with limits and output path.
    Supports cancellation via cancel_event (threading.Event).
    """
    t0 = time.time()

    folder = Path(folder)
    files = sorted(
        [Path(p) for p in glob.glob(str(folder / pattern))], key=_natural_key
    )
    files = [f for f in files if "coordinate" not in f.name.lower()]
    if len(files) == 0:
        raise FileNotFoundError(f"No MAT files found in {folder} matching '{pattern}'")

    if settings is None:
        settings = PlotSettings()

    # Limit frames for test mode
    if hasattr(settings, "test_mode") and getattr(settings, "test_mode", False):
        test_frames = getattr(settings, "test_frames", 50)
        files = files[:test_frames]

    # Limits
    vmin, vmax, use_two, actual_min, actual_max = _compute_global_limits_from_files(
        files, pick, settings
    )

    # LUT (1024)
    lut = _make_lut(settings.cmap, use_two, vmin, vmax)

    # Size
    arrs0 = read_mat_contents(str(files[0]))
    arr0, b0 = _select_variable_from_arrs(arrs0, str(files[0]), pick)

    H, W = arr0.shape
    if H == 0 or W == 0:
        raise ValueError(
            f"Invalid image dimensions {H}x{W} in {files[0]}. Check your MAT file data."
        )
    Hout, Wout = _resolve_upscale(H, W, settings.upscale)

    # Writer: require ffmpeg (no fallback)
    if not settings.use_ffmpeg:
        raise RuntimeError(
            "Only FFmpeg-based writing supported; set settings.use_ffmpeg = True"
        )
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

    # Stream frames
    total_frames = len(files)
    for i, f in enumerate(files):
        if cancel_event and cancel_event.is_set():
            break
        arrs = read_mat_contents(str(f))
        try:
            field, b_mask = _select_variable_from_arrs(arrs, str(f), pick)
        except Exception:
            continue
        H_f, W_f = field.shape

        # Apply noise reduction if enabled
        if getattr(settings, "apply_smoothing", True):
            # Convert to float for processing
            field_smooth = field.astype(np.float32)

            # Apply median filter first to remove salt-and-pepper noise
            median_size = getattr(settings, "median_filter_size", 3)
            if median_size > 1:
                # Create a mask for valid data to avoid smoothing masked regions
                # valid_mask = np.ones_like(field_smooth, dtype=bool)
                # if b_mask is not None:
                # ~b_mask.astype(bool)

                # Apply median filter only to valid regions
                field_smooth = cv2.medianBlur(field_smooth, median_size)

            # Apply light Gaussian smoothing to reduce high-frequency noise
            sigma = getattr(settings, "smoothing_sigma", 0.8)
            if sigma > 0:
                field_smooth = cv2.GaussianBlur(field_smooth, (0, 0), sigma)

            # Use smoothed field
            field = field_smooth

        idx = _to_uint16_idx(field, vmin, vmax)  # 0..1023
        rgb = lut[idx]  # (H,W,3) uint8 RGB

        # Upscale image
        if Hout != H or Wout != W:
            rgb = cv2.resize(rgb, (Wout, Hout), interpolation=cv2.INTER_LANCZOS4)
            # Upscale mask with nearest-neighbor for sharp edges
            if b_mask is not None:
                b_mask_up = cv2.resize(
                    b_mask.astype(np.uint8),
                    (Wout, Hout),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                b_mask_up = None
        else:
            b_mask_up = b_mask.astype(bool) if b_mask is not None else None

        # Apply mask after resizing
        if b_mask_up is not None:
            rgb[b_mask_up] = settings.mask_rgb

        writer.write(rgb)
        # Progress callback
        if hasattr(settings, "progress_callback") and callable(
            settings.progress_callback
        ):
            settings.progress_callback(i + 1, total_frames)

    writer.release()

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
        "variable": pick,
        "cmap": settings.cmap,
        "elapsed_sec": round(t1 - t0, 3),
        "writer": "ffmpeg",
        "pix_fmt": getattr(settings, "pix_fmt", None),
        "crf": getattr(settings, "crf", None),
        "codec": getattr(settings, "codec", None),
    }


# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    frames_dir = Path(
        "C:\\Users\\ees1u24\\Desktop\\Planar_Images_with_wall\\test\\calibrated_piv\\1000\\Cam1\\instantaneous"
    )
    # Optional: coordinates (not used by video)
    try:
        from post_processing.vector_loading import load_coords_from_directory

        coords_x, coords_y = load_coords_from_directory(frames_dir)
    except Exception:
        coords_x = coords_y = None

    ps = PlotSettings(
        variableName=r"$u_y$",
        variableUnits=r"mm/s",
        cmap=None,  # None → automatic based on sign/symmetry
        lower_limit=-5,
        upper_limit=200,  # or set explicit limits
        symmetric_around_zero=True,
        fps=60,
        out_path="uy_hq.mp4",
        use_ffmpeg=True,
        crf=18,
        codec="libx264",
        pix_fmt="yuv420p",
        preset="slow",
        dither=True,
        upscale=4.0,  # moderate upscale to improve apparent resolution
        ffmpeg_extra_args=(),
        ffmpeg_loglevel="info",
    )

    meta = make_video_from_scalar(
        folder=frames_dir,
        pick="ux",  # 'ux' or 'uy'
        pattern="[0-9]*.mat",  # exclude coordinate file
        settings=ps,
    )
    print("Video written:", meta)
