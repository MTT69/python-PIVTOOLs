import glob
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from scipy.io import loadmat

from post_processing.vector_loading import read_mat_contents

# ------------------------- Settings -------------------------


@dataclass
class PlotSettings:
    corners: tuple = (None, None, None, None)  # (x0, y0, x1, y1)

    variableName: str = r"Variable Name"
    variableUnits: str = r"unit"

    save_name: str = ""
    save_extension: str = ".png"
    save_pickle: bool = False

    cmap: str | None = None
    levels: int = 30
    lower_limit: float | None = None
    upper_limit: float | None = None
    symmetric_around_zero: bool = True

    coordinate_units = r"m"
    _xlabel: str = r"$x$" + f" ({coordinate_units})"
    _ylabel: str = r"$y$" + f" ({coordinate_units})"
    _fontsize: int = 14
    _title_fontsize: int = 16

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
    dither: bool = True  # blue-noise pre-LUT to kill banding
    upscale: float | tuple | None = (
        None  # e.g. 2.0 or (H_out, W_out) or None (keep native)
    )

    # Extra ffmpeg args (appended to the ffmpeg command) - use this to tune quality further
    ffmpeg_extra_args: tuple | list = ()
    ffmpeg_loglevel: str = "warning"

    @property
    def title(self):
        return f"{self.variableName} ({self.variableUnits}) Plot"


# ------------------------- Helpers -------------------------

_num_re = re.compile(r"(\d+)")


def _resolve_upscale(h, w, upscale):
    """Return (H_out, W_out). `upscale` can be None, a float factor, or (H, W)."""
    if upscale is None:
        return h, w
    if isinstance(upscale, (int, float)):
        H = int(round(h * float(upscale)))
        W = int(round(w * float(upscale)))
    else:  # assume (H, W)
        H, W = int(upscale[0]), int(upscale[1])
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
    except Exception:
        pass

    # If arrs is dict-like, try to pull key directly
    try:
        if hasattr(arrs, "get"):
            if pick in arrs:
                arr = np.asarray(arrs[pick])
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

    # If many files exist, randomly sample up to N files to speed up limit computation.
    N = 50
    if len(files) > N:
        rng = np.random.default_rng()
        # choose indices without replacement and keep them sorted to preserve rough ordering
        idx = rng.choice(len(files), size=N, replace=False)
        files_to_check = [files[i] for i in sorted(idx)]
    else:
        files_to_check = files

    gmin, gmax = np.inf, -np.inf
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
        gmin = min(gmin, float(masked.min()))
        gmax = max(gmax, float(masked.max()))

    if not np.isfinite(gmin) or not np.isfinite(gmax):
        gmin, gmax = -1.0, 1.0

    use_two = False
    actual_min, actual_max = gmin, gmax
    if settings.symmetric_around_zero and gmin < 0 < gmax:
        vabs = max(abs(gmin), abs(gmax))
        gmin, gmax = -vabs, vabs
        use_two = True

    if gmax == gmin:
        gmax = gmin + 1e-9

    return gmin, gmax, use_two, actual_min, actual_max


def _make_lut(cmap_name: str | None, use_two_slope: bool, vmin, vmax):
    # 1024-step LUT to reduce banding before codec quantization
    if cmap_name is not None:
        cmap = cm.get_cmap(cmap_name, 1024)
    else:
        if use_two_slope:
            cmap = cm.get_cmap("bwr", 1024)
        elif vmax <= 0:
            cmap = cm.get_cmap("Blues_r", 1024)
        else:
            cmap = cm.get_cmap("Reds", 1024)
    lut = (cmap(np.linspace(0, 1, 1024))[:, :3] * 255).astype(np.uint8)  # (1024,3) RGB
    return lut


def _to_uint16_idx(frame, vmin, vmax):
    norm = (frame - vmin) / (vmax - vmin)
    return np.clip((norm * 1023.0).round(), 0, 1023).astype(np.uint16)


def _blue_noise(shape, strength=1.0 / 1024.0):
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
        crf=12,
        codec="libx264",
        pix_fmt="yuv444p",
        preset="slow",
        extra_args=None,
        loglevel="warning",
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
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
        stdin.write(rgb_frame_uint8.tobytes())

    def release(self):
        stdin = self.proc.stdin
        if stdin is not None:
            stdin.close()
        # wait and capture stderr for diagnostics
        _, stderr = self.proc.communicate()
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
    pick: str = "uy",  # 'ux' or 'uy' or any variable name/index
    pattern: str = "[0-9]*.mat",  # exclude coordinate files
    settings: PlotSettings | None = None,
) -> dict:
    """
    Reads a sequence of MAT files containing piv_result.{ux,uy,b_mask} and writes a high-quality MP4.
    Returns metadata dict with limits and output path.
    """
    t0 = time.time()

    folder = Path(folder)
    files = sorted(
        [Path(p) for p in glob.glob(str(folder / pattern))], key=_natural_key
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No MAT files found in {folder} matching '{pattern}'")

    if settings is None:
        settings = PlotSettings()

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

    # Optional blue-noise (static frame, extremely subtle)
    noise = _blue_noise((H, W)) if settings.dither else None

    # Stream frames
    for f in files:
        arrs = read_mat_contents(str(f))
        try:
            field, b_mask = _select_variable_from_arrs(arrs, str(f), pick)
        except Exception:
            # skip frames where variable is not available
            continue

        if settings.dither:
            field = field.astype(np.float32) + noise

        idx = _to_uint16_idx(field, vmin, vmax)  # 0..1023
        rgb = lut[idx]  # (H,W,3) uint8 RGB

        if b_mask is not None:
            m = b_mask.astype(bool)
            rgb[m] = settings.mask_rgb

        if settings.upscale:
            rgb = cv2.resize(rgb, (Wout, Hout), interpolation=cv2.INTER_LANCZOS4)

        writer.write(rgb)

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
        "cmap": (
            settings.cmap
            if settings.cmap
            else ("bwr" if use_two else ("Blues_r" if vmax <= 0 else "Reds"))
        ),
        "elapsed_sec": round(t1 - t0, 3),
        "writer": "ffmpeg",
        "pix_fmt": getattr(settings, "pix_fmt", None),
        "crf": getattr(settings, "crf", None),
        "codec": getattr(settings, "codec", None),
    }


# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    frames_dir = Path("calibrated_piv/1000/Cam1/instantaneous")

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
