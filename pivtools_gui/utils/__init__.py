"""Common utility helpers shared across blueprints.

Centralizes small duplicated snippets so updates (e.g. image encoding or
camera folder normalization) propagate consistently.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Union

import numpy as np
from loguru import logger
from PIL import Image
from scipy.ndimage import maximum_filter

# --- Particle-aware display window (image-pair viewer) -----------------------
# The frame is split into TILE_GRID x TILE_GRID tiles so that a glare patch or
# a dark wall occupying a minority of tiles cannot drag the window: the final
# black/white levels are medians over tiles, not area percentiles.
TILE_GRID = 8
# A particle peak must clear the tile background by this many noise sigmas.
# 5 sigma keeps Gaussian-noise false maxima to ~0 per tile while still
# catching dim particles (validated on the clean channel and BFS_1500 sets).
PEAK_K_SIGMA = 5.0
# Side (px) of the local-maximum neighbourhood; 3 is the smallest footprint
# that defines a pixel peak and matches the ~2-3 px PIV particle image size.
MAXIMA_FOOTPRINT = 3
# Converts the median absolute deviation to a Gaussian-consistent sigma.
MAD_TO_SIGMA = 1.4826
# The 8-bit encode of a display window [lo, hi] (counts) spans
# lo .. lo + H*(hi - lo), linear. H = 2 keeps 128 grey levels inside the window
# while letting the client preview a drag of the white bound up to 2x before
# the server re-encodes on release.
ENCODE_HEADROOM = 2.0
# Fewer tiles than this with detected peaks means the frame has no
# trustworthy particle population (blank, flat, or a lone hot pixel).
MIN_PEAK_TILES = 4
# Side (px) of the full-resolution central sub-block analysed per tile. Caps
# the work at TILE_GRID^2 * TILE_SAMPLE^2 ~ 1 MP regardless of sensor size;
# medians differ from full-tile results by <3 % on the validation frames.
TILE_SAMPLE = 128

# Fallback window: the old auto rule's 5/95 area percentiles. The old rule
# took them in the sqrt domain; percentiles commute with a monotonic
# transform, so in counts they are exactly the 5/95 percentiles of the frame.
_FALLBACK_PCT_LO, _FALLBACK_PCT_HI = 5.0, 95.0
# Above this many pixels the percentile window is estimated on a strided
# sub-sample (~1 MP) for speed, matching numpy_to_base64.
_PERCENTILE_SUBSAMPLE_ABOVE = 4_000_000
# 8-bit display range.
_DISPLAY_LEVELS = 255


def camera_number(camera: Union[str, int]) -> int:
    """Return the numeric camera id from a value like 1, "1", "Cam1".

    Raises ValueError if it cannot parse a positive int.
    """
    if isinstance(camera, int):
        return camera
    s = str(camera).strip()
    if s.lower().startswith("cam"):
        s = s[3:]
    try:
        cam_int = int(s)
    except (TypeError, ValueError):
        logger.error(f"Invalid camera identifier (non-parsable): {camera!r}")
        raise
    if cam_int < 0:
        raise ValueError("camera must be positive integer")
    return cam_int


def numpy_to_png_base64(arr: np.ndarray, compress_level: int = 1) -> str:
    """Convert a numpy array (uint8 or convertible) to a base64 PNG string.

    Args:
        arr: Input numpy array
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
                       Default is 1 for speed. Use 6 for better compression.
    """
    return numpy_to_base64(arr, format="png", compress_level=compress_level)


def get_display_contrast_stats(arr: np.ndarray) -> dict:
    """Compute contrast slider positions for the sqrt-normalised 8-bit encoding.

    Returns vmin_pct/vmax_pct as percentages (0-100) of the encoded 8-bit
    range, representing a tighter window than the full [0, 100] default.
    This gives the frontend slider a meaningful starting position: the user
    can widen toward [0, 100] to see more, or tighten to boost contrast.

    The percentiles (5th–95th) are computed in the sqrt domain to match
    the encoding in numpy_to_base64().
    """
    if arr.dtype == np.uint8:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    a = np.sqrt(np.maximum(arr.astype(np.float32, copy=False), 0))
    if not a.size:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    total = a.size
    if total > 4_000_000:
        stride = max(2, int(np.sqrt(total / 1_000_000)))
        sampled = a[::stride, ::stride].ravel()
    else:
        sampled = a.ravel()

    # The encoding window used by numpy_to_base64
    enc_lo = float(np.percentile(sampled, 0.5))
    enc_hi = float(np.percentile(sampled, 99.5))
    enc_range = enc_hi - enc_lo
    if enc_range <= 0:
        return {"vmin_pct": 0.0, "vmax_pct": 100.0}

    # Tighter auto-scale window within the encoded range
    p5 = float(np.percentile(sampled, 5))
    p95 = float(np.percentile(sampled, 95))

    vmin_pct = max(0.0, 100.0 * (p5 - enc_lo) / enc_range)
    vmax_pct = min(100.0, 100.0 * (p95 - enc_lo) / enc_range)

    return {"vmin_pct": round(vmin_pct, 2), "vmax_pct": round(vmax_pct, 2)}


def _percentile_window(arr: np.ndarray) -> dict:
    """Fallback display window: the 5/95 area percentiles of the frame (counts).

    This is the old auto rule. It was evaluated in the sqrt domain, but
    percentiles commute with a monotonic transform, so the bounds are
    identical when taken directly on counts.
    """
    a = arr
    if a.size > _PERCENTILE_SUBSAMPLE_ABOVE:
        stride = max(2, int(np.sqrt(a.size / 1_000_000)))
        a = a[::stride, ::stride]
    if not a.size:
        return {"black": 0.0, "white": 0.0, "method": "percentile_fallback"}
    black, white = (
        float(v) for v in np.percentile(a, [_FALLBACK_PCT_LO, _FALLBACK_PCT_HI])
    )
    return {"black": black, "white": white, "method": "percentile_fallback"}


def _tile_blocks(arr: np.ndarray, tile_sample: int | None) -> np.ndarray:
    """Stack the central sub-block of each of the TILE_GRID^2 tiles.

    Tiles are ``(H // TILE_GRID, W // TILE_GRID)``; the up-to-``TILE_GRID - 1``
    remainder rows/columns are ignored. Each tile contributes its central
    ``min(tile, tile_sample)`` block (the whole tile when ``tile_sample`` is
    None). Returns a float32 copy of shape ``(TILE_GRID**2, sh, sw)``, so the
    caller's (possibly read-only) frame is never written.
    """
    h, w = arr.shape
    th, tw = h // TILE_GRID, w // TILE_GRID
    sh = th if tile_sample is None else min(th, tile_sample)
    sw = tw if tile_sample is None else min(tw, tile_sample)
    blocks = np.empty((TILE_GRID * TILE_GRID, sh, sw), dtype=np.float32)
    for i in range(TILE_GRID):
        y0 = i * th + (th - sh) // 2
        for j in range(TILE_GRID):
            x0 = j * tw + (tw - sw) // 2
            blocks[i * TILE_GRID + j] = arr[y0 : y0 + sh, x0 : x0 + sw]
    return blocks


def particle_display_window(
    arr: np.ndarray, tile_sample: int | None = TILE_SAMPLE
) -> dict:
    """Display window from the background level to the typical particle peak.

    Area percentiles put the window where most *pixels* are, so glare, walls
    and dark borders decide it. This instead measures the two quantities a PIV
    viewer cares about, per tile of a TILE_GRID x TILE_GRID grid:

    - background ``b`` = median intensity,
    - noise ``sigma`` = MAD_TO_SIGMA * median|f - b|,
    - particle peaks = 3x3 local maxima above ``b + PEAK_K_SIGMA * sigma``,
      and the tile white ``w`` = median peak intensity.

    Black and white are the medians of ``b`` and ``w`` over the tiles that had
    peaks, so a glare patch covering a minority of tiles is ignored and
    allowed to saturate.

    Tiles with ``sigma == 0`` (at least half their pixels share one value,
    e.g. a masked or clipped region, or a background-subtracted residual that
    is mostly exactly zero) are excluded: with no noise scale the 5-sigma
    test degenerates to "any value above the median" and would count noise
    as particles. If fewer than MIN_PEAK_TILES tiles have peaks, or the
    medians give ``white <= black``, the old percentile window is returned
    with ``method="percentile_fallback"`` and a warning is logged.

    Args:
        arr: 2-D frame of any real dtype (raw counts or a processed float frame).
        tile_sample: Side of the central sub-block analysed per tile (speed);
            None analyses whole tiles.

    Returns:
        dict with ``black``/``white`` (intensity units of ``arr``) and
        ``method`` ("particle" or "percentile_fallback").
    """
    if arr.ndim != 2:
        raise ValueError(
            f"particle_display_window expects a 2-D frame, got {arr.shape}"
        )

    h, w = arr.shape
    if h // TILE_GRID < MAXIMA_FOOTPRINT or w // TILE_GRID < MAXIMA_FOOTPRINT:
        logger.warning(
            f"Frame {arr.shape} too small for {TILE_GRID}x{TILE_GRID} tile "
            "analysis; using percentile display window"
        )
        return _percentile_window(arr)

    blocks = _tile_blocks(arr, tile_sample)
    n_tiles = blocks.shape[0]
    flat = blocks.reshape(n_tiles, -1)
    bg = np.median(flat, axis=1)
    sigma = MAD_TO_SIGMA * np.median(np.abs(flat - bg[:, None]), axis=1)

    # size=(1, F, F): maxima are found within each tile, never across tiles.
    local_max = blocks == maximum_filter(
        blocks, size=(1, MAXIMA_FOOTPRINT, MAXIMA_FOOTPRINT)
    )
    is_peak = local_max & (blocks > (bg + PEAK_K_SIGMA * sigma)[:, None, None])

    tile_bg, tile_white = [], []
    for k in range(n_tiles):
        if sigma[k] == 0:
            continue
        peaks = blocks[k][is_peak[k]]
        if peaks.size:
            tile_bg.append(bg[k])
            tile_white.append(np.median(peaks))

    if len(tile_bg) < MIN_PEAK_TILES:
        logger.warning(
            f"Particle display window: only {len(tile_bg)}/{n_tiles} tiles have "
            f"particle peaks (need {MIN_PEAK_TILES}); using percentile window"
        )
        return _percentile_window(arr)

    black = float(np.median(tile_bg))
    white = float(np.median(tile_white))
    if white <= black:
        logger.warning(
            f"Particle display window: white {white:.4g} <= black {black:.4g}; "
            "using percentile window"
        )
        return _percentile_window(arr)

    return {"black": black, "white": white, "method": "particle"}


def parse_display_bounds(vmin, vmax) -> tuple[float, float] | None:
    """Parse an optional manual display window from request values.

    Args:
        vmin: Lower bound (intensity units) as str/number, or None.
        vmax: Upper bound (intensity units) as str/number, or None.

    Returns:
        ``(vmin, vmax)`` as floats, or None when both are absent (auto window).

    Raises:
        ValueError: if only one bound is given, either is not a finite
            number, or ``vmax <= vmin``.
    """
    if vmin is None and vmax is None:
        return None
    if vmin is None or vmax is None:
        raise ValueError("Display window needs both vmin and vmax, or neither")
    try:
        lo, hi = float(vmin), float(vmax)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Display window bounds must be numbers: {e}") from e
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"Display window bounds must be finite, got {lo}, {hi}")
    if hi <= lo:
        raise ValueError(f"Display window needs vmax > vmin, got {lo}, {hi}")
    return lo, hi


def encode_display_frame(
    arr: np.ndarray,
    img_format: str,
    bounds: tuple[float, float] | None = None,
) -> tuple[str, dict]:
    """Encode one image-pair-viewer frame linearly and return ``(base64, stats)``.

    The display window ``[lo, hi]`` is ``bounds`` when given (a manual window
    in intensity units), otherwise the auto window from
    particle_display_window. The 8-bit image spans
    ``lo .. lo + ENCODE_HEADROOM * (hi - lo)`` linearly, capped at the frame
    maximum (but never below ``hi``), so the client can remap within it while
    a slider is dragged and still show the window exactly.

    Args:
        arr: 2-D frame (raw counts or a processed float frame).
        img_format: "png" or "jpeg".
        bounds: Manual ``(vmin, vmax)`` from parse_display_bounds, or None.

    Returns:
        ``(base64, stats)`` with stats ``{"black", "white", "method"}`` (the
        auto window, always computed so the client can return to it),
        ``{"enc_lo", "enc_hi"}`` (intensity span of the 8-bit levels 0..255)
        and ``{"data_min", "data_max"}`` (frame range, for the slider).
    """
    auto = particle_display_window(arr)
    data_min, data_max = float(arr.min()), float(arr.max())
    lo, hi = bounds if bounds is not None else (auto["black"], auto["white"])
    enc_lo = lo
    enc_hi = min(lo + ENCODE_HEADROOM * (hi - lo), max(hi, data_max))
    if enc_hi <= enc_lo:
        # Flat auto window (constant frame): any positive span encodes it black.
        enc_hi = enc_lo + 1.0

    q = arr.astype(np.float32)
    q -= enc_lo
    q *= _DISPLAY_LEVELS / (enc_hi - enc_lo)
    np.clip(q, 0, _DISPLAY_LEVELS, out=q)
    np.rint(q, out=q)
    b64 = numpy_to_base64(q.astype(np.uint8), format=img_format)

    stats = {
        "black": auto["black"],
        "white": auto["white"],
        "method": auto["method"],
        "enc_lo": enc_lo,
        "enc_hi": enc_hi,
        "data_min": data_min,
        "data_max": data_max,
    }
    return b64, stats


def numpy_to_base64(
    arr: np.ndarray,
    format: str = "png",
    compress_level: int = 1,
    jpeg_quality: int = 85,
    vmin: float = None,
    vmax: float = None,
) -> str:
    """Convert a numpy array to a base64 encoded image string.

    For high-bit-depth images (12/16-bit), applies a sqrt transform before
    the linear percentile clip.  This is the variance-stabilising transform
    for Poisson (photon-counting) noise and compresses the dynamic range so
    that dim PIV particles are visible in the 8-bit output without affecting
    well-exposed images (sqrt is nearly linear for large values).

    When vmin/vmax are not provided, automatic 0.5th–99.5th percentile
    clipping is applied on the sqrt-transformed data.  This rejects hot
    pixels and sensor defects while preserving the full particle signal.

    Args:
        arr: Input numpy array
        format: Image format - "png" or "jpeg"
        compress_level: PNG compression level (0-9). Lower = faster, higher = smaller.
        jpeg_quality: JPEG quality (1-95). Higher = better quality, larger files.
        vmin: If provided, clip sqrt-domain values below this before normalising
              to 0-255. When omitted, the 0.5th percentile is used.
        vmax: If provided, clip sqrt-domain values above this before normalising
              to 0-255. When omitted, the 99.5th percentile is used.
    """
    if arr.dtype != np.uint8:
        # Apply sqrt for variance-stabilised normalisation of photon-count data.
        # This dramatically improves contrast for low-count PIV images (12/16-bit
        # cameras using <10% of sensor range) while being nearly invisible on
        # well-exposed images where sqrt(x) ≈ linear for large x.
        a = np.sqrt(np.maximum(arr.astype(np.float32, copy=False), 0))
        if a.size:
            if vmin is not None and vmax is not None:
                mn, mx = float(vmin), float(vmax)
            else:
                # Auto percentile clipping in sqrt domain
                total = a.size
                if total > 4_000_000:
                    stride = max(2, int(np.sqrt(total / 1_000_000)))
                    sampled = a[::stride, ::stride].ravel()
                else:
                    sampled = a.ravel()
                mn = float(np.percentile(sampled, 0.5))
                mx = float(np.percentile(sampled, 99.5))
            if mx > mn:
                a = (255 * (np.clip(a, mn, mx) - mn) / (mx - mn)).astype(np.uint8)
            else:
                logger.debug("Flat image (min==max); producing black output")
                a = np.zeros_like(a, dtype=np.uint8)
        else:
            logger.debug("Empty array; substituting 1x1 black pixel")
            a = np.zeros((1, 1), dtype=np.uint8)
        arr = a

    img = Image.fromarray(arr)
    buf = BytesIO()

    if format.lower() == "jpeg":
        # PIL handles 'L' (grayscale) mode JPEG natively — no need to convert to RGB.
        # Skipping the conversion avoids tripling the data size before compression.
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        # Default to PNG
        img.save(buf, format="PNG", compress_level=compress_level, optimize=False)

    return base64.b64encode(buf.getvalue()).decode("utf-8")
